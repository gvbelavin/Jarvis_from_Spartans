"""
web_audio.py — аудиотракт через браузер телефона вместо микрофона/динамика платы.

Два режима работы страницы, оба ведут в один и тот же пайплайн:

    / (webapp/orb.html)      постоянный поток с микрофона по WebSocket.
                             Wake word и VAD считает ПЛАТА — теми же
                             wake_word.detect и CommandRecorder, что и для
                             железного микрофона. Телефон в этом режиме —
                             просто микрофон с динамиком на проводе.

    /classic (webapp/...)    «нажал — говори — нажал»: одна запись уходит
                             POST-запросом /api/talk. Wake word не нужен.

Дальше всё как обычно: тот же Speaker ID, тот же Whisper, та же память и
LLM. Ответ Piper возвращается тому, кто задал вопрос: в теле POST-запроса
или сообщением в WebSocket.

Адаптер реализует контракт AudioModuleInterface (contracts.py), поэтому
оркестратор о вебе не знает ничего:

    wait_for_wake_word()  ->  ждём фразу с телефона (из любого режима)
    record_phrase()       ->  отдаём эту запись (int16, 16 кГц, моно)
    identify_speaker()    ->  модуль Б (через engine = AudioAdapter)
    speech_to_text()      ->  модуль Б
    play_tts()            ->  синтез Piper, WAV уходит на телефон

Необязательные хуки, которые вызывает app.py, если они есть:
`open()` поднимает HTTP(S)-сервер, `end_turn()` закрывает ход,
`aclose()` гасит сервер.

Модели второй раз НЕ грузятся: engine — тот же AudioAdapter (или заглушка
в режиме --mock), веб-сервер живёт в том же процессе и том же event loop.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from aiohttp import WSMsgType, web

from contracts import GUEST_NAME, SpeakerResult
from settings import WEBAPP_DIR

logger = logging.getLogger(__name__)

# Сколько ждать, пока оркестратор заберёт запись. Если он занят или
# микрофон замьючен кнопкой на индикаторе — телефон получит 503, а не
# будет висеть вечно.
PICKUP_TIMEOUT = 15.0

# Потолок на весь ход: STT + LLM + TTS на RK3588 укладываются в секунды,
# но LLM-сервер бывает холодным.
TURN_TIMEOUT = 180.0

MIN_SECONDS = 0.3
MAX_SECONDS = 60.0

# Потолок на одну фразу в потоковом режиме. CommandRecorder завершает
# запись только по тишине: в шумной комнате он не закончит никогда.
MAX_PHRASE_SECONDS = 20.0

# Сколько ждать первое сообщение с токеном в WebSocket.
AUTH_TIMEOUT = 10.0

# Как часто проверять, не положили ли на диск новый сертификат.
CERT_RELOAD_INTERVAL = 3600.0


class _HttpSink:
    """Ответы копятся и уезжают телом POST-запроса /api/talk."""

    def __init__(self) -> None:
        self.replies: list[dict[str, str]] = []

    async def state(self, name: str) -> None:
        pass

    async def reply(self, text: str, wav_bytes: bytes) -> None:
        self.replies.append(
            {"text": text, "audio": base64.b64encode(wav_bytes).decode("ascii")}
        )


class _WsSink:
    """Ответы и состояния уезжают в WebSocket — по ним живёт «шар»."""

    def __init__(self, ws: web.WebSocketResponse) -> None:
        self._ws = ws

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._ws.closed:
            return
        try:
            await self._ws.send_json(payload)
        except (ConnectionResetError, RuntimeError):
            logger.debug("[web] Сокет закрылся раньше отправки.", exc_info=True)

    async def state(self, name: str) -> None:
        await self._send({"type": "state", "state": name})

    async def send_error(self, message: str) -> None:
        await self._send({"type": "error", "code": "turn", "message": message})

    async def reply(self, text: str, wav_bytes: bytes) -> None:
        await self._send(
            {
                "type": "reply",
                "text": text,
                "audio": base64.b64encode(wav_bytes).decode("ascii"),
            }
        )


@dataclass
class _Turn:
    """Один ход разговора: запись с телефона и всё, что вернём обратно."""

    audio: np.ndarray
    sink: Any
    picked: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    transcript: str = ""
    speaker: Optional[SpeakerResult] = None


class _StreamSession:
    """
    Wake word и VAD поверх сетевого потока.

    Повторяет конечный автомат `_ListenSession` из audio_adapter.py, только
    чанки приходят не из PortAudio, а из WebSocket. Модуль Б ждёт ровно
    `CHUNK_SIZE` отсчётов за вызов (openWakeWord — 80 мс при 16 кГц),
    а браузер шлёт чем придётся, поэтому здесь свой буфер.
    """

    WAKE = "wake"
    PHRASE = "phrase"

    def __init__(
        self,
        detect,
        recorder_cls: type,
        chunk_size: int,
        sample_rate: int,
    ) -> None:
        self._detect = detect
        self._recorder_cls = recorder_cls
        self._chunk_size = chunk_size
        self._max_samples = int(MAX_PHRASE_SECONDS * sample_rate)

        self._tail = np.empty(0, dtype=np.int16)
        self._recorder: Any = None
        self._recorded = 0

        # Пока плата отвечает, поток игнорируется: железный микрофон в это
        # время тоже закрыт, и Джарвис не должен будить сам себя.
        self.paused = False

    def feed(self, chunk: np.ndarray) -> list[tuple[str, Optional[np.ndarray]]]:
        """Скармливает кусок потока; возвращает события (wake / готовая фраза)."""
        events: list[tuple[str, Optional[np.ndarray]]] = []

        if self.paused:
            return events

        self._tail = np.concatenate((self._tail, chunk))

        while self._tail.size >= self._chunk_size:
            frame = self._tail[: self._chunk_size]
            self._tail = self._tail[self._chunk_size :]

            if self._recorder is None:
                if self._detect(frame):
                    self._recorder = self._recorder_cls()
                    self._recorded = 0
                    events.append((self.WAKE, None))
                continue

            self._recorder.process(frame)
            self._recorded += frame.size

            if self._recorder.finished or self._recorded >= self._max_samples:
                audio = (
                    self._recorder.get_audio()
                    if self._recorder.audio_chunks
                    else None
                )
                self._recorder = None
                self.paused = True
                events.append((self.PHRASE, audio))

        return events

    def resume(self) -> None:
        """Плата ответила — снова слушаем wake word с чистого листа."""
        self._tail = np.empty(0, dtype=np.int16)
        self._recorder = None
        self.paused = False


class WebAudioAdapter:
    def __init__(
        self,
        engine: Any,
        sample_rate: int,
        host: str,
        port: int,
        token: Optional[str] = None,
        cert_file: Optional[str] = None,
        key_file: Optional[str] = None,
    ) -> None:
        # engine обязан уметь identify_speaker, speech_to_text,
        # synthesize(text) -> WAV bytes и stream_tools() для потокового
        # режима (AudioAdapter или AudioEngineMock).
        self._engine = engine
        self.sample_rate = sample_rate
        self._host = host
        self._port = port
        self._token = token or None
        self._cert_file = cert_file
        self._key_file = key_file

        self._queue: asyncio.Queue[_Turn] = asyncio.Queue(maxsize=1)
        self._current: Optional[_Turn] = None
        self._busy = False

        self._stream_ws: Optional[web.WebSocketResponse] = None
        self._stream_session: Optional[_StreamSession] = None
        self._stream_tools: Optional[tuple] = None

        self._runner: Optional[web.AppRunner] = None
        self._reload_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Контракт AudioModuleInterface
    # ------------------------------------------------------------------

    async def wait_for_wake_word(self) -> None:
        # Страховка: если оркестратор не вызвал end_turn, ход закрывается
        # здесь, иначе телефон ждал бы до TURN_TIMEOUT.
        await self._finish_turn()

        turn = await self._queue.get()
        turn.picked.set()
        self._current = turn
        logger.info(
            "[web] Фраза с телефона: %.1f с.",
            turn.audio.size / self.sample_rate,
        )
        await turn.sink.state("thinking")

    async def record_phrase(self) -> np.ndarray:
        if self._current is None:
            raise RuntimeError(
                "record_phrase() called before wait_for_wake_word()."
            )
        return self._current.audio

    async def identify_speaker(self, audio_data) -> SpeakerResult:
        result = await self._engine.identify_speaker(audio_data)
        if self._current is not None:
            self._current.speaker = result
        return result

    async def speech_to_text(self, audio_data) -> str:
        text = await self._engine.speech_to_text(audio_data)
        if self._current is not None:
            self._current.transcript = text
        return text

    async def play_tts(self, text: str) -> None:
        turn = self._current

        # Фразы вне хода (приветствие при старте, «спокойной ночи») слать
        # некому — и синтезировать их незачем.
        if turn is None:
            logger.info("[web] Нет активного запроса, фраза пропущена: %r", text)
            return

        wav_bytes = await self._engine.synthesize(text)
        await turn.sink.state("speaking")
        await turn.sink.reply(text, wav_bytes)

    async def listen_once(self) -> tuple[SpeakerResult, str]:
        await self.wait_for_wake_word()
        audio_data = await self.record_phrase()

        speaker_result, transcript = await asyncio.gather(
            self.identify_speaker(audio_data),
            self.speech_to_text(audio_data),
        )
        return speaker_result, transcript

    # ------------------------------------------------------------------
    # Хуки жизненного цикла (вызываются из app.py, если есть)
    # ------------------------------------------------------------------

    async def open(self) -> None:
        app = web.Application(client_max_size=8 * 1024 * 1024)
        app.router.add_get("/", self._handle_orb)
        app.router.add_get("/classic", self._handle_classic)
        app.router.add_get("/api/status", self._handle_status)
        app.router.add_get("/api/stream", self._handle_stream)
        app.router.add_post("/api/talk", self._handle_talk)
        app.router.add_static("/static/", WEBAPP_DIR)

        ssl_context = self._make_ssl_context()

        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(
            self._runner, self._host, self._port, ssl_context=ssl_context
        )
        await site.start()

        scheme = "https" if ssl_context else "http"
        logger.info(
            "[web] Веб-интерфейс: %s://%s:%d/", scheme, self._host, self._port
        )

        if ssl_context is None and self._host not in ("127.0.0.1", "localhost"):
            logger.warning(
                "[web] Сервер без TLS: Safari на iPhone не даст доступ к "
                "микрофону по http://. Укажите --web-cert и --web-key."
            )

        if self._token is None:
            logger.warning(
                "[web] JARVIS_WEB_TOKEN не задан: говорить с Джарвисом может "
                "любой пир NetBird-сети."
            )

        if ssl_context is not None:
            self._reload_task = asyncio.create_task(
                self._reload_cert_forever(ssl_context)
            )

    async def end_turn(self) -> None:
        await self._finish_turn()

    async def aclose(self) -> None:
        await self._finish_turn()

        if self._reload_task is not None:
            self._reload_task.cancel()
            self._reload_task = None

        if self._stream_ws is not None and not self._stream_ws.closed:
            await self._stream_ws.close()

        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def close(self) -> None:
        turn, self._current = self._current, None
        if turn is not None:
            turn.done.set()

        close = getattr(self._engine, "close", None)
        if callable(close):
            close()

    # ------------------------------------------------------------------
    # Страницы
    # ------------------------------------------------------------------

    async def _handle_orb(self, request: web.Request) -> web.StreamResponse:
        return self._page("orb.html")

    async def _handle_classic(self, request: web.Request) -> web.StreamResponse:
        return self._page("index.html")

    def _page(self, name: str) -> web.StreamResponse:
        return web.FileResponse(
            WEBAPP_DIR / name, headers={"Cache-Control": "no-cache"}
        )

    # ------------------------------------------------------------------
    # HTTP API
    # ------------------------------------------------------------------

    async def _handle_status(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "busy": self._busy,
                "streaming": self._stream_ws is not None,
                "sample_rate": self.sample_rate,
                "auth_required": self._token is not None,
                "authorized": self._authorized(
                    request.headers.get("X-Jarvis-Token", "")
                ),
            }
        )

    async def _handle_talk(self, request: web.Request) -> web.Response:
        if not self._authorized(request.headers.get("X-Jarvis-Token", "")):
            return _error(401, "Неверный токен.")

        if self._busy:
            return _error(409, "Джарвис ещё отвечает на прошлый вопрос.")

        try:
            rate = int(request.headers.get("X-Sample-Rate", "0"))
        except ValueError:
            rate = 0
        if rate != self.sample_rate:
            return _error(400, f"Ожидается PCM {self.sample_rate} Гц.")

        body = await request.read()
        if len(body) % 2:
            return _error(400, "Ожидается int16 little-endian PCM.")

        audio = np.frombuffer(body, dtype="<i2").astype(np.int16)
        seconds = audio.size / self.sample_rate
        if seconds < MIN_SECONDS:
            return _error(400, "Слишком короткая запись.")
        if seconds > MAX_SECONDS:
            return _error(413, "Слишком длинная запись.")

        self._busy = True
        sink = _HttpSink()
        turn = _Turn(audio=audio, sink=sink)
        try:
            self._queue.put_nowait(turn)

            try:
                await asyncio.wait_for(turn.picked.wait(), PICKUP_TIMEOUT)
            except asyncio.TimeoutError:
                if not turn.picked.is_set():
                    self._queue.get_nowait()
                    return _error(
                        503,
                        "Джарвис сейчас не слушает (занят или выключен "
                        "микрофон на индикаторе).",
                    )

            try:
                await asyncio.wait_for(turn.done.wait(), TURN_TIMEOUT)
            except asyncio.TimeoutError:
                return _error(504, "Джарвис не успел ответить.")

            speaker = turn.speaker
            return web.json_response(
                {
                    "transcript": turn.transcript,
                    "speaker": {
                        "name": speaker.user_name if speaker else GUEST_NAME,
                        "confidence": speaker.confidence if speaker else 0.0,
                    },
                    "replies": sink.replies,
                }
            )
        finally:
            self._busy = False

    # ------------------------------------------------------------------
    # Поток с микрофона (режим «шара»)
    # ------------------------------------------------------------------

    async def _handle_stream(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30.0, max_msg_size=4 * 1024 * 1024)
        await ws.prepare(request)

        # Токен приходит первым сообщением, а не в заголовке: из браузера
        # заголовки к WebSocket не приставишь, а в URL токен светился бы
        # в логах и истории.
        if not await self._authenticate(ws):
            return ws

        if self._stream_ws is not None:
            await ws.send_json(
                {"type": "error", "code": "busy", "message": "Джарвис уже слушает другое устройство."}
            )
            await ws.close()
            return ws

        try:
            tools = await self._get_stream_tools()
        except Exception as exc:
            logger.exception("[web] Модуль wake word недоступен.")
            await ws.send_json(
                {"type": "error", "code": "wake_word", "message": str(exc)}
            )
            await ws.close()
            return ws

        session = _StreamSession(*tools, sample_rate=self.sample_rate)
        sink = _WsSink(ws)
        self._stream_ws = ws
        self._stream_session = session

        logger.info("[web] Телефон слушает: %s", request.remote)
        await ws.send_json({"type": "ready", "sample_rate": self.sample_rate})
        await sink.state("idle")

        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    chunk = np.frombuffer(msg.data, dtype="<i2")
                    events = await asyncio.to_thread(session.feed, chunk)

                    for kind, audio in events:
                        if kind == _StreamSession.WAKE:
                            logger.info("[web] Wake word с телефона.")
                            await sink.state("listening")
                        elif kind == _StreamSession.PHRASE:
                            await self._submit_phrase(audio, sink, session)

                elif msg.type == WSMsgType.TEXT:
                    data = _json_or_none(msg.data) or {}
                    if data.get("type") == "bye":
                        break

                elif msg.type == WSMsgType.ERROR:
                    logger.info("[web] Сокет оборвался: %s", ws.exception())
                    break
        finally:
            self._stream_ws = None
            self._stream_session = None
            logger.info("[web] Телефон отключился.")

        return ws

    async def _authenticate(self, ws: web.WebSocketResponse) -> bool:
        if self._token is None:
            return True

        try:
            msg = await ws.receive(timeout=AUTH_TIMEOUT)
        except asyncio.TimeoutError:
            await ws.close()
            return False

        data = _json_or_none(msg.data) if msg.type == WSMsgType.TEXT else None

        if not data or not self._authorized(str(data.get("token", ""))):
            await ws.send_json({"type": "error", "code": "auth", "message": "Неверный токен."})
            await ws.close()
            return False

        return True

    async def _submit_phrase(
        self,
        audio: Optional[np.ndarray],
        sink: _WsSink,
        session: _StreamSession,
    ) -> None:
        """Фраза записана — отдаём её оркестратору тем же путём, что и POST."""
        seconds = 0.0 if audio is None else audio.size / self.sample_rate

        if audio is None or seconds < MIN_SECONDS:
            logger.info("[web] После wake word ничего не сказано.")
            await sink.state("idle")
            session.resume()
            return

        if self._busy:
            await sink.state("idle")
            await sink.send_error("Джарвис ещё отвечает на прошлый вопрос.")
            session.resume()
            return

        self._busy = True
        turn = _Turn(audio=audio.astype(np.int16), sink=sink)

        try:
            self._queue.put_nowait(turn)
        except asyncio.QueueFull:
            self._busy = False
            await sink.state("idle")
            session.resume()
            return

        # Ход живёт дальше сам: ответ уедет в сокет из play_tts, а
        # _finish_turn снимет busy и вернёт поток к ожиданию wake word.
        asyncio.create_task(self._await_turn(turn, session))

    async def _await_turn(self, turn: _Turn, session: _StreamSession) -> None:
        try:
            await asyncio.wait_for(turn.done.wait(), TURN_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("[web] Оркестратор не ответил за %.0f с.", TURN_TIMEOUT)
            await turn.sink.state("idle")
        finally:
            self._busy = False
            session.resume()

    async def _get_stream_tools(self) -> tuple:
        """
        (detect, CommandRecorder, CHUNK_SIZE) от модуля Б.

        Лениво и в отдельном потоке: openWakeWord поднимает ONNX-сессию
        при импорте, а держать это в момент старта сервера незачем —
        вдруг телефон в режиме «шара» сегодня и не подключится.
        """
        if self._stream_tools is None:
            self._stream_tools = await asyncio.to_thread(self._engine.stream_tools)
        return self._stream_tools

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------

    def _authorized(self, given: str) -> bool:
        if self._token is None:
            return True
        return hmac.compare_digest(given.encode(), self._token.encode())

    async def _finish_turn(self) -> None:
        turn, self._current = self._current, None
        if turn is None:
            return

        turn.done.set()
        await turn.sink.state("idle")

    def _make_ssl_context(self) -> Optional[ssl.SSLContext]:
        if not self._cert_file and not self._key_file:
            return None
        if not (self._cert_file and self._key_file):
            raise SystemExit("--web-cert и --web-key задаются только вместе.")

        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(self._cert_file, self._key_file)
        return context

    async def _reload_cert_forever(self, context: ssl.SSLContext) -> None:
        """
        Продлённый сертификат копируется (scp) по тем же путям.
        load_cert_chain на живом контексте применяется к новым соединениям,
        так что перезапускать оркестратор не нужно.
        """
        cert = Path(self._cert_file)
        last_mtime = cert.stat().st_mtime

        while True:
            await asyncio.sleep(CERT_RELOAD_INTERVAL)
            try:
                mtime = cert.stat().st_mtime
                if mtime != last_mtime:
                    context.load_cert_chain(self._cert_file, self._key_file)
                    last_mtime = mtime
                    logger.info("[web] Сертификат TLS перечитан.")
            except (OSError, ssl.SSLError):
                logger.exception("[web] Не удалось перечитать сертификат.")


def _json_or_none(raw: Any) -> Optional[dict]:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)
