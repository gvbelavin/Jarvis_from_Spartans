"""
web_audio.py — аудиотракт через браузер телефона вместо микрофона/динамика платы.

Телефон открывает страницу (webapp/) по NetBird, пишет фразу по нажатию
кнопки и отправляет её POST-запросом /api/talk. Дальше всё как обычно:
тот же Speaker ID, тот же Whisper, та же память и LLM. Ответ Piper
возвращается в теле того же запроса и играет на телефоне.

Адаптер реализует контракт AudioModuleInterface (contracts.py), поэтому
оркестратор о вебе не знает ничего:

    wait_for_wake_word()  ->  ждём, пока с телефона придёт запись
    record_phrase()       ->  отдаём эту запись (int16, 16 кГц, моно)
    identify_speaker()    ->  модуль Б (через engine = AudioAdapter)
    speech_to_text()      ->  модуль Б
    play_tts()            ->  синтез Piper, WAV складывается в ответ телефону

Необязательные хуки, которые вызывает app.py, если они есть:
`open()` поднимает HTTP(S)-сервер, `end_turn()` отдаёт ответ телефону,
`aclose()` гасит сервер.

Модели второй раз НЕ грузятся: engine — тот же AudioAdapter (или заглушка
в режиме --mock), веб-сервер живёт в том же процессе и том же event loop.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import logging
import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from aiohttp import web

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

# Как часто проверять, не положили ли на диск новый сертификат.
CERT_RELOAD_INTERVAL = 3600.0


@dataclass
class _Turn:
    """Один ход разговора: запись с телефона и всё, что вернём обратно."""

    audio: np.ndarray
    picked: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    transcript: str = ""
    speaker: Optional[SpeakerResult] = None
    replies: list[dict[str, str]] = field(default_factory=list)


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
        # engine обязан уметь identify_speaker, speech_to_text и
        # synthesize(text) -> WAV bytes (AudioAdapter или AudioEngineMock).
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

        self._runner: Optional[web.AppRunner] = None
        self._reload_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Контракт AudioModuleInterface
    # ------------------------------------------------------------------

    async def wait_for_wake_word(self) -> None:
        # Страховка: если оркестратор не вызвал end_turn, ход закрывается
        # здесь, иначе телефон ждал бы до TURN_TIMEOUT.
        self._finish_turn()

        turn = await self._queue.get()
        turn.picked.set()
        self._current = turn
        logger.info(
            "[web] Получена запись с телефона: %.1f с.",
            turn.audio.size / self.sample_rate,
        )

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
        turn.replies.append(
            {
                "text": text,
                "audio": base64.b64encode(wav_bytes).decode("ascii"),
            }
        )

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
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/api/status", self._handle_status)
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
        self._finish_turn()

    async def aclose(self) -> None:
        self._finish_turn()

        if self._reload_task is not None:
            self._reload_task.cancel()
            self._reload_task = None

        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def close(self) -> None:
        self._finish_turn()
        close = getattr(self._engine, "close", None)
        if callable(close):
            close()

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    async def _handle_index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(
            WEBAPP_DIR / "index.html",
            headers={"Cache-Control": "no-cache"},
        )

    async def _handle_status(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "busy": self._busy,
                "sample_rate": self.sample_rate,
                "auth_required": self._token is not None,
                "authorized": self._authorized(request),
            }
        )

    async def _handle_talk(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
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
        turn = _Turn(audio=audio)
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
                    "replies": turn.replies,
                }
            )
        finally:
            self._busy = False

    def _authorized(self, request: web.Request) -> bool:
        if self._token is None:
            return True
        given = request.headers.get("X-Jarvis-Token", "")
        return hmac.compare_digest(given.encode(), self._token.encode())

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------

    def _finish_turn(self) -> None:
        turn, self._current = self._current, None
        if turn is not None:
            turn.done.set()

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


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)
