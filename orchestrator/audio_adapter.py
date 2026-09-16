"""
audio_adapter.py — тонкий адаптер между оркестратором (модуль А)
и аудиомодулем (модуль Б, git submodule `orchestrator/audio_module`).

ЗАЧЕМ НУЖЕН ЭТОТ ФАЙЛ
---------------------

В аудиомодуле НЕТ класса `AudioEngine` и НЕТ функции `run_pipeline()`.
Точка входа `pipeline.py::main()` — это готовый сценарий целиком:
он печатает `"Command: ..."`, жёстко подставляет ответ
`"Я получил твою команду."`, сам вызывает Piper и возвращает `None`.
Переиспользовать его как функцию нельзя.

Поэтому адаптер вызывает те же самые функции модуля Б по отдельности,
в том же порядке, что и `pipeline.py`, но без его печати и без его
заглушки ответа. Ни один алгоритм внутри submodule не меняется
и ни один файл submodule не редактируется.

ФАКТИЧЕСКИЕ ФУНКЦИИ МОДУЛЯ Б (сверено с кодом):

    audio.start_audio(callback)          -> sounddevice.InputStream (не запущен)
    wake_word.detect(np.ndarray[int16])  -> bool
    comand_record.CommandRecorder()      -> .process(audio), .finished, .get_audio()
    speaker.identify_speaker(audio)      -> {"user_id", "user_name", "confidence"}
    stt.transcribe(audio)                -> str
    tts.synthesize(text)                 -> bytes (WAV в памяти)
    player.play_audio(wav_bytes)         -> None (блокирующий)

Модули Б используют «плоские» импорты (`import config`), поэтому папка
`audio_module` добавляется в `sys.path`, а её модули импортируются
как top-level. Импорты ленивые: `speaker.py` тянет torch+WeSpeaker
и считает эмбеддинги эталонов, `stt.py` грузит модель Whisper,
`tts.py` — модель Piper. Держать всё это в момент `import audio_adapter`
не нужно, иначе оркестратор нельзя будет даже импортировать без железа.

ТРЕБОВАНИЕ К ОКРУЖЕНИЮ
----------------------

`speaker.py` импортирует `paths.py`, где задан `REFERENCE_DIR`
(каталог `audio_dataset/references/`). Файл `paths.py` в git НЕ хранится
(он в `.gitignore`), поэтому модуль Б нужно предварительно настроить
по его README: положить модели, эталонные записи и создать `paths.py`.
Адаптер этим не занимается — он только связывает готовый модуль Б.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from contracts import GUEST_NAME, SpeakerResult

# Папка orchestrator/ (там же лежат contracts.py, app.py)
BASE_DIR = Path(__file__).resolve().parent

# Папка git submodule с аудиомодулем
AUDIO_MODULE_DIR = BASE_DIR / "audio_module"

# Модули Б импортируются как top-level (`import config`), поэтому их папка
# должна быть в sys.path. Ставим её в начало, чтобы `import config` попал
# именно в config.py модуля Б, а не в какой-нибудь одноимённый модуль.
if str(AUDIO_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIO_MODULE_DIR))

# Модуль Б завершает запись только по тишине и не имеет собственного
# таймаута: если после wake word никто ничего не скажет, запись не
# закончится никогда. Адаптер добавляет страховочный таймаут сверху.
DEFAULT_RECORD_TIMEOUT = 10.0

logger = logging.getLogger(__name__)


def _submodule(name: str):
    """Импортирует модуль аудиомодуля по имени (`wake_word`, `stt`, ...)."""
    return importlib.import_module(name)


class _ListenSession:
    """
    Состояние одного цикла «ожидание wake word -> запись команды».

    Повторяет конечный автомат `pipeline.py` (waiting -> recording),
    но не печатает результат и не вызывает TTS: его задача — только
    отдать PCM-аудио команды наружу.

    Один и тот же InputStream остаётся открытым и на этапе ожидания,
    и на этапе записи. Так сделано в самом `pipeline.py`, и это важно:
    если переоткрывать поток между wake word и записью, начало команды
    (примерно один блок = 80 мс) теряется.
    """

    def __init__(self, detect: Callable, recorder_cls: type, device: Optional[int]):
        self._detect = detect
        self._recorder_cls = recorder_cls
        self._device = device

        self.state = "waiting"
        self.recorder: Any = None
        self.stream: Any = None

        self.wake = threading.Event()      # взведён, когда услышали «Джарвис»
        self.finished = threading.Event()  # взведён, когда команда записана
        self.error: Optional[BaseException] = None

    def callback(self, indata, frames, time_info, status) -> None:
        """Callback микрофона. Работает в аудиопотоке sounddevice."""
        if status:
            # Переполнения буфера и прочие проблемы устройства — не повод
            # ломать цикл, но знать о них нужно.
            logger.debug("Аудиопоток: %s", status)

        try:
            # Как в pipeline.py: берём один канал
            audio = indata[:, 0].copy()

            if self.state == "waiting":
                if self._detect(audio):
                    logger.info("Wake word обнаружен — записываю команду.")
                    self.recorder = self._recorder_cls()
                    self.state = "recording"
                    self.wake.set()

            elif self.state == "recording":
                self.recorder.process(audio)
                if self.recorder.finished:
                    self.state = "done"
                    self.finished.set()

        except BaseException as exc:  # noqa: BLE001 — ошибку надо вынести в поток orchestrator
            self.state = "error"
            self.error = exc
            self.wake.set()
            self.finished.set()


class AudioAdapter:
    """
    Приводит функции модуля Б к асинхронному контракту
    `contracts.AudioModuleInterface`.

    Все вызовы модуля Б блокирующие (микрофон, NPU/CPU-инференс, звук),
    поэтому каждый из них выполняется в отдельном потоке через
    `asyncio.to_thread`, чтобы не блокировать event loop оркестратора.
    """

    def __init__(
        self,
        device: Optional[int] = None,
        record_timeout: float = DEFAULT_RECORD_TIMEOUT,
        user_id_map: Optional[dict[str, str]] = None,
    ) -> None:
        """
        :param device: переопределить номер аудиоустройства. По умолчанию
            используется `audio_module/config.py::DEVICE_NUMBER`.
        :param record_timeout: максимум секунд на запись команды после
            wake word.
        :param user_id_map: соответствие «имя эталонной папки модуля Б ->
            user_id модуля памяти». Модуль Б берёт идентификатор из имени
            каталога в `audio_dataset/references/` (например `shelestov`),
            а `jarvis_memory` знает пользователей как `anton`/`masha`.
            Если словарь не задан, ID передаётся как есть.
        """
        self._device = device
        self._record_timeout = record_timeout
        self._user_id_map = user_id_map or {}

        self._session: Optional[_ListenSession] = None
        self._session_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Публичный контракт (см. contracts.AudioModuleInterface)
    # ------------------------------------------------------------------

    async def wait_for_wake_word(self) -> None:
        """Открывает микрофон и ждёт слово «Джарвис».

        Возвращает управление сразу после детекции, НЕ закрывая поток:
        запись команды продолжается в том же InputStream.
        """
        await asyncio.to_thread(self._start_listening)

    async def record_phrase(self):
        """Дожидается конца команды и возвращает PCM-аудио.

        :return: `np.ndarray` (int16, 16 kHz, mono) — ровно тот формат,
            который ожидают `speaker.identify_speaker()` и `stt.transcribe()`.
            `None`, если речи так и не было.
        """
        return await asyncio.to_thread(self._finish_recording)

    async def identify_speaker(self, audio_data) -> SpeakerResult:
        """Speaker ID по PCM-аудио (WeSpeaker, модуль Б)."""
        raw = await asyncio.to_thread(self._identify_blocking, audio_data)
        return self._to_speaker_result(raw)

    async def speech_to_text(self, audio_data) -> str:
        """STT по PCM-аудио (whisper.cpp, модуль Б)."""
        text = await asyncio.to_thread(self._transcribe_blocking, audio_data)
        return (text or "").strip()

    async def play_tts(self, text: str) -> None:
        """Синтез (Piper) и воспроизведение ответа. Блокирует до конца речи."""
        await asyncio.to_thread(self._play_tts_blocking, text)

    async def listen_once(self) -> tuple[SpeakerResult, str]:
        """
        Полный аудиоэтап одной команды:

            wake word -> запись команды (VAD)
                      -> параллельно Speaker ID + STT

        :return: `(SpeakerResult, transcript)`. При пустом/несостоявшемся
            аудио возвращает `(гость, "")`, чтобы оркестратор произнёс
            «я не расслышал» и вернулся к ожиданию wake word.
        """
        await self.wait_for_wake_word()

        audio_data = await self.record_phrase()

        if audio_data is None or getattr(audio_data, "size", 1) == 0:
            logger.warning("Команда не записана: речи после wake word не было.")
            return SpeakerResult(user_id=None, user_name=GUEST_NAME, confidence=0.0), ""

        speaker_task = asyncio.create_task(self.identify_speaker(audio_data))
        stt_task = asyncio.create_task(self.speech_to_text(audio_data))

        try:
            speaker_result, transcript = await asyncio.gather(
                speaker_task, stt_task
            )
        except BaseException:
            # gather не отменяет «соседа» при падении одной задачи
            speaker_task.cancel()
            stt_task.cancel()
            raise

        logger.info(
            "Аудио: speaker=%s (confidence=%.2f), transcript=%r",
            speaker_result.user_name,
            speaker_result.confidence,
            transcript,
        )
        return speaker_result, transcript

    def close(self) -> None:
        """Закрывает микрофон, если он остался открытым."""
        with self._session_lock:
            self._close_session_locked()

    async def __aenter__(self) -> "AudioAdapter":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Синхронная часть: работа с модулем Б
    # ------------------------------------------------------------------

    def _open_stream(self, callback: Callable):
        """Создаёт InputStream. По умолчанию — ровно как в модуле Б."""
        if self._device is None:
            return _submodule("audio").start_audio(callback)

        # Устройство переопределено: создаём поток с теми же параметрами
        # модуля Б, но с другим device.
        import sounddevice as sd

        cfg = _submodule("config")
        return sd.InputStream(
            samplerate=cfg.SAMPLE_RATE,
            blocksize=cfg.CHUNK_SIZE,
            channels=cfg.CHANNELS,
            dtype="int16",
            callback=callback,
            device=self._device,
        )

    def _start_listening(self) -> None:
        """Открывает микрофон и ждёт wake word (блокирующая часть)."""
        # Модели грузятся один раз — при первом обращении, дальше кэшируются
        # интерпретатором Python в sys.modules.
        detect = _submodule("wake_word").detect
        recorder_cls = _submodule("comand_record").CommandRecorder

        with self._session_lock:
            self._close_session_locked()

            session = _ListenSession(detect, recorder_cls, self._device)
            try:
                # start_audio() возвращает НЕ запущенный InputStream:
                # в pipeline.py его запускает контекстный менеджер `with`.
                session.stream = self._open_stream(session.callback)
                session.stream.start()
            except BaseException:
                if session.stream is not None:
                    try:
                        session.stream.close()
                    except Exception:  # noqa: BLE001
                        logger.debug("Не удалось закрыть аудиопоток.", exc_info=True)
                raise

            self._session = session

        logger.info("Ожидание wake word «Джарвис»...")

        # Ждём либо wake word, либо ошибку из callback.
        session.wake.wait()

        if session.error is not None:
            with self._session_lock:
                self._close_session_locked()
            raise session.error

    def _finish_recording(self):
        """Дожидается конца команды и забирает PCM (блокирующая часть)."""
        session = self._session

        if session is None:
            raise RuntimeError(
                "record_phrase() вызван без предварительного "
                "wait_for_wake_word(): микрофон не открыт."
            )

        if not session.wake.is_set():
            session.wake.wait()

        if session.error is not None:
            with self._session_lock:
                self._close_session_locked()
            raise session.error

        if not session.finished.wait(self._record_timeout):
            logger.warning(
                "Запись команды прервана по таймауту %.1f с: "
                "тишина так и не наступила.",
                self._record_timeout,
            )

        audio = None
        if session.recorder is not None and session.recorder.audio_chunks:
            audio = session.recorder.get_audio()

        with self._session_lock:
            self._close_session_locked()

        return audio

    def _close_session_locked(self) -> None:
        """Закрывает поток текущей сессии. Вызывать под `_session_lock`."""
        session = self._session
        self._session = None

        if session is None or session.stream is None:
            return

        try:
            session.stream.stop()
            session.stream.close()
        except Exception:  # noqa: BLE001
            logger.debug("Ошибка при закрытии аудиопотока.", exc_info=True)

    @staticmethod
    def _identify_blocking(audio_data):
        return _submodule("speaker").identify_speaker(audio_data)

    @staticmethod
    def _transcribe_blocking(audio_data) -> str:
        return _submodule("stt").transcribe(audio_data)

    @staticmethod
    def _play_tts_blocking(text: str) -> None:
        wav_bytes = _submodule("tts").synthesize(text)
        _submodule("player").play_audio(wav_bytes)

    # ------------------------------------------------------------------
    # Приведение ответов модуля Б к контрактам оркестратора
    # ------------------------------------------------------------------

    def _to_speaker_result(self, raw: Any) -> SpeakerResult:
        """`speaker.identify_speaker()` возвращает dict — приводим к dataclass."""
        if isinstance(raw, SpeakerResult):
            return raw

        if not isinstance(raw, dict):
            raise TypeError(
                "Модуль Б должен вернуть dict с ключами "
                "user_id/user_name/confidence, получено: "
                f"{type(raw).__name__}"
            )

        user_id = raw.get("user_id")
        if user_id is not None:
            user_id = self._user_id_map.get(user_id, user_id)

        # Модуль Б при неуверенном распознавании отдаёт user_name=None —
        # оркестратор и модуль памяти ожидают строку.
        user_name = raw.get("user_name") or user_id or GUEST_NAME

        confidence = raw.get("confidence")
        confidence = float(confidence) if confidence is not None else 0.0

        return SpeakerResult(
            user_id=user_id,
            user_name=str(user_name),
            confidence=confidence,
        )
