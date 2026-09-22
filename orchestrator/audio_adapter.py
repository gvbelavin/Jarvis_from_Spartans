from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from contracts import GUEST_NAME, SpeakerResult

# ВАЖНО: импорт из settings, а НЕ из config.
# `config` — это модуль внутри подмодуля аудио (модуль Б), он подключается
# ниже через sys.path. Имя settings выбрано специально, чтобы не было
# коллизии имён: см. пояснение в settings.py.
from settings import AUDIO_MODULE_DIR, DEFAULT_RECORD_TIMEOUT

if str(AUDIO_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIO_MODULE_DIR))

logger = logging.getLogger(__name__)


def _submodule(name: str):
    """
    Импортирует модуль аудиомодуля по имени (`wake_word`, `stt`, ...).

    Ленивый импорт: `speaker.py` тянет torch и WeSpeaker и считает
    эмбеддинги эталонов, `stt.py` грузит модель Whisper, `tts.py` — Piper.
    Держать всё это в момент `import audio_adapter` нельзя, иначе
    оркестратор не импортируется без железа и моделей.
    """
    return importlib.import_module(name)


def _speaker_threshold() -> float:
    """
    Порог уверенности Speaker ID берётся из настроек МОДУЛЯ Б.

    Здесь намеренно нет своей копии значения: единственный источник
    правды — `audio_module/config.py::SPEAKER_THRESHOLD`. Иначе при правке
    чужого модуля наш дубль молча разошёлся бы с оригиналом, и адаптер
    начал бы считать гостями тех, кого модуль Б уверенно узнал.
    """
    return float(_submodule("config").SPEAKER_THRESHOLD)


# Импорты, которые модуль Б выполняет на уровне модуля. Ровно они решают,
# запустится ли аудиотракт вообще: без `wespeaker` не будет Speaker ID,
# без `pywhispercpp` — STT, без `piper` — TTS.
_REQUIRED_MODULES = (
    ("numpy", "numpy"),
    ("sounddevice", "sounddevice"),
    ("soundfile", "soundfile"),
    ("openwakeword.model", "openwakeword"),
    ("wespeaker", "wespeaker"),
    ("torch", "torch"),
    ("pywhispercpp.model", "pywhispercpp"),
    ("piper", "piper-tts"),
)


def check_ready() -> list[str]:
    """
    Проверяет, что аудиомодуль вообще может быть загружен.

    Зачем: без этой проверки первый же вызов уходит в `_submodule("wake_word")`,
    падает сырым `ModuleNotFoundError` из глубины стека (в отдельном потоке,
    обёрнутый в `asyncio.to_thread`), и по логу непонятно, чего не хватает —
    пакета, модели или настроенного `paths.py`.

    Возвращает список проблемных мест; пустой список означает «готово».
    """
    problems: list[str] = []

    if not AUDIO_MODULE_DIR.is_dir():
        problems.append(
            f"нет папки модуля Б: {AUDIO_MODULE_DIR} "
            "(подмодуль не инициализирован? git submodule update --init)"
        )
        return problems

    import importlib.util

    for module_name, package_name in _REQUIRED_MODULES:
        try:
            found = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            found = False

        if not found:
            problems.append(
                f"нет пакета {package_name!r} (нужен для модуля Б): "
                "python -m pip install -r audio_module/requirements.txt"
            )

    # `paths.py` в git модуля Б не хранится (он в .gitignore подмодуля),
    # но `speaker.py` импортирует его на уровне модуля.
    if not (AUDIO_MODULE_DIR / "paths.py").is_file():
        problems.append(
            "нет audio_module/paths.py — speaker.py импортирует его "
            "и без него Speaker ID не загрузится. Нужен REFERENCE_DIR "
            "на каталог audio_dataset/references/"
        )

    # Модели и эталонные записи.
    missing_assets = []
    if not (AUDIO_MODULE_DIR / "whisper.cpp" / "ggml-base.bin").is_file():
        missing_assets.append("whisper.cpp/ggml-base.bin")
    if not (AUDIO_MODULE_DIR / "ru_RU-dmitri-medium.onnx").is_file():
        missing_assets.append("ru_RU-dmitri-medium.onnx")

    if missing_assets:
        problems.append(
            "нет моделей модуля Б: " + ", ".join(missing_assets)
        )

    if not (AUDIO_MODULE_DIR / "audio_dataset" / "references").is_dir():
        problems.append(
            "нет audio_module/audio_dataset/references/ — эталонные "
            "записи голосов, без них Speaker ID всегда вернёт гостя"
        )

    return problems


class _ListenSession:
    def __init__(
        self,
        detect: Callable,
        recorder_cls: type,
        device: Optional[int],
    ) -> None:
        self._detect = detect
        self._recorder_cls = recorder_cls
        self._device = device

        self.state = "waiting"
        self.recorder: Any = None
        self.stream: Any = None

        self.wake = threading.Event()
        self.finished = threading.Event()
        self.error: Optional[BaseException] = None

    def callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug("Audio stream status: %s", status)

        try:
            audio = indata[:, 0].copy()

            if self.state == "waiting":
                if self._detect(audio):
                    logger.info("Wake word detected.")
                    self.recorder = self._recorder_cls()
                    self.state = "recording"
                    self.wake.set()

            elif self.state == "recording":
                self.recorder.process(audio)

                if self.recorder.finished:
                    self.state = "done"
                    self.finished.set()

        except BaseException as exc:
            self.state = "error"
            self.error = exc
            self.wake.set()
            self.finished.set()


class AudioAdapter:
    def __init__(
        self,
        device: Optional[int] = None,
        record_timeout: float = DEFAULT_RECORD_TIMEOUT,
        user_id_map: Optional[dict[str, str]] = None,
    ) -> None:
        self._device = device
        self._record_timeout = record_timeout
        self._user_id_map = user_id_map or {}

        self._session: Optional[_ListenSession] = None
        self._session_lock = threading.Lock()

    async def wait_for_wake_word(self) -> None:
        await asyncio.to_thread(self._start_listening)

    async def record_phrase(self):
        return await asyncio.to_thread(self._finish_recording)

    async def identify_speaker(self, audio_data) -> SpeakerResult:
        raw = await asyncio.to_thread(self._identify_blocking, audio_data)
        return self._to_speaker_result(raw)

    async def speech_to_text(self, audio_data) -> str:
        text = await asyncio.to_thread(self._transcribe_blocking, audio_data)
        return (text or "").strip()

    async def play_tts(self, text: str) -> None:
        await asyncio.to_thread(self._play_tts_blocking, text)

    async def synthesize(self, text: str) -> bytes:
        """WAV от Piper без проигрывания — для веб-интерфейса (web_audio.py)."""
        return await asyncio.to_thread(_submodule("tts").synthesize, text)

    async def listen_once(self) -> tuple[SpeakerResult, str]:
        await self.wait_for_wake_word()

        audio_data = await self.record_phrase()

        if audio_data is None or getattr(audio_data, "size", 1) == 0:
            logger.warning("No speech recorded after wake word.")
            return SpeakerResult(
                user_id=None,
                user_name=GUEST_NAME,
                confidence=0.0,
            ), ""

        speaker_task = asyncio.create_task(
            self.identify_speaker(audio_data)
        )
        stt_task = asyncio.create_task(
            self.speech_to_text(audio_data)
        )

        try:
            speaker_result, transcript = await asyncio.gather(
                speaker_task,
                stt_task,
            )
        except BaseException:
            speaker_task.cancel()
            stt_task.cancel()
            raise

        logger.info(
            "speaker=%s confidence=%.2f transcript=%r",
            speaker_result.user_name,
            speaker_result.confidence,
            transcript,
        )

        return speaker_result, transcript

    def close(self) -> None:
        with self._session_lock:
            self._close_session_locked()

    async def __aenter__(self) -> "AudioAdapter":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _open_stream(self, callback: Callable):
        if self._device is None:
            return _submodule("audio").start_audio(callback)

        import sounddevice as sd

        audio_config = _submodule("config")

        return sd.InputStream(
            samplerate=audio_config.SAMPLE_RATE,
            blocksize=audio_config.CHUNK_SIZE,
            channels=audio_config.CHANNELS,
            dtype="int16",
            callback=callback,
            device=self._device,
        )

    def _start_listening(self) -> None:
        detect = _submodule("wake_word").detect
        recorder_cls = _submodule("comand_record").CommandRecorder

        with self._session_lock:
            self._close_session_locked()

            session = _ListenSession(
                detect=detect,
                recorder_cls=recorder_cls,
                device=self._device,
            )

            try:
                session.stream = self._open_stream(session.callback)
                session.stream.start()
            except BaseException:
                if session.stream is not None:
                    try:
                        session.stream.close()
                    except Exception:
                        logger.debug(
                            "Failed to close audio stream.",
                            exc_info=True,
                        )
                raise

            self._session = session

        logger.info("Waiting for wake word.")

        session.wake.wait()

        if session.error is not None:
            with self._session_lock:
                self._close_session_locked()
            raise session.error

    def _finish_recording(self):
        session = self._session

        if session is None:
            raise RuntimeError(
                "record_phrase() called before wait_for_wake_word()."
            )

        if not session.wake.is_set():
            session.wake.wait()

        if session.error is not None:
            with self._session_lock:
                self._close_session_locked()
            raise session.error

        if not session.finished.wait(self._record_timeout):
            logger.warning(
                "Recording timed out after %.1f seconds.",
                self._record_timeout,
            )

        audio = None

        if session.recorder is not None and session.recorder.audio_chunks:
            audio = session.recorder.get_audio()

        with self._session_lock:
            self._close_session_locked()

        return audio

    def _close_session_locked(self) -> None:
        session = self._session
        self._session = None

        if session is None or session.stream is None:
            return

        try:
            session.stream.stop()
            session.stream.close()
        except Exception:
            logger.debug(
                "Failed to close audio stream.",
                exc_info=True,
            )

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

    def _to_speaker_result(self, raw: Any) -> SpeakerResult:
        if isinstance(raw, SpeakerResult):
            return raw

        if not isinstance(raw, dict):
            raise TypeError(
                "speaker.identify_speaker() must return dict or SpeakerResult; "
                f"got {type(raw).__name__}"
            )

        user_id = raw.get("user_id")

        if user_id is not None:
            user_id = self._user_id_map.get(user_id, user_id)

        confidence = raw.get("confidence")
        confidence = float(confidence) if confidence is not None else 0.0

        if not user_id or confidence < _speaker_threshold():
            return SpeakerResult(
                user_id=None,
                user_name=GUEST_NAME,
                confidence=confidence,
            )

        user_name = raw.get("user_name") or user_id

        return SpeakerResult(
            user_id=str(user_id),
            user_name=str(user_name),
            confidence=confidence,
        )