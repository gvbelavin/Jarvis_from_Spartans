"""
mocks_for_testing.py — изолированные заглушки для проверки оркестратора
без микрофона, аудиомоделей и локальной LLM.

Запуск полной связки на заглушках:

    python app.py --mock --once

Заглушки реализуют ровно те же контракты, что и реальные модули
(см. contracts.py), поэтому `app.py` в обоих режимах выполняется по
одному и тому же коду:

    AudioEngineMock    -> AudioModuleInterface      (модуль Б)
    MemoryModuleMock   -> DatabaseModuleInterface   (модуль Ц)
    LLMEngineMock      -> LLMModuleInterface        (модуль Г)

Модуль памяти при возможности берётся настоящий — см. app.py
(`build_mock_orchestrator`). Заглушка ниже нужна только для случая,
когда подмодуль `memory_module` не инициализирован.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from contracts import GUEST_NAME, SpeakerResult

logger = logging.getLogger("jarvis.mocks")


class AudioEngineMock:
    """Заглушка модуля Б: wake word, VAD, Speaker ID, STT, TTS."""

    def __init__(
        self,
        transcript: str = "Какое у меня сегодня расписание?",
        speaker: Optional[SpeakerResult] = None,
        max_commands: Optional[int] = None,
    ) -> None:
        self.transcript = transcript
        self.speaker = speaker or SpeakerResult(
            user_id="daniil", user_name="Даниил", confidence=0.91
        )
        self.max_commands = max_commands
        self.spoken: list[str] = []

    async def wait_for_wake_word(self) -> None:
        logger.info("[mock audio] Ожидание wake word «Джарвис»...")
        await asyncio.sleep(0.2)
        logger.info("[mock audio] Wake word обнаружен.")

    async def record_phrase(self) -> Any:
        logger.info("[mock audio] Запись команды (VAD)...")
        await asyncio.sleep(0.2)
        # Модуль Б отдаёт np.ndarray; для заглушки достаточно непустого
        # объекта с атрибутом size, как у массива.
        return _FakeAudio(size=16000)

    async def identify_speaker(self, audio_data: Any) -> SpeakerResult:
        logger.info("[mock audio] Speaker ID (WeSpeaker)...")
        await asyncio.sleep(0.05)
        return self.speaker

    async def speech_to_text(self, audio_data: Any) -> str:
        logger.info("[mock audio] STT (whisper.cpp)...")
        await asyncio.sleep(0.1)
        return self.transcript

    async def play_tts(self, text: str) -> None:
        self.spoken.append(text)
        logger.info("[mock audio] Динамик (Piper TTS): %s", text)
        await asyncio.sleep(0.05)

    def stream_tools(self):
        """Потоковый режим без моделей: wake word по громкому звуку."""
        return _MockWakeWord(), _MockRecorder, 1280

    async def synthesize(self, text: str) -> bytes:
        """Вместо голоса Piper — короткий гудок, чтобы веб-режим было слышно."""
        return _beep_wav()

    async def listen_once(self) -> tuple[SpeakerResult, str]:
        """Тот же контракт, что у audio_adapter.AudioAdapter.listen_once()."""
        await self.wait_for_wake_word()
        audio = await self.record_phrase()

        speaker, transcript = await asyncio.gather(
            self.identify_speaker(audio),
            self.speech_to_text(audio),
        )
        return speaker, transcript

    def close(self) -> None:
        logger.info("[mock audio] Микрофон закрыт.")


class _MockWakeWord:
    """Срабатывает на любой достаточно громкий чанк — вместо openWakeWord."""

    THRESHOLD = 800

    def __call__(self, frame) -> bool:
        import numpy as np

        return bool(np.abs(frame).mean() > self.THRESHOLD)


class _MockRecorder:
    """Мини-VAD вместо CommandRecorder модуля Б: тишина = конец фразы."""

    SILENCE_THRESHOLD = 500
    SILENCE_CHUNKS = 13

    def __init__(self) -> None:
        self.audio_chunks: list[Any] = []
        self.finished = False
        self._started = False
        self._silence = 0

    def process(self, frame) -> None:
        import numpy as np

        self.audio_chunks.append(frame.copy())

        if np.abs(frame).mean() > self.SILENCE_THRESHOLD:
            self._started = True
            self._silence = 0
        elif self._started:
            self._silence += 1
            if self._silence >= self.SILENCE_CHUNKS:
                self.finished = True

    def get_audio(self):
        import numpy as np

        return np.concatenate(self.audio_chunks)


def _beep_wav(seconds: float = 0.3, freq: float = 660.0) -> bytes:
    import io
    import math
    import struct
    import wave

    rate = 22050
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate)))
        for i in range(int(seconds * rate))
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)
    return buf.getvalue()


class _FakeAudio:
    """Минимальная имитация np.ndarray: `size` и `copy()`."""

    def __init__(self, size: int = 16000) -> None:
        self.size = size

    def copy(self) -> "_FakeAudio":
        return _FakeAudio(self.size)

    def __len__(self) -> int:
        return self.size


class _MockContext:
    """Минимальный аналог jarvis_memory.PromptContext."""

    def __init__(self, system_prompt: str, transcript: str, user_id: Optional[str]) -> None:
        self.system_prompt = system_prompt
        self.history: list[dict[str, str]] = []
        self.transcript = transcript
        self.user_id = user_id
        self.intent = "general"
        self.debug = {"profile_found": user_id is not None}

    def to_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            *self.history,
            {"role": "user", "content": self.transcript},
        ]


class MemoryModuleMock:
    """Изолированная заглушка модуля Ц: профили, расписание, история."""

    def __init__(self) -> None:
        self.profiles = {
            "daniil": {
                "name": "Даниил",
                "schedule": "В 16:00 лекция по матанализу, в 19:30 тренировка.",
            },
            "fedor": {
                "name": "Фёдор",
                "schedule": "В 09:00 английский, в 14:00 встреча с научруком.",
            },
        }
        self.history: dict[str, list[dict[str, str]]] = {}

    def build_context(self, user_id: Optional[str], transcript: str) -> _MockContext:
        """user_id=None — гость: нейтральный контекст без личных данных."""
        profile = self.profiles.get(user_id) if user_id else None

        if profile is None:
            system_prompt = (
                "Ты — Джарвис, локальный голосовой ассистент. "
                "Говорящий не опознан: личные данные не раскрывай, "
                "попроси подойти ближе и повторить."
            )
        else:
            system_prompt = (
                "Ты — Джарвис, локальный голосовой ассистент. "
                f"С тобой говорит {profile['name']}. "
                f"Расписание на сегодня: {profile['schedule']} "
                "Отвечай коротко, одно-два предложения."
            )

        ctx = _MockContext(system_prompt, transcript, user_id)
        ctx.history = list(self.history.get(user_id or "_unknown", []))
        return ctx

    def record_answer(
        self,
        user_id: Optional[str],
        transcript: str,
        answer: str,
    ) -> None:
        buf = self.history.setdefault(user_id or "_unknown", [])
        buf.append({"role": "user", "content": transcript})
        buf.append({"role": "assistant", "content": answer})
        del buf[:-6]  # последние 3 обмена

    def clear(self, user_id: Optional[str] = None) -> None:
        if user_id is None:
            self.history.clear()
        else:
            self.history.pop(user_id or "_unknown", None)


class LLMEngineMock:
    """
    Заглушка модуля Г: синхронный движок, обёрнутый адаптером.

    Специально сделан синхронным: это проверяет, что оркестратор
    действительно уводит генерацию в отдельный поток и не блокирует
    event loop (в логе перед ответом продолжает идти heartbeat).
    """

    def generate(self, messages: list[dict[str, str]]) -> str:
        import time

        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        query = user_messages[-1] if user_messages else ""
        turns = sum(1 for m in messages if m["role"] == "assistant")

        time.sleep(0.3)  # имитация локального инференса

        logger.info(
            "[mock llm] сообщений=%d, прошлых обменов=%d, вопрос=%r",
            len(messages),
            turns,
            query,
        )

        if "расписан" in query.lower():
            if "Даниил" in messages[0]["content"]:
                return "Даниил, в четыре часа дня лекция по матанализу, а в половине восьмого тренировка."
            if "Фёдор" in messages[0]["content"]:
                return "Фёдор, в девять утра английский, а в два часа встреча с научруком."
            return "Я не знаю, кто спрашивает, поэтому расписание показать не могу."

        if "повтор" in query.lower():
            return "Повторяю: это тестовый ответ локального Джарвиса."

        return "Я услышал ваш запрос. Это тестовый ответ локального Джарвиса."


async def _self_test() -> None:
    """Проверка самих заглушек, без оркестратора."""
    audio = AudioEngineMock()
    speaker, transcript = await audio.listen_once()
    assert isinstance(speaker, SpeakerResult)
    assert transcript

    memory = MemoryModuleMock()
    ctx = memory.build_context(speaker.user_id, transcript)
    assert ctx.to_messages()[0]["role"] == "system"
    assert ctx.to_messages()[-1]["content"] == transcript

    guest_ctx = memory.build_context(None, transcript)
    assert "не опознан" in guest_ctx.system_prompt

    llm = LLMEngineMock()
    answer = llm.generate(ctx.to_messages())
    assert isinstance(answer, str)

    memory.record_answer(speaker.user_id, transcript, answer)
    assert len(memory.build_context(speaker.user_id, transcript).history) == 2

    print("mocks_for_testing: OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(_self_test())