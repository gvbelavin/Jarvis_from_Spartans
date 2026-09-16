"""
contracts.py — контракты интерфейсов между модулями проекта «Джарвис».

ВАЖНО (сверено с реальным кодом submodule, а не с ТЗ):

* Модуль Б (`orchestrator/audio_module`, репозиторий Edge_NSU_b_part)
  НЕ содержит класса `AudioEngine` и НЕ содержит функции `run_pipeline()`.
  Его публичные точки входа — обычные модульные функции:

      from wake_word import detect          # detect(np.ndarray[int16]) -> bool
      from comand_record import CommandRecorder
      from audio import start_audio        # start_audio(callback) -> sd.InputStream
      from speaker import identify_speaker # identify_speaker(audio) -> dict
      from stt import transcribe           # transcribe(audio) -> str
      from tts import synthesize           # synthesize(text) -> bytes (WAV)
      from player import play_audio        # play_audio(bytes) -> None

  Поэтому весь доступ к модулю Б идёт через тонкий адаптер
  `orchestrator/audio_adapter.py`, который приводит эти функции
  к контракту `AudioModuleInterface` ниже.

* Модуль Ц (`orchestrator/memory_module`, пакет `jarvis_memory`) реализует
  ровно тот публичный API, который зафиксирован в `DatabaseModuleInterface`:
  `build_context(user_id, transcript)` -> ctx с `ctx.to_messages()`
  и `record_answer(user_id, transcript, answer)`.

* Модуль Г (LLM) получает уже готовый список chat-сообщений:
  `await llm.generate(messages) -> str`.

Эти контракты — то, что нужно передать участникам Б, Ц, Г.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class SpeakerResult:
    """Результат идентификации говорящего (модуль Б -> оркестратор).

    Соответствует фактическому dict, который возвращает
    `audio_module/speaker.py::identify_speaker()`:

        {"user_id": ..., "user_name": ..., "confidence": ...}

    Если голос не опознан (похожесть ниже SPEAKER_THRESHOLD),
    модуль Б возвращает user_id=None и user_name=None.
    """

    user_id: Optional[str]   # ID пользователя ("anton") или None для гостя
    user_name: str           # Отображаемое имя; для гостя — GUEST_NAME
    confidence: float        # Уверенность модели (0.0 - 1.0)


GUEST_NAME = "Гость"


@runtime_checkable
class AudioModuleInterface(Protocol):
    """
    Контракт аудио-модуля (модуль Б).

    Реализуется адаптером `audio_adapter.AudioAdapter`, который
    оборачивает функции реального submodule и переводит их
    в асинхронный вид через `asyncio.to_thread`.
    """

    async def wait_for_wake_word(self) -> None:
        """Блокирует выполнение, пока не детектировано слово «Джарвис»."""
        ...

    async def record_phrase(self) -> "object":
        """Записывает фразу до паузы (VAD).

        Возвращает np.ndarray (int16, 16 kHz, mono) — именно это
        ожидают `speaker.identify_speaker()` и `stt.transcribe()`.
        """
        ...

    async def identify_speaker(self, audio_data) -> SpeakerResult:
        """Инференс Speaker ID (WeSpeaker). Возвращает SpeakerResult."""
        ...

    async def speech_to_text(self, audio_data) -> str:
        """Распознавание речи (whisper.cpp). Возвращает текст фразы."""
        ...

    async def play_tts(self, text: str) -> None:
        """Синтез (Piper) и воспроизведение ответа.

        Блокирует выполнение до конца проговаривания.
        """
        ...


@runtime_checkable
class DatabaseModuleInterface(Protocol):
    """
    Контракт модуля Ц: пользовательская память, профиль, история, RAG.

    Реальная реализация — пакет `jarvis_memory`
    (подмодуль `orchestrator/memory_module`). Импортируется как:

        import jarvis_memory as memory
    """

    def build_context(self, user_id: Optional[str], transcript: str):
        """Формирует контекст для LLM.

        `user_id=None` означает, что модуль Б не узнал говорящего:
        `jarvis_memory` в этом случае отдаёт нейтральный контекст
        и запрещает раскрывать личные данные.

        Возвращает объект ctx, у которого есть:
            ctx.to_messages() -> list[dict[str, str]]
        """
        ...

    def record_answer(
        self,
        user_id: Optional[str],
        transcript: str,
        answer: str,
    ) -> None:
        """Сохраняет пару «вопрос — ответ» в историю конкретного пользователя.

        Вызов обязателен: без него не работает ни «повтори»,
        ни многоходовой диалог.
        """
        ...


@runtime_checkable
class LLMModuleInterface(Protocol):
    """
    Контракт модуля LLM (модуль Г).

    Системный промпт, профиль, расписание, RAG-блоки, история диалога
    и текущий вопрос уже собраны `jarvis_memory` в `ctx.to_messages()`.
    LLM НЕ принимает system_prompt и user_query отдельно.
    """

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Принимает сообщения в стандартном chat-формате:

        [
            {"role": "system",    "content": "Ты — Джарвис..."},
            {"role": "user",      "content": "Предыдущий вопрос"},
            {"role": "assistant", "content": "Предыдущий ответ"},
            {"role": "user",      "content": "Текущий вопрос"},
        ]

        Возвращает только текст ответа ассистента.
        """
        ...
