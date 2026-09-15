"""
contracts.py — Контракты интерфейсов между модулями проекта Джарвис.
Зафиксируйте эти сигнатуры и передайте участникам Б, Ц, Г для реализации.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SpeakerResult:
    """Результат идентификации говорящего (модуль Б -> модуль А)"""
    user_id: Optional[str]  # Уникальный ID ("alex", "elena") или None для гостя
    user_name: str          # Отображаемое имя ("Алексей", "Гость")
    confidence: float       # Уверенность модели (0.0 - 1.0)


@dataclass
class UserContext:
    """Контекст пользователя из БД (модуль Ц -> модуль А)"""
    user_id: str
    user_name: str
    system_prompt: str      # Сформированный промпт с расписанием и фактами


class AudioModuleInterface:
    """
    Интерфейс аудио-модуля (Задача Б).
    Участник Б должен реализовать класс AudioEngine с этими методами.
    """

    async def wait_for_wake_word(self) -> None:
        """
        Блокирующая функция. Возвращает управление,
        когда детектировано кодовое слово (wake word).
        """
        pass

    async def record_phrase(self) -> bytes:
        """
        Записывает фразу пользователя до паузы (VAD).
        Возвращает PCM-байты (16kHz, mono, 16-bit).
        """
        pass

    async def identify_speaker(self, audio_data: bytes) -> SpeakerResult:
        """
        Запускает инференс RKNN-модели Speaker ID на NPU.
        Возвращает результат идентификации.
        """
        pass

    async def speech_to_text(self, audio_data: bytes) -> str:
        """
        Распознаёт речь (STT).
        Возвращает текст сказанной фразы.
        """
        pass

    async def play_tts(self, text: str) -> None:
        """
        Озвучивает текст через динамик (TTS).
        Блокирует выполнение до завершения проговаривания.
        """
        pass


class DatabaseModuleInterface:
    """
    Интерфейс модуля БД (Задача Ц).
    Участник Ц должен реализовать класс UserDatabase с этими методами.
    """

    async def get_user_context(
        self,
        speaker: SpeakerResult,
        user_query: str
    ) -> UserContext:
        """
        Возвращает контекст пользователя (профиль + RAG) для LLM.
        Если пользователь не найден — возвращает контекст для гостя.
        """
        pass


class LLMModuleInterface:
    """
    Интерфейс модуля LLM (Задача Г).
    Участник Г должен реализовать класс LLMEngine с этими методами.
    """

    async def generate_response(
        self,
        system_prompt: str,
        user_query: str
    ) -> str:
        """
        Генерирует текстовый ответ на запрос пользователя.
        Возвращает текст ответа.
        """
        pass
