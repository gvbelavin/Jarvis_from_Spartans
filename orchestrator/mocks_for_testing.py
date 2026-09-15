"""
mocks_for_testing.py — Изолированные моки для тестирования оркестратора.
Используйте этот файл, чтобы тестировать app.py без реальных модулей Б, Ц, Г.
"""

import asyncio
import logging
from contracts import SpeakerResult, UserContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AudioEngineMock:
    """Заглушка модуля Б: Wake Word, VAD, NPU Speaker ID, STT, TTS"""

    async def wait_for_wake_word(self) -> None:
        logging.info("Ожидание wake-word 'Джарвис'...")
        await asyncio.sleep(2.0)
        logging.info(">>> Wake-word обнаружен!")

    async def record_phrase(self) -> bytes:
        logging.info("Запись голосовой команды (VAD)...")
        await asyncio.sleep(1.5)
        return b"fake_pcm_audio_data"

    async def identify_speaker(self, audio_data: bytes) -> SpeakerResult:
        logging.info("NPU: Инференс Speaker ID модели...")
        await asyncio.sleep(0.05)  # Имитация ~50 мс инференса
        return SpeakerResult(user_id="user_1", user_name="Алексей", confidence=0.89)

    async def speech_to_text(self, audio_data: bytes) -> str:
        logging.info("STT: Распознавание речи...")
        await asyncio.sleep(0.3)
        return "Какое у меня сегодня расписание?"

    async def play_tts(self, text: str) -> None:
        logging.info(f"Динамик (TTS): '{text}'")
        await asyncio.sleep(1.5)


class UserDatabaseMock:
    """Заглушка модуля Ц: Профили и векторная БД"""

    def __init__(self):
        self.profiles = {
            "user_1": {
                "name": "Алексей",
                "schedule": "В 14:00 защита проекта на Firefly RK3588S, в 18:00 спортзал."
            },
            "user_2": {
                "name": "Елена",
                "schedule": "В 10:00 созвон по работе, вечером покупка продуктов."
            }
        }

    async def get_user_context(
        self,
        speaker: SpeakerResult,
        user_query: str
    ) -> UserContext:
        user_data = self.profiles.get(speaker.user_id)
        if not user_data:
            return UserContext(
                user_id="guest",
                user_name="Гость",
                system_prompt="Пользователь не опознан. Отвечай нейтрально и вежливо."
            )

        prompt = (
            f"Ты — локальный домашний ассистент Джарвис. "
            f"Ты разговариваешь с жильцом по имени {user_data['name']}. "
            f"Вот его актуальное расписание на сегодня: {user_data['schedule']}. "
            f"Отвечай кратко, дружелюбно, 1-2 предложениями."
        )
        return UserContext(
            user_id=speaker.user_id,
            user_name=user_data["name"],
            system_prompt=prompt
        )


class LLMEngineMock:
    """Заглушка модуля Г: Qwen2.5 через rknn-llm или llama.cpp"""

    async def generate_response(
        self,
        system_prompt: str,
        user_query: str
    ) -> str:
        logging.info("LLM: Генерация ответа...")
        await asyncio.sleep(0.8)
        return "Алексей, сегодня в 14:00 у вас защита проекта на Firefly RK3588S, а в 18:00 спортзал."
