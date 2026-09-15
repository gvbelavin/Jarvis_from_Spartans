"""
app.py — Оркестратор проекта Джарвис (Задача А).
Собирает все модули в единое приложение, управляет циклом состояний.

ИНТЕГРАЦИЯ МОДУЛЕЙ:
- Модуль Б (аудио): интегрирован как git submodule из Edge_NSU_b_part
- Модуль Ц (БД): раскомментировать import при готовности
- Модуль Г (LLM): раскомментировать import при готовности
"""

import asyncio
import logging
import time
from typing import Optional

# Импорты интерфейсов
from contracts import SpeakerResult, UserContext

# =============================================================================
# IMPORTS: Модули
# =============================================================================

# Модуль Б (аудио): git submodule из Edge_NSU_b_part
from audio_module import AudioEngine  # <-- ИНТЕГРИРОВАНО

# Модуль Ц (БД)
# from database_module import UserDatabase  # <-- РАСКОММЕНТИРОВАТЬ ПРИ ГОТОВНОСТИ

# Модуль Г (LLM)
# from llm_module import LLMEngine  # <-- РАСКОММЕНТИРОВАТЬ ПРИ ГОТОВНОСТИ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# =============================================================================
# ЗАГЛУШКИ МОДУЛЕЙ (MOCKS)
# Замените эти классы на реальные импорты из модулей Б, Ц, Г
# =============================================================================

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


# =============================================================================
# ОРКЕСТРАТОР (Задача А)
# =============================================================================

class JarvisOrchestrator:
    """
    Главный цикл управления системой Джарвис.
    Управляет состояниями: IDLE -> RECORDING -> PROCESSING -> SPEAKING -> IDLE
    """

    def __init__(self):
        # Инициализация модулей (замените моки на реальные классы)
        self.voice = AudioEngine()       # <-- ИНТЕГРИРОВАНО: AudioEngine из audio_module
        self.db = UserDatabaseMock()     # <-- UserDatabase() после готовности модуля Ц
        self.llm = LLMEngineMock()       # <-- LLMEngine() после готовности модуля Г
        self.is_running = True

    async def run_cycle(self):
        """Основной бесконечный цикл обработки запросов"""
        logging.info("=" * 60)
        logging.info("Джарвис запущен. Ожидание команды...")
        logging.info("=" * 60)

        while self.is_running:
            try:
                # --- ШАГ 1: Ожидание активационной фразы ---
                await self.voice.wait_for_wake_word()

                # --- ШАГ 2: Запись пользовательской фразы ---
                audio_frames = await self.voice.record_phrase()

                t_start = time.perf_counter()

                # --- ШАГ 3: Параллельный запуск Speaker ID (на NPU) и STT ---
                speaker_task = asyncio.create_task(
                    self.voice.identify_speaker(audio_frames)
                )
                stt_task = asyncio.create_task(
                    self.voice.speech_to_text(audio_frames)
                )

                speaker_res, transcript = await asyncio.gather(
                    speaker_task,
                    stt_task
                )

                logging.info(
                    f"Распознан: {speaker_res.user_name} "
                    f"(уверенность: {speaker_res.confidence:.2f}) | "
                    f"Текст: '{transcript}'"
                )

                # --- ШАГ 4: Извлечение профиля из БД (RAG) ---
                context = await self.db.get_user_context(speaker_res, transcript)
                logging.info(f"Пользователь: {context.user_name}")

                # --- ШАГ 5: Генерация ответа LLM ---
                reply_text = await self.llm.generate_response(
                    context.system_prompt,
                    transcript
                )

                latency = time.perf_counter() - t_start
                logging.info(f"Общее время обработки до озвучки: {latency:.2f} с")

                # --- ШАГ 6: Воспроизведение ответа в динамик ---
                await self.voice.play_tts(reply_text)

            except KeyboardInterrupt:
                logging.info("Оркестратор остановлен пользователем.")
                self.is_running = False
                break

            except Exception as e:
                logging.error(f"Ошибка в цикле оркестратора: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    def stop(self):
        """Метод для остановки оркестратора"""
        self.is_running = False


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

async def main():
    orchestrator = JarvisOrchestrator()
    try:
        await orchestrator.run_cycle()
    except KeyboardInterrupt:
        logging.info("Получен сигнал остановки (Ctrl+C).")
    finally:
        orchestrator.stop()
        logging.info("Джарвис завершил работу.")


if __name__ == "__main__":
    asyncio.run(main())
