import asyncio
import logging
import sys
import time
from pathlib import Path

from contracts import SpeakerResult

# Корень папки orchestrator
BASE_DIR = Path(__file__).resolve().parent

# Позволяет импортировать пакет jarvis_memory,
# находящийся в orchestrator/memory_module/jarvis_memory
MEMORY_MODULE_DIR = BASE_DIR / "memory_module"
if MEMORY_MODULE_DIR.exists():
    sys.path.insert(0, str(MEMORY_MODULE_DIR))

# Модуль Б: git submodule Edge_NSU_b_part.
# ВАЖНО: имя класса/пути нужно сверить с реальной структурой репозитория друга.
from audio_module import AudioEngine

# Модуль Ц: git submodule LessVegetables/jarvis-memory.
# Пока submodule не добавлен, этот импорт упадёт.
import jarvis_memory as memory

# Модуль Г — подключите, когда участник Г пришлёт реализацию.
# Например:
# from llm_module import LLMEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class LLMEngineMock:
    """
    Временная заглушка модуля Г.

    Финальный контракт модуля Г:
        await llm.generate(messages) -> str

    messages имеет формат:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
    """

    async def generate(self, messages: list[dict[str, str]]) -> str:
        logging.info("LLM: генерация ответа...")

        for message in reversed(messages):
            if message.get("role") == "user":
                query = message.get("content", "")
                break
        else:
            query = ""

        await asyncio.sleep(0.8)

        if "расписан" in query.lower():
            return (
                "Сегодня у вас в 14:00 защита проекта "
                "на Firefly RK3588S, а в 18:00 спортзал."
            )

        return "Я услышал ваш запрос. Пока это тестовый ответ локального Джарвиса."


class JarvisOrchestrator:
    """
    Главный цикл:

    Wake word
        -> запись команды
        -> параллельно: Speaker ID + STT
        -> memory.build_context(user_id, transcript)
        -> LLM.generate(ctx.to_messages())
        -> memory.record_answer(user_id, transcript, answer)
        -> TTS
    """

    def __init__(self) -> None:
        self.voice = AudioEngine()
        self.llm = LLMEngineMock()
        self.is_running = True

    async def process_command(self, audio_frames: bytes) -> None:
        """
        Обрабатывает один аудиофрагмент после wake word и VAD.
        Speaker ID и STT стартуют параллельно.
        """

        started_at = time.perf_counter()

        speaker_task = asyncio.create_task(
            self.voice.identify_speaker(audio_frames)
        )
        stt_task = asyncio.create_task(
            self.voice.speech_to_text(audio_frames)
        )

        speaker_result, transcript = await asyncio.gather(
            speaker_task,
            stt_task,
        )

        if not isinstance(speaker_result, SpeakerResult):
            raise TypeError(
                "Метод identify_speaker() должен возвращать SpeakerResult "
                "из contracts.py"
            )

        transcript = transcript.strip()

        if not transcript:
            await self.voice.play_tts(
                "Извините, я не смог разобрать команду. Повторите, пожалуйста."
            )
            return

        user_id = speaker_result.user_id

        logging.info(
            "Speaker ID: %s | confidence=%.2f | transcript=%r",
            speaker_result.user_name,
            speaker_result.confidence,
            transcript,
        )

        # Модуль Ц:
        # Собирает system prompt, пользовательские факты и историю диалога.
        # user_id может быть None, если голос не распознан.
        ctx = memory.build_context(user_id, transcript)

        # Модуль Г:
        # Получает готовый стандартный список chat messages.
        messages = ctx.to_messages()
        answer = await self.llm.generate(messages)

        answer = answer.strip()

        if not answer:
            answer = "Извините, я не смог подготовить ответ."

        # Модуль Ц:
        # Сохраняет вопрос и ответ в историю конкретного пользователя.
        # Для неизвестного пользователя user_id == None;
        # модуль памяти должен корректно это обработать.
        memory.record_answer(user_id, transcript, answer)

        total_latency = time.perf_counter() - started_at

        logging.info(
            "Ответ готов за %.2f с | user_id=%r | answer=%r",
            total_latency,
            user_id,
            answer,
        )

        await self.voice.play_tts(answer)

    async def run(self) -> None:
        """Бесконечный цикл голосового ассистента."""

        logging.info("=" * 60)
        logging.info("Джарвис запущен. Ожидание wake word...")
        logging.info("=" * 60)

        while self.is_running:
            try:
                # 1. Ожидание слова «Джарвис»
                await self.voice.wait_for_wake_word()

                # 2. Запись команды после wake word, завершение через VAD
                audio_frames = await self.voice.record_phrase()

                if not audio_frames:
                    logging.warning("Аудиофрагмент пустой. Возврат к ожиданию.")
                    continue

                # 3–6. Полный pipeline
                await self.process_command(audio_frames)

            except asyncio.CancelledError:
                raise

            except KeyboardInterrupt:
                self.stop()

            except Exception:
                logging.exception(
                    "Ошибка в обработке команды. Возврат к ожиданию wake word."
                )

                try:
                    await self.voice.play_tts(
                        "Произошла ошибка. Я снова готов слушать."
                    )
                except Exception:
                    logging.exception("Не удалось озвучить сообщение об ошибке.")

                await asyncio.sleep(0.5)

    def stop(self) -> None:
        """Запрашивает корректную остановку приложения."""
        self.is_running = False


async def main() -> None:
    orchestrator = JarvisOrchestrator()

    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logging.info("Остановка по Ctrl+C.")
    finally:
        orchestrator.stop()
        logging.info("Джарвис завершил работу.")


if __name__ == "__main__":
    asyncio.run(main())