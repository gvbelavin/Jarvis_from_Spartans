#!/usr/bin/env python3
"""
app.py — оркестратор «Джарвиса» (модуль А).

Последовательность обработки одной команды:

    wake word -> VAD/запись -> Speaker ID + STT   (audio_adapter, модуль Б)
              -> memory.build_context(user_id, transcript)   (jarvis_memory, модуль Ц)
              -> await llm.generate(ctx.to_messages())       (модуль Г)
              -> memory.record_answer(user_id, transcript, answer)
              -> Piper TTS -> динамик

Ни один файл submodule не редактируется: доступ к модулю Б идёт через
`audio_adapter.py`, к модулю Ц — через публичный API пакета `jarvis_memory`,
к модулю Г — через `llm_adapter.py`.

Запуск:

    python app.py                # реальное железо (микрофон, модели модуля Б)
    python app.py --mock         # офлайн-проверка связки без микрофона и моделей
    python app.py --once         # обработать одну команду и выйти
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

from contracts import SpeakerResult

# Корень папки orchestrator
BASE_DIR = Path(__file__).resolve().parent

# Позволяет импортировать пакет jarvis_memory,
# находящийся в orchestrator/memory_module/jarvis_memory
MEMORY_MODULE_DIR = BASE_DIR / "memory_module"
if MEMORY_MODULE_DIR.exists() and str(MEMORY_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_MODULE_DIR))

logger = logging.getLogger("jarvis")

# Фразы самого ассистента (не модели): произносятся, когда до LLM дело не дошло.
NO_SPEECH_PHRASE = "Я не расслышал команду. Повторите, пожалуйста."
ERROR_PHRASE = "Произошла ошибка. Я снова готов слушать."
NO_ANSWER_PHRASE = "Извините, я не смог подготовить ответ."
LLM_NOT_CONNECTED_PHRASE = (
    "Модуль языковой модели ещё не подключён, поэтому ответить я не могу."
)


class LLMNotConnected:
    """
    Заглушка на время отсутствия модуля Г.

    Реализует тот же контракт, что и настоящая LLM: `await generate(messages) -> str`.
    Существует только для того, чтобы оркестратор можно было запустить и
    проверить связку аудио + память до появления реализации участника Г.
    """

    async def generate(self, messages: list[dict[str, str]]) -> str:
        logger.warning(
            "Модуль LLM не подключён: получено %d сообщений, генерация недоступна.",
            len(messages),
        )
        return LLM_NOT_CONNECTED_PHRASE


class JarvisOrchestrator:
    """Главный цикл голосового ассистента."""

    def __init__(
        self,
        audio: Any,
        llm: Any,
        memory: Any,
        warn_on_unknown_profile: bool = True,
    ) -> None:
        self.audio = audio
        self.llm = llm
        self.memory = memory
        self.warn_on_unknown_profile = warn_on_unknown_profile
        self.is_running = True

    # ------------------------------------------------------------------
    # Одна команда
    # ------------------------------------------------------------------

    async def process_once(self) -> None:
        started_at = time.perf_counter()

        # 1. Аудиоэтап целиком: wake word -> запись (VAD) -> Speaker ID + STT.
        speaker_result, transcript = await self.audio.listen_once()

        if not isinstance(speaker_result, SpeakerResult):
            raise TypeError(
                "Аудиомодуль обязан вернуть SpeakerResult из contracts.py, "
                f"получено: {type(speaker_result).__name__}"
            )

        # 2. STT не вернул текст — на этом всё, отвечаем сами и ждём снова.
        if not transcript.strip():
            logger.info("Пустой transcript: команда не распознана.")
            await self.audio.play_tts(NO_SPEECH_PHRASE)
            return

        user_id: Optional[str] = speaker_result.user_id

        # 3. Память и RAG: профиль, расписание, факты, история — всё
        #    именно этого пользователя. user_id=None (гость) поддержан
        #    модулем памяти отдельной веткой.
        ctx = await asyncio.to_thread(
            self.memory.build_context, user_id, transcript
        )

        self._check_profile(user_id, ctx)

        # 4. LLM получает готовый список chat-сообщений:
        #    system + история + текущий вопрос. Ничего не склеиваем сами.
        messages = ctx.to_messages()
        reply_text = await self.llm.generate(messages)

        reply_text = (reply_text or "").strip()

        if not reply_text:
            logger.warning("LLM вернула пустой ответ.")
            reply_text = NO_ANSWER_PHRASE

        # 5. История: сохраняем пару «вопрос — ответ» для этого пользователя.
        #    Без этого вызова не работают ни «повтори», ни многоходовой диалог.
        await asyncio.to_thread(
            self.memory.record_answer, user_id, transcript, reply_text
        )

        latency = time.perf_counter() - started_at
        logger.info(
            "Ответ за %.2f с | user_id=%r | intent=%r | answer=%r",
            latency,
            user_id,
            getattr(ctx, "intent", None),
            reply_text,
        )

        # 6. TTS и динамик.
        await self.audio.play_tts(reply_text)

    def _check_profile(self, user_id: Optional[str], ctx: Any) -> None:
        """Предупреждает, если голос узнан, а профиля в памяти нет.

        Модуль Б берёт user_id из имени папки с эталонными записями
        (`audio_dataset/references/<user_id>/`), а модуль памяти знает
        свои ключи (`anton`, `masha`). Если это разные наборы строк,
        персональные данные молча не подтянутся. Лучше сказать об этом вслух.
        """
        if (
            not self.warn_on_unknown_profile
            or user_id is None
            or not isinstance(getattr(ctx, "debug", None), dict)
        ):
            return

        if not ctx.debug.get("profile_found", True):
            logger.warning(
                "Голос опознан как %r, но профиля с таким user_id в модуле "
                "памяти нет — контекст будет нейтральным. Свяжите ID: "
                "--user-map <имя_папки_модуля_Б>=<user_id_модуля_памяти>.",
                user_id,
            )

    # ------------------------------------------------------------------
    # Главный цикл
    # ------------------------------------------------------------------

    async def run(self, max_commands: Optional[int] = None) -> None:
        """Бесконечный цикл, либо ограниченное число команд при max_commands."""

        logger.info("=" * 64)
        logger.info("Джарвис запущен. Ожидание wake word...")
        logger.info("=" * 64)

        attempts = 0

        try:
            while self.is_running:
                if max_commands is not None and attempts >= max_commands:
                    logger.info("Попыток обработки: %d. Завершаю.", attempts)
                    break

                # Считаем попытки, а не только успехи: иначе сбойная команда
                # в режиме --once зацикливала бы оркестратор.
                attempts += 1

                try:
                    await self.process_once()

                except asyncio.CancelledError:
                    raise

                except KeyboardInterrupt:
                    self.stop()
                    break

                except Exception:
                    logger.exception(
                        "Ошибка при обработке команды. "
                        "Возврат к ожиданию wake word."
                    )
                    await self._say_safely(ERROR_PHRASE)
                    await asyncio.sleep(0.5)
        finally:
            self.close()

    async def _say_safely(self, phrase: str) -> None:
        """Озвучивает фразу, не давая сбою TTS уронить цикл."""
        try:
            await self.audio.play_tts(phrase)
        except Exception:
            logger.exception("Не удалось озвучить сообщение.")

    def close(self) -> None:
        """Закрывает аудиоресурсы."""
        close = getattr(self.audio, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Ошибка при закрытии аудиомодуля.", exc_info=True)

    def stop(self) -> None:
        """Запрашивает корректную остановку приложения."""
        self.is_running = False


# ----------------------------------------------------------------------
# Сборка зависимостей
# ----------------------------------------------------------------------


def parse_user_map(raw: Optional[str]) -> dict[str, str]:
    """Разбирает `--user-map shelestov=anton,puchkov=masha`."""
    if not raw:
        return {}

    mapping: dict[str, str] = {}

    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"Некорректная пара в --user-map: {pair!r}. "
                "Ожидается формат имя_папки=user_id."
            )
        source, target = pair.split("=", 1)
        mapping[source.strip()] = target.strip()

    return mapping


def build_orchestrator(args: argparse.Namespace) -> JarvisOrchestrator:
    """Собирает реальный оркестратор: модуль Б + модуль Ц + модуль Г."""

    from audio_adapter import AudioAdapter, check_ready
    from llm_adapter import LLMAdapter, load_llm_engine

    import jarvis_memory as memory

    # --- Модуль Б: аудио -------------------------------------------------
    # Проверяем готовность ДО первого обращения к железу: иначе сырой
    # ModuleNotFoundError прилетит из отдельного потока, и по логу будет
    # непонятно, чего именно не хватает.
    problems = check_ready()

    if problems:
        logger.error("Аудиомодуль (модуль Б) не готов к запуску:")
        for problem in problems:
            logger.error("  - %s", problem)
        logger.error(
            "Подробности по установке: orchestrator/audio_module/README.md. "
            "Проверить связку без железа: python app.py --mock --once"
        )
        raise SystemExit(2)

    audio = AudioAdapter(
        device=args.device,
        record_timeout=args.record_timeout,
        user_id_map=parse_user_map(args.user_map),
    )

    # --- Модуль Г: локальная LLM -----------------------------------------
    if args.llm_module:
        llm = load_llm_engine(args.llm_module, args.llm_class)
    else:
        try:
            llm = load_llm_engine()
        except ImportError as exc:
            logger.warning("%s", exc)
            logger.warning(
                "Оркестратор стартует без LLM: связка аудио + память "
                "проверяется, ответы будет заглушка."
            )
            llm = LLMAdapter(LLMNotConnected())

    return JarvisOrchestrator(audio=audio, llm=llm, memory=memory)


def build_mock_orchestrator(args: argparse.Namespace) -> JarvisOrchestrator:
    """
    Собирает оркестратор на заглушках: без микрофона, моделей и LLM.

    Модуль памяти при этом берётся НАСТОЯЩИЙ (`jarvis_memory`), если он
    доступен: у него нет внешних зависимостей, и так проверяется реальная
    связка «профиль -> контекст -> история», включая сценарий user_id=None.
    Если подмодуль памяти не подтянут, используется изолированная заглушка.
    """
    from llm_adapter import LLMAdapter
    from mocks_for_testing import AudioEngineMock, LLMEngineMock, MemoryModuleMock

    try:
        import jarvis_memory as memory
    except ImportError:
        logger.warning(
            "Пакет jarvis_memory недоступен — использую заглушку памяти "
            "(orchestrator/memory_module не инициализирован?)."
        )
        memory = MemoryModuleMock()
        warn_on_unknown_profile = False
    else:
        warn_on_unknown_profile = True

    return JarvisOrchestrator(
        audio=AudioEngineMock(
            transcript=args.mock_transcript,
            max_commands=args.max_commands,
        ),
        # Заглушка LLM синхронная — оборачиваем её тем же адаптером,
        # что и реальный модуль Г: так проверяется и путь to_thread.
        llm=LLMAdapter(LLMEngineMock()),
        memory=memory,
        warn_on_unknown_profile=warn_on_unknown_profile,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оркестратор локального голосового ассистента «Джарвис».",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="работать на заглушках (без микрофона, моделей и LLM)",
    )
    parser.add_argument(
        "--mock-transcript",
        default="Какое у меня сегодня расписание?",
        help="распознанный текст для режима --mock",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="обработать одну команду и завершиться",
    )
    parser.add_argument(
        "--max-commands",
        type=int,
        default=None,
        help="ограничить число обработанных команд и выйти",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="номер аудиоустройства; по умолчанию config.DEVICE_NUMBER модуля Б",
    )
    parser.add_argument(
        "--record-timeout",
        type=float,
        default=10.0,
        help="максимум секунд на запись команды после wake word",
    )
    parser.add_argument(
        "--user-map",
        default=None,
        help=(
            "соответствие ID говорящего и профиля памяти, например "
            "shelestov=anton,puchkov=masha"
        ),
    )
    parser.add_argument(
        "--llm-module",
        default=None,
        help="имя модуля Г (по умолчанию llm_module)",
    )
    parser.add_argument(
        "--llm-class",
        default="LLMEngine",
        help="имя класса LLM в модуле Г (по умолчанию LLMEngine)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="уровень логирования (по умолчанию INFO)",
    )
    return parser.parse_args(argv)


async def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.mock:
        orchestrator = build_mock_orchestrator(args)
        logger.info("Режим заглушек: реальные модули не загружаются.")
    else:
        orchestrator = build_orchestrator(args)

    max_commands = 1 if args.once else args.max_commands

    try:
        await orchestrator.run(max_commands=max_commands)
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C.")
    finally:
        orchestrator.stop()

    logger.info("Джарвис завершил работу.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass