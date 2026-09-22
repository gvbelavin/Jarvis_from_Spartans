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
    python app.py --web --web-host 100.107.17.63 \\
        --web-cert ~/.jarvis-tls/fullchain.pem --web-key ~/.jarvis-tls/key.pem
                                 # говорить через браузер телефона (web_audio.py)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from contracts import SpeakerResult
from debug_audio import maybe_wrap as maybe_debug_audio
from indicator import IndicatorEsp32, IndicatorNoop
from indicating_audio import IndicatingAudioAdapter
from settings import DEFAULT_WEB_PORT, INDICATOR_PORT
from timing import astage

# Корень папки orchestrator
BASE_DIR = Path(__file__).resolve().parent

# Позволяет импортировать пакет jarvis_memory,
# находящийся в orchestrator/memory_module/jarvis_memory
MEMORY_MODULE_DIR = BASE_DIR / "memory_module"
if MEMORY_MODULE_DIR.exists() and str(MEMORY_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_MODULE_DIR))

# Числа словами. Модуль Ц уже убирает цифры из промпта, но модель может
# написать их и сама; это последний рубеж перед динамиком. Импорт отдельно
# от self.memory: там может стоять мок, а функция нужна всегда.
try:
    from jarvis_memory import spell as spell_numbers
except ImportError:                                # модуль Ц не подключён
    def spell_numbers(text: str) -> str:
        return text

logger = logging.getLogger("jarvis")

# Фразы самого ассистента (не модели): произносятся, когда до LLM дело не дошло.
NO_SPEECH_PHRASE = "Я не расслышал команду. Повторите, пожалуйста."
ERROR_PHRASE = "Произошла ошибка. Я снова готов слушать."
NO_ANSWER_PHRASE = "Извините, я не смог подготовить ответ."
STARTUP_PHRASE = "Доброе утро! Джарвис проснулся."
SHUTDOWN_PHRASE = "Спокойной ночи! Джарвис спать."
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
        indicator: Optional[Any] = None,
        warn_on_unknown_profile: bool = True,
    ) -> None:
        self.audio = audio
        self.llm = llm
        self.memory = memory
        self.indicator = indicator if indicator is not None else IndicatorNoop()
        self.warn_on_unknown_profile = warn_on_unknown_profile
        self.is_running = True

        # Кнопка BOOT на плате индикатора переключает микрофон.
        # Пока mute=True — оркестратор не вызывает listen_once (аудиомодуль
        # не открывает микрофон, wake word не запускается).
        self.mute = False
        self.indicator.set_mute_callback(self._on_mute_change)

    async def _on_mute_change(self, muted: bool) -> None:
        """Callback от IndicatorEsp32: пользователь нажал кнопку BOOT."""
        self.mute = muted
        logger.info(
            "Микрофон %s (кнопка на плате индикатора).",
            "ВЫКЛЮЧЕН" if muted else "включён",
        )

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

        # Гостя подсвечиваем отдельным паттерном — человек видит, что
        # его не узнали. Иначе оставляем/ставим «думаю»: генерация на
        # RKLLM занимает ~3 с, и без индикатора пауза выглядит как зависон.
        if user_id is None:
            await self.indicator.set_state("guest")
        else:
            await self.indicator.set_state("thinking")

        # 3. Память и RAG: профиль, расписание, факты, история — всё
        #    именно этого пользователя. user_id=None (гость) поддержан
        #    модулем памяти отдельной веткой.
        async with astage("memory"):
            ctx = await asyncio.to_thread(
                self.memory.build_context, user_id, transcript
            )

        self._check_profile(user_id, ctx)

        # 4. LLM получает готовый список chat-сообщений:
        #    system + история + текущий вопрос. Ничего не склеиваем сами.
        messages = ctx.to_messages()
        logger.info("Думаю... (запрос к LLM, %d сообщений)", len(messages))
        logger.debug(messages)
        async with astage("llm"):
            reply_text = await self.llm.generate(messages)

        reply_text = (reply_text or "").strip()

        if not reply_text:
            logger.warning("LLM вернула пустой ответ.")
            reply_text = NO_ANSWER_PHRASE

        # «14:30» синтезатор прочитает как набор символов, а не как время.
        # Делается до записи в историю, чтобы «повтори» вернуло ровно то,
        # что было произнесено.
        reply_text = spell_numbers(reply_text)

        # 5. История: сохраняем пару «вопрос — ответ» для этого пользователя.
        #    Без этого вызова не работают ни «повтори», ни многоходовой диалог.
        async with astage("memory_write"):
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
        свои ключи (`daniel`, `daniil`, `fedor`, `stepan`, `gleb`).
        Сейчас они совпадают, так что ключ нужен редко. Если наборы строк
        разойдутся,
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

        await self.indicator.open()
        await self.indicator.set_state("idle")
        # Веб-режим поднимает здесь HTTP(S)-сервер; у AudioAdapter хука нет.
        await self._audio_hook("open")

        logger.info("=" * 64)
        logger.info("Джарвис запущен. Ожидание wake word...")
        logger.info("=" * 64)
        
        await self._say_safely(STARTUP_PHRASE)

        attempts = 0

        try:
            while self.is_running:
                if max_commands is not None and attempts >= max_commands:
                    logger.info("Попыток обработки: %d. Завершаю.", attempts)
                    break

                # Микрофон замьючен по кнопке — не тратим ресурсы на
                # запуск wake word / записи. Считаем это НЕ попыткой,
                # чтобы --once не заканчивался «пустым» кругом при mute.
                if self.mute:
                    await asyncio.sleep(0.3)
                    continue

                # Считаем попытки, а не только успехи: иначе сбойная команда
                # в режиме --once зацикливала бы оркестратор.
                attempts += 1

                try:
                    await self.process_once()

                    # Ответ произнесён (или пропущен) — возвращаемся в idle.
                    # IndicatingAudioAdapter уже мог поставить idle сам после
                    # TTS, но лишний вызов безопасен и покрывает ветку с
                    # пустым transcript, когда TTS играет NO_SPEECH_PHRASE.
                    await self.indicator.set_state("idle")

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
                    await self.indicator.set_state("error")
                    await self._say_safely(ERROR_PHRASE)
                    await asyncio.sleep(1.5)
                    await self.indicator.set_state("idle")

                finally:
                    # Веб-режим: всё, что было сказано за ход (включая
                    # ERROR_PHRASE), уходит телефону одним ответом.
                    await self._audio_hook("end_turn")
        finally:
            await self._say_safely(SHUTDOWN_PHRASE)
            try:
                await self._audio_hook("aclose")
            except Exception:
                logger.debug("Ошибка при остановке веб-сервера.", exc_info=True)
            try:
                await self.indicator.close()
            except Exception:
                logger.debug("Ошибка при закрытии индикатора.", exc_info=True)
            self.close()

    async def _audio_hook(self, name: str) -> None:
        """Вызывает необязательный хук аудиомодуля (есть у WebAudioAdapter)."""
        method = getattr(self.audio, name, None)
        if callable(method):
            await method()

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
    """Разбирает `--user-map daniel_new=daniel,fedya=fedor`."""
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


def _maybe_web(engine: Any, args: argparse.Namespace, sample_rate: int) -> Any:
    """
    В режиме --web микрофон и динамик платы заменяются браузером телефона.

    Модели модуля Б при этом те же: engine (AudioAdapter или заглушка)
    остаётся внутри и делает Speaker ID, STT и синтез Piper.
    """
    if not args.web:
        return engine

    from web_audio import WebAudioAdapter

    return WebAudioAdapter(
        engine,
        sample_rate=sample_rate,
        host=args.web_host,
        port=args.web_port,
        token=os.environ.get("JARVIS_WEB_TOKEN"),
        cert_file=args.web_cert,
        key_file=args.web_key,
    )


def _build_indicator(args: argparse.Namespace) -> Any:
    """
    Создаёт индикатор (ESP32 на USB-serial) или заглушку.

    Флаг --no-indicator полностью отключает интеграцию с платой —
    полезно для запуска на машине, где ESP32 не подключён вовсе.
    """
    if getattr(args, "no_indicator", False):
        return IndicatorNoop()
    return IndicatorEsp32(port=args.indicator_port)


def build_orchestrator(args: argparse.Namespace) -> JarvisOrchestrator:
    """Собирает реальный оркестратор: модуль Б + модуль Ц + модуль Г + индикатор."""

    from audio_adapter import AudioAdapter, _submodule, check_ready
    from llm_adapter import LLMAdapter, load_llm_engine

    import jarvis_memory as memory
    from jarvis_memory import db as memory_db

    if not memory_db.is_seeded():
        logger.warning(
            "База памяти пуста (%s). Профили команды не подтянутся. "
            "Наполнить: cd orchestrator/memory_module && python -m jarvis_memory.seed",
            memory_db.db_path(),
        )

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
    sample_rate = _submodule("config").SAMPLE_RATE
    audio = _maybe_web(audio, args, sample_rate)
    # --debug: каждая фраза сохраняется в WAV (debug_audio.py).
    audio = maybe_debug_audio(
        audio, args, source="web" if args.web else "mic", sample_rate=sample_rate
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

    # --- Модуль 5: индикатор ---------------------------------------------
    # Обёртка нужна, чтобы вставить set_state в те точки внутри
    # listen_once, которые скрыты от оркестратора.
    indicator = _build_indicator(args)
    audio = IndicatingAudioAdapter(audio, indicator)

    return JarvisOrchestrator(
        audio=audio,
        llm=llm,
        memory=memory,
        indicator=indicator,
    )


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
        from jarvis_memory import db as memory_db
        from jarvis_memory import seed as memory_seed
    except ImportError:
        logger.warning(
            "Пакет jarvis_memory недоступен — использую заглушку памяти "
            "(orchestrator/memory_module не инициализирован?)."
        )
        memory = MemoryModuleMock()
        warn_on_unknown_profile = False
    else:
        warn_on_unknown_profile = True
        # --mock без сида даёт нейтральный контекст и FOREIGN KEY при
        # записи истории: профили живут в SQLite, а не в коде.
        if not memory_db.is_seeded():
            logger.info(
                "База памяти пуста — наполняю профилями команды (jarvis_memory.seed)."
            )
            memory_seed.seed()

    indicator = _build_indicator(args)
    engine = AudioEngineMock(
        transcript=args.mock_transcript,
        max_commands=args.max_commands,
    )
    audio = maybe_debug_audio(
        _maybe_web(engine, args, sample_rate=16000),
        args,
        source="web" if args.web else "mock",
        sample_rate=16000,
    )
    audio = IndicatingAudioAdapter(audio, indicator)

    return JarvisOrchestrator(
        audio=audio,
        # Заглушка LLM синхронная — оборачиваем её тем же адаптером,
        # что и реальный модуль Г: так проверяется и путь to_thread.
        llm=LLMAdapter(LLMEngineMock()),
        memory=memory,
        indicator=indicator,
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
            "daniel_new=daniel. Нужно только если имя папки эталонов "
            "не совпадает с user_id в базе"
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
        "--indicator-port",
        default=INDICATOR_PORT,
        help=(
            f"serial-порт LED-индикатора (модуль 5); по умолчанию "
            f"{INDICATOR_PORT}. Если порт недоступен, оркестратор "
            f"стартует без индикации."
        ),
    )
    parser.add_argument(
        "--no-indicator",
        action="store_true",
        help="полностью отключить интеграцию с платой индикатора",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help=(
            "слушать не микрофон платы, а браузер телефона (web_audio.py); "
            "токен доступа — из переменной окружения JARVIS_WEB_TOKEN"
        ),
    )
    parser.add_argument(
        "--web-host",
        default="127.0.0.1",
        help=(
            "адрес веб-интерфейса; на плате — IP в NetBird, чтобы страница "
            "не была видна в локальной сети (по умолчанию 127.0.0.1)"
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=DEFAULT_WEB_PORT,
        help=f"порт веб-интерфейса (по умолчанию {DEFAULT_WEB_PORT})",
    )
    parser.add_argument(
        "--web-cert",
        default=None,
        help="fullchain.pem для HTTPS (без HTTPS iPhone не даст микрофон)",
    )
    parser.add_argument(
        "--web-key",
        default=None,
        help="приватный ключ к --web-cert",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "режим отладки: лог DEBUG и сохранение каждой фразы в WAV "
            "(debug_audio.py); то же включает --log-level DEBUG или JARVIS_DEBUG=1"
        ),
    )
    parser.add_argument(
        "--debug-audio-dir",
        default=None,
        help="куда сохранять записи в режиме отладки (по умолчанию orchestrator/debug_audio/)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="уровень логирования (по умолчанию INFO)",
    )
    return parser.parse_args(argv)


async def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if args.debug:
        args.log_level = "DEBUG"

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.getLogger("numba").setLevel(logging.WARNING)

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
