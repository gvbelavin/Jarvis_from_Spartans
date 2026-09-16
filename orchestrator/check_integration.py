#!/usr/bin/env python3
"""
check_integration.py — офлайн-проверка связки оркестратора.

Запускается без микрофона, без аудиомоделей и без локальной LLM:

    python check_integration.py

Что проверяется (именно это ломается при интеграции):

1. Один и тот же вопрос от РАЗНЫХ пользователей даёт РАЗНЫЕ ответы
   (профиль и расписание берутся из памяти, а не из воздуха).
2. Гость (user_id=None) не получает личные данные.
3. Пустой transcript приводит к фразе «я не расслышал» и не доходит до LLM.
4. история: record_answer действительно попадает в следующий
   ctx.to_messages() как пары user/assistant (многоходовой диалог).
5. LLM получает корректный chat-формат, а не строка.
6. `audio_adapter` ИМПОРТИРУЕТСЯ и приводит ответ модуля Б к SpeakerResult.

Проверка 6 добавлена после реальной поломки: адаптер обращался к
`from config import ...`, а файла `orchestrator/config.py` не существовало.
Режим `--mock` и проверки 1-5 этого не замечали, потому что не трогают
адаптер, — падал только запуск на железе. Теперь такой разрыв ловится здесь.

Модуль памяти берётся НАСТОЯЩИЙ (`jarvis_memory`), заглушка только
у аудио и у LLM. Поэтому проверяется именно тот код, что пойдёт на плату.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MEMORY_MODULE_DIR = BASE_DIR / "memory_module"
if MEMORY_MODULE_DIR.exists() and str(MEMORY_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_MODULE_DIR))

from app import JarvisOrchestrator  # noqa: E402
from contracts import SpeakerResult  # noqa: E402
from llm_adapter import LLMAdapter, validate_messages  # noqa: E402
from mocks_for_testing import AudioEngineMock, LLMEngineMock  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jarvis.check")

SCHEDULE_QUESTION = "Какое у меня сегодня расписание?"


def make_orchestrator(
    memory,
    user_id,
    user_name,
    transcript=SCHEDULE_QUESTION,
    confidence=0.9,
) -> JarvisOrchestrator:
    audio = AudioEngineMock(
        transcript=transcript,
        speaker=SpeakerResult(
            user_id=user_id,
            user_name=user_name,
            confidence=confidence,
        ),
    )
    return JarvisOrchestrator(
        audio=audio,
        llm=LLMAdapter(LLMEngineMock()),
        memory=memory,
        warn_on_unknown_profile=False,
    )


async def main() -> int:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        status = "OK  " if condition else "FAIL"
        print(f"[{status}] {message}")
        if not condition:
            failures.append(message)

    import jarvis_memory as memory

    memory.clear()

    # --- 1. Разные пользователи, одинаковый вопрос ----------------------
    anton = make_orchestrator(memory, "anton", "Антон")
    masha = make_orchestrator(memory, "masha", "Маша")

    await anton.process_once()
    await masha.process_once()

    anton_answer = anton.audio.spoken[-1]
    masha_answer = masha.audio.spoken[-1]

    print(f"       anton -> {anton_answer}")
    print(f"       masha -> {masha_answer}")

    check(
        anton_answer != masha_answer,
        "разные пользователи получили разные ответы на один вопрос",
    )
    check("Антон" in anton_answer, "ответ Антона построен на его профиле")
    check("Маша" in masha_answer, "ответ Маши построен на её профиле")
    check(
        "лекция" in anton_answer and "английский" in masha_answer,
        "в ответах разные расписания из памяти",
    )

    # --- 2. Гость --------------------------------------------------------
    guest = make_orchestrator(memory, None, "Гость", confidence=0.1)
    await guest.process_once()
    guest_answer = guest.audio.spoken[-1]

    print(f"       guest -> {guest_answer}")
    check(
        "не знаю, кто спрашивает" in guest_answer,
        "гость (user_id=None) не получил личных данных",
    )
    check(
        "лекция" not in guest_answer and "английский" not in guest_answer,
        "гость не увидел чужое расписание",
    )

    # --- 3. Пустой transcript не доходит до LLM --------------------------
    silent = make_orchestrator(memory, "anton", "Антон", transcript="   ")
    await silent.process_once()
    silent_answer = silent.audio.spoken[-1]

    print(f"       silence -> {silent_answer}")
    check(
        "не расслышал" in silent_answer,
        "пустой transcript обработан без вызова LLM",
    )

    # --- 4. История попадает в следующий запрос --------------------------
    memory.clear()
    turns = make_orchestrator(memory, "anton", "Антон")

    await turns.process_once()

    ctx = memory.build_context("anton", "а тренировка?")
    history = ctx.history

    check(
        len(history) == 2,
        f"record_answer сохранил обмен (в истории {len(history)} сообщений)",
    )
    check(
        history[0]["role"] == "user" and history[1]["role"] == "assistant",
        "история имеет формат user/assistant",
    )

    messages = ctx.to_messages()
    check(messages[0]["role"] == "system", "первое сообщение — system")
    check(
        messages[-1] == {"role": "user", "content": "а тренировка?"},
        "последнее сообщение — текущий вопрос",
    )
    check(len(messages) == 4, "system + 2 сообщения истории + вопрос")

    # --- 5. Формат сообщений валиден для LLM -----------------------------
    try:
        validate_messages(messages)
        ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"       validate_messages: {exc}")
        ok = False
    check(ok, "ctx.to_messages() проходит валидацию chat-формата")

    try:
        validate_messages("не список")
        rejected = False
    except TypeError:
        rejected = True
    check(rejected, "строка вместо списка сообщений отвергается адаптером")

    # --- 6. Адаптер модуля Б импортируется и приводит ответы ---------------
    # Без этой проверки поломка вида `ModuleNotFoundError` в audio_adapter.py
    # не видна: и --mock, и проверки выше его не трогают.
    try:
        import audio_adapter
        from settings import AUDIO_MODULE_DIR, MEMORY_MODULE_DIR

        imported = True
    except Exception as exc:  # noqa: BLE001
        print(f"       import audio_adapter: {type(exc).__name__}: {exc}")
        imported = False

    check(imported, "audio_adapter импортируется (реальный режим запустится)")

    if imported:
        check(
            AUDIO_MODULE_DIR.is_dir(),
            f"папка модуля Б существует: {AUDIO_MODULE_DIR.name}",
        )
        check(
            (MEMORY_MODULE_DIR / "jarvis_memory").is_dir(),
            "пакет jarvis_memory на месте",
        )

        adapter = audio_adapter.AudioAdapter()

        # Неузнанный голос: модуль Б отдаёт user_name=None.
        # Раньше это проходило как user_name=None и ломало контракт.
        unknown = adapter._to_speaker_result(
            {"user_id": None, "user_name": None, "confidence": 0.31}
        )
        check(
            unknown.user_id is None and unknown.user_name == "Гость",
            "неузнанный голос -> гость с непустым user_name",
        )

        # Уверенный голос проходит как есть.
        known = adapter._to_speaker_result(
            {"user_id": "anton", "user_name": "anton", "confidence": 0.88}
        )
        check(
            known.user_id == "anton" and known.confidence == 0.88,
            "узнанный голос -> его user_id и confidence",
        )

        # Низкая уверенность ниже порога модуля Б -> гость.
        weak = adapter._to_speaker_result(
            {"user_id": "anton", "user_name": "anton", "confidence": 0.2}
        )
        check(
            weak.user_id is None,
            "уверенность ниже SPEAKER_THRESHOLD модуля Б -> гость",
        )

        # Переопределение ID через --user-map.
        mapped = audio_adapter.AudioAdapter(
            user_id_map={"shelestov": "anton"}
        )._to_speaker_result(
            {"user_id": "shelestov", "user_name": None, "confidence": 0.9}
        )
        check(
            mapped.user_id == "anton",
            "--user-map связывает ID модуля Б с профилем памяти",
        )

        # Порог берётся из config.py модуля Б, а не из копии в оркестраторе.
        check(
            audio_adapter._speaker_threshold() == 0.6,
            "SPEAKER_THRESHOLD читается из модуля Б (0.6)",
        )

    print()
    if failures:
        print(f"ПРОВАЛЕНО проверок: {len(failures)}")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("Все проверки интеграции пройдены.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))