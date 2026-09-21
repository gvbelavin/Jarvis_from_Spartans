#!/usr/bin/env python3
"""
check_integration.py — офлайн-проверка связки оркестратора.

Запускается без микрофона, без аудиомоделей и без живого RKLLM-сервера:

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
7. `llm_module.LLMEngine` шлёт ctx.to_messages() на OpenAI-совместимый
   эндпоинт и забирает choices[0].message.content. Сервер — заглушка
   на localhost, живой NPU не нужен.
8. На время генерации индикатор в состоянии thinking (гость — guest).

Проверка 6 добавлена после реальной поломки: адаптер обращался к
`from config import ...`, а файла `orchestrator/config.py` не существовало.
Режим `--mock` и проверки 1-5 этого не замечали, потому что не трогают
адаптер, — падал только запуск на железе. Теперь такой разрыв ловится здесь.

Модуль памяти берётся НАСТОЯЩИЙ (`jarvis_memory`), заглушка только
у аудио и у LLM. Поэтому проверяется именно тот код, что пойдёт на плату.

База — временный файл: на плате `JARVIS_DB` указывает на боевую sqlite,
и этот скрипт её не должен ни сидировать, ни чистить.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent

MEMORY_MODULE_DIR = BASE_DIR / "memory_module"
if MEMORY_MODULE_DIR.exists() and str(MEMORY_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_MODULE_DIR))

# До любого import jarvis_memory: иначе проверки сотрут боевую базу на плате.
_TMP_DB = Path(tempfile.mkdtemp(prefix="jarvis-orch-")) / "test.db"
os.environ["JARVIS_DB"] = str(_TMP_DB)

from app import JarvisOrchestrator  # noqa: E402
from contracts import SpeakerResult  # noqa: E402
from llm_adapter import LLMAdapter, load_llm_engine, validate_messages  # noqa: E402
from llm_module import LLMEngine, LLMServerError  # noqa: E402
from mocks_for_testing import AudioEngineMock, LLMEngineMock  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jarvis.check")

SCHEDULE_QUESTION = "Какое у меня сегодня расписание?"


class RecordingIndicator:
    """Пишет вызовы set_state, чтобы проверить UX «думаю» без ESP32."""

    def __init__(self) -> None:
        self.states: list[str] = []

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def set_state(self, state: str) -> None:
        self.states.append(state)

    def set_mute_callback(self, callback: Any) -> None:
        return None


class _FakeRKLLM(BaseHTTPRequestHandler):
    """OpenAI-совместитая заглушка /v1/chat/completions."""

    last_body: dict | None = None

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).last_body = body
        payload = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Короткий ответ модели.",
                        }
                    }
                ]
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return None


def make_orchestrator(
    memory,
    user_id,
    user_name,
    transcript=SCHEDULE_QUESTION,
    confidence=0.9,
    indicator: Optional[Any] = None,
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
        indicator=indicator,
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
    from jarvis_memory import seed as memory_seed

    memory_seed.seed()
    memory.clear()

    # --- 1. Разные пользователи, одинаковый вопрос ----------------------
    anton_ind = RecordingIndicator()
    masha_ind = RecordingIndicator()
    anton = make_orchestrator(memory, "anton", "Антон", indicator=anton_ind)
    masha = make_orchestrator(memory, "masha", "Маша", indicator=masha_ind)

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
    check(
        "thinking" in anton_ind.states,
        "перед LLM индикатор в состоянии thinking",
    )

    # --- 2. Гость --------------------------------------------------------
    guest_ind = RecordingIndicator()
    guest = make_orchestrator(
        memory, None, "Гость", confidence=0.1, indicator=guest_ind
    )
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
    check("guest" in guest_ind.states, "неузнанный голос подсвечен как guest")

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
    check(
        "Антон" in messages[0]["content"],
        "system-промпт из памяти содержит профиль говорящего",
    )

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

    # --- 7. Клиент RKLLM: ctx.to_messages() уходит на :8080 как есть ------
    loaded = load_llm_engine()
    check(
        type(loaded._engine).__name__ == "LLMEngine",
        "load_llm_engine() находит llm_module.LLMEngine",
    )

    _FakeRKLLM.last_body = None
    httpd = HTTPServer(("127.0.0.1", 0), _FakeRKLLM)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions"
        engine = LLMEngine(url=url, timeout=5)
        payload_messages = memory.build_context(
            "anton", SCHEDULE_QUESTION
        ).to_messages()
        answer = engine.generate(payload_messages)
        body = _FakeRKLLM.last_body or {}
        sent = body.get("messages") or []
        roles = [item.get("role") for item in sent]
        system_text = sent[0]["content"] if sent else ""

        check(answer == "Короткий ответ модели.", "клиент забирает choices[0].message.content")
        check(roles[:1] == ["system"] and roles[-1:] == ["user"], "на сервер уходят system и user")
        check("Антон" in system_text, "до LLM доезжает system-промпт с профилем")
        check(body.get("model") == "rkllm", "в запросе model=rkllm")
        check(body.get("stream") is False, "stream=False: Piper ждёт целую фразу")
    finally:
        httpd.shutdown()
        httpd.server_close()

    downed = LLMEngine(url="http://127.0.0.1:1/v1/chat/completions", timeout=1)
    try:
        downed.generate([{"role": "user", "content": "ping"}])
        unreachable = False
        hint = ""
    except LLMServerError as exc:
        unreachable = True
        hint = str(exc)
    check(unreachable, "выключенный сервер даёт LLMServerError, а не сырой URLError")
    check(
        "flask_server.py" in hint and "8080" in hint,
        "ошибка недоступности называет команду запуска сервера",
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
