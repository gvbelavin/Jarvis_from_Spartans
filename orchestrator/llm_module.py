"""
llm_module.py — модуль Г со стороны оркестратора: клиент RKLLM-сервера.

ЗАЧЕМ ЭТОТ ФАЙЛ
---------------

`llm_adapter.load_llm_engine()` по умолчанию ищет модуль `llm_module`
с классом `LLMEngine`. Участник Г (Фёдор) поставил модуль НЕ как
Python-пакет, а как HTTP-сервер: пропатченный `flask_server.py` из
`rknn-llm` крутится на плате в `~/rkllm_server/` и слушает порт 8080
с OpenAI-совместимым эндпоинтом `/v1/chat/completions`.

Этот файл — недостающее звено между ними. Он не загружает модель и не
трогает NPU: вся генерация происходит в чужом процессе, здесь только
HTTP-запрос и разбор ответа.

Контракт ровно тот, которого ждёт `LLMAdapter`:

    LLMEngine().generate(messages: list[dict[str, str]]) -> str

Метод синхронный — `LLMAdapter` сам вызовет его через `asyncio.to_thread`,
чтобы генерация (~3 с на 0.5B) не блокировала event loop оркестратора.

ПОЧЕМУ urllib, А НЕ requests
----------------------------

`requests` тянет за собой urllib3, charset-normalizer и certifi. На плате
с 20 ГБ диска и общим окружением лишняя зависимость не нужна: запрос здесь
один, он локальный (`localhost`, без TLS), и стандартной библиотеки хватает.
Модуль Ц (`jarvis_memory`) по той же причине ходит в погоду через urllib.

НАСТРОЙКА
---------

Значения по умолчанию соответствуют инструкции Фёдора и менять их не нужно.
Переопределяются переменными окружения:

    JARVIS_LLM_URL      полный URL эндпоинта (default: локальный порт 8080)
    JARVIS_LLM_MODEL    имя модели в теле запроса (default: rkllm)
    JARVIS_LLM_TIMEOUT  таймаут одного запроса в секундах (default: 120)

ПРОВЕРКА, ЧТО СЕРВЕР ЖИВ
------------------------

    curl http://localhost:8080/v1/models
    python llm_module.py "Привет, кто ты?"
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:8080/v1/chat/completions"
DEFAULT_MODEL = "rkllm"

# Генерация на 0.5B занимает ~3 с, на 1.5B — заметно дольше, а первый
# запрос после старта сервера ещё и ждёт загрузки модели на NPU.
# Таймаут здесь — защита от зависшего сервера, а не инструмент управления
# задержкой: слишком маленький превратит медленный ответ в ошибку.
DEFAULT_TIMEOUT = 120.0


class LLMServerError(RuntimeError):
    """Сервер LLM недоступен или вернул ответ, который не разобрать."""


class LLMEngine:
    """Клиент RKLLM-сервера, реализующий контракт модуля Г."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.url = url or os.environ.get("JARVIS_LLM_URL", DEFAULT_URL)
        self.model = model or os.environ.get("JARVIS_LLM_MODEL", DEFAULT_MODEL)
        self.timeout = timeout or float(
            os.environ.get("JARVIS_LLM_TIMEOUT", DEFAULT_TIMEOUT)
        )

        logger.info("LLM: RKLLM-сервер %s (модель %r)", self.url, self.model)

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Отправляет chat-сообщения на сервер и возвращает текст ответа.

        `messages` приходит из `jarvis_memory` как `ctx.to_messages()`:
        system-промпт с профилем пользователя, история диалога и текущий
        вопрос. Ничего склеивать и переупаковывать не нужно — формат
        сервера ровно такой же (OpenAI chat completions).
        """
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                # Потоковый режим сервер умеет, но оркестратор всё равно
                # ждёт целую фразу: Piper синтезирует ответ целиком.
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise LLMServerError(
                f"RKLLM-сервер вернул HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            # Самый частый случай на плате: сервер просто не запущен.
            # Сообщение должно сразу называть команду, которая это чинит,
            # иначе по логу оркестратора причина неочевидна.
            raise LLMServerError(
                f"RKLLM-сервер недоступен по {self.url} ({exc.reason}). "
                "Проверьте: curl http://localhost:8080/v1/models ; "
                "поднять: cd ~/rkllm_server && nohup ~/miniconda3/bin/python3 "
                "flask_server.py --rkllm_model_path ~/Qwen2.5-0.5B-Instruct_"
                "W8A8_RK3588.rkllm --target_platform rk3588 "
                "> ~/rkllm_server.log 2>&1 & disown"
            ) from exc
        except TimeoutError as exc:
            raise LLMServerError(
                f"RKLLM-сервер не ответил за {self.timeout:.0f} с."
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMServerError(
                "RKLLM-сервер вернул не JSON — вероятно, это HTML-страница "
                "ошибки Flask. Смотрите ~/rkllm_server.log."
            ) from exc

        return self._extract_text(body)

    @staticmethod
    def _extract_text(body: object) -> str:
        """Достаёт текст из ответа OpenAI-совместимого сервера.

        Разбор намеренно защитный: `flask_server.py` из rknn-llm пропатчен
        вручную, и структура ответа может разойтись с оригиналом. Лучше
        внятное сообщение с куском ответа, чем KeyError из потока.
        """
        if not isinstance(body, dict):
            raise LLMServerError(f"Ожидался JSON-объект, получено: {type(body).__name__}")

        choices = body.get("choices")

        if not isinstance(choices, list) or not choices:
            # Сервер сообщает об ошибке генерации в поле error.
            error = body.get("error")
            if error:
                raise LLMServerError(f"RKLLM-сервер сообщил об ошибке: {error}")
            raise LLMServerError(f"В ответе нет choices: {str(body)[:300]}")

        first = choices[0]

        if not isinstance(first, dict):
            raise LLMServerError(f"choices[0] не объект: {str(first)[:300]}")

        message = first.get("message")

        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]

        # Некоторые сборки сервера отдают текст в choices[0].text
        # (формат legacy completions) — принимаем и его.
        if isinstance(first.get("text"), str):
            return first["text"]

        raise LLMServerError(
            f"Не нашёл текст ответа в choices[0]: {str(first)[:300]}"
        )


def main() -> int:
    """Ручная проверка: `python llm_module.py "вопрос"`."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    question = sys.argv[1] if len(sys.argv) > 1 else "Привет! Кратко: кто ты?"

    engine = LLMEngine()

    try:
        answer = engine.generate([{"role": "user", "content": question}])
    except LLMServerError as exc:
        print(f"ОШИБКА: {exc}")
        return 1

    print(f"Вопрос: {question}")
    print(f"Ответ:  {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
