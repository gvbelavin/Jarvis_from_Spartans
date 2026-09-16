"""
llm_adapter.py — тонкий адаптер между оркестратором и модулем LLM (модуль Г).

ЧТО ДЕЛАЕТ
----------

Приводит любой локальный движок генерации к единственному контракту
`contracts.LLMModuleInterface`:

    await llm.generate(messages) -> str

где `messages` — уже готовый список chat-сообщений, собранный
`jarvis_memory` (`ctx.to_messages()`): system + история + текущий вопрос.

Ключевая деталь из требований: локальная LLM (Qwen2.5 через rkllm-llama.cpp,
llama.cpp, ONNX Runtime и т. д.) почти всегда имеет СИНХРОННЫЙ API.
Вызов такой модели инлайном заблокировал бы event loop оркестратора —
на время генерации перестали бы обслуживаться задачи и не обрабатывался
бы Ctrl+C. Поэтому синхронный движок вызывается через `asyncio.to_thread`.

ПОДКЛЮЧЕНИЕ РЕАЛЬНОГО МОДУЛЯ Г
------------------------------

Когда участник Г пришлёт реализацию, ожидается один из вариантов:

1. Модуль `llm_module` с классом `LLMEngine` и методом `generate(messages)`:

       from llm_module import LLMEngine
       llm = LLMAdapter(LLMEngine())

2. Готовая функция:

       llm = LLMAdapter.from_callable(my_generate, is_async=False)

3. Готовый объект, уже реализующий `async generate(messages)`:
   он передаётся в `LLMAdapter` как есть и не оборачивается.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
from typing import Any, AsyncIterator, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# Имя модуля, который должен предоставить участник Г.
DEFAULT_LLM_MODULE = "llm_module"
DEFAULT_LLM_CLASS = "LLMEngine"

_VALID_ROLES = {"system", "user", "assistant"}


def validate_messages(messages: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Проверяет, что LLM получает корректный chat-формат.

    Проверка дешёвая, а ловит она самую частую ошибку интеграции:
    в модель уезжает не список сообщений, а строка или объект контекста.
    """
    if not isinstance(messages, (list, tuple)):
        raise TypeError(
            "LLM.generate() ожидает list[dict[str, str]], получено: "
            f"{type(messages).__name__}. "
            "Передавайте ctx.to_messages(), а не сам ctx."
        )

    normalized: list[dict[str, str]] = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(
                f"messages[{index}] должен быть dict, получено: "
                f"{type(message).__name__}"
            )

        role = message.get("role")
        content = message.get("content")

        if role not in _VALID_ROLES:
            raise ValueError(
                f"messages[{index}]['role'] = {role!r}; "
                f"допустимо: {sorted(_VALID_ROLES)}"
            )

        if not isinstance(content, str):
            raise TypeError(
                f"messages[{index}]['content'] должен быть str, получено: "
                f"{type(content).__name__}"
            )

        normalized.append({"role": role, "content": content})

    if not normalized:
        raise ValueError("Список messages пуст: LLM нечего генерировать.")

    if normalized[-1]["role"] != "user":
        # Это не ошибка формата, но обычно признак того, что контекст
        # собран неправильно.
        logger.warning(
            "Последнее сообщение имеет роль %r, а не 'user'.",
            normalized[-1]["role"],
        )

    return normalized


class CallableLLMEngine:
    """Обёртка над функцией генерации: `fn(messages) -> str`."""

    def __init__(self, fn: Callable[..., Any], is_async: bool = False) -> None:
        self._fn = fn
        self.is_async = is_async

    def generate(self, messages: list[dict[str, str]]) -> Any:
        return self._fn(messages)


class LLMAdapter:
    """
    Нормализует произвольный локальный движок к контракту:

        await llm.generate(messages) -> str

    Синхронные движки выполняются в отдельном потоке и не блокируют
    event loop оркестратора.
    """

    def __init__(self, engine: Any) -> None:
        generate = getattr(engine, "generate", None)

        if generate is None:
            raise TypeError(
                "Движок LLM должен иметь метод generate(messages). "
                f"Получено: {type(engine).__name__}"
            )

        self._engine = engine
        self._engine_is_async = inspect.iscoroutinefunction(generate)

        if self._engine_is_async:
            logger.info(
                "LLM: %s.generate() асинхронный — вызывается напрямую.",
                type(engine).__name__,
            )
        else:
            logger.info(
                "LLM: %s.generate() синхронный — вызывается через asyncio.to_thread().",
                type(engine).__name__,
            )

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., Any],
        is_async: bool = False,
    ) -> "LLMAdapter":
        """Создаёт адаптер из обычной функции генерации."""
        return cls(CallableLLMEngine(fn, is_async=is_async))

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Единственный метод контракта. Возвращает текст ответа."""
        normalized = validate_messages(messages)

        if self._engine_is_async:
            raw = await self._engine.generate(normalized)
        else:
            raw = await asyncio.to_thread(self._engine.generate, normalized)

        if not isinstance(raw, str):
            raise TypeError(
                "LLM должна вернуть str, получено: "
                f"{type(raw).__name__}. "
                "Проверьте модуль Г."
            )

        return raw.strip()

    async def stream(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Необязательный потоковый режим.

        Реальная локальная LLM умеет отдавать токены по мере генерации —
        это заметно сокращает время до первого звука. Если движок такого
        не умеет, метод отдаёт готовый ответ одним куском. Контракт
        оркестратора от этого не меняется: `generate()` остаётся основным.
        """
        stream_method = getattr(self._engine, "stream", None)
        normalized = validate_messages(messages)

        if stream_method is None:
            yield await self.generate(normalized)
            return

        if inspect.isasyncgenfunction(stream_method):
            async for chunk in stream_method(normalized):
                yield chunk
            return

        if inspect.iscoroutinefunction(stream_method):
            result = await stream_method(normalized)
            if hasattr(result, "__aiter__"):
                async for chunk in result:
                    yield chunk
            elif isinstance(result, str):
                yield result
            else:
                for chunk in result:
                    yield chunk
            return

        # Синхронный генератор: собираем чанки в потоке, отдаём по мере выхода.
        iterator = await asyncio.to_thread(stream_method, normalized)
        for chunk in iterator:
            yield chunk


def load_llm_engine(
    module_name: str = DEFAULT_LLM_MODULE,
    class_name: str = DEFAULT_LLM_CLASS,
) -> LLMAdapter:
    """
    Загружает реальный модуль Г по соглашению об именах.

    Используется, когда модуль Г уже лежит в `orchestrator/llm_module/`
    (как обычная папка или как ещё один git submodule).

    :raises ImportError: если модуль или класс ещё не поставлены.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Модуль LLM {module_name!r} не найден. "
            "Передайте готовый движок явно: "
            "LLMAdapter(engine) или LLMAdapter.from_callable(fn)."
        ) from exc

    engine_cls = getattr(module, class_name, None)

    if engine_cls is None:
        raise ImportError(
            f"В модуле {module_name!r} нет класса {class_name!r}. "
            "Ожидается класс с методом generate(messages)."
        )

    return LLMAdapter(engine_cls())