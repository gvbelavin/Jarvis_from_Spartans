"""
timing.py — единообразные INFO-логи «сколько заняла каждая стадия».

Формат:

    INFO  identification start
    INFO  identification 0.87s

Один контекст-менеджер для sync-кода (`with stage("memory"): ...`) и
async-вариант (`async with astage("llm"): ...`). Логгер один (`jarvis.perf`),
чтобы в проде можно было гранулярно выкрутить/приглушить только тайминги.

Использование:

    from timing import stage, astage, log_llm_perf

    async with astage("llm"):
        text = await self._engine.generate(messages)

    log_llm_perf(prompt_tokens=145, completion_tokens=42, elapsed=3.2)
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Optional

logger = logging.getLogger("jarvis.perf")


@contextlib.contextmanager
def stage(name: str):
    """Синхронный тайминг стадии: пишет «name start» и «name Ns»."""
    logger.info("%s start", name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s %.2fs", name, time.perf_counter() - t0)


@contextlib.asynccontextmanager
async def astage(name: str):
    """Асинхронный вариант stage() для использования в async-коде."""
    logger.info("%s start", name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s %.2fs", name, time.perf_counter() - t0)


def log_llm_perf(
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    total_tokens: Optional[int],
    elapsed: float,
) -> None:
    """Логирует производительность LLM: токены на входе/выходе и tok/s.

    Считаем tok/s только по completion_tokens: prompt-часть на большинстве
    серверов обрабатывается заметно быстрее, и включать её в знаменатель
    исказило бы фактическую скорость генерации.
    """
    if (
        isinstance(completion_tokens, (int, float))
        and completion_tokens > 0
        and elapsed > 0
    ):
        tps = completion_tokens / elapsed
        logger.info(
            "llm perf: prompt=%s completion=%s total=%s | %.1f tok/s | %.2fs",
            prompt_tokens,
            completion_tokens,
            total_tokens,
            tps,
            elapsed,
        )
    else:
        logger.info(
            "llm perf: prompt=%s completion=%s total=%s | %.2fs (нет данных для tok/s)",
            prompt_tokens,
            completion_tokens,
            total_tokens,
            elapsed,
        )
