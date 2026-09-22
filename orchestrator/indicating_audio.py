"""
indicating_audio.py — обёртка над аудиомодулем, которая подсвечивает этапы
на LED-индикаторе (модуль 5, Степан).

Проблема, которую решает: `AudioAdapter.listen_once()` внутри себя склеивает
wake word + запись + STT + Speaker ID в один вызов. Извне (из app.py) не
получится вставить `set_state("listening")` СРАЗУ после wake word и до
начала записи — этот момент проходит внутри listen_once, скрыто от
оркестратора.

Решение: обёртка декомпозирует listen_once на его составляющие
(они уже открыты как публичные методы в AudioAdapter) и вставляет
переходы состояний между ними. Реальному аудиомодулю знать про
индикатор ничего не надо.

Все остальные методы (`play_tts`, `close` и т.п.) просто делегируются.
`play_tts` дополнительно перед началом выставляет "speaking".
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from contracts import GUEST_NAME, SpeakerResult
from timing import astage

logger = logging.getLogger(__name__)


async def _timed(name: str, coro):
    """Обёртка для тайминга задач, которые уходят в asyncio.gather().

    Хочется отдельно видеть, сколько заняли identification и stt, но они
    идут параллельно — просто обернуть блок stage()-ом нельзя. Поэтому
    таймим каждую корутину сама по себе, и лог остаётся ровно того же
    формата, что у последовательных стадий.
    """
    async with astage(name):
        return await coro


class IndicatingAudioAdapter:
    """
    Обёртка вокруг любого объекта, реализующего контракт AudioModuleInterface
    (AudioAdapter или AudioEngineMock). Шлёт состояния на индикатор в тех
    точках пайплайна, где оркестратор их сам увидеть не может.
    """

    def __init__(self, inner: Any, indicator: Any) -> None:
        self._inner = inner
        self._indicator = indicator

    # --- декомпозированный listen_once ---------------------------------

    async def listen_once(self) -> tuple[SpeakerResult, str]:
        # ждём wake word — плата в это время в "idle"
        async with astage("wake_word"):
            await self._inner.wait_for_wake_word()
        await self._indicator.set_state("listening")

        async with astage("record"):
            audio_data = await self._inner.record_phrase()

        if audio_data is None or getattr(audio_data, "size", 1) == 0:
            logger.warning("No speech recorded after wake word.")
            await self._indicator.set_state("idle")
            return (
                SpeakerResult(
                    user_id=None,
                    user_name=GUEST_NAME,
                    confidence=0.0,
                ),
                "",
            )

        await self._indicator.set_state("thinking")

        # identification и stt идут параллельно (asyncio.gather), поэтому
        # обёрнуты индивидуально через _timed — иначе в логе будет одна
        # цифра на два этапа и по ней не понять, кто из них тормозил.
        speaker_task = asyncio.create_task(
            _timed("identification", self._inner.identify_speaker(audio_data))
        )
        stt_task = asyncio.create_task(
            _timed("stt", self._inner.speech_to_text(audio_data))
        )

        try:
            speaker_result, transcript = await asyncio.gather(
                speaker_task,
                stt_task,
            )
        except BaseException:
            speaker_task.cancel()
            stt_task.cancel()
            await self._indicator.set_state("idle")
            raise

        logger.info(
            "speaker=%s confidence=%.2f transcript=%r",
            speaker_result.user_name,
            speaker_result.confidence,
            transcript,
        )

        return speaker_result, transcript

    # --- проброс остальных методов -------------------------------------

    async def wait_for_wake_word(self) -> None:
        await self._inner.wait_for_wake_word()

    async def record_phrase(self):
        return await self._inner.record_phrase()

    async def identify_speaker(self, audio_data) -> SpeakerResult:
        return await self._inner.identify_speaker(audio_data)

    async def speech_to_text(self, audio_data) -> str:
        return await self._inner.speech_to_text(audio_data)

    async def play_tts(self, text: str) -> None:
        await self._indicator.set_state("speaking")
        try:
            async with astage("tts"):
                await self._inner.play_tts(text)
        finally:
            # После проговаривания вернём индикатор в idle, чтобы
            # цикл-фраза-цикл-фраза выглядел естественно. Ошибка
            # индикатора не должна ломать TTS.
            try:
                await self._indicator.set_state("idle")
            except Exception:
                logger.debug("Indicator idle after TTS failed.", exc_info=True)

    # --- необязательные хуки (есть у WebAudioAdapter) -------------------

    async def open(self) -> None:
        await self._call_optional("open")

    async def end_turn(self) -> None:
        await self._call_optional("end_turn")

    async def aclose(self) -> None:
        await self._call_optional("aclose")

    async def _call_optional(self, name: str) -> None:
        method = getattr(self._inner, name, None)
        if callable(method):
            await method()

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Inner audio close failed.", exc_info=True)
