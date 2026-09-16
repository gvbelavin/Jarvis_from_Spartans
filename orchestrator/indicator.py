"""
indicator.py — Модуль индикации + кнопки для оркестратора Джарвиса (задача 5, Степан).

Управляет светодиодным индикатором на ESP32 через USB-serial и слушает
события от кнопки BOOT на плате (mute toggle).

Интерфейс (async, чтобы не блокировать asyncio-цикл app.py):

    ind = IndicatorEsp32(port="/dev/ttyUSB0", on_mute_change=cb)
    await ind.open()
    await ind.set_state("listening")
    ...
    await ind.close()

Где cb — асинхронная функция `async def cb(muted: bool) -> None`.
Она вызывается когда пользователь физически нажал кнопку BOOT на плате.

Особенности:
- RX платы (host -> board) работает надёжно на 115200
- TX платы (board -> host) сначала плюётся мусором от ROM-загрузчика,
  но потом даёт чистые строки READY / OK / EVT
- Кнопка обрабатывается локально на плате (mute toggle) + шлёт
  событие `EVT btn_short mute=on|off` в оркестратор
- Если порт не открыт или отвалился — set_state молча логирует и не падает
  (индикация — не критичная функция, пайплайн должен продолжать работать)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

import serial

logger = logging.getLogger("indicator")

VALID_STATES = frozenset({
    "boot", "idle", "listening", "thinking", "speaking", "guest", "error",
    "mute", "unmute",
})

MuteCallback = Callable[[bool], Awaitable[None]]


class IndicatorEsp32:
    """Отправляет команды состояния индикатора на ESP32 по serial
    и слушает EVT-сообщения от кнопки."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 115200,
        reset_on_open: bool = False,
        on_mute_change: Optional[MuteCallback] = None,
    ) -> None:
        self._port = port
        self._baud = baud
        self._reset_on_open = reset_on_open
        self._ser: Optional[serial.Serial] = None
        self._lock = asyncio.Lock()
        self._on_mute_change = on_mute_change
        self._reader_task: Optional[asyncio.Task] = None
        self._stop_reader = False

    async def open(self) -> None:
        """Открыть порт и запустить reader-таск. Идемпотентно."""
        if self._ser and self._ser.is_open:
            return
        try:
            await asyncio.to_thread(self._open_blocking)
            logger.info("Indicator: opened %s @ %d", self._port, self._baud)
        except Exception as e:
            logger.warning("Indicator: cannot open %s: %s (продолжаю без индикации)",
                           self._port, e)
            self._ser = None
            return

        self._stop_reader = False
        loop = asyncio.get_running_loop()
        self._reader_task = loop.create_task(self._reader_loop())

    def _open_blocking(self) -> None:
        s = serial.Serial()
        s.port = self._port
        s.baudrate = self._baud
        s.timeout = 0.2
        s.dtr = False
        s.rts = False
        s.open()
        if not self._reset_on_open:
            s.setDTR(False)
            s.setRTS(False)
        self._ser = s

    async def close(self) -> None:
        self._stop_reader = True
        if self._reader_task:
            try:
                await asyncio.wait_for(self._reader_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._reader_task.cancel()
            self._reader_task = None

        if not self._ser:
            return
        try:
            await self._write_line("idle")
        except Exception:
            pass
        try:
            await asyncio.to_thread(self._ser.close)
        finally:
            self._ser = None
            logger.info("Indicator: closed")

    def set_mute_callback(self, callback: Optional[MuteCallback]) -> None:
        """Назначить/сменить callback, вызываемый при нажатии кнопки BOOT.

        Позволяет создать индикатор в билдере, а затем прицепить к нему
        обработчик из уже готового оркестратора — без магии через setattr.
        """
        self._on_mute_change = callback

    async def set_state(self, state: str) -> None:
        """
        Установить состояние индикатора.
        Допустимые: boot, idle, listening, thinking, speaking, guest, error,
                    mute, unmute
        Не бросает исключений — просто логирует ошибку.
        """
        state = state.strip().lower()
        if state not in VALID_STATES:
            logger.warning("Indicator: unknown state %r, ignoring", state)
            return
        await self._write_line(state)

    async def _write_line(self, line: str) -> None:
        if not self._ser or not self._ser.is_open:
            logger.debug("Indicator: no port, skip send %r", line)
            return
        payload = (line + "\n").encode()
        async with self._lock:
            try:
                await asyncio.to_thread(self._ser.write, payload)
                await asyncio.to_thread(self._ser.flush)
                logger.debug("Indicator: -> %s", line)
            except Exception as e:
                logger.warning("Indicator: write failed (%s), dropping port", e)
                try:
                    await asyncio.to_thread(self._ser.close)
                except Exception:
                    pass
                self._ser = None

    async def _reader_loop(self) -> None:
        """Читает строки от платы и обрабатывает EVT-события."""
        buf = b""
        while not self._stop_reader:
            if not self._ser or not self._ser.is_open:
                await asyncio.sleep(0.5)
                continue
            try:
                chunk = await asyncio.to_thread(self._ser.read, 128)
            except Exception as e:
                logger.warning("Indicator: read failed (%s)", e)
                await asyncio.sleep(0.5)
                continue

            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, _, buf = buf.partition(b"\n")
                try:
                    text = line.decode(errors="replace").strip()
                except Exception:
                    continue
                if not text:
                    continue
                await self._handle_line(text)

    async def _handle_line(self, text: str) -> None:
        logger.debug("Indicator: <- %s", text)
        if not text.startswith("EVT"):
            return
        # Формат: "EVT btn_short mute=on" / "EVT btn_short mute=off" / "EVT btn_long"
        parts = text.split()
        if len(parts) >= 3 and parts[1] == "btn_short":
            key, _, val = parts[2].partition("=")
            if key == "mute" and self._on_mute_change:
                muted = val.lower() == "on"
                try:
                    await self._on_mute_change(muted)
                except Exception:
                    logger.exception("Indicator: on_mute_change callback failed")


class IndicatorNoop:
    """Заглушка на случай, когда плата не подключена — использовать в CI/тестах."""

    async def open(self) -> None: pass
    async def close(self) -> None: pass
    async def set_state(self, state: str) -> None:
        logger.debug("IndicatorNoop: %s", state)

    def set_mute_callback(self, callback) -> None:
        # Кнопки нет — callback никогда не сработает, но контракт совпадает.
        return None
