#!/usr/bin/env python3
"""
debug_audio.py — дамп всех записанных фраз в папку для отладки STT.

Работает только в режиме отладки (`python app.py --debug ...`,
`--log-level DEBUG` или переменная окружения JARVIS_DEBUG=1).

Обёртка встаёт между источником звука (AudioAdapter — микрофон платы,
WebAudioAdapter — браузер телефона) и IndicatingAudioAdapter. Сохраняется
ровно тот массив, который уходит в Whisper (int16, 16 кГц, моно) — то есть
в WAV слышно именно то, что «услышала» модель.

На каждую фразу в папке появляются:

    20260923-142501-123_web.wav    — сама запись
    index.jsonl                    — строка с метаданными: источник,
                                     длительность, пик/RMS/клиппинг,
                                     распознанный текст, говорящий

WAV пишется ДО распознавания: если STT упадёт, запись всё равно останется.
Чтобы не забить диск платы, хранятся только последние DEFAULT_KEEP записей.

Просмотр того, что накопилось (на плате):

    python debug_audio.py                 # последние 20 записей
    python debug_audio.py -n 100          # последние 100
    python debug_audio.py --dir /tmp/x    # другая папка

Забрать на ноутбук и послушать:

    rsync -av <плата>:<путь>/orchestrator/debug_audio/ ./debug_audio/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np

from settings import BASE_DIR

logger = logging.getLogger(__name__)

DEFAULT_DEBUG_AUDIO_DIR = BASE_DIR / "debug_audio"
DEFAULT_KEEP = 500
INDEX_NAME = "index.jsonl"


def debug_enabled(args: argparse.Namespace) -> bool:
    """Режим отладки: --debug, --log-level DEBUG или JARVIS_DEBUG=1."""
    if getattr(args, "debug", False):
        return True
    if str(getattr(args, "log_level", "")).upper() == "DEBUG":
        return True
    return os.environ.get("JARVIS_DEBUG", "").lower() in ("1", "true", "yes")


def maybe_wrap(inner: Any, args: argparse.Namespace, source: str, sample_rate: int) -> Any:
    """Оборачивает аудиомодуль в DebugAudioRecorder, только если включена отладка."""
    if not debug_enabled(args):
        return inner

    directory = Path(getattr(args, "debug_audio_dir", None) or DEFAULT_DEBUG_AUDIO_DIR)
    logger.info("[debug] Записи фраз сохраняются в %s", directory)
    return DebugAudioRecorder(inner, directory, source=source, sample_rate=sample_rate)


class DebugAudioRecorder:
    """
    Прозрачная обёртка над AudioModuleInterface: всё делегирует inner,
    по пути сохраняя запись и результаты STT / Speaker ID.
    """

    def __init__(
        self,
        inner: Any,
        directory: Path,
        source: str,
        sample_rate: int,
        keep: int = DEFAULT_KEEP,
    ) -> None:
        self._inner = inner
        self._dir = Path(directory)
        self._source = source
        self._sample_rate = sample_rate
        self._keep = keep
        # id(audio) -> запись index.jsonl, пока не пришли и STT, и Speaker ID.
        self._pending: dict[int, dict[str, Any]] = {}

    # --- перехватываемые методы -----------------------------------------

    async def record_phrase(self):
        audio = await self._inner.record_phrase()
        try:
            self._save(audio)
        except Exception:
            logger.warning("[debug] Не удалось сохранить запись.", exc_info=True)
        return audio

    async def speech_to_text(self, audio_data) -> str:
        try:
            text = await self._inner.speech_to_text(audio_data)
        except BaseException as exc:
            self._update(audio_data, transcript=None, stt_error=repr(exc))
            raise
        self._update(audio_data, transcript=text)
        return text

    async def identify_speaker(self, audio_data):
        try:
            result = await self._inner.identify_speaker(audio_data)
        except BaseException as exc:
            self._update(audio_data, speaker=None, speaker_error=repr(exc))
            raise
        self._update(
            audio_data,
            speaker={
                "user_id": result.user_id,
                "user_name": result.user_name,
                "confidence": round(float(result.confidence), 3),
            },
        )
        return result

    # --- всё остальное (play_tts, open, end_turn, aclose, close, ...) ----

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    # --- внутреннее ------------------------------------------------------

    def _save(self, audio: Any) -> None:
        # В --mock приходит не ndarray, а заглушка — сохранять нечего.
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return

        self._dir.mkdir(parents=True, exist_ok=True)

        now = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        name = f"{stamp}-{int(now * 1000) % 1000:03d}_{self._source}.wav"

        pcm = _to_int16(audio)
        with wave.open(str(self._dir / name), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self._sample_rate)
            wav.writeframes(pcm.tobytes())

        peak = int(np.abs(pcm.astype(np.int32)).max())
        rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
        clipped = float(np.mean(np.abs(pcm.astype(np.int32)) >= 32767))

        self._pending[id(audio)] = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "file": name,
            "source": self._source,
            "seconds": round(pcm.size / self._sample_rate, 2),
            "sample_rate": self._sample_rate,
            "input_dtype": str(audio.dtype),
            "peak": peak,
            "rms": round(rms, 1),
            "clipped_pct": round(clipped * 100, 2),
        }
        logger.debug(
            "[debug] %s: %.2f с, peak=%d, rms=%.0f, clipped=%.2f%%",
            name, pcm.size / self._sample_rate, peak, rms, clipped * 100,
        )
        self._prune()

    def _update(self, audio: Any, **fields: Any) -> None:
        entry = self._pending.get(id(audio))
        if entry is None:
            return
        entry.update(fields)

        has_stt = "transcript" in entry
        has_speaker = "speaker" in entry
        if not (has_stt and has_speaker):
            return

        del self._pending[id(audio)]
        try:
            with open(self._dir / INDEX_NAME, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("[debug] Не удалось записать %s.", INDEX_NAME, exc_info=True)

    def _prune(self) -> None:
        files = sorted(self._dir.glob("*.wav"))
        for old in files[: max(0, len(files) - self._keep)]:
            try:
                old.unlink()
            except OSError:
                pass


def _to_int16(audio: np.ndarray) -> np.ndarray:
    """Приводит запись к int16 моно (так её видит stt.transcribe)."""
    if audio.ndim > 1:
        audio = audio[:, 0]
    if np.issubdtype(audio.dtype, np.floating):
        return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return audio.astype(np.int16)


# ----------------------------------------------------------------------
# CLI: посмотреть, что накопилось
# ----------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(description="Последние отладочные записи фраз.")
    parser.add_argument("--dir", default=str(DEFAULT_DEBUG_AUDIO_DIR))
    parser.add_argument("-n", type=int, default=20, help="сколько записей показать")
    args = parser.parse_args()

    index = Path(args.dir) / INDEX_NAME
    if not index.is_file():
        print(f"Нет {index}. Запустите app.py с --debug и скажите что-нибудь.")
        return

    lines = index.read_text(encoding="utf-8").splitlines()[-args.n:]
    for line in lines:
        e = json.loads(line)
        speaker = e.get("speaker") or {}
        exists = (Path(args.dir) / e["file"]).is_file()
        print(
            f"{e['file']}{'' if exists else ' (удалён)'}\n"
            f"    {e['seconds']:>5} с  peak={e['peak']:<5} rms={e['rms']:<7} "
            f"clip={e['clipped_pct']}%  "
            f"speaker={speaker.get('user_name', '?')}({speaker.get('confidence', '?')})\n"
            f"    «{e.get('transcript') or e.get('stt_error', '')}»"
        )


if __name__ == "__main__":
    _main()
