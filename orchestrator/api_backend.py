"""
api_backend.py — временная подмена модуля Г и STT модуля Б на облачное API.

ЗАЧЕМ ЭТОТ ФАЙЛ
---------------

Нужно уметь ответить на вопрос «это железо тормозит или это наши модели
плохие?». Для этого весь пайплайн остаётся прежним — wake word, запись,
Speaker ID (WeSpeaker), Piper TTS, память, индикатор, — а две самые
подозрительные ступени подменяются эталоном:

    локальный Whisper (ggml-base) -> /v1/audio/transcriptions (whisper-1)
    локальный RKLLM (Qwen2.5-0.5B) -> /v1/chat/completions   (gpt-4o-mini)

Включается одним флагом:

    python app.py --use_api

Файл целиком опциональный: без `--use_api` он даже не импортируется,
в app.py от него ровно три строки. Это инструмент для замера, а не
часть итогового продукта — на защите ассистент работает локально.

НАСТРОЙКА
---------

Сервер-релей говорит на OpenAI один в один, отличий два: base URL и ключ.

    RELAY_TOKEN              ключ (обязателен; ищется также в
                             JARVIS_API_KEY и OPENAI_API_KEY)
    JARVIS_API_BASE_URL      база API (default: https://api.gehrman.me/v1)
    JARVIS_API_LLM_MODEL     модель генерации (default: gpt-4o-mini)
    JARVIS_API_STT_MODEL     модель распознавания (default: whisper-1)
    JARVIS_API_STT_LANGUAGE  язык распознавания (default: ru — как в stt.py)
    JARVIS_API_TIMEOUT       таймаут одного запроса, секунды (default: 60)

ПОЧЕМУ urllib, А НЕ openai/requests
-----------------------------------

По той же причине, что и в llm_module.py: на плате общее окружение и
20 ГБ диска, а запросов здесь два и оба тривиальные. Ставить SDK ради
временной проверки смысла нет.

ПРОВЕРКА БЕЗ ОРКЕСТРАТОРА
-------------------------

    export RELAY_TOKEN=...
    python api_backend.py "Привет, кто ты?"      # LLM
    python api_backend.py --stt запись.wav       # распознавание
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
import wave
from typing import Any, Optional

from timing import log_llm_perf

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.gehrman.me/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_STT_MODEL = "whisper-1"
DEFAULT_LANGUAGE = "ru"
DEFAULT_TIMEOUT = 60.0

# Частота записи модуля Б (audio_module/config.py::SAMPLE_RATE).
# Дублировать её здесь нельзя, но и импортировать config ради одного
# числа рано: api_backend должен импортироваться и без sys.path модуля Б.
# Значение приходит снаружи, это лишь запасной вариант для CLI-проверки.
FALLBACK_SAMPLE_RATE = 16000


class ApiError(RuntimeError):
    """Релей недоступен или вернул ответ, который не разобрать."""


def api_key() -> str:
    """Ключ релея. Без него смысла стартовать нет — падаем сразу."""
    for name in ("RELAY_TOKEN", "JARVIS_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value.strip()

    raise ApiError(
        "Не задан ключ API. Режим --use_api требует токен релея: "
        "export RELAY_TOKEN=<токен> (принимаются также JARVIS_API_KEY "
        "и OPENAI_API_KEY)."
    )


def base_url() -> str:
    return os.environ.get("JARVIS_API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _timeout() -> float:
    return float(os.environ.get("JARVIS_API_TIMEOUT", DEFAULT_TIMEOUT))


def _request(
    path: str,
    data: bytes,
    content_type: str,
    timeout: float,
) -> dict:
    """POST на релей с Bearer-авторизацией. Возвращает разобранный JSON."""
    url = f"{base_url()}{path}"

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": content_type,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        # 401 здесь — почти всегда протухший или пустой RELAY_TOKEN.
        raise ApiError(f"{url} вернул HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ApiError(
            f"{url} недоступен ({exc.reason}). Проверьте сеть на плате: "
            f"curl {base_url()}/models -H \"Authorization: Bearer $RELAY_TOKEN\""
        ) from exc
    except TimeoutError as exc:
        raise ApiError(f"{url} не ответил за {timeout:.0f} с.") from exc
    except json.JSONDecodeError as exc:
        raise ApiError(f"{url} вернул не JSON.") from exc

    if not isinstance(body, dict):
        raise ApiError(f"Ожидался JSON-объект, получено: {type(body).__name__}")

    return body


# ----------------------------------------------------------------------
# LLM: /v1/chat/completions
# ----------------------------------------------------------------------


class ApiLLMEngine:
    """
    Модуль Г через облако. Контракт тот же, что у llm_module.LLMEngine:

        generate(messages: list[dict[str, str]]) -> str

    Метод синхронный — LLMAdapter сам уведёт его в asyncio.to_thread.
    Формат сообщений уже OpenAI-совместимый (ctx.to_messages()), так что
    переупаковывать нечего: ровно та же полезная нагрузка, что уходит
    на RKLLM-сервер, только с другим адресом и ключом.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.model = model or os.environ.get("JARVIS_API_LLM_MODEL", DEFAULT_LLM_MODEL)
        self.timeout = timeout or _timeout()

        logger.info(
            "LLM: облачный релей %s/chat/completions (модель %r)",
            base_url(),
            self.model,
        )

    def generate(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        t0 = time.perf_counter()
        body = _request(
            "/chat/completions",
            payload,
            "application/json",
            self.timeout,
        )
        elapsed = time.perf_counter() - t0

        # Тот же формат лога, что у локального движка: цифры двух режимов
        # можно класть рядом в отчёт без пересчёта.
        usage = body.get("usage")
        if isinstance(usage, dict):
            log_llm_perf(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                elapsed=elapsed,
            )
        else:
            logger.info("llm perf: usage missing | %.2fs", elapsed)

        return _extract_text(body)


def _extract_text(body: dict) -> str:
    choices = body.get("choices")

    if not isinstance(choices, list) or not choices:
        error = body.get("error")
        if error:
            raise ApiError(f"Релей сообщил об ошибке: {error}")
        raise ApiError(f"В ответе нет choices: {str(body)[:300]}")

    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None

    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]

    raise ApiError(f"Не нашёл текст ответа в choices[0]: {str(first)[:300]}")


# ----------------------------------------------------------------------
# STT: /v1/audio/transcriptions
# ----------------------------------------------------------------------


def _to_wav_bytes(audio: Any, sample_rate: int) -> bytes:
    """
    Запись модуля Б -> WAV в памяти.

    Рекордер отдаёт моно int16 (ровно это же видит stt.transcribe, где
    оно делится на 32768). Float на всякий случай тоже принимаем: через
    web_audio.py приходит массив из браузера, и однажды он может
    оказаться нормализованным.
    """
    import numpy as np

    if isinstance(audio, (bytes, bytearray)):
        raw = bytes(audio)
    else:
        array = np.asarray(audio)

        if array.ndim > 1:                       # стерео -> моно
            array = array[:, 0]

        if array.dtype != np.int16:
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(array, -1.0, 1.0) * 32767.0
            array = array.astype(np.int16)

        raw = array.tobytes()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(raw)

    return buffer.getvalue()


def _multipart(fields: dict[str, str], wav_bytes: bytes) -> tuple[bytes, str]:
    """Собирает multipart/form-data вручную: SDK ради одного запроса не ставим."""
    boundary = f"----jarvis{uuid.uuid4().hex}"
    parts: list[bytes] = []

    for name, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )

    parts.append(
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; '
            'filename="command.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(wav_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcribe(audio: Any, sample_rate: int = FALLBACK_SAMPLE_RATE) -> str:
    """Распознаёт запись облачным Whisper. Синхронная, как stt.transcribe."""
    wav_bytes = _to_wav_bytes(audio, sample_rate)

    model = os.environ.get("JARVIS_API_STT_MODEL", DEFAULT_STT_MODEL)
    language = os.environ.get("JARVIS_API_STT_LANGUAGE", DEFAULT_LANGUAGE)

    data, content_type = _multipart(
        {
            "model": model,
            "language": language,
            "response_format": "json",
        },
        wav_bytes,
    )

    t0 = time.perf_counter()
    body = _request("/audio/transcriptions", data, content_type, _timeout())
    elapsed = time.perf_counter() - t0

    text = body.get("text")

    if not isinstance(text, str):
        error = body.get("error")
        if error:
            raise ApiError(f"Релей сообщил об ошибке распознавания: {error}")
        raise ApiError(f"В ответе нет поля text: {str(body)[:300]}")

    seconds = (len(wav_bytes) - 44) / 2 / max(int(sample_rate), 1)
    logger.info(
        "stt perf (api, %s): %.1f с аудио за %.2f с",
        model,
        seconds,
        elapsed,
    )

    return text.strip()


class ApiSttAudio:
    """
    Обёртка над аудиодвижком модуля Б: подменяет ТОЛЬКО speech_to_text.

    Wake word, VAD, Speaker ID и Piper остаются локальными и работают через
    `__getattr__` — обёртка прозрачна и для web_audio.py (ему нужны
    `stream_tools()` и `synthesize()`), и для debug_audio.py.

    Побочный выигрыш: `audio_adapter._submodule("stt")` теперь не вызывается
    вовсе, так что ggml-base даже не грузится в память платы.
    """

    def __init__(self, inner: Any, sample_rate: int = FALLBACK_SAMPLE_RATE) -> None:
        self._inner = inner
        self._sample_rate = sample_rate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def speech_to_text(self, audio_data) -> str:
        import asyncio

        text = await asyncio.to_thread(transcribe, audio_data, self._sample_rate)
        return (text or "").strip()

    async def listen_once(self):
        # listen_once внутри AudioAdapter зовёт СВОЙ self.speech_to_text,
        # то есть локальный Whisper. Поэтому связку «запись -> Speaker ID +
        # STT» собираем здесь заново поверх публичных методов движка.
        import asyncio

        from contracts import GUEST_NAME, SpeakerResult

        await self._inner.wait_for_wake_word()
        audio_data = await self._inner.record_phrase()

        if audio_data is None or getattr(audio_data, "size", 1) == 0:
            logger.warning("Запись пуста — распознавать нечего.")
            return SpeakerResult(
                user_id=None,
                user_name=GUEST_NAME,
                confidence=0.0,
            ), ""

        speaker_result, transcript = await asyncio.gather(
            self._inner.identify_speaker(audio_data),
            self.speech_to_text(audio_data),
        )

        logger.info(
            "speaker=%s confidence=%.2f transcript=%r",
            speaker_result.user_name,
            speaker_result.confidence,
            transcript,
        )

        return speaker_result, transcript


# ----------------------------------------------------------------------
# Точки подключения для app.py
# ----------------------------------------------------------------------


def wrap_audio(engine: Any, sample_rate: int = FALLBACK_SAMPLE_RATE) -> ApiSttAudio:
    """Подменяет STT у аудиодвижка. Вызывается из app.py при --use_api."""
    logger.info(
        "STT: облачный релей %s/audio/transcriptions (модель %r), "
        "локальный Whisper не загружается",
        base_url(),
        os.environ.get("JARVIS_API_STT_MODEL", DEFAULT_STT_MODEL),
    )
    return ApiSttAudio(engine, sample_rate=sample_rate)


def build_llm() -> Any:
    """Готовый LLM для оркестратора: облачный движок в штатном LLMAdapter."""
    from llm_adapter import LLMAdapter

    return LLMAdapter(ApiLLMEngine())


def drop_local_stt_problems(problems: list[str]) -> list[str]:
    """
    Убирает из отчёта check_ready() претензии к локальному Whisper.

    В режиме --use_api ни pywhispercpp, ни ggml-base.bin не нужны:
    распознаёт облако. Всё остальное (WeSpeaker, openWakeWord, Piper)
    по-прежнему обязательно, и эти строки остаются как были.
    """
    kept: list[str] = []

    for problem in problems:
        if "pywhispercpp" in problem:
            continue

        if "нет моделей модуля Б:" in problem and "ggml-base.bin" in problem:
            prefix, _, assets = problem.partition(":")
            rest = [
                asset.strip()
                for asset in assets.split(",")
                if asset.strip() and "ggml-base.bin" not in asset
            ]
            if not rest:
                continue
            problem = f"{prefix}: " + ", ".join(rest)

        kept.append(problem)

    return kept


def main() -> int:
    """Ручная проверка: `python api_backend.py "вопрос"` / `--stt файл.wav`."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    argv = sys.argv[1:]

    try:
        if argv and argv[0] == "--stt":
            if len(argv) < 2:
                print("Использование: python api_backend.py --stt файл.wav")
                return 2

            import numpy as np

            with wave.open(argv[1], "rb") as wav:
                sample_rate = wav.getframerate()
                frames = wav.readframes(wav.getnframes())

            audio = np.frombuffer(frames, dtype=np.int16)
            print(f"Текст: {transcribe(audio, sample_rate)}")
            return 0

        question = argv[0] if argv else "Привет! Кратко: кто ты?"
        answer = ApiLLMEngine().generate([{"role": "user", "content": question}])
        print(f"Вопрос: {question}")
        print(f"Ответ:  {answer}")
        return 0
    except ApiError as exc:
        print(f"ОШИБКА: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
