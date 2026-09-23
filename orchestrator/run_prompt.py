#!/usr/bin/env python3
"""
Single-prompt sampling-knob harness for the local RKLLM / OpenAI-compatible server.

Unlike eval_intents.py (many rows, one config) this is one prompt, many configs —
for when the model gives you one bogus answer and you want to know which knob fixes it.

    python run_prompt.py                                  # replay the built-in bogus case
    python run_prompt.py --temp 0 --top-p 1 --top-k 1     # greedy
    python run_prompt.py --temp 0.7 --repeat 5            # how unstable is it?
    python run_prompt.py --sweep temp=0,0.3,0.7,1.0       # one knob, several values
    python run_prompt.py --messages conv.json             # your own conversation
    python run_prompt.py --no-history                     # drop prior turns, keep system+last user
    python run_prompt.py --raw                            # dump the whole JSON response

IMPORTANT: only the flags you actually pass are put in the payload. Passing nothing
reproduces production exactly, because orchestrator/llm_module.py sends no sampling
params at all — the server decides. That also means: if a knob changes nothing here,
the likely answer is that this patched flask_server.py ignores it, not that the knob
doesn't matter. Use --raw / --echo-payload to check what actually went over the wire.
"""

import argparse
import ast
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("JARVIS_LLM_URL", "http://localhost:8080/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("JARVIS_LLM_MODEL", "rkllm")

# The exact conversation from the 2026-09-23 12:49 DEBUG line, so `python run_prompt.py`
# with no arguments reproduces the bad answer before you start turning knobs.
DEFAULT_MESSAGES = [
    {
        "role": "system",
        "content": (
            "Ты — Джарвис, голосовой помощник в доме. Тебя слушают, а не читают.\n\n"
            "Правила ответа:\n"
            "- Коротко: одно-два предложения, без списков и разметки.\n"
            "- Разговорный русский. Все числа ниже уже написаны словами — так их и "
            "произноси, не переводи обратно в цифры.\n"
            "- Если нужных данных нет в контексте ниже — скажи об этом прямо. Не выдумывай.\n"
            "В твоём ответе должны быть только буквы и знаки препинания. Никаких чисел, "
            "никаких специальных символов. Всё прописывай словами.\n"
            "С тобой говорит Даниил. Возраст: двадцать два года.\n"
            "Сейчас: среда, двадцать третье сентября, двенадцать часов сорок девять минут\n\n"
            "Расписание (завтра):\n"
            "- четырнадцать часов тридцать минут Теория статистических решений "
            "(практическое занятие), Аудитория: т четыре тысячи двести десять\n"
            "- шестнадцать часов двадцать минут Механика (лекция), Аудитория: двести девять КПА"
        ),
    },
    {"role": "user", "content": "Привет! Какого нет, сбисание на завтра."},
    {"role": "assistant", "content": "Джарвис: Привет! Как у тебя дела?"},
    {"role": "user", "content": "Привет! Какое у меня расписание на завтра?"},
]

# flag dest -> JSON key in the request body. Add a line here if you patch the server
# to accept another knob; nothing else in the script needs to change.
PARAM_KEYS = {
    "temp": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repeat_penalty": "repeat_penalty",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "max_tokens": "max_tokens",
    "seed": "seed",
    "n_keep": "n_keep",
}


def coerce(text):
    """'0.7' -> 0.7, '40' -> 40, 'true' -> True, anything else stays a string."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        low = text.lower()
        if low in ("true", "false"):
            return low == "true"
        return text


def load_messages(path):
    """Accepts a JSON array, a {"messages": [...]} object, or a Python-repr list.

    The Python-repr case is deliberate: you can paste the `[{'role': 'system', ...}]`
    straight out of the DEBUG log into a file without converting the quotes.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = ast.literal_eval(text)
    if isinstance(data, dict):
        data = data["messages"]
    if not isinstance(data, list) or not all(isinstance(m, dict) for m in data):
        raise SystemExit(f"{path}: expected a list of message dicts")
    return data


def trim_history(messages):
    """Keep the system prompt and the final user turn; drop everything in between.

    Worth a run on its own: the pasted log has a garbled ASR turn and an assistant
    reply that starts with 'Джарвис:', and a 0.5B model will happily copy both the
    garbling and the name prefix. If --no-history fixes the answer, the problem is
    your history, not your sampling.
    """
    system = [m for m in messages if m.get("role") == "system"][:1]
    users = [m for m in messages if m.get("role") == "user"]
    return system + users[-1:]


def build_payload(args, messages, overrides=None):
    payload = {"model": args.model, "messages": messages, "stream": False}
    for dest, key in PARAM_KEYS.items():
        value = getattr(args, dest)
        if value is not None:
            payload[key] = value
    if args.stop:
        payload["stop"] = args.stop
    for item in args.set or []:
        key, _, value = item.partition("=")
        payload[key.strip()] = coerce(value.strip())
    payload.update(overrides or {})
    return payload


def call(url, payload, timeout):
    """Returns (text, latency, body). text is None on failure."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        print(f"  [http {exc.code}] {detail}", file=sys.stderr)
        return None, time.perf_counter() - t0, None
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  [error] {exc}  (server up? curl {url.rsplit('/v1/', 1)[0]}/v1/models)",
              file=sys.stderr)
        return None, time.perf_counter() - t0, None
    except json.JSONDecodeError as exc:
        print(f"  [error] response was not JSON: {exc}", file=sys.stderr)
        return None, time.perf_counter() - t0, None
    dt = time.perf_counter() - t0
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print(f"  [error] unexpected shape: {json.dumps(body, ensure_ascii=False)[:300]}",
              file=sys.stderr)
        return None, dt, body
    return text, dt, body


def show(text, dt, body, raw):
    if raw and body is not None:
        print(json.dumps(body, ensure_ascii=False, indent=2))
    usage = (body or {}).get("usage") or {}
    completion = usage.get("completion_tokens")
    rate = f"  {completion / dt:.1f} tok/s" if completion and dt else ""
    meta = f"{dt:.2f}s"
    if usage:
        meta += (f"  prompt {usage.get('prompt_tokens', '?')}"
                 f"  completion {completion if completion is not None else '?'}{rate}")
    print(f"  [{meta}]")
    if text is None:
        return
    for line in text.splitlines() or [""]:
        print(f"  > {line}")
    # Cheap checks against the house rules in the system prompt: digits are forbidden
    # (Piper reads the text aloud) and the model likes to prefix its own name.
    flags = []
    if any(ch.isdigit() for ch in text):
        flags.append("contains digits")
    if text.lstrip().lower().startswith(("джарвис:", "jarvis:", "assistant:")):
        flags.append("name prefix")
    if flags:
        print(f"  !! {', '.join(flags)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=float, default=120.0)

    ap.add_argument("--messages", help="JSON (or Python-repr) file with the conversation")
    ap.add_argument("--system", help="override the system prompt with this text")
    ap.add_argument("--system-file", help="override the system prompt from a file")
    ap.add_argument("--user", help="replace the final user turn with this text")
    ap.add_argument("--no-history", action="store_true",
                    help="keep only the system prompt and the last user turn")

    ap.add_argument("--temp", "--temperature", dest="temp", type=float)
    ap.add_argument("--top-p", dest="top_p", type=float)
    ap.add_argument("--top-k", dest="top_k", type=int)
    ap.add_argument("--min-p", dest="min_p", type=float)
    ap.add_argument("--repeat-penalty", dest="repeat_penalty", type=float)
    ap.add_argument("--frequency-penalty", dest="frequency_penalty", type=float)
    ap.add_argument("--presence-penalty", dest="presence_penalty", type=float)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--n-keep", dest="n_keep", type=int)
    ap.add_argument("--stop", action="append", help="stop string; repeatable")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="any other payload field, repeatable (e.g. --set stream=false)")

    ap.add_argument("--repeat", type=int, default=1, help="run the same config N times")
    ap.add_argument("--sweep", metavar="PARAM=V1,V2,...",
                    help="run once per value, e.g. --sweep temp=0,0.3,0.7")
    ap.add_argument("--raw", action="store_true", help="print the full JSON response")
    ap.add_argument("--echo-payload", action="store_true",
                    help="print the request body before sending")
    args = ap.parse_args()

    messages = load_messages(args.messages) if args.messages else [
        dict(m) for m in DEFAULT_MESSAGES
    ]
    if args.system_file:
        with open(args.system_file, encoding="utf-8") as f:
            args.system = f.read().strip()
    if args.system is not None:
        messages = [m for m in messages if m.get("role") != "system"]
        messages.insert(0, {"role": "system", "content": args.system})
    if args.user is not None:
        last = max((i for i, m in enumerate(messages) if m.get("role") == "user"),
                   default=None)
        if last is None:
            messages.append({"role": "user", "content": args.user})
        else:
            messages[last] = {"role": "user", "content": args.user}
    if args.no_history:
        messages = trim_history(messages)

    if args.sweep:
        name, _, values = args.sweep.partition("=")
        name = name.strip()
        key = PARAM_KEYS.get(name, name)
        configs = [{key: coerce(v.strip())} for v in values.split(",") if v.strip()]
    else:
        configs = [{}]

    last_user = next((m["content"] for m in reversed(messages)
                      if m.get("role") == "user"), "")
    print(f"endpoint : {args.url}")
    print(f"model    : {args.model}")
    print(f"messages : {len(messages)} "
          f"({', '.join(m.get('role', '?') for m in messages)})")
    print(f"question : {last_user!r}")

    for overrides in configs:
        payload = build_payload(args, messages, overrides)
        knobs = {k: v for k, v in payload.items() if k not in ("model", "messages", "stream")}
        label = ", ".join(f"{k}={v}" for k, v in knobs.items()) or "server defaults (no params sent)"
        print(f"\n{'=' * 62}\n{label}\n{'=' * 62}")
        if args.echo_payload:
            print(json.dumps(payload, ensure_ascii=False, indent=2))

        texts, lats = [], []
        for i in range(args.repeat):
            if args.repeat > 1:
                print(f"  --- run {i + 1}/{args.repeat}")
            text, dt, body = call(args.url, payload, args.timeout)
            show(text, dt, body, args.raw)
            texts.append(text)
            lats.append(dt)

        if args.repeat > 1:
            unique = {t for t in texts}
            print(f"  {len(unique)} distinct answer(s) in {args.repeat} runs   "
                  f"median {statistics.median(lats):.2f}s")


if __name__ == "__main__":
    main()
