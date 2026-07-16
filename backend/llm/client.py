"""
Multi-provider LLM client: gemini | openai | anthropic | ollama.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dotenv = os.path.join(_BACKEND_DIR, ".env")
if os.path.exists(_dotenv):
    load_dotenv(dotenv_path=_dotenv)

PROVIDER_DEFAULTS = {
    "gemini": "gemini-2.5-flash-lite",
    "openai": "gpt-4o",
    "anthropic": "claude-opus-4-7",
    "ollama": "llama3.2",
}

_provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
_model = os.getenv("LLM_MODEL", "").strip() or PROVIDER_DEFAULTS.get(_provider, "")
_client = None
_ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
_ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))
_ollama_think = os.getenv("OLLAMA_THINK", "false").strip().lower() in (
    "1", "true", "yes", "on",
)
_ollama_num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "8192"))

SYSTEM_PROMPT = """\
You are AlgoTrader AI's code engine: a quantitative Python engineer specializing \
in the backtrader library for stock strategy backtesting and optimization.

Hard rules:
1. Follow the user message instructions exactly.
2. Output ONLY the requested fenced blocks (```python and/or ```json). \
No prose outside those fences.
3. Assume `bt`, `numpy` (`np`), and `datetime` are already in scope — do not import them \
unless the user prompt explicitly allows it.
4. Never implement `notify_trade` or `notify_order` unless asked.
5. Prefer configurable `params` over hardcoded magic numbers.
6. When returning JSON, use only the parameter keys requested.
7. Prefer standard `bt.indicators.*` APIs.
8. Do not emit <think> tags or chain-of-thought outside code/json fences.
"""


def provider_name() -> str:
    return _provider


def is_ready() -> bool:
    return _client is not None


def not_ready_message() -> str:
    if _provider == "ollama":
        return (
            f"Ollama client not initialised (base URL: {_ollama_base}). "
            "Start Ollama (`ollama serve`), set LLM_PROVIDER=ollama in .env, "
            f"and pull a model (`ollama pull {_model or 'llama3.2'}`)."
        )
    return (
        f"LLM client not initialised. "
        f"Check {_provider.upper()}_API_KEY and LLM_PROVIDER in your .env."
    )


def _extract_ollama_text(message) -> str:
    if message is None:
        return ""
    if isinstance(message, dict):
        content = (message.get("content") or "").strip()
        thinking = (message.get("thinking") or "").strip()
    else:
        content = (getattr(message, "content", None) or "").strip()
        thinking = (getattr(message, "thinking", None) or "").strip()
    if content:
        return content
    if thinking and ("```python" in thinking or "```json" in thinking):
        return thinking
    return ""


def _ollama_chat(prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    options = {"num_predict": _ollama_num_predict}

    try:
        import ollama as ollama_sdk  # type: ignore

        client = (
            _client
            if _client is not None and hasattr(_client, "chat")
            else ollama_sdk.Client(host=_ollama_base, timeout=_ollama_timeout)
        )
        try:
            response = client.chat(
                model=_model, messages=messages, options=options, think=_ollama_think
            )
        except TypeError:
            response = client.chat(model=_model, messages=messages, options=options)
        if isinstance(response, dict):
            text = _extract_ollama_text(response.get("message"))
        else:
            text = _extract_ollama_text(getattr(response, "message", None))
        if text:
            return text
        raise RuntimeError(
            f"Ollama returned empty content for model {_model!r} (think={_ollama_think})."
        )
    except ImportError:
        pass
    except RuntimeError:
        raise
    except Exception as e:
        if "empty content" in str(e):
            raise
        print(f"WARNING: ollama SDK chat failed ({e}); falling back to HTTP")

    payload = json.dumps({
        "model": _model,
        "messages": messages,
        "stream": False,
        "think": _ollama_think,
        "options": options,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{_ollama_base}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_ollama_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {_ollama_base}: {e}. Is `ollama serve` running?"
        ) from e

    content = _extract_ollama_text(data.get("message"))
    if not content:
        raise RuntimeError(
            f"Ollama returned empty content for model {_model!r} "
            f"(think={_ollama_think}, done_reason={data.get('done_reason')!r})."
        )
    return content


def _init_client() -> None:
    global _client
    print(f"Attempting to configure LLM (provider: {_provider}, model: {_model}) ...")
    if _provider == "gemini":
        import google.genai as genai

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or "YOUR_API_KEY_HERE" in api_key:
            raise ValueError("GEMINI_API_KEY missing or still set to placeholder.")
        _client = genai.Client(api_key=api_key)
    elif _provider == "openai":
        import openai as openai_sdk

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY missing.")
        _client = openai_sdk.OpenAI(api_key=api_key)
    elif _provider == "anthropic":
        import anthropic as anthropic_sdk

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY missing.")
        _client = anthropic_sdk.Anthropic(api_key=api_key)
    elif _provider == "ollama":
        try:
            with urllib.request.urlopen(f"{_ollama_base}/api/tags", timeout=5) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
            full_names = [m.get("name", "") for m in tags.get("models", [])]
            available = [n.split(":")[0] for n in full_names]
            model_base = _model.split(":")[0]
            if full_names and _model not in full_names and model_base not in available:
                print(
                    f"WARNING: model {_model!r} not found in Ollama "
                    f"(available: {full_names or 'none'}). Pull: ollama pull {_model}"
                )
        except Exception as probe_err:
            raise ValueError(
                f"Ollama not reachable at {_ollama_base}: {probe_err}."
            ) from probe_err
        try:
            import ollama as ollama_sdk  # type: ignore

            _client = ollama_sdk.Client(host=_ollama_base, timeout=_ollama_timeout)
            print(f"Using official ollama Python client → {_ollama_base}")
        except ImportError:
            _client = {"type": "ollama", "base_url": _ollama_base}
            print(f"Using Ollama HTTP API → {_ollama_base}")
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER {_provider!r}. "
            "Supported: gemini, openai, anthropic, ollama."
        )
    print(f"LLM client initialised (provider: {_provider}, model: {_model}).")


try:
    print("--- Initializing Application ---")
    _init_client()
except Exception as e:
    print(f"CRITICAL ERROR configuring LLM: {e}")
    _client = None


def call_llm(prompt: str) -> str:
    if _client is None:
        raise RuntimeError(not_ready_message())
    if _provider == "gemini":
        return _client.models.generate_content(model=_model, contents=prompt).text
    if _provider == "openai":
        return _client.chat.completions.create(
            model=_model,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content
    if _provider == "anthropic":
        return _client.messages.create(
            model=_model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        ).content[0].text
    if _provider == "ollama":
        return _ollama_chat(prompt)
    raise RuntimeError(f"Unhandled provider: {_provider}")
