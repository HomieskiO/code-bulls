"""
graph.py — Two LangGraph workflows:
  1. Single-ticker  : yfinance data, configurable start date (no 2015 limit)
  2. Screened multi : Gemini generates stock-screener code → runs on Kaggle
                      dataset → CSV of daily ticker picks → multi-asset backtest
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import json
import backtrader as bt
import backtrader.linebuffer as _bt_lb
import yfinance as yf
from evaluator import (
    Expectancy,
    get_metrics,
    CAGRAnalyzer,
    PortfolioValueAnalyzer,
    TradeLogAnalyzer,
)
import os
import re
import tempfile
import numpy as np
import pandas as pd
import datetime as dt_module
from datetime import datetime
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Patch backtrader's LinesOperation to tolerate ZeroDivisionError
# ---------------------------------------------------------------------------
# LinesOperation._once_op (batch path) and .next (bar-by-bar path) apply
# binary operations with raw operator.truediv and no zero-guard.
# RSI raises ZeroDivisionError when avg_loss = 0 (e.g. stock opens with
# 14+ consecutive up-days).  Substituting NaN is safe: NaN comparisons
# in strategy conditions (e.g. rsi < 35) evaluate to False.

def _safe_lines_once_op(self, start, end):
    dst  = self.array
    srca = self.a.array
    srcb = self.b.array
    op   = self.operation
    for i in range(start, end):
        try:
            dst[i] = op(srca[i], srcb[i])
        except ZeroDivisionError:
            dst[i] = float("nan")


def _safe_lines_next(self):
    try:
        if self.bline:
            self[0] = self.operation(self.a[0], self.b[0])
        elif not self.r:
            if not self.btime:
                self[0] = self.operation(self.a[0], self.b)
            else:
                self[0] = self.operation(self.a.time(), self.b)
        else:
            self[0] = self.operation(self.a, self.b[0])
    except ZeroDivisionError:
        self[0] = float("nan")


_bt_lb.LinesOperation._once_op = _safe_lines_once_op
_bt_lb.LinesOperation.next      = _safe_lines_next


# ---------------------------------------------------------------------------
# Environment / API key loading
# ---------------------------------------------------------------------------

print("--- Initializing Application ---")
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    print("Found .env file, loading environment variables.")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print("WARNING: .env file not found. Please create one with your API key(s).")

# ---------------------------------------------------------------------------
# Multi-provider LLM client
#
# Set in .env:
#   LLM_PROVIDER = gemini | openai | anthropic | ollama   (default: gemini)
#   LLM_MODEL    = <model name>                           (optional – provider default)
#
# Cloud keys (provider-specific):
#   GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
#
# Local Ollama (no API key required):
#   OLLAMA_BASE_URL = http://localhost:11434   (optional)
#   OLLAMA_TIMEOUT  = 600                      (seconds; local models can be slow)
#   LLM_MODEL       = llama3.2 | qwen2.5-coder | mistral | …
#   Ensure `ollama serve` is running and the model is pulled (`ollama pull <model>`).
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS = {
    "gemini":    "gemini-2.5-flash",
    "openai":    "gpt-4o",
    "anthropic": "claude-opus-4-7",
    "ollama":    "llama3.2",
}

_llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
_model_name   = os.getenv("LLM_MODEL", "").strip() or _PROVIDER_DEFAULTS.get(_llm_provider, "")
_llm_client   = None
_ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
_ollama_timeout  = int(os.getenv("OLLAMA_TIMEOUT", "600"))
# Qwen3 / similar "thinking" models put chain-of-thought in message.thinking and
# often leave message.content empty until the think budget finishes — or forever
# if num_predict is exhausted mid-thought. Default OFF for reliable code gen.
_ollama_think = os.getenv("OLLAMA_THINK", "false").strip().lower() in (
    "1", "true", "yes", "on",
)
_ollama_num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "8192"))

# Fixed system message for local models — keeps format discipline without
# duplicating the large task prompts (code gen / optimize / screening).
_OLLAMA_SYSTEM_PROMPT = """\
You are AlgoTrader AI's code engine: a quantitative Python engineer specializing \
in the backtrader library for stock strategy backtesting and optimization.

Your job is to turn natural-language trading ideas into valid backtrader strategy \
code, screening scripts, or parameter JSON for iterative optimization.

Hard rules:
1. Follow the user message instructions exactly.
2. Output ONLY the requested fenced blocks (```python and/or ```json). \
No prose, preambles, markdown headings, or explanations outside those fences.
3. Assume `bt`, `numpy` (`np`), and `datetime` are already in the global scope — \
do not import them (or anything else) unless the user prompt explicitly allows it.
4. Never implement `notify_trade` or `notify_order` unless the user explicitly asks.
5. Prefer configurable `params` over hardcoded magic numbers.
6. When returning JSON, use only the parameter keys requested — do not rename, \
add, or remove keys.
7. Write correct, runnable Python. Prefer standard `bt.indicators.*` APIs.
8. Do not use internal chain-of-thought or <think> tags in the output.
"""


def _ollama_messages(prompt: str) -> list:
    """Build chat messages with the project system prompt for Ollama."""
    return [
        {"role": "system", "content": _OLLAMA_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]


def _extract_ollama_text(message) -> str:
    """
    Pull the user-facing answer from an Ollama message.

    Qwen3.x returns:
      message.content  -> final answer (often empty while thinking)
      message.thinking -> chain-of-thought (not usable as code)

    Prefer content; never treat pure thinking as a successful code response.
    """
    if message is None:
        return ""
    if isinstance(message, dict):
        content  = (message.get("content") or "").strip()
        thinking = (message.get("thinking") or "").strip()
    else:
        content  = (getattr(message, "content", None) or "").strip()
        thinking = (getattr(message, "thinking", None) or "").strip()
    if content:
        return content
    # Some older builds put the whole answer only in thinking if think is on —
    # try to salvage fenced blocks if present, otherwise fail clearly.
    if thinking and ("```python" in thinking or "```json" in thinking):
        return thinking
    return ""


def _ollama_chat(prompt: str) -> str:
    """
    Call a local Ollama instance via the native /api/chat endpoint.
    Uses the official `ollama` package when installed; otherwise stdlib HTTP.
    Always injects the project system prompt for format discipline.

    Disables model "thinking" by default (OLLAMA_THINK=true to enable) so
    Qwen3 etc. put tokens into message.content instead of message.thinking.
    """
    messages = _ollama_messages(prompt)
    options  = {"num_predict": _ollama_num_predict}

    # Prefer the official Python client when available
    try:
        import ollama as _ollama_sdk  # type: ignore
        client = (
            _llm_client
            if _llm_client is not None and hasattr(_llm_client, "chat")
            else _ollama_sdk.Client(host=_ollama_base_url, timeout=_ollama_timeout)
        )
        chat_kwargs = {
            "model":    _model_name,
            "messages": messages,
            "options":  options,
        }
        # ollama Python SDK accepts think= on recent versions; ignore if unsupported
        try:
            response = client.chat(**chat_kwargs, think=_ollama_think)
        except TypeError:
            response = client.chat(**chat_kwargs)

        if isinstance(response, dict):
            text = _extract_ollama_text(response.get("message"))
        else:
            text = _extract_ollama_text(getattr(response, "message", None))
        if text:
            return text
        raise RuntimeError(
            f"Ollama returned empty content for model {_model_name!r} "
            f"(think={_ollama_think}). For Qwen3 set OLLAMA_THINK=false and "
            f"raise OLLAMA_NUM_PREDICT if needed."
        )
    except ImportError:
        pass
    except RuntimeError:
        raise
    except Exception as e:
        # Fall through to HTTP if the SDK path fails for non-import reasons
        # that look like missing-think-param issues; re-raise hard failures.
        if "empty content" in str(e):
            raise
        print(f"WARNING: ollama SDK chat failed ({e}); falling back to HTTP")

    # Fallback: raw HTTP to Ollama REST API (no extra dependency)
    import urllib.error
    import urllib.request

    payload_obj = {
        "model":    _model_name,
        "messages": messages,
        "stream":   False,
        "think":    _ollama_think,   # Qwen3: false → fill content, not thinking
        "options":  options,
    }
    payload = json.dumps(payload_obj).encode("utf-8")
    req = urllib.request.Request(
        f"{_ollama_base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_ollama_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {_ollama_base_url}: {e}. "
            "Is `ollama serve` running? Try: ollama list"
        ) from e

    content = _extract_ollama_text(data.get("message"))
    if not content:
        msg = data.get("message") or {}
        thinking_len = len((msg.get("thinking") or "") if isinstance(msg, dict) else "")
        raise RuntimeError(
            f"Ollama returned empty content for model {_model_name!r} "
            f"(think={_ollama_think}, thinking_chars={thinking_len}, "
            f"done_reason={data.get('done_reason')!r}). "
            "Qwen3 models write chain-of-thought into `thinking` and leave "
            "`content` empty — set OLLAMA_THINK=false (default) and restart, "
            "or increase OLLAMA_NUM_PREDICT."
        )
    return content


try:
    print(f"Attempting to configure LLM (provider: {_llm_provider}, model: {_model_name}) ...")
    if _llm_provider == "gemini":
        import google.genai as genai
        _api_key = os.getenv("GEMINI_API_KEY", "")
        if not _api_key or "YOUR_API_KEY_HERE" in _api_key:
            raise ValueError("GEMINI_API_KEY missing or still set to placeholder.")
        _llm_client = genai.Client(api_key=_api_key)
    elif _llm_provider == "openai":
        import openai as _openai_sdk
        _api_key = os.getenv("OPENAI_API_KEY", "")
        if not _api_key:
            raise ValueError("OPENAI_API_KEY missing.")
        _llm_client = _openai_sdk.OpenAI(api_key=_api_key)
    elif _llm_provider == "anthropic":
        import anthropic as _anthropic_sdk
        _api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not _api_key:
            raise ValueError("ANTHROPIC_API_KEY missing.")
        _llm_client = _anthropic_sdk.Anthropic(api_key=_api_key)
    elif _llm_provider == "ollama":
        # Probe that Ollama is reachable; model existence is checked on first call.
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"{_ollama_base_url}/api/tags", timeout=5
            ) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
            available = [
                m.get("name", "").split(":")[0]
                for m in tags.get("models", [])
            ]
            full_names = [m.get("name", "") for m in tags.get("models", [])]
            model_base = _model_name.split(":")[0]
            if full_names and _model_name not in full_names and model_base not in available:
                print(
                    f"WARNING: model {_model_name!r} not found in Ollama "
                    f"(available: {full_names or 'none'}). "
                    f"Pull it with: ollama pull {_model_name}"
                )
        except Exception as probe_err:
            raise ValueError(
                f"Ollama not reachable at {_ollama_base_url}: {probe_err}. "
                "Start it with `ollama serve` and ensure OLLAMA_BASE_URL is correct."
            ) from probe_err

        # Prefer official client when installed; otherwise _call_llm uses HTTP.
        try:
            import ollama as _ollama_sdk  # type: ignore
            _llm_client = _ollama_sdk.Client(host=_ollama_base_url, timeout=_ollama_timeout)
            print(f"Using official ollama Python client → {_ollama_base_url}")
        except ImportError:
            _llm_client = {"type": "ollama", "base_url": _ollama_base_url}
            print(f"Using Ollama HTTP API → {_ollama_base_url} (install `ollama` package for native client)")
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER {_llm_provider!r}. "
            "Supported values: gemini, openai, anthropic, ollama."
        )
    print(f"LLM client initialised (provider: {_llm_provider}, model: {_model_name}).")
except Exception as e:
    print(f"CRITICAL ERROR configuring LLM: {e}")
    _llm_client = None


def _llm_not_ready_message() -> str:
    if _llm_provider == "ollama":
        return (
            f"Ollama client not initialised (base URL: {_ollama_base_url}). "
            "Start Ollama (`ollama serve`), set LLM_PROVIDER=ollama in .env, "
            f"and pull a model (`ollama pull {_model_name or 'llama3.2'}`)."
        )
    return (
        f"LLM client not initialised. "
        f"Check {_llm_provider.upper()}_API_KEY and LLM_PROVIDER in your .env."
    )


def _call_llm(prompt: str) -> str:
    if _llm_client is None:
        raise RuntimeError(_llm_not_ready_message())
    if _llm_provider == "gemini":
        response = _llm_client.models.generate_content(model=_model_name, contents=prompt)
        return response.text
    if _llm_provider == "openai":
        response = _llm_client.chat.completions.create(
            model=_model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    if _llm_provider == "anthropic":
        response = _llm_client.messages.create(
            model=_model_name,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    if _llm_provider == "ollama":
        return _ollama_chat(prompt)
    raise RuntimeError(f"Unhandled provider: {_llm_provider}")


_call_gemini = _call_llm  # backward-compat alias (used by main.py)


# ---------------------------------------------------------------------------
# Kaggle dataset path
# ---------------------------------------------------------------------------

KAGGLE_STOCKS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "Stock Market Dataset", "Stocks")
)

# ---------------------------------------------------------------------------
# Backtrader API documentation (injected into Gemini prompts)
# ---------------------------------------------------------------------------

def _build_bt_api_docs() -> str:
    """
    Build a compact, accurate backtrader API reference from bt_api.json
    (method signatures / docs) and live introspection (indicator params / lines).
    Returns a string ready to embed in LLM prompts.
    """
    lines = []

    # ── 1. Indicator catalogue (live introspection is authoritative for params) ──
    _SKIP_LINE_ATTRS = {
        "advance","backwards","buflen","extend","extrasize","forward","fullsize",
        "get","getlinealiases","home","itersize","reset","rewind","size",
    }
    _SKIP_PARAMS = {"movav", "_movav", "_rocperiod"}

    indicator_specs = [
        ("RSI",             bt.indicators.RSI),
        ("SMA",             bt.indicators.SMA),
        ("EMA",             bt.indicators.EMA),
        ("DEMA",            bt.indicators.DEMA),
        ("TEMA",            bt.indicators.TEMA),
        ("MACD",            bt.indicators.MACD),
        ("MACDHisto",       bt.indicators.MACDHisto),
        ("BollingerBands",  bt.indicators.BollingerBands),
        ("ATR",             bt.indicators.ATR),
        ("Stochastic",      bt.indicators.Stochastic),
        ("StochasticFull",  bt.indicators.StochasticFull),
        ("CrossOver",       bt.indicators.CrossOver),
        ("WilliamsR",       bt.indicators.WilliamsR),
        ("CCI",             bt.indicators.CCI),
        ("Momentum",        bt.indicators.Momentum),
        ("ROC",             bt.indicators.ROC),
        ("ROC100",          bt.indicators.ROC100),
        ("Highest",         bt.indicators.Highest),
        ("Lowest",          bt.indicators.Lowest),
        ("Trix",            bt.indicators.Trix),
        ("AroonUp",         bt.indicators.AroonUp),
        ("AroonDown",       bt.indicators.AroonDown),
        ("AroonOscillator", bt.indicators.AroonOscillator),
    ]

    lines.append("## Available bt.indicators (use bt.indicators.<Name>)")
    for name, cls in indicator_specs:
        try:
            params = {
                k: (v if not isinstance(v, type) else v.__name__)
                for k, v in cls.params._getpairs().items()
                if k not in _SKIP_PARAMS
            }
            output_lines = sorted(
                {a for a in dir(cls.lines) if not a.startswith("_") and a not in _SKIP_LINE_ATTRS}
            )
            lines.append(
                f"  bt.indicators.{name}({', '.join(f'{k}={v!r}' for k,v in params.items())})"
                f"  →  lines: {output_lines}"
            )
        except Exception:
            lines.append(f"  bt.indicators.{name}  (params unavailable)")

    # ── 2. Strategy lifecycle & trading methods (from bt_api.json) ──
    lines.append("")
    lines.append("## Strategy API  (self.<method>)")

    bt_api_path = os.path.join(os.path.dirname(__file__), "bt_api.json")
    api_data = {}
    try:
        with open(bt_api_path) as fh:
            api_data = json.load(fh)
    except Exception:
        pass

    strat_members = api_data.get("backtrader.strategy.Strategy", {}).get("members", {})
    for mname in ["buy", "sell", "close", "getposition", "order_target_percent",
                  "order_target_size", "order_target_value", "cancel"]:
        m = strat_members.get(mname, {})
        if m:
            sig  = m.get("signature", mname)
            doc  = (m.get("doc") or "").strip().splitlines()[0][:100]
            lines.append(f"  {sig}")
            if doc:
                lines.append(f"    # {doc}")

    lines.append("  self.broker.getcash()          # current available cash")
    lines.append("  self.broker.getvalue()         # total portfolio value (cash + positions)")
    lines.append("  self.getposition(data).size    # shares held for a feed (0 if flat)")
    lines.append("  self.getposition(data).price   # avg entry price for current position")

    # ── 3. Trade object attributes ──
    lines.append("")
    lines.append("## Trade object  (received in notify_trade — DO NOT implement notify_trade)")
    lines.append("  trade.pnl           # gross profit/loss")
    lines.append("  trade.pnlcomm       # net profit/loss after commission")
    lines.append("  trade.commission    # total commission paid")
    lines.append("  trade.size          # position size when closed")
    lines.append("  trade.price         # avg entry price")
    lines.append("  trade.value         # position value")
    lines.append("  trade.isclosed      # True when trade is closed")
    lines.append("  trade.data._name    # symbol name of the data feed")
    lines.append("  NOTE: 'trade.comm' does NOT exist — use trade.pnlcomm or trade.commission")

    # ── 4. Order object attributes ──
    lines.append("")
    lines.append("## Order object  (received in notify_order — DO NOT implement notify_order)")
    order_members = api_data.get("backtrader.order.Order", {}).get("members", {})
    status_attrs = {k: v["value_repr"] for k, v in order_members.items()
                    if v.get("kind") == "attribute" and not k.startswith("_")
                    and k[0].isupper() and v["value_repr"].lstrip("-").isdigit()}
    lines.append(f"  Status constants: {status_attrs}")
    lines.append("  order.status                   # current status integer")
    lines.append("  order.executed.price           # fill price")
    lines.append("  order.executed.value           # fill value")
    lines.append("  order.executed.comm            # commission on this fill")
    lines.append("  order.executed.size            # filled size")
    lines.append("  order.isbuy() / order.issell() # direction helpers")

    return "\n".join(lines)


_BT_API_DOCS = _build_bt_api_docs()


# ---------------------------------------------------------------------------
# Best-config scoring
# ---------------------------------------------------------------------------

def _config_score(metrics: dict) -> float:
    cagr = float(metrics.get("cagr",         0) or 0)
    dd   = abs(float(metrics.get("max_drawdown", 0) or 0))
    return cagr * 0.65 - dd * 0.35


# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    strategy_prompt: str
    start_date: str          # "YYYY-MM-DD" — inclusive lower bound for performance
    end_date: str            # "YYYY-MM-DD" — inclusive upper bound for performance
    position_size_pct: float # % of available cash per trade (default 100 for single)
    generated_code: str
    current_iteration_number: int
    current_config: dict
    all_iteration_results: List[dict]
    best_config_so_far: dict
    error: str


class ScreenGraphState(TypedDict):
    strategy_prompt: str
    screening_prompt: str
    start_date: str          # "YYYY-MM-DD" lower bound
    end_date: str            # "YYYY-MM-DD" upper bound (dataset max 2017-11-10)
    position_size_pct: float # % of available cash per new position (default 10 multi)
    generated_code: str
    screening_code: str
    screening_csv_path: str
    screening_dict: dict     # {"YYYY-MM-DD": ["AAPL", "TSLA", ...]}
    current_iteration_number: int
    current_config: dict
    all_iteration_results: List[dict]
    best_config_so_far: dict
    error: str


def _position_size_pct(state: dict, default: float = 100.0) -> float:
    try:
        v = float(state.get("position_size_pct", default) or default)
    except (TypeError, ValueError):
        v = default
    return max(1.0, min(100.0, v))


def _position_sizing_instructions(pct: float) -> str:
    """Text injected into LLM prompts so generated strategies respect stake size."""
    frac = round(pct / 100.0, 4)
    return (
        f"POSITION SIZING (mandatory):\n"
        f"- Allocate exactly {pct:g}% of currently available cash to each new entry "
        f"(stake_pct = {frac}).\n"
        f"- Include ('stake_pct', {frac}) in params and in the ```json defaults.\n"
        f"- On buy, either call self.buy() / self.buy(data=d) with NO size= argument "
        f"(the runtime PercentSizer applies stake_pct), OR compute "
        f"size = int((self.broker.getcash() * self.p.stake_pct) / price) and pass size=.\n"
        f"- Do NOT hardcode a different stake. Do NOT use 95% or all-in unless stake_pct says so.\n"
    )


def _ensure_stake_pct_in_config(config: dict, pct: float) -> dict:
    cfg = dict(config or {})
    cfg["stake_pct"] = round(float(pct) / 100.0, 4)
    return cfg



def _strategy_param_names(cls) -> set:
    """Names declared on a bt.Strategy params block."""
    try:
        return set(cls.params._getkeys())
    except Exception:
        try:
            return set(dict(cls.params._getpairs()).keys())
        except Exception:
            return set()


def _inject_stake_pct_param(code: str, pct: float) -> str:
    """
    Ensure generated strategy source declares stake_pct so it stays visible
    in the code tab and matches the runtime sizer.
    """
    frac = round(float(pct) / 100.0, 4)
    if re.search(r"""['"]stake_pct['"]""", code):
        # rewrite existing default if present
        code = re.sub(
            r"""\(\s*['"]stake_pct['"]\s*,\s*[^)]+\)""",
            f"('stake_pct', {frac})",
            code,
        )
        return code

    # Insert into params = ( ... ) block after opening paren
    m = re.search(r"params\s*=\s*\(\s*\n?", code)
    if m:
        insert_at = m.end()
        line = f"        ('stake_pct', {frac}),\n"
        return code[:insert_at] + line + code[insert_at:]

    # Fallback: prepend a comment (still enforced via cerebro sizer)
    return (
        f"# position size: {pct:g}% of available cash per trade (stake_pct={frac})\n"
        + code
    )


# ---------------------------------------------------------------------------
# Shared utility helpers
# ---------------------------------------------------------------------------

def get_ticker_from_prompt(prompt: str) -> str:
    """Extract the first uppercase 1–5 letter word as the ticker symbol."""
    match = re.search(r"\b([A-Z]{1,5})\b", prompt)
    if match:
        return match.group(1)
    raise ValueError("Could not extract a valid stock ticker from the prompt.")


def _extract_fenced_block(text: str, lang: str) -> str | None:
    """
    Extract a ```lang ... ``` fenced block. Tolerates missing closing fence
    (truncated local-model outputs) and optional language tags.
    """
    # Closed fence: ```python\n...\n```
    m = re.search(
        rf"```{lang}\s*\n(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # Unclosed fence (model stopped mid-generation)
    m = re.search(
        rf"```{lang}\s*\n(.*)$",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


def _looks_truncated(code: str) -> bool:
    """Heuristic: strategy source cut off mid-statement."""
    if not code or not code.strip():
        return True
    s = code.rstrip()
    if s.endswith("\\"):
        return True
    # incomplete logical/continuation tails
    if re.search(
        r"(?:\b(?:and|or|not|if|elif|else|return|with|for|while|def|class)\b"
        r"|[,=\(\[\{+\-*/:])\s*$",
        s,
    ):
        return True
    # rough bracket balance
    opens  = s.count("(") + s.count("[") + s.count("{")
    closes = s.count(")") + s.count("]") + s.count("}")
    if opens > closes:
        return True
    # must define a Strategy class to be considered complete enough
    if "class " not in s or "bt.Strategy" not in s:
        return True
    if "def next" not in s:
        return True
    return False


def _infer_config_from_code(code: str) -> dict:
    """
    Build a default config dict from a strategy's `params = (...)` declaration
    when the model omitted the ```json block.
    """
    config: dict = {}

    def _parse_literal(raw: str):
        val = raw.strip().rstrip(",")
        if val in ("None", "null"):
            return None
        if val in ("True", "False"):
            return val == "True"
        try:
            return json.loads(val)
        except Exception:
            try:
                return float(val) if "." in val else int(val)
            except Exception:
                return val.strip("'\"")

    # Collect ('name', value) pairs from a params = ( ... ) region when present
    params_idx = re.search(r"\bparams\s*=", code)
    search_region = code[params_idx.start():params_idx.start() + 1500] if params_idx else code
    for name, raw in re.findall(
        r"""\(\s*['"](\w+)['"]\s*,\s*(None|True|False|null|-?\d+\.?\d*)\s*\)""",
        search_region,
    ):
        config[name] = _parse_literal(raw)

    # Fallback: self.params.x = ... assignments in __init__
    if not config:
        for name, raw in re.findall(
            r"""self\.params\.(\w+)\s*=\s*([^\n#]+)""",
            code,
        ):
            config[name] = _parse_literal(raw)

    config.setdefault("stop_loss", None)
    config.setdefault("take_profit", None)
    return config


def _extract_code_and_config(response_text: str, allow_infer_config: bool = True):
    """
    Parse the LLM response for a ```python block and a ```json block.

    Local models often omit the json fence or truncate the closing ``` —
    we tolerate that and, when needed, infer config from the strategy params.
    """
    code = _extract_fenced_block(response_text, "python")
    if not code:
        # bare class without fences
        m = re.search(
            r"(class\s+\w+\s*\(\s*bt\.Strategy\s*\).*)$",
            response_text,
            re.DOTALL,
        )
        if m:
            code = m.group(1).strip()
    if not code:
        raise ValueError(
            "LLM response is missing the required ```python code block.\n"
            f"Response was:\n{response_text[:500]}"
        )

    if _looks_truncated(code):
        raise ValueError(
            "LLM strategy code looks truncated (incomplete statement / missing next()).\n"
            f"Code tail:\n{code[-300:]}"
        )

    json_raw = _extract_fenced_block(response_text, "json")
    config = None
    if json_raw:
        # tolerate trailing commas / surrounding noise
        try:
            config = json.loads(json_raw)
        except json.JSONDecodeError:
            # try first {...} object
            obj = re.search(r"\{.*\}", json_raw, re.DOTALL)
            if obj:
                try:
                    config = json.loads(obj.group(0))
                except json.JSONDecodeError:
                    config = None

    if config is None and allow_infer_config:
        config = _infer_config_from_code(code)
        if len(config) <= 2 and set(config.keys()) <= {"stop_loss", "take_profit"}:
            # only defaults — not enough
            config = None

    if config is None:
        raise ValueError(
            "LLM response is missing a usable ```json config block "
            "(and could not infer params from the code).\n"
            f"Response was:\n{response_text[:500]}"
        )

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object, got {type(config)}")

    config.setdefault("stop_loss", None)
    config.setdefault("take_profit", None)
    return code, config


def _flatten_yfinance_df(df):
    """Flatten MultiIndex yfinance columns to single-level OHLCV."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if "adj close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj close": "close"})
    return df[["open", "high", "low", "close", "volume"]].copy()


def _make_exec_namespace() -> dict:
    """Return a namespace dict pre-populated with common imports."""
    import math
    return {
        "bt":       bt,
        "numpy":    np,
        "np":       np,
        "math":     math,
        "datetime": dt_module,
    }


_CODE_FIXES = {
    "bt.indicators.Crossover":   "bt.indicators.CrossOver",
    "bt.ind.Crossover":          "bt.indicators.CrossOver",
    "indicators.Crossover":      "indicators.CrossOver",
    "bt.indicators.crossover":   "bt.indicators.CrossOver",
    "bt.indicators.ema(":        "bt.indicators.EMA(",
    "bt.indicators.sma(":        "bt.indicators.SMA(",
    "bt.indicators.rsi(":        "bt.indicators.RSI(",
    "bt.indicators.macd(":       "bt.indicators.MACD(",
    "bt.indicators.bollinger":   "bt.indicators.BollingerBands",
    "bt.indicators.Bollinger(":  "bt.indicators.BollingerBands(",
}

def _sanitize_code(code: str) -> str:
    """Fix predictable naming mistakes in LLM-generated backtrader code."""
    for wrong, right in _CODE_FIXES.items():
        code = code.replace(wrong, right)
    return code


# ===========================================================================
# WORKFLOW 1 — SINGLE-TICKER  (yfinance, no date limit)
# ===========================================================================

_CODE_GEN_PROMPT = (
"""You are an expert in the `backtrader` Python library.
Convert the natural language trading strategy below into a complete,
valid `backtrader.Strategy` class.

Crucial requirements:
1. Configurable via `params` dict — do NOT hardcode numeric values.
2. The class MUST import nothing; assume `bt`, `numpy as np`, and `datetime`
   are already available in the global scope.
3. Output Format: provide EXACTLY one ```python block (the strategy class)
   and one ```json block (the default parameter values).
4. Do NOT define `notify_trade` or `notify_order` methods — they are not needed and frequently cause AttributeError.
5. Write NO comments in the code — no inline comments, no docstrings, nothing.
6. Always include `stop_loss = None` and `take_profit = None` in `params`.
   In `next()`, after your normal entry/exit logic, add a risk-exit block:
     if self.position.size > 0:
         if self.params.stop_loss is not None and self.data.close[0] <= self.position.price * (1 - self.params.stop_loss):
             self.close()
         elif self.params.take_profit is not None and self.data.close[0] >= self.position.price * (1 + self.params.take_profit):
             self.close()
   Set both to `null` in the ```json config block unless the user explicitly requested them.
7. POSITION SIZING — follow the POSITION SIZING block; include stake_pct in params and json.
   Prefer self.buy() with no size= argument (runtime PercentSizer enforces stake).

{position_sizing}

--- BACKTRADER API REFERENCE ---
{api_docs}
--- END REFERENCE ---

User Strategy: "{{prompt}}"
"""
.replace("{api_docs}", _BT_API_DOCS.replace("{", "{{").replace("}", "}}"))
.replace("{{prompt}}", "{prompt}")
)

# Compact prompt for local models (Ollama) — full API docs blow the context
# window and cause truncated code / missing JSON.
_CODE_GEN_PROMPT_OLLAMA = """\
Convert this trading strategy into a complete backtrader.Strategy class.

User strategy:
{prompt}

{position_sizing}

MANDATORY output — nothing else, both fences closed:
```python
class MyStrategy(bt.Strategy):
    params = (
        ('fast', 15),
        ('slow', 50),
        ('stake_pct', {stake_frac}),
        ('stop_loss', None),
        ('take_profit', None),
    )
    def __init__(self):
        self.fast_ema = bt.indicators.EMA(period=self.p.fast)
        self.slow_ema = bt.indicators.EMA(period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ema, self.slow_ema)
    def next(self):
        if not self.position:
            if self.crossover[0] > 0:
                self.buy()
        else:
            if self.crossover[0] < 0:
                self.close()
        if self.position.size > 0:
            if self.p.stop_loss is not None and self.data.close[0] <= self.position.price * (1 - self.p.stop_loss):
                self.close()
            elif self.p.take_profit is not None and self.data.close[0] >= self.position.price * (1 + self.p.take_profit):
                self.close()
```
```json
{{"fast": 15, "slow": 50, "stake_pct": {stake_frac}, "stop_loss": null, "take_profit": null}}
```

Rules:
- Adapt the example to the user strategy (indicators, periods, conditions).
- Use params = ( ('name', default), ... ) — never assign self.params.x = ... in __init__.
- Use self.p.name to read params. No imports. No notify_trade/notify_order. No comments.
- Always include stop_loss, take_profit, and stake_pct={stake_frac}.
- On entry call self.buy() with NO size= (sizer uses stake_pct).
- Common indicators: bt.indicators.EMA/SMA/RSI/MACD/BollingerBands/ATR/CrossOver/Stochastic.
- Close BOTH fences. Output the full class and the full json object.
"""

_REPAIR_CODE_PROMPT = """\
Your previous answer was incomplete or invalid for a backtrader strategy generator.

Error:
{error}

Previous output (truncated):
{prev}

User strategy:
{prompt}

{position_sizing}

Rewrite from scratch. Output EXACTLY:
1) one complete ```python block with a full bt.Strategy subclass using params = (...)
2) one complete ```json block with default param values (include stop_loss/take_profit as null and stake_pct={stake_frac})

No prose. Close both fences. Finish the entire class including def next(self). Use self.buy() without size=.
"""


def _generate_strategy_from_llm(strategy_prompt: str, position_size_pct: float = 100.0) -> tuple:
    """
    Call the LLM for strategy code + config, with one repair retry on failure.
    Returns (code, config).
    """
    pct = max(1.0, min(100.0, float(position_size_pct)))
    stake_frac = round(pct / 100.0, 4)
    sizing = _position_sizing_instructions(pct)

    if _llm_provider == "ollama":
        prompt = _CODE_GEN_PROMPT_OLLAMA.format(
            prompt=strategy_prompt,
            position_sizing=sizing,
            stake_frac=stake_frac,
        )
    else:
        prompt = _CODE_GEN_PROMPT.format(
            prompt=strategy_prompt,
            position_sizing=sizing,
        )

    def _parse(resp: str):
        code, config = _extract_code_and_config(resp)
        code = _normalize_python_source(code)
        code = _inject_stake_pct_param(code, pct)
        _validate_python_syntax(code)
        config = _ensure_stake_pct_in_config(config, pct)
        return code, config

    response_text = _call_llm(prompt)
    try:
        return _parse(response_text)
    except Exception as first_err:
        print(f"  First code-gen parse failed ({first_err}); retrying with repair prompt …")
        repair = _REPAIR_CODE_PROMPT.format(
            error=str(first_err)[:400],
            prev=response_text[:1500],
            prompt=strategy_prompt,
            position_sizing=sizing,
            stake_frac=stake_frac,
        )
        response_text = _call_llm(repair)
        return _parse(response_text)


def generate_strategy_code(state: GraphState) -> GraphState:
    print("--- Node: generate_strategy_code ---")
    if _llm_client is None:
        return {**state, "error": _llm_not_ready_message()}

    try:
        pct = _position_size_pct(state, 100.0)
        print(f"  Position size: {pct:g}% of available cash per trade")
        generated_code, default_config = _generate_strategy_from_llm(
            state["strategy_prompt"],
            position_size_pct=pct,
        )
        print(f"  Generated strategy ({len(generated_code)} chars), config={default_config}")
        return {
            **state,
            "generated_code":           generated_code,
            "current_config":           default_config,
            "current_iteration_number": 1,
            "all_iteration_results":    [],
            "best_config_so_far":       {},
            "error":                    None,
        }
    except Exception as e:
        print(f"ERROR in generate_strategy_code: {e}")
        return {**state, "error": f"Failed to generate/parse LLM response: {e}"}


def run_backtest(state: GraphState) -> GraphState:
    iteration = state["current_iteration_number"]
    print(f"--- Node: run_backtest  (iteration {iteration}) ---")

    try:
        ticker_symbol = get_ticker_from_prompt(state["strategy_prompt"])
        print(f"  Fetching data for {ticker_symbol} via yfinance …")

        start = state.get("start_date") or "2010-01-01"
        end   = state.get("end_date")   or "2017-11-10"
        # yfinance `end` is exclusive — bump by 1 day so the user's end date is included
        end_exclusive = (
            dt_module.datetime.strptime(end, "%Y-%m-%d") + dt_module.timedelta(days=1)
        ).strftime("%Y-%m-%d")
        print(f"  Backtest period: {start} → {end}")

        raw = yf.download(
            ticker_symbol,
            start=start,
            end=end_exclusive,
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise ValueError(
                f"No data returned by yfinance for '{ticker_symbol}' "
                f"between {start} and {end}."
            )

        data_df = _flatten_yfinance_df(raw)
        data_df.index = pd.to_datetime(data_df.index)
        # Enforce inclusive bounds so performance is relative to the requested window
        data_df = data_df.loc[
            (data_df.index >= pd.Timestamp(start)) &
            (data_df.index <= pd.Timestamp(end))
        ]
        if data_df.empty:
            raise ValueError(
                f"No bars for '{ticker_symbol}' within {start} → {end}."
            )

        from_dt = dt_module.datetime.strptime(start, "%Y-%m-%d")
        to_dt   = dt_module.datetime.strptime(end,   "%Y-%m-%d")
        data_feed = bt.feeds.PandasData(
            dataname=data_df,
            open="open", high="high", low="low", close="close",
            volume="volume", openinterest=-1,
            fromdate=from_dt,
            todate=to_dt,
        )

        namespace = _make_exec_namespace()
        sanitized = _sanitize_code(state["generated_code"])
        try:
            exec(sanitized, namespace)
        except SyntaxError as se:
            raise ValueError(f"Syntax error in generated strategy code: {se}")

        StrategyClass = next(
            (
                obj for obj in namespace.values()
                if isinstance(obj, type)
                and issubclass(obj, bt.Strategy)
                and obj is not bt.Strategy
            ),
            None,
        )
        if StrategyClass is None:
            raise ImportError(
                "No bt.Strategy subclass found in generated code.\n"
                f"Code:\n{state['generated_code'][:400]}"
            )

        config = state["current_config"]
        # Drop stake_pct from strategy kwargs if the class has no such param —
        # still applied via PercentSizer. Prefer passing it when declared.
        pct = _position_size_pct(state, 100.0)
        config = _ensure_stake_pct_in_config(config, pct)
        strat_kwargs = dict(config)
        param_names = _strategy_param_names(StrategyClass)
        if param_names and "stake_pct" not in param_names:
            strat_kwargs.pop("stake_pct", None)

        cerebro = bt.Cerebro()
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(100_000.0)
        cerebro.broker.setcommission(commission=0.001)
        # Position size: % of available cash on each buy() without explicit size=
        cerebro.addsizer(bt.sizers.PercentSizer, percents=pct)
        print(f"  PercentSizer: {pct:g}% of cash per trade")
        cerebro.addstrategy(StrategyClass, **strat_kwargs)
        cerebro.addanalyzer(Expectancy,              _name="expectancy")
        cerebro.addanalyzer(bt.analyzers.DrawDown,   _name="drawdown")
        cerebro.addanalyzer(
            bt.analyzers.TimeReturn,
            _name="cagr",
            timeframe=bt.TimeFrame.Years,
        )
        cerebro.addanalyzer(PortfolioValueAnalyzer,  _name="portfolio")
        cerebro.addanalyzer(TradeLogAnalyzer,        _name="tradelog")

        print(f"  Running cerebro with config: {config}")
        results = cerebro.run()
        metrics = get_metrics(
            cerebro, results,
            period_start=start,
            period_end=end,
        )
        print(f"  Metrics: {metrics}")

        all_results = state.get("all_iteration_results", []) + [
            {"iteration": iteration, "config": config, "metrics": metrics}
        ]
        best = state.get("best_config_so_far", {})
        if not best or _config_score(metrics) > _config_score(best.get("metrics", {})):
            best = {"config": config, "metrics": metrics}

        return {
            **state,
            "all_iteration_results": all_results,
            "best_config_so_far":    best,
            "error":                 None,
        }

    except Exception as e:
        print(f"ERROR in run_backtest: {e}")
        return {**state, "error": str(e)}


_OPTIMIZE_PROMPT = """
You are a quantitative analyst AI optimising a `backtrader` strategy.

Original strategy prompt: "{prompt}"

Strategy code:
```python
{code}
```

Iteration {prev_iter} results:
  Configuration: {prev_config}
  Metrics: {prev_metrics}

Analyse the results and propose new VALUES that improve CAGR while keeping Max Drawdown low.

CRITICAL RULES:
- You MUST use EXACTLY these parameter names (no others): {valid_keys}
- Do NOT rename, add, or remove any keys.
- Output ONLY a single ```json block with the updated values.
- No explanation, no other text.
- `stop_loss` and `take_profit` accept either `null` (disabled) or a positive float (e.g. 0.05 = 5%).
  Use the metrics to decide: high max_drawdown → try enabling stop_loss; low win_rate or expectancy → try
  enabling take_profit.  You may freely switch between null and a float for these two keys each iteration.
"""

def optimize_strategy(state: GraphState) -> GraphState:
    iteration = state["current_iteration_number"]
    print(f"--- Node: optimize_strategy  (was iteration {iteration}) ---")
    if _llm_client is None:
        return {**state, "error": _llm_not_ready_message()}

    prev       = state["all_iteration_results"][-1]
    valid_keys = list(prev["config"].keys())

    prompt = _OPTIMIZE_PROMPT.format(
        prompt=state["strategy_prompt"],
        code=state["generated_code"],
        prev_iter=prev["iteration"],
        prev_config=json.dumps(prev["config"],  indent=2),
        prev_metrics=json.dumps(prev["metrics"], indent=2),
        valid_keys=json.dumps(valid_keys),
    )

    try:
        response_text = _call_llm(prompt)
        json_match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
        if not json_match:
            raise ValueError(
                "Optimisation response missing ```json block.\n"
                f"Response: {response_text[:400]}"
            )
        raw_config = json.loads(json_match.group(1).strip())
        new_config = {k: raw_config.get(k, prev["config"][k]) for k in valid_keys}
        # Keep user position sizing fixed across optimization iterations
        if "stake_pct" in prev["config"]:
            new_config["stake_pct"] = prev["config"]["stake_pct"]
        else:
            new_config = _ensure_stake_pct_in_config(
                new_config, _position_size_pct(state, 100.0)
            )
        print(f"  Optimised config (sanitised): {new_config}")
        return {
            **state,
            "current_config":           new_config,
            "current_iteration_number": iteration + 1,
        }
    except Exception as e:
        print(f"ERROR in optimize_strategy: {e}")
        return {**state, "error": f"Failed to generate/parse new config: {e}"}


def should_continue_after_generation(state: GraphState) -> str:
    if state.get("error"):
        print(f"Stopping: error after code generation — {state['error']}")
        return END
    return "run_backtest"


def should_continue_after_run(state: GraphState) -> str:
    if state.get("error"):
        print(f"Stopping: error during backtest — {state['error']}")
        return END
    completed = state["current_iteration_number"]
    if completed >= 3:
        print("All 3 iterations complete.")
        return END
    return "optimize_strategy"


workflow = StateGraph(GraphState)
workflow.add_node("generate_strategy_code", generate_strategy_code)
workflow.add_node("run_backtest",           run_backtest)
workflow.add_node("optimize_strategy",      optimize_strategy)
workflow.set_entry_point("generate_strategy_code")
workflow.add_conditional_edges(
    "generate_strategy_code", should_continue_after_generation,
    {"run_backtest": "run_backtest", END: END},
)
workflow.add_conditional_edges(
    "run_backtest", should_continue_after_run,
    {"optimize_strategy": "optimize_strategy", END: END},
)
workflow.add_edge("optimize_strategy", "run_backtest")
app = workflow.compile()


# ===========================================================================
# WORKFLOW 2 — SCREENED MULTI-TICKER  (Kaggle dataset)
# ===========================================================================

# Use <<SCREENING_PROMPT>> as placeholder so the template can freely contain
# curly braces (Python code examples) without needing .format()-escaping.
_SCREENING_CODE_GEN_PROMPT = """\
You are a quantitative analyst. Generate optimised Python code to screen stocks from a local CSV dataset.

PERFORMANCE REQUIREMENT: Must complete in under 10 minutes for ~7000 stock files.
The ONLY way to achieve this is to read files in parallel with ThreadPoolExecutor.

The following names are already defined in the script — do NOT import or redefine them:
  pd                 : pandas
  os                 : os module
  ThreadPoolExecutor : from concurrent.futures
  DATASET_PATH       : str — directory containing *.us.txt stock files
  OUTPUT_PATH        : str — file path to write the output CSV
  START_DATE         : pd.Timestamp or None — inclusive lower bound
  END_DATE           : pd.Timestamp or None — inclusive upper bound

Each stock file (e.g. 'aapl.us.txt') is a CSV with header:
  Date,Open,High,Low,Close,Volume,OpenInt

DATE RULES (critical — avoid TypeError):
  - START_DATE / END_DATE are pandas Timestamps or None (injected by the runtime).
  - Always do: df['Date'] = pd.to_datetime(df['Date'], errors='coerce') before filtering.
  - Compare only Timestamp-to-Timestamp, e.g. df[df['Date'] >= START_DATE].
  - Never compare a Timestamp cell to a raw string in Python (iterrows). Convert first:
      d = pd.Timestamp(row['Date']);  d >= START_DATE
  - When writing output, use d.strftime('%Y-%m-%d') for the date column.

Screening logic to implement:
  <<SCREENING_PROMPT>>

MANDATORY CODE STRUCTURE — follow this pattern exactly:

  STEP 1: Define a single-file loader function _load_one(args) where args=(path, ticker).
    - Read only the columns you need with usecols=[...]. Do NOT pass parse_dates.
    - After reading: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    - Immediately filter rows to the needed date window:
        if START_DATE is not None: df = df[df['Date'] >= START_DATE - pd.Timedelta(days=90)]
        if END_DATE is not None:   df = df[df['Date'] <= END_DATE]
    - Add a 'ticker' column, return the filtered df, or return None on any error.

  STEP 2: Parallel-load ALL files with ThreadPoolExecutor(max_workers=16).
    Collect non-None results, then pd.concat into one combined DataFrame.
    NEVER use filter(None, ...) on DataFrames — bool(DataFrame) raises ValueError.
    Correct pattern:
      frames = [df for df in executor.map(_load_one, jobs) if df is not None]

  STEP 3: Compute the screening metric using ONLY vectorised pandas operations.
    CRITICAL — keep 'Date' and 'ticker' as plain columns; do NOT use set_index(). The combined
    DataFrame has many rows per date (one per ticker), so a date index would have duplicates and
    cause "cannot reindex on an axis with duplicate labels" errors.
    For per-ticker rolling or pct_change columns always use .transform():
      combined_df['metric'] = combined_df.groupby('ticker')['Close'].transform(
          lambda x: x.pct_change(periods=30)   # or .rolling(30).mean(), etc.
      )
    Never use .rolling(...).mean().reset_index(...) — that pattern fails with duplicate dates.
    For cross-sectional ranking per date use:
      combined_df['rank_pct'] = combined_df.groupby('Date')['metric'].rank(pct=True, ascending=False)

  STEP 4: Prefer vectorised filters (no iterrows):
    if START_DATE is not None: selected = selected[selected['Date'] >= START_DATE]
    if END_DATE is not None:   selected = selected[selected['Date'] <= END_DATE]
    out = selected[['Date', 'ticker']].copy()
    out['date'] = out['Date'].dt.strftime('%Y-%m-%d')
    out = out[['date', 'ticker']]

  STEP 5: Write the output:
    out.to_csv(OUTPUT_PATH, index=False)

Output ONLY a single ```python code block implementing all five steps. No explanation, no other text.
"""

# Compact, example-driven prompt for local models (avoids long prose that
# leads to broken indentation / truncated scripts).
_SCREENING_CODE_GEN_PROMPT_OLLAMA = """\
Write a COMPLETE, valid Python script body that screens stocks.

Screening criteria:
<<SCREENING_PROMPT>>

These names already exist — do NOT import or redefine them:
  pd, os, ThreadPoolExecutor, DATASET_PATH, OUTPUT_PATH, START_DATE, END_DATE

Each file in DATASET_PATH is named like 'aapl.us.txt' with columns:
  Date,Open,High,Low,Close,Volume,OpenInt

Copy this skeleton and adapt ONLY the metric / selection to the criteria.
Use 4-space indentation. No tabs. No comments. Close the ``` fence.

```python
def _load_one(args):
    path, ticker = args
    try:
        df = pd.read_csv(path, usecols=['Date', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if START_DATE is not None:
            df = df[df['Date'] >= START_DATE - pd.Timedelta(days=90)]
        if END_DATE is not None:
            df = df[df['Date'] <= END_DATE]
        if df.empty:
            return None
        df = df.copy()
        df['ticker'] = ticker
        return df
    except Exception:
        return None

files = []
for fn in os.listdir(DATASET_PATH):
    if fn.endswith('.us.txt'):
        ticker = fn[:-7].upper()
        files.append((os.path.join(DATASET_PATH, fn), ticker))

frames = []
with ThreadPoolExecutor(max_workers=16) as ex:
    for df in ex.map(_load_one, files):
        if df is not None:
            frames.append(df)

if not frames:
    pd.DataFrame(columns=['date', 'ticker']).to_csv(OUTPUT_PATH, index=False)
else:
    combined = pd.concat(frames, ignore_index=True)
    combined['Date'] = pd.to_datetime(combined['Date'], errors='coerce')
    combined['metric'] = combined.groupby('ticker')['Close'].transform(
        lambda x: x.pct_change(periods=21)
    )
    combined['rank_pct'] = combined.groupby('Date')['metric'].rank(
        pct=True, ascending=False
    )
    selected = combined[combined['rank_pct'] <= 0.01].copy()
    if START_DATE is not None:
        selected = selected[selected['Date'] >= START_DATE]
    if END_DATE is not None:
        selected = selected[selected['Date'] <= END_DATE]
    out = selected[['Date', 'ticker']].copy()
    out['date'] = out['Date'].dt.strftime('%Y-%m-%d')
    out = out[['date', 'ticker']]
    out['ticker'] = out['ticker'].astype(str).str.upper()
    out.to_csv(OUTPUT_PATH, index=False)
```

Rules:
- Output ONLY one ```python block with the full script body (no prose).
- START_DATE/END_DATE are Timestamps or None — compare with datetime Date only.
- Keep Date and ticker as columns — never set_index on Date.
- Use groupby(...).transform(...) for rolling/pct_change.
- Must write CSV to OUTPUT_PATH with columns date,ticker.
- NEVER use filter(None, dfs) — use [df for df in dfs if df is not None].
- Valid Python syntax is mandatory — balanced brackets, consistent 4-space indent.
"""

_REPAIR_SCREENING_PROMPT = """\
The screening Python script you wrote is invalid.

Error:
{error}

Previous code (truncate):
```python
{prev}
```

Screening criteria:
{criteria}

Rewrite the FULL script body from scratch.
Predefined names (do not import): pd, os, ThreadPoolExecutor, DATASET_PATH, OUTPUT_PATH, START_DATE, END_DATE.
Output ONLY one complete ```python block. 4-space indent, no tabs, no comments.
Must parallel-load *.us.txt from DATASET_PATH, screen by criteria, write date,ticker CSV to OUTPUT_PATH.
NEVER use filter(None, ...) on DataFrames — use [x for x in items if x is not None].
"""


def _normalize_python_source(code: str) -> str:
    """Fix common LLM source issues: CRLF, tabs, whole-block over-indent."""
    import textwrap
    import ast as _ast

    code = (code or "").replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    # Drop accidental markdown fences left inside the block
    code = re.sub(r"^```(?:python)?\s*\n?", "", code.strip(), flags=re.I)
    code = re.sub(r"\n?```\s*$", "", code.strip())
    code = textwrap.dedent(code)

    lines = code.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""

    # If every non-empty line shares a common leading indent, strip it
    indents = []
    for ln in lines:
        if not ln.strip():
            continue
        indents.append(len(ln) - len(ln.lstrip(" ")))
    if indents:
        common = min(indents)
        if common > 0:
            lines = [ln[common:] if ln.strip() else ln for ln in lines]

    code = "\n".join(lines) + "\n"

    # Last resort: if still invalid, try compile as-is for caller to handle
    try:
        _ast.parse(code)
    except SyntaxError:
        pass
    return code


def _validate_python_syntax(code: str) -> None:
    import ast as _ast
    try:
        _ast.parse(code)
    except SyntaxError as e:
        raise ValueError(
            f"Generated Python has SyntaxError at line {e.lineno}: {e.msg}\n"
            f"  {e.text or ''}"
        ) from e


def _sanitize_screening_code(code: str) -> str:
    """
    Fix common LLM screening anti-patterns that pass syntax checks but crash
    at runtime (especially pandas DataFrame truthiness).

    Main bug: ``list(filter(None, dfs))`` calls ``bool(DataFrame)`` → ValueError.
    """
    def _rewrite_list_filter_none(src: str) -> str:
        """Rewrite list(filter(None, EXPR)) with balanced-paren-aware scan."""
        pattern = re.compile(r"list\s*\(\s*filter\s*\(\s*None\s*,\s*")
        out = []
        i = 0
        while True:
            m = pattern.search(src, i)
            if not m:
                out.append(src[i:])
                break
            out.append(src[i:m.start()])
            expr_start = m.end()
            # Walk EXPR until the ')' that closes filter(
            depth = 1
            j = expr_start
            while j < len(src) and depth > 0:
                ch = src[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                j += 1
            if depth != 0:
                # malformed — keep original slice
                out.append(src[m.start():expr_start])
                i = expr_start
                continue
            expr = src[expr_start:j - 1]
            # consume optional whitespace + closing ')' of list(
            k = j
            while k < len(src) and src[k].isspace():
                k += 1
            if k < len(src) and src[k] == ")":
                k += 1
                out.append(f"[_x for _x in ({expr}) if _x is not None]")
                i = k
            else:
                out.append(src[m.start():j])
                i = j
        return "".join(out)

    return _rewrite_list_filter_none(code)


def _extract_python_only(response_text: str) -> str:
    """Extract a python fenced block (or bare code) and normalize it."""
    code = _extract_fenced_block(response_text, "python")
    if not code:
        # bare script without fences
        stripped = response_text.strip()
        if "def " in stripped or "for " in stripped or "class " in stripped:
            code = stripped
    if not code:
        raise ValueError(
            "LLM response missing ```python block.\n"
            f"Response:\n{response_text[:500]}"
        )
    code = _normalize_python_source(code)
    code = _sanitize_screening_code(code)
    _validate_python_syntax(code)
    return code


def _generate_screening_from_llm(screening_prompt: str) -> str:
    """Generate screening script with syntax check + one repair retry."""
    if _llm_provider == "ollama":
        prompt = _SCREENING_CODE_GEN_PROMPT_OLLAMA.replace(
            "<<SCREENING_PROMPT>>", screening_prompt
        )
    else:
        prompt = _SCREENING_CODE_GEN_PROMPT.replace(
            "<<SCREENING_PROMPT>>", screening_prompt
        )

    response_text = _call_llm(prompt)
    try:
        return _extract_python_only(response_text)
    except Exception as first_err:
        print(f"  Screening code parse/syntax failed ({first_err}); repairing …")
        repair = _REPAIR_SCREENING_PROMPT.format(
            error=str(first_err)[:500],
            prev=(response_text or "")[:2000],
            criteria=screening_prompt,
        )
        response_text = _call_llm(repair)
        return _extract_python_only(response_text)


def generate_screening_code(state: ScreenGraphState) -> ScreenGraphState:
    print("--- Node: generate_screening_code ---")
    if _llm_client is None:
        return {**state, "error": _llm_not_ready_message()}

    try:
        code = _generate_screening_from_llm(state["screening_prompt"])
        print(f"  Screening code OK ({len(code)} chars)")
        return {
            **state,
            "screening_code": code,
            "error":          None,
        }
    except Exception as e:
        print(f"ERROR in generate_screening_code: {e}")
        return {**state, "error": f"Failed to generate screening code: {e}"}


_SCREENING_TIMEOUT_SECS = 900  # 15 minutes


def run_screening(state: ScreenGraphState) -> ScreenGraphState:
    """Execute the generated screening code as a subprocess with a 15-minute timeout."""
    print("--- Node: run_screening ---")
    import subprocess
    import sys
    import textwrap

    output_path = tempfile.mktemp(suffix=".csv")
    code_file   = tempfile.mktemp(suffix=".py")

    screening_body = _normalize_python_source(state.get("screening_code") or "")
    screening_body = _sanitize_screening_code(screening_body)
    try:
        _validate_python_syntax(screening_body)
    except Exception as e:
        return {**state, "error": f"Screening code failed syntax check: {e}"}

    # Build a standalone Python script that embeds all variables and runs the
    # generated code.  Using repr() for paths/strings is injection-safe.
    #
    # Date safety: LLM code often mixes datetime64 Date columns with string
    # START_DATE/END_DATE (or parse_dates=True then compares to str), which
    # raises TypeError on scalar comparisons (e.g. in iterrows).  Inject
    # Timestamp bounds + normalize Date on every read_csv so comparisons work.
    _start_raw = (state.get("start_date") or "").strip()
    _end_raw   = (state.get("end_date")   or "").strip()
    header = textwrap.dedent(f"""\
        import pandas as pd
        import os
        from concurrent.futures import ThreadPoolExecutor

        DATASET_PATH = {repr(KAGGLE_STOCKS_PATH)}
        OUTPUT_PATH  = {repr(output_path)}
        _START_RAW   = {repr(_start_raw)}
        _END_RAW     = {repr(_end_raw)}
        # Prefer Timestamps so comparisons with datetime Date columns work.
        # Also keep string forms for scripts that do strftime-style filters.
        START_DATE   = pd.Timestamp(_START_RAW) if _START_RAW else None
        END_DATE     = pd.Timestamp(_END_RAW)   if _END_RAW   else None
        START_DATE_STR = _START_RAW
        END_DATE_STR   = _END_RAW

        def _normalize_date_col(df):
            if not isinstance(df, pd.DataFrame) or 'Date' not in df.columns:
                return df
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            return df

        _orig_read_csv = pd.read_csv
        def _read_csv_normalized(*args, **kwargs):
            kwargs = dict(kwargs)
            # Let us control Date dtype; ignore LLM parse_dates choices
            kwargs.pop('parse_dates', None)
            df = _orig_read_csv(*args, **kwargs)
            return _normalize_date_col(df)
        pd.read_csv = _read_csv_normalized

    """)

    try:
        with open(code_file, "w") as fh:
            fh.write(header + screening_body + "\n")

        print(
            f"  Running screening subprocess "
            f"(timeout: {_SCREENING_TIMEOUT_SECS}s = 15 min) …"
        )
        proc = subprocess.run(
            [sys.executable, code_file],
            timeout=_SCREENING_TIMEOUT_SECS,
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            stderr_tail = proc.stderr.strip()[-1000:] if proc.stderr.strip() else "no stderr"
            # Keep a debug copy of the failed script for local inspection
            debug_path = os.path.join(tempfile.gettempdir(), "code_bulls_last_screening.py")
            try:
                with open(debug_path, "w") as dfh:
                    dfh.write(header + screening_body + "\n")
                print(f"  Saved failed screening script → {debug_path}")
            except Exception:
                pass
            raise RuntimeError(
                f"Screening script exited with code {proc.returncode}:\n{stderr_tail}"
            )

        if proc.stderr.strip():
            # Non-fatal warnings from the script
            print(f"  Screening script warnings:\n{proc.stderr.strip()[-400:]}")

    except subprocess.TimeoutExpired:
        return {
            **state,
            "error": (
                f"Screening timed out after {_SCREENING_TIMEOUT_SECS // 60} minutes. "
                "Narrow the date range (start_date / end_date) or simplify the "
                "screening criteria to speed it up."
            ),
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {**state, "error": f"Screening failed: {e}"}
    finally:
        try:
            os.unlink(code_file)
        except Exception:
            pass

    # ── Parse the CSV written by the screening script ──────────────────────
    if not os.path.exists(output_path):
        return {**state, "error": "Screening script did not create the OUTPUT_PATH file."}

    try:
        screening_df = pd.read_csv(output_path, dtype=str)
    except Exception as e:
        return {**state, "error": f"Failed to read screening output CSV: {e}"}

    if (
        screening_df.empty
        or "date"   not in screening_df.columns
        or "ticker" not in screening_df.columns
    ):
        return {
            **state,
            "error": "Screening CSV is empty or missing 'date'/'ticker' columns.",
        }

    screening_dict: dict = {}
    for _, row in screening_df.iterrows():
        date_str = str(row["date"]).strip()
        ticker   = str(row["ticker"]).strip().upper()
        screening_dict.setdefault(date_str, []).append(ticker)

    total_pairs = sum(len(v) for v in screening_dict.values())
    print(
        f"  Screening complete: {len(screening_dict)} dates, "
        f"{total_pairs} ticker-day pairs."
    )

    if not screening_dict:
        return {
            **state,
            "error": "Screening returned no results — no tickers passed the criteria.",
        }

    return {
        **state,
        "screening_csv_path": output_path,
        "screening_dict":     screening_dict,
        "error":              None,
    }


_MULTI_STRATEGY_PROMPT = (
"""You are an expert in the `backtrader` Python library.
Write a multi-asset `backtrader.Strategy` class for the trading strategy below.

IMPORTANT CONTEXT:
- Multiple data feeds are loaded, one per stock ticker.  Each has `data._name` set to its symbol.
- `self.params.screening` is a dict: {{"YYYY-MM-DD": ["AAPL", "TSLA", ...]}}
  On each bar, ONLY open NEW positions for tickers that appear in today's list.
  Do NOT force-close a position just because a ticker is no longer in the screening
  list — existing positions stay open and are exited only by the strategy's own
  sell/stop conditions.
- Get today's date: `today = self.datetime.date(0).strftime('%Y-%m-%d')`
- Tickers in screening are UPPERCASE; d._name is UPPERCASE — match exactly.

CRITICAL strategy/screener compatibility:
- If the universe is momentum/breakout/top-gainers, BUY when `d._name in today_screened`
  (do NOT require RSI oversold for entry — those names are rarely oversold).
- Use RSI/stops mainly for exits unless the user explicitly wants mean-reversion entries.


CRITICAL — indicator initialisation MUST use this exact try/except pattern per feed:
    def __init__(self):
        self.inds = {{}}
        for d in self.datas:
            try:
                self.inds[d._name] = {{
                    'ind': bt.indicators.RSI(d, period=self.params.rsi_period),
                }}
            except Exception:
                pass

CRITICAL — in `next()`, always guard before using an indicator:
    def next(self):
        today_screened = self.params.screening.get(
            self.datetime.date(0).strftime('%Y-%m-%d'), []
        )
        for d in self.datas:
            if d._name not in self.inds:
                continue
            if len(d) <= self.params.warmup_period:
                continue

Requirements:
1. `screening` must be a param with default `{{}}`.  All other params numeric — NO hardcoded values.
2. Assume `bt`, `np` (numpy), and `datetime` are already in the global scope. Do NOT import anything.
3. Output EXACTLY one ```python block and one ```json block (default numeric params, WITHOUT `screening`).
4. Do NOT define `notify_trade` or `notify_order` methods — they are not needed and frequently cause AttributeError.
5. Write NO comments in the code — no inline comments, no docstrings, nothing.
6. Always include `stop_loss = None` and `take_profit = None` in `params`.
   Inside the `for d in self.datas:` loop in `next()`, after your normal entry/exit logic, add:
     pos = self.getposition(d)
     if pos.size > 0:
         if self.params.stop_loss is not None and d.close[0] <= pos.price * (1 - self.params.stop_loss):
             self.close(d)
         elif self.params.take_profit is not None and d.close[0] >= pos.price * (1 + self.params.take_profit):
             self.close(d)
   Set both to `null` in the ```json config block unless the user explicitly requested them.
7. POSITION SIZING — follow the POSITION SIZING block; include stake_pct in params and json.
   Use self.buy(data=d) with no size= (runtime PercentSizer enforces stake per new position).

{position_sizing}

--- BACKTRADER API REFERENCE ---
{api_docs}
--- END REFERENCE ---

User Strategy: "{{prompt}}"
"""
.replace("{api_docs}", _BT_API_DOCS.replace("{", "{{").replace("}", "}}"))
.replace("{{prompt}}", "{prompt}")
)

_MULTI_STRATEGY_PROMPT_OLLAMA = """\
Write a multi-asset backtrader.Strategy for this idea:

{prompt}

{position_sizing}

Context:
- Many data feeds; each has d._name = ticker symbol (UPPERCASE, e.g. AAPL).
- self.p.screening is dict {{"YYYY-MM-DD": ["AAPL", ...]}} with UPPERCASE tickers.
- On each bar, ONLY open NEW positions for tickers in today's screening list.
- Do NOT force-close just because a ticker left the list unless the user asked.
- today = self.datetime.date(0).strftime('%Y-%m-%d')
- Compare with: d._name in today_screened  (both uppercase).

CRITICAL compatibility with stock SCREENERS:
- Momentum screens (top gainers / breakouts) almost NEVER have RSI < 30/35.
  For those screens, BUY when the ticker is in today's list (optionally confirm
  with a short trend filter), and use RSI / trailing rules only for EXITS.
- Only use RSI-oversold as an ENTRY filter if the user explicitly asked for
  mean-reversion entries (not for "top movers" screens).

Output EXACTLY these two closed fences (adapt to the user strategy):

```python
class MultiScreenStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_exit', 70),
        ('warmup_period', 30),
        ('stake_pct', {stake_frac}),
        ('stop_loss', None),
        ('take_profit', None),
        ('screening', {{}}),
    )
    def __init__(self):
        self.inds = {{}}
        for d in self.datas:
            try:
                self.inds[d._name] = {{
                    'rsi': bt.indicators.RSI(d, period=self.p.rsi_period),
                }}
            except Exception:
                pass
    def next(self):
        today = self.datetime.date(0).strftime('%Y-%m-%d')
        today_screened = self.p.screening.get(today, [])
        for d in self.datas:
            if d._name not in self.inds:
                continue
            if len(d) <= self.p.warmup_period:
                continue
            pos = self.getposition(d)
            rsi = self.inds[d._name]['rsi'][0]
            if pos.size == 0:
                if d._name in today_screened:
                    self.buy(data=d)
            else:
                if rsi > self.p.rsi_exit:
                    self.close(data=d)
            pos = self.getposition(d)
            if pos.size > 0:
                if self.p.stop_loss is not None and d.close[0] <= pos.price * (1 - self.p.stop_loss):
                    self.close(data=d)
                elif self.p.take_profit is not None and d.close[0] >= pos.price * (1 + self.p.take_profit):
                    self.close(data=d)
```
```json
{{"rsi_period": 14, "rsi_exit": 70, "warmup_period": 30, "stake_pct": {stake_frac}, "stop_loss": null, "take_profit": null}}
```

Rules:
- params include screening={{}} and stake_pct={stake_frac}. JSON must NOT include screening.
- self.buy(data=d) with NO size=. No imports, no notify_*, no comments. Close both fences.
- Init indicators with try/except per feed; guard with if d._name not in self.inds.
- Prefer buy-when-screened for universe/momentum strategies.
"""

_REPAIR_MULTI_PROMPT = """\
Your multi-asset strategy response was invalid.

Error:
{error}

Previous output (truncate):
{prev}

User strategy:
{prompt}

{position_sizing}

Rewrite from scratch. Output EXACTLY:
1) one complete ```python bt.Strategy class (params including screening={{}}, stake_pct={stake_frac}, try/except inds, next with screening gate)
2) one complete ```json defaults WITHOUT screening (include stake_pct={stake_frac})

No prose. Close both fences. Valid Python only. Use self.buy(data=d) without size=.
"""


def _generate_multi_strategy_from_llm(strategy_prompt: str, position_size_pct: float = 10.0) -> tuple:
    pct = max(1.0, min(100.0, float(position_size_pct)))
    stake_frac = round(pct / 100.0, 4)
    sizing = _position_sizing_instructions(pct)

    if _llm_provider == "ollama":
        prompt = _MULTI_STRATEGY_PROMPT_OLLAMA.format(
            prompt=strategy_prompt,
            position_sizing=sizing,
            stake_frac=stake_frac,
        )
    else:
        prompt = _MULTI_STRATEGY_PROMPT.format(
            prompt=strategy_prompt,
            position_sizing=sizing,
        )

    def _parse(resp: str):
        code, config = _extract_code_and_config(resp)
        code = _normalize_python_source(code)
        code = _inject_stake_pct_param(code, pct)
        _validate_python_syntax(code)
        config = _ensure_stake_pct_in_config(config, pct)
        return code, config

    response_text = _call_llm(prompt)
    try:
        return _parse(response_text)
    except Exception as first_err:
        print(f"  Multi-strategy parse failed ({first_err}); repairing …")
        repair = _REPAIR_MULTI_PROMPT.format(
            error=str(first_err)[:400],
            prev=(response_text or "")[:1500],
            prompt=strategy_prompt,
            position_sizing=sizing,
            stake_frac=stake_frac,
        )
        response_text = _call_llm(repair)
        return _parse(response_text)


def generate_multi_strategy_code(state: ScreenGraphState) -> ScreenGraphState:
    print("--- Node: generate_multi_strategy_code ---")
    if _llm_client is None:
        return {**state, "error": _llm_not_ready_message()}

    try:
        pct = _position_size_pct(state, 10.0)
        print(f"  Position size: {pct:g}% of available cash per new position")
        generated_code, default_config = _generate_multi_strategy_from_llm(
            state["strategy_prompt"],
            position_size_pct=pct,
        )
        # screening is injected at run time — drop if model put it in config
        default_config.pop("screening", None)
        print(f"  Multi strategy OK ({len(generated_code)} chars), config={default_config}")
        return {
            **state,
            "generated_code":           generated_code,
            "current_config":           default_config,
            "current_iteration_number": 1,
            "all_iteration_results":    [],
            "best_config_so_far":       {},
            "error":                    None,
        }
    except Exception as e:
        print(f"ERROR in generate_multi_strategy_code: {e}")
        return {**state, "error": f"Failed to generate multi-ticker strategy code: {e}"}


def _fast_date_parse(date_string: str) -> dt_module.datetime:
    """String-slicing date parser — much faster than strptime for YYYY-MM-DD."""
    return dt_module.datetime(
        int(date_string[0:4]),
        int(date_string[5:7]),
        int(date_string[8:10]),
    )


def _has_enough_data(filepath: str, min_bytes: int = 6_000) -> bool:
    """
    Fast file-size proxy for row count (≈100 bytes/row → 6 KB ≈ 60 rows).
    ZeroDivisionError in indicators is handled by the _safe_once_op patch,
    so we only need to filter completely empty / near-empty files.
    """
    return os.path.getsize(filepath) >= min_bytes


def _price_series_sane(
    filepath: str,
    from_dt,
    to_dt,
    max_median: float = 10_000.0,
    min_median: float = 0.05,
) -> bool:
    """
    Reject reverse-split / unit-corrupted series (e.g. DRYS closes in 1e8).
    Samples closes in [from_dt, to_dt]; keeps names with a sane median price.
    """
    try:
        closes = []
        with open(filepath, "r", errors="ignore") as fh:
            fh.readline()  # header
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) < 5:
                    continue
                ds = parts[0]
                try:
                    y, m, d = int(ds[0:4]), int(ds[5:7]), int(ds[8:10])
                except Exception:
                    continue
                if from_dt is not None and (y, m, d) < (from_dt.year, from_dt.month, from_dt.day):
                    continue
                if to_dt is not None and (y, m, d) > (to_dt.year, to_dt.month, to_dt.day):
                    continue
                try:
                    closes.append(float(parts[4]))
                except Exception:
                    continue
                if len(closes) >= 400:
                    break
        if len(closes) < 20:
            return False
        closes.sort()
        med = closes[len(closes) // 2]
        if med <= 0 or med > max_median or med < min_median:
            return False
        if closes[-1] > max_median * 50:
            return False
        return True
    except Exception:
        return False


def run_multi_backtest(state: ScreenGraphState) -> ScreenGraphState:
    iteration = state["current_iteration_number"]
    print(f"--- Node: run_multi_backtest  (iteration {iteration}) ---")
    try:
        screening_dict = state["screening_dict"]
        period_start = state.get("start_date") or "2010-01-01"
        period_end   = state.get("end_date")   or "2017-11-10"
        print(f"  Backtest period: {period_start} → {period_end}")

        # Load data from (period_start - warmup) so indicators are ready by
        # period_start; performance metrics are computed on period_start→end.
        period_from = dt_module.datetime.strptime(period_start, "%Y-%m-%d")
        from_dt = period_from - dt_module.timedelta(days=100)
        to_dt   = dt_module.datetime.strptime(period_end, "%Y-%m-%d")

        # Normalize screening keys/tickers to YYYY-MM-DD + UPPERCASE symbols
        norm_screening = {}
        for raw_date, tickers in (screening_dict or {}).items():
            dkey = str(raw_date).strip()[:10]
            cleaned = []
            for t in tickers or []:
                sym = str(t).strip().upper().replace(".US", "")
                if sym:
                    cleaned.append(sym)
            if cleaned:
                norm_screening[dkey] = sorted(set(cleaned))
        screening_dict = norm_screening
        state = {**state, "screening_dict": screening_dict}

        all_tickers = sorted({t for tickers in screening_dict.values() for t in tickers})
        if not all_tickers:
            raise ValueError("Screening dict is empty — no tickers to trade.")
        n_days = len(screening_dict)
        n_pairs = sum(len(v) for v in screening_dict.values())
        sample_dates = sorted(screening_dict.keys())
        print(
            f"  Screening universe: {len(all_tickers)} tickers, "
            f"{n_days} dates, {n_pairs} ticker-day pairs"
        )
        if sample_dates:
            print(
                f"  Screening date span: {sample_dates[0]} → {sample_dates[-1]} "
                f"(e.g. {sample_dates[n_days//2]} has "
                f"{len(screening_dict[sample_dates[n_days//2]])} names)"
            )

        cerebro = bt.Cerebro()
        # First feed clocks the engine — prefer a liquid long-history name if present
        preferred = [t for t in ("SPY", "AAPL", "MSFT", "GE", "IBM") if t in all_tickers]
        ordered_tickers = preferred + [t for t in all_tickers if t not in preferred]
        loaded = 0
        skipped_invalid = 0
        skipped_bad_price = 0

        for ticker in ordered_tickers:
            filepath = os.path.join(KAGGLE_STOCKS_PATH, f"{ticker.lower()}.us.txt")
            if not os.path.exists(filepath) or not _has_enough_data(filepath):
                skipped_invalid += 1
                continue
            # Drop reverse-split / corrupt series (e.g. DRYS prices in the millions)
            if not _price_series_sane(filepath, from_dt, to_dt):
                skipped_bad_price += 1
                continue
            try:
                feed = bt.feeds.GenericCSVData(
                    dataname=filepath,
                    name=ticker,
                    dtformat=_fast_date_parse,
                    date=0,
                    open=1,
                    high=2,
                    low=3,
                    close=4,
                    volume=5,
                    openinterest=6,
                    fromdate=from_dt,
                    todate=to_dt,
                    preload=True,
                )
                cerebro.adddata(feed, name=ticker)
                loaded += 1
            except Exception as ex:
                print(f"  WARNING: skipping {ticker}: {ex}")

        if loaded == 0:
            raise ValueError(
                "No ticker data could be loaded from the Kaggle dataset. "
                "Check that Stock Market Dataset/Stocks/ contains the screened tickers."
            )
        print(
            f"  Loaded {loaded}/{len(all_tickers)} tickers "
            f"({skipped_invalid} insufficient data, {skipped_bad_price} bad/split prices skipped)."
        )

        namespace = _make_exec_namespace()
        sanitized = _sanitize_code(state["generated_code"])
        try:
            exec(sanitized, namespace)
        except SyntaxError as se:
            raise ValueError(f"Syntax error in generated strategy code: {se}")

        StrategyClass = next(
            (
                obj for obj in namespace.values()
                if isinstance(obj, type)
                and issubclass(obj, bt.Strategy)
                and obj is not bt.Strategy
            ),
            None,
        )
        if StrategyClass is None:
            raise ImportError("No bt.Strategy subclass found in generated code.")

        config     = state["current_config"]
        pct = _position_size_pct(state, 10.0)
        config = _ensure_stake_pct_in_config(config, pct)
        run_config = {**config, "screening": screening_dict}
        param_names = _strategy_param_names(StrategyClass)
        if param_names and "stake_pct" not in param_names:
            run_config.pop("stake_pct", None)

        cerebro.broker.setcash(100_000.0)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.addsizer(bt.sizers.PercentSizer, percents=pct)
        print(f"  PercentSizer: {pct:g}% of cash per new position")
        cerebro.addstrategy(StrategyClass, **run_config)
        cerebro.addanalyzer(Expectancy,              _name="expectancy")
        cerebro.addanalyzer(bt.analyzers.DrawDown,   _name="drawdown")
        cerebro.addanalyzer(
            bt.analyzers.TimeReturn,
            _name="cagr",
            timeframe=bt.TimeFrame.Years,
        )
        cerebro.addanalyzer(PortfolioValueAnalyzer,  _name="portfolio")
        cerebro.addanalyzer(TradeLogAnalyzer,        _name="tradelog")

        print(f"  Running multi-ticker cerebro with config: {config}")
        results = cerebro.run(stdstats=False, tradehistory=True)
        metrics = get_metrics(
            cerebro, results,
            period_start=period_start,
            period_end=period_end,
        )
        print(f"  Metrics: {metrics}")

        all_results = state.get("all_iteration_results", []) + [
            {"iteration": iteration, "config": config, "metrics": metrics}
        ]
        best = state.get("best_config_so_far", {})
        if not best or _config_score(metrics) > _config_score(best.get("metrics", {})):
            best = {"config": config, "metrics": metrics}

        return {
            **state,
            "all_iteration_results": all_results,
            "best_config_so_far":    best,
            "error":                 None,
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"ERROR in run_multi_backtest: {e}")
        return {**state, "error": str(e)}


def optimize_multi_strategy(state: ScreenGraphState) -> ScreenGraphState:
    iteration = state["current_iteration_number"]
    print(f"--- Node: optimize_multi_strategy  (was iteration {iteration}) ---")
    if _llm_client is None:
        return {**state, "error": _llm_not_ready_message()}

    prev       = state["all_iteration_results"][-1]
    valid_keys = list(prev["config"].keys())

    prompt = _OPTIMIZE_PROMPT.format(
        prompt=state["strategy_prompt"],
        code=state["generated_code"],
        prev_iter=prev["iteration"],
        prev_config=json.dumps(prev["config"],  indent=2),
        prev_metrics=json.dumps(prev["metrics"], indent=2),
        valid_keys=json.dumps(valid_keys),
    )
    try:
        response_text = _call_llm(prompt)
        json_match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
        if not json_match:
            raise ValueError(
                "Optimisation response missing ```json block.\n"
                f"Response: {response_text[:400]}"
            )
        raw_config = json.loads(json_match.group(1).strip())
        new_config = {k: raw_config.get(k, prev["config"][k]) for k in valid_keys}
        if "stake_pct" in prev["config"]:
            new_config["stake_pct"] = prev["config"]["stake_pct"]
        else:
            new_config = _ensure_stake_pct_in_config(
                new_config, _position_size_pct(state, 10.0)
            )
        print(f"  Optimised config: {new_config}")
        return {
            **state,
            "current_config":           new_config,
            "current_iteration_number": iteration + 1,
        }
    except Exception as e:
        print(f"ERROR in optimize_multi_strategy: {e}")
        return {**state, "error": f"Failed to generate/parse new config: {e}"}


# --- Edge conditions ---

def _screen_after_screening_gen(state: ScreenGraphState) -> str:
    return END if state.get("error") else "run_screening"

def _screen_after_screening_run(state: ScreenGraphState) -> str:
    return END if state.get("error") else "generate_multi_strategy_code"

def _screen_after_multi_gen(state: ScreenGraphState) -> str:
    return END if state.get("error") else "run_multi_backtest"

def _screen_after_multi_run(state: ScreenGraphState) -> str:
    if state.get("error"):
        return END
    if state["current_iteration_number"] >= 3:
        print("All 3 multi-ticker iterations complete.")
        return END
    return "optimize_multi_strategy"


# --- Build multi_app ---

multi_workflow = StateGraph(ScreenGraphState)
multi_workflow.add_node("generate_screening_code",      generate_screening_code)
multi_workflow.add_node("run_screening",                run_screening)
multi_workflow.add_node("generate_multi_strategy_code", generate_multi_strategy_code)
multi_workflow.add_node("run_multi_backtest",           run_multi_backtest)
multi_workflow.add_node("optimize_multi_strategy",      optimize_multi_strategy)

multi_workflow.set_entry_point("generate_screening_code")
multi_workflow.add_conditional_edges(
    "generate_screening_code", _screen_after_screening_gen,
    {"run_screening": "run_screening", END: END},
)
multi_workflow.add_conditional_edges(
    "run_screening", _screen_after_screening_run,
    {"generate_multi_strategy_code": "generate_multi_strategy_code", END: END},
)
multi_workflow.add_conditional_edges(
    "generate_multi_strategy_code", _screen_after_multi_gen,
    {"run_multi_backtest": "run_multi_backtest", END: END},
)
multi_workflow.add_conditional_edges(
    "run_multi_backtest", _screen_after_multi_run,
    {"optimize_multi_strategy": "optimize_multi_strategy", END: END},
)
multi_workflow.add_edge("optimize_multi_strategy", "run_multi_backtest")

multi_app = multi_workflow.compile()
