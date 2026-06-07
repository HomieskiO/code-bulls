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
from evaluator import Expectancy, get_metrics, CAGRAnalyzer, PortfolioValueAnalyzer
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
#   LLM_PROVIDER = gemini | openai | anthropic   (default: gemini)
#   LLM_MODEL    = <model name>                  (optional – uses provider default)
#   GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS = {
    "gemini":    "gemini-2.5-flash-lite",
    "openai":    "gpt-4o",
    "anthropic": "claude-opus-4-7",
}

_llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
_model_name   = os.getenv("LLM_MODEL", "").strip() or _PROVIDER_DEFAULTS.get(_llm_provider, "")
_llm_client   = None

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
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER {_llm_provider!r}. "
            "Supported values: gemini, openai, anthropic."
        )
    print(f"LLM client initialised (provider: {_llm_provider}, model: {_model_name}).")
except Exception as e:
    print(f"CRITICAL ERROR configuring LLM: {e}")
    _llm_client = None


def _call_llm(prompt: str) -> str:
    if _llm_client is None:
        raise RuntimeError(
            f"LLM client not initialised. "
            f"Check {_llm_provider.upper()}_API_KEY and LLM_PROVIDER in your .env."
        )
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
    start_date: str          # "" or "YYYY-MM-DD" — no hard 2015 limit
    generated_code: str
    current_iteration_number: int
    current_config: dict
    all_iteration_results: List[dict]
    best_config_so_far: dict
    error: str


class ScreenGraphState(TypedDict):
    strategy_prompt: str
    screening_prompt: str
    start_date: str          # "" = no lower limit
    end_date: str            # "" = today
    generated_code: str
    screening_code: str
    screening_csv_path: str
    screening_dict: dict     # {"YYYY-MM-DD": ["AAPL", "TSLA", ...]}
    current_iteration_number: int
    current_config: dict
    all_iteration_results: List[dict]
    best_config_so_far: dict
    error: str


# ---------------------------------------------------------------------------
# Shared utility helpers
# ---------------------------------------------------------------------------

def get_ticker_from_prompt(prompt: str) -> str:
    """Extract the first uppercase 1–5 letter word as the ticker symbol."""
    match = re.search(r"\b([A-Z]{1,5})\b", prompt)
    if match:
        return match.group(1)
    raise ValueError("Could not extract a valid stock ticker from the prompt.")


def _extract_code_and_config(response_text: str):
    """Parse the LLM response for a ```python block and a ```json block."""
    code_match = re.search(r"```python\n(.*?)```", response_text, re.DOTALL)
    json_match = re.search(r"```json\n(.*?)```",   response_text, re.DOTALL)

    if not code_match:
        raise ValueError(
            "LLM response is missing the required ```python code block.\n"
            f"Response was:\n{response_text[:500]}"
        )
    if not json_match:
        raise ValueError(
            "LLM response is missing the required ```json config block.\n"
            f"Response was:\n{response_text[:500]}"
        )

    code   = code_match.group(1).strip()
    config = json.loads(json_match.group(1).strip())
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

--- BACKTRADER API REFERENCE ---
{api_docs}
--- END REFERENCE ---

User Strategy: "{{prompt}}"
"""
.replace("{api_docs}", _BT_API_DOCS.replace("{", "{{").replace("}", "}}"))
.replace("{{prompt}}", "{prompt}")
)

def generate_strategy_code(state: GraphState) -> GraphState:
    print("--- Node: generate_strategy_code ---")
    if _llm_client is None:
        return {**state, "error": "Gemini client not initialised. Check GEMINI_API_KEY."}

    prompt = _CODE_GEN_PROMPT.format(prompt=state["strategy_prompt"])
    try:
        response_text = _call_llm(prompt)
        generated_code, default_config = _extract_code_and_config(response_text)
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

        start = state.get("start_date") or "2000-01-01"
        raw = yf.download(
            ticker_symbol,
            start=start,
            end=datetime.now().strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise ValueError(f"No data returned by yfinance for '{ticker_symbol}'.")

        data_df = _flatten_yfinance_df(raw)
        data_df.index = pd.to_datetime(data_df.index)

        data_feed = bt.feeds.PandasData(
            dataname=data_df,
            open="open", high="high", low="low", close="close",
            volume="volume", openinterest=-1,
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
        cerebro = bt.Cerebro()
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(100_000.0)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
        cerebro.addstrategy(StrategyClass, **config)
        cerebro.addanalyzer(Expectancy,              _name="expectancy")
        cerebro.addanalyzer(bt.analyzers.DrawDown,   _name="drawdown")
        cerebro.addanalyzer(
            bt.analyzers.TimeReturn,
            _name="cagr",
            timeframe=bt.TimeFrame.Years,
        )
        cerebro.addanalyzer(PortfolioValueAnalyzer,  _name="portfolio")

        print(f"  Running cerebro with config: {config}")
        results = cerebro.run()
        metrics = get_metrics(cerebro, results)
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
        return {**state, "error": "Gemini client not initialised."}

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
  START_DATE         : str — "YYYY-MM-DD" lower bound filter (empty = no limit)
  END_DATE           : str — "YYYY-MM-DD" upper bound filter (empty = no limit)

Each stock file (e.g. 'aapl.us.txt') is a CSV with header:
  Date,Open,High,Low,Close,Volume,OpenInt

Screening logic to implement:
  <<SCREENING_PROMPT>>

MANDATORY CODE STRUCTURE — follow this pattern exactly:

  STEP 1: Define a single-file loader function _load_one(args) where args=(path, ticker).
    - Read only the columns you need with usecols=[...].
    - After reading, immediately filter rows to the needed date window:
        if START_DATE: df = df[df['Date'] >= START_DATE]
        if END_DATE:   df = df[df['Date'] <= END_DATE]
      (add a lookback buffer, e.g. subtract 90 days from START_DATE, for rolling calculations)
    - Add a 'ticker' column, return the filtered df, or return None on any error.

  STEP 2: Parallel-load ALL files with ThreadPoolExecutor(max_workers=16).
    Collect non-None results, then pd.concat into one combined DataFrame.

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

  STEP 4: For each date, select tickers that pass the criteria.
    Build a list of dicts: [{"date": "YYYY-MM-DD", "ticker": "AAPL"}, ...]
    Iterate over the filtered DataFrame rows (use .itertuples() or vectorised construction).
    Apply START_DATE / END_DATE string filters on the final date column.

  STEP 5: Write the output:
    pd.DataFrame(results, columns=['date', 'ticker']).to_csv(OUTPUT_PATH, index=False)

Output ONLY a single ```python code block implementing all five steps. No explanation, no other text.
"""


def generate_screening_code(state: ScreenGraphState) -> ScreenGraphState:
    print("--- Node: generate_screening_code ---")
    if _llm_client is None:
        return {**state, "error": "Gemini client not initialised."}

    # Use .replace() so the template's curly braces (code examples) don't need escaping
    prompt = _SCREENING_CODE_GEN_PROMPT.replace("<<SCREENING_PROMPT>>", state["screening_prompt"])
    try:
        response_text = _call_llm(prompt)
        code_match = re.search(r"```python\n(.*?)```", response_text, re.DOTALL)
        if not code_match:
            raise ValueError(
                "LLM response missing ```python block.\n"
                f"Response:\n{response_text[:500]}"
            )
        return {
            **state,
            "screening_code": code_match.group(1).strip(),
            "error":          None,
        }
    except Exception as e:
        print(f"ERROR in generate_screening_code: {e}")
        return {**state, "error": f"Failed to generate screening code: {e}"}


_SCREENING_TIMEOUT_SECS = 900  # 15 minutes


def run_screening(state: ScreenGraphState) -> ScreenGraphState:
    """Execute the Gemini-generated screening code as a subprocess with a 15-minute timeout."""
    print("--- Node: run_screening ---")
    import subprocess
    import sys
    import textwrap

    output_path = tempfile.mktemp(suffix=".csv")
    code_file   = tempfile.mktemp(suffix=".py")

    # Build a standalone Python script that embeds all variables and runs the
    # generated code.  Using repr() for paths/strings is injection-safe.
    header = textwrap.dedent(f"""\
        import pandas as pd
        import os
        from concurrent.futures import ThreadPoolExecutor

        DATASET_PATH = {repr(KAGGLE_STOCKS_PATH)}
        OUTPUT_PATH  = {repr(output_path)}
        START_DATE   = {repr(state.get("start_date") or "")}
        END_DATE     = {repr(state.get("end_date")   or "")}

    """)

    try:
        with open(code_file, "w") as fh:
            fh.write(header + state["screening_code"] + "\n")

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

--- BACKTRADER API REFERENCE ---
{api_docs}
--- END REFERENCE ---

User Strategy: "{{prompt}}"
"""
.replace("{api_docs}", _BT_API_DOCS.replace("{", "{{").replace("}", "}}"))
.replace("{{prompt}}", "{prompt}")
)


def generate_multi_strategy_code(state: ScreenGraphState) -> ScreenGraphState:
    print("--- Node: generate_multi_strategy_code ---")
    if _llm_client is None:
        return {**state, "error": "Gemini client not initialised."}

    prompt = _MULTI_STRATEGY_PROMPT.format(prompt=state["strategy_prompt"])
    try:
        response_text = _call_llm(prompt)
        generated_code, default_config = _extract_code_and_config(response_text)
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


def run_multi_backtest(state: ScreenGraphState) -> ScreenGraphState:
    iteration = state["current_iteration_number"]
    print(f"--- Node: run_multi_backtest  (iteration {iteration}) ---")
    try:
        screening_dict = state["screening_dict"]
        end_date   = state.get("end_date")   or None

        # Derive fromdate from the screening window so idle pre-screening years
        # don't distort CAGR and the equity curve.  Add a 100-day warmup buffer
        # so indicators (e.g. RSI-14) have enough bars before the first trade.
        if screening_dict:
            earliest_screen = min(screening_dict.keys())
            from_dt = (dt_module.datetime.strptime(earliest_screen, "%Y-%m-%d")
                       - dt_module.timedelta(days=100))
        else:
            # fall back to explicit start_date or a sane default
            raw = state.get("start_date") or "2000-01-01"
            from_dt = dt_module.datetime.strptime(raw, "%Y-%m-%d")

        all_tickers = sorted({t for tickers in screening_dict.values() for t in tickers})
        if not all_tickers:
            raise ValueError("Screening dict is empty — no tickers to trade.")
        print(f"  Unique tickers in screening: {len(all_tickers)}")

        cerebro = bt.Cerebro()
        loaded = 0
        skipped_invalid = 0

        for ticker in all_tickers:
            filepath = os.path.join(KAGGLE_STOCKS_PATH, f"{ticker.lower()}.us.txt")
            if not os.path.exists(filepath) or not _has_enough_data(filepath):
                skipped_invalid += 1
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
                    todate=dt_module.datetime.strptime(end_date,   "%Y-%m-%d") if end_date   else None,
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
            f"({skipped_invalid} skipped — insufficient/flat price data)."
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
        run_config = {**config, "screening": screening_dict}

        cerebro.broker.setcash(100_000.0)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.addstrategy(StrategyClass, **run_config)
        cerebro.addanalyzer(Expectancy,              _name="expectancy")
        cerebro.addanalyzer(bt.analyzers.DrawDown,   _name="drawdown")
        cerebro.addanalyzer(
            bt.analyzers.TimeReturn,
            _name="cagr",
            timeframe=bt.TimeFrame.Years,
        )
        cerebro.addanalyzer(PortfolioValueAnalyzer,  _name="portfolio")

        print(f"  Running multi-ticker cerebro with config: {config}")
        results = cerebro.run(stdstats=False, tradehistory=False)
        metrics = get_metrics(cerebro, results)
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
        return {**state, "error": "Gemini client not initialised."}

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
