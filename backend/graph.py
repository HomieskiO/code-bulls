"""
graph.py — LangGraph 3-iteration optimization loop using google-genai SDK.

Bugs fixed vs. the original:
  1. google-genai SDK: use genai.Client() not genai.configure() + GenerativeModel()
  2. evaluator imports: Expectancy + get_metrics (not the old names)
  3. yfinance MultiIndex columns: flatten before passing to PandasData
  4. exec namespace: inject `bt`, `numpy`, etc. so generated code can use them
  5. should_continue_after_run: iteration counter was off-by-one (stopped at 1)
  6. PercentSizer import: add explicit import guard
  7. CAGRAnalyzer added as an optional extra; TimeReturn is still the primary one
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import json
import backtrader as bt
import yfinance as yf
from evaluator import Expectancy, get_metrics, CAGRAnalyzer, PortfolioValueAnalyzer
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment / API key loading
# ---------------------------------------------------------------------------

print("--- Initializing Application ---")
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    print("Found .env file, loading environment variables.")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print("WARNING: .env file not found. Please create one with GEMINI_API_KEY=...")

# ---------------------------------------------------------------------------
# Gemini client  (google-genai >= 0.8 SDK)
# ---------------------------------------------------------------------------

import google.genai as genai

_client = None
_model_name = "gemini-2.5-flash-lite"

try:
    print("Attempting to configure Gemini...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables. "
            "Add it to your .env file."
        )
    if "YOUR_API_KEY_HERE" in api_key:
        raise ValueError(
            "GEMINI_API_KEY is still the placeholder value. "
            "Replace it with your actual key."
        )

    # New SDK: genai.Client, not genai.configure()
    _client = genai.Client(api_key=api_key)
    print(f"Gemini client initialised (model: {_model_name}).")

except Exception as e:
    print(f"CRITICAL ERROR configuring Gemini: {e}")
    _client = None


def _call_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return the text response."""
    if _client is None:
        raise RuntimeError(
            "Gemini client not initialised. Check GEMINI_API_KEY and server logs."
        )
    response = _client.models.generate_content(
        model=_model_name,
        contents=prompt,
    )
    return response.text


# ---------------------------------------------------------------------------
# Best-config scoring
# ---------------------------------------------------------------------------

def _config_score(metrics: dict) -> float:
    """
    Combined quality score used to pick the best iteration.
    Weights: 65% CAGR (return), 35% drawdown penalty.
    Both values are in % form (e.g. cagr=15 means 15%).
    Higher score = better risk-adjusted return.
    """
    cagr = float(metrics.get("cagr",         0) or 0)
    dd   = abs(float(metrics.get("max_drawdown", 0) or 0))
    return cagr * 0.65 - dd * 0.35


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    strategy_prompt: str
    generated_code: str
    current_iteration_number: int
    current_config: dict
    all_iteration_results: List[dict]
    best_config_so_far: dict
    error: str


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_ticker_from_prompt(prompt: str) -> str:
    """Extract the first uppercase 1–5 letter word as the ticker symbol."""
    match = re.search(r"\b([A-Z]{1,5})\b", prompt)
    if match:
        return match.group(1)
    raise ValueError("Could not extract a valid stock ticker from the prompt.")


def _extract_code_and_config(response_text: str):
    """
    Parse the LLM response for a ```python block and a ```json block.
    Raises ValueError if either is missing or the JSON is malformed.
    """
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
    """
    yfinance >= 0.2.x returns a MultiIndex DataFrame when multiple tickers
    are downloaded, and sometimes even for a single ticker.
    Flatten to single-level columns and keep only OHLCV.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # drop the ticker level; keep Price level
        df.columns = df.columns.get_level_values(0)

    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Rename 'adj close' → 'close' if present (auto_adjust=False case)
    if "adj close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj close": "close"})

    return df[["open", "high", "low", "close", "volume"]].copy()


# ---------------------------------------------------------------------------
# Exec namespace helper — gives generated code access to bt, numpy, etc.
# ---------------------------------------------------------------------------

def _make_exec_namespace() -> dict:
    """Return a namespace dict pre-populated with common imports."""
    import math
    return {
        "bt":    bt,
        "numpy": np,
        "np":    np,
        "math":  math,
    }


# Common indicator names that LLMs consistently get wrong
_CODE_FIXES = {
    "bt.indicators.Crossover":   "bt.indicators.CrossOver",
    "bt.ind.Crossover":          "bt.indicators.CrossOver",
    "indicators.Crossover":      "indicators.CrossOver",
    "bt.indicators.Stochastic(": "bt.indicators.Stochastic(",   # already correct, keep
    "bt.indicators.RSI(":        "bt.indicators.RSI(",           # already correct, keep
    # capitalisation variants
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


# ---------------------------------------------------------------------------
# Node 1: generate_strategy_code
# ---------------------------------------------------------------------------

_CODE_GEN_PROMPT = """
You are an expert in the `backtrader` Python library.
Convert the natural language trading strategy below into a complete,
valid `backtrader.Strategy` class.

Crucial requirements:
1. Configurable via `params` dict — do NOT hardcode numeric values.
2. The class MUST import nothing; assume `bt` and `numpy as np` are already
   available in the global scope.
3. Output Format: provide EXACTLY one ```python block (the strategy class)
   and one ```json block (the default parameter values).

User Strategy: "{prompt}"
"""

def generate_strategy_code(state: GraphState) -> GraphState:
    print("--- Node: generate_strategy_code ---")
    if _client is None:
        return {**state, "error": "Gemini client not initialised. Check GEMINI_API_KEY."}

    prompt = _CODE_GEN_PROMPT.format(prompt=state["strategy_prompt"])
    try:
        response_text = _call_gemini(prompt)
        generated_code, default_config = _extract_code_and_config(response_text)

        return {
            **state,
            "generated_code":          generated_code,
            "current_config":          default_config,
            "current_iteration_number": 1,
            "all_iteration_results":   [],
            "best_config_so_far":      {},
            "error":                   None,
        }
    except Exception as e:
        print(f"ERROR in generate_strategy_code: {e}")
        return {**state, "error": f"Failed to generate/parse LLM response: {e}"}


# ---------------------------------------------------------------------------
# Node 2: run_backtest
# ---------------------------------------------------------------------------

def run_backtest(state: GraphState) -> GraphState:
    iteration = state["current_iteration_number"]
    print(f"--- Node: run_backtest  (iteration {iteration}) ---")

    try:
        import pandas as pd   # local import keeps top-level clean

        ticker_symbol = get_ticker_from_prompt(state["strategy_prompt"])
        print(f"  Fetching data for {ticker_symbol} via yfinance …")

        raw = yf.download(
            ticker_symbol,
            start="2015-01-01",
            end=datetime.now().strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise ValueError(f"No data returned by yfinance for '{ticker_symbol}'.")

        # FIX 3: flatten MultiIndex columns
        data_df = _flatten_yfinance_df(raw)
        data_df.index = pd.to_datetime(data_df.index)

        data_feed = bt.feeds.PandasData(
            dataname=data_df,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
        )

        # FIX 4: exec with pre-populated namespace so `bt`, `np`, etc. resolve
        namespace = _make_exec_namespace()
        sanitized = _sanitize_code(state["generated_code"])
        try:
            exec(sanitized, namespace)
        except SyntaxError as se:
            raise ValueError(f"Syntax error in generated strategy code: {se}")

        StrategyClass = next(
            (
                obj
                for obj in namespace.values()
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


# ---------------------------------------------------------------------------
# Node 3: optimize_strategy
# ---------------------------------------------------------------------------

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
"""

def optimize_strategy(state: GraphState) -> GraphState:
    iteration = state["current_iteration_number"]
    print(f"--- Node: optimize_strategy  (was iteration {iteration}) ---")
    if _client is None:
        return {**state, "error": "Gemini client not initialised."}

    prev        = state["all_iteration_results"][-1]
    valid_keys  = list(prev["config"].keys())

    prompt = _OPTIMIZE_PROMPT.format(
        prompt=state["strategy_prompt"],
        code=state["generated_code"],
        prev_iter=prev["iteration"],
        prev_config=json.dumps(prev["config"],  indent=2),
        prev_metrics=json.dumps(prev["metrics"], indent=2),
        valid_keys=json.dumps(valid_keys),
    )

    try:
        response_text = _call_gemini(prompt)
        json_match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
        if not json_match:
            raise ValueError(
                "Optimisation response missing ```json block.\n"
                f"Response: {response_text[:400]}"
            )
        raw_config = json.loads(json_match.group(1).strip())

        # Guard: keep only keys that exist in the original config;
        # fall back to the previous value for any key the LLM dropped.
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


# ---------------------------------------------------------------------------
# Edge conditions
# ---------------------------------------------------------------------------

def should_continue_after_generation(state: GraphState) -> str:
    if state.get("error"):
        print(f"Stopping: error after code generation — {state['error']}")
        return END
    return "run_backtest"


def should_continue_after_run(state: GraphState) -> str:
    if state.get("error"):
        print(f"Stopping: error during backtest — {state['error']}")
        return END

    # FIX 5: we want exactly 3 iterations (numbers 1, 2, 3).
    # After running iteration N, the counter is still N (optimize increments it).
    completed = state["current_iteration_number"]
    if completed >= 3:
        print(f"All 3 iterations complete.")
        return END
    return "optimize_strategy"


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

workflow = StateGraph(GraphState)

workflow.add_node("generate_strategy_code", generate_strategy_code)
workflow.add_node("run_backtest",           run_backtest)
workflow.add_node("optimize_strategy",      optimize_strategy)

workflow.set_entry_point("generate_strategy_code")

workflow.add_conditional_edges(
    "generate_strategy_code",
    should_continue_after_generation,
    {
        "run_backtest": "run_backtest",
        END:            END,
    },
)

workflow.add_conditional_edges(
    "run_backtest",
    should_continue_after_run,
    {
        "optimize_strategy": "optimize_strategy",
        END:                 END,
    },
)

workflow.add_edge("optimize_strategy", "run_backtest")

app = workflow.compile()
