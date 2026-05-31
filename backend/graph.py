"""
graph.py — LangGraph 3-iteration optimization loop.

Features:
  - Kaggle or Yahoo Finance data via data_loader
  - Multi-stock scanning with daily selection (scanner.py)
  - Risk profile injection into code generation
  - Broader optimization: stop_loss, take_profit, position_size always included
  - Per-iteration optimization explanations
  - Configurable optimization scope
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import json
import backtrader as bt
import re
import numpy as np
import math
import pandas as pd
from datetime import datetime

from llm_client import call_gemini
from data_loader import load_ticker_data
from scanner import compute_daily_selections
from evaluator import Expectancy, get_metrics, CAGRAnalyzer, PortfolioValueAnalyzer

# Keep backward-compat alias for main.py
_call_gemini = call_gemini

BACKTEST_START = "2015-01-01"

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _config_score(metrics: dict) -> float:
    cagr = float(metrics.get("cagr",         0) or 0)
    dd   = abs(float(metrics.get("max_drawdown", 0) or 0))
    return cagr * 0.65 - dd * 0.35


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    # Core (always required)
    strategy_prompt:          str
    generated_code:           str
    current_iteration_number: int
    current_config:           dict
    all_iteration_results:    List[dict]
    best_config_so_far:       dict
    error:                    str

    # Data source
    data_source:  str           # "yfinance" | "kaggle"

    # Tickers
    tickers:      List[str]     # [single] or [list for multi-stock]

    # Multi-stock scanning
    is_multi_stock:    bool
    daily_selections:  dict     # {"2020-01-03": ["AAPL", "MSFT"], ...}
    scan_rule:         str
    scan_top_n:        int

    # Risk profile (drives default param values in generated code)
    risk_profile: dict          # {level, stop_loss_pct, take_profit_pct, ...}

    # Optimization
    optimization_scope:        List[str]   # ["all"] | ["risk"] | ["strategy"]
    optimization_explanations: List[str]   # per-iteration explanation from LLM


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_ticker_from_prompt(prompt: str) -> str:
    m = re.search(r"\b([A-Z]{1,5})\b", prompt)
    if m:
        return m.group(1)
    raise ValueError("Could not extract a ticker symbol from the prompt.")


def _extract_code_and_config(text: str):
    code_m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    json_m = re.search(r"```json\n(.*?)```",   text, re.DOTALL)
    if not code_m:
        raise ValueError(f"LLM response missing ```python block.\n{text[:500]}")
    if not json_m:
        raise ValueError(f"LLM response missing ```json block.\n{text[:500]}")
    return code_m.group(1).strip(), json.loads(json_m.group(1).strip())


_CODE_FIXES = {
    "bt.indicators.crossover":   "bt.indicators.CrossOver",
    "bt.ind.Crossover":          "bt.indicators.CrossOver",
    "bt.indicators.Crossover":   "bt.indicators.CrossOver",
    "bt.indicators.ema(":        "bt.indicators.EMA(",
    "bt.indicators.sma(":        "bt.indicators.SMA(",
    "bt.indicators.rsi(":        "bt.indicators.RSI(",
    "bt.indicators.macd(":       "bt.indicators.MACD(",
    "bt.indicators.bollinger":   "bt.indicators.BollingerBands",
    "bt.indicators.Bollinger(":  "bt.indicators.BollingerBands(",
}

def _sanitize_code(code: str) -> str:
    for wrong, right in _CODE_FIXES.items():
        code = code.replace(wrong, right)
    return code


def _make_exec_namespace(daily_selections: dict = None) -> dict:
    return {
        "bt":                bt,
        "numpy":             np,
        "np":                np,
        "math":              math,
        "daily_selections":  daily_selections or {},
    }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_MULTI_STOCK_INSTRUCTIONS = """
MULTI-STOCK MODE — your strategy will run on multiple data feeds simultaneously:
- `self.datas` contains all loaded ticker feeds (one per ticker)
- Access a feed's name: `d._name` for each `d` in `self.datas`
- The global variable `daily_selections` (dict: {{date_str: [ticker_names]}}) tells you which tickers to trade each day
- Each bar, ONLY trade tickers in today's selection:
    today    = self.datetime.date().isoformat()
    selected = [t.upper() for t in daily_selections.get(today, [])]
    for d in self.datas:
        pos = self.getposition(d)
        if d._name.upper() not in selected:
            if pos.size > 0: self.close(data=d)
            continue
        # apply your entry/exit logic here, using `d` instead of `self.data`
        # e.g. self.buy(data=d)  self.close(data=d)
"""

_CODE_GEN_PROMPT = """
You are an expert in the `backtrader` Python library.
Convert the natural language trading strategy below into a complete, valid `backtrader.Strategy` class.

Crucial requirements:
1. Configure ALL numeric values via `params` — never hardcode them.
2. No imports in the class body — `bt`, `np`, and `daily_selections` are available as globals.
3. ALWAYS include these three risk parameters in `params` (use suggested defaults below):
   - stop_loss_pct   : float  (e.g. 0.05 = 5%; 0.0 = disabled)
   - take_profit_pct : float  (e.g. 0.10 = 10%; 0.0 = disabled)
   - position_size_pct : float  (e.g. 0.95 = allocate 95% of cash per trade)
4. Track entry price after a buy: `self.entry_price = self.data.close[0]`
5. Check stop-loss and take-profit on every bar if position is open.
{multi_stock_block}
6. Output EXACTLY one ```python block (the strategy class) and one ```json block (default param values).

Risk profile: {risk_level}
Suggested defaults:
  stop_loss_pct   = {stop_loss_pct}
  take_profit_pct = {take_profit_pct}
  position_size_pct = {position_size_pct}

User strategy: "{prompt}"
"""

_SCOPE_HINTS = {
    "all":      "You may adjust ANY parameter including stop_loss_pct, take_profit_pct, and position_size_pct.",
    "risk":     "ONLY adjust: stop_loss_pct, take_profit_pct, position_size_pct. Leave all others unchanged.",
    "strategy": "ONLY adjust strategy-specific parameters (indicators, periods, thresholds). Do NOT change stop_loss_pct, take_profit_pct, or position_size_pct.",
}

_OPTIMIZE_PROMPT = """
You are a quantitative analyst optimising a `backtrader` strategy.

Original strategy: "{prompt}"

Strategy code:
```python
{code}
```

Iteration {prev_iter} results:
  Config  : {prev_config}
  Metrics : {prev_metrics}

Goal: improve CAGR while keeping Max Drawdown low.

Scope: {scope_hint}

RULES:
- Use EXACTLY these parameter keys (no others): {valid_keys}
- Output ONLY a ```json block with the updated values.
- On a NEW LINE after the json block, write:
  EXPLANATION: <one sentence describing which params you changed and why>

Example:
```json
{{"fast_period": 10, "stop_loss_pct": 0.03}}
```
EXPLANATION: Tightened fast_period to capture shorter signals; reduced stop_loss_pct to limit drawdown.
"""


# ---------------------------------------------------------------------------
# Node 1: generate_strategy_code
# ---------------------------------------------------------------------------

def generate_strategy_code(state: GraphState) -> GraphState:
    print("--- Node: generate_strategy_code ---")

    rp            = state.get("risk_profile") or {}
    risk_level    = rp.get("level",              "moderate")
    stop_loss     = rp.get("stop_loss_pct",      0.05)
    take_profit   = rp.get("take_profit_pct",    0.10)
    position_size = rp.get("position_size_pct",  0.50)
    is_multi      = state.get("is_multi_stock",  False)

    multi_block = _MULTI_STOCK_INSTRUCTIONS if is_multi else ""

    prompt = _CODE_GEN_PROMPT.format(
        prompt=state["strategy_prompt"],
        risk_level=risk_level,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        position_size_pct=position_size,
        multi_stock_block=multi_block,
    )

    try:
        raw             = call_gemini(prompt)
        code, config    = _extract_code_and_config(raw)
        # Ensure risk params are always present in config
        config.setdefault("stop_loss_pct",   stop_loss)
        config.setdefault("take_profit_pct", take_profit)
        config.setdefault("position_size_pct", position_size)

        return {
            **state,
            "generated_code":           code,
            "current_config":           config,
            "current_iteration_number": 1,
            "all_iteration_results":    [],
            "best_config_so_far":       {},
            "optimization_explanations": [],
            "error":                    None,
        }
    except Exception as e:
        print(f"ERROR in generate_strategy_code: {e}")
        return {**state, "error": str(e)}


# ---------------------------------------------------------------------------
# Node 2: run_backtest
# ---------------------------------------------------------------------------

def run_backtest(state: GraphState) -> GraphState:
    iteration = state["current_iteration_number"]
    print(f"--- Node: run_backtest (iter {iteration}) ---")

    try:
        data_source     = state.get("data_source",      "yfinance")
        tickers         = state.get("tickers",          [])
        is_multi        = state.get("is_multi_stock",   False)
        daily_sel       = state.get("daily_selections", {})
        rp              = state.get("risk_profile",     {})
        position_pct    = rp.get("position_size_pct",   0.95)
        scan_top_n      = state.get("scan_top_n",       1)
        end_date        = datetime.now().strftime("%Y-%m-%d")

        # Resolve tickers
        if not tickers:
            tickers = [get_ticker_from_prompt(state["strategy_prompt"])]

        # --- Pre-compute daily selections for multi-stock ---
        if is_multi and len(tickers) > 1 and not daily_sel:
            print(f"  [scanner] Computing daily selections for {tickers} …")
            daily_sel = compute_daily_selections(
                tickers=tickers,
                start=BACKTEST_START,
                end=end_date,
                source=data_source,
                rule=state.get("scan_rule", "top_volume"),
                top_n=scan_top_n,
            )
            # Persist back to state so future iterations reuse it
            state = {**state, "daily_selections": daily_sel}

        # --- Load data feeds ---
        data_feeds = []
        for ticker in tickers:
            print(f"  Loading {ticker} via {data_source} …")
            df = load_ticker_data(ticker, BACKTEST_START, end_date, data_source)
            df.index = pd.to_datetime(df.index)
            feed = bt.feeds.PandasData(
                dataname=df,
                name=ticker,
                open="open", high="high", low="low",
                close="close", volume="volume",
                openinterest=-1,
            )
            data_feeds.append(feed)

        # --- Compile strategy ---
        namespace = _make_exec_namespace(daily_sel)
        sanitized = _sanitize_code(state["generated_code"])
        try:
            exec(sanitized, namespace)
        except SyntaxError as se:
            raise ValueError(f"Syntax error in generated code: {se}")

        StrategyClass = next(
            (v for v in namespace.values()
             if isinstance(v, type) and issubclass(v, bt.Strategy) and v is not bt.Strategy),
            None,
        )
        if StrategyClass is None:
            raise ImportError(f"No bt.Strategy subclass found in:\n{state['generated_code'][:400]}")

        # --- Cerebro setup ---
        config       = state["current_config"]
        alloc_pct    = max(5, min(95, int(position_pct * 100 / max(scan_top_n, 1))))

        cerebro = bt.Cerebro()
        for feed in data_feeds:
            cerebro.adddata(feed)
        cerebro.broker.setcash(100_000.0)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.addsizer(bt.sizers.PercentSizer, percents=alloc_pct)
        cerebro.addstrategy(StrategyClass, **config)
        cerebro.addanalyzer(Expectancy,             _name="expectancy")
        cerebro.addanalyzer(bt.analyzers.DrawDown,  _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="cagr", timeframe=bt.TimeFrame.Years)
        cerebro.addanalyzer(PortfolioValueAnalyzer,  _name="portfolio")

        print(f"  Config: {config}")
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
            "tickers":               tickers,
            "daily_selections":      daily_sel,
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

def optimize_strategy(state: GraphState) -> GraphState:
    iteration = state["current_iteration_number"]
    print(f"--- Node: optimize_strategy (was iter {iteration}) ---")

    prev       = state["all_iteration_results"][-1]
    valid_keys = list(prev["config"].keys())

    # Build scope hint from optimization_scope list
    scope      = state.get("optimization_scope", ["all"])
    scope_hint = " ".join(_SCOPE_HINTS.get(s, "") for s in scope) or _SCOPE_HINTS["all"]

    prompt = _OPTIMIZE_PROMPT.format(
        prompt=state["strategy_prompt"],
        code=state["generated_code"],
        prev_iter=prev["iteration"],
        prev_config=json.dumps(prev["config"],  indent=2),
        prev_metrics=json.dumps(prev["metrics"], indent=2),
        valid_keys=json.dumps(valid_keys),
        scope_hint=scope_hint,
    )

    try:
        raw       = call_gemini(prompt)
        json_m    = re.search(r"```json\n(.*?)```", raw, re.DOTALL)
        if not json_m:
            raise ValueError(f"No ```json block in optimize response:\n{raw[:400]}")

        raw_cfg    = json.loads(json_m.group(1).strip())
        new_config = {k: raw_cfg.get(k, prev["config"][k]) for k in valid_keys}

        # Extract explanation
        exp_m       = re.search(r"EXPLANATION:\s*(.+)", raw)
        explanation = exp_m.group(1).strip() if exp_m else f"Iteration {iteration+1}: parameters adjusted."

        print(f"  New config: {new_config}")
        print(f"  Explanation: {explanation}")

        return {
            **state,
            "current_config":           new_config,
            "current_iteration_number": iteration + 1,
            "optimization_explanations": state.get("optimization_explanations", []) + [explanation],
        }
    except Exception as e:
        print(f"ERROR in optimize_strategy: {e}")
        return {**state, "error": str(e)}


# ---------------------------------------------------------------------------
# Edge conditions
# ---------------------------------------------------------------------------

def should_continue_after_generation(state: GraphState) -> str:
    if state.get("error"):
        return END
    return "run_backtest"


def should_continue_after_run(state: GraphState) -> str:
    if state.get("error"):
        return END
    if state["current_iteration_number"] >= 3:
        return END
    return "optimize_strategy"


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

workflow = StateGraph(GraphState)
workflow.add_node("generate_strategy_code", generate_strategy_code)
workflow.add_node("run_backtest",           run_backtest)
workflow.add_node("optimize_strategy",      optimize_strategy)

workflow.set_entry_point("generate_strategy_code")

workflow.add_conditional_edges(
    "generate_strategy_code",
    should_continue_after_generation,
    {"run_backtest": "run_backtest", END: END},
)
workflow.add_conditional_edges(
    "run_backtest",
    should_continue_after_run,
    {"optimize_strategy": "optimize_strategy", END: END},
)
workflow.add_edge("optimize_strategy", "run_backtest")

app = workflow.compile()