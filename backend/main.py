from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
import pandas as pd
import yfinance as yf

from graph import app as langgraph_app, _call_gemini, BACKTEST_START
from database import SessionLocal, Strategy, BacktestIteration

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def _fetch_benchmark(ticker: str, start: str, end: str, initial_value: float = 100_000) -> list:
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            return []
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        close   = raw["close"].dropna()
        scale   = initial_value / float(close.iloc[0])
        monthly = close.resample("MS").first()
        return [
            {"date": str(d.date()), "value": round(float(v) * scale, 2)}
            for d, v in monthly.items()
        ]
    except Exception as e:
        print(f"WARNING: could not fetch benchmark {ticker}: {e}")
        return []


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    prompt:             str
    benchmark_ticker:   str            = "SPY"
    data_source:        str            = "yfinance"
    tickers:            List[str]      = Field(default_factory=list)
    is_multi_stock:     bool           = False
    scan_rule:          str            = "top_volume"
    scan_top_n:         int            = 1
    risk_profile:       Dict[str, Any] = Field(default_factory=dict)
    optimization_scope: List[str]      = Field(default_factory=lambda: ["all"])


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message:    str


# ---------------------------------------------------------------------------
# Explain prompt
# ---------------------------------------------------------------------------

_EXPLAIN_PROMPT = """
You are a quantitative trading analyst. An AI just optimized a trading strategy.
Explain the results clearly to the user.

Original request: "{prompt}"
Best configuration: {config}
Performance metrics:
  CAGR: {cagr}% | Total Return: {total_return}% | Max Drawdown: {max_drawdown}%
  Win Rate: {win_rate}% | Expectancy: ${expectancy}/trade | Total Trades: {total_trades}
  Final portfolio: ${final_value} (started at $100,000)

Optimization journey: {opt_explanations}

Write 3-4 concise sentences covering:
1. What the best configuration parameters mean in plain English
2. How the strategy performed (strengths)
3. Key risks or weaknesses
4. What changed most across iterations and why it helped (if applicable)

Plain prose only — no markdown headers or bullet points.
"""


# ---------------------------------------------------------------------------
# POST /api/chat — conversational setup
# ---------------------------------------------------------------------------

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    from conversation import process_message
    return process_message(request.session_id, request.message)


# ---------------------------------------------------------------------------
# POST /api/backtest — run optimization loop
# ---------------------------------------------------------------------------

@app.post("/api/backtest")
def run_backtest_endpoint(request: BacktestRequest, db: Session = Depends(get_db)):
    inputs = {
        "strategy_prompt":    request.prompt,
        "data_source":        request.data_source,
        "tickers":            request.tickers or [],
        "is_multi_stock":     request.is_multi_stock,
        "scan_rule":          request.scan_rule,
        "scan_top_n":         request.scan_top_n,
        "daily_selections":   {},
        "risk_profile":       request.risk_profile or {},
        "optimization_scope": request.optimization_scope or ["all"],
        "optimization_explanations": [],
        # defaults for TypedDict fields the graph expects
        "generated_code":           "",
        "current_iteration_number": 1,
        "current_config":           {},
        "all_iteration_results":    [],
        "best_config_so_far":       {},
        "error":                    None,
    }

    final_state = langgraph_app.invoke(inputs)
    end_date    = datetime.now().strftime("%Y-%m-%d")

    if final_state.get("error"):
        return {"error": final_state["error"]}

    best = final_state["best_config_so_far"]
    m    = best.get("metrics", {})
    opt_explanations = final_state.get("optimization_explanations", [])

    # Generate explanation
    explanation = ""
    try:
        explanation = _call_gemini(_EXPLAIN_PROMPT.format(
            prompt=request.prompt,
            config=json.dumps(best.get("config", {}), indent=2),
            cagr=m.get("cagr", 0),
            total_return=m.get("total_return_pct", 0),
            max_drawdown=m.get("max_drawdown", 0),
            win_rate=m.get("win_rate", 0),
            expectancy=m.get("expectancy", 0),
            total_trades=m.get("total_trades", 0),
            final_value=m.get("final_portfolio_value", 0),
            opt_explanations="; ".join(opt_explanations) or "No iterations completed.",
        ))
    except Exception as e:
        explanation = f"Could not generate explanation: {e}"

    # Persist to DB
    strategy_record = Strategy(
        user_prompt=request.prompt,
        generated_python_code=final_state["generated_code"],
        timestamp=datetime.now().isoformat(),
        data_source=request.data_source,
        risk_profile_json=json.dumps(request.risk_profile),
        is_multi_stock=request.is_multi_stock,
        tickers_json=json.dumps(final_state.get("tickers", request.tickers)),
    )
    db.add(strategy_record)
    db.commit()
    db.refresh(strategy_record)

    for i, it in enumerate(final_state["all_iteration_results"]):
        met = it["metrics"]
        exp = opt_explanations[i - 1] if i > 0 and i - 1 < len(opt_explanations) else ""
        db.add(BacktestIteration(
            strategy_id=strategy_record.id,
            iteration_number=it["iteration"],
            config_json=json.dumps(it["config"]),
            cagr=met.get("cagr"),
            max_drawdown=met.get("max_drawdown"),
            avg_win=met.get("avg_win"),
            avg_loss=met.get("avg_loss"),
            win_rate=met.get("win_rate"),
            expectancy=met.get("expectancy"),
            opt_explanation=exp,
        ))
    db.commit()

    benchmark_values = _fetch_benchmark(request.benchmark_ticker, BACKTEST_START, end_date)

    return {
        "strategy_id":              strategy_record.id,
        "best_configuration":       best,
        "all_iterations":           final_state["all_iteration_results"],
        "generated_code":           final_state["generated_code"],
        "explanation":              explanation,
        "benchmark_ticker":         request.benchmark_ticker,
        "benchmark_values":         benchmark_values,
        "optimization_explanations": opt_explanations,
        "risk_profile":             request.risk_profile,
        "tickers":                  final_state.get("tickers", request.tickers),
        "data_source":              request.data_source,
    }


# ---------------------------------------------------------------------------
# GET /api/history
# ---------------------------------------------------------------------------

@app.get("/api/history")
def get_history(db: Session = Depends(get_db)):
    strategies = db.query(Strategy).order_by(Strategy.id.desc()).all()
    result = []
    for s in strategies:
        best_iter = None
        best_exp  = None
        iters = []
        for it in s.iterations:
            iters.append({
                "iteration":       it.iteration_number,
                "config":          it.config_json,
                "cagr":            it.cagr,
                "max_drawdown":    it.max_drawdown,
                "win_rate":        it.win_rate,
                "expectancy":      it.expectancy,
                "opt_explanation": it.opt_explanation or "",
            })
            if best_exp is None or (it.expectancy or 0) > best_exp:
                best_exp  = it.expectancy
                best_iter = it
        result.append({
            "id":               s.id,
            "prompt":           s.user_prompt,
            "timestamp":        s.timestamp,
            "data_source":      s.data_source or "yfinance",
            "is_multi_stock":   s.is_multi_stock or False,
            "best_cagr":        best_iter.cagr        if best_iter else None,
            "best_win_rate":    best_iter.win_rate     if best_iter else None,
            "best_drawdown":    best_iter.max_drawdown if best_iter else None,
            "best_expectancy":  best_iter.expectancy   if best_iter else None,
            "iterations":       iters,
        })
    return result


# ---------------------------------------------------------------------------
# GET /api/kaggle/tickers — list available Kaggle tickers
# ---------------------------------------------------------------------------

@app.get("/api/kaggle/tickers")
def list_kaggle_tickers(subdir: str = "Stocks"):
    from data_loader import list_kaggle_tickers as _list
    tickers = _list(subdir)
    return {"tickers": tickers, "count": len(tickers)}