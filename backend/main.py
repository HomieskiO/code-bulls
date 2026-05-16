from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
import json
from datetime import datetime
import pandas as pd
import yfinance as yf

from graph import app as langgraph_app, _call_gemini
from database import SessionLocal, Strategy, BacktestIteration

app = FastAPI()

BACKTEST_START = "2015-01-01"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _fetch_benchmark(ticker: str, start: str, end: str, initial_value: float = 100_000) -> list:
    """Return monthly benchmark portfolio values normalised to initial_value."""
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


class BacktestRequest(BaseModel):
    prompt:           str
    benchmark_ticker: str = "SPY"


_EXPLAIN_PROMPT = """
You are a quantitative trading analyst. An AI just optimized a trading strategy and found the best configuration. Explain it clearly to the user.

Original strategy request: "{prompt}"

Best configuration found: {config}

Performance metrics:
- CAGR: {cagr}%
- Total Return: {total_return}%
- Max Drawdown: {max_drawdown}%
- Win Rate: {win_rate}%
- Expectancy: ${expectancy} per trade
- Total Trades: {total_trades}
- Final Portfolio Value: ${final_value} (started at $100,000)

Write 3–4 concise sentences covering:
1. What the best configuration parameters mean in plain English
2. How the strategy performed overall (strengths)
3. Key risks or weaknesses to be aware of

Use plain language. No markdown headers or bullet points — just flowing prose.
"""


@app.post("/api/backtest")
def run_backtest_endpoint(request: BacktestRequest, db: Session = Depends(get_db)):
    inputs = {"strategy_prompt": request.prompt}
    final_state = langgraph_app.invoke(inputs)
    end_date = datetime.now().strftime("%Y-%m-%d")

    if final_state.get("error"):
        return {"error": final_state["error"]}

    best = final_state["best_config_so_far"]
    m    = best.get("metrics", {})

    # Generate plain-English explanation
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
        ))
    except Exception as e:
        explanation = f"Could not generate explanation: {e}"

    # Persist to DB
    strategy_record = Strategy(
        user_prompt=request.prompt,
        generated_python_code=final_state["generated_code"],
        timestamp=datetime.now().isoformat(),
    )
    db.add(strategy_record)
    db.commit()
    db.refresh(strategy_record)

    for it in final_state["all_iteration_results"]:
        met = it["metrics"]
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
        ))
    db.commit()

    benchmark_values = _fetch_benchmark(
        request.benchmark_ticker, BACKTEST_START, end_date
    )

    return {
        "strategy_id":        strategy_record.id,
        "best_configuration":  best,
        "all_iterations":     final_state["all_iteration_results"],
        "generated_code":     final_state["generated_code"],
        "explanation":        explanation,
        "benchmark_ticker":   request.benchmark_ticker,
        "benchmark_values":   benchmark_values,
    }


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
                "iteration":    it.iteration_number,
                "config":       it.config_json,
                "cagr":         it.cagr,
                "max_drawdown": it.max_drawdown,
                "win_rate":     it.win_rate,
                "expectancy":   it.expectancy,
            })
            if best_exp is None or (it.expectancy or 0) > best_exp:
                best_exp  = it.expectancy
                best_iter = it
        result.append({
            "id":        s.id,
            "prompt":    s.user_prompt,
            "timestamp": s.timestamp,
            "best_cagr":        best_iter.cagr        if best_iter else None,
            "best_win_rate":    best_iter.win_rate     if best_iter else None,
            "best_drawdown":    best_iter.max_drawdown if best_iter else None,
            "best_expectancy":  best_iter.expectancy   if best_iter else None,
            "iterations":       iters,
        })
    return result
