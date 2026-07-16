"""Orchestration: invoke workflows, explain, persist, benchmark."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from database import BacktestIteration, Strategy
from fixtures import fixture_explanation, fixtures_enabled
from graph import multi_app, app as single_app, _call_gemini
from prompts import explain_prompt


def fetch_benchmark(
    ticker: str, start: str, end: str, initial_value: float = 100_000
) -> list:
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            return []
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        close = raw["close"].dropna()
        scale = initial_value / float(close.iloc[0])
        monthly = close.resample("MS").first()
        return [
            {"date": str(d.date()), "value": round(float(v) * scale, 2)}
            for d, v in monthly.items()
        ]
    except Exception as e:
        print(f"WARNING: could not fetch benchmark {ticker}: {e}")
        return []


def _explain(prompt: str, start: str, end: str, best: dict) -> str:
    m = best.get("metrics", {})
    try:
        return _call_gemini(
            explain_prompt(
                prompt=prompt,
                period_start=start,
                period_end=end,
                config=json.dumps(best.get("config", {}), indent=2),
                cagr=m.get("cagr", 0),
                total_return=m.get("total_return_pct", 0),
                max_drawdown=m.get("max_drawdown", 0),
                win_rate=m.get("win_rate", 0),
                expectancy=m.get("expectancy", 0),
                total_trades=m.get("total_trades", 0),
                final_value=m.get("final_portfolio_value", 0),
            )
        )
    except Exception as e:
        return f"Could not generate explanation: {e}"


def persist_run(
    db: Session,
    *,
    user_prompt: str,
    generated_code: str,
    final_state: dict,
    start_date: str,
    end_date: str,
    position_size_pct: float,
    is_multi: bool = False,
    screening_prompt: str = "",
    screening_code: str = "",
    explanation: str = "",
) -> Strategy:
    best = final_state.get("best_config_so_far") or {}
    metrics = best.get("metrics") or {}
    trades = metrics.get("trades") or []
    unique_tickers = []
    screening_dict = final_state.get("screening_dict") or {}
    if screening_dict:
        unique_tickers = sorted({t for v in screening_dict.values() for t in v})

    # Slim metrics for DB (drop huge series if needed — keep trades)
    best_metrics = {
        k: v for k, v in metrics.items() if k != "portfolio_values"
    }
    # keep portfolio_values length capped in storage
    pv = metrics.get("portfolio_values") or []
    if len(pv) > 500:
        step = max(1, len(pv) // 500)
        best_metrics["portfolio_values"] = pv[::step]
    else:
        best_metrics["portfolio_values"] = pv

    rec = Strategy(
        user_prompt=user_prompt,
        generated_python_code=generated_code,
        timestamp=datetime.now().isoformat(),
        data_source="kaggle" if is_multi else "yfinance",
        is_multi_stock=is_multi,
        tickers_json=json.dumps(unique_tickers),
        start_date=start_date,
        end_date=end_date,
        position_size_pct=position_size_pct,
        screening_prompt=screening_prompt,
        screening_code=screening_code,
        best_metrics_json=json.dumps(best_metrics),
        trades_json=json.dumps(trades),
        explanation=explanation,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    for it in final_state.get("all_iteration_results") or []:
        met = it.get("metrics") or {}
        slim = {k: v for k, v in met.items() if k not in ("portfolio_values", "trades")}
        db.add(
            BacktestIteration(
                strategy_id=rec.id,
                iteration_number=it.get("iteration"),
                config_json=json.dumps(it.get("config") or {}),
                cagr=met.get("cagr"),
                max_drawdown=met.get("max_drawdown"),
                avg_win=met.get("avg_win"),
                avg_loss=met.get("avg_loss"),
                win_rate=met.get("win_rate"),
                expectancy=met.get("expectancy"),
                metrics_json=json.dumps(slim),
            )
        )
    db.commit()
    return rec


def run_single(
    db: Session,
    *,
    prompt: str,
    start_date: str,
    end_date: str,
    position_size_pct: float,
    benchmark_ticker: str = "SPY",
    use_fixtures: bool = False,
) -> Dict[str, Any]:
    use_fx = fixtures_enabled(use_fixtures)
    inputs = {
        "strategy_prompt": prompt,
        "start_date": start_date,
        "end_date": end_date,
        "position_size_pct": position_size_pct,
        "use_fixtures": use_fx,
    }
    final_state = single_app.invoke(inputs)
    if final_state.get("error"):
        return {
            "error": final_state["error"],
            "generated_code": final_state.get("generated_code", ""),
            "period": {"start": start_date, "end": end_date},
            "position_size_pct": position_size_pct,
            "used_fixtures": use_fx,
        }

    best = final_state["best_config_so_far"]
    m = best.get("metrics", {})
    if use_fx:
        explanation = fixture_explanation("single-stock")
    else:
        explanation = _explain(prompt, start_date, end_date, best)
    rec = persist_run(
        db,
        user_prompt=prompt,
        generated_code=final_state["generated_code"],
        final_state=final_state,
        start_date=start_date,
        end_date=end_date,
        position_size_pct=position_size_pct,
        is_multi=False,
        explanation=explanation,
    )
    pv = m.get("portfolio_values") or []
    bm_start = m.get("period_start") or (pv[0]["date"] if pv else start_date)
    bm_end = m.get("period_end") or (pv[-1]["date"] if pv else end_date)
    return {
        "strategy_id": rec.id,
        "best_configuration": best,
        "all_iterations": final_state["all_iteration_results"],
        "generated_code": final_state["generated_code"],
        "explanation": explanation,
        "benchmark_ticker": benchmark_ticker,
        "benchmark_values": fetch_benchmark(benchmark_ticker, bm_start, bm_end),
        "period": {"start": start_date, "end": end_date},
        "position_size_pct": position_size_pct,
        "used_fixtures": use_fx,
    }


def run_screened(
    db: Session,
    *,
    strategy_prompt: str,
    screening_prompt: str,
    start_date: str,
    end_date: str,
    position_size_pct: float,
    benchmark_ticker: str = "SPY",
    use_fixtures: bool = False,
) -> Dict[str, Any]:
    use_fx = fixtures_enabled(use_fixtures)
    inputs = {
        "strategy_prompt": strategy_prompt,
        "screening_prompt": screening_prompt,
        "start_date": start_date,
        "end_date": end_date,
        "position_size_pct": position_size_pct,
        "use_fixtures": use_fx,
    }
    final_state = multi_app.invoke(inputs)
    if final_state.get("error"):
        return {
            "error": final_state["error"],
            "generated_code": final_state.get("generated_code", ""),
            "screening_code": final_state.get("screening_code", ""),
            "period": {"start": start_date, "end": end_date},
            "position_size_pct": position_size_pct,
            "used_fixtures": use_fx,
        }

    best = final_state["best_config_so_far"]
    m = best.get("metrics", {})
    prompt = f"[Screening] {screening_prompt}\n[Strategy] {strategy_prompt}"
    if use_fx:
        explanation = fixture_explanation("multi-stock")
    else:
        explanation = _explain(prompt, start_date, end_date, best)
    rec = persist_run(
        db,
        user_prompt=f"[SCREENED] {screening_prompt} | {strategy_prompt}",
        generated_code=final_state.get("generated_code", ""),
        final_state=final_state,
        start_date=start_date,
        end_date=end_date,
        position_size_pct=position_size_pct,
        is_multi=True,
        screening_prompt=screening_prompt,
        screening_code=final_state.get("screening_code", ""),
        explanation=explanation,
    )
    screening_dict = final_state.get("screening_dict") or {}
    unique_tickers = sorted({t for v in screening_dict.values() for t in v})
    pv = m.get("portfolio_values") or []
    bm_start = m.get("period_start") or (pv[0]["date"] if pv else start_date)
    bm_end = m.get("period_end") or (pv[-1]["date"] if pv else end_date)
    return {
        "strategy_id": rec.id,
        "best_configuration": best,
        "all_iterations": final_state.get("all_iteration_results", []),
        "generated_code": final_state.get("generated_code", ""),
        "screening_code": final_state.get("screening_code", ""),
        "explanation": explanation,
        "benchmark_ticker": benchmark_ticker,
        "benchmark_values": fetch_benchmark(benchmark_ticker, bm_start, bm_end),
        "period": {"start": start_date, "end": end_date},
        "position_size_pct": position_size_pct,
        "used_fixtures": use_fx,
        "screening_summary": {
            "unique_tickers": unique_tickers,
            "total_ticker_days": sum(len(v) for v in screening_dict.values()),
            "date_range": {
                "start": min(screening_dict.keys()) if screening_dict else None,
                "end": max(screening_dict.keys()) if screening_dict else None,
            },
            "max_unique_tickers": __import__(
                "screening.engine", fromlist=["MAX_UNIQUE_TICKERS"]
            ).MAX_UNIQUE_TICKERS,
        },
    }
