"""FastAPI entrypoint — thin validation layer over services."""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import SessionLocal, Strategy
from services.backtest_service import run_screened, run_single

app = FastAPI()

DATASET_MAX_END = "2017-11-10"
DEFAULT_START = "2010-01-01"
DEFAULT_END = DATASET_MAX_END


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _parse_ymd(value: str, field: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400, detail=f"{field} must be YYYY-MM-DD, got {value!r}"
        )


def _normalize_period(start_date: str, end_date: str) -> tuple[str, str]:
    start = (start_date or DEFAULT_START).strip()
    end = (end_date or DEFAULT_END).strip()
    start_d = _parse_ymd(start, "start_date")
    end_d = _parse_ymd(end, "end_date")
    max_d = _parse_ymd(DATASET_MAX_END, "DATASET_MAX_END")
    if end_d > max_d:
        end = DATASET_MAX_END
        end_d = max_d
    if start_d >= end_d:
        raise HTTPException(
            status_code=400,
            detail=f"start_date ({start}) must be before end_date ({end})",
        )
    return start, end


def _normalize_position_size_pct(value, default: float) -> float:
    try:
        v = float(value if value is not None else default)
    except (TypeError, ValueError):
        v = default
    return max(1.0, min(100.0, v))


class BacktestRequest(BaseModel):
    prompt: str
    start_date: str = DEFAULT_START
    end_date: str = DEFAULT_END
    benchmark_ticker: str = "SPY"
    position_size_pct: float = 100.0


class ScreenedBacktestRequest(BaseModel):
    strategy_prompt: str
    screening_prompt: str
    start_date: str = DEFAULT_START
    end_date: str = DEFAULT_END
    benchmark_ticker: str = "SPY"
    position_size_pct: float = 10.0


@app.post("/api/backtest")
def run_backtest_endpoint(request: BacktestRequest, db: Session = Depends(get_db)):
    start_date, end_date = _normalize_period(request.start_date, request.end_date)
    position_size_pct = _normalize_position_size_pct(request.position_size_pct, 100.0)
    return run_single(
        db,
        prompt=request.prompt,
        start_date=start_date,
        end_date=end_date,
        position_size_pct=position_size_pct,
        benchmark_ticker=request.benchmark_ticker,
    )


@app.post("/api/screen-backtest")
def run_screened_backtest(
    request: ScreenedBacktestRequest, db: Session = Depends(get_db)
):
    start_date, end_date = _normalize_period(request.start_date, request.end_date)
    position_size_pct = _normalize_position_size_pct(request.position_size_pct, 10.0)
    return run_screened(
        db,
        strategy_prompt=request.strategy_prompt,
        screening_prompt=request.screening_prompt,
        start_date=start_date,
        end_date=end_date,
        position_size_pct=position_size_pct,
        benchmark_ticker=request.benchmark_ticker,
    )


@app.get("/api/history")
def get_history(db: Session = Depends(get_db)):
    strategies = db.query(Strategy).order_by(Strategy.id.desc()).all()
    result = []
    for s in strategies:
        best_iter = None
        best_exp = None
        iters = []
        for it in s.iterations:
            iters.append({
                "iteration": it.iteration_number,
                "config": it.config_json,
                "cagr": it.cagr,
                "max_drawdown": it.max_drawdown,
                "win_rate": it.win_rate,
                "expectancy": it.expectancy,
            })
            if best_exp is None or (it.expectancy or 0) > best_exp:
                best_exp = it.expectancy
                best_iter = it
        result.append({
            "id": s.id,
            "prompt": s.user_prompt,
            "timestamp": s.timestamp,
            "start_date": s.start_date,
            "end_date": s.end_date,
            "position_size_pct": s.position_size_pct,
            "is_multi_stock": s.is_multi_stock,
            "best_cagr": best_iter.cagr if best_iter else None,
            "best_win_rate": best_iter.win_rate if best_iter else None,
            "best_drawdown": best_iter.max_drawdown if best_iter else None,
            "best_expectancy": best_iter.expectancy if best_iter else None,
            "iterations": iters,
        })
    return result
