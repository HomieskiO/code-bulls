"""Orchestration: staged generate / adapt / optimize, explain, persist, benchmark."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from database import BacktestIteration, Strategy
from fixtures import fixture_explanation, fixtures_enabled
from graph import _call_gemini
from prompts import explain_prompt
from screening.engine import MAX_UNIQUE_TICKERS
from services.draft_store import get_draft, save_draft, update_draft
from workflows.multi import run_multi_adapt, run_multi_generate, run_multi_optimize
from workflows.single import run_single_adapt, run_single_generate, run_single_optimize


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

    best_metrics = {
        k: v for k, v in metrics.items() if k != "portfolio_values"
    }
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


def _screening_summary(screening_dict: dict) -> dict:
    unique_tickers = sorted({t for v in screening_dict.values() for t in v}) if screening_dict else []
    return {
        "unique_tickers": unique_tickers,
        "total_ticker_days": sum(len(v) for v in (screening_dict or {}).values()),
        "date_range": {
            "start": min(screening_dict.keys()) if screening_dict else None,
            "end": max(screening_dict.keys()) if screening_dict else None,
        },
        "max_unique_tickers": MAX_UNIQUE_TICKERS,
    }


def _draft_payload_from_single(state: dict, *, use_fx: bool, start: str, end: str, pct: float) -> dict:
    return {
        "mode": "single",
        "strategy_prompt": state.get("strategy_prompt") or "",
        "start_date": start,
        "end_date": end,
        "position_size_pct": pct,
        "use_fixtures": use_fx,
        "generated_code": state.get("generated_code") or "",
        "current_config": state.get("current_config") or {},
        "status": "awaiting_decision",
    }


def _draft_payload_from_multi(state: dict, *, use_fx: bool, start: str, end: str, pct: float) -> dict:
    return {
        "mode": "screened",
        "strategy_prompt": state.get("strategy_prompt") or "",
        "screening_prompt": state.get("screening_prompt") or "",
        "start_date": start,
        "end_date": end,
        "position_size_pct": pct,
        "use_fixtures": use_fx,
        "generated_code": state.get("generated_code") or "",
        "current_config": state.get("current_config") or {},
        "screening_code": state.get("screening_code") or "",
        "screening_params": state.get("screening_params") or {},
        "screening_dict": state.get("screening_dict") or {},
        "status": "awaiting_decision",
    }


def _public_generate_response(
    draft_id: str,
    draft: dict,
    *,
    is_multi: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "status": "awaiting_decision",
        "draft_id": draft_id,
        "generated_code": draft.get("generated_code") or "",
        "current_config": draft.get("current_config") or {},
        "period": {
            "start": draft.get("start_date"),
            "end": draft.get("end_date"),
        },
        "position_size_pct": draft.get("position_size_pct"),
        "used_fixtures": bool(draft.get("use_fixtures")),
        "message": (
            "Strategy code is ready. Review it, then either optimize config parameters "
            "or adapt the code with a follow-up prompt."
        ),
    }
    if is_multi:
        out["screening_code"] = draft.get("screening_code") or ""
        out["screening_summary"] = _screening_summary(draft.get("screening_dict") or {})
        out["screening_prompt"] = draft.get("screening_prompt") or ""
        out["strategy_prompt"] = draft.get("strategy_prompt") or ""
    else:
        out["strategy_prompt"] = draft.get("strategy_prompt") or ""
    return out


def generate_single(
    *,
    prompt: str,
    start_date: str,
    end_date: str,
    position_size_pct: float,
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
    state = run_single_generate(inputs)
    if state.get("error"):
        return {
            "error": state["error"],
            "status": "error",
            "generated_code": state.get("generated_code", ""),
            "period": {"start": start_date, "end": end_date},
            "position_size_pct": position_size_pct,
            "used_fixtures": use_fx,
        }
    draft = _draft_payload_from_single(
        state, use_fx=use_fx, start=start_date, end=end_date, pct=position_size_pct
    )
    draft_id = save_draft(draft)
    return _public_generate_response(draft_id, draft, is_multi=False)


def generate_screened(
    *,
    strategy_prompt: str,
    screening_prompt: str,
    start_date: str,
    end_date: str,
    position_size_pct: float,
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
    state = run_multi_generate(inputs)
    if state.get("error"):
        return {
            "error": state["error"],
            "status": "error",
            "generated_code": state.get("generated_code", ""),
            "screening_code": state.get("screening_code", ""),
            "period": {"start": start_date, "end": end_date},
            "position_size_pct": position_size_pct,
            "used_fixtures": use_fx,
        }
    draft = _draft_payload_from_multi(
        state, use_fx=use_fx, start=start_date, end=end_date, pct=position_size_pct
    )
    draft_id = save_draft(draft)
    return _public_generate_response(draft_id, draft, is_multi=True)


def adapt_draft(*, draft_id: str, adapt_instruction: str) -> Dict[str, Any]:
    draft = get_draft(draft_id)
    if not draft:
        return {"error": "Draft not found or expired. Generate the strategy again.", "status": "error"}

    instruction = (adapt_instruction or "").strip()
    if not instruction:
        return {"error": "Adapt instruction is required.", "status": "error"}

    mode = draft.get("mode")
    if mode == "screened":
        state = run_multi_adapt(draft, instruction)
    else:
        state = run_single_adapt(draft, instruction)

    if state.get("error"):
        return {
            "error": state["error"],
            "status": "error",
            "draft_id": draft_id,
            "generated_code": state.get("generated_code") or draft.get("generated_code") or "",
        }

    updates = {
        "strategy_prompt": state.get("strategy_prompt") or draft.get("strategy_prompt"),
        "generated_code": state.get("generated_code") or "",
        "current_config": state.get("current_config") or {},
        "status": "awaiting_decision",
    }
    updated = update_draft(draft_id, **updates)
    return _public_generate_response(draft_id, updated or {**draft, **updates}, is_multi=(mode == "screened"))


def _finalize_optimized(
    db: Session,
    *,
    draft: dict,
    final_state: dict,
    benchmark_ticker: str,
) -> Dict[str, Any]:
    is_multi = draft.get("mode") == "screened"
    start_date = draft.get("start_date") or ""
    end_date = draft.get("end_date") or ""
    position_size_pct = float(draft.get("position_size_pct") or (10.0 if is_multi else 100.0))
    use_fx = bool(draft.get("use_fixtures"))
    code = final_state.get("generated_code") or draft.get("generated_code") or ""

    best = final_state.get("best_config_so_far") or {}
    m = best.get("metrics") or {}

    if is_multi:
        screening_prompt = draft.get("screening_prompt") or ""
        strategy_prompt = final_state.get("strategy_prompt") or draft.get("strategy_prompt") or ""
        user_prompt = f"[SCREENED] {screening_prompt} | {strategy_prompt}"
        explain_src = f"[Screening] {screening_prompt}\n[Strategy] {strategy_prompt}"
    else:
        user_prompt = final_state.get("strategy_prompt") or draft.get("strategy_prompt") or ""
        explain_src = user_prompt
        screening_prompt = ""

    if use_fx:
        explanation = fixture_explanation("multi-stock" if is_multi else "single-stock")
    else:
        explanation = _explain(explain_src, start_date, end_date, best)

    rec = persist_run(
        db,
        user_prompt=user_prompt,
        generated_code=code,
        final_state=final_state,
        start_date=start_date,
        end_date=end_date,
        position_size_pct=position_size_pct,
        is_multi=is_multi,
        screening_prompt=screening_prompt,
        screening_code=final_state.get("screening_code") or draft.get("screening_code") or "",
        explanation=explanation,
    )

    pv = m.get("portfolio_values") or []
    bm_start = (pv[0]["date"] if pv else None) or m.get("period_start") or start_date
    bm_end = (pv[-1]["date"] if pv else None) or m.get("period_end") or end_date

    out: Dict[str, Any] = {
        "status": "complete",
        "draft_id": draft.get("draft_id"),
        "strategy_id": rec.id,
        "best_configuration": best,
        "all_iterations": final_state.get("all_iteration_results", []),
        "generated_code": code,
        "explanation": explanation,
        "benchmark_ticker": benchmark_ticker,
        "benchmark_values": fetch_benchmark(benchmark_ticker, bm_start, bm_end),
        "period": {"start": start_date, "end": end_date},
        "data_period": {"start": bm_start, "end": bm_end},
        "position_size_pct": position_size_pct,
        "used_fixtures": use_fx,
    }
    if is_multi:
        screening_dict = final_state.get("screening_dict") or draft.get("screening_dict") or {}
        out["screening_code"] = final_state.get("screening_code") or draft.get("screening_code") or ""
        out["screening_summary"] = _screening_summary(screening_dict)
    return out


def optimize_draft(
    db: Session,
    *,
    draft_id: str,
    benchmark_ticker: str = "SPY",
) -> Dict[str, Any]:
    draft = get_draft(draft_id)
    if not draft:
        return {"error": "Draft not found or expired. Generate the strategy again.", "status": "error"}

    mode = draft.get("mode")
    if mode == "screened":
        final_state = run_multi_optimize(draft)
    else:
        final_state = run_single_optimize(draft)

    if final_state.get("error"):
        return {
            "error": final_state["error"],
            "status": "error",
            "draft_id": draft_id,
            "generated_code": final_state.get("generated_code") or draft.get("generated_code") or "",
            "screening_code": final_state.get("screening_code") or draft.get("screening_code") or "",
        }

    update_draft(
        draft_id,
        status="complete",
        generated_code=final_state.get("generated_code"),
        current_config=(final_state.get("best_config_so_far") or {}).get("config")
        or final_state.get("current_config"),
        all_iteration_results=final_state.get("all_iteration_results"),
        best_config_so_far=final_state.get("best_config_so_far"),
    )
    return _finalize_optimized(
        db, draft=draft, final_state=final_state, benchmark_ticker=benchmark_ticker
    )


# ── Legacy full-pipeline (generate + optimize in one call) ───────────────────

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
    gen = generate_single(
        prompt=prompt,
        start_date=start_date,
        end_date=end_date,
        position_size_pct=position_size_pct,
        use_fixtures=use_fixtures,
    )
    if gen.get("error"):
        return gen
    return optimize_draft(
        db, draft_id=gen["draft_id"], benchmark_ticker=benchmark_ticker
    )


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
    gen = generate_screened(
        strategy_prompt=strategy_prompt,
        screening_prompt=screening_prompt,
        start_date=start_date,
        end_date=end_date,
        position_size_pct=position_size_pct,
        use_fixtures=use_fixtures,
    )
    if gen.get("error"):
        return gen
    return optimize_draft(
        db, draft_id=gen["draft_id"], benchmark_ticker=benchmark_ticker
    )
