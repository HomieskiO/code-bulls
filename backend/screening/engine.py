"""
Screening engine: run generated (or fixed) screeners → screening_dict.

Public contract consumed by strategies / multi backtest:
  screening_dict: {"YYYY-MM-DD": ["AAPL", ...], ...}

CSV contract produced by generated build_screening_csv(...):
  columns: date,ticker
"""
from __future__ import annotations

import math
import os
import re
import tempfile
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

KAGGLE_STOCKS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Stock Market Dataset", "Stocks")
)

# Hard cap multi-stock universe size (unique tickers loaded into cerebro)
MAX_UNIQUE_TICKERS = int(os.getenv("MAX_UNIQUE_TICKERS", "400"))
MAX_TICKERS_PER_DAY = int(os.getenv("MAX_TICKERS_PER_DAY", "50"))


def _load_one(
    args: Tuple[str, str, Optional[str], Optional[str], int],
) -> Optional[pd.DataFrame]:
    path, ticker, start_date, end_date, lookback_days = args
    try:
        df = pd.read_csv(path, usecols=["Date", "Close", "Volume"])
        lookback_start = start_date
        if start_date:
            d0 = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(
                days=lookback_days + 60
            )
            lookback_start = d0.strftime("%Y-%m-%d")
        if lookback_start:
            df = df[df["Date"] >= lookback_start]
        if end_date:
            df = df[df["Date"] <= end_date]
        if df is None or df.empty or len(df) < lookback_days + 1:
            return None
        df = df.copy()
        df["ticker"] = str(ticker).upper()
        # Drop absurd prices (reverse-split corruption)
        med = float(df["Close"].median())
        if med <= 0 or med > 10_000 or med < 0.05:
            return None
        if float(df["Close"].max()) > 500_000:
            return None
        return df
    except Exception:
        return None


def cap_screening_time_balanced(
    screening_dict: Dict[str, List[str]],
    max_unique_tickers: int = MAX_UNIQUE_TICKERS,
) -> Tuple[Dict[str, List[str]], Set[str], str]:
    """
    Cap the unique-ticker universe in a time-balanced way.

    Instead of keeping only globally most-frequent names (which biases toward
    late-sample churn), take the top names **per calendar year**, then union.
    That preserves names that led the screen in 2010–2016, not only 2017.

    Returns (filtered_dict, keep_set, log_line).
    """
    if not screening_dict:
        return {}, set(), "empty screening_dict"

    # Frequency within each year
    by_year: Dict[str, Dict[str, int]] = {}
    for date_str, tickers in screening_dict.items():
        year = str(date_str)[:4]
        bucket = by_year.setdefault(year, {})
        for t in tickers or []:
            sym = str(t).strip().upper()
            if sym:
                bucket[sym] = bucket.get(sym, 0) + 1

    years = sorted(by_year.keys())
    n_years = max(1, len(years))
    # Share budget across years; allow overlap so union stays near max_unique
    per_year = max(15, (max_unique_tickers + n_years - 1) // n_years)

    keep: Set[str] = set()
    year_picks: Dict[str, int] = {}
    for y in years:
        ranked = sorted(by_year[y].items(), key=lambda kv: (-kv[1], kv[0]))
        picked = [t for t, _ in ranked[:per_year]]
        year_picks[y] = len(picked)
        keep.update(picked)

    # If union still too large (rare), trim by total frequency but keep at least
    # a few names from each year.
    if len(keep) > max_unique_tickers:
        global_freq: Dict[str, int] = {}
        for tickers in screening_dict.values():
            for t in tickers or []:
                sym = str(t).strip().upper()
                if sym in keep:
                    global_freq[sym] = global_freq.get(sym, 0) + 1
        # Guarantee min floor per year
        floor = max(5, per_year // 3)
        guaranteed: Set[str] = set()
        for y in years:
            ranked = sorted(by_year[y].items(), key=lambda kv: (-kv[1], kv[0]))
            guaranteed.update(t for t, _ in ranked[:floor] if t in keep)
        rest_budget = max(0, max_unique_tickers - len(guaranteed))
        rest = [
            t
            for t, _ in sorted(global_freq.items(), key=lambda kv: (-kv[1], kv[0]))
            if t not in guaranteed
        ][:rest_budget]
        keep = guaranteed | set(rest)

    filtered = {
        d: [t for t in ts if t in keep]
        for d, ts in screening_dict.items()
    }
    filtered = {d: ts for d, ts in filtered.items() if ts}

    # Coverage diagnostics
    years_with_days = sorted({d[:4] for d in filtered})
    pairs = sum(len(v) for v in filtered.values())
    log = (
        f"time-balanced cap: keep={len(keep)} unique "
        f"(per_year_budget={per_year}, years={years_with_days}), "
        f"dates_left={len(filtered)}, pairs={pairs}, year_picks={year_picks}"
    )
    return filtered, keep, log


def screening_date_coverage(screening_dict: Dict[str, List[str]]) -> str:
    if not screening_dict:
        return "coverage: empty"
    dates = sorted(screening_dict.keys())
    by_year: Dict[str, int] = {}
    for d, ts in screening_dict.items():
        y = d[:4]
        by_year[y] = by_year.get(y, 0) + len(ts)
    return (
        f"coverage: {dates[0]} → {dates[-1]} "
        f"({len(dates)} dates); pairs_by_year={dict(sorted(by_year.items()))}"
    )


def load_screening_csv(csv_path: str) -> Dict[str, List[str]]:
    """Load date,ticker CSV → screening_dict (UPPERCASE tickers, YYYY-MM-DD dates)."""
    df = pd.read_csv(csv_path)
    if df is None or df.empty:
        return {}
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    date_col = lower_map.get("date")
    ticker_col = lower_map.get("ticker") or lower_map.get("symbol")
    if date_col is None or ticker_col is None:
        raise ValueError(
            f"Screening CSV must have date,ticker columns; got {list(df.columns)}"
        )
    out: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        d = str(row[date_col])[:10]
        t = str(row[ticker_col]).strip().upper()
        if not d or not t or t == "NAN" or d.lower() == "nan":
            continue
        bucket = out.setdefault(d, [])
        if t not in bucket:
            bucket.append(t)
    for d in list(out.keys()):
        out[d] = sorted(out[d])
    return out


def cap_screening_per_day(
    screening_dict: Dict[str, List[str]],
    max_tickers_per_day: int = MAX_TICKERS_PER_DAY,
) -> Dict[str, List[str]]:
    """Keep at most N tickers per date (stable alphabetical if no rank)."""
    if max_tickers_per_day <= 0:
        return screening_dict
    return {
        d: sorted(ts)[:max_tickers_per_day]
        for d, ts in screening_dict.items()
        if ts
    }


def finalize_screening_dict(
    screening_dict: Dict[str, List[str]],
    *,
    max_unique_tickers: int = MAX_UNIQUE_TICKERS,
    max_tickers_per_day: int = MAX_TICKERS_PER_DAY,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Post-process any screening_dict (from CSV or fixed engine):
    per-day cap + time-balanced unique-ticker cap.
    Returns (final_dict, log_lines).
    """
    logs: List[str] = []
    if not screening_dict:
        return {}, ["result: empty"]

    logs.append(f"pre-cap {screening_date_coverage(screening_dict)}")
    screening_dict = cap_screening_per_day(
        screening_dict, max_tickers_per_day=max_tickers_per_day
    )
    screening_dict, _keep, cap_log = cap_screening_time_balanced(
        screening_dict, max_unique_tickers=max_unique_tickers
    )
    logs.append(cap_log)
    logs.append(f"post-cap {screening_date_coverage(screening_dict)}")
    n_tickers = len({t for v in screening_dict.values() for t in v})
    n_pairs = sum(len(v) for v in screening_dict.values())
    logs.append(
        f"result: {len(screening_dict)} dates, {n_tickers} unique tickers, "
        f"{n_pairs} ticker-day pairs"
    )
    logs.append(
        f"caps: MAX_UNIQUE_TICKERS={max_unique_tickers}, "
        f"MAX_TICKERS_PER_DAY={max_tickers_per_day}"
    )
    return screening_dict, logs


def _screening_exec_namespace() -> dict:
    """Safe-ish namespace for generated screening scripts (no builtins.import)."""
    import builtins as _bi

    safe_builtins = {
        k: getattr(_bi, k)
        for k in (
            "abs",
            "all",
            "any",
            "bool",
            "dict",
            "enumerate",
            "float",
            "int",
            "len",
            "list",
            "max",
            "min",
            "print",
            "range",
            "round",
            "set",
            "sorted",
            "str",
            "sum",
            "tuple",
            "zip",
            "True",
            "False",
            "None",
            "Exception",
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
        )
        if hasattr(_bi, k)
    }
    # True/False/None are not always in builtins as names on all versions
    safe_builtins["True"] = True
    safe_builtins["False"] = False
    safe_builtins["None"] = None

    return {
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np,
        "os": os,
        "Path": Path,
        "datetime": datetime,
        "timedelta": timedelta,
        "math": math,
        "re": re,
        "defaultdict": defaultdict,
    }


def execute_build_screening_csv(
    code: str,
    *,
    stocks_dir: str = KAGGLE_STOCKS_PATH,
    start_date: str,
    end_date: str,
    out_csv: str,
    timeout_sec: int = 600,
) -> None:
    """
    Exec generated code and call build_screening_csv(...).
    Raises ValueError on failure.
    """
    ns = _screening_exec_namespace()
    try:
        exec(code, ns, ns)  # noqa: S102 — intentional, validated screening scripts
    except Exception as e:
        raise ValueError(f"Screening code failed to load: {e}") from e

    fn = ns.get("build_screening_csv")
    if not callable(fn):
        raise ValueError("Screening code must define callable build_screening_csv(...)")

    def _call():
        fn(stocks_dir, start_date, end_date, out_csv)

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        try:
            fut.result(timeout=timeout_sec)
        except FuturesTimeout as e:
            raise ValueError(
                f"Screening code timed out after {timeout_sec}s"
            ) from e
        except Exception as e:
            raise ValueError(f"build_screening_csv failed: {e}") from e

    if not os.path.isfile(out_csv):
        raise ValueError(f"Screening code did not create CSV at {out_csv}")


def run_generated_screener(
    code: str,
    *,
    start_date: str = "",
    end_date: str = "",
    dataset_path: str = KAGGLE_STOCKS_PATH,
    max_unique_tickers: int = MAX_UNIQUE_TICKERS,
    max_tickers_per_day: int = MAX_TICKERS_PER_DAY,
    timeout_sec: int = 600,
    out_dir: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], str, str]:
    """
    Run LLM-generated screening Python → CSV → screening_dict (+ caps).

    Returns (screening_dict, screening_code_for_ui, csv_path).
    screening_code_for_ui = original code + run summary comments.
    """
    start_date = (start_date or "").strip()
    end_date = (end_date or "").strip()
    if not start_date or not end_date:
        raise ValueError("start_date and end_date are required for screening")

    out_dir = out_dir or os.path.join(
        tempfile.gettempdir(), "code_bulls_screening"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"screen_{uuid.uuid4().hex[:12]}.csv")

    execute_build_screening_csv(
        code,
        stocks_dir=dataset_path,
        start_date=start_date,
        end_date=end_date,
        out_csv=csv_path,
        timeout_sec=timeout_sec,
    )

    raw = load_screening_csv(csv_path)
    final, logs = finalize_screening_dict(
        raw,
        max_unique_tickers=max_unique_tickers,
        max_tickers_per_day=max_tickers_per_day,
    )

    footer = "\n\n# ── Run summary ─────────────────────────────────────────\n"
    footer += f"# csv: {csv_path}\n"
    footer += f"# period: {start_date} → {end_date}\n"
    footer += f"# dataset: {dataset_path}\n"
    for line in logs:
        footer += f"# {line}\n"

    ui_code = (code or "").rstrip() + footer
    return final, ui_code, csv_path


def _normalize_run_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp engine params (generic — no rule-specific defaults)."""
    lookback = int(params.get("lookback_days", 21))
    top_pct = float(params.get("top_pct", 0.01))
    metric = str(params.get("metric", "pct_change")).strip().lower()
    rank_asc = bool(params.get("rank_ascending", False))
    min_raw = params.get("min_metric", None)
    try:
        min_metric = None if min_raw is None or min_raw == "" else float(min_raw)
    except (TypeError, ValueError):
        min_metric = None
    if metric not in ("pct_change", "abs_change", "volume_ratio", "near_high"):
        metric = "pct_change"
    return {
        "lookback_days": max(2, min(252, lookback)),
        "top_pct": max(0.001, min(1.0, top_pct)),
        "metric": metric,
        "rank_ascending": rank_asc,
        "min_metric": min_metric,
    }


def render_screening_python(
    params: Dict[str, Any],
    *,
    user_rule: str = "",
    start_date: str = "",
    end_date: str = "",
    run_stats: Optional[List[str]] = None,
    max_tickers_per_day: int = MAX_TICKERS_PER_DAY,
) -> str:
    """
    Generate screening Python from params that were derived from the screening prompt.

    Generic template: only PARAMS and metric branch change with the prompt.
    Not hardcoded to any single user rule (volume, gainers, etc.).
    """
    p = _normalize_run_params(params or {})
    lookback = p["lookback_days"]
    top_pct = p["top_pct"]
    metric = p["metric"]
    rank_asc = p["rank_ascending"]
    min_metric = p["min_metric"]
    min_repr = "None" if min_metric is None else repr(min_metric)
    rule_line = (user_rule or "").replace("\n", " ").strip() or "(from screening prompt)"
    rank_dir = "lowest" if rank_asc else "highest"
    min_periods = max(2, lookback // 2)

    # Metric body — selected by params["metric"] only
    metric_bodies = {
        "pct_change": f"return close.pct_change(periods={lookback})",
        "abs_change": f"return close - close.shift({lookback})",
        "volume_ratio": (
            f"avg = volume.rolling({lookback}, min_periods={min_periods}).mean()\n"
            f"    return volume / avg.replace(0, float('nan'))"
        ),
        "near_high": (
            f"hi = close.rolling({lookback}, min_periods={min_periods}).max()\n"
            f"    return close / hi.replace(0, float('nan'))"
        ),
    }
    metric_body = metric_bodies[metric]

    filter_block: List[str] = []
    if top_pct < 1.0:
        filter_block.append(
            f"    # Rank filter: keep the {rank_dir} {top_pct * 100:g}% each day"
        )
        filter_block.append(f"    keep = keep[keep['rank_pct'] <= {top_pct}]")
    else:
        filter_block.append("    # top_pct=1.0 → no percentile rank cut")
    if min_metric is not None:
        op = "<=" if rank_asc else ">="
        filter_block.append(
            f"    # Threshold filter from screening prompt (min_metric={min_metric})"
        )
        filter_block.append(f"    keep = keep[keep['metric'] {op} {min_metric}]")
    if len(filter_block) == 1 and "no percentile" in filter_block[0]:
        filter_block.append("    # (all names that pass other filters are eligible)")

    stats_block = ""
    if run_stats:
        lines = [f"# {s.lstrip('# ').rstrip()}" for s in run_stats if str(s).strip()]
        stats_block = "\n\n# ── Run summary ─────────────────────────────────────────\n" + "\n".join(lines) + "\n"

    return f'''"""
Daily universe screener generated from the screening prompt.

User screening prompt:
  {rule_line}

Period: {start_date or "*"} → {end_date or "*"}

Params below are produced from that prompt (LLM or heuristic), then this
module is filled in. The same PARAMS drive the live screener engine.
"""

PARAMS = {{
    "lookback_days": {lookback},
    "top_pct": {top_pct},
    "metric": {metric!r},
    "rank_ascending": {rank_asc},
    "min_metric": {min_repr},
}}


def compute_metric(close, volume):
    """Score series for one ticker. Branch chosen by PARAMS['metric']."""
    # active metric for this prompt: {metric!r}
    {metric_body}


def select_universe(panel):
    """
    panel: DataFrame[Date, ticker, Close, Volume]
    returns: {{"YYYY-MM-DD": ["AAPL", ...], ...}}
    """
    lookback = PARAMS["lookback_days"]
    top_pct = PARAMS["top_pct"]
    rank_ascending = PARAMS["rank_ascending"]
    min_metric = PARAMS["min_metric"]

    panel = panel.copy()
    panel["metric"] = (
        panel.groupby("ticker", group_keys=False)
        .apply(lambda g: compute_metric(g["Close"], g["Volume"]))
        .reset_index(level=0, drop=True)
    )
    panel = panel.dropna(subset=["metric"])

    # rank within each day (ascending=False → highest metric ranks best)
    panel["rank_pct"] = panel.groupby("Date")["metric"].rank(
        pct=True, ascending=rank_ascending
    )

    keep = panel
{chr(10).join(filter_block)}

    keep = (
        keep.sort_values(["Date", "rank_pct"])
        .groupby("Date", group_keys=False)
        .head({int(max_tickers_per_day)})
    )

    screening = {{}}
    for day, grp in keep.groupby("Date"):
        screening[str(day)[:10]] = sorted(set(grp["ticker"].astype(str).str.upper()))
    return screening
{stats_block}'''


def run_fixed_screener(
    params: Dict[str, Any],
    start_date: str = "",
    end_date: str = "",
    dataset_path: str = KAGGLE_STOCKS_PATH,
    max_unique_tickers: int = MAX_UNIQUE_TICKERS,
    max_tickers_per_day: int = MAX_TICKERS_PER_DAY,
    user_rule: str = "",
) -> Tuple[Dict[str, List[str]], str]:
    """
    Run the deterministic screener using params derived from the screening prompt.

    Returns (screening_dict, generated screening Python for the Code tab).
    """
    p = _normalize_run_params(params or {})
    lookback = p["lookback_days"]
    top_pct = p["top_pct"]
    metric = p["metric"]
    rank_asc = p["rank_ascending"]
    min_metric = p["min_metric"]

    start_date = (start_date or "").strip()
    end_date = (end_date or "").strip()

    files = []
    for fn in os.listdir(dataset_path):
        if not fn.endswith(".us.txt"):
            continue
        ticker = fn[:-7].upper()
        files.append(
            (
                os.path.join(dataset_path, fn),
                ticker,
                start_date or None,
                end_date or None,
                lookback,
            )
        )

    frames: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for df in ex.map(_load_one, files):
            if df is not None:
                frames.append(df)

    run_stats = [
        f"files scanned: {len(files)}, loaded series: {len(frames)}",
    ]

    if not frames:
        code = render_screening_python(
            p,
            user_rule=user_rule,
            start_date=start_date,
            end_date=end_date,
            run_stats=run_stats + ["result: empty"],
            max_tickers_per_day=max_tickers_per_day,
        )
        return {}, code

    combined = pd.concat(frames, ignore_index=True)
    gclose = combined.groupby("ticker")["Close"]
    gvol = combined.groupby("ticker")["Volume"]
    min_periods = max(2, min(lookback, max(2, lookback // 2)))

    if metric == "volume_ratio":
        roll = gvol.transform(
            lambda x: x.rolling(lookback, min_periods=min_periods).mean()
        )
        combined["metric"] = combined["Volume"] / roll.replace(0, pd.NA)
    elif metric == "abs_change":
        combined["metric"] = gclose.transform(lambda x: x - x.shift(lookback))
    elif metric == "near_high":
        roll_hi = gclose.transform(
            lambda x: x.rolling(lookback, min_periods=min_periods).max()
        )
        combined["metric"] = combined["Close"] / roll_hi.replace(0, pd.NA)
    else:
        combined["metric"] = gclose.transform(
            lambda x: x.pct_change(periods=lookback)
        )

    combined = combined.dropna(subset=["metric"]).copy()
    combined["rank_pct"] = combined.groupby("Date")["metric"].rank(
        pct=True, ascending=False if not rank_asc else True
    )

    mask = pd.Series(True, index=combined.index)
    if top_pct < 1.0:
        mask &= combined["rank_pct"] <= top_pct
    if min_metric is not None:
        if rank_asc:
            mask &= combined["metric"] <= float(min_metric)
        else:
            mask &= combined["metric"] >= float(min_metric)

    selected = combined[mask].copy()
    if start_date:
        selected = selected[selected["Date"] >= start_date]
    if end_date:
        selected = selected[selected["Date"] <= end_date]

    selected["ticker"] = selected["ticker"].astype(str).str.upper()
    selected["date"] = selected["Date"].astype(str).str.slice(0, 10)

    selected = (
        selected.sort_values(["date", "rank_pct"])
        .groupby("date", group_keys=False)
        .head(max_tickers_per_day)
    )

    screening_dict: Dict[str, List[str]] = {}
    for date_str, grp in selected.groupby("date"):
        screening_dict[str(date_str)] = sorted(set(grp["ticker"].tolist()))

    run_stats.append(f"pre-cap {screening_date_coverage(screening_dict)}")

    screening_dict, keep, cap_log = cap_screening_time_balanced(
        screening_dict, max_unique_tickers=max_unique_tickers
    )
    run_stats.append(cap_log)
    run_stats.append(f"post-cap {screening_date_coverage(screening_dict)}")

    n_tickers = len({t for v in screening_dict.values() for t in v})
    n_pairs = sum(len(v) for v in screening_dict.values())
    run_stats.append(
        f"result: {len(screening_dict)} dates, {n_tickers} unique tickers, "
        f"{n_pairs} ticker-day pairs"
    )
    run_stats.append(
        f"caps: MAX_UNIQUE_TICKERS={max_unique_tickers}, "
        f"MAX_TICKERS_PER_DAY={max_tickers_per_day}"
    )

    code = render_screening_python(
        p,
        user_rule=user_rule,
        start_date=start_date,
        end_date=end_date,
        run_stats=run_stats,
        max_tickers_per_day=max_tickers_per_day,
    )
    return screening_dict, code
