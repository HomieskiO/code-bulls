"""
Fixed screening engine — no LLM-generated Python execution.

Returns screening_dict: {"YYYY-MM-DD": ["AAPL", ...], ...}
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

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


def run_fixed_screener(
    params: Dict[str, Any],
    start_date: str = "",
    end_date: str = "",
    dataset_path: str = KAGGLE_STOCKS_PATH,
    max_unique_tickers: int = MAX_UNIQUE_TICKERS,
    max_tickers_per_day: int = MAX_TICKERS_PER_DAY,
) -> Tuple[Dict[str, List[str]], str]:
    """
    Run the deterministic screener.

    Returns (screening_dict, human_readable_summary_code).
    """
    lookback = int(params.get("lookback_days", 21))
    top_pct = float(params.get("top_pct", 0.01))
    metric = str(params.get("metric", "pct_change"))
    rank_asc = bool(params.get("rank_ascending", False))

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

    summary_lines = [
        f"# Fixed screener (no LLM Python exec)",
        f"# params: lookback_days={lookback}, top_pct={top_pct}, metric={metric},",
        f"#         rank_ascending={rank_asc}",
        f"# period: {start_date or '*'} → {end_date or '*'}",
        f"# files scanned: {len(files)}, loaded series: {len(frames)}",
    ]

    if not frames:
        return {}, "\n".join(summary_lines + ["# result: empty"])

    combined = pd.concat(frames, ignore_index=True)
    gclose = combined.groupby("ticker")["Close"]
    gvol = combined.groupby("ticker")["Volume"]

    if metric == "volume_ratio":
        roll = gvol.transform(
            lambda x: x.rolling(lookback, min_periods=max(2, lookback // 2)).mean()
        )
        combined["metric"] = combined["Volume"] / roll.replace(0, pd.NA)
    elif metric == "abs_change":
        combined["metric"] = gclose.transform(lambda x: x - x.shift(lookback))
    else:
        combined["metric"] = gclose.transform(
            lambda x: x.pct_change(periods=lookback)
        )

    combined = combined.dropna(subset=["metric"]).copy()
    # Highest metric → smallest rank_pct when ascending=False
    combined["rank_pct"] = combined.groupby("Date")["metric"].rank(
        pct=True, ascending=False if not rank_asc else True
    )
    selected = combined[combined["rank_pct"] <= top_pct].copy()
    if start_date:
        selected = selected[selected["Date"] >= start_date]
    if end_date:
        selected = selected[selected["Date"] <= end_date]

    selected["ticker"] = selected["ticker"].astype(str).str.upper()
    selected["date"] = selected["Date"].astype(str).str.slice(0, 10)

    # Cap per-day tickers
    selected = (
        selected.sort_values(["date", "rank_pct"])
        .groupby("date", group_keys=False)
        .head(max_tickers_per_day)
    )

    screening_dict: Dict[str, List[str]] = {}
    for date_str, grp in selected.groupby("date"):
        screening_dict[str(date_str)] = sorted(set(grp["ticker"].tolist()))

    pre_cap_coverage = screening_date_coverage(screening_dict)
    summary_lines.append(f"# pre-cap {pre_cap_coverage}")

    # Time-balanced unique-ticker cap (avoids late-sample-only universes)
    screening_dict, keep, cap_log = cap_screening_time_balanced(
        screening_dict, max_unique_tickers=max_unique_tickers
    )
    summary_lines.append(f"# {cap_log}")
    summary_lines.append(f"# post-cap {screening_date_coverage(screening_dict)}")

    n_tickers = len({t for v in screening_dict.values() for t in v})
    n_pairs = sum(len(v) for v in screening_dict.values())
    summary_lines.append(
        f"# result: {len(screening_dict)} dates, {n_tickers} unique tickers, "
        f"{n_pairs} ticker-day pairs"
    )
    summary_lines.append(
        f"# caps: MAX_UNIQUE_TICKERS={max_unique_tickers}, "
        f"MAX_TICKERS_PER_DAY={max_tickers_per_day}"
    )
    return screening_dict, "\n".join(summary_lines) + "\n"
