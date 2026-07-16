"""
Fixed screening engine — no LLM-generated Python execution.

Returns screening_dict: {"YYYY-MM-DD": ["AAPL", ...], ...}
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

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

    combined = combined.dropna(subset=["metric"])
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

    # Cap unique tickers by frequency (keep most frequently screened names)
    freq: Dict[str, int] = {}
    for tickers in screening_dict.values():
        for t in tickers:
            freq[t] = freq.get(t, 0) + 1
    if len(freq) > max_unique_tickers:
        keep = {
            t
            for t, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[
                :max_unique_tickers
            ]
        }
        screening_dict = {
            d: [t for t in ts if t in keep]
            for d, ts in screening_dict.items()
        }
        screening_dict = {d: ts for d, ts in screening_dict.items() if ts}

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
