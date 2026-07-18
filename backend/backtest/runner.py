"""Single- and multi-ticker backtest execution (shared cerebro builder)."""
from __future__ import annotations

import datetime as dt_module
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Type

import backtrader as bt
import pandas as pd
import yfinance as yf

from codegen.extract import (
    ensure_stake_pct_in_config,
    load_strategy_class,
    position_size_pct,
    strategy_param_names,
)
from screening.engine import (
    KAGGLE_STOCKS_PATH,
    MAX_UNIQUE_TICKERS,
    cap_screening_time_balanced,
    screening_date_coverage,
)
from .cerebro_builder import build_cerebro, config_score, run_and_metrics

# Dataset roots (SPY lives under ETFs/, equities under Stocks/)
_KAGGLE_ETFS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(KAGGLE_STOCKS_PATH), "ETFs")
)

# Master clock candidates for multi-stock Cerebro (data0).
# SPY is preferred when it covers the requested start; this Kaggle dump's SPY
# only begins 2005-02-25, so earlier periods fall back to long equities (AAPL…).
# Not required to be in the screening list; strategy only buys when screened.
CLOCK_CANDIDATES = ("SPY", "AAPL", "MSFT", "GE", "IBM")


def get_ticker_from_prompt(prompt: str) -> str:
    match = re.search(r"\b([A-Z]{1,5})\b", prompt)
    if match:
        return match.group(1)
    raise ValueError("Could not extract a valid stock ticker from the prompt.")


def flatten_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if "adj close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj close": "close"})
    return df[["open", "high", "low", "close", "volume"]].copy()


def _strategy_kwargs(StrategyClass: Type[bt.Strategy], config: dict) -> dict:
    kwargs = dict(config)
    names = strategy_param_names(StrategyClass)
    if names and "stake_pct" not in names:
        kwargs.pop("stake_pct", None)
    return kwargs


def run_single_backtest(
    *,
    strategy_code: str,
    config: dict,
    strategy_prompt: str,
    start_date: str,
    end_date: str,
    position_size_pct_val: float = 100.0,
) -> dict:
    """Run one single-ticker backtest; returns metrics dict."""
    ticker = get_ticker_from_prompt(strategy_prompt)
    start = start_date or "2010-01-01"
    end = end_date or "2017-11-10"
    end_exclusive = (
        dt_module.datetime.strptime(end, "%Y-%m-%d") + dt_module.timedelta(days=1)
    ).strftime("%Y-%m-%d")

    raw = yf.download(
        ticker, start=start, end=end_exclusive, auto_adjust=True, progress=False
    )
    if raw.empty:
        raise ValueError(f"No data returned by yfinance for '{ticker}' between {start} and {end}.")

    data_df = flatten_yfinance_df(raw)
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.loc[
        (data_df.index >= pd.Timestamp(start)) & (data_df.index <= pd.Timestamp(end))
    ]
    if data_df.empty:
        raise ValueError(f"No bars for '{ticker}' within {start} → {end}.")

    from_dt = dt_module.datetime.strptime(start, "%Y-%m-%d")
    to_dt = dt_module.datetime.strptime(end, "%Y-%m-%d")
    feed = bt.feeds.PandasData(
        dataname=data_df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
        fromdate=from_dt,
        todate=to_dt,
    )

    StrategyClass = load_strategy_class(strategy_code)
    pct = max(1.0, min(100.0, float(position_size_pct_val)))
    config = ensure_stake_pct_in_config(config, pct)
    kwargs = _strategy_kwargs(StrategyClass, config)

    cerebro = build_cerebro(
        strategy_cls=StrategyClass,
        strategy_kwargs=kwargs,
        feeds=[feed],
        stake_pct=pct,
        multi=False,
    )
    print(f"  Single backtest {ticker} {start}→{end}, stake={pct:g}%")
    return run_and_metrics(cerebro, period_start=start, period_end=end, multi=False)


def _fast_date_parse(date_string: str) -> dt_module.datetime:
    return dt_module.datetime(
        int(date_string[0:4]),
        int(date_string[5:7]),
        int(date_string[8:10]),
    )


def _has_enough_data(filepath: str, min_bytes: int = 6_000) -> bool:
    return os.path.getsize(filepath) >= min_bytes


def _resolve_csv_path(ticker: str) -> Optional[str]:
    """Locate ticker CSV under Stocks/ then ETFs/ (SPY is an ETF in this dataset)."""
    name = f"{ticker.lower()}.us.txt"
    for root in (KAGGLE_STOCKS_PATH, _KAGGLE_ETFS_PATH):
        path = os.path.join(root, name)
        if os.path.exists(path):
            return path
    return None


def _csv_first_date(filepath: str) -> Optional[str]:
    """Return first YYYY-MM-DD in a Kaggle CSV, or None."""
    try:
        with open(filepath, "r", errors="ignore") as fh:
            fh.readline()
            for line in fh:
                if not line.strip():
                    continue
                return line.split(",", 1)[0].strip()[:10]
    except Exception:
        return None
    return None


# Strategies often use SMA(50) / Highest(252) — need enough bars in window
_MIN_BARS_FOR_INDICATORS = int(os.getenv("MIN_BARS_FOR_INDICATORS", "280"))


def _price_series_sane(
    filepath: str,
    from_dt,
    to_dt,
    *,
    min_bars: int = _MIN_BARS_FOR_INDICATORS,
) -> bool:
    """
    Reject corrupt prices and series too short for long indicators
    (e.g. Highest period=252 → empty max() crash in runonce mode).
    """
    try:
        closes = []
        with open(filepath, "r", errors="ignore") as fh:
            fh.readline()
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) < 5:
                    continue
                ds = parts[0]
                try:
                    y, m, d = int(ds[0:4]), int(ds[5:7]), int(ds[8:10])
                except Exception:
                    continue
                if from_dt and (y, m, d) < (from_dt.year, from_dt.month, from_dt.day):
                    continue
                if to_dt and (y, m, d) > (to_dt.year, to_dt.month, to_dt.day):
                    continue
                try:
                    closes.append(float(parts[4]))
                except Exception:
                    continue
        if len(closes) < min_bars:
            return False
        sample = closes[:: max(1, len(closes) // 400)][:400]
        sample_sorted = sorted(sample)
        med = sample_sorted[len(sample_sorted) // 2]
        if med <= 0 or med > 10_000 or med < 0.05:
            return False
        if max(closes) > 500_000:
            return False
        return True
    except Exception:
        return False


def _ticker_eligible_for_multi(
    ticker: str,
    *,
    from_dt: dt_module.datetime,
    to_dt: dt_module.datetime,
    allow_etf: bool = False,
) -> Tuple[bool, str]:
    """Whether a ticker has usable Kaggle price history in the window."""
    if allow_etf:
        filepath = _resolve_csv_path(ticker)
    else:
        filepath = os.path.join(KAGGLE_STOCKS_PATH, f"{ticker.lower()}.us.txt")
        if not os.path.exists(filepath):
            filepath = None
    if not filepath or not _has_enough_data(filepath):
        return False, "missing_or_tiny"
    if not _csv_first_date(filepath):
        return False, "no_dates"
    if not _price_series_sane(filepath, from_dt, to_dt):
        return False, "bad_or_short"
    return True, "ok"


def _load_ohlcv_df(
    filepath: str,
    from_dt: dt_module.datetime,
    to_dt: dt_module.datetime,
) -> Optional[pd.DataFrame]:
    """Load Kaggle OHLCV CSV into a DatetimeIndex DataFrame (unsorted-safe)."""
    try:
        df = pd.read_csv(
            filepath,
            usecols=["Date", "Open", "High", "Low", "Close", "Volume"],
            dtype={
                "Open": "float64",
                "High": "float64",
                "Low": "float64",
                "Close": "float64",
                "Volume": "float64",
            },
        )
        if df is None or df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"])
        df = df.set_index("Date").sort_index()
        df = df.loc[(df.index >= pd.Timestamp(from_dt)) & (df.index <= pd.Timestamp(to_dt))]
        if df.empty:
            return None
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as ex:
        print(f"  WARNING: could not parse {filepath}: {ex}")
        return None


def _align_to_calendar(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Reindex a stock onto the master session calendar.

    Backtrader multi-data only calls strategy.next() once every feed has a first
    bar, so late IPOs would otherwise delay the whole run. Aligning onto the
    clock calendar gives every feed the same start/end.

    Inactive days (pre-IPO / gaps / post-delist):
      - volume = 0  (strategy must skip these; also used to force-exit)
      - OHLC = 0 before first print (never NaN — NaN poisons broker.getvalue)
      - OHLC ffilled after first print so open positions can be marked/closed
    """
    aligned = df.reindex(calendar)
    had_bar = aligned["close"].notna()
    aligned["volume"] = aligned["volume"].where(had_bar, 0.0).fillna(0.0)
    ohlc = ["open", "high", "low", "close"]
    # Carry last print forward (delist / gaps); zeros only before first print
    aligned[ohlc] = aligned[ohlc].ffill().fillna(0.0)
    return aligned


def _make_pandas_feed(
    df: pd.DataFrame,
    name: str,
    from_dt: dt_module.datetime,
    to_dt: dt_module.datetime,
) -> bt.feeds.PandasData:
    return bt.feeds.PandasData(
        dataname=df,
        name=name,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
        fromdate=from_dt,
        todate=to_dt,
    )


def _make_csv_feed(
    ticker: str,
    from_dt: dt_module.datetime,
    to_dt: dt_module.datetime,
    *,
    name: Optional[str] = None,
    allow_etf: bool = False,
    calendar: Optional[pd.DatetimeIndex] = None,
) -> Optional[bt.feeds.PandasData]:
    """Load one Kaggle feed as PandasData, optionally calendar-aligned.

    ``allow_etf=True`` also searches ETFs/ (needed for SPY clock).
    When ``calendar`` is provided, the series is reindexed to it (pad volume=0).
    """
    if allow_etf:
        filepath = _resolve_csv_path(ticker)
    else:
        filepath = os.path.join(KAGGLE_STOCKS_PATH, f"{ticker.lower()}.us.txt")
        if not os.path.exists(filepath):
            filepath = None
    if not filepath or not _has_enough_data(filepath):
        return None
    if not _price_series_sane(filepath, from_dt, to_dt):
        return None
    df = _load_ohlcv_df(filepath, from_dt, to_dt)
    if df is None or df.empty:
        return None
    if calendar is not None:
        df = _align_to_calendar(df, calendar)
    feed_name = name or ticker
    try:
        return _make_pandas_feed(df, feed_name, from_dt, to_dt)
    except Exception as ex:
        print(f"  WARNING: could not load {ticker}: {ex}")
        return None


def _peek_csv_date_range(ticker: str) -> str:
    """First/last date strings from a stock/ETF file (for logging)."""
    filepath = _resolve_csv_path(ticker)
    if not filepath:
        return "missing"
    try:
        with open(filepath, "r", errors="ignore") as fh:
            fh.readline()
            first = None
            last = None
            for line in fh:
                if not line.strip():
                    continue
                d = line.split(",", 1)[0]
                if first is None:
                    first = d
                last = d
        return f"{first} → {last}"
    except Exception:
        return "unknown"


def _select_clock_ticker(period_start: str) -> Optional[str]:
    """
    Choose data0 so the multi-data calendar covers the requested start date.

    Prefer SPY when its file starts on/before period_start; otherwise the first
    candidate that does. Without this, a late SPY series (e.g. 2005+) silently
    truncates a 1998→… backtest to mid-2000s.
    """
    covering: List[str] = []
    fallback: List[str] = []
    for cand in CLOCK_CANDIDATES:
        path = _resolve_csv_path(cand)
        if not path or not _has_enough_data(path):
            continue
        first = _csv_first_date(path)
        if not first:
            continue
        fallback.append(cand)
        if first <= period_start:
            covering.append(cand)
    if covering:
        # Prefer SPY among those that actually cover the window
        if "SPY" in covering:
            return "SPY"
        return covering[0]
    return fallback[0] if fallback else None


def run_multi_backtest_core(
    *,
    strategy_code: str,
    config: dict,
    screening_dict: Dict[str, List[str]],
    start_date: str,
    end_date: str,
    position_size_pct_val: float = 10.0,
    max_unique_tickers: int = MAX_UNIQUE_TICKERS,
) -> dict:
    """Run multi-ticker backtest with screening dict; returns metrics."""
    period_start = start_date or "2010-01-01"
    period_end = end_date or "2017-11-10"
    period_from = dt_module.datetime.strptime(period_start, "%Y-%m-%d")
    from_dt = period_from - dt_module.timedelta(days=100)
    to_dt = dt_module.datetime.strptime(period_end, "%Y-%m-%d")

    # Normalize screening
    norm: Dict[str, List[str]] = {}
    for raw_date, tickers in (screening_dict or {}).items():
        dkey = str(raw_date).strip()[:10]
        cleaned = sorted(
            {
                str(t).strip().upper().replace(".US", "")
                for t in (tickers or [])
                if str(t).strip()
            }
        )
        if cleaned:
            norm[dkey] = cleaned
    screening_dict = norm

    print(f"  Pre-cap {screening_date_coverage(screening_dict)}")

    # Drop unusable series (missing / corrupt / too short) before the universe cap.
    # Late IPOs are KEPT — they are calendar-aligned onto the clock so they do not
    # delay strategy.next() (backtrader otherwise waits for every feed's first bar).
    raw_unique = {t for ts in screening_dict.values() for t in ts}
    eligible: set = set()
    skip_reasons: Dict[str, int] = {}
    for t in raw_unique:
        ok, reason = _ticker_eligible_for_multi(t, from_dt=from_dt, to_dt=to_dt)
        if ok:
            eligible.add(t)
        else:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    if not eligible:
        raise ValueError(
            "No screened tickers have usable price history in the requested window."
        )
    n_dropped = len(raw_unique) - len(eligible)
    if n_dropped:
        print(
            f"  Pre-cap eligibility: keep={len(eligible)}/{len(raw_unique)} "
            f"(dropped={n_dropped}, reasons={skip_reasons})"
        )
    screening_dict = {
        d: [t for t in ts if t in eligible]
        for d, ts in screening_dict.items()
    }
    screening_dict = {d: ts for d, ts in screening_dict.items() if ts}
    print(f"  Post-eligibility {screening_date_coverage(screening_dict)}")

    # Time-balanced universe cap (per-year top names) — avoids 2017-only bias
    n_unique = len({t for ts in screening_dict.values() for t in ts})
    if n_unique > max_unique_tickers:
        screening_dict, keep, cap_log = cap_screening_time_balanced(
            screening_dict, max_unique_tickers=max_unique_tickers
        )
        print(f"  {cap_log}")
    else:
        keep = {t for ts in screening_dict.values() for t in ts}
        print(f"  Universe under cap ({n_unique} unique) — no ticker trim")

    print(f"  Post-cap {screening_date_coverage(screening_dict)}")

    all_tickers = sorted({t for ts in screening_dict.values() for t in ts})
    if not all_tickers:
        raise ValueError("Screening dict is empty — no tickers to trade.")

    print(
        f"  Multi universe: {len(all_tickers)} tickers, "
        f"{len(screening_dict)} dates, "
        f"{sum(len(v) for v in screening_dict.values())} pairs"
    )

    feeds: List[Tuple[Any, str]] = []
    skipped_invalid = skipped_bad = 0

    # --- Master clock (data0): cover period_start when possible (not always SPY) ---
    # This Kaggle SPY series starts 2005-02-25. Using it as data0 for a 1998 start
    # silently dropped 1998–2005. Fall back to AAPL/GE/… that cover the window.
    # Equities are then calendar-aligned onto the clock (volume=0 pre-IPO pads).
    clock_ticker = None
    calendar: Optional[pd.DatetimeIndex] = None
    chosen = _select_clock_ticker(period_start)
    candidates = []
    if chosen:
        candidates.append(chosen)
    candidates.extend(c for c in CLOCK_CANDIDATES if c != chosen)
    for cand in candidates:
        clock_path = _resolve_csv_path(cand)
        if not clock_path or not _has_enough_data(clock_path):
            continue
        clock_df = _load_ohlcv_df(clock_path, from_dt, to_dt)
        if clock_df is None or clock_df.empty:
            continue
        calendar = clock_df.index
        clock_ticker = cand
        feeds.append((_make_pandas_feed(clock_df, cand, from_dt, to_dt), cand))
        file_range = _peek_csv_date_range(cand)
        cal_start = str(calendar[0].date()) if len(calendar) else "?"
        note = ""
        if cand != "SPY":
            note = " (SPY Kaggle history too short for requested start — using equity clock)"
        print(
            f"  Clock feed (data0): {cand} "
            f"file_range={file_range} "
            f"bars={len(calendar)} calendar_start={cal_start} "
            f"fromdate={from_dt.date()} todate={to_dt.date()}"
            f"{note}"
        )
        if cal_start > period_start:
            print(
                f"  WARNING: clock {cand} first bar {cal_start} is after "
                f"requested start {period_start} — early years still truncated"
            )
        break
    if clock_ticker is None or calendar is None:
        print(
            "  WARNING: no preferred clock feed found; "
            "using first screened ticker as data0 (may truncate timeline)"
        )

    # Remaining screened names, reindexed onto the clock calendar
    for ticker in all_tickers:
        if ticker == clock_ticker:
            continue
        ok, reason = _ticker_eligible_for_multi(ticker, from_dt=from_dt, to_dt=to_dt)
        if not ok:
            if reason == "bad_or_short":
                skipped_bad += 1
            else:
                skipped_invalid += 1
            continue
        feed = _make_csv_feed(
            ticker, from_dt, to_dt, calendar=calendar if calendar is not None else None
        )
        if feed is None:
            skipped_invalid += 1
            continue
        feeds.append((feed, ticker))

    if not feeds:
        raise ValueError("No ticker data could be loaded from the Kaggle dataset.")
    if clock_ticker is None and feeds:
        print(
            f"  Clock feed (data0) fallback: {feeds[0][1]} "
            f"file_range={_peek_csv_date_range(feeds[0][1])}"
        )

    loaded_names = {name for _, name in feeds}
    screening_dict = {
        d: [t for t in ts if t in loaded_names]
        for d, ts in screening_dict.items()
    }
    screening_dict = {d: ts for d, ts in screening_dict.items() if ts}

    print(
        f"  Loaded {len(feeds)} feeds "
        f"(clock={clock_ticker or feeds[0][1]}, "
        f"{skipped_invalid} insufficient, {skipped_bad} bad/short skipped; "
        f"calendar-aligned={calendar is not None})."
    )

    StrategyClass = load_strategy_class(strategy_code)
    pct = max(1.0, min(100.0, float(position_size_pct_val)))
    config = ensure_stake_pct_in_config(config, pct)
    run_config = {**config, "screening": screening_dict}
    names = strategy_param_names(StrategyClass)
    if names and "stake_pct" not in names:
        run_config.pop("stake_pct", None)

    cerebro = build_cerebro(
        strategy_cls=StrategyClass,
        strategy_kwargs=run_config,
        feeds=feeds,
        stake_pct=pct,
        multi=True,
    )
    return run_and_metrics(
        cerebro, period_start=period_start, period_end=period_end, multi=True
    )
