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
from screening.engine import KAGGLE_STOCKS_PATH, MAX_UNIQUE_TICKERS
from .cerebro_builder import build_cerebro, config_score, run_and_metrics


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


def _price_series_sane(filepath: str, from_dt, to_dt) -> bool:
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
                if len(closes) >= 400:
                    break
        if len(closes) < 20:
            return False
        closes.sort()
        med = closes[len(closes) // 2]
        if med <= 0 or med > 10_000 or med < 0.05:
            return False
        if closes[-1] > 500_000:
            return False
        return True
    except Exception:
        return False


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

    all_tickers = sorted({t for ts in screening_dict.values() for t in ts})
    if not all_tickers:
        raise ValueError("Screening dict is empty — no tickers to trade.")

    # Cap unique tickers
    if len(all_tickers) > max_unique_tickers:
        freq: Dict[str, int] = {}
        for ts in screening_dict.values():
            for t in ts:
                freq[t] = freq.get(t, 0) + 1
        keep = {
            t
            for t, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[
                :max_unique_tickers
            ]
        }
        screening_dict = {
            d: [t for t in ts if t in keep] for d, ts in screening_dict.items()
        }
        screening_dict = {d: ts for d, ts in screening_dict.items() if ts}
        all_tickers = sorted(keep)
        print(f"  Capped universe to {len(all_tickers)} tickers (MAX_UNIQUE_TICKERS)")

    print(
        f"  Multi universe: {len(all_tickers)} tickers, "
        f"{len(screening_dict)} dates, "
        f"{sum(len(v) for v in screening_dict.values())} pairs"
    )

    preferred = [t for t in ("SPY", "AAPL", "MSFT", "GE", "IBM") if t in all_tickers]
    ordered = preferred + [t for t in all_tickers if t not in preferred]
    feeds = []
    skipped_invalid = skipped_bad = 0

    for ticker in ordered:
        filepath = os.path.join(KAGGLE_STOCKS_PATH, f"{ticker.lower()}.us.txt")
        if not os.path.exists(filepath) or not _has_enough_data(filepath):
            skipped_invalid += 1
            continue
        if not _price_series_sane(filepath, from_dt, to_dt):
            skipped_bad += 1
            continue
        try:
            feed = bt.feeds.GenericCSVData(
                dataname=filepath,
                name=ticker,
                dtformat=_fast_date_parse,
                date=0,
                open=1,
                high=2,
                low=3,
                close=4,
                volume=5,
                openinterest=6,
                fromdate=from_dt,
                todate=to_dt,
                preload=True,
            )
            feeds.append((feed, ticker))
        except Exception as ex:
            print(f"  WARNING: skipping {ticker}: {ex}")

    if not feeds:
        raise ValueError("No ticker data could be loaded from the Kaggle dataset.")
    print(
        f"  Loaded {len(feeds)} tickers "
        f"({skipped_invalid} insufficient, {skipped_bad} bad prices skipped)."
    )

    StrategyClass = load_strategy_class(strategy_code)
    pct = max(1.0, min(100.0, float(position_size_pct_val)))
    config = ensure_stake_pct_in_config(config, pct)
    run_config = {**config, "screening": screening_dict}
    names = strategy_param_names(StrategyClass)
    if names and "stake_pct" not in names:
        run_config.pop("stake_pct", None)
    if names and "screening" not in names:
        # still need screening — inject via params default if missing is rare
        pass

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
