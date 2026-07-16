"""Shared Cerebro construction and backtrader patches."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import backtrader as bt
import backtrader.linebuffer as bt_lb

from evaluator import (
    Expectancy,
    PortfolioValueAnalyzer,
    TradeLogAnalyzer,
    get_metrics,
)


def patch_backtrader_divzero() -> None:
    """Tolerate ZeroDivisionError in LinesOperation (e.g. RSI with zero losses)."""

    def _safe_once_op(self, start, end):
        dst = self.array
        srca = self.a.array
        srcb = self.b.array
        op = self.operation
        for i in range(start, end):
            try:
                dst[i] = op(srca[i], srcb[i])
            except ZeroDivisionError:
                dst[i] = float("nan")

    def _safe_next(self):
        try:
            if self.bline:
                self[0] = self.operation(self.a[0], self.b[0])
            elif not self.r:
                if not self.btime:
                    self[0] = self.operation(self.a[0], self.b)
                else:
                    self[0] = self.operation(self.a.time(), self.b)
            else:
                self[0] = self.operation(self.a, self.b[0])
        except ZeroDivisionError:
            self[0] = float("nan")

    bt_lb.LinesOperation._once_op = _safe_once_op
    bt_lb.LinesOperation.next = _safe_next


patch_backtrader_divzero()


def config_score(metrics: dict) -> float:
    cagr = float(metrics.get("cagr", 0) or 0)
    dd = abs(float(metrics.get("max_drawdown", 0) or 0))
    return cagr * 0.65 - dd * 0.35


def build_cerebro(
    *,
    strategy_cls: Type[bt.Strategy],
    strategy_kwargs: Dict[str, Any],
    feeds: List[Any],
    stake_pct: float = 100.0,
    cash: float = 100_000.0,
    commission: float = 0.001,
    multi: bool = False,
) -> bt.Cerebro:
    """
    Single place for broker, sizer, strategy, and analyzers.
    ``stake_pct`` is percent of available cash (1–100).
    """
    cerebro = bt.Cerebro()
    for feed in feeds:
        if isinstance(feed, tuple):
            data, name = feed
            cerebro.adddata(data, name=name)
        else:
            cerebro.adddata(feed)

    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=float(stake_pct))
    cerebro.addstrategy(strategy_cls, **strategy_kwargs)
    cerebro.addanalyzer(Expectancy, _name="expectancy")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="cagr", timeframe=bt.TimeFrame.Years
    )
    cerebro.addanalyzer(PortfolioValueAnalyzer, _name="portfolio")
    cerebro.addanalyzer(TradeLogAnalyzer, _name="tradelog")
    return cerebro


def run_and_metrics(
    cerebro: bt.Cerebro,
    *,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    multi: bool = False,
) -> dict:
    if multi:
        # runonce=True precomputes indicators in batch; with many multi-length
        # feeds that yields ValueError: max() iterable argument is empty
        # (e.g. Highest/SMA once() on empty slices). Bar-by-bar is safer.
        results = cerebro.run(stdstats=False, tradehistory=True, runonce=False)
    else:
        results = cerebro.run()
    return get_metrics(
        cerebro, results, period_start=period_start, period_end=period_end
    )
