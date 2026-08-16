"""TradeLogAnalyzer must record size/exit via trade.ref (not id)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import backtrader as bt

from evaluator import TradeLogAnalyzer


class _S(bt.Strategy):
    def __init__(self):
        self.sma = {d: bt.ind.SMA(d, period=3) for d in self.datas}

    def next(self):
        for d in self.datas:
            if len(d) < 5:
                continue
            pos = self.getposition(d)
            if not pos.size and d.close[0] > self.sma[d][0]:
                self.buy(data=d)
            elif pos.size and d.close[0] < self.sma[d][0]:
                self.close(data=d)


def _mk(n, seed, name):
    idx = pd.date_range("2010-01-04", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    px = 50 + np.cumsum(rng.normal(0, 0.8, n))
    df = pd.DataFrame(
        {"open": px, "high": px + 0.5, "low": px - 0.5, "close": px, "volume": 1e5},
        index=idx,
    )
    return bt.feeds.PandasData(dataname=df, name=name)


def test_tradelog_size_and_exit():
    c = bt.Cerebro()
    c.adddata(_mk(100, 1, "AAA"))
    c.adddata(_mk(100, 2, "BBB"))
    c.broker.setcash(100_000)
    c.addsizer(bt.sizers.PercentSizer, percents=10)
    c.addstrategy(_S)
    c.addanalyzer(TradeLogAnalyzer, _name="tradelog")
    r = c.run(tradehistory=True, stdstats=False)
    trades = r[0].analyzers.tradelog.get_analysis()
    assert len(trades) > 0
    assert all(t["size"] and t["size"] > 0 for t in trades), trades[:3]
    assert all(t["exit_price"] is not None for t in trades), trades[:3]
    assert all(t["entry_price"] for t in trades)
    assert all(t["entry_date"] and t["exit_date"] for t in trades)


if __name__ == "__main__":
    test_tradelog_size_and_exit()
    print("test_tradelog OK")
