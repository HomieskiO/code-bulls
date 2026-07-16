"""get_metrics period clipping tests (no full cerebro)."""
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluator import get_metrics


class FakeAnalyzer:
    def __init__(self, data):
        self._data = data

    def get_analysis(self):
        return self._data


def test_period_clip_and_cagr():
    portfolio = [
        {"date": "2009-12-01", "value": 100000},
        {"date": "2010-01-01", "value": 100000},
        {"date": "2012-01-01", "value": 121000},
        {"date": "2015-01-01", "value": 150000},
        {"date": "2017-11-01", "value": 180000},
        {"date": "2018-01-01", "value": 190000},
    ]
    analyzers = SimpleNamespace(
        expectancy=FakeAnalyzer({
            "total_trades": 2,
            "win_rate": 50.0,
            "avg_win": 100.0,
            "avg_loss": 50.0,
            "expectancy": 25.0,
        }),
        drawdown=FakeAnalyzer({"max": {"drawdown": 12.5}}),
        portfolio=FakeAnalyzer(portfolio),
        tradelog=FakeAnalyzer([
            {"pnl": 50.0, "pnl_pct": 10.0},
            {"pnl": -25.0, "pnl_pct": -5.0},
        ]),
    )
    strat = SimpleNamespace(analyzers=analyzers)
    broker = SimpleNamespace(startingcash=100000.0, getvalue=lambda: 180000.0)
    cerebro = SimpleNamespace(broker=broker)
    m = get_metrics(
        cerebro, [strat], period_start="2010-01-01", period_end="2017-11-10"
    )
    dates = [e["date"] for e in m["portfolio_values"]]
    assert dates[0] >= "2010-01-01"
    assert dates[-1] <= "2017-11-10"
    assert m["total_return_pct"] == 80.0  # 180k/100k - 1 within window
    assert "trades" in m
    # avg win/loss/expectancy reported as trade return % (not $)
    assert m["avg_win"] == 10.0
    assert m["avg_loss"] == 5.0
    # E = 0.5*10 - 0.5*5 = 2.5%
    assert m["expectancy"] == 2.5


if __name__ == "__main__":
    test_period_clip_and_cagr()
    print("test_metrics OK")
