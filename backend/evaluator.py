"""
evaluator.py — Custom Backtrader analyzers and metrics extraction.

Exports used by graph.py:
  - Expectancy  (bt.Analyzer subclass, add via cerebro.addanalyzer)
  - get_metrics(cerebro, results) -> dict
"""

import numpy as np
import backtrader as bt
import backtrader.analyzers as btanalyzers


# ---------------------------------------------------------------------------
# Analyzer: Expectancy
# ---------------------------------------------------------------------------

class Expectancy(bt.Analyzer):
    """
    Computes win-rate, average win, average loss, and expectancy
    by tracking every closed trade's net PnL (after commission).

    Formula: Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    """

    def start(self):
        self._trades = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self._trades.append(trade.pnlcomm)   # net PnL after commission

    def get_analysis(self):
        trades = self._trades
        total  = len(trades)

        if total == 0:
            return {
                "total_trades": 0,
                "win_rate":     0.0,
                "loss_rate":    0.0,
                "avg_win":      0.0,
                "avg_loss":     0.0,
                "expectancy":   0.0,
            }

        wins   = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]

        win_rate  = len(wins)   / total if total > 0 else 0.0
        loss_rate = len(losses) / total if total > 0 else 0.0
        avg_win   = float(np.mean(wins)) if wins else 0.0
        avg_loss  = float(np.mean([abs(l) for l in losses])) if losses else 0.0
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        return {
            "total_trades": total,
            "win_rate":     win_rate,
            "loss_rate":    loss_rate,
            "avg_win":      avg_win,
            "avg_loss":     avg_loss,
            "expectancy":   expectancy,
        }


# ---------------------------------------------------------------------------
# Analyzer: CAGR
# ---------------------------------------------------------------------------

class CAGRAnalyzer(bt.Analyzer):
    """Compound Annual Growth Rate over the full backtest period."""

    def start(self):
        self._start_value = None
        self._start_date  = None

    def next(self):
        if self._start_value is None:
            self._start_value = self.strategy.broker.getvalue()
            self._start_date  = self.strategy.datetime.date()

    def stop(self):
        self._end_value = self.strategy.broker.getvalue()
        self._end_date  = self.strategy.datetime.date()

    def get_analysis(self):
        if not self._start_value or self._start_value <= 0:
            return {"cagr": 0.0}

        days  = (self._end_date - self._start_date).days
        years = days / 365.25

        if years <= 0 or self._end_value <= 0:
            return {"cagr": 0.0}

        cagr = (self._end_value / self._start_value) ** (1.0 / years) - 1.0
        return {"cagr": cagr}


# ---------------------------------------------------------------------------
# get_metrics — called by graph.py after cerebro.run()
# ---------------------------------------------------------------------------

def get_metrics(cerebro, results):
    """
    Extract all performance metrics from a completed cerebro run.

    Expects the following analyzers to have been added before running:
        cerebro.addanalyzer(Expectancy,              _name='expectancy')
        cerebro.addanalyzer(bt.analyzers.DrawDown,   _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='cagr', ...)

    Returns a flat dict with keys matching what graph.py / the frontend expect.
    All percentage-based metrics are returned as ratios (e.g., 0.5 for 50%).
    """
    strat         = results[0]
    initial_cash  = cerebro.broker.startingcash
    final_value   = cerebro.broker.getvalue()
    total_return  = (final_value / initial_cash - 1) if initial_cash > 0 else 0.0

    # --- Expectancy analyzer ---
    exp_data = strat.analyzers.expectancy.get_analysis()
    expectancy_value = exp_data.get("expectancy", 0.0)
    expectancy_ratio = (expectancy_value / initial_cash) if initial_cash > 0 else 0.0

    # --- DrawDown ---
    dd_data  = strat.analyzers.drawdown.get_analysis()
    max_dd_pct   = dd_data.get("max", {}).get("drawdown", 0.0)  # This is in %, e.g. 23.8 for 23.8%
    max_dd_ratio = max_dd_pct / 100.0

    # --- CAGR  (support both CAGRAnalyzer and bt.analyzers.TimeReturn) ---
    cagr = 0.0
    if hasattr(strat.analyzers, "cagr"):
        raw = strat.analyzers.cagr.get_analysis()
        if isinstance(raw, dict):
            if "cagr" in raw:                           # CAGRAnalyzer
                cagr = raw["cagr"]                      # Already a ratio
            else:                                       # TimeReturn → {date: float}
                annual_returns = list(raw.values())
                if annual_returns:
                    cagr = float(np.mean(annual_returns)) # Already a ratio

    return {
        "cagr":                  round(cagr, 4),
        "max_drawdown":          round(max_dd_ratio, 4),
        "win_rate":              round(exp_data.get("win_rate", 0.0), 4),
        "avg_win":               round(exp_data.get("avg_win", 0.0), 2),
        "avg_loss":              round(exp_data.get("avg_loss", 0.0), 2),
        "expectancy":            round(expectancy_ratio, 4),
        "total_trades":          exp_data.get("total_trades", 0),
        "final_portfolio_value": round(final_value, 2),
        "total_return":          round(total_return, 4),
    }
