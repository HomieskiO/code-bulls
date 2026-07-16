"""
evaluator.py — Custom Backtrader analyzers and metrics extraction.

Exports used by graph.py:
  - Expectancy  (bt.Analyzer subclass, add via cerebro.addanalyzer)
  - get_metrics(cerebro, results) -> dict
"""

import numpy as np
import backtrader as bt
import backtrader.analyzers as btanalyzers
from datetime import datetime


# ---------------------------------------------------------------------------
# Analyzer: PortfolioValue  (monthly samples for equity-curve chart)
# ---------------------------------------------------------------------------

class PortfolioValueAnalyzer(bt.Analyzer):
    """Records monthly portfolio value for equity-curve visualisation."""

    def start(self):
        self._values     = []
        self._last_month = None

    def next(self):
        d = self.strategy.datetime.date()
        m = (d.year, d.month)
        if m != self._last_month:
            self._values.append({
                "date":  d.isoformat(),
                "value": round(self.strategy.broker.getvalue(), 2),
            })
            self._last_month = m

    def get_analysis(self):
        return self._values


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

        win_rate  = len(wins)   / total
        loss_rate = len(losses) / total
        avg_win   = float(np.mean(wins))                       if wins   else 0.0
        avg_loss  = float(np.mean([abs(l) for l in losses]))   if losses else 0.0
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        return {
            "total_trades": total,
            "win_rate":     round(win_rate  * 100, 4),   # % form
            "loss_rate":    round(loss_rate * 100, 4),   # % form
            "avg_win":      round(avg_win,   2),
            "avg_loss":     round(avg_loss,  2),
            "expectancy":   round(expectancy, 2),
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
        return {"cagr": round(cagr * 100, 4)}   # % form


# ---------------------------------------------------------------------------
# Analyzer: TradeLog  (per-trade rows for the UI Trades tab)
# ---------------------------------------------------------------------------

class TradeLogAnalyzer(bt.Analyzer):
    """
    One row per closed trade for the frontend Trades tab.

    Notes:
    - On close, ``trade.size`` is 0 in backtrader — size must be captured while open.
    - Prefer trade.history when tradehistory=True for entry/exit prices.
    """

    def start(self):
        self._trades = []
        self._open   = {}   # id(trade) -> open snapshot
        self._seq    = 0

    def _ticker(self, trade) -> str:
        data = trade.data
        if data is None:
            return "UNKNOWN"
        ticker = getattr(data, "_name", None) or getattr(data, "_dataname", None) or "UNKNOWN"
        return str(ticker).upper()

    def _bar_dt(self, trade):
        data = trade.data
        if data is None:
            return None
        try:
            return data.datetime.datetime(0)
        except Exception:
            return None

    def notify_trade(self, trade):
        data = trade.data
        if data is None:
            return

        key = id(trade)
        now = self._bar_dt(trade)
        ticker = self._ticker(trade)

        # Capture / refresh open snapshot while the trade has a non-zero size
        size_now = abs(float(trade.size or 0.0))
        if trade.isopen and size_now > 0:
            if key not in self._open:
                self._seq += 1
                self._open[key] = {
                    "id":          self._seq,
                    "ticker":      ticker,
                    "side":        "long" if float(trade.size or 0) >= 0 else "short",
                    "size":        size_now,
                    "entry_date":  now.date().isoformat() if now else None,
                    "entry_price": round(float(trade.price or 0.0), 4),
                }
            else:
                # Partial fills: keep max absolute size and avg entry price
                self._open[key]["size"] = max(self._open[key]["size"], size_now)
                if trade.price:
                    self._open[key]["entry_price"] = round(float(trade.price), 4)

        if not trade.isclosed:
            return

        opened = self._open.pop(key, None)

        # Prefer history events when available (tradehistory=True)
        entry_date = opened["entry_date"] if opened else None
        entry_price = opened["entry_price"] if opened else None
        size = opened["size"] if opened else 0.0
        side = opened["side"] if opened else "long"
        trade_id = opened["id"] if opened else None
        exit_price = None

        hist = getattr(trade, "history", None) or []
        if hist:
            try:
                # history events: each has status, event, size, price, ...
                first = hist[0]
                last = hist[-1]
                # event objects vary by bt version — use duck typing
                def _ev_price(ev):
                    return float(getattr(ev, "price", None) or getattr(getattr(ev, "event", None), "price", 0) or 0)

                def _ev_size(ev):
                    return abs(float(getattr(ev, "size", None) or getattr(getattr(ev, "event", None), "size", 0) or 0))

                def _ev_dt(ev):
                    raw = getattr(ev, "datetime", None) or getattr(getattr(ev, "event", None), "datetime", None)
                    if raw is None:
                        return None
                    try:
                        return bt.num2date(raw).date().isoformat()
                    except Exception:
                        return str(raw)[:10]

                if not entry_price:
                    entry_price = _ev_price(first) or None
                if not size:
                    size = max((_ev_size(ev) for ev in hist), default=0.0)
                if not entry_date:
                    entry_date = _ev_dt(first)
                exit_price = _ev_price(last) or None
            except Exception:
                pass

        if trade_id is None:
            self._seq += 1
            trade_id = self._seq

        if not entry_price and trade.price:
            entry_price = round(float(trade.price), 4)
        entry_price = float(entry_price or 0.0)
        size = float(size or 0.0)

        pnl = float(trade.pnlcomm or 0.0)
        if exit_price is None and size > 0 and entry_price:
            try:
                gross = float(trade.pnl or 0.0)
                if side == "short":
                    exit_price = entry_price - (gross / size)
                else:
                    exit_price = entry_price + (gross / size)
            except Exception:
                exit_price = None

        exit_date = now.date().isoformat() if now else None
        # Fall back to dtclose ordinal
        if exit_date is None and getattr(trade, "dtclose", None):
            try:
                exit_date = bt.num2date(trade.dtclose).date().isoformat()
            except Exception:
                pass
        if entry_date is None and getattr(trade, "dtopen", None):
            try:
                entry_date = bt.num2date(trade.dtopen).date().isoformat()
            except Exception:
                pass

        self._trades.append({
            "id":          trade_id,
            "ticker":      ticker if not opened else opened["ticker"],
            "side":        side,
            "size":        round(size, 4),
            "entry_date":  entry_date,
            "entry_price": round(entry_price, 4) if entry_price else None,
            "exit_date":   exit_date,
            "exit_price":  round(exit_price, 4) if exit_price is not None else None,
            "pnl":         round(pnl, 2),
            "pnl_pct":     round(
                (pnl / (entry_price * size) * 100) if entry_price and size else 0.0,
                2,
            ),
            "commission":  round(float(trade.commission or 0.0), 2),
        })

    def get_analysis(self):
        return self._trades


# ---------------------------------------------------------------------------
# get_metrics — called by graph.py after cerebro.run()
# ---------------------------------------------------------------------------

def get_metrics(cerebro, results, period_start: str = None, period_end: str = None):
    """
    Extract all performance metrics from a completed cerebro run.

    When ``period_start`` / ``period_end`` (YYYY-MM-DD) are provided, the equity
    curve is clipped to that window and CAGR / total return are computed over
    those dates so performance is always relative to the user-requested range.

    Expects the following analyzers to have been added before running:
        cerebro.addanalyzer(Expectancy,              _name='expectancy')
        cerebro.addanalyzer(bt.analyzers.DrawDown,   _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='cagr', ...)
        cerebro.addanalyzer(TradeLogAnalyzer,        _name='tradelog')
        cerebro.addanalyzer(PortfolioValueAnalyzer,  _name='portfolio')

    Returns a flat dict with keys matching what graph.py / the frontend expect.
    """
    strat         = results[0]
    initial_cash  = cerebro.broker.startingcash
    final_value   = cerebro.broker.getvalue()

    # --- Expectancy analyzer ---
    exp_data = strat.analyzers.expectancy.get_analysis()

    # --- DrawDown ---
    dd_data  = strat.analyzers.drawdown.get_analysis()
    max_dd   = dd_data.get("max", {}).get("drawdown", 0.0)   # already in %

    # --- Trade log ---
    trades = []
    if hasattr(strat.analyzers, "tradelog"):
        trades = list(strat.analyzers.tradelog.get_analysis() or [])
        if period_start or period_end:
            filtered = []
            for t in trades:
                # Keep trades that exit inside the window (or open if no exit)
                d = t.get("exit_date") or t.get("entry_date") or ""
                if period_start and d and d < period_start:
                    continue
                if period_end and d and d > period_end:
                    continue
                filtered.append(t)
            trades = filtered

    portfolio_values = []
    if hasattr(strat.analyzers, "portfolio"):
        portfolio_values = list(strat.analyzers.portfolio.get_analysis() or [])

    # Clip equity curve to the requested performance window (drop warmup bars)
    if portfolio_values and (period_start or period_end):
        clipped = []
        for e in portfolio_values:
            d = e["date"]
            if period_start and d < period_start:
                continue
            if period_end and d > period_end:
                continue
            clipped.append(e)
        # Keep the last bar before period_start as the starting capital anchor
        # so return is measured from period open, not from mid-window cash.
        if clipped and period_start:
            pre = [e for e in portfolio_values if e["date"] < period_start]
            if pre and clipped[0]["date"] > period_start:
                anchor = dict(pre[-1])
                anchor["date"] = period_start
                clipped = [anchor] + clipped
        if clipped:
            portfolio_values = clipped

    # CAGR + total return over the (possibly clipped) equity window
    cagr = 0.0
    total_return = round((final_value / initial_cash - 1) * 100, 4) if initial_cash else 0.0
    period_start_out = period_start
    period_end_out   = period_end

    if portfolio_values and len(portfolio_values) >= 2:
        pv_first = portfolio_values[0]["value"]
        pv_last  = portfolio_values[-1]["value"]
        d0 = portfolio_values[0]["date"]
        d1 = portfolio_values[-1]["date"]
        period_start_out = period_start_out or d0
        period_end_out   = period_end_out   or d1
        days = (
            datetime.strptime(d1, "%Y-%m-%d") -
            datetime.strptime(d0, "%Y-%m-%d")
        ).days
        if pv_first > 0:
            total_return = round((pv_last / pv_first - 1) * 100, 4)
            if days > 0 and pv_last > 0:
                cagr = round(((pv_last / pv_first) ** (365.25 / days) - 1) * 100, 4)
    else:
        # Fallback: TimeReturn / CAGRAnalyzer if no equity curve
        if hasattr(strat.analyzers, "cagr"):
            raw = strat.analyzers.cagr.get_analysis()
            if isinstance(raw, dict):
                if "cagr" in raw:
                    cagr = raw["cagr"]
                else:
                    annual_returns = list(raw.values())
                    if annual_returns:
                        cagr = round(float(np.mean(annual_returns)) * 100, 4)

        # If explicit dates given, recompute CAGR from cash endpoints + duration
        if period_start and period_end and initial_cash > 0 and final_value > 0:
            days = (
                datetime.strptime(period_end,   "%Y-%m-%d") -
                datetime.strptime(period_start, "%Y-%m-%d")
            ).days
            if days > 0:
                cagr = round(
                    ((final_value / initial_cash) ** (365.25 / days) - 1) * 100, 4
                )
                total_return = round((final_value / initial_cash - 1) * 100, 4)

    return {
        "cagr":                  cagr,
        "max_drawdown":          round(max_dd, 4),
        "win_rate":              exp_data.get("win_rate",   0.0),
        "avg_win":               exp_data.get("avg_win",    0.0),
        "avg_loss":              exp_data.get("avg_loss",   0.0),
        "expectancy":            exp_data.get("expectancy", 0.0),
        "total_trades":          exp_data.get("total_trades", 0) or len(trades),
        "final_portfolio_value": round(final_value, 2),
        "total_return_pct":      total_return,
        "portfolio_values":      portfolio_values,
        "trades":                trades,
        "period_start":          period_start_out,
        "period_end":            period_end_out,
    }
