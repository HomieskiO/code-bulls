# report_builder.py

from datetime import datetime
from typing import Any, Dict, List, Optional

import Dict


def build_summary_report(strategy) -> Dict[str, Any]:
    """
    Build a generic summary report from a backtrader strategy instance.

    Assumes that the following analyzers were added (with these names):
      - AnnualReturn  -> _name='annual_return'
      - TradeAnalyzer -> _name='trades'
      - TradeLog      -> _name='trade_log'   (הקובץ שבנינו לפני כן)

    Returns a dict with:
      - meta
      - performance
      - trades_summary
      - trades_table
    """

    broker = strategy.broker
    data = strategy.datas[0]

    analyzers = strategy.analyzers

    # ---- pull analyzers safely ----
    annual_ret = _safe_get_analyzer(analyzers, "annual_return")
    trades_analysis = _safe_get_analyzer(analyzers, "trades")
    trades_table = _safe_get_analyzer(analyzers, "trade_log", default=[])

    # ---- meta info ----
    meta: Dict[str, Any] = {}

    meta["strategy_name"] = strategy.__class__.__name__
    meta["data_name"] = getattr(data, "_name", None)

    # start / end datetime
    meta["start_datetime"] = _get_data_datetime(data, 0)
    meta["end_datetime"] = _get_data_datetime(data, -1)

    meta["bars"] = len(strategy)

    # סכום התיק בסוף הריצה
    meta["equity_end"] = float(broker.getvalue())
    meta["cash_end"] = float(broker.getcash())
    meta["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    # אם שמרת ערך התחלתי באסטרטגיה (כמו starting_cash / starting_value)
    starting_value = getattr(strategy, "starting_cash", None)
    if starting_value is None:
        starting_value = getattr(strategy, "starting_value", None)
    meta["equity_start"] = float(starting_value) if starting_value is not None else None

    # ---- performance section ----
    performance = _build_performance_from_annual_returns(annual_ret)

    # ---- trade summary from TradeAnalyzer ----
    trades_summary = _build_trades_summary(trades_analysis)

    # ---- assemble full report ----
    report: Dict[str, Any] = {
        "meta": meta,
        "performance": performance,
        "trades_summary": trades_summary,
        "trades_table": trades_table,
    }

    return report


# ================== helpers ================== #

def _safe_get_analyzer(analyzers, name: str, default: Any = None) -> Any:
    """
    Try to fetch analyzer by name and call get_analysis().
    Return default if missing or on error.
    """
    if not hasattr(analyzers, name):
        return default
    analyzer = getattr(analyzers, name)
    try:
        return analyzer.get_analysis()
    except Exception:
        return default


def _get_data_datetime(data, index: int) -> Optional[str]:
    """
    Gets ISO datetime from a data feed at given index.
    Returns None on failure.
    """
    try:
        dt: datetime = data.datetime.datetime(index)
        return dt.isoformat()
    except Exception:
        return None


def _build_performance_from_annual_returns(annual_ret: Any) -> Dict[str, Any]:
    """
    annual_ret is the dict from backtrader.analyzers.AnnualReturn:
      {year: return_decimal}

    We compute:
      - annual_returns: 그대로
      - total_return: compounded across years
      - cagr: compounded annual growth rate
    """
    perf: Dict[str, Any] = {}

    if not isinstance(annual_ret, dict) or not annual_ret:
        return perf

    # העתק פשוט למפת year -> return
    annual_returns: Dict[int, float] = {}
    for year in annual_ret.keys():
        annual_returns[int(year)] = float(annual_ret[year])

    years = sorted(annual_returns.keys())

    total_factor = 1.0
    for year in years:
        r = annual_returns[year]
        total_factor = total_factor * (1.0 + r)

    total_return = total_factor - 1.0

    num_years = len(years)
    if num_years > 0:
        cagr = total_factor ** (1.0 / float(num_years)) - 1.0
    else:
        cagr = None

    perf["annual_returns"] = annual_returns
    perf["total_return"] = total_return
    perf["cagr"] = cagr

    return perf


def _build_trades_summary(trades_analysis: Any) -> Dict[str, Any]:
    """
    trades_analysis is the dict from backtrader.analyzers.TradeAnalyzer.
    We extract a few useful aggregate stats.
    """
    summary: Dict[str, Any] = {}

    if not isinstance(trades_analysis, dict):
        return summary

    total = trades_analysis.get("total", {})
    won = trades_analysis.get("won", {})
    lost = trades_analysis.get("lost", {})

    total_trades = total.get("total", 0)
    wins = won.get("total", 0)
    losses = lost.get("total", 0)

    summary["total_trades"] = int(total_trades)
    summary["wins"] = int(wins)
    summary["losses"] = int(losses)

    if total_trades:
        summary["win_rate"] = wins / float(total_trades)
    else:
        summary["win_rate"] = None

    # ממוצעי רווח/הפסד אם קיימים
    won_pnl = won.get("pnl", {})
    lost_pnl = lost.get("pnl", {})
    # report_builder.py
    from datetime import datetime
    from typing import Any, Dict, Optional

    def build_comprehensive_report(strategy) -> Dict[str, Any]:
        """
        Builds a data-rich summary report.
        REQUIRES the following analyzers to be added to Cerebro:
          - bt.analyzers.TradeAnalyzer (name='trades')
          - bt.analyzers.DrawDown (name='drawdown')
          - bt.analyzers.SharpeRatio (name='sharpe')
          - bt.analyzers.SQN (name='sqn')
          - AdvancedTradeLog (name='trade_log')
        """

        broker = strategy.broker
        data = strategy.datas[0]
        analyzers = strategy.analyzers

        # ---- 1. Pull Analyzers ----
        trades_analysis = _safe_get_analyzer(analyzers, "trades")
        drawdown_analysis = _safe_get_analyzer(analyzers, "drawdown")
        sharpe_analysis = _safe_get_analyzer(analyzers, "sharpe")
        sqn_analysis = _safe_get_analyzer(analyzers, "sqn")

        # Get the raw list of trades from our custom logger
        trade_log_data = _safe_get_analyzer(analyzers, "trade_log", default=[])

        # ---- 2. Meta Data ----
        meta: Dict[str, Any] = {}
        meta["strategy"] = strategy.__class__.__name__
        meta["asset"] = getattr(data, "_name", "Unknown")
        meta["start_date"] = _get_data_datetime(data, 0)
        meta["end_date"] = _get_data_datetime(data, -1)
        meta["total_bars"] = len(strategy)
        meta["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Equity Info
        equity_end = float(broker.getvalue())
        starting_cash = getattr(strategy, "starting_cash", 100000.0)  # Fallback if not set
        total_return_abs = equity_end - starting_cash
        total_return_pct = (total_return_abs / starting_cash) * 100.0

        meta["equity_start"] = starting_cash
        meta["equity_end"] = equity_end
        meta["return_abs"] = total_return_abs
        meta["return_pct"] = total_return_pct

        # ---- 3. Risk Metrics (Drawdown & Sharpe) ----
        risk: Dict[str, Any] = {}

        # Drawdown extraction
        if isinstance(drawdown_analysis, dict):
            # max.drawdown is the deep percentage drop
            risk["max_drawdown_pct"] = drawdown_analysis.get("max", {}).get("drawdown", 0.0)
            risk["max_drawdown_money"] = drawdown_analysis.get("max", {}).get("moneydown", 0.0)
            risk["max_drawdown_len_bars"] = drawdown_analysis.get("max", {}).get("len", 0)

        # Sharpe extraction
        if isinstance(sharpe_analysis, dict):
            risk["sharpe_ratio"] = sharpe_analysis.get("sharperatio", 0.0)

        # SQN extraction (System Quality Number)
        if isinstance(sqn_analysis, dict):
            risk["sqn_score"] = sqn_analysis.get("sqn", 0.0)

        # ---- 4. Trade Statistics (Deep Dive) ----
        stats: Dict[str, Any] = {}

        if isinstance(trades_analysis, dict):
            total = trades_analysis.get("total", {})
            won = trades_analysis.get("won", {})
            lost = trades_analysis.get("lost", {})

            # Counts
            total_closed = total.get("total", 0)
            stats["total_trades"] = total_closed
            stats["wins"] = won.get("total", 0)
            stats["losses"] = lost.get("total", 0)

            # Win Rate
            stats["win_rate_pct"] = (stats["wins"] / total_closed * 100) if total_closed > 0 else 0.0

            # PnL Stats
            pnl_data = trades_analysis.get("pnl", {})
            stats["pnl_net_total"] = pnl_data.get("net", {}).get("total", 0.0)
            stats["pnl_gross_total"] = pnl_data.get("gross", {}).get("total", 0.0)

            # Averages
            stats["avg_profit_net"] = pnl_data.get("net", {}).get("average", 0.0)
            stats["avg_win_money"] = won.get("pnl", {}).get("average", 0.0)
            stats["avg_loss_money"] = lost.get("pnl", {}).get("average", 0.0)

            # Best / Worst
            stats["largest_win"] = won.get("pnl", {}).get("max", 0.0)
            stats["largest_loss"] = lost.get("pnl", {}).get("max", 0.0)

            # Streaks
            streak = trades_analysis.get("streak", {})
            stats["max_win_streak"] = streak.get("won", {}).get("longest", 0)
            stats["max_loss_streak"] = streak.get("lost", {}).get("longest", 0)

            # Derived Metrics: Profit Factor
            # (Gross Win / Absolute Gross Loss)
            gross_profit = 0.0
            gross_loss = 0.0
            # We need to iterate the log or trust the TradeAnalyzer dict structure slightly more deeply
            # Simpler way: TradeAnalyzer usually aggregates this, but if not, we use:
            # avg_win * wins vs avg_loss * losses approximation or exact totals if avail.
            # Let's calculate from totals directly if available:
            # Note: Backtrader's TradeAnalyzer structure is complex, manual calculation is safer for Profit Factor
            total_won_money = stats["avg_win_money"] * stats["wins"]
            total_lost_money = abs(stats["avg_loss_money"] * stats["losses"])

            if total_lost_money > 0:
                stats["profit_factor"] = total_won_money / total_lost_money
            else:
                stats["profit_factor"] = 999.0 if total_won_money > 0 else 0.0

        # ---- Assemble Report ----
        report: Dict[str, Any] = {
            "meta": meta,
            "risk": risk,
            "stats": stats,
            "trade_log": trade_log_data
        }

        return report

    # ================== Helpers ================== #

    def _safe_get_analyzer(analyzers, name: str, default: Any = None) -> Any:
        """
        Helper to fetch analyzer results safely.
        """
        if not hasattr(analyzers, name):
            return default
        analyzer = getattr(analyzers, name)
        try:
            return analyzer.get_analysis()
        except Exception:
            return default

    def _get_data_datetime(data, index: int) -> Optional[str]:
        try:
            dt = data.datetime.datetime(index)
            return dt.isoformat()
        except Exception:
            return None
    summary["avg_win"] = float(won_pnl.get("avg")) if "avg" in won_pnl else None
    summary["avg_loss"] = float(lost_pnl.get("avg")) if "avg" in lost_pnl else None

    # אפשר להרחיב בעתיד (max win, max loss, consecutive וכו')
    return summary
