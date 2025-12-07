# stats/summary_report.py

from datetime import datetime
from typing import Any, Dict, List, Optional


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

    summary["avg_win"] = float(won_pnl.get("avg")) if "avg" in won_pnl else None
    summary["avg_loss"] = float(lost_pnl.get("avg")) if "avg" in lost_pnl else None

    # אפשר להרחיב בעתיד (max win, max loss, consecutive וכו')
    return summary
