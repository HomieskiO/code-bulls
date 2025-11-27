from pathlib import Path
import csv
from typing import List, Dict, Any


def export_trade_log_csv(trade_table: List[Dict[str, Any]],
                         strategy_name: str,
                         log_dir: str = "logs") -> str:
    """
    Save the trade log (list of dicts) into logs/trade_log_{strategy}.csv
    Returns the full path of the saved file.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    filename = f"trade_log_{strategy_name}.csv"
    filepath = log_path / filename

    if trade_table:
        keys = trade_table[0].keys()
        with filepath.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(trade_table)

    return str(filepath)


def export_summary_report_csv(report: Dict[str, Any],
                              strategy_name: str,
                              out_dir: str = "reports") -> str:
    """
    Export a *summary* row (meta + performance + trades_summary)
    as CSV file:
        reports/summary_{strategy_name}.csv

    trades_table לא נכלל כאן כי הוא כבר נשמר כקובץ לוג נפרד.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = f"summary_{strategy_name}.csv"
    filepath = out_path / filename

    # נבנה שורה אחת "שטוחה" מ-3 החלקים:
    meta = report.get("meta", {}) or {}
    performance = report.get("performance", {}) or {}
    trades_summary = report.get("trades_summary", {}) or {}

    flat_row: Dict[str, Any] = {}

    # נוסיף prefix לשדות כדי שיהיה ברור מאיפה הם מגיעים
    for k, v in meta.items():
        flat_row[f"meta_{k}"] = v

    for k, v in performance.items():
        flat_row[f"performance_{k}"] = v

    for k, v in trades_summary.items():
        flat_row[f"trades_{k}"] = v

    # כתיבה ל-CSV – שורה אחת
    fieldnames = list(flat_row.keys())
    with filepath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(flat_row)

    return str(filepath)
