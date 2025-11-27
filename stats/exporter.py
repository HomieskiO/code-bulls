from pathlib import Path
import csv
from typing import List, Dict, Any, Union


def export_csv(data: Union[List[Dict[str, Any]], Dict[str, Any]],
               strategy_name: str) -> str:
    """
    Export either:
      - trade log (list[dict])  -> logs/trade_log_{strategy_name}.csv
      - summary report (dict)   -> report/summary_{strategy_name}.csv

    Heuristics:
      - If 'data' is a list of dicts -> treated as trade log (one row per trade)
      - If 'data' is a dict and has keys like 'meta'/'performance' ->
            treated as summary report.
    Returns the full path of the saved CSV file.
    """

    # ---- case 1: trade log (list of dict rows) ----
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return _export_trade_log_csv(data, strategy_name)

    # ---- case 2: summary report (dict with sections) ----
    if isinstance(data, dict):
        # אם זה נראה כמו דוח מסכם
        if any(k in data for k in ("meta", "performance", "trades_summary")):
            return _export_summary_report_csv(data, strategy_name)

    raise ValueError("export_csv: unsupported data format for CSV export")


def _export_trade_log_csv(trade_table: List[Dict[str, Any]],
                          strategy_name: str,
                          log_dir: str = "logs") -> str:
    """
    Internal: export trade log (list of dict) to logs/trade_log_{strategy}.csv
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


def _export_summary_report_csv(report: Dict[str, Any],
                               strategy_name: str,
                               out_dir: str = "reports") -> str:
    """
    Internal: export summary report (meta + performance + trades_summary)
    as a single-row CSV:
        reports/summary_{strategy_name}.csv
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = f"summary_{strategy_name}.csv"
    filepath = out_path / filename

    meta = report.get("meta", {}) or {}
    performance = report.get("performance", {}) or {}
    trades_summary = report.get("trades_summary", {}) or {}

    flat_row: Dict[str, Any] = {}

    for k, v in meta.items():
        flat_row[f"meta_{k}"] = v

    for k, v in performance.items():
        flat_row[f"performance_{k}"] = v

    for k, v in trades_summary.items():
        flat_row[f"trades_{k}"] = v

    fieldnames = list(flat_row.keys())

    with filepath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(flat_row)

    return str(filepath)
