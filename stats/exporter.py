import csv
from dataclasses import asdict
# CRITICAL FIX: We import the 'datetime' class specifically, not just the module.
from datetime import datetime
import os
from typing import List


def _add_timestamp(filename: str) -> str:
    """
    Adds a timestamp (YYYYMMDD_HHMM) to the filename.
    Example: 'trade_log.csv' -> 'trade_log_20260113_2045.csv'
    """
    base, ext = os.path.splitext(filename)
    # Get current time using the class method .now()
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{base}_{ts}{ext}"


def save_csv(data_list: List[object], filename: str):
    """
    Saves a list of dataclasses (like Trade Log or Symbol Stats) to CSV
    with a timestamped filename.
    """
    if not data_list:
        print(f"Warning: No data to save for {filename}")
        return

    final_name = _add_timestamp(filename)

    # Convert dataclasses to dicts
    rows = [asdict(item) for item in data_list]

    if not rows:
        return

    headers = rows[0].keys()

    try:
        with open(final_name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved report: {final_name}")
    except Exception as e:
        print(f"Error saving {final_name}: {e}")


def save_single_object_csv(obj: object, filename: str):
    """
    Saves a single dataclass object (like Strategy Summary) to CSV.
    """
    if not obj:
        return

    final_name = _add_timestamp(filename)
    row = asdict(obj)

    try:
        with open(final_name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
        print(f"Saved report: {final_name}")
    except Exception as e:
        print(f"Error saving {final_name}: {e}")