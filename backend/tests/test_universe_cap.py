"""Time-balanced universe cap should keep names from early years."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from screening.engine import cap_screening_time_balanced


def test_time_balanced_keeps_early_and_late():
    # Early-only and late-only leaders + one common name
    screening = {}
    for i in range(1, 20):
        d = f"2010-01-{i:02d}"
        screening[d] = ["EARLY_A", "EARLY_B", "COMMON"]
    for i in range(1, 100):
        # late period appears more often → global-frequency would prefer LATE_*
        d = f"2017-06-{(i % 28) + 1:02d}"
        if d in screening:
            d = f"2017-07-{(i % 28) + 1:02d}"
        screening[d] = [f"LATE_{i % 30}", "COMMON", "LATE_X"]

    filtered, keep, log = cap_screening_time_balanced(screening, max_unique_tickers=40)
    assert "EARLY_A" in keep or "EARLY_B" in keep, keep
    assert "COMMON" in keep
    years = {d[:4] for d in filtered}
    assert "2010" in years
    assert "2017" in years
    assert "time-balanced" in log


if __name__ == "__main__":
    test_time_balanced_keeps_early_and_late()
    print("test_universe_cap OK")
