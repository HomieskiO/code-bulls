"""Screening param inference tests."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from screening.params import infer_screening_params, parse_screening_params_from_llm


def test_top_1_month():
    p = infer_screening_params("Top 1% of stocks with biggest price move over past 1 month")
    assert p["top_pct"] == 0.01
    assert p["lookback_days"] == 21
    assert p["metric"] == "pct_change"
    assert p["rank_ascending"] is False


def test_volume():
    p = infer_screening_params("Stocks with 3x average volume spike")
    assert p["metric"] == "volume_ratio"


def test_parse_llm_json():
    resp = '```json\n{"lookback_days": 10, "top_pct": 0.05, "metric": "pct_change", "rank_ascending": false}\n```'
    p = parse_screening_params_from_llm(resp, "top 5% 2 week movers")
    assert p["lookback_days"] == 10
    assert p["top_pct"] == 0.05


def test_parse_pct_as_percent_number():
    resp = '```json\n{"lookback_days": 21, "top_pct": 3, "metric": "pct_change", "rank_ascending": false}\n```'
    p = parse_screening_params_from_llm(resp, "top 3%")
    assert p["top_pct"] == 0.03


if __name__ == "__main__":
    test_top_1_month()
    test_volume()
    test_parse_llm_json()
    test_parse_pct_as_percent_number()
    print("test_screening_params OK")
