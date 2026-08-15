"""Screening param inference tests."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from screening.params import (
    SCREENING_PARAMS_PROMPT,
    infer_screening_params,
    normalize_screening_params,
    parse_screening_params_from_llm,
)


def test_top_1_month():
    p = infer_screening_params("Top 1% of stocks with biggest price move over past 1 month")
    assert p["top_pct"] == 0.01
    assert p["lookback_days"] == 21
    assert p["metric"] == "pct_change"
    assert p["rank_ascending"] is False


def test_volume_3x():
    p = infer_screening_params("Stocks with 3x average volume spike")
    assert p["metric"] == "volume_ratio"
    assert p["min_metric"] == 3.0
    assert p["top_pct"] == 1.0


def test_52_week_high():
    p = infer_screening_params("Stocks breaking above their 52-week high")
    assert p["metric"] == "near_high"
    assert p["lookback_days"] == 252
    assert p["min_metric"] == 0.98


def test_bottom_losers():
    p = infer_screening_params("Bottom 2% worst 1-month performers")
    assert p["top_pct"] == 0.02
    assert p["rank_ascending"] is True
    assert p["metric"] == "pct_change"


def test_parse_llm_json():
    resp = '```json\n{"lookback_days": 10, "top_pct": 0.05, "metric": "pct_change", "rank_ascending": false, "min_metric": null}\n```'
    p = parse_screening_params_from_llm(resp, "top 5% 2 week movers")
    assert p["lookback_days"] == 10
    assert p["top_pct"] == 0.05
    assert p["min_metric"] is None


def test_parse_pct_as_percent_number():
    resp = '```json\n{"lookback_days": 21, "top_pct": 3, "metric": "pct_change", "rank_ascending": false}\n```'
    p = parse_screening_params_from_llm(resp, "top 3%")
    assert p["top_pct"] == 0.03


def test_parse_volume_threshold():
    resp = '```json\n{"lookback_days": 20, "top_pct": 1.0, "metric": "volume_ratio", "rank_ascending": false, "min_metric": 3.0}\n```'
    p = parse_screening_params_from_llm(resp, "stocks with 3x volume")
    assert p["metric"] == "volume_ratio"
    assert p["min_metric"] == 3.0


def test_prompt_has_fewshots():
    assert "Example 1" in SCREENING_PARAMS_PROMPT
    assert "volume_ratio" in SCREENING_PARAMS_PROMPT
    assert "near_high" in SCREENING_PARAMS_PROMPT
    assert "Do NOT default to top 1%" in SCREENING_PARAMS_PROMPT


def test_normalize_clamps():
    p = normalize_screening_params({"lookback_days": 999, "top_pct": 50, "metric": "bogus"})
    assert p["lookback_days"] == 252
    assert p["top_pct"] == 0.5
    assert p["metric"] == "pct_change"


if __name__ == "__main__":
    test_top_1_month()
    test_volume_3x()
    test_52_week_high()
    test_bottom_losers()
    test_parse_llm_json()
    test_parse_pct_as_percent_number()
    test_parse_volume_threshold()
    test_prompt_has_fewshots()
    test_normalize_clamps()
    print("test_screening_params OK")
