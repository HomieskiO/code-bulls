"""
Pregenerated strategy/config fixtures for offline full-pipeline testing.

Enable via:
  - request body: use_fixtures=true
  - env: USE_FIXTURES=1  (forces fixtures when request also allows / or always)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from codegen.extract import ensure_stake_pct_in_config, inject_stake_pct_param

_DIR = Path(__file__).resolve().parent

# Default UI prompts (must match frontend defaults for automatic fixture match)
DEFAULT_SINGLE_PROMPT = (
    "Trade AAPL. Buy when the 15-day EMA crosses above the 50-day EMA. "
    "Sell when it crosses below."
)
DEFAULT_SCREENING_PROMPT = (
    "Top 1% of stocks with the biggest price move over the past 1 month"
)
DEFAULT_MULTI_STRATEGY_PROMPT = (
    "Buy when the 10-day SMA is above the 20-day SMA and both SMAs are sloping up. "
    "Sell when the 10-day SMA is below the 20-day SMA and both SMAs are sloping down."
)

DEFAULT_SCREENING_PARAMS = {
    "lookback_days": 21,
    "top_pct": 0.01,
    "metric": "pct_change",
    "rank_ascending": False,
}


def env_forces_fixtures() -> bool:
    return os.getenv("USE_FIXTURES", "").strip().lower() in ("1", "true", "yes", "on")


def fixtures_enabled(request_flag: bool = False) -> bool:
    """True if this run should skip LLM code generation."""
    return bool(request_flag) or env_forces_fixtures()


def _load_text(name: str) -> str:
    path = _DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Fixture missing: {path}")
    return path.read_text(encoding="utf-8")


def _load_json(name: str) -> dict:
    return json.loads(_load_text(name))


def load_single_fixture(position_size_pct: float = 100.0) -> Tuple[str, dict]:
    code = _load_text("single_ema_crossover.py")
    config = _load_json("single_ema_crossover.json")
    code = inject_stake_pct_param(code, position_size_pct)
    config = ensure_stake_pct_in_config(config, position_size_pct)
    return code, config


def load_multi_fixture(position_size_pct: float = 10.0) -> Tuple[str, dict]:
    code = _load_text("multi_sma_slope.py")
    config = _load_json("multi_sma_slope.json")
    code = inject_stake_pct_param(code, position_size_pct)
    config = ensure_stake_pct_in_config(config, position_size_pct)
    config.pop("screening", None)
    return code, config


def load_screening_params_fixture() -> dict:
    return dict(DEFAULT_SCREENING_PARAMS)


def fixture_explanation(mode: str) -> str:
    return (
        f"[Fixture mode] This run used pregenerated {mode} strategy code "
        f"(no LLM generation or optimization). Metrics still come from a real backtest."
    )
