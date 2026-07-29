"""Unit tests for codegen helpers (no LLM, no network)."""
import ast
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codegen.extract import (
    ensure_stake_pct_in_config,
    extract_code_and_config,
    extract_fenced_block,
    inject_stake_pct_param,
    inject_standard_strategy_params,
    load_strategy_class,
    normalize_python_source,
    position_size_pct,
    sanitize_bt_code,
    strategy_param_names,
    validate_python_syntax,
)


def test_position_size_pct_clamp():
    assert position_size_pct({"position_size_pct": 150}, 100) == 100.0
    assert position_size_pct({"position_size_pct": 0}, 100) == 1.0
    assert position_size_pct({}, 10) == 10.0


def test_inject_stake_pct():
    code = "class S(bt.Strategy):\n    params = (\n        ('fast', 15),\n    )\n"
    out = inject_stake_pct_param(code, 10)
    assert "stake_pct" in out
    assert "0.1" in out
    ast.parse(out)


def test_inject_standard_params_multi():
    code = (
        "class MultiAssetBollingerStrategy(bt.Strategy):\n"
        "    params = (\n"
        "        ('bb_period', 20),\n"
        "        ('bb_dev', 2),\n"
        "        ('stake_pct', 0.1),\n"
        "    )\n"
        "    def next(self):\n"
        "        pass\n"
    )
    out = inject_standard_strategy_params(code, multi=True)
    assert "stop_loss" in out
    assert "take_profit" in out
    assert "screening" in out
    # idempotent
    out2 = inject_standard_strategy_params(out, multi=True)
    assert out2.count("stop_loss") == out.count("stop_loss")
    cls = load_strategy_class(out)
    names = strategy_param_names(cls)
    assert {"stop_loss", "take_profit", "screening", "stake_pct"} <= names


def test_sanitize_indicator_data_kwarg():
    raw = (
        "class S(bt.Strategy):\n"
        "    def __init__(self):\n"
        "        self.bb = bt.indicators.BollingerBands(data=d, period=20, dev=2)\n"
        "        self.sma = bt.indicators.SMA(period=10, data=d)\n"
    )
    out = sanitize_bt_code(raw)
    assert "data=" not in out
    assert "BollingerBands(d," in out.replace(" ", "")
    assert "devfactor=" in out
    assert "dev=" not in out.replace("devfactor", "")
    # SMA data= moved to first positional
    assert "SMA(d," in out.replace(" ", "")
    validate_python_syntax(out)


def test_ensure_stake():
    assert ensure_stake_pct_in_config({"a": 1}, 100)["stake_pct"] == 1.0
    assert ensure_stake_pct_in_config({}, 10)["stake_pct"] == 0.1


def test_extract_fenced():
    text = "```python\nclass A(bt.Strategy):\n    def next(self):\n        pass\n```\n```json\n{\"x\": 1}\n```"
    code, config = extract_code_and_config(text)
    assert "class A" in code
    assert config["x"] == 1


def test_normalize_dedent():
    raw = "    def foo():\n        return 1\n"
    out = normalize_python_source(raw)
    validate_python_syntax(out)
    assert out.lstrip().startswith("def foo")


if __name__ == "__main__":
    test_position_size_pct_clamp()
    test_inject_stake_pct()
    test_inject_standard_params_multi()
    test_sanitize_indicator_data_kwarg()
    test_ensure_stake()
    test_extract_fenced()
    test_normalize_dedent()
    print("test_codegen OK")
