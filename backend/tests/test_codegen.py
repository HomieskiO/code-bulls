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
    normalize_python_source,
    position_size_pct,
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
    test_ensure_stake()
    test_extract_fenced()
    test_normalize_dedent()
    print("test_codegen OK")
