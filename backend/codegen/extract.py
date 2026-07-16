"""Code extraction, sanitization, and strategy loading helpers."""
from __future__ import annotations

import ast
import json
import math
import re
import textwrap
from typing import Any, Dict, Optional, Tuple, Type

import backtrader as bt
import numpy as np
import datetime as dt_module

_CODE_FIXES = {
    "bt.indicators.Crossover": "bt.indicators.CrossOver",
    "bt.ind.Crossover": "bt.indicators.CrossOver",
    "indicators.Crossover": "indicators.CrossOver",
    "bt.indicators.crossover": "bt.indicators.CrossOver",
    "bt.indicators.ema(": "bt.indicators.EMA(",
    "bt.indicators.sma(": "bt.indicators.SMA(",
    "bt.indicators.rsi(": "bt.indicators.RSI(",
    "bt.indicators.macd(": "bt.indicators.MACD(",
    "bt.indicators.bollinger": "bt.indicators.BollingerBands",
    "bt.indicators.Bollinger(": "bt.indicators.BollingerBands(",
}


def position_size_pct(state: dict, default: float = 100.0) -> float:
    try:
        raw = state.get("position_size_pct", default)
        v = float(default if raw is None else raw)
    except (TypeError, ValueError):
        v = default
    return max(1.0, min(100.0, v))


def position_sizing_instructions(pct: float) -> str:
    frac = round(pct / 100.0, 4)
    return (
        f"POSITION SIZING (mandatory):\n"
        f"- Allocate exactly {pct:g}% of currently available cash to each new entry "
        f"(stake_pct = {frac}).\n"
        f"- Include ('stake_pct', {frac}) in params and in the ```json defaults.\n"
        f"- On buy, call self.buy() / self.buy(data=d) with NO size= argument "
        f"(runtime PercentSizer applies stake_pct).\n"
        f"- Do NOT hardcode a different stake.\n"
    )


def ensure_stake_pct_in_config(config: dict, pct: float) -> dict:
    cfg = dict(config or {})
    cfg["stake_pct"] = round(float(pct) / 100.0, 4)
    return cfg


def inject_stake_pct_param(code: str, pct: float) -> str:
    frac = round(float(pct) / 100.0, 4)
    if re.search(r"""['"]stake_pct['"]""", code):
        return re.sub(
            r"""\(\s*['"]stake_pct['"]\s*,\s*[^)]+\)""",
            f"('stake_pct', {frac})",
            code,
        )
    m = re.search(r"params\s*=\s*\(\s*\n?", code)
    if m:
        return code[: m.end()] + f"        ('stake_pct', {frac}),\n" + code[m.end() :]
    return f"# position size: {pct:g}% cash per trade (stake_pct={frac})\n" + code


def strategy_param_names(cls) -> set:
    try:
        return set(cls.params._getkeys())
    except Exception:
        try:
            return set(dict(cls.params._getpairs()).keys())
        except Exception:
            return set()


def sanitize_bt_code(code: str) -> str:
    for wrong, right in _CODE_FIXES.items():
        code = code.replace(wrong, right)
    return code


def normalize_python_source(code: str) -> str:
    code = (code or "").replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    code = re.sub(r"^```(?:python)?\s*\n?", "", code.strip(), flags=re.I)
    code = re.sub(r"\n?```\s*$", "", code.strip())
    code = textwrap.dedent(code)
    lines = code.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    indents = [len(ln) - len(ln.lstrip(" ")) for ln in lines if ln.strip()]
    if indents:
        common = min(indents)
        if common > 0:
            lines = [ln[common:] if ln.strip() else ln for ln in lines]
    return "\n".join(lines) + "\n"


def validate_python_syntax(code: str) -> None:
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise ValueError(
            f"Generated Python has SyntaxError at line {e.lineno}: {e.msg}\n"
            f"  {e.text or ''}"
        ) from e


def extract_fenced_block(text: str, lang: str) -> Optional[str]:
    m = re.search(rf"```{lang}\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(rf"```{lang}\s*\n(.*)$", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _looks_truncated(code: str) -> bool:
    if not code or not code.strip():
        return True
    s = code.rstrip()
    if s.endswith("\\"):
        return True
    if re.search(
        r"(?:\b(?:and|or|not|if|elif|else|return|with|for|while|def|class)\b"
        r"|[,=\(\[\{+\-*/:])\s*$",
        s,
    ):
        return True
    opens = s.count("(") + s.count("[") + s.count("{")
    closes = s.count(")") + s.count("]") + s.count("}")
    if opens > closes:
        return True
    if "class " not in s or "bt.Strategy" not in s:
        return True
    if "def next" not in s:
        return True
    return False


def _infer_config_from_code(code: str) -> dict:
    config: dict = {}

    def _parse_literal(raw: str):
        val = raw.strip().rstrip(",")
        if val in ("None", "null"):
            return None
        if val in ("True", "False"):
            return val == "True"
        try:
            return json.loads(val)
        except Exception:
            try:
                return float(val) if "." in val else int(val)
            except Exception:
                return val.strip("'\"")

    params_idx = re.search(r"\bparams\s*=", code)
    search_region = code[params_idx.start() : params_idx.start() + 1500] if params_idx else code
    for name, raw in re.findall(
        r"""\(\s*['"](\w+)['"]\s*,\s*(None|True|False|null|-?\d+\.?\d*)\s*\)""",
        search_region,
    ):
        config[name] = _parse_literal(raw)
    if not config:
        for name, raw in re.findall(r"""self\.params\.(\w+)\s*=\s*([^\n#]+)""", code):
            config[name] = _parse_literal(raw)
    config.setdefault("stop_loss", None)
    config.setdefault("take_profit", None)
    return config


def extract_code_and_config(
    response_text: str, allow_infer_config: bool = True
) -> Tuple[str, dict]:
    code = extract_fenced_block(response_text, "python")
    if not code:
        m = re.search(
            r"(class\s+\w+\s*\(\s*bt\.Strategy\s*\).*)$",
            response_text,
            re.DOTALL,
        )
        if m:
            code = m.group(1).strip()
    if not code:
        raise ValueError(
            "LLM response is missing the required ```python code block.\n"
            f"Response was:\n{response_text[:500]}"
        )
    if _looks_truncated(code):
        raise ValueError(
            "LLM strategy code looks truncated (incomplete statement / missing next()).\n"
            f"Code tail:\n{code[-300:]}"
        )

    json_raw = extract_fenced_block(response_text, "json")
    config = None
    if json_raw:
        try:
            config = json.loads(json_raw)
        except json.JSONDecodeError:
            obj = re.search(r"\{.*\}", json_raw, re.DOTALL)
            if obj:
                try:
                    config = json.loads(obj.group(0))
                except json.JSONDecodeError:
                    config = None

    if config is None and allow_infer_config:
        config = _infer_config_from_code(code)
        if len(config) <= 2 and set(config.keys()) <= {"stop_loss", "take_profit"}:
            config = None

    if config is None:
        raise ValueError(
            "LLM response is missing a usable ```json config block "
            "(and could not infer params from the code).\n"
            f"Response was:\n{response_text[:500]}"
        )
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object, got {type(config)}")
    config.setdefault("stop_loss", None)
    config.setdefault("take_profit", None)
    return code, config


def make_exec_namespace() -> dict:
    return {
        "bt": bt,
        "numpy": np,
        "np": np,
        "math": math,
        "datetime": dt_module,
    }


def load_strategy_class(code: str) -> Type[bt.Strategy]:
    namespace = make_exec_namespace()
    sanitized = sanitize_bt_code(code)
    try:
        exec(sanitized, namespace)
    except SyntaxError as se:
        raise ValueError(f"Syntax error in generated strategy code: {se}") from se
    StrategyClass = next(
        (
            obj
            for obj in namespace.values()
            if isinstance(obj, type)
            and issubclass(obj, bt.Strategy)
            and obj is not bt.Strategy
        ),
        None,
    )
    if StrategyClass is None:
        raise ImportError(
            "No bt.Strategy subclass found in generated code.\n"
            f"Code:\n{code[:400]}"
        )
    return StrategyClass
