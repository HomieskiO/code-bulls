"""LLM screening-code generation: prompt, extract, validate, fixture template."""
from __future__ import annotations

import ast
import re
from typing import Optional, Set

from codegen.extract import extract_fenced_block, normalize_python_source, validate_python_syntax

# Injected into exec namespace — generated code must NOT import.
ALLOWED_BUILTINS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "print",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}

FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
        "exit",
        "quit",
    }
)

SCREENING_CODE_PROMPT = """\
Write a Python screener that implements the USER rule and writes a CSV of daily screened stocks.

User rule:
{prompt}

Backtest period (filter output to this range):
  start_date = {start_date}
  end_date   = {end_date}

Dataset:
  stocks_dir contains files named like AAPL.us.txt
  each file has columns: Date,Open,High,Low,Close,Volume
  Date is YYYY-MM-DD. Tickers = filename without ".us.txt", UPPERCASE.

REQUIRED API (exact name + signature):
```python
def build_screening_csv(stocks_dir, start_date, end_date, out_csv):
    ...
```

Contract:
- Write CSV to out_csv with header: date,ticker
- date = YYYY-MM-DD, ticker = UPPERCASE symbol
- Only include rows with start_date <= date <= end_date
- One row per (date, ticker) that PASSES the user rule that day
- No imports (pd, np, os, Path, datetime, timedelta, math, re, defaultdict are already available)
- No comments, no prose outside the code fence
- Output EXACTLY one ```python block, closed

## Few-shot examples (adapt logic to the USER rule — do not copy blindly)

### Example A — top 1% by 21-day return
```python
def build_screening_csv(stocks_dir, start_date, end_date, out_csv):
    lookback = 21
    top_pct = 0.01
    rows = []
    paths = [os.path.join(stocks_dir, f) for f in os.listdir(stocks_dir) if f.endswith('.us.txt')]
    frames = []
    for path in paths:
        ticker = os.path.basename(path)[:-7].upper()
        try:
            df = pd.read_csv(path, usecols=['Date', 'Close', 'Volume'])
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.copy()
        df['ticker'] = ticker
        df['metric'] = df['Close'].pct_change(lookback)
        frames.append(df)
    if not frames:
        pd.DataFrame(columns=['date', 'ticker']).to_csv(out_csv, index=False)
        return
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=['metric'])
    all_df = all_df[(all_df['Date'] >= start_date) & (all_df['Date'] <= end_date)]
    all_df['rank_pct'] = all_df.groupby('Date')['metric'].rank(pct=True, ascending=False)
    keep = all_df[all_df['rank_pct'] <= top_pct]
    out = keep[['Date', 'ticker']].copy()
    out.columns = ['date', 'ticker']
    out['ticker'] = out['ticker'].astype(str).str.upper()
    out = out.drop_duplicates()
    out.to_csv(out_csv, index=False)
```

### Example B — volume >= 2.5x 15-day average
```python
def build_screening_csv(stocks_dir, start_date, end_date, out_csv):
    lookback = 15
    min_ratio = 2.5
    frames = []
    for f in os.listdir(stocks_dir):
        if not f.endswith('.us.txt'):
            continue
        path = os.path.join(stocks_dir, f)
        ticker = f[:-7].upper()
        try:
            df = pd.read_csv(path, usecols=['Date', 'Close', 'Volume'])
        except Exception:
            continue
        if df is None or len(df) < lookback + 1:
            continue
        df = df.copy()
        df['ticker'] = ticker
        avg = df['Volume'].rolling(lookback, min_periods=max(2, lookback // 2)).mean()
        df['metric'] = df['Volume'] / avg.replace(0, pd.NA)
        frames.append(df)
    if not frames:
        pd.DataFrame(columns=['date', 'ticker']).to_csv(out_csv, index=False)
        return
    all_df = pd.concat(frames, ignore_index=True).dropna(subset=['metric'])
    all_df = all_df[(all_df['Date'] >= start_date) & (all_df['Date'] <= end_date)]
    keep = all_df[all_df['metric'] >= min_ratio]
    out = keep[['Date', 'ticker']].rename(columns={{'Date': 'date'}})
    out['ticker'] = out['ticker'].astype(str).str.upper()
    out.drop_duplicates().to_csv(out_csv, index=False)
```

Now implement build_screening_csv for the USER rule only.
"""


SCREENING_REPAIR_PROMPT = """\
Your previous screening script was invalid.

Error:
{error}

Previous code (truncated):
{prev}

User rule:
{prompt}

Period: {start_date} → {end_date}

Rewrite from scratch. Output EXACTLY one ```python block with:
  def build_screening_csv(stocks_dir, start_date, end_date, out_csv):
that writes CSV columns date,ticker. No imports. No prose.
"""


def extract_screening_code(response_text: str) -> str:
    code = extract_fenced_block(response_text or "", "python")
    if not code:
        m = re.search(
            r"(def\s+build_screening_csv\s*\(.*)$",
            response_text or "",
            re.DOTALL,
        )
        if m:
            code = m.group(1).strip()
    if not code:
        raise ValueError(
            "LLM response missing ```python block with build_screening_csv.\n"
            f"Response was:\n{(response_text or '')[:500]}"
        )
    code = normalize_python_source(code)
    if "def build_screening_csv" not in code:
        raise ValueError("Screening code must define build_screening_csv(...)")
    validate_python_syntax(code)
    validate_screening_code_safety(code)
    return code


def validate_screening_code_safety(code: str) -> None:
    """Reject imports and dangerous builtins/calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Screening code SyntaxError: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError(
                "Screening code must not use import statements "
                "(pd/np/os/Path/… are pre-injected)."
            )
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"Forbidden name in screening code: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"Forbidden dunder attribute: {node.attr}")
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in FORBIDDEN_NAMES:
                raise ValueError(f"Forbidden call: {fn.id}()")


def fixture_screening_code() -> str:
    """Default offline screener: top 1% by 21-day pct change (matches DEFAULT params)."""
    return '''def build_screening_csv(stocks_dir, start_date, end_date, out_csv):
    lookback = 21
    top_pct = 0.01
    frames = []
    for f in os.listdir(stocks_dir):
        if not f.endswith('.us.txt'):
            continue
        path = os.path.join(stocks_dir, f)
        ticker = f[:-7].upper()
        try:
            df = pd.read_csv(path, usecols=['Date', 'Close', 'Volume'])
        except Exception:
            continue
        if df is None or df.empty or len(df) < lookback + 1:
            continue
        df = df.copy()
        df['ticker'] = ticker
        df['metric'] = df['Close'].pct_change(lookback)
        frames.append(df)
    if not frames:
        pd.DataFrame(columns=['date', 'ticker']).to_csv(out_csv, index=False)
        return
    all_df = pd.concat(frames, ignore_index=True).dropna(subset=['metric'])
    all_df = all_df[(all_df['Date'] >= start_date) & (all_df['Date'] <= end_date)]
    all_df['rank_pct'] = all_df.groupby('Date')['metric'].rank(pct=True, ascending=False)
    keep = all_df[all_df['rank_pct'] <= top_pct]
    out = keep[['Date', 'ticker']].copy()
    out.columns = ['date', 'ticker']
    out['ticker'] = out['ticker'].astype(str).str.upper()
    out.drop_duplicates().to_csv(out_csv, index=False)
'''
