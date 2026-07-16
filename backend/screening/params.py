"""Parse NL screening criteria into structured params (no free-form Python)."""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from codegen.extract import extract_fenced_block

SCREENING_PARAMS_PROMPT = """\
Convert this stock-screening rule into a JSON config for a daily universe screener.

Rule:
{prompt}

Output ONLY one ```json block with exactly these keys:
```json
{{
  "lookback_days": 21,
  "top_pct": 0.01,
  "metric": "pct_change",
  "rank_ascending": false
}}
```

Field meanings:
- lookback_days: trading-day window (≈21 for 1 month, ≈5 for 1 week, ≈63 for 3 months)
- top_pct: fraction kept each day (0.01 = top 1%)
- metric: "pct_change" | "abs_change" | "volume_ratio"
- rank_ascending: false = keep highest metric (gainers); true = keep lowest

If ambiguous, use the defaults above. No prose outside the json fence.
"""


def infer_screening_params(prompt: str) -> Dict[str, Any]:
    p = (prompt or "").lower()
    lookback = 21
    if re.search(r"\b(1|one)\s*week\b|\b7\s*days?\b", p):
        lookback = 5
    elif re.search(r"\b(2|two)\s*weeks?\b|\b14\s*days?\b", p):
        lookback = 10
    elif re.search(r"\b(3|three)\s*months?\b|\b90\s*days?\b", p):
        lookback = 63
    elif re.search(r"\b(6|six)\s*months?\b", p):
        lookback = 126
    elif re.search(r"\b(1|one)\s*month\b|\b30\s*days?\b|\b21\s*days?\b", p):
        lookback = 21

    top_pct = 0.01
    m = re.search(r"top\s+(\d+(?:\.\d+)?)\s*%", p)
    if m:
        top_pct = max(0.001, min(0.5, float(m.group(1)) / 100.0))
    else:
        m = re.search(r"top\s+(\d+(?:\.\d+)?)\s*percent", p)
        if m:
            top_pct = max(0.001, min(0.5, float(m.group(1)) / 100.0))

    metric = "pct_change"
    if "volume" in p:
        metric = "volume_ratio"
    elif re.search(r"\babsolute\b|\bpoints?\b", p):
        metric = "abs_change"

    rank_ascending = bool(re.search(r"loser|declin|drop|worst|smallest", p))
    return {
        "lookback_days": lookback,
        "top_pct": top_pct,
        "metric": metric,
        "rank_ascending": rank_ascending,
    }


def parse_screening_params_from_llm(response_text: str, screening_prompt: str) -> Dict[str, Any]:
    defaults = infer_screening_params(screening_prompt)
    raw = extract_fenced_block(response_text or "", "json")
    data = None
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            obj = re.search(r"\{.*\}", raw, re.DOTALL)
            if obj:
                try:
                    data = json.loads(obj.group(0))
                except json.JSONDecodeError:
                    data = None
    if not isinstance(data, dict):
        obj = re.search(r"\{[^{}]+\}", response_text or "", re.DOTALL)
        if obj:
            try:
                data = json.loads(obj.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        print(f"  Using heuristic screening params: {defaults}")
        return defaults

    out = dict(defaults)
    if "lookback_days" in data:
        try:
            out["lookback_days"] = max(2, min(252, int(data["lookback_days"])))
        except (TypeError, ValueError):
            pass
    if "top_pct" in data:
        try:
            v = float(data["top_pct"])
            if v > 1:
                v = v / 100.0
            out["top_pct"] = max(0.001, min(0.5, v))
        except (TypeError, ValueError):
            pass
    metric = str(data.get("metric", out["metric"])).strip().lower()
    if metric in ("pct_change", "abs_change", "volume_ratio"):
        out["metric"] = metric
    if "rank_ascending" in data:
        out["rank_ascending"] = bool(data["rank_ascending"])
    return out
