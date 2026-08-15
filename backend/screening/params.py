"""Parse NL screening criteria into structured params (no free-form Python)."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from codegen.extract import extract_fenced_block

# Allowed metric names for the deterministic screener engine
VALID_METRICS = frozenset({"pct_change", "abs_change", "volume_ratio", "near_high"})

SCREENING_PARAMS_PROMPT = """\
Convert the USER screening rule into a JSON config for a daily stock-universe screener.

User rule:
{prompt}

CRITICAL:
- Derive EVERY field from the user rule text.
- Do NOT default to top 1% / 1-month gainers unless the user asked for that.
- Output ONLY one ```json block (no prose) with exactly these keys:
  lookback_days, top_pct, metric, rank_ascending, min_metric

Schema:
```json
{{
  "lookback_days": 21,
  "top_pct": 0.01,
  "metric": "pct_change",
  "rank_ascending": false,
  "min_metric": null
}}
```

Field meanings:
- lookback_days: bar window (≈5=1 week, ≈10=2 weeks, ≈21=1 month, ≈63=3 months, ≈126=6 months, ≈252=1 year / 52 weeks)
- top_pct: fraction kept each day by rank (0.01 = top 1%, 0.05 = top 5%). Use 1.0 when selection is by min_metric threshold only.
- metric:
  - "pct_change" — % price change over lookback (typical gainers/losers)
  - "abs_change" — absolute price change over lookback
  - "volume_ratio" — today's volume / lookback average volume
  - "near_high" — close / lookback rolling high (≈1.0 at 52-week high)
- rank_ascending: false = keep highest metric (gainers / high volume / near highs); true = keep lowest (losers / oversold)
- min_metric: optional floor on the metric after ranking is applied, or the main filter when top_pct is 1.0.
  Examples: 3.0 for "3x average volume"; 0.98 for "within 2% of 52-week high". Use null when unused.

## Few-shot examples (match JSON shape; values come from EACH rule)

### Example 1 — "Top 1% of stocks with the biggest price move over the past 1 month"
```json
{{"lookback_days": 21, "top_pct": 0.01, "metric": "pct_change", "rank_ascending": false, "min_metric": null}}
```

### Example 2 — "Top 5% of stocks by 2-week return"
```json
{{"lookback_days": 10, "top_pct": 0.05, "metric": "pct_change", "rank_ascending": false, "min_metric": null}}
```

### Example 3 — "Bottom 2% worst 1-month performers (biggest losers)"
```json
{{"lookback_days": 21, "top_pct": 0.02, "metric": "pct_change", "rank_ascending": true, "min_metric": null}}
```

### Example 4 — "Stocks with 3x average volume spike over the past 20 days"
```json
{{"lookback_days": 20, "top_pct": 1.0, "metric": "volume_ratio", "rank_ascending": false, "min_metric": 3.0}}
```

### Example 5 — "Top 3% by absolute dollar move over 1 week"
```json
{{"lookback_days": 5, "top_pct": 0.03, "metric": "abs_change", "rank_ascending": false, "min_metric": null}}
```

### Example 6 — "Stocks breaking above / near their 52-week high"
```json
{{"lookback_days": 252, "top_pct": 0.05, "metric": "near_high", "rank_ascending": false, "min_metric": 0.98}}
```

### Example 7 — "Top 10% volume-ratio names over 1 month (relative volume leaders)"
```json
{{"lookback_days": 21, "top_pct": 0.10, "metric": "volume_ratio", "rank_ascending": false, "min_metric": null}}
```

Now convert the USER rule only. One ```json block.
"""


def _parse_min_metric(prompt: str) -> Optional[float]:
    p = (prompt or "").lower()
    # "3x average volume", "2.5x their 15-day average volume", "volume 3 times"
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*[x×]\s*(?:their\s+|the\s+)?(?:\d+[\s\-]*day\s+)?(?:average\s+)?volume"
        r"|volume\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*[x×]"
        r"|(\d+(?:\.\d+)?)\s*times\s+(?:the\s+|their\s+)?(?:average\s+)?volume"
        r"|at\s+least\s+(\d+(?:\.\d+)?)\s*[x×]",
        p,
    )
    if m:
        for g in m.groups():
            if g is not None:
                return max(0.5, float(g))
    # near high: "within 2% of high" → 0.98
    m = re.search(r"within\s+(\d+(?:\.\d+)?)\s*%\s*of", p)
    if m:
        return max(0.5, min(1.0, 1.0 - float(m.group(1)) / 100.0))
    if re.search(r"52[\s\-]*week\s+high|new\s+highs?|break(?:ing)?\s+out|near\s+(?:their\s+)?high", p):
        return 0.98
    return None


def infer_screening_params(prompt: str) -> Dict[str, Any]:
    """Heuristic fallback when LLM params are unavailable/invalid."""
    p = (prompt or "").lower()
    lookback = 21
    # Explicit N-day window first (e.g. "15-day average", "10-day return")
    m_days = re.search(r"\b(\d{1,3})[\s\-]*days?\b", p)
    if re.search(r"52[\s\-]*week|1\s*year|\b252\b", p):
        lookback = 252
    elif re.search(r"\b(1|one)[\s\-]*weeks?\b|\b7\s*days?\b", p) and not m_days:
        lookback = 5
    elif re.search(r"\b(2|two)[\s\-]*weeks?\b|\b14\s*days?\b", p):
        lookback = 10
    elif re.search(r"\b(3|three)[\s\-]*months?\b|\b90\s*days?\b", p):
        lookback = 63
    elif re.search(r"\b(6|six)[\s\-]*months?\b", p):
        lookback = 126
    elif re.search(r"\b(1|one)[\s\-]*months?\b|\b30\s*days?\b|\b21\s*days?\b", p):
        lookback = 21
    elif m_days:
        lookback = max(2, min(252, int(m_days.group(1))))
    elif re.search(r"\b(5|five)\s*days?\b", p):
        lookback = 5

    top_pct = 0.01
    m = re.search(r"top\s+(\d+(?:\.\d+)?)\s*%", p)
    if m:
        top_pct = max(0.001, min(1.0, float(m.group(1)) / 100.0))
    else:
        m = re.search(r"top\s+(\d+(?:\.\d+)?)\s*percent", p)
        if m:
            top_pct = max(0.001, min(1.0, float(m.group(1)) / 100.0))
        else:
            m = re.search(r"bottom\s+(\d+(?:\.\d+)?)\s*%", p)
            if m:
                top_pct = max(0.001, min(1.0, float(m.group(1)) / 100.0))

    metric = "pct_change"
    if re.search(r"52[\s\-]*week\s+high|near\s+(?:their\s+)?high|new\s+highs?|break(?:ing)?\s+(?:above|out)", p):
        metric = "near_high"
    elif "volume" in p:
        metric = "volume_ratio"
    elif re.search(r"\babsolute\b|\bpoints?\b|\bdollar\b", p):
        metric = "abs_change"

    rank_ascending = bool(re.search(r"loser|declin|drop|worst|smallest|bottom\s+\d", p))
    min_metric = _parse_min_metric(prompt)

    # Threshold-only screens (e.g. 3x volume): keep all that pass min_metric
    if min_metric is not None and metric == "volume_ratio" and not re.search(r"top\s+\d", p):
        top_pct = 1.0
    if min_metric is not None and metric == "near_high" and not re.search(r"top\s+\d", p):
        # still rank-limit near-highs unless user gave top N%
        if top_pct == 0.01 and not re.search(r"\b1\s*%", p):
            top_pct = 0.05

    return {
        "lookback_days": lookback,
        "top_pct": top_pct,
        "metric": metric,
        "rank_ascending": rank_ascending,
        "min_metric": min_metric,
    }


def _coerce_min_metric(raw: Any) -> Optional[float]:
    if raw is None or raw == "" or (isinstance(raw, str) and raw.lower() in ("null", "none")):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def normalize_screening_params(data: Dict[str, Any], fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Clamp / validate a params dict; fill missing keys from fallback or safe defaults."""
    base = {
        "lookback_days": 21,
        "top_pct": 0.01,
        "metric": "pct_change",
        "rank_ascending": False,
        "min_metric": None,
    }
    if fallback:
        base.update({k: fallback[k] for k in base if k in fallback})
    out = dict(base)
    src = data or {}

    if "lookback_days" in src:
        try:
            out["lookback_days"] = max(2, min(252, int(src["lookback_days"])))
        except (TypeError, ValueError):
            pass
    if "top_pct" in src:
        try:
            v = float(src["top_pct"])
            if v > 1:
                v = v / 100.0
            out["top_pct"] = max(0.001, min(1.0, v))
        except (TypeError, ValueError):
            pass
    metric = str(src.get("metric", out["metric"])).strip().lower()
    if metric in VALID_METRICS:
        out["metric"] = metric
    if "rank_ascending" in src:
        out["rank_ascending"] = bool(src["rank_ascending"])
    if "min_metric" in src:
        out["min_metric"] = _coerce_min_metric(src.get("min_metric"))
    return out


def parse_screening_params_from_llm(response_text: str, screening_prompt: str) -> Dict[str, Any]:
    """Parse LLM JSON into validated screener params; fall back to heuristics from the user prompt."""
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
        print(f"  Using heuristic screening params from prompt: {defaults}")
        return normalize_screening_params(defaults)

    out = normalize_screening_params(data, fallback=defaults)
    print(f"  LLM screening params: {out} (from user rule)")
    return out
