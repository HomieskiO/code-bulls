"""Single compact prompt pack for strategy generation / optimize / explain."""
from __future__ import annotations

import json
import os

import backtrader as bt


def build_bt_api_docs() -> str:
    lines = ["## Available bt.indicators (use bt.indicators.<Name>)"]
    specs = [
        ("RSI", bt.indicators.RSI),
        ("SMA", bt.indicators.SMA),
        ("EMA", bt.indicators.EMA),
        ("MACD", bt.indicators.MACD),
        ("BollingerBands", bt.indicators.BollingerBands),
        ("ATR", bt.indicators.ATR),
        ("CrossOver", bt.indicators.CrossOver),
        ("Stochastic", bt.indicators.Stochastic),
        ("Highest", bt.indicators.Highest),
        ("Lowest", bt.indicators.Lowest),
    ]
    skip = {"movav", "_movav", "_rocperiod"}
    for name, cls in specs:
        try:
            params = {
                k: (v if not isinstance(v, type) else v.__name__)
                for k, v in cls.params._getpairs().items()
                if k not in skip
            }
            lines.append(
                f"  bt.indicators.{name}(data, {', '.join(f'{k}={v!r}' for k, v in params.items())})"
            )
        except Exception:
            lines.append(f"  bt.indicators.{name}(data, ...)")
    lines.append("")
    lines.append(
        "## Multi-feed indicators: first arg is the feed, POSITIONAL only "
        "(never data= keyword)"
    )
    lines.append("  bt.indicators.SMA(d, period=20)")
    lines.append("  bt.indicators.BollingerBands(d, period=20, devfactor=2.0)")
    lines.append("  # WRONG: bt.indicators.BollingerBands(data=d, ...)")
    lines.append("")
    lines.append("## Strategy API: self.buy() self.sell() self.close() self.getposition(data)")
    lines.append("  self.broker.getcash()  self.broker.getvalue()")
    return "\n".join(lines)


_BT_API_DOCS = build_bt_api_docs()


def strategy_code_prompt(user_prompt: str, position_sizing: str, stake_frac: float) -> str:
    return f"""Convert this trading strategy into a complete backtrader.Strategy class.

{user_prompt}

{position_sizing}

Rules:
- params = (('name', default), ...) — never assign self.params.x in __init__.
- No imports. No notify_trade/notify_order. No comments.
- Include stop_loss, take_profit, stake_pct={stake_frac}.
- On entry call self.buy() with NO size=.
- Output EXACTLY one ```python block and one ```json block. Close both fences.
- Adapt indicators/logic to the user strategy.

API reference:
{_BT_API_DOCS}
"""


def multi_strategy_code_prompt(user_prompt: str, position_sizing: str, stake_frac: float) -> str:
    return f"""Write a multi-asset backtrader.Strategy for this idea:

{user_prompt}

{position_sizing}

Context:
- Many data feeds; each has d._name = ticker (UPPERCASE).
- Feeds share a master calendar (SPY). Pre-IPO / gap / delist days have volume=0 (OHLC may be 0 or ffilled) — skip trading those bars; close any open position if volume hits 0.
- self.p.screening is dict {{"YYYY-MM-DD": ["AAPL", ...]}} with UPPERCASE tickers.
- Only OPEN new positions for tickers in today's screening list.
- today = self.datetime.date(0).strftime('%Y-%m-%d')
- Match with: d._name in today_screened

Rules:
- params include screening={{}}, stake_pct={stake_frac}, stop_loss=None, take_profit=None.
- JSON must NOT include screening; may include stop_loss/take_profit (null or float).
- self.buy(data=d) with NO size=. No imports, no notify_*, no comments. Close both fences.
- Indicators: bt.indicators.X(d, period=...) — feed is first POSITIONAL arg. Never data=.
- BollingerBands uses devfactor= (not dev=). Access bands via .lines.bot / .lines.mid / .lines.top (or .bot/.mid/.top).
- try/except per feed for indicators; guard with if d._name not in self.inds.
- Skip inactive bars: volume<=0 (calendar padding). Close open positions when volume hits 0.
- Adapt indicators/logic to the user strategy while keeping screening gate for entries when relevant.

API reference:
{_BT_API_DOCS}
"""


def repair_strategy_prompt(
    *,
    error: str,
    prev: str,
    user_prompt: str,
    position_sizing: str,
    stake_frac: float,
    multi: bool = False,
) -> str:
    kind = "multi-asset screening strategy" if multi else "backtrader strategy"
    return f"""Your previous {kind} answer was invalid.

Error:
{error}

Previous output (truncated):
{prev}

User strategy:
{user_prompt}

{position_sizing}

Rewrite from scratch. Output EXACTLY one ```python block and one ```json block.
Include stake_pct={stake_frac}. No prose. Close both fences. Use buy() without size=.
{"Include screening={{}} in params; omit screening from JSON." if multi else ""}
"""


def adapt_strategy_prompt(
    *,
    user_prompt: str,
    adapt_instruction: str,
    prev_code: str,
    prev_config: dict,
    position_sizing: str,
    stake_frac: float,
    multi: bool = False,
) -> str:
    kind = "multi-asset screening strategy" if multi else "backtrader strategy"
    multi_bits = ""
    if multi:
        multi_bits = """
Multi-asset rules (keep):
- params include screening={{}} and stake_pct.
- Indicators take feed as first POSITIONAL arg: bt.indicators.X(d, ...). Never data=.
- Only OPEN new positions when d._name in today_screened; skip volume<=0 bars.
"""
    return f"""You are adapting an existing {kind}.

Original strategy idea:
{user_prompt}

User adaptation request (apply this change):
{adapt_instruction}

Current strategy code:
```python
{prev_code}
```

Previous default config:
```json
{json.dumps(prev_config, indent=2)}
```

{position_sizing}
{multi_bits}
Rewrite the FULL strategy with the adaptation applied.
Output EXACTLY one ```python block and one ```json block.
Include stake_pct={stake_frac}, stop_loss, take_profit.
No imports, no notify_*, no comments, no prose. Close both fences.
Use buy() without size=.
{"Include screening={{}} in params; omit screening from JSON." if multi else ""}
"""


def optimize_prompt(
    *,
    user_prompt: str,
    code: str,
    prev_iter: int,
    prev_config: dict,
    prev_metrics: dict,
    valid_keys: list,
) -> str:
    return f"""You are optimising a backtrader strategy.

Original strategy: {user_prompt}

```python
{code}
```

Iteration {prev_iter} results:
  Configuration: {json.dumps(prev_config, indent=2)}
  Metrics: {json.dumps(prev_metrics, indent=2)}

Propose new VALUES that improve CAGR while keeping Max Drawdown low.

CRITICAL:
- Use EXACTLY these keys: {json.dumps(valid_keys)}
- Do NOT rename/add/remove keys.
- Output ONLY one ```json block.
- stake_pct must stay unchanged if present.
- stop_loss/take_profit: null or positive float (e.g. 0.05).
"""


def explain_prompt(
    *,
    prompt: str,
    period_start: str,
    period_end: str,
    config: str,
    cagr,
    total_return,
    max_drawdown,
    win_rate,
    expectancy,
    total_trades,
    final_value,
) -> str:
    return f"""You are a quantitative trading analyst. Explain this optimized strategy to the user.

Original strategy request: "{prompt}"
Backtest period: {period_start} → {period_end}

Best configuration found: {config}

Performance metrics (relative to the backtest period):
- CAGR: {cagr}%
- Total Return: {total_return}%
- Max Drawdown: {max_drawdown}%
- Win Rate: {win_rate}%
- Expectancy: {expectancy}% expected return per trade
- Total Trades: {total_trades}
- Final Portfolio Value: ${final_value} (started at $100,000)

Write 3–4 concise sentences covering:
1. What the best configuration parameters mean in plain English
2. How the strategy performed over the backtest period (strengths)
3. Key risks or weaknesses

Plain language. No markdown headers or bullet points — flowing prose only.
"""
