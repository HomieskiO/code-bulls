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
                f"  bt.indicators.{name}({', '.join(f'{k}={v!r}' for k, v in params.items())})"
            )
        except Exception:
            lines.append(f"  bt.indicators.{name}")
    lines.append("")
    lines.append("## Strategy API: self.buy() self.sell() self.close() self.getposition(data)")
    lines.append("  self.broker.getcash()  self.broker.getvalue()")
    return "\n".join(lines)


_BT_API_DOCS = build_bt_api_docs()


def strategy_code_prompt(user_prompt: str, position_sizing: str, stake_frac: float) -> str:
    return f"""Convert this trading strategy into a complete backtrader.Strategy class.

User strategy:
{user_prompt}

{position_sizing}

MANDATORY output — nothing else, both fences closed:
```python
class MyStrategy(bt.Strategy):
    params = (
        ('fast', 15),
        ('slow', 50),
        ('stake_pct', {stake_frac}),
        ('stop_loss', None),
        ('take_profit', None),
    )
    def __init__(self):
        self.fast_ema = bt.indicators.EMA(period=self.p.fast)
        self.slow_ema = bt.indicators.EMA(period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ema, self.slow_ema)
    def next(self):
        if not self.position:
            if self.crossover[0] > 0:
                self.buy()
        else:
            if self.crossover[0] < 0:
                self.close()
        if self.position.size > 0:
            if self.p.stop_loss is not None and self.data.close[0] <= self.position.price * (1 - self.p.stop_loss):
                self.close()
            elif self.p.take_profit is not None and self.data.close[0] >= self.position.price * (1 + self.p.take_profit):
                self.close()
```
```json
{{"fast": 15, "slow": 50, "stake_pct": {stake_frac}, "stop_loss": null, "take_profit": null}}
```

Rules:
- Adapt the example to the user strategy (indicators, periods, conditions).
- params = (('name', default), ...) — never assign self.params.x in __init__.
- No imports. No notify_trade/notify_order. No comments.
- Include stop_loss, take_profit, stake_pct={stake_frac}.
- On entry call self.buy() with NO size=.
- Common indicators: EMA/SMA/RSI/MACD/BollingerBands/ATR/CrossOver/Highest/Lowest.
- Close BOTH fences.

API reference:
{_BT_API_DOCS}
"""


def multi_strategy_code_prompt(user_prompt: str, position_sizing: str, stake_frac: float) -> str:
    return f"""Write a multi-asset backtrader.Strategy for this idea:

{user_prompt}

{position_sizing}

Context:
- Many data feeds; each has d._name = ticker (UPPERCASE).
- self.p.screening is dict {{"YYYY-MM-DD": ["AAPL", ...]}} with UPPERCASE tickers.
- Only OPEN new positions for tickers in today's screening list.
- today = self.datetime.date(0).strftime('%Y-%m-%d')
- Match with: d._name in today_screened

Compatibility:
- For momentum/top-gainer screens, BUY when d._name in today_screened (do not require RSI oversold for entry).
- Use RSI/stops mainly for exits unless the user explicitly wants mean-reversion entries.
- For ATH breakouts: use Highest indicator; for SMA exits use SMA(period=50).

Output EXACTLY two closed fences:
```python
class MultiScreenStrategy(bt.Strategy):
    params = (
        ('sma_period', 50),
        ('warmup_period', 50),
        ('stake_pct', {stake_frac}),
        ('stop_loss', None),
        ('take_profit', None),
        ('screening', {{}}),
    )
    def __init__(self):
        self.inds = {{}}
        for d in self.datas:
            try:
                self.inds[d._name] = {{
                    'sma': bt.indicators.SMA(d, period=self.p.sma_period),
                    'ath': bt.indicators.Highest(d.close, period=252),
                }}
            except Exception:
                pass
    def next(self):
        today = self.datetime.date(0).strftime('%Y-%m-%d')
        today_screened = self.p.screening.get(today, [])
        for d in self.datas:
            if d._name not in self.inds:
                continue
            if len(d) <= self.p.warmup_period:
                continue
            pos = self.getposition(d)
            ath = self.inds[d._name]['ath'][0]
            sma = self.inds[d._name]['sma'][0]
            if pos.size == 0:
                if d._name in today_screened and d.close[0] >= ath:
                    self.buy(data=d)
            else:
                if d.close[0] < sma:
                    self.close(data=d)
            pos = self.getposition(d)
            if pos.size > 0:
                if self.p.stop_loss is not None and d.close[0] <= pos.price * (1 - self.p.stop_loss):
                    self.close(data=d)
                elif self.p.take_profit is not None and d.close[0] >= pos.price * (1 + self.p.take_profit):
                    self.close(data=d)
```
```json
{{"sma_period": 50, "warmup_period": 50, "stake_pct": {stake_frac}, "stop_loss": null, "take_profit": null}}
```

Rules:
- params include screening={{}} and stake_pct={stake_frac}. JSON must NOT include screening.
- self.buy(data=d) with NO size=. No imports, no notify_*, no comments. Close both fences.
- try/except per feed for indicators; guard with if d._name not in self.inds.
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
- Expectancy: ${expectancy} per trade
- Total Trades: {total_trades}
- Final Portfolio Value: ${final_value} (started at $100,000)

Write 3–4 concise sentences covering:
1. What the best configuration parameters mean in plain English
2. How the strategy performed over the backtest period (strengths)
3. Key risks or weaknesses

Plain language. No markdown headers or bullet points — flowing prose only.
"""
