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


def _single_fewshots(stake_frac: float) -> str:
    """Compact single-asset few-shots (format + API patterns only)."""
    return f"""## Few-shot examples (match this OUTPUT FORMAT; adapt logic to the user idea)

### Example A — idea: "Buy when 15-EMA crosses above 50-EMA; sell on cross below."
```python
class EmaCrossoverStrategy(bt.Strategy):
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

### Example B — idea: "Buy RSI(14) < 30; sell RSI > 70."
```python
class RsiMeanReversionStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_low', 30),
        ('rsi_high', 70),
        ('stake_pct', {stake_frac}),
        ('stop_loss', None),
        ('take_profit', None),
    )
    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
    def next(self):
        if not self.position:
            if self.rsi[0] < self.p.rsi_low:
                self.buy()
        else:
            if self.rsi[0] > self.p.rsi_high:
                self.close()
        if self.position.size > 0:
            if self.p.stop_loss is not None and self.data.close[0] <= self.position.price * (1 - self.p.stop_loss):
                self.close()
            elif self.p.take_profit is not None and self.data.close[0] >= self.position.price * (1 + self.p.take_profit):
                self.close()
```
```json
{{"rsi_period": 14, "rsi_low": 30, "rsi_high": 70, "stake_pct": {stake_frac}, "stop_loss": null, "take_profit": null}}
```
"""


def _multi_fewshots(stake_frac: float) -> str:
    """Compact multi-asset few-shots (screening gate, volume pad, positional feeds)."""
    return f"""## Few-shot examples (match this OUTPUT FORMAT; adapt logic to the user idea)

### Example A — idea: "Buy when SMA10 > SMA20 and both sloping up; sell when both sloping down."
```python
class MultiSmaSlopeStrategy(bt.Strategy):
    params = (
        ('sma_fast', 10),
        ('sma_slow', 20),
        ('warmup_period', 30),
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
                    'fast': bt.indicators.SMA(d, period=self.p.sma_fast),
                    'slow': bt.indicators.SMA(d, period=self.p.sma_slow),
                }}
            except Exception:
                pass
    def next(self):
        today = self.datetime.date(0).strftime('%Y-%m-%d')
        today_screened = self.p.screening.get(today, [])
        warm = int(self.p.warmup_period)
        for d in self.datas:
            if d._name not in self.inds:
                continue
            if len(d) <= warm:
                continue
            pos = self.getposition(d)
            try:
                if float(d.volume[0]) <= 0:
                    if pos.size > 0:
                        self.close(data=d)
                    continue
                if any(float(d.volume[-i]) <= 0 for i in range(warm)):
                    continue
                px = float(d.close[0])
                if px != px or px <= 0:
                    continue
            except Exception:
                continue
            fast = self.inds[d._name]['fast']
            slow = self.inds[d._name]['slow']
            try:
                f0, s0 = float(fast[0]), float(slow[0])
                f1, s1 = float(fast[-1]), float(slow[-1])
            except Exception:
                continue
            if f0 != f0 or s0 != s0:
                continue
            fast_up, slow_up = f0 > f1, s0 > s1
            fast_dn, slow_dn = f0 < f1, s0 < s1
            if pos.size == 0:
                if d._name in today_screened and f0 > s0 and fast_up and slow_up:
                    self.buy(data=d)
            else:
                if f0 < s0 and fast_dn and slow_dn:
                    self.close(data=d)
            pos = self.getposition(d)
            if pos.size > 0:
                if self.p.stop_loss is not None and px <= pos.price * (1 - self.p.stop_loss):
                    self.close(data=d)
                elif self.p.take_profit is not None and px >= pos.price * (1 + self.p.take_profit):
                    self.close(data=d)
```
```json
{{"sma_fast": 10, "sma_slow": 20, "warmup_period": 30, "stake_pct": {stake_frac}, "stop_loss": null, "take_profit": null}}
```

### Example B — idea: "Buy when close is below lower Bollinger (20, 2); sell when close is above middle band."
```python
class MultiBollingerStrategy(bt.Strategy):
    params = (
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('warmup_period', 30),
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
                    'bb': bt.indicators.BollingerBands(d, period=self.p.bb_period, devfactor=self.p.bb_dev),
                }}
            except Exception:
                pass
    def next(self):
        today = self.datetime.date(0).strftime('%Y-%m-%d')
        today_screened = self.p.screening.get(today, [])
        warm = int(self.p.warmup_period)
        for d in self.datas:
            if d._name not in self.inds:
                continue
            if len(d) <= warm:
                continue
            pos = self.getposition(d)
            try:
                if float(d.volume[0]) <= 0:
                    if pos.size > 0:
                        self.close(data=d)
                    continue
                if any(float(d.volume[-i]) <= 0 for i in range(warm)):
                    continue
                px = float(d.close[0])
                if px != px or px <= 0:
                    continue
            except Exception:
                continue
            bb = self.inds[d._name]['bb']
            try:
                bot = float(bb.bot[0])
                mid = float(bb.mid[0])
            except Exception:
                continue
            if bot != bot or mid != mid:
                continue
            if pos.size == 0:
                if d._name in today_screened and px < bot:
                    self.buy(data=d)
            else:
                if px > mid:
                    self.close(data=d)
            pos = self.getposition(d)
            if pos.size > 0:
                if self.p.stop_loss is not None and px <= pos.price * (1 - self.p.stop_loss):
                    self.close(data=d)
                elif self.p.take_profit is not None and px >= pos.price * (1 + self.p.take_profit):
                    self.close(data=d)
```
```json
{{"bb_period": 20, "bb_dev": 2.0, "warmup_period": 30, "stake_pct": {stake_frac}, "stop_loss": null, "take_profit": null}}
```
"""


def strategy_code_prompt(user_prompt: str, position_sizing: str, stake_frac: float) -> str:
    return f"""Convert this trading strategy into a complete backtrader.Strategy class.

User strategy:
{user_prompt}

{position_sizing}

{_single_fewshots(stake_frac)}
Rules:
- params = (('name', default), ...) — never assign self.params.x in __init__.
- No imports. No notify_trade/notify_order. No comments.
- Include stop_loss, take_profit, stake_pct={stake_frac}.
- On entry call self.buy() with NO size=.
- Output EXACTLY one ```python block and one ```json block for the USER strategy only (not the examples). Close both fences.
- Adapt indicators/logic to the user strategy; do not copy an example blindly.

API reference:
{_BT_API_DOCS}
"""


def multi_strategy_code_prompt(user_prompt: str, position_sizing: str, stake_frac: float) -> str:
    return f"""Write a multi-asset backtrader.Strategy for this idea:

User strategy:
{user_prompt}

{position_sizing}

Context:
- Many data feeds; each has d._name = ticker (UPPERCASE).
- Feeds share a master calendar (SPY). Pre-IPO / gap / delist days have volume=0 (OHLC may be 0 or ffilled) — skip trading those bars; close any open position if volume hits 0.
- self.p.screening is dict {{"YYYY-MM-DD": ["AAPL", ...]}} with UPPERCASE tickers.
- Only OPEN new positions for tickers in today's screening list.
- today = self.datetime.date(0).strftime('%Y-%m-%d')
- Match with: d._name in today_screened

{_multi_fewshots(stake_frac)}
Rules:
- params include screening={{}}, stake_pct={stake_frac}, stop_loss=None, take_profit=None.
- JSON must NOT include screening; may include stop_loss/take_profit (null or float).
- self.buy(data=d) with NO size=. No imports, no notify_*, no comments. Close both fences.
- Indicators: bt.indicators.X(d, period=...) — feed is first POSITIONAL arg. Never data=.
- BollingerBands uses devfactor= (not dev=). Access bands via .bot / .mid / .top.
- try/except per feed for indicators; guard with if d._name not in self.inds.
- Skip inactive bars: volume<=0 (calendar padding). Close open positions when volume hits 0.
- Output EXACTLY one ```python block and one ```json block for the USER strategy only (not the examples).
- Adapt indicators/logic to the user strategy while keeping the screening gate for entries when relevant.

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
