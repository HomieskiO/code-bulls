class MultiScreenStrategy(bt.Strategy):
    params = (
        ('sma_fast', 10),
        ('sma_slow', 20),
        ('warmup_period', 30),
        ('stake_pct', 0.1),
        ('stop_loss', None),
        ('take_profit', None),
        ('screening', {}),
    )

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            try:
                self.inds[d._name] = {
                    'fast': bt.indicators.SMA(d, period=self.p.sma_fast),
                    'slow': bt.indicators.SMA(d, period=self.p.sma_slow),
                }
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
            # Calendar-aligned feeds: volume=0 means no real bar (pre-IPO / delist)
            try:
                vol0 = float(d.volume[0])
            except Exception:
                continue
            if vol0 <= 0:
                if pos.size > 0:
                    self.close(data=d)
                continue
            # Need a full warmup of live bars so SMAs are not polluted by zero-pads
            try:
                if any(float(d.volume[-i]) <= 0 for i in range(warm)):
                    continue
                px = float(d.close[0])
            except Exception:
                continue
            if px != px or px <= 0:
                continue
            fast = self.inds[d._name]['fast']
            slow = self.inds[d._name]['slow']
            try:
                f0, s0 = float(fast[0]), float(slow[0])
                f1, s1 = float(fast[-1]), float(slow[-1])
            except Exception:
                continue
            if f0 != f0 or s0 != s0 or f1 != f1 or s1 != s1:
                continue
            fast_up = f0 > f1
            slow_up = s0 > s1
            fast_dn = f0 < f1
            slow_dn = s0 < s1
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
