class EmaCrossoverStrategy(bt.Strategy):
    params = (
        ('fast', 15),
        ('slow', 50),
        ('stake_pct', 1.0),
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
