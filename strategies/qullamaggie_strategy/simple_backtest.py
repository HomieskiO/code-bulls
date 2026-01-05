import backtrader as bt
import yfinance as yf
import pandas as pd
import datetime

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CSV_FILE = 'qullamaggie_candidates_close_only.csv'
TICKER = 'NVDA'
START_DATE = '1999-01-22'  # NVDA IPO
END_DATE = '2016-12-31'


# -----------------------------------------------------------------------------
# STRATEGY CLASS
# -----------------------------------------------------------------------------
class QullamaggieBreakout(bt.Strategy):
    params = (
        ('sma_period', 20),
        ('signals_file', None),
        ('target_ticker', 'NVDA')
    )

    def __init__(self):
        self.sma20 = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.sma_period
        )

        # Load valid buy dates from the CSV
        self.buy_dates = self.load_buy_signals()

    def load_buy_signals(self):
        """Loads the CSV and returns a set of date strings for the target ticker."""
        if self.params.signals_file is None:
            return set()

        print(f"Loading signals for {self.params.target_ticker}...")
        df = pd.read_csv(self.params.signals_file)

        # Filter for the specific ticker
        df = df[df['Ticker'] == self.params.target_ticker]

        # Convert 'Date' column to a set of strings (YYYY-MM-DD)
        # We ensure they are strings to match Backtrader's date format easily
        valid_dates = set(df['Date'].astype(str).tolist())

        print(f"Loaded {len(valid_dates)} buy signals.")
        return valid_dates

    def next(self):
        # Get the current date in YYYY-MM-DD string format
        current_date_str = self.data.datetime.date(0).isoformat()

        # 1. EXIT LOGIC: If we are in a position
        if self.position:
            # Sell if Close is below 20 SMA
            if self.data.close[0] < self.sma20[0]:
                self.close()  # Sell at next Open
                # Optional: print(f"{current_date_str}: Sell Signal (Close < 20SMA)")

        # 2. ENTRY LOGIC: If we are NOT in a position
        # (We check !self.position so we don't try to buy when already 100% invested)
        if not self.position:
            if current_date_str in self.buy_dates:
                # Buy at next Open
                # size=0.99 ensures we don't get rejected for insufficient cash due to rounding
                self.order_target_percent(target=0.99)
                print(f"{current_date_str}: Buy Signal Triggered")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED at {order.executed.price:.2f} on {bt.num2date(order.executed.dt).date()}")
            elif order.issell():
                print(f"SELL EXECUTED at {order.executed.price:.2f} on {bt.num2date(order.executed.dt).date()}")


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Setup Cerebro Engine
    cerebro = bt.Cerebro()

    # 2. Load Data from Yahoo Finance
    print("Downloading data from Yahoo Finance...")
    data = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
    data.columns = ["Open", "High", "Low", "Close", "Volume"]
    datafeed = bt.feeds.PandasData(dataname=data)

    cerebro.adddata(datafeed)

    # 3. Add Strategy
    cerebro.addstrategy(
        QullamaggieBreakout,
        signals_file=CSV_FILE,
        target_ticker=TICKER
    )

    # 4. Broker Settings
    cerebro.broker.setcash(10000.0)  # Starting Cash
    cerebro.broker.setcommission(commission=0.0)  # Zero commission for simplicity

    # 5. Run Backtest
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # 6. Plot Result
    # Note: Plotting might fail on some headless servers.
    # If running locally, this will pop up a chart.
    try:
        cerebro.plot(style='candlestick')
    except Exception as e:
        print("Plotting skipped.")