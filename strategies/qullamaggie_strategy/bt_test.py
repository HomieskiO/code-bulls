import bt
import pandas as pd
import glob
import os
import time

# 1. Data Preparation (Load all into one big DataFrame)
DATA_DIR = '/Users/omerchomsky/mta/code_bulls/Stock Market Dataset/Stocks/'
df_gainers = pd.read_csv('top_percent_gainers.csv')
target_tickers = [t.lower() + ".us.txt" for t in df_gainers['Ticker'].unique()]

prices = {}

print("Loading data...")
for file_name in target_tickers:
    file_path = os.path.join(DATA_DIR, file_name)
    if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
        try:
            # We only need the Close price for a simple Buy & Hold / Rebalance
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            ticker_name = file_name.replace('.us.txt', '').upper()
            prices[ticker_name] = df['Close']
        except Exception:
            continue

# Create a single DataFrame (Index=Date, Columns=Tickers)
price_data = pd.DataFrame(prices)
# Forward fill missing data (if a stock didn't trade that day) so we don't drop the row
price_data = price_data.fillna(method='ffill')

print(f"Loaded {len(price_data.columns)} assets.")

# 2. Define the Strategy
# "SelectAll" -> "WeighEqually" -> "Rebalance" mimics buying the whole basket.
# logic: Run once (at start), select everything available, give equal weight, send orders.
s = bt.Strategy('Simple_ETF_Strategy', [
    bt.algos.RunDaily(),
    bt.algos.SelectAll(),
    bt.algos.WeighEqually(),
    bt.algos.Rebalance()
])

# 3. Run Backtest
test = bt.Backtest(s, price_data, initial_capital=100000.0)
start_time = time.time()
res = bt.run(test)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")

# 4. Results
res.plot()
print("Final Value: %.2f" % res.stats.loc['total_return'])
res.display() # Prints a nice table of stats

