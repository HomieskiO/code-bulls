import vectorbt as vbt
import pandas as pd
import os
import time

# Define paths
DATA_DIR = '/Users/omerchomsky/mta/code_bulls/Stock Market Dataset/Stocks/'

# 1. Load the list of tickers
df_gainers = pd.read_csv('top_percent_gainers.csv')
unique_tickers = df_gainers['Ticker'].unique()
target_filenames = {f"{ticker.lower()}.us.txt" for ticker in unique_tickers}

print(f"Targeting {len(target_filenames)} files from top gainers.")

# 2. Efficiently load data into a dictionary
data_dict = {}
valid_files = 0

print("Loading data...")
for file_name in target_filenames:
    file_path = os.path.join(DATA_DIR, file_name)

    if not os.path.isfile(file_path):
        continue

    if os.path.getsize(file_path) == 0:
        print(f"Skipping empty file: {file_name}")
        continue

    try:
        # We only need the 'Close' column for a simple Buy & Hold or Close-based strategy
        # To mimic the exact backtrader strategy, we usually just need price data.
        df = pd.read_csv(
            file_path,
            parse_dates=['Date'],
            index_col='Date'
        )

        # Store just the Close price for the main simulation DataFrame
        # The key will be the filename (or ticker name)
        data_dict[file_name] = df['Close']
        valid_files += 1

    except Exception as e:
        print(f"Error loading {file_name}: {e}")

print(f"Successfully loaded {valid_files} files.")

if valid_files == 0:
    print("No valid files loaded. Exiting.")
    exit()

# 3. Combine into a single DataFrame (Rows=Dates, Columns=Tickers)
# This aligns all dates automatically, filling missing values with NaN
price_data = pd.DataFrame(data_dict)

# 4. Define the Strategy (Simple Buy and Hold)
print('Starting Backtest...')
start_time = time.time()

# In VectorBT, a "Buy and Hold" is equivalent to being in the market at the first valid index.
# We generate a signals DataFrame of the same shape as price_data
# True at the first valid timestamp (Buy), False otherwise.
entries = (~price_data.isnull()).vbt.signals.first()
exits = pd.DataFrame(False, index=entries.index, columns=entries.columns)

# 5. Run Portfolio Simulation
portfolio = vbt.Portfolio.from_signals(
    price_data,
    entries,
    exits,
    init_cash=100000.0,
    # In your backtrader code, you bought "size=1" (1 share).
    # To match that exactly:
    size=1,
    size_type='amount',  # 'amount' means fixed number of shares (1), not value or percent
    freq='1D'  # Daily frequency for annualization metrics
)

end_time = time.time()

# 6. Results
print(f'Starting Portfolio Value: {portfolio.init_cash.sum():.2f}')
print(
    f'Final Portfolio Value: {portfolio.final_value().sum():.2f}')  # Sum of all individual "stock strategies" if treated independently, or use total_value for a unified portfolio view depending on how you want to aggregate.

# Note: The above creates a "portfolio" object that actually simulates ONE portfolio per column (stock).
# To get the aggregate value of holding 1 share of ALL these stocks together starting from 100k cash *each* (default behavior) or shared cash:
# VectorBT defaults to 100k per column.
# To mimic "One single account buying 1 share of each":
total_final_value = portfolio.cash().iloc[-1].sum() + portfolio.asset_value().iloc[-1].sum()
# Actually, a cleaner way to see the "Total Equity" of the entire operation:
print(f'Total Final Assets Value (All holdings combined): {portfolio.asset_value().iloc[-1].sum():.2f}')

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")