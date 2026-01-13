import backtrader as bt
import os
import datetime
import pandas as pd
import time

def fast_date_parse(date_string):
    # Assumes format 'YYYY-MM-DD'
    # Slicing strings is much faster than regex/strptime
    return datetime.datetime(
        int(date_string[0:4]),
        int(date_string[5:7]),
        int(date_string[8:10])
    )

class SimpleStrategy(bt.Strategy):
    def __init__(self):
        self.closes = {data._name: data.close for data in self.datas}

    def next(self):
        for data in self.datas:
            pos = self.getposition(data).size
            if not pos:
                self.buy(data=data, size=1)

cerebro = bt.Cerebro()
cerebro.addstrategy(SimpleStrategy)

# print("--- STARTING PROFILE ---")
# profiler = cProfile.Profile()
# profiler.enable()

DATA_DIR = '/Users/omerchomsky/mta/code_bulls/Stock Market Dataset/Stocks/'

df_gainers = pd.read_csv('top_percent_gainers.csv')
unique_tickers = df_gainers['Ticker'].unique()
target_filenames = {f"{ticker.lower()}.us.txt" for ticker in unique_tickers}

print(f"Targeting {len(target_filenames)} files from top gainers.")

for file_name in target_filenames:
    file_path = os.path.join(DATA_DIR, file_name)

    if not os.path.isfile(file_path):
        continue

    if os.path.getsize(file_path) == 0:
        print(f"Skipping empty file: {file_name}")
        continue

    try:
        data = bt.feeds.GenericCSVData(
            dataname=file_path,
            name=file_name,
            dtformat=fast_date_parse,
            date=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=6,
            preload=True
        )
        cerebro.adddata(data)

    except Exception as e:
        print(f"Error loading {file_name}: {e}")

cerebro.broker.setcash(100000.0)

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
start_time = time.time()
cerebro.run(stdstats=False, tradehistory=False)
end_time = time.time()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")