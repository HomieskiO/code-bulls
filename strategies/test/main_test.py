import sys

import backtrader as bt
import pandas as pd
import os
import datetime
import time
import random
import stats
from stats.analyzers import ThreeLayerAnalyzer
from stats.exporter import save_single_object_csv, save_csv


# --- CONFIGURATION ---
DATA_DIR = '/Users/noamfishbain/PycharmProjects/code-bulls/strategies/test'
CSV_FILE = 'top_percent_gainers.csv'
CASH = 100000.0


def fast_date_parse(date_string):
    """
    Parses dates in 'YYYY-MM-DD' format efficiently.
    """
    return datetime.datetime(
        int(date_string[0:4]),
        int(date_string[5:7]),
        int(date_string[8:10])
    )


class DummyTestStrategy(bt.Strategy):
    """
    A dummy strategy designed specifically to generate trades for testing reports.
    Logic:
    1. If not in position: BUY.
    2. If in position for 10 bars: SELL.
    """

    def __init__(self):
        # Dictionary to track how many bars we've held each position
        self.hold_time = {data._name: 0 for data in self.datas}

    def next(self):
        for data in self.datas:
            name = data._name
            pos = self.getposition(data).size

            if not pos:
                # No position? Buy!
                # We use a small size to allow buying multiple stocks
                self.buy(data=data, size=10)
                self.hold_time[name] = 0
            else:
                # Have position? Increment hold counter
                self.hold_time[name] += 1

                # Sell after holding for 10 bars (guarantees a closed trade)
                if self.hold_time[name] >= 10:
                    self.close(data=data)
                    self.hold_time[name] = 0


# --- MAIN EXECUTION ---

def run_test():
    print("--- STARTING TEST RUN ---")
    cerebro = bt.Cerebro()

    # 1. Add the dummy strategy
    cerebro.addstrategy(DummyTestStrategy)

    # 2. Add Analyzers for the reports
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
    cerebro.addanalyzer(ThreeLayerAnalyzer, _name='mega_report')

    # 3. Load Data from CSV list
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found in current directory.")
        return

    print(f"Reading tickers from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    unique_tickers = df['Ticker'].unique()

    # Limit to first 20 tickers to speed up the test
    test_tickers = unique_tickers[:20]
    print(f"Testing with {len(test_tickers)} tickers: {test_tickers}")

    files_loaded = 0
    for ticker in test_tickers:
        filename = f"{ticker.lower()}.us.txt"
        file_path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue

        try:
            data = bt.feeds.GenericCSVData(
                dataname=file_path,
                name=ticker,  # Important: This name appears in the reports
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
            files_loaded += 1
        except Exception as e:
            print(f"Error loading {ticker}: {e}")

    if files_loaded == 0:
        print("CRITICAL: No data files were loaded. Please check DATA_DIR path.")
        return

    # 4. Setup Broker
    cerebro.broker.setcash(CASH)
    print(f'Starting Portfolio Value: {CASH:.2f}')

    # 5. Run
    start_time = time.time()
    results = cerebro.run()
    end_time = time.time()

    # 6. Generate Reports
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    if results:
        strat = results[0]
        report_data = strat.analyzers.mega_report.get_results()

        print("--- GENERATING REPORTS ---")

        # Report 1: Trade Log
        log_file = 'test_report_1_trade_log.csv'
        save_csv(report_data['log'], log_file)

        # Report 2: Symbol Performance
        symbol_file = 'test_report_2_by_symbol.csv'
        save_csv(report_data['by_symbol'], symbol_file)

        # Report 3: Strategy Summary
        summary_file = 'test_report_3_strategy_summary.csv'
        save_single_object_csv(report_data['global'], summary_file)

        print(f"SUCCESS! Check the files:\n1. {log_file}\n2. {symbol_file}\n3. {summary_file}")
    else:
        print("Error: Strategy returned no results.")


if __name__ == '__main__':
    run_test()