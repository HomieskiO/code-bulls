import backtrader as bt
import datetime
import sys
import os

import pandas as pd

# --- 1. SETUP & IMPORTS ---
# וודא שהנתיב הזה נכון לתיקייה שבה שמרת את models.py, analyzers.py
# TOOLS_PATH = '/Users/noamfishbain/PycharmProjects/code-bulls/stats'
# sys.path.append(TOOLS_PATH)

try:
    import yfinance as yf
except ImportError:
    print("CRITICAL: yfinance not installed. Run 'pip install yfinance'")
    sys.exit(1)

# ניסיון לייבא את הכלים שלך
try:
    from analyzers import ThreeLayerAnalyzer
    from exporter import save_csv, save_single_object_csv
except ImportError:
    # נסיון גיבוי למצוא את הקבצים בתיקייה למעלה
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    try:
        from stats.analyzers import ThreeLayerAnalyzer
        from stats.exporter import save_csv, save_single_object_csv
    except ImportError:
        print("Error: Could not import 'analyzers' or 'exporter'. Check your paths.")
        sys.exit(1)

# --- 2. CONFIGURATION ---
START_CASH = 100000.0
HOLD_PERIOD = 30  # ימים להחזיק במניה

# רשימת 20 מניות גדולות בנאסד"ק לבדיקה
TOP_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX',
    'AMD', 'INTC', 'CSCO', 'CMCSA', 'PEP', 'ADBE', 'AVGO', 'TXN',
    'QCOM', 'TMUS', 'COST', 'SBUX'
]


class BuyAndHoldStrategy(bt.Strategy):
    """
    אסטרטגיה פשוטה לייצור דוחות:
    1. קונה כל מניה ביום הראשון שהיא מופיעה.
    2. מחזיקה אותה 30 יום.
    3. מוכרת (כדי שיהיה Closed Trade ללוג).
    """

    def __init__(self):
        self.hold_counters = {data._name: 0 for data in self.datas}

    def next(self):
        for data in self.datas:
            name = data._name
            pos = self.getposition(data).size

            if not pos:
                # קנייה בנר הראשון
                # קונים כמות קטנה (10 יחידות) כדי שישאר כסף לכולן
                self.buy(data=data, size=10)
                self.hold_counters[name] = 0
            else:
                # ניהול פוזיציה קיימת
                self.hold_counters[name] += 1

                # מכירה אחרי X ימים
                if self.hold_counters[name] >= HOLD_PERIOD:
                    self.close(data=data)
                    self.hold_counters[name] = 0


def run_nasdaq_test():
    print(f"--- STARTING NASDAQ TEST ({len(TOP_STOCKS)} Stocks) ---")
    cerebro = bt.Cerebro()

    # --- הוספת דאטה מ-Yahoo Finance ---
    print("Downloading data...")
    start_date = datetime.datetime(2023, 1, 1)

    loaded_count = 0
    for ticker in TOP_STOCKS:
        try:
            # הורדת נתונים
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)

            # --- תיקון קריטי ל-YFINANCE החדש ---
            # אם העמודות הן MultiIndex (כמו ('Close', 'AAPL')), אנחנו משטחים אותן
            if isinstance(df.columns, pd.MultiIndex):
                # לוקחים רק את השם של העמודה (Price) וזורקים את שם הטיקר
                df.columns = df.columns.get_level_values(0)

            # וידוא שאין עמודות כפולות או בעיות אחרות
            df = df.dropna()

            if len(df) > 0:
                data = bt.feeds.PandasData(dataname=df, name=ticker)
                cerebro.adddata(data)
                loaded_count += 1
                print(f"Loaded: {ticker}")
            else:
                print(f"Empty data for {ticker}")

        except Exception as e:
            print(f"Failed to load {ticker}: {e}")

    if loaded_count == 0:
        print("No data loaded. Check internet connection.")
        return

    # --- הגדרת אסטרטגיה ודוחות ---
    cerebro.addstrategy(BuyAndHoldStrategy)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
    cerebro.addanalyzer(ThreeLayerAnalyzer, _name='mega_report')

    cerebro.broker.setcash(START_CASH)

    print(f'\nStarting Portfolio Value: {START_CASH:,.2f}')
    results = cerebro.run()
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():,.2f}')

    # --- שמירת הדוחות ---
    if results:
        strat = results[0]
        report_data = strat.analyzers.mega_report.get_results()

        print("\n--- SAVING REPORTS ---")
        save_csv(report_data['log'], 'nasdaq_1_trade_log.csv')
        save_csv(report_data['by_symbol'], 'nasdaq_2_symbol_report.csv')
        save_single_object_csv(report_data['global'], 'nasdaq_3_strategy_summary.csv')

        print("Done! Files created successfully.")

if __name__ == '__main__':
    run_nasdaq_test()