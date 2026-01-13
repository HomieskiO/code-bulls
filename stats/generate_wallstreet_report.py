import pandas as pd
import quantstats as qs
import webbrowser
import os

# --- הגדרות ---
# שים כאן את השם של קובץ הלוג האחרון שלך
LOG_FILE = '/Users/noamfishbain/PycharmProjects/code-bulls/strategies/test/nasdaq_1_trade_log_20260113_2007.csv'  # עדכן לשם האמיתי!
START_CASH = 100000.0


def create_quantstats_report():
    print("Generating Wall St. Grade Report...")

    # 1. טעינת הנתונים
    try:
        df = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        print(f"Error: File {LOG_FILE} not found.")
        return

    # 2. המרה לפורמט ש-QuantStats מבין (Daily Returns)
    # אנחנו צריכים להמיר את רשימת העסקאות לטור של רווח/הפסד יומי
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df = df.set_index('exit_time')
    df = df.sort_index()

    # קיבוץ רווחים לפי ימים (אם היו כמה עסקאות ביום)
    daily_pnl = df['pnl_net'].resample('D').sum().fillna(0)

    # יצירת סדרת תשואות יומית (באחוזים מהתיק)
    # כדי שזה יהיה מדויק, אנחנו צריכים לחשב את שווי התיק המצטבר
    cumulative_pnl = daily_pnl.cumsum()
    equity_curve = START_CASH + cumulative_pnl

    # חישוב אחוז השינוי היומי (Returns)
    returns = equity_curve.pct_change().fillna(0)

    # 3. יצירת הדוח
    # הספרייה מורידה אוטומטית את מדד ה-S&P500 (SPY) להשוואה!
    output_file = "stats_report.html"

    qs.reports.html(
        returns,
        benchmark='SPY',
        output=output_file,
        title='My Algo Strategy Performance'
    )

    print(f"Success! Opening {output_file}...")
    webbrowser.open('file://' + os.path.realpath(output_file))


if __name__ == '__main__':
    create_quantstats_report()