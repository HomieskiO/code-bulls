import pandas as pd
import numpy as np
import glob
import os
import argparse
import yfinance as yf

# CONFIGURATION (Defaults)
# Update this path to where your extracted "Stocks" folder is located
DATA_PATH = '/Users/omerchomsky/mta/code_bulls/Stock Market Dataset/Stocks/*'

# Simulation Range
START_DATE = '1995-01-01'
END_DATE = '2015-12-31'
OUTPUT_FILE = 'top_percent_gainers.csv'

# Default Parameters
DEFAULT_TOP_PERCENT = 1.0  # Top 2%
DEFAULT_MIN_DOLLAR_VOL = 5_000_000  # $5 Million
DEFAULT_MIN_ADR = 3.5  # 3.5% Average Daily Range


def get_market_regime_dates():
    """
    Downloads Nasdaq Composite (^IXIC) from Yahoo Finance.
    Returns a set of dates where:
    1. SMA10 > SMA20
    2. SMA10 is trending up
    3. SMA20 is trending up
    """
    print("Fetching Nasdaq Composite (^IXIC) data from Yahoo Finance...")
    try:
        # Download data with enough buffer before START_DATE for MA calculations
        # We go back to 1990 to be safe.
        df_index = yf.download('^IXIC', start='1990-01-01', end=pd.Timestamp(END_DATE) + pd.Timedelta(days=30),
                               progress=False)

        # Handle yfinance MultiIndex columns (common in newer versions)
        if isinstance(df_index.columns, pd.MultiIndex):
            df_index.columns = df_index.columns.get_level_values(0)

        # Reset index so Date is a column
        df_index.reset_index(inplace=True)

        # Ensure proper datetime format
        df_index['Date'] = pd.to_datetime(df_index['Date'])
        df_index = df_index.sort_values('Date')

        # Calculate SMAs
        df_index['SMA10'] = df_index['Close'].rolling(window=10).mean()
        df_index['SMA20'] = df_index['Close'].rolling(window=20).mean()

        # Check Trends (Current > Previous)
        df_index['SMA10_Up'] = df_index['SMA10'] > df_index['SMA10'].shift(1)
        df_index['SMA20_Up'] = df_index['SMA20'] > df_index['SMA20'].shift(1)

        # Define Allowed Dates Mask
        mask_bullish = (
                (df_index['SMA10'] > df_index['SMA20']) &
                (df_index['SMA10_Up']) &
                (df_index['SMA20_Up'])
        )

        # Extract only the dates that meet criteria
        allowed_dates = set(df_index[mask_bullish]['Date'])

        print(
            f"Market Regime Analysis: Found {len(allowed_dates)} valid bullish days out of {len(df_index)} total days.")
        return allowed_dates

    except Exception as e:
        print(f"CRITICAL ERROR fetching market data: {e}")
        return None


def process_stock_file(filepath, allowed_dates):
    """
    Reads a single stock file.
    Filters by Date (Market Regime), Liquidity, and ADR.
    """
    try:
        # Skip empty files
        if os.stat(filepath).st_size == 0:
            return None

        filename = os.path.basename(filepath)
        ticker = filename.split('.')[0].upper()

        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])

        # Optimization: Pre-filter to reduce memory (keep 200 day buffer for local indicators)
        mask_load = df['Date'] >= pd.Timestamp(START_DATE) - pd.Timedelta(days=200)
        df = df[mask_load].copy()

        if df.empty:
            return None

        df = df.sort_values('Date')

        # --- CALCULATE INDICATORS (Must happen BEFORE filtering by allowed_dates) ---

        # 1. Dollar Volume (SMA50)
        df['Dollar_Vol'] = df['Close'] * df['Volume']
        df['Dollar_Vol_SMA50'] = df['Dollar_Vol'].rolling(window=50).mean()

        # 2. ADR (20-day)
        # Handle zero division for Low
        df['Low'] = df['Low'].replace(0, np.nan)
        df['Daily_Range_Pct'] = (df['High'] / df['Low'] - 1) * 100
        df['ADR_20'] = df['Daily_Range_Pct'].rolling(window=20).mean()

        # 3. Returns (Momentum)
        df['Ret_1m'] = df['Close'].pct_change(periods=21)
        df['Ret_3m'] = df['Close'].pct_change(periods=63)
        df['Ret_6m'] = df['Close'].pct_change(periods=126)

        # --- FILTER 1: User Date Range ---
        mask_range = (df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)
        df = df[mask_range].copy()

        # --- FILTER 2: Market Regime (Nasdaq Check) ---
        # Keep only rows where the Date exists in the allowed_dates set
        if allowed_dates is not None:
            df = df[df['Date'].isin(allowed_dates)].copy()

        if df.empty:
            return None

        df['Ticker'] = ticker

        cols_to_keep = ['Date', 'Ticker', 'Close', 'Volume',
                        'Ret_1m', 'Ret_3m', 'Ret_6m',
                        'Dollar_Vol_SMA50', 'ADR_20']

        return df[cols_to_keep]

    except Exception as e:
        # Silently fail on data errors to keep the loop moving
        return None


def main():
    # --- ARGS ---
    parser = argparse.ArgumentParser(
        description="Filter stocks by Market Regime, Performance, Liquidity, and Volatility.")

    parser.add_argument('--top_percent', type=float, default=DEFAULT_TOP_PERCENT,
                        help=f'Top percentage threshold (default: {DEFAULT_TOP_PERCENT})')

    parser.add_argument('--min_volume', type=float, default=DEFAULT_MIN_DOLLAR_VOL,
                        help=f'Minimum Avg Dollar Volume (default: {DEFAULT_MIN_DOLLAR_VOL})')

    parser.add_argument('--min_adr', type=float, default=DEFAULT_MIN_ADR,
                        help=f'Minimum 20-day ADR %% (default: {DEFAULT_MIN_ADR})')

    args = parser.parse_args()

    # --- STEP 1: GET MARKET REGIME ---
    allowed_dates = get_market_regime_dates()

    if not allowed_dates:
        print("Error: Could not determine market regime dates. Aborting.")
        return

    # --- STEP 2: PROCESS FILES ---
    print("Finding files...")
    all_files = glob.glob(DATA_PATH)
    print(f"Found {len(all_files)} files.")

    processed_dfs = []

    print("Processing stocks (this may take time)...")
    for i, f in enumerate(all_files):
        res = process_stock_file(f, allowed_dates)
        if res is not None and not res.empty:
            processed_dfs.append(res)

        if i % 500 == 0 and i > 0:
            print(f"Processed {i} files...")

    print("Concatenating data...")
    if not processed_dfs:
        print("No stocks found matching the criteria!")
        return

    full_df = pd.concat(processed_dfs, ignore_index=True)

    # Free memory
    del processed_dfs

    print(f"Total rows passed Market Regime check: {len(full_df)}")

    # --- STEP 3: APPLY STOCK-SPECIFIC FILTERS ---
    print(f"Applying Filters: Min ${args.min_volume:,.0f} Vol, Min {args.min_adr}% ADR")

    mask_valid = (
            (full_df['Dollar_Vol_SMA50'] >= args.min_volume) &
            (full_df['ADR_20'] >= args.min_adr)
    )

    full_df = full_df[mask_valid].copy()

    print(f"Rows after Filters: {len(full_df)}")

    if full_df.empty:
        print("No stocks met the Liquidity/ADR criteria.")
        return

    # --- STEP 4: CALCULATE RELATIVE PERFORMANCE (Top %) ---
    quantile_val = 1 - (args.top_percent / 100.0)
    print(f"Calculating daily top {args.top_percent}% (Quantile: {quantile_val:.4f})...")

    # 1. Group by Date to find the threshold for that specific day
    daily_thresholds = full_df.groupby('Date')[['Ret_1m', 'Ret_3m', 'Ret_6m']].quantile(quantile_val)
    daily_thresholds.columns = [f'{c}_thresh' for c in daily_thresholds.columns]

    # 2. Merge thresholds back
    full_df = full_df.merge(daily_thresholds, on='Date', how='left')

    # 3. Filter for Top Performers
    mask_top = (
            (full_df['Ret_1m'] >= full_df['Ret_1m_thresh']) |
            (full_df['Ret_3m'] >= full_df['Ret_3m_thresh']) |
            (full_df['Ret_6m'] >= full_df['Ret_6m_thresh'])
    )

    top_gainers = full_df[mask_top].copy()

    # Optional: Add flags
    top_gainers['Is_Top_1m'] = top_gainers['Ret_1m'] >= top_gainers['Ret_1m_thresh']
    top_gainers['Is_Top_3m'] = top_gainers['Ret_3m'] >= top_gainers['Ret_3m_thresh']
    top_gainers['Is_Top_6m'] = top_gainers['Ret_6m'] >= top_gainers['Ret_6m_thresh']

    # Cleanup
    top_gainers.drop(columns=['Ret_1m_thresh', 'Ret_3m_thresh', 'Ret_6m_thresh'], inplace=True)

    print(f"Filtering complete. Final Top Rows: {len(top_gainers)}")

    print(f"Saving to {OUTPUT_FILE}...")
    top_gainers.to_csv(OUTPUT_FILE, index=False)
    print("Done.")


if __name__ == "__main__":
    main()