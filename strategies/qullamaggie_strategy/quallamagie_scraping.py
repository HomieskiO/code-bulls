import pandas as pd
import numpy as np
import glob
import os

# CONFIGURATION
# Update this path to where your extracted "Stocks" folder is located
DATA_PATH = '/Users/omerchomsky/mta/code_bulls/Stock Market Dataset/Stocks/*'
START_DATE = '1995-01-01'
END_DATE = '2015-12-31'
OUTPUT_FILE = 'top_3_percent_gainers.csv'


def process_stock_file(filepath):
    """
    Reads a single stock file, calculates returns, and filters by date.
    Returns a DataFrame or None if data is insufficient.
    """
    try:
        # Extract Ticker from filename (assuming format 'aapl.us.txt')
        filename = os.path.basename(filepath)
        ticker = filename.split('.')[0].upper()

        # Read CSV (The dataset typically has no header or specific headers, adjust accordingly)
        # Using common column names for this specific Kaggle dataset
        df = pd.read_csv(filepath)

        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Filter mostly for range first to reduce memory, but keep a buffer for rolling calc
        # We need ~126 days prior to 1995 for the 6-month calculation
        mask_load = df['Date'] >= pd.Timestamp(START_DATE) - pd.Timedelta(days=200)
        df = df[mask_load].copy()

        if df.empty:
            return None

        # Sort is critical for rolling calculations
        df = df.sort_values('Date')

        # Calculate Returns (Momentum)
        # 1 month ~ 21 trading days
        # 3 months ~ 63 trading days
        # 6 months ~ 126 trading days
        df['Ret_1m'] = df['Close'].pct_change(periods=21)
        df['Ret_3m'] = df['Close'].pct_change(periods=63)
        df['Ret_6m'] = df['Close'].pct_change(periods=126)

        # Now filter strictly for the requested user range
        mask_final = (df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)
        df = df[mask_final].copy()

        # Add Ticker column
        df['Ticker'] = ticker

        # Keep only necessary columns to save memory
        cols_to_keep = ['Date', 'Ticker', 'Close', 'Volume', 'Ret_1m', 'Ret_3m', 'Ret_6m']
        return df[cols_to_keep]

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def main():
    print("Finding files...")
    all_files = glob.glob(DATA_PATH)
    print(f"Found {len(all_files)} files.")

    processed_dfs = []

    print("Processing files (this may take time)...")
    for i, f in enumerate(all_files):
        res = process_stock_file(f)
        if res is not None and not res.empty:
            processed_dfs.append(res)

        if i % 500 == 0:
            print(f"Processed {i} files...")

    print("Concatenating data...")
    if not processed_dfs:
        print("No data found!")
        return

    full_df = pd.concat(processed_dfs, ignore_index=True)

    # Garbage collection to free up memory before the heavy lifting
    del processed_dfs

    print("Calculating daily quantiles...")

    # We need to find the top 3% for each timeframe independently.
    # We will create a mask for each timeframe.

    # 1. Group by Date to find the 97th percentile (0.97 quantile) for that day
    daily_thresholds = full_df.groupby('Date')[['Ret_1m', 'Ret_3m', 'Ret_6m']].quantile(0.97)

    # Rename columns to join
    daily_thresholds.columns = [f'{c}_thresh' for c in daily_thresholds.columns]

    # 2. Merge thresholds back to the main dataframe
    full_df = full_df.merge(daily_thresholds, on='Date', how='left')

    # 3. Filter: Keep row if it is in the top 3% for ANY of the timeframes
    # (If you want it to be top 3% in ALL timeframes, change | to &)

    mask_top = (
            (full_df['Ret_1m'] >= full_df['Ret_1m_thresh']) |
            (full_df['Ret_3m'] >= full_df['Ret_3m_thresh']) |
            (full_df['Ret_6m'] >= full_df['Ret_6m_thresh'])
    )

    top_gainers = full_df[mask_top].copy()

    # Optional: Add flags to know which specific timeframe it qualified for
    top_gainers['Is_Top_1m'] = top_gainers['Ret_1m'] >= top_gainers['Ret_1m_thresh']
    top_gainers['Is_Top_3m'] = top_gainers['Ret_3m'] >= top_gainers['Ret_3m_thresh']
    top_gainers['Is_Top_6m'] = top_gainers['Ret_6m'] >= top_gainers['Ret_6m_thresh']

    # Drop threshold columns to clean up
    top_gainers.drop(columns=['Ret_1m_thresh', 'Ret_3m_thresh', 'Ret_6m_thresh'], inplace=True)

    print(f"Filtering complete. Original rows: {len(full_df)}, Top rows: {len(top_gainers)}")

    print(f"Saving to {OUTPUT_FILE}...")
    top_gainers.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()