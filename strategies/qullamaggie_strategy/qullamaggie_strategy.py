import pandas as pd
import numpy as np

# CONFIGURATION
INPUT_FILE = 'top_3_percent_gainers.parquet'
OUTPUT_FILE = 'qullamaggie_candidates_close_only.csv'


def find_breakouts(df):
    """
    Identifies Qullamaggie setups using only Close/Volume data.
    """
    print(f"Processing {len(df)} rows...")

    # 1. SETUP MOVING AVERAGES (The "Surf")
    # We use .transform() to keep the data aligned with the original DataFrame
    df['SMA_10'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(10).mean())
    df['SMA_20'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).mean())
    df['Vol_SMA_20'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(20).mean())

    # 2. DEFINE CONSOLIDATION (The "Flag") - CLOSE PRICE PROXY
    # Since we lack High/Low, we look for tightness in the CLOSING prices.
    # We look back 10 days (2 weeks).

    # Highest Close in the last 10 days
    df['Close_Max_10d'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(10).max())
    # Lowest Close in the last 10 days
    df['Close_Min_10d'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(10).min())

    # Calculate "Closing Tightness"
    # Formula: (Max Close - Min Close) / Max Close
    # Note: We use a stricter threshold (0.10 or 10%) because closing ranges are naturally
    # tighter than High-Low ranges.
    df['Tightness'] = (df['Close_Max_10d'] - df['Close_Min_10d']) / df['Close_Max_10d']

    # 3. IDENTIFY THE BREAKOUT

    # Condition A: Surfing the Moving Average
    # The price should hold the 20-day MA.
    surf_condition = df['Close'] > df['SMA_20']

    # Condition B: Tightness Setup (Pre-Breakout)
    # The stock must have been tight *yesterday* (before today's move).
    # We look for tightness < 10% in closing prices.
    tight_condition = df['Tightness'].shift(1) < 0.10

    # Condition C: The Breakout Trigger
    # Today's Close > The Highest Close of the previous 10 days
    prior_high_close = df['Close_Max_10d'].shift(1)
    breakout_condition = df['Close'] > prior_high_close

    # Condition D: Momentum Context
    # We only want setups that are happening while the stock is a "Top Gainer".
    # We use the columns already in your file: Is_Top_1m, Is_Top_3m, or Is_Top_6m.
    # You can change '|' (OR) to '&' (AND) if you want to be stricter.
    momentum_condition = (
            (df['Is_Top_1m'] == True) |
            (df['Is_Top_3m'] == True) |
            (df['Is_Top_6m'] == True)
    )

    # Condition E: Volume Surge (Optional)
    # Volume > Average Volume
    volume_condition = df['Volume'] > df['Vol_SMA_20']

    # COMBINE ALL FILTERS
    setup_mask = (
            momentum_condition &
            surf_condition &
            tight_condition &
            breakout_condition &
            volume_condition
    )

    result = df[setup_mask].copy()

    # Clean up helper columns for the final output
    cols_to_drop = ['SMA_10', 'SMA_20', 'Vol_SMA_20', 'Close_Max_10d', 'Close_Min_10d']
    result.drop(columns=cols_to_drop, inplace=True)

    return result


def main():
    try:
        # Load the parquet file
        print("Loading Parquet file...")
        df = pd.read_parquet(INPUT_FILE)

        # Sort is critical for rolling window calculations
        df = df.sort_values(['Ticker', 'Date'])

        # Run the detection
        breakouts = find_breakouts(df)

        print(f"Found {len(breakouts)} potential breakout setups.")

        if not breakouts.empty:
            print("\nSample Breakouts:")
            print(breakouts[['Date', 'Ticker', 'Close', 'Volume', 'Tightness']].head())

            print(f"\nSaving results to {OUTPUT_FILE}...")
            breakouts.to_csv(OUTPUT_FILE, index=False)
            print("Done.")
        else:
            print("No breakouts found with current thresholds.")

    except FileNotFoundError:
        print(f"Error: Could not find file '{INPUT_FILE}'. Check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()