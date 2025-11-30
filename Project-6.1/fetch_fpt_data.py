import pandas as pd
from vnstock import Vnstock
from datetime import datetime, timedelta
import os

# Configuration
SYMBOL = 'FPT'
START_DATE = '2025-03-11'
REQUIRED_DAYS = 100
OUTPUT_FILE = 'data/FPT_hidden-test.csv'

def fetch_data():
    print(f"Fetching data for {SYMBOL} starting from {START_DATE}...")
    
    # Calculate a safe end date
    start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_dt = start_dt + timedelta(days=300) # Increased buffer
    end_date_str = end_dt.strftime('%Y-%m-%d')
    
    try:
        # Fetch data using vnstock 3.3.0 API
        stock = Vnstock().stock(symbol=SYMBOL, source='VCI')
        df = stock.quote.history(start=START_DATE, end=end_date_str)
        
        if df is None or df.empty:
            print("No data returned from vnstock.")
            return

        # Add symbol column if missing
        if 'symbol' not in df.columns:
            df['symbol'] = SYMBOL
            
        # Ensure 'time' is datetime and sorted
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        # Filter to ensure we start from START_DATE (in case API returns earlier data)
        df = df[df['time'] >= pd.to_datetime(START_DATE)]
        
        # Filter for business days (vnstock usually returns trading days only)
        # We take the first 100 rows
        if len(df) < REQUIRED_DAYS:
            print(f"Warning: Only fetched {len(df)} days. Expected at least {REQUIRED_DAYS}.")
        
        df_subset = df.head(REQUIRED_DAYS)
        
        print(f"Fetched {len(df_subset)} days.")
        if not df_subset.empty:
            print(f"Date range: {df_subset['time'].min().date()} to {df_subset['time'].max().date()}")
        
        # Save to CSV
        # Format time as YYYY-MM-DD
        df_subset['time'] = df_subset['time'].dt.strftime('%Y-%m-%d')
        
        # Reorder columns to match: time,open,high,low,close,volume,symbol
        cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        # Check if all columns exist
        available_cols = [c for c in cols if c in df_subset.columns]
        df_subset = df_subset[available_cols]
        
        df_subset.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fetch_data()
