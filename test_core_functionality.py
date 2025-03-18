"""
Core functionality test script for the trading bot.
Tests API connection, data fetching, and indicator calculation with improved error handling.
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import BINANCE_CONFIG
from data.data_collector import DataCollector
from data.indicators import TechnicalIndicators

def test_api_connection(max_retries=3):
    """Test the Binance API connection with retries."""
    print("\nTesting Binance API Connection...")
    
    for attempt in range(max_retries):
        try:
            client = Client(
                BINANCE_CONFIG["API_KEY"],
                BINANCE_CONFIG["API_SECRET"],
                testnet=BINANCE_CONFIG["USE_TESTNET"]
            )
            
            # Test server time
            server_time = client.get_server_time()
            server_time_str = datetime.fromtimestamp(server_time['serverTime'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            print(f"✓ Successfully connected to Binance {'Testnet' if BINANCE_CONFIG['USE_TESTNET'] else 'Live'}")
            print(f"✓ Server time: {server_time_str}")
            
            # Test account info
            account = client.get_account()
            print(f"✓ Successfully retrieved account info")
            
            return True, client
            
        except BinanceAPIException as e:
            print(f"✗ Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                print("✗ All connection attempts failed")
                return False, None
        except Exception as e:
            print(f"✗ Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                print("✗ All connection attempts failed")
                return False, None
        
def test_data_fetching(client, max_retries=3):
    """Test fetching historical data with retries."""
    print("\nTesting Historical Data Fetching...")
    
    for attempt in range(max_retries):
        try:
            data_collector = DataCollector(client)
            symbol = "BTCUSDT"
            interval = "5m"
            
            # Get recent data
            df = data_collector.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str="1 day ago UTC"
            )
            
            if df is not None and not df.empty:
                print(f"✓ Successfully fetched {len(df)} datapoints for {symbol}")
                print(f"✓ Time range: {df.index[0]} to {df.index[-1]}")
                
                # Validate data quality
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"✗ Missing required columns: {missing_columns}")
                    continue
                
                # Check for NaN values
                nan_columns = df[required_columns].isna().sum()
                if nan_columns.any():
                    print(f"⚠ Found NaN values in columns: {nan_columns[nan_columns > 0]}")
                    # Forward fill NaN values
                    df = df.ffill().bfill()
                
                return True, df
            else:
                print(f"✗ Attempt {attempt + 1}/{max_retries}: No data received")
                if attempt == max_retries - 1:
                    return False, None
                
        except Exception as e:
            print(f"✗ Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                return False, None

def test_indicators(df, max_retries=3):
    """Test technical indicator calculations with retries."""
    print("\nTesting Technical Indicators...")
    
    for attempt in range(max_retries):
        try:
            ti = TechnicalIndicators()
            
            # Create a copy of the dataframe to avoid modifying the original
            df_copy = df.copy()
            
            # Forward fill any NaN values before calculation
            df_copy = df_copy.ffill().bfill()
            
            # Calculate indicators
            df_with_indicators = ti.add_all_indicators(df_copy)
            
            # Check which indicators were successfully calculated
            new_columns = set(df_with_indicators.columns) - set(df.columns)
            
            if new_columns:
                print(f"✓ Successfully calculated {len(new_columns)} indicators:")
                for col in sorted(new_columns):
                    # Check for NaN values in the indicator
                    nan_count = df_with_indicators[col].isna().sum()
                    if nan_count > 0:
                        print(f"  - {col} (⚠ {nan_count} NaN values)")
                    else:
                        print(f"  - {col}")
                
                return True
            else:
                print(f"✗ Attempt {attempt + 1}/{max_retries}: No indicators were calculated")
                if attempt == max_retries - 1:
                    return False
                
        except Exception as e:
            print(f"✗ Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                return False

def main():
    """Run all tests with improved error handling."""
    print("=== Running Core Functionality Tests ===")
    
    # Test 1: API Connection
    success, client = test_api_connection()
    if not success:
        print("✗ API connection test failed. Stopping further tests.")
        return
    
    # Test 2: Data Fetching
    success, df = test_data_fetching(client)
    if not success:
        print("✗ Data fetching test failed. Stopping further tests.")
        return
    
    # Test 3: Indicators
    success = test_indicators(df)
    if not success:
        print("✗ Indicator calculation test failed.")
        return
    
    print("\n=== All tests completed successfully! ===")

if __name__ == "__main__":
    main() 