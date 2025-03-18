"""
Basic functionality test script for the trading bot.
Tests API connection, data fetching, and indicator calculation.
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import BINANCE_CONFIG
from data.data_collector import DataCollector
from data.indicators import TechnicalIndicators

def test_api_connection():
    """Test the Binance API connection."""
    print("\nTesting Binance API Connection...")
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
        
        return True, client
    except BinanceAPIException as e:
        print(f"✗ Binance API Error: {e}")
        return False, None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False, None

def test_data_fetching(client):
    """Test fetching historical data."""
    print("\nTesting Historical Data Fetching...")
    try:
        data_collector = DataCollector(client)
        symbol = "BTCUSDT"
        interval = "5m"
        limit = 100
        
        # Get recent data
        df = data_collector.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str="1 day ago UTC"
        )
        
        if df is not None and not df.empty:
            print(f"✓ Successfully fetched {len(df)} datapoints for {symbol}")
            print(f"✓ Time range: {df.index[0]} to {df.index[-1]}")
            return True, df
        else:
            print("✗ No data received")
            return False, None
            
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return False, None

def test_indicators(df):
    """Test technical indicator calculations."""
    print("\nTesting Technical Indicators...")
    try:
        ti = TechnicalIndicators()
        df_with_indicators = ti.add_all_indicators(df)
        
        # Check which indicators were successfully calculated
        new_columns = set(df_with_indicators.columns) - set(df.columns)
        print(f"✓ Successfully calculated {len(new_columns)} indicators:")
        for col in sorted(new_columns):
            print(f"  - {col}")
            
        return True
    except Exception as e:
        print(f"✗ Error calculating indicators: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Running Basic Functionality Tests ===")
    
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