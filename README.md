# Binance Trading Bot

A cryptocurrency trading bot that automates trading strategies on the Binance exchange platform. This bot implements various technical indicators and trading strategies with real-time data analysis for cryptocurrency trading.

## Features

- **Real-time Data Collection**: Fetches and processes cryptocurrency market data from Binance
- **Technical Analysis**: Implements multiple technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Strategy Implementation**: Includes multi-indicator strategies and DOGEBTC high-frequency strategies
- **Advanced Trading Patterns**: Optimized patterns for scalping, trend-following, and hybrid trading approaches
- **Risk Management**: Implements trailing stops, maximum drawdown limits, and position size management
- **Backtesting**: Tests strategies on historical data before live trading
- **Machine Learning Integration**: Optional ML prediction models for enhanced trading signals
- **Testnet Support**: Safely test trading strategies on Binance Testnet

## Project Structure

```
├── config/                  # Configuration files
│   └── config.py            # Main configuration settings
├── data/                    # Data handling modules
│   ├── data_collector.py    # Market data collection from Binance
│   └── indicators.py        # Technical indicators implementation
├── models/                  # Machine learning models
│   └── prediction_model.py  # Price prediction models
├── patterns/                # Trading pattern definitions
│   ├── default_pattern.json          # Basic default pattern
│   ├── scalping_pattern.json         # High-frequency trading pattern
│   ├── trend_following_pattern.json  # Long-term trend following pattern
│   ├── hybridized_pattern.json       # Optimized hybrid pattern
│   └── optimized_pattern.json        # Advanced optimized pattern
├── strategies/              # Trading strategies
│   ├── base_strategy.py             # Base strategy class
│   ├── multi_indicator_strategy.py  # Multi-indicator strategy implementation
│   └── dogebtc_hf_strategy.py       # DOGEBTC HF strategy implementation
├── utils/                   # Utility modules
│   ├── logger.py            # Logging utilities
│   ├── order_executor.py    # Order execution handler
│   └── pattern_loader.py    # Pattern loading and validation
├── tests/                   # Test scripts
│   └── test_indicators_demo.py  # Demo for technical indicators
├── logs/                    # Trading logs
├── trading_bot.py           # Main trading bot class
├── start_trading_bot.py     # Entry point for running the bot
├── test_trading_workflow.py # End-to-end testing script
├── GET_STARTED.md           # Comprehensive guide to using the bot
└── requirements.txt         # Project dependencies
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Binance account (or Testnet account for testing)
- API key and secret from Binance

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/binance_trading_bot.git
   cd binance_trading_bot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root directory:
   ```
   BINANCE_API_KEY=your_api_key
   BINANCE_API_SECRET=your_api_secret
   BINANCE_TESTNET_API_KEY=your_testnet_api_key
   BINANCE_TESTNET_API_SECRET=your_testnet_api_secret
   ```

## Usage

### Running Tests

Before live trading, test the bot's functionality:

```bash
python test_trading_workflow.py
```

This will run a complete end-to-end test of all components.

### Running Backtests

To run a backtest with a specific trading pattern:

```bash
python start_trading_bot.py --mode backtest --symbol BTCUSDT --timeframe 5m --pattern scalping_pattern.json --testnet
```

Supported timeframes include: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

### Running Live Trading

To run in live trading mode (use with caution):

```bash
python start_trading_bot.py --mode live --symbol BTCUSDT --timeframe 5m --pattern hybridized_pattern.json --testnet
```

### Command Line Options

- `--mode`: `backtest` or `live` (default: `backtest`)
- `--symbol`: Trading pair to watch (e.g., `BTCUSDT`)
- `--timeframe`: Data timeframe (e.g., `5m`, `15m`, `1h`)
- `--pattern`: Trading pattern JSON file to use (default: `default_pattern.json`)
- `--testnet`: Use Binance testnet (recommended for testing)
- `--log-level`: Logging level (e.g., `INFO`, `DEBUG`)
- `--risk`: Risk percentage per trade (default from config)
- `--max-positions`: Maximum number of concurrent positions (default from config)

## Trading Patterns

The bot includes several optimized trading patterns:

- **default_pattern.json**: Basic pattern with standard indicators
- **scalping_pattern.json**: High-frequency trading pattern for 1m-5m timeframes
- **trend_following_pattern.json**: Long-term trend pattern for 15m-1h timeframes
- **hybridized_pattern.json**: Balanced pattern that combines multiple approaches
- **optimized_pattern.json**: Advanced pattern with optimized indicators

For detailed information on how to create and customize patterns, please refer to the [GET_STARTED.md](GET_STARTED.md) file.

## Risk Warning

This trading bot is provided for educational and research purposes only. Cryptocurrency trading carries significant risk, and you should never invest money that you cannot afford to lose. The authors take no responsibility for financial losses incurred through the use of this software.

## Development

### Adding a New Pattern

1. Create a new JSON file in the `patterns` directory
2. Define buy and sell conditions with appropriate indicators and operators
3. Set risk parameters like take profit, stop loss, and trailing stop values
4. Test the pattern with backtesting before live trading

See [GET_STARTED.md](GET_STARTED.md) for detailed instructions on pattern creation.

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
