# Binance Trading Bot

A cryptocurrency trading bot that automates trading strategies on the Binance exchange platform. This bot implements various technical indicators and trading strategies with real-time data analysis for cryptocurrency trading.

## Features

- **Real-time Data Collection**: Fetches and processes cryptocurrency market data from Binance
- **Technical Analysis**: Implements multiple technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Strategy Implementation**: Includes multi-indicator strategies and DOGEBTC high-frequency strategies
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
├── strategies/              # Trading strategies
│   ├── base_strategy.py             # Base strategy class
│   ├── multi_indicator_strategy.py  # Multi-indicator strategy implementation
│   └── dogebtc_hf_strategy.py       # DOGEBTC HF strategy implementation
├── utils/                   # Utility modules
│   ├── logger.py            # Logging utilities
│   └── order_executor.py    # Order execution handler
├── tests/                   # Test scripts
│   └── test_indicators_demo.py  # Demo for technical indicators
├── logs/                    # Trading logs
├── trading_bot.py           # Main trading bot class
├── run_bot.py               # Entry point for running the bot
├── test_trading_workflow.py # End-to-end testing script
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

This will run a complete end-to-end test of all components, including:
- API connection
- Data collection
- Technical indicator calculation
- Strategy signal generation
- Order execution (simulation mode)

### Running the Bot

To start the trading bot in backtest mode:

```bash
python run_bot.py --mode backtest --symbols BTCUSDT ETHUSDT --start-date 2023-01-01
```

To run in live trading mode (use with caution):

```bash
python run_bot.py --mode live --symbols BTCUSDT --interval 15
```

### Command Line Options

- `--mode`: `backtest` or `live` (default: `backtest`)
- `--symbols`: Trading pairs to watch (e.g., `BTCUSDT ETHUSDT`)
- `--interval`: Update interval in minutes for live mode (default: 15)
- `--start-date`: Start date for backtest (format: YYYY-MM-DD)
- `--end-date`: End date for backtest (format: YYYY-MM-DD)
- `--initial-balance`: Initial balance for backtest (default: 1000)
- `--use-ml`: Enable machine learning predictions

## Risk Warning

This trading bot is provided for educational and research purposes only. Cryptocurrency trading carries significant risk, and you should never invest money that you cannot afford to lose. The authors take no responsibility for financial losses incurred through the use of this software.

## Development

### Adding a New Strategy

1. Create a new strategy file in the `strategies` directory
2. Extend the `BaseStrategy` class
3. Implement the `generate_signals` method
4. Update the `should_buy` and `should_sell` methods

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
