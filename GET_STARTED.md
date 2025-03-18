# Getting Started with Binance Trading Bot

This guide provides detailed instructions on how to set up, configure, and use the Binance Trading Bot effectively. It includes explanations on creating custom trading patterns, optimizing backtesting strategies, and troubleshooting common issues.

## Table of Contents

1. [Setting Up the Bot](#setting-up-the-bot)
2. [Understanding the Pattern System](#understanding-the-pattern-system)
3. [Creating Custom Patterns](#creating-custom-patterns)
4. [Backtesting Strategies](#backtesting-strategies)
5. [Optimizing Trading Parameters](#optimizing-trading-parameters)
6. [Going Live](#going-live)
7. [Troubleshooting](#troubleshooting)

## Setting Up the Bot

### Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher installed
- A Binance account (or Testnet account for testing)
- API keys from Binance with appropriate permissions

### Installation Steps

1. Clone the repository and navigate to the project directory
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your API keys in a `.env` file:
   ```
   BINANCE_API_KEY=your_api_key
   BINANCE_API_SECRET=your_api_secret
   BINANCE_TESTNET_API_KEY=your_testnet_api_key
   BINANCE_TESTNET_API_SECRET=your_testnet_api_secret
   ```

### Initial Testing

Run the basic test to ensure everything is working:

```bash
python test_trading_workflow.py
```

This will verify your connection to Binance, test data collection, and validate the indicator calculations.

## Understanding the Pattern System

The Binance Trading Bot uses a flexible pattern-based system to define trading strategies. Each pattern is a JSON file that contains:

1. **Buy Conditions**: Rules that determine when to enter a trade
2. **Sell Conditions**: Rules that determine when to exit a trade
3. **Risk Parameters**: Settings for risk management like take profit, stop loss, etc.
4. **Signal Threshold**: Minimum signal strength required to execute trades

### Pattern Components

A typical pattern includes:

```json
{
    "name": "Pattern Name",
    "description": "Pattern description",
    "version": "1.0",
    "signal_threshold": 3.0,
    "risk_params": {
        "take_profit_pct": 1.2,
        "stop_loss_pct": 0.5,
        "max_trade_duration": 12,
        "trailing_stop_pct": 0.8,
        "max_drawdown_pct": 2.0,
        "position_sizing": "risk_adjusted"
    },
    "buy_conditions": [
        // Buy conditions go here
    ],
    "sell_conditions": [
        // Sell conditions go here
    ]
}
```

### Condition Structure

Each condition has the following structure:

```json
{
    "name": "Condition Name",
    "description": "Condition description",
    "indicator": "indicator_name",
    "operator": ">",
    "value": 50,
    "weight": 1.0
}
```

Or for more complex conditions:

```json
{
    "name": "Complex Condition",
    "description": "Condition with multiple parts",
    "conditions": [
        {
            "indicator": "rsi",
            "operator": ">",
            "indicator_compare": "rsi_prev",
            "weight": 0.8
        },
        {
            "indicator": "rsi",
            "operator": "<",
            "value": 50,
            "weight": 0.7
        }
    ],
    "weight": 1.2
}
```

### Available Indicators

The bot supports numerous technical indicators:

- **Price Data**: `open`, `high`, `low`, `close`, `volume`
- **Moving Averages**: `sma_[period]`, `ema_[period]`
- **Oscillators**: `rsi`, `stoch_k`, `stoch_d`
- **MACD**: `macd`, `macd_signal`, `macd_hist`
- **Bollinger Bands**: `bb_upper`, `bb_middle`, `bb_lower`, `bb_width`
- **Other**: `atr`, `adx`, `volatility`, and many more

### Comparative Operators

You can use these operators in conditions:
- `>`: Greater than
- `<`: Less than
- `>=`: Greater than or equal
- `<=`: Less than or equal
- `==`: Equal to
- `!=`: Not equal to

## Creating Custom Patterns

### Step 1: Choose Pattern Type

Decide what type of trading strategy you want to implement:
- **Scalping**: High-frequency trades with small profits (1m-5m timeframes)
- **Day Trading**: Medium-term trades (15m-1h timeframes)
- **Trend Following**: Longer-term trades following major trends (1h+ timeframes)
- **Hybrid**: Balanced approach combining multiple strategies

### Step 2: Define Risk Parameters

Set appropriate risk management parameters based on your strategy:

```json
"risk_params": {
    "take_profit_pct": 1.5,     // Target profit percentage
    "stop_loss_pct": 0.7,       // Maximum loss percentage
    "max_trade_duration": 24,   // Maximum trade duration in hours
    "trailing_stop_pct": 0.8,   // Trailing stop loss percentage
    "max_drawdown_pct": 2.0,    // Maximum portfolio drawdown allowed
    "position_sizing": "risk_adjusted"  // Position sizing method
}
```

### Step 3: Create Buy Conditions

Define conditions that must be met to generate buy signals:

```json
"buy_conditions": [
    {
        "name": "RSI Oversold",
        "description": "RSI below oversold threshold",
        "indicator": "rsi",
        "operator": "<",
        "value": 30,
        "weight": 1.5
    },
    {
        "name": "Price Above MA",
        "description": "Price above 20-period moving average",
        "indicator": "close",
        "operator": ">",
        "indicator_compare": "sma_20",
        "weight": 1.0
    }
]
```

### Step 4: Create Sell Conditions

Define conditions that must be met to generate sell signals:

```json
"sell_conditions": [
    {
        "name": "RSI Overbought",
        "description": "RSI above overbought threshold",
        "indicator": "rsi",
        "operator": ">",
        "value": 70,
        "weight": 1.5
    },
    {
        "name": "Take Profit",
        "description": "Take profit when target reached",
        "indicator": "unrealized_pnl_pct",
        "operator": ">=",
        "value": 1.2,
        "weight": 2.0
    }
]
```

### Step 5: Save and Test

Save your pattern as a JSON file in the `patterns` directory (e.g., `my_custom_pattern.json`) and test it with backtesting.

## Backtesting Strategies

### Basic Backtesting

To run a backtest with your custom pattern:

```bash
python start_trading_bot.py --mode backtest --symbol BTCUSDT --timeframe 5m --pattern my_custom_pattern.json --testnet
```

### Understanding Backtest Results

The backtest will output performance metrics:
- **Return**: Overall percentage return
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted return metric
- **Max Drawdown**: Largest percentage drop in portfolio value
- **Number of Trades**: Total trades executed

### Timeframe Selection

Different timeframes are suitable for different strategies:
- **1m, 3m, 5m**: Scalping strategies (high frequency)
- **15m, 30m**: Short-term trading
- **1h, 2h, 4h**: Medium-term trading
- **6h, 12h, 1d**: Long-term trading

Example:
```bash
python start_trading_bot.py --mode backtest --symbol BTCUSDT --timeframe 1m --pattern scalping_pattern.json --testnet
```

## Optimizing Trading Parameters

### Pattern Optimization

Based on backtesting results, you may need to adjust:

1. **Signal Threshold**: Increase for more selective trades, decrease for more frequent trades
2. **Risk Parameters**: Adjust take profit and stop loss values
3. **Condition Weights**: Change weights of different conditions
4. **Indicator Values**: Modify indicator thresholds

### Multi-Timeframe Approach

For optimal results, consider a multi-timeframe approach:
1. Use higher timeframes (1h) to determine overall trend direction
2. Use medium timeframes (15m) to confirm trend
3. Use lower timeframes (5m, 1m) for entry and exit points

### Recommended Patterns by Timeframe

Based on our testing, we recommend:

1. **For High-Frequency Trading (1m-5m timeframes)**:
   - Use the scalping_pattern.json with:
     - stop_loss_pct: 0.3%
     - take_profit_pct: 0.5%
     - trailing_stop_pct: 0.4%
     - signal_threshold: 2.8

2. **For Balanced Trading (15m timeframe)**:
   - Use the hybridized_pattern.json with:
     - stop_loss_pct: 0.5%
     - take_profit_pct: 1.2%
     - trailing_stop_pct: 0.8%
     - signal_threshold: 3.0

3. **For Longer-Term Trend Following (1h+ timeframes)**:
   - Use the trend_following_pattern.json with:
     - stop_loss_pct: 1.5%
     - take_profit_pct: 3.5%
     - trailing_stop_pct: 2.0%
     - signal_threshold: 4.2

## Going Live

### Testing on Testnet First

Always test your strategies on the Binance Testnet before trading with real funds:

```bash
python start_trading_bot.py --mode live --symbol BTCUSDT --timeframe 5m --pattern my_custom_pattern.json --testnet
```

### Live Trading

When ready to trade with real funds, remove the `--testnet` flag (use with caution):

```bash
python start_trading_bot.py --mode live --symbol BTCUSDT --timeframe 5m --pattern my_custom_pattern.json
```

### Monitoring Performance

Monitor your bot's performance with logging:

```bash
python start_trading_bot.py --mode live --symbol BTCUSDT --timeframe 5m --pattern my_custom_pattern.json --testnet --log-level DEBUG
```

## Troubleshooting

### Missing Indicators

If you see warnings about missing indicators, you may need to add them to the `TechnicalIndicators` class in `data/indicators.py`. Common missing indicators include:

```python
def add_ema(self, data, periods=[5, 12, 26]):
    for period in periods:
        data[f'ema_{period}'] = ta.ema(data['close'], length=period)
        data[f'ema_{period}_prev'] = data[f'ema_{period}'].shift(1)
    return data

def add_bollinger_width(self, data, period=20, std=2.0):
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    data['bb_width_prev'] = data['bb_width'].shift(1)
    return data

def add_adx(self, data, period=14):
    data['adx'] = ta.adx(data['high'], data['low'], data['close'], length=period)
    return data
```

### Invalid Pattern Format

If you see error messages about invalid pattern format, check:
- JSON syntax is valid
- Conditions have either `value` or `indicator_compare` (not both/neither)
- All required fields are present

### Performance Warnings

If you see warnings about DataFrame fragmentation, consider optimizing the indicator calculations:

```python
# Instead of:
for col in ['open', 'high', 'low', 'close']:
    data[f'{col}_sma10'] = data[col].rolling(10).mean()

# Use:
new_cols = {}
for col in ['open', 'high', 'low', 'close']:
    new_cols[f'{col}_sma10'] = data[col].rolling(10).mean()
data = pd.concat([data, pd.DataFrame(new_cols)], axis=1)
```

### API Errors

Common API errors and solutions:
- **Invalid interval**: Make sure you're using valid timeframes (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
- **API key permission issues**: Ensure your API keys have proper trading permissions
- **Rate limit errors**: Reduce the frequency of API calls or implement retry mechanisms

## Examples and Templates

### Example Pattern: RSI + MACD Strategy

```json
{
    "name": "RSI-MACD Strategy",
    "description": "Simple strategy using RSI and MACD indicators",
    "version": "1.0",
    "signal_threshold": 2.5,
    "risk_params": {
        "take_profit_pct": 1.0,
        "stop_loss_pct": 0.5,
        "max_trade_duration": 24,
        "trailing_stop_pct": 0.7,
        "max_drawdown_pct": 2.0,
        "position_sizing": "risk_adjusted"
    },
    "buy_conditions": [
        {
            "name": "RSI Oversold",
            "description": "RSI below oversold threshold",
            "indicator": "rsi",
            "operator": "<",
            "value": 35,
            "weight": 1.5
        },
        {
            "name": "MACD Bullish Cross",
            "description": "MACD crossing above signal line",
            "conditions": [
                {
                    "indicator": "macd",
                    "operator": ">",
                    "indicator_compare": "macd_signal",
                    "weight": 1.0
                },
                {
                    "indicator": "macd_prev",
                    "operator": "<",
                    "indicator_compare": "macd_signal_prev",
                    "weight": 0.5
                }
            ],
            "weight": 1.2
        }
    ],
    "sell_conditions": [
        {
            "name": "RSI Overbought",
            "description": "RSI above overbought threshold",
            "indicator": "rsi",
            "operator": ">",
            "value": 70,
            "weight": 1.5
        },
        {
            "name": "MACD Bearish Cross",
            "description": "MACD crossing below signal line",
            "conditions": [
                {
                    "indicator": "macd",
                    "operator": "<",
                    "indicator_compare": "macd_signal",
                    "weight": 1.0
                },
                {
                    "indicator": "macd_prev",
                    "operator": ">",
                    "indicator_compare": "macd_signal_prev",
                    "weight": 0.5
                }
            ],
            "weight": 1.2
        },
        {
            "name": "Take Profit",
            "description": "Take profit when target reached",
            "indicator": "unrealized_pnl_pct",
            "operator": ">=",
            "value": 1.0,
            "weight": 2.0
        }
    ]
}
```

### Template for Creating Your Own Pattern

```json
{
    "name": "Your Pattern Name",
    "description": "Describe your pattern's strategy",
    "version": "1.0",
    "signal_threshold": 3.0,
    "risk_params": {
        "take_profit_pct": 1.5,
        "stop_loss_pct": 0.7,
        "max_trade_duration": 24,
        "trailing_stop_pct": 0.8,
        "max_drawdown_pct": 2.0,
        "position_sizing": "risk_adjusted"
    },
    "buy_conditions": [
        {
            "name": "Condition 1",
            "description": "Description of condition 1",
            "indicator": "your_indicator",
            "operator": "your_operator",
            "value": 50,
            "weight": 1.0
        },
        {
            "name": "Condition 2",
            "description": "Description of condition 2",
            "conditions": [
                {
                    "indicator": "indicator1",
                    "operator": "operator1",
                    "indicator_compare": "indicator2",
                    "weight": 0.8
                },
                {
                    "indicator": "indicator3",
                    "operator": "operator2",
                    "value": 50,
                    "weight": 0.7
                }
            ],
            "weight": 1.2
        }
    ],
    "sell_conditions": [
        // Add your sell conditions following the same structure as buy conditions
    ]
}
```

Feel free to experiment with different combinations of indicators and parameters to find the strategy that works best for your trading goals. Remember to always test thoroughly with backtesting before deploying any strategy with real funds. 