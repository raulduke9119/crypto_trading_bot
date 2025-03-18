#!/usr/bin/env python3
"""
Starting script for the Binance Trading Bot.
This script initializes and starts the trading bot based on command line arguments.
"""

import os
import argparse
import sys
import logging
from datetime import datetime, timedelta

from trading_bot import TradingBot
from config.config import (
    DATA_DIRECTORY, LOG_LEVEL, LOG_FILE, BINANCE_API_KEY, 
    BINANCE_API_SECRET, RISK_PERCENTAGE, MAX_POSITIONS, 
    TRAILING_STOP_PCT, MAX_DRAWDOWN, INITIAL_CAPITAL
)

def parse_arguments():
    """Parse command line arguments for bot configuration."""
    parser = argparse.ArgumentParser(description='Start Binance Trading Bot')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['live', 'paper', 'backtest'], default='paper',
                        help='Trading mode: live (real trading), paper (simulated), or backtest')
    
    # Symbol selection
    parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., BTCUSDT)')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of trading symbols')
    
    # Timeframe selection
    parser.add_argument('--timeframe', type=str, default='1h', 
                        help='Trading timeframe (e.g., 5m, 15m, 1h, 4h, 1d)')
    
    # Backtest parameters
    parser.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=INITIAL_CAPITAL,
                        help=f'Initial capital for backtest/paper trading (default: {INITIAL_CAPITAL})')
    
    # Risk parameters
    parser.add_argument('--risk', type=float, default=RISK_PERCENTAGE,
                        help=f'Risk percentage per trade (default: {RISK_PERCENTAGE}%)')
    parser.add_argument('--max-positions', type=int, default=MAX_POSITIONS,
                        help=f'Maximum number of open positions (default: {MAX_POSITIONS})')
    parser.add_argument('--trailing-stop', type=float, default=TRAILING_STOP_PCT,
                        help=f'Trailing stop percentage (default: {TRAILING_STOP_PCT}%)')
    parser.add_argument('--max-drawdown', type=float, default=MAX_DRAWDOWN,
                        help=f'Maximum allowed drawdown (default: {MAX_DRAWDOWN}%)')
    
    # Strategy selection
    parser.add_argument('--strategy', type=str, default='multi_indicator',
                        help='Trading strategy to use (default: multi_indicator)')
    parser.add_argument('--pattern', type=str, help='JSON pattern file for signal generation')
    
    # ML parameters
    parser.add_argument('--use-ml', action='store_true', help='Use machine learning predictions')
    
    # API parameters
    parser.add_argument('--api-key', type=str, help='Binance API key (overrides config)')
    parser.add_argument('--api-secret', type=str, help='Binance API secret (overrides config)')
    parser.add_argument('--testnet', action='store_true', help='Use Binance testnet API')
    
    # Logging parameters
    parser.add_argument('--log-level', type=str, default=LOG_LEVEL,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help=f'Logging level (default: {LOG_LEVEL})')
    parser.add_argument('--log-file', type=str, default=LOG_FILE,
                        help=f'Log file name (default: {LOG_FILE})')
    
    # New order book manager parameters
    parser.add_argument('--use-order-book', action='store_true', 
                        help='Use order book for liquidity analysis and trade execution')
    parser.add_argument('--order-book-depth', type=int, default=10,
                        help='Depth of order book to use (default: 10)')
    
    # Performance tracker parameters
    parser.add_argument('--kelly-factor', type=float, default=0.5,
                        help='Kelly criterion factor for position sizing (default: 0.5, half-Kelly)')
    parser.add_argument('--min-trades', type=int, default=10,
                        help='Minimum number of trades before using Kelly criterion (default: 10)')
    parser.add_argument('--history-file', type=str, 
                        help='Path to trade history file for performance tracking')
    
    return parser.parse_args()

def main():
    """Main function to initialize and start the trading bot."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up symbols list
    symbols = []
    if args.symbol:
        symbols.append(args.symbol)
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        # Default to BTC if no symbol specified
        symbols = ['BTCUSDT']
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(DATA_DIRECTORY, args.log_file)),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('trading_bot')
    logger.info(f"Starting trading bot in {args.mode} mode")
    
    # Create config dictionary
    config = {
        'symbols': symbols,
        'timeframe': args.timeframe,
        'api_key': args.api_key or BINANCE_API_KEY,
        'api_secret': args.api_secret or BINANCE_API_SECRET,
        'testnet': args.testnet,
        'risk_percentage': args.risk,
        'max_positions': args.max_positions,
        'trailing_stop_pct': args.trailing_stop,
        'max_drawdown': args.max_drawdown,
        'strategy': args.strategy,
        'log_level': args.log_level,
        'log_file': args.log_file,
        'use_ml_predictions': args.use_ml,
        'pattern_file': args.pattern,
        'use_order_book': args.use_order_book,
        'order_book_depth': args.order_book_depth,
        'kelly_factor': args.kelly_factor,
        'min_trades': args.min_trades,
        'history_file': args.history_file
    }
    
    # Initialize trading bot
    is_backtest = args.mode == 'backtest'
    is_paper_trading = args.mode == 'paper'
    
    bot = TradingBot(config, is_backtest=is_backtest, is_paper_trading=is_paper_trading, 
                     initial_balance=args.initial_capital)
    
    # Start trading based on mode
    if is_backtest:
        # Run backtest
        if not args.start_date:
            # Default to 1 month ago if not specified
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        else:
            start_date = args.start_date
            
        end_date = args.end_date  # Can be None
        
        logger.info(f"Running backtest from {start_date} to {end_date or 'today'}")
        
        # Run backtest
        results = bot.backtest(start_date, end_date, args.initial_capital)
        
        # Display results
        logger.info("Backtest Results:")
        overall = results.get('overall', {})
        logger.info(f"Initial Capital: ${overall.get('start_capital', 0):.2f}")
        logger.info(f"Final Capital: ${overall.get('end_capital', 0):.2f}")
        logger.info(f"Return: {overall.get('total_return', 0):.2f}%")
        logger.info(f"Number of Trades: {overall.get('total_trades', 0)}")
        logger.info(f"Win Rate: {overall.get('win_rate', 0):.2f}%")
        logger.info(f"Maximum Drawdown: {overall.get('max_drawdown', 0):.2f}%")
        logger.info(f"Sharpe Ratio: {overall.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Sortino Ratio: {overall.get('sortino_ratio', 0):.2f}")
        
        # Print symbol-specific results
        for symbol in symbols:
            if symbol in results:
                symbol_results = results[symbol]
                logger.info(f"\n{symbol} Results:")
                logger.info(f"Number of Trades: {symbol_results.get('total_trades', 0)}")
                logger.info(f"Win Rate: {symbol_results.get('win_rate', 0):.2f}%")
                logger.info(f"Profit Factor: {symbol_results.get('profit_factor', 0):.2f}")
                logger.info(f"Average Win: ${symbol_results.get('average_win', 0):.2f}")
                logger.info(f"Average Loss: ${symbol_results.get('average_loss', 0):.2f}")
        
        # Display performance summary from performance tracker if available
        if hasattr(bot, 'performance_tracker'):
            logger.info("\nPerformance Summary:")
            logger.info(bot.performance_tracker.get_performance_summary())
            
    elif is_paper_trading:
        # Run paper trading mode
        logger.info("Starting paper trading mode")
        bot.run()
    else:
        # Run live trading mode
        logger.info("Starting live trading mode")
        bot.run()

if __name__ == "__main__":
    main() 