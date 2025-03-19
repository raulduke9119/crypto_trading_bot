"""
Hauptmodul des Trading Bots.
Führt alle Komponenten zusammen und implementiert die Hauptlogik.
"""
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import schedule
from typing import Dict, List, Optional, Union, Tuple, Any
from binance.client import Client
from binance.exceptions import BinanceAPIException
import importlib
import traceback
import json
import logging

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from data.data_collector import DataCollector
from data.indicators import TechnicalIndicators
from strategies.multi_indicator_strategy import MultiIndicatorStrategy, PositionStatus
from strategies.dogebtc_hf_strategy import DogebtcHFStrategy
from utils.order_executor import OrderExecutor
from models.prediction_model import PredictionModel
from utils.position import Position
from config.config import (
    API_KEY, API_SECRET, USE_TESTNET, TRADING_SYMBOLS, DEFAULT_TIMEFRAME,
    RISK_PERCENTAGE, MAX_POSITIONS, INITIAL_CAPITAL, HISTORICAL_DATA_DAYS,
    DATA_DIRECTORY, MODEL_DIRECTORY, LOG_LEVEL, LOG_FILE, USE_ML,
    DEFAULT_PATTERN, TRAILING_STOP_PCT, MAX_DRAWDOWN
)
from utils.order_book_manager import OrderBookManager
from utils.performance_tracker import PerformanceTracker

# Logger einrichten
logger = setup_logger(LOG_FILE, LOG_LEVEL)

class TradingBot:
    """
    Hauptklasse des Trading Bots.
    Koordiniert alle Komponenten und führt die Handelslogik aus.
    """
    
    def __init__(self, 
                config: Dict[str, Any],
                is_backtest: bool = False,
                is_paper_trading: bool = False,
                initial_balance: float = INITIAL_CAPITAL):
        """
        Initialisiert den Trading Bot.
        
        Args:
            config: Dictionary mit Konfigurationseinstellungen
            is_backtest: Ob der Bot im Backtest-Modus läuft
            is_paper_trading: Ob der Bot im Paper-Trading-Modus läuft
            initial_balance: Anfangsguthaben für Backtest oder Paper-Trading
        """
        self.api_key = config.get('api_key', API_KEY)
        self.api_secret = config.get('api_secret', API_SECRET)
        self.use_testnet = config.get('testnet', USE_TESTNET)
        self.symbols = config.get('symbols', TRADING_SYMBOLS)
        self.timeframe = config.get('timeframe', DEFAULT_TIMEFRAME)
        self.risk_percentage = config.get('risk_percentage', RISK_PERCENTAGE)
        self.max_positions = config.get('max_positions', MAX_POSITIONS)
        self.initial_capital = initial_balance
        self.use_ml = config.get('use_ml_predictions', USE_ML)
        
        # Pattern configuration
        if 'pattern_file' in config and config['pattern_file']:
            self.pattern_name = config['pattern_file']
        else:
            self.pattern_name = DEFAULT_PATTERN
        
        # Trading mode
        self.is_backtest = is_backtest
        self.is_paper_trading = is_paper_trading
        
        # Log configuration
        self.log_level = config.get('log_level', LOG_LEVEL)
        self.log_file = config.get('log_file', LOG_FILE)
        
        # Risk parameters
        self.trailing_stop_pct = config.get('trailing_stop_pct', TRAILING_STOP_PCT)
        self.max_drawdown = config.get('max_drawdown', MAX_DRAWDOWN)
        
        # Order book and Kelly parameters
        self.use_order_book = config.get('use_order_book', False)
        self.order_book_depth = config.get('order_book_depth', 10)
        self.kelly_factor = config.get('kelly_factor', 0.5)
        self.min_trades_for_kelly = config.get('min_trades_for_kelly', 10)
        self.history_file = config.get('history_file', None)
        
        # Prüfe API-Schlüssel
        if not self.api_key or not self.api_secret:
            logger.error("API-Schlüssel oder Secret fehlen. Bitte in config.py oder .env-Datei konfigurieren.")
            raise ValueError("API-Schlüssel oder Secret fehlen")
        
        # Initialisiere Binance-Client
        try:
            self.client = Client(self.api_key, self.api_secret, testnet=self.use_testnet)
            logger.info(f"Binance-Client initialisiert (Testnet: {self.use_testnet})")
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren des Binance-Clients: {e}")
            raise
        
        # Initialisiere Komponenten
        self.data_collector = DataCollector(self.client)
        self.indicators = TechnicalIndicators()
        self.strategy = MultiIndicatorStrategy(
            self.risk_percentage, 
            self.max_positions, 
            self.use_ml,
            trailing_stop_pct=self.trailing_stop_pct,
            max_drawdown_pct=self.max_drawdown,
            pattern_name=self.pattern_name
        )
        self.order_executor = OrderExecutor(self.client)
        
        # Initialisiere Prediction Models für jeden Symbol
        self.prediction_models = {}
        if self.use_ml:
            for symbol in self.symbols:
                self.prediction_models[symbol] = PredictionModel(symbol)
            logger.info(f"Vorhersagemodelle für {len(self.symbols)} Symbole initialisiert")
        
        # Tracking von Positionen und Orders
        self.positions = {}  # Aktive Positionen
        self.open_orders = {}  # Offene Orders
        self.trade_history = {}  # Trade history for each symbol
        self.equity_curve = {}  # Equity curve for each symbol
        
        # Backtest-Daten
        self.backtest_data = {}  # Initialisiere backtest_data als leeres Dictionary
        
        # Performance-Metriken
        self.trades_executed = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_profit_loss = 0.0
        
        # Initialize new components
        if self.use_order_book:
            self.order_book_manager = OrderBookManager(
                self.client, 
                self.symbols,
                update_interval=5, 
                depth=self.order_book_depth
            )
            logger.info("OrderBookManager initialized for liquidity analysis")
        else:
            self.order_book_manager = None
            
        self.performance_tracker = PerformanceTracker(
            initial_capital=self.initial_capital, 
            history_file=self.history_file
        )
        
        logger.info(f"Trading Bot initialized with {len(self.symbols)} symbols in {'backtest' if is_backtest else 'paper trading' if is_paper_trading else 'live'} mode")
    
    def test_connection(self) -> bool:
        """
        Testet die Verbindung zu Binance und zeigt Kontoinformationen an.
        
        Returns:
            True bei erfolgreicher Verbindung, False bei Fehler
        """
        try:
            # Teste Serverzeit
            server_time = self.client.get_server_time()
            server_time_str = datetime.fromtimestamp(server_time['serverTime'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Binance-Serverzeit: {server_time_str}")
            
            # Hole Kontoinformationen
            account_info = self.client.get_account()
            
            # Zeige Guthaben an
            balances = account_info['balances']
            non_zero_balances = [b for b in balances if float(b['free']) > 0 or float(b['locked']) > 0]
            
            logger.info("Kontoguthaben:")
            for balance in non_zero_balances:
                logger.info(f"{balance['asset']}: Frei={balance['free']}, Gesperrt={balance['locked']}")
            
            return True
        except BinanceAPIException as e:
            logger.error(f"Binance API-Fehler: {e}")
            return False
        except Exception as e:
            logger.error(f"Unerwarteter Fehler: {e}")
            return False
    
    def update_market_data(self, days_back: int = HISTORICAL_DATA_DAYS) -> Dict[str, pd.DataFrame]:
        """
        Aktualisiert Marktdaten für alle Symbole.
        
        Args:
            days_back: Wie viele Tage zurück Daten geladen werden sollen
            
        Returns:
            Dictionary mit Symbol als Schlüssel und DataFrame als Wert
        """
        try:
            logger.info(f"Aktualisiere Marktdaten für {len(self.symbols)} Symbole...")
            
            market_data = {}
            for symbol in self.symbols:
                # Hole historische Kerzendaten
                df = self.data_collector.get_historical_klines(
                    symbol=symbol,
                    interval=self.timeframe,
                    start_str=f"{days_back} days ago"
                )
                
                if df.empty:
                    logger.warning(f"Keine Daten gefunden für {symbol}")
                    continue
                
                # Füge technische Indikatoren hinzu
                df = self.indicators.add_all_indicators(df)
                
                # Berechne Signale
                df = self.strategy.generate_signals(df)
                
                # Speichere die Daten
                market_data[symbol] = df
                
                logger.info(f"Marktdaten für {symbol} aktualisiert: {len(df)} Zeilen")
                
                # Warte kurz, um Rate-Limits zu vermeiden
                time.sleep(0.5)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren der Marktdaten: {e}")
            return {}
    
    def train_prediction_models(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """
        Trainiert Vorhersagemodelle für jedes Symbol.
        
        Args:
            market_data: Dictionary mit Marktdaten
        """
        if not self.use_ml:
            return
        
        try:
            logger.info("Trainiere Vorhersagemodelle...")
            for symbol, df in market_data.items():
                if symbol in self.prediction_models:
                    self.prediction_models[symbol].train(df)
            logger.info("Training der Vorhersagemodelle abgeschlossen")
        except Exception as e:
            logger.error(f"Fehler beim Training der Vorhersagemodelle: {e}")
            raise
    
    def backtest(self, start_date=None, end_date=None, initial_balance=1000.0):
        """
        Run a backtest simulation using historical data.
        
        Args:
            start_date: Optional start date for backtest (datetime or string)
            end_date: Optional end date for backtest (datetime or string)
            initial_balance: Initial account balance for backtest
            
        Returns:
            Dictionary with backtest results
        """
        start_time = time.time()
        
        # Validate inputs
        if start_date and isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        if end_date and isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        if not end_date:
            end_date = datetime.now()
            
        # Reset simulation state
        self.initial_capital = initial_balance
        self.backtest_stats = {}
        self.trade_history = {}
        self.equity_curve = {}
        self.positions = {}

        # Reset performance tracker with initial balance
        self.performance_tracker.reset()
        self.performance_tracker.initial_capital = initial_balance
        self.performance_tracker.current_capital = initial_balance
        
        logger.info(f"Starting backtest from {start_date} to {end_date} with ${initial_balance} initial capital")
        
        # Get historical data for each symbol
        all_market_data = {}
        for symbol in self.symbols:
            try:
                # Get historical data
                logger.info(f"Fetching historical data for {symbol}...")
                market_data = self.data_collector.get_historical_data(symbol, self.timeframe, start_date, end_date)
                
                if market_data.empty:
                    logger.warning(f"No historical data found for {symbol}")
                    continue
                
                # Reset index to ensure we have a proper datetime column
                market_data = market_data.reset_index()
                if 'datetime' not in market_data.columns and 'date' in market_data.columns:
                    market_data = market_data.rename(columns={'date': 'datetime'})
                
                # Ensure datetime column exists and is sorted
                if 'datetime' not in market_data.columns:
                    logger.error(f"No datetime column found in market data for {symbol}")
                    continue
                
                # Sort by date and remove duplicates
                market_data = market_data.sort_values('datetime')
                market_data = market_data.drop_duplicates(subset=['datetime'])
                
                # Add technical indicators
                market_data = self._add_indicators(market_data)
                
                # Generate signals using strategy
                market_data = self.strategy.generate_signals(market_data)
                
                logger.info(f"Loaded {len(market_data)} candles for {symbol}")
                logger.info(f"Buy signals: {market_data['buy_signal'].sum()}, Sell signals: {market_data['sell_signal'].sum()}")
                
                all_market_data[symbol] = market_data
                
            except Exception as e:
                logger.error(f"Error getting historical data for {symbol}: {e}")
                continue
        
        if not all_market_data:
            logger.error("No valid market data for any symbol, aborting backtest")
            return {'error': 'No valid market data'}
        
        # Combined dataframe index for iteration
        dates = sorted(set(pd.to_datetime(date) for symbol in all_market_data 
                          for date in all_market_data[symbol]['datetime'].values))
        
        # Initialize equity curve with starting balance
        for symbol in self.symbols:
            self.equity_curve[symbol] = [(dates[0] if dates else datetime.now(), initial_balance)]
        
        # Simulate trading
        logger.info("Simulating trades...")
        
        # Loop through each date
        for current_date in dates:
            # Process each symbol
            for symbol in all_market_data:
                # Get current day's data
                current_data = all_market_data[symbol][all_market_data[symbol]['datetime'] == current_date]
                
                if current_data.empty:
                    # No data for this symbol on this date
                    continue
                
                current_row = current_data.iloc[0]
                current_price = current_row['close']
                
                # Track unrealized P&L
                if symbol in self.positions:
                    position = self.positions[symbol]
                    entry_price = position['entry_price']
                    quantity = position['quantity']
                    direction = position['direction']
                    
                    # Calculate unrealized P&L
                    if direction == 'long':
                        unrealized_pnl = (current_price - entry_price) * quantity
                        unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100
                    else:  # short
                        unrealized_pnl = (entry_price - current_price) * quantity
                        unrealized_pnl_pct = (entry_price - current_price) / entry_price * 100
                    
                    # Update position info
                    position['unrealized_pnl'] = unrealized_pnl
                    position['unrealized_pnl_pct'] = unrealized_pnl_pct
                    position['current_price'] = current_price
                    
                    # Check trailing stop
                    if 'highest_price' in position and direction == 'long':
                        if current_price > position['highest_price']:
                            position['highest_price'] = current_price
                            
                            # Update trailing stop if enabled
                            if self.trailing_stop_pct > 0:
                                new_stop = current_price * (1 - self.trailing_stop_pct / 100)
                                if 'stop_loss' not in position or new_stop > position['stop_loss']:
                                    position['stop_loss'] = new_stop
                                    position['stop_type'] = 'trailing'
                        
                        # Check if trailing stop triggered
                        if 'stop_loss' in position and current_price <= position['stop_loss'] and position['stop_type'] == 'trailing':
                            # Close position at trailing stop
                            trade = self._handle_closed_trade(symbol, position, current_price, 
                                                            'trailing_stop', current_date)
                            del self.positions[symbol]
                            
                    elif 'lowest_price' in position and direction == 'short':
                        if current_price < position['lowest_price']:
                            position['lowest_price'] = current_price
                            
                            # Update trailing stop if enabled
                            if self.trailing_stop_pct > 0:
                                new_stop = current_price * (1 + self.trailing_stop_pct / 100)
                                if 'stop_loss' not in position or new_stop < position['stop_loss']:
                                    position['stop_loss'] = new_stop
                                    position['stop_type'] = 'trailing'
                        
                        # Check if trailing stop triggered
                        if 'stop_loss' in position and current_price >= position['stop_loss'] and position['stop_type'] == 'trailing':
                            # Close position at trailing stop
                            trade = self._handle_closed_trade(symbol, position, current_price, 
                                                            'trailing_stop', current_date)
                            del self.positions[symbol]
                    
                    # Check fixed stop loss and take profit
                    if direction == 'long':
                        # Check stop loss
                        if 'stop_loss' in position and current_price <= position['stop_loss'] and position['stop_type'] == 'fixed':
                            # Close position at stop loss
                            trade = self._handle_closed_trade(symbol, position, current_price, 
                                                            'stop_loss', current_date)
                            del self.positions[symbol]
                            
                        # Check take profit
                        elif 'take_profit' in position and current_price >= position['take_profit']:
                            # Close position at take profit
                            trade = self._handle_closed_trade(symbol, position, current_price, 
                                                            'take_profit', current_date)
                            del self.positions[symbol]
                            
                    else:  # short
                        # Check stop loss
                        if 'stop_loss' in position and current_price >= position['stop_loss'] and position['stop_type'] == 'fixed':
                            # Close position at stop loss
                            trade = self._handle_closed_trade(symbol, position, current_price, 
                                                            'stop_loss', current_date)
                            del self.positions[symbol]
                            
                        # Check take profit
                        elif 'take_profit' in position and current_price <= position['take_profit']:
                            # Close position at take profit
                            trade = self._handle_closed_trade(symbol, position, current_price, 
                                                            'take_profit', current_date)
                            del self.positions[symbol]
                
                # Process signals
                # Buy signal for long positions
                if current_row['buy_signal'] == 1 and symbol not in self.positions and len(self.positions) < self.max_positions:
                    # Calculate signal strength (0.5 to 1.0 based on indicators)
                    signal_strength = self._calculate_signal_strength(current_data)
                    
                    # Calculate position size
                    quantity = self._calculate_position_size(symbol, all_market_data[symbol][:current_data.index[0]+1], signal_strength)
                    
                    if quantity > 0:
                        # Calculate stop loss based on ATR
                        atr_col = 'atr_14'
                        atr = current_row[atr_col] if atr_col in current_row and not np.isnan(current_row[atr_col]) else current_price * 0.01
                        
                        # Check if the OrderBookManager can provide optimal stop loss levels
                        stop_loss_level = current_price * (1 - self.risk_percentage / 100)  # Default based on risk percentage
                        take_profit_level = current_price * (1 + self.risk_percentage * 2 / 100)  # Default 2:1 risk/reward
                        
                        # In a real system, we would use the OrderBookManager to determine optimal levels
                        
                        # Open long position
                        position = {
                            'direction': 'long',
                            'entry_price': current_price,
                            'quantity': quantity,
                            'entry_time': current_date,
                            'stop_loss': stop_loss_level,
                            'stop_type': 'fixed',
                            'take_profit': take_profit_level,
                            'highest_price': current_price,
                            'unrealized_pnl': 0,
                            'unrealized_pnl_pct': 0,
                            'signal_strength': signal_strength
                        }
                        
                        self.positions[symbol] = position
                        logger.info(f"BACKTEST: {current_date} - Opening LONG position for {symbol} at {current_price}, "
                                 f"qty: {quantity}, stop: {stop_loss_level:.2f}, target: {take_profit_level:.2f}")
                
                # Sell signal for short positions
                elif current_row['sell_signal'] == 1 and symbol not in self.positions and len(self.positions) < self.max_positions:
                    # Calculate signal strength (0.5 to 1.0 based on indicators)
                    signal_strength = self._calculate_signal_strength(current_data)
                    
                    # Calculate position size
                    quantity = self._calculate_position_size(symbol, all_market_data[symbol][:current_data.index[0]+1], signal_strength)
                    
                    if quantity > 0:
                        # Calculate stop loss based on ATR
                        atr_col = 'atr_14'
                        atr = current_row[atr_col] if atr_col in current_row and not np.isnan(current_row[atr_col]) else current_price * 0.01
                        
                        # Default stop loss and take profit levels
                        stop_loss_level = current_price * (1 + self.risk_percentage / 100)  # Default based on risk percentage
                        take_profit_level = current_price * (1 - self.risk_percentage * 2 / 100)  # Default 2:1 risk/reward
                        
                        # Open short position
                        position = {
                            'direction': 'short',
                            'entry_price': current_price,
                            'quantity': quantity,
                            'entry_time': current_date,
                            'stop_loss': stop_loss_level,
                            'stop_type': 'fixed',
                            'take_profit': take_profit_level,
                            'lowest_price': current_price,
                            'unrealized_pnl': 0,
                            'unrealized_pnl_pct': 0,
                            'signal_strength': signal_strength
                        }
                        
                        self.positions[symbol] = position
                        logger.info(f"BACKTEST: {current_date} - Opening SHORT position for {symbol} at {current_price}, "
                                 f"qty: {quantity}, stop: {stop_loss_level:.2f}, target: {take_profit_level:.2f}")
                
                # Exit signal for existing positions
                elif (current_row['sell_signal'] == 1 and symbol in self.positions and self.positions[symbol]['direction'] == 'long') or \
                     (current_row['buy_signal'] == 1 and symbol in self.positions and self.positions[symbol]['direction'] == 'short'):
                    # Close position on exit signal
                    position = self.positions[symbol]
                    trade = self._handle_closed_trade(symbol, position, current_price, 'exit_signal', current_date)
                    del self.positions[symbol]
            
            # Update equity curve at end of day
            for symbol in self.symbols:
                if symbol in self.equity_curve:
                    self.equity_curve[symbol].append((current_date, self.initial_capital))
        
        # Close any remaining positions at the end of the simulation
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            last_price = all_market_data[symbol]['close'].iloc[-1]
            trade = self._handle_closed_trade(symbol, position, last_price, 'end_of_simulation', end_date)
            del self.positions[symbol]
        
        # Calculate backtest metrics
        results = {}
        for symbol in self.symbols:
            if symbol in self.trade_history and self.trade_history[symbol]:
                metrics = self._calculate_backtest_metrics(self.trade_history[symbol], self.initial_capital, self.initial_capital)
                results[symbol] = metrics
        
        # Overall results
        total_trades = sum(len(self.trade_history.get(symbol, [])) for symbol in self.symbols)
        winning_trades = sum(sum(1 for trade in self.trade_history.get(symbol, []) if trade['pnl'] > 0) for symbol in self.symbols)
        total_pnl = sum(sum(trade['pnl'] for trade in self.trade_history.get(symbol, [])) for symbol in self.symbols)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        return_pct = (self.initial_capital - initial_balance) / initial_balance * 100 if initial_balance > 0 else 0
        
        # Get metrics from the performance tracker
        overall_metrics = self.performance_tracker.get_metrics()
        
        results['overall'] = {
            'start_capital': initial_balance,
            'end_capital': self.initial_capital,
            'total_return': return_pct,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate * 100,
            'profit_factor': overall_metrics.get('profit_factor', 0),
            'sharpe_ratio': overall_metrics.get('sharpe_ratio', 0),
            'sortino_ratio': overall_metrics.get('sortino_ratio', 0),
            'max_drawdown': overall_metrics.get('max_drawdown', 0),
            'backtest_duration': time.time() - start_time
        }
        
        logger.info(f"Backtest completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Start capital: ${initial_balance:.2f}, End capital: ${self.initial_capital:.2f}")
        logger.info(f"Return: {return_pct:.2f}%, Number of trades: {total_trades}, Win rate: {win_rate*100:.2f}%")
        logger.info(f"Maximum drawdown: {overall_metrics.get('max_drawdown', 0):.2f}%, Sharpe Ratio: {overall_metrics.get('sharpe_ratio', 0):.2f}")
        
        return results

    def _calculate_position_size(self, symbol: str, market_data: pd.DataFrame, signal_strength: float = 1.0) -> float:
        """
        Calculate position size based on account balance, current price, and signal strength.
        Handles both live trading and backtesting cases.
        
        Args:
            symbol: Trading pair symbol
            market_data: Historical market data
            signal_strength: Signal strength from 0.5 to 1.0
            
        Returns:
            Quantity to trade
        """
        try:
            # Get current account balance (using initial_capital for backtest/paper trading)
            if self.is_backtest or self.is_paper_trading:
                account_balance = self.initial_capital
            else:
                account_balance = self.order_executor.get_account_balance('USDT')
            
            if account_balance <= 0:
                logger.warning(f"Invalid account balance: {account_balance}")
                return 0
            
            # Get current price from market data
            if market_data is None or market_data.empty:
                logger.warning(f"No market data available for {symbol}")
                return 0
            
            current_price = market_data['close'].iloc[-1]
            if current_price <= 0:
                logger.warning(f"Invalid current price for {symbol}: {current_price}")
                return 0
            
            # Check liquidity if not using testnet and not in backtest mode
            if not self.use_testnet and not self.is_backtest:
                if 'volume' in market_data.columns:
                    # Volume in base asset, convert to USD value
                    avg_volume = market_data['volume'].rolling(window=24).mean().iloc[-1] * current_price
                    
                    if avg_volume < 100000:  # Less than $100k average volume
                        logger.warning(f"Low liquidity for {symbol}: ${avg_volume:.2f} avg volume")
            
            # Use Kelly criterion for position sizing if we have enough trades
            min_position_size = 0.001  # Minimum position size (0.1% of account)
            min_trade_value = 10.0     # Minimum trade value in USDT
            
            # Calculate risk percentage based on signal strength
            risk_pct = self.risk_percentage * signal_strength
            
            # If we have enough trade history, use Kelly position sizing
            if hasattr(self, 'performance_tracker') and \
               symbol in self.performance_tracker.symbols_performance and \
               self.performance_tracker.symbols_performance[symbol]['trades'] >= self.min_trades_for_kelly:
                try:
                    kelly_pct = self.performance_tracker.calculate_optimal_position_size(
                        symbol, account_balance, self.kelly_factor, self.min_trades_for_kelly
                    )
                    
                    # Apply a floor to Kelly percentage to prevent tiny positions
                    # Use at least 30% of the basic risk percentage if Kelly is very small
                    kelly_pct = max(kelly_pct, risk_pct * 0.3)
                    
                    logger.info(f"Using Kelly position sizing for {symbol}: {kelly_pct:.2f}%")
                    position_size = (account_balance * kelly_pct / 100) / current_price
                except Exception as e:
                    logger.warning(f"Error calculating Kelly position size: {e}. Using standard sizing.")
                    position_size = self._calculate_position_size_simple(
                        account_balance, current_price, risk_pct, signal_strength
                    )
            else:
                # Standard position sizing based on fixed risk percentage
                logger.info(f"Using standard position sizing for {symbol}: {risk_pct:.2f}%")
                position_size = self._calculate_position_size_simple(
                    account_balance, current_price, risk_pct, signal_strength
                )
            
            # Apply minimum position constraints for meaningful backtesting
            min_qty_by_value = min_trade_value / current_price
            min_qty_by_pct = (account_balance * min_position_size / 100) / current_price
            min_position_qty = max(min_qty_by_value, min_qty_by_pct)
            
            if self.is_backtest:
                # During backtesting, ensure we have a meaningful position size
                position_size = max(position_size, min_position_qty)
            
            # Round position size according to lot size rules
            if not self.is_backtest:
                try:
                    # Get symbol info for proper rounding
                    symbol_info = self.order_executor.get_symbol_filters(symbol)
                    lot_size_filter = symbol_info.get('LOT_SIZE', {})
                    step_size = lot_size_filter.get('stepSize', 0.00001)
                    position_size = self.order_executor.round_step_size(position_size, float(step_size))
                except Exception as e:
                    logger.error(f"Error getting symbol info for {symbol}: {e}")
                    # Use a reasonable default if error occurs
                    position_size = round(position_size, 5)
            else:
                # For backtesting, use a simpler rounding approach
                position_size = round(position_size, 5)
                
            # Calculate position value
            position_value = position_size * current_price
            logger.info(f"Calculated position size for {symbol}: {position_size} (~${position_value:.2f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    def _handle_closed_trade(self, symbol, position, exit_price, exit_reason, exit_time=None):
        """
        Process a closed trade, update statistics and trade history.
        
        Args:
            symbol: Trading symbol
            position: The position that was closed
            exit_price: The exit price
            exit_reason: Reason for exit (take_profit, stop_loss, etc.)
            exit_time: The exit time (optional, defaults to current time)
        """
        if not exit_time:
            exit_time = datetime.now()
            
        entry_time = position.get('entry_time', datetime.now())
        entry_price = position.get('entry_price', 0)
        quantity = position.get('quantity', 0)
        direction = position.get('direction', 'long')
        
        # Calculate P&L
        if direction == 'long':
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:  # short
            pnl = (entry_price - exit_price) * quantity
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            
        # Update account balance for backtesting/paper trading
        if self.use_testnet:
            self.initial_capital += pnl
            
        # Create trade record
        trade = {
            'symbol': symbol,
            'direction': direction,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason
        }
        
        # Add trade to history
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []
        self.trade_history[symbol].append(trade)
        
        # Log the result
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        pnl_pct_str = f"+{pnl_pct:.2f}%" if pnl_pct >= 0 else f"-{abs(pnl_pct):.2f}%"
        logger.info(f"Closed {direction} position for {symbol} - {exit_reason}. PnL: {pnl_str} ({pnl_pct_str})")
        
        # Update equity curve
        timestamp = exit_time
        if self.use_testnet:
            timestamp = exit_time  # Use the provided timestamp for backtests
        
        if symbol not in self.equity_curve:
            self.equity_curve[symbol] = []
        
        self.equity_curve[symbol].append((timestamp, self.initial_capital if self.use_testnet else 0))
        
        # Add to performance tracker
        self.performance_tracker.add_trade(trade)
        
        return trade

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        if df.empty:
            logger.warning("Cannot add indicators to empty DataFrame")
            return df
            
        try:
            # Make a copy to avoid warnings
            result_df = df.copy()
            
            # Add technical indicators using the indicators object
            result_df = self.indicators.add_all_indicators(result_df)
            
            # Fill NaN values that might be created during indicator calculations
            result_df = result_df.ffill().bfill()
            
            return result_df
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return df

    def _calculate_signal_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate the signal strength based on various indicators.
        
        Args:
            data: DataFrame with market data and indicators
            
        Returns:
            Signal strength as a float between 0.5 and 1.0
        """
        try:
            strength = 0.5  # Base strength
            
            # Use the last row if data is a DataFrame
            if isinstance(data, pd.DataFrame) and not data.empty:
                row = data.iloc[-1]
            else:
                # Handle case where data is a Series
                row = data if isinstance(data, pd.Series) else None
                
            if row is None:
                return strength
                
            # Add strength based on RSI
            if 'rsi' in row:
                rsi = row['rsi']
                # RSI below 30 (oversold) for buy signals
                if rsi < 30:
                    strength += 0.1
                # RSI below 20 (extremely oversold) for buy signals
                if rsi < 20:
                    strength += 0.1
                    
            # Add strength based on MACD
            if 'macd_hist' in row:
                macd_hist = row['macd_hist']
                # Strong MACD histogram for buy signals
                if macd_hist > 0:
                    strength += 0.05
                if macd_hist > 10:
                    strength += 0.05
                    
            # Add strength based on Bollinger Bands
            if 'bb_lower' in row and 'close' in row:
                bb_lower = row['bb_lower']
                close = row['close']
                # Price near or below lower Bollinger Band for buy signals
                if close < bb_lower * 1.01:
                    strength += 0.1
                    
            # Cap strength between 0.5 and 1.0
            return min(max(strength, 0.5), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.5  # Default to medium strength

    def _calculate_position_size_simple(self, balance: float, price: float, risk_pct: float, signal_strength: float = 0.5) -> float:
        """
        Simple position size calculation for backtesting
        
        Args:
            balance: Current account balance
            price: Current price of the asset
            risk_pct: Risk percentage
            signal_strength: Signal strength (0.0-1.0)
            
        Returns:
            Position size in units of the asset
        """
        if balance <= 0 or price <= 0:
            return 0.0
            
        # Adjust risk based on signal strength
        adjusted_risk = risk_pct * (0.8 + signal_strength * 0.4)
        
        # Calculate position value based on risk
        position_value = (adjusted_risk / 100) * balance
        
        # Ensure we don't risk more than a certain percentage per trade
        max_position_value = balance / self.max_positions
        position_value = min(position_value, max_position_value)
        
        # Calculate quantity
        quantity = position_value / price
        
        # Apply signal strength adjustment to final size
        quantity *= (0.7 + signal_strength * 0.5)
        
        # Round to 8 decimal places (common for crypto)
        return round(quantity, 8)

    def _get_empty_backtest_results(self) -> Dict[str, Any]:
        """Gibt leere Backtest-Ergebnisse zurück."""
        return {
            'initial_balance': 0.0,
            'final_balance': 0.0,
            'return_percentage': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'equity_curve': [],
            'trades': []
        }

    def _calculate_backtest_metrics(self, trades: List[Dict[str, Any]], initial_balance: float, final_balance: float) -> Dict[str, Any]:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            trades: List of completed trades from backtest
            initial_balance: Starting account balance
            final_balance: Ending account balance
            
        Returns:
            Dictionary of calculated performance metrics
        """
        try:
            results = {}
            
            # Overall metrics
            total_trades = len(trades)
            results['total_trades'] = total_trades
            
            if total_trades == 0:
                logger.warning("No trades were executed in backtest, returning empty metrics")
                return self._get_empty_backtest_results()
            
            # Calculate P&L metrics
            results['start_capital'] = initial_balance
            results['end_capital'] = final_balance
            results['absolute_return'] = final_balance - initial_balance
            results['total_return'] = ((final_balance - initial_balance) / initial_balance) * 100 if initial_balance > 0 else 0
            
            # Standardize profit field (could be 'profit' or 'pnl' depending on the trade object)
            for trade in trades:
                if 'profit' not in trade and 'pnl' in trade:
                    trade['profit'] = trade['pnl']
                elif 'profit' not in trade:
                    trade['profit'] = 0.0
            
            # Win/Loss metrics
            winning_trades = [t for t in trades if t.get('profit', 0) > 0]
            losing_trades = [t for t in trades if t.get('profit', 0) <= 0]
            
            results['winning_trades'] = len(winning_trades)
            results['losing_trades'] = len(losing_trades)
            results['win_rate'] = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
            
            # Calculate average win/loss
            total_profit = sum(t.get('profit', 0) for t in winning_trades)
            total_loss = sum(t.get('profit', 0) for t in losing_trades)
            
            results['total_profit'] = total_profit
            results['total_loss'] = total_loss
            results['net_profit'] = total_profit + total_loss
            
            results['average_win'] = total_profit / len(winning_trades) if winning_trades else 0
            results['average_loss'] = total_loss / len(losing_trades) if losing_trades else 0
            
            # Calculate profit factor
            results['profit_factor'] = abs(total_profit / total_loss) if total_loss != 0 else (1 if total_profit == 0 else float('inf'))
            
            # Calculate drawdown
            if trades:
                # Create equity curve
                equity_curve = [initial_balance]
                for trade in trades:
                    equity_curve.append(equity_curve[-1] + trade.get('profit', 0))
                
                # Calculate max drawdown
                max_equity = initial_balance
                max_drawdown = 0
                max_drawdown_pct = 0
                
                for equity in equity_curve:
                    max_equity = max(max_equity, equity)
                    drawdown = max_equity - equity
                    drawdown_pct = (drawdown / max_equity) * 100 if max_equity > 0 else 0
                    
                    max_drawdown = max(max_drawdown, drawdown)
                    max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
                
                results['max_drawdown'] = max_drawdown
                results['max_drawdown_pct'] = max_drawdown_pct
            else:
                results['max_drawdown'] = 0
                results['max_drawdown_pct'] = 0
            
            # Calculate risk metrics if there are enough trades
            if total_trades >= 5:
                # Convert trades to returns for risk calculations
                returns = []
                for trade in trades:
                    profit_pct = trade.get('profit_pct', 0)
                    returns.append(profit_pct / 100)  # Convert percentage to decimal
                
                returns = np.array(returns)
                
                # Calculate Sharpe ratio (assuming risk-free rate of 0)
                avg_return = np.mean(returns)
                std_dev = np.std(returns)
                results['sharpe_ratio'] = (avg_return / std_dev) * np.sqrt(252) if std_dev > 0 else 0
                
                # Calculate Sortino ratio (using negative returns only)
                negative_returns = returns[returns < 0]
                downside_dev = np.std(negative_returns) if len(negative_returns) > 0 else 0.0001
                results['sortino_ratio'] = (avg_return / downside_dev) * np.sqrt(252) if downside_dev > 0 else 0
            else:
                results['sharpe_ratio'] = 0
                results['sortino_ratio'] = 0
            
            # Calculate average trade metrics
            results['average_trade'] = results['net_profit'] / total_trades if total_trades > 0 else 0
            results['average_trade_pct'] = results['total_return'] / total_trades if total_trades > 0 else 0
            
            # Calculate expectancy
            results['expectancy'] = (results['win_rate'] / 100 * results['average_win']) + \
                                   ((1 - results['win_rate'] / 100) * results['average_loss'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating backtest metrics: {e}")
            return self._get_empty_backtest_results()

    def run(self):
        """
        Start the trading bot in live or paper trading mode.
        """
        logger.info(f"Starting trading bot in {'paper trading' if self.is_paper_trading else 'live'} mode")
        
        # Initialize market data
        market_data = self.update_market_data()
        
        # Start order book manager if configured
        if self.use_order_book and self.order_book_manager:
            self.order_book_manager.start()
            logger.info("Order book manager started")
        
        # Trading loop will be implemented here
        logger.info("Trading bot ready")
        
    def stop(self):
        """
        Stop the trading bot and clean up resources.
        """
        logger.info("Stopping trading bot")
        
        # Stop order book manager if running
        if self.use_order_book and self.order_book_manager:
            self.order_book_manager.stop()
            logger.info("Order book manager stopped")
        
        logger.info("Trading bot stopped")
