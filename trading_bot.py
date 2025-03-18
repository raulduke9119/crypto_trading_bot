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
    DATA_DIRECTORY, MODEL_DIRECTORY, LOG_LEVEL, LOG_FILE, USE_ML
)

# Logger einrichten
logger = setup_logger(LOG_FILE, LOG_LEVEL)

class TradingBot:
    """
    Hauptklasse des Trading Bots.
    Koordiniert alle Komponenten und führt die Handelslogik aus.
    """
    
    def __init__(self, 
                api_key: str = API_KEY, 
                api_secret: str = API_SECRET,
                use_testnet: bool = USE_TESTNET,
                symbols: List[str] = TRADING_SYMBOLS,
                timeframe: str = DEFAULT_TIMEFRAME,
                risk_percentage: float = RISK_PERCENTAGE,
                max_positions: int = MAX_POSITIONS,
                initial_capital: float = INITIAL_CAPITAL,
                use_ml: bool = USE_ML,
                pattern_name: str = "default_pattern.json"):
        """
        Initialisiert den Trading Bot.
        
        Args:
            api_key: Binance API-Schlüssel
            api_secret: Binance API-Secret
            use_testnet: Ob Testnet verwendet werden soll
            symbols: Liste von Trading-Paaren
            timeframe: Zeitintervall
            risk_percentage: Risikoprozentsatz pro Trade
            max_positions: Maximale Anzahl gleichzeitiger Positionen
            initial_capital: Anfangskapital
            use_ml: Ob ML-Vorhersagen verwendet werden sollen
            pattern_name: Name des zu verwendenden Trading-Patterns
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet
        self.symbols = symbols
        self.timeframe = timeframe
        self.risk_percentage = risk_percentage
        self.max_positions = max_positions
        self.initial_capital = initial_capital
        self.use_ml = use_ml
        self.pattern_name = pattern_name
        
        # Prüfe API-Schlüssel
        if not api_key or not api_secret:
            logger.error("API-Schlüssel oder Secret fehlen. Bitte in config.py oder .env-Datei konfigurieren.")
            raise ValueError("API-Schlüssel oder Secret fehlen")
        
        # Initialisiere Binance-Client
        try:
            self.client = Client(api_key, api_secret, testnet=use_testnet)
            logger.info(f"Binance-Client initialisiert (Testnet: {use_testnet})")
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren des Binance-Clients: {e}")
            raise
        
        # Initialisiere Komponenten
        self.data_collector = DataCollector(self.client)
        self.indicators = TechnicalIndicators()
        self.strategy = MultiIndicatorStrategy(
            risk_percentage, 
            max_positions, 
            use_ml,
            trailing_stop_pct=1.5,
            max_drawdown_pct=5.0,
            pattern_name=pattern_name
        )
        self.order_executor = OrderExecutor(self.client)
        
        # Initialisiere Prediction Models für jeden Symbol
        self.prediction_models = {}
        if use_ml:
            for symbol in symbols:
                self.prediction_models[symbol] = PredictionModel(symbol)
            logger.info(f"Vorhersagemodelle für {len(symbols)} Symbole initialisiert")
        
        # Tracking von Positionen und Orders
        self.positions = {}  # Aktive Positionen
        self.open_orders = {}  # Offene Orders
        
        # Backtest-Daten
        self.backtest_data = {}  # Initialisiere backtest_data als leeres Dictionary
        
        # Performance-Metriken
        self.trades_executed = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_profit_loss = 0.0
        
        logger.info(f"Trading Bot initialisiert mit {len(symbols)} Symbolen")
    
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
    
    def backtest(self,
                start_date: str,
                end_date: Optional[str] = None,
                initial_balance: float = INITIAL_CAPITAL) -> Dict[str, Any]:
        """
        Führt einen Backtest der Handelsstrategie durch.
        
        Args:
            start_date: Startdatum im Format 'YYYY-MM-DD'
            end_date: Enddatum im Format 'YYYY-MM-DD' (optional)
            initial_balance: Anfangskapital für den Backtest
            
        Returns:
            Dictionary mit Backtest-Ergebnissen
        """
        try:
            # Validiere Eingabeparameter
            if not isinstance(start_date, str) or not start_date:
                raise ValueError("start_date muss ein gültiger String sein")
            if end_date and not isinstance(end_date, str):
                raise ValueError("end_date muss ein gültiger String sein")
            if not isinstance(initial_balance, (int, float)) or initial_balance <= 0:
                raise ValueError("initial_balance muss eine positive Zahl sein")
            
            logger.info(f"Starte Backtest von {start_date} bis {end_date or 'heute'} mit {initial_balance} USDT")
            
            # Hole historische Daten
            market_data: Dict[str, pd.DataFrame] = {}
            for symbol in self.symbols:
                try:
                    # Hole historische Daten
                    df = self.data_collector.get_historical_klines(
                        symbol=symbol,
                        interval=self.timeframe,
                        start_str=start_date,
                        end_str=end_date
                    )
                    
                    if df.empty:
                        logger.warning(f"Keine Daten gefunden für {symbol}")
                        continue
                    
                    # Reset index to avoid any issues with duplicates
                    df = df.reset_index()
                    
                    # Ensure we have a datetime column
                    if 'datetime' in df.columns:
                        date_col = 'datetime'
                    elif 'date' in df.columns:
                        date_col = 'date'
                    else:
                        # Create a datetime column from the index if needed
                        df['datetime'] = pd.to_datetime(df.index)
                        date_col = 'datetime'
                    
                    # Convert to datetime and ensure it's in the correct format
                    df[date_col] = pd.to_datetime(df[date_col])
                    
                    # Sort by date
                    df = df.sort_values(by=date_col)
                    
                    # Check for and remove duplicate timestamps
                    if df.duplicated(subset=[date_col]).any():
                        logger.warning(f"Found {df.duplicated(subset=[date_col]).sum()} duplicate timestamps in data for {symbol}")
                        df = df.drop_duplicates(subset=[date_col], keep='first')
                    
                    # Set index to datetime
                    df = df.set_index(date_col)
                    
                    # Füge technische Indikatoren hinzu
                    df = self.indicators.add_all_indicators(df)
                    if df.empty:
                        logger.warning(f"Fehler beim Hinzufügen der Indikatoren für {symbol}")
                        continue
                    
                    # Fülle NaN-Werte
                    df = df.ffill().bfill()
                    
                    # Berechne Signale
                    df = self.strategy.generate_signals(df)
                    if df.empty:
                        logger.warning(f"Fehler beim Generieren der Signale für {symbol}")
                        continue
                    
                    # Speichere DataFrame
                    market_data[symbol] = df
                    logger.info(f"Daten für {symbol} erfolgreich geladen: {len(df)} Datenpunkte")
                    
                except Exception as e:
                    logger.error(f"Fehler beim Laden der Daten für {symbol}: {e}")
                    continue
            
            if not market_data:
                raise ValueError("Keine Marktdaten für Backtest verfügbar")
            
            # Trainiere ML-Modelle wenn aktiviert
            if self.use_ml:
                self.train_prediction_models(market_data)
            
            # Initialisiere Backtest-Variablen
            balance: float = initial_balance
            equity_curve: List[Tuple[str, float]] = [(start_date, initial_balance)]
            trades: List[Dict[str, Any]] = []
            max_balance: float = initial_balance
            winning_trades: int = 0
            losing_trades: int = 0
            
            # Tracking variables for positions
            positions = {}
            for symbol in self.symbols:
                positions[symbol] = []
                
            # Initialize backtest data structures
            self.backtest_data = {}
            for symbol in self.symbols:
                self.backtest_data[symbol] = {
                    'trades': [],
                    'equity_curve': equity_curve.copy(),
                    'results': {}
                }
            
            # Directly process data without using daterange
            try:
                for symbol, df in market_data.items():
                    # Process this symbol's data
                    # First, create a clean dataset for processing
                    processed_df = df.copy()
                    
                    # Track open positions for this symbol
                    open_positions = []
                    trade_history = []
                    
                    # Iterate through each row in the dataframe sequentially
                    for idx, row in processed_df.iterrows():
                        current_datetime = idx
                        current_date_str = current_datetime.strftime('%Y-%m-%d')
                        
                        # Update existing positions with current price
                        updated_positions = []
                        for pos in open_positions:
                            # Update position with current price
                            pos['current_price'] = row['close']
                            
                            # Update highest/lowest prices
                            if row['close'] > pos.get('highest_price', 0):
                                pos['highest_price'] = row['close']
                                # Update trailing stop if price is higher (for long positions)
                                new_trailing_stop = row['close'] * (1 - self.strategy.trailing_stop_pct / 100)
                                pos['trailing_stop'] = max(pos.get('trailing_stop', 0), new_trailing_stop)
                                
                            if row['close'] < pos.get('lowest_price', float('inf')):
                                pos['lowest_price'] = row['close']
                            
                            # Calculate unrealized P&L
                            entry_price = pos.get('entry_price', row['close'])
                            price_diff = row['close'] - entry_price
                            pos['unrealized_pnl'] = price_diff * pos.get('quantity', 0)
                            pos['unrealized_pnl_pct'] = (price_diff / entry_price) * 100 if entry_price > 0 else 0
                            
                            # Check if position should be closed
                            sell_reason = None
                            
                            # Check stop loss
                            if row['close'] <= pos.get('stop_loss', 0):
                                sell_reason = "stop_loss"
                            # Check take profit
                            elif row['close'] >= pos.get('take_profit', float('inf')):
                                sell_reason = "take_profit"
                            # Check trailing stop
                            elif row['close'] <= pos.get('trailing_stop', 0):
                                sell_reason = "trailing_stop"
                            # Check max drawdown
                            elif pos.get('unrealized_pnl_pct', 0) <= -self.strategy.max_drawdown_pct:
                                sell_reason = "max_drawdown"
                            # Check sell signal
                            elif row.get('sell_signal', 0) > 0:
                                # Create a slice of the dataframe up to current row
                                current_slice = processed_df.loc[:idx].tail(10)  # Use last 10 rows for context
                                should_sell, _ = self.strategy.should_sell(current_slice, pos)
                                if should_sell:
                                    sell_reason = "sell_signal"
                            
                            # If we have a reason to sell, close the position
                            if sell_reason:
                                # Close the position
                                close_price = row['close']
                                position_value = pos.get('quantity', 0) * close_price
                                
                                # Calculate final P&L
                                pnl = pos.get('unrealized_pnl', 0)
                                pnl_pct = pos.get('unrealized_pnl_pct', 0)
                                
                                # Update balance
                                balance += position_value
                                
                                # Update trade stats
                                if pnl > 0:
                                    winning_trades += 1
                                else:
                                    losing_trades += 1
                                
                                # Record the closed trade
                                pos['status'] = 'closed'
                                pos['exit_reason'] = sell_reason
                                pos['exit_time'] = current_datetime
                                pos['exit_price'] = close_price
                                pos['realized_pnl'] = pnl
                                pos['realized_pnl_pct'] = pnl_pct
                                
                                trade_history.append(pos)
                                
                                logger.info(
                                    f"Verkauf: {symbol} zu {close_price:.2f} USDT, "
                                    f"PnL: {pnl:.2f} USDT ({pnl_pct:.2f}%), "
                                    f"Grund: {sell_reason}"
                                )
                            else:
                                # Keep the position open
                                updated_positions.append(pos)
                        
                        # Update open positions list
                        open_positions = updated_positions
                        
                        # Check for buy signals if we have capacity for more positions
                        if len(open_positions) < self.max_positions and row.get('buy_signal', 0) > 0:
                            # Create a slice of the dataframe up to current row for context
                            current_slice = processed_df.loc[:idx].tail(10)  # Use last 10 rows for context
                            
                            # Check if we should buy
                            should_buy, signal_strength = self.strategy.should_buy(current_slice, open_positions)
                            
                            if should_buy:
                                # Calculate position size
                                price = row['close']
                                position_size = self._calculate_position_size_simple(
                                    balance=balance,
                                    price=price,
                                    risk_pct=self.risk_percentage,
                                    signal_strength=signal_strength
                                )
                                
                                position_value = position_size * price
                                
                                # Check if we have enough balance
                                if position_value <= balance:
                                    # Create new position
                                    new_position = {
                                        'symbol': symbol,
                                        'entry_price': price,
                                        'quantity': position_size,
                                        'entry_time': current_datetime,
                                        'current_price': price,
                                        'highest_price': price,
                                        'lowest_price': price,
                                        'stop_loss': price * (1 - self.strategy.stop_loss_pct / 100),
                                        'take_profit': price * (1 + self.strategy.take_profit_pct / 100),
                                        'trailing_stop': price * (1 - self.strategy.trailing_stop_pct / 100),
                                        'unrealized_pnl': 0.0,
                                        'unrealized_pnl_pct': 0.0,
                                        'status': 'open',
                                        'signal_strength': signal_strength
                                    }
                                    
                                    # Add position to open positions
                                    open_positions.append(new_position)
                                    
                                    # Update balance
                                    balance -= position_value
                                    
                                    logger.info(
                                        f"Kauf: {symbol} zu {price:.2f} USDT, "
                                        f"Größe: {position_size:.8f}, "
                                        f"Signal-Stärke: {signal_strength:.1f}"
                                    )
                        
                        # Record equity at this point
                        # Calculate total value of all open positions
                        open_value = sum(pos.get('quantity', 0) * pos.get('current_price', 0) for pos in open_positions)
                        total_equity = balance + open_value
                        
                        if total_equity > max_balance:
                            max_balance = total_equity
                            
                        equity_curve.append((current_date_str, total_equity))
                    
                    # After processing all data for this symbol, save results
                    self.backtest_data[symbol]['trades'] = trade_history
                    self.backtest_data[symbol]['equity_curve'] = equity_curve
                
                # Calculate overall backtest results
                total_trades = winning_trades + losing_trades
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
                return_percentage = ((equity_curve[-1][1] - initial_balance) / initial_balance * 100) if equity_curve else 0.0
                
                # Calculate drawdowns
                drawdowns = []
                peak = initial_balance
                for _, value in equity_curve:
                    if value > peak:
                        peak = value
                    drawdown = ((peak - value) / peak) * 100
                    drawdowns.append(drawdown)
                    
                max_drawdown = max(drawdowns) if drawdowns else 0.0
                
                # Calculate Sharpe ratio
                returns = []
                for i in range(1, len(equity_curve)):
                    prev_balance = equity_curve[i-1][1]
                    curr_balance = equity_curve[i][1]
                    if prev_balance > 0:
                        daily_return = (curr_balance - prev_balance) / prev_balance
                        returns.append(daily_return)
                        
                if returns:
                    avg_return = sum(returns) / len(returns)
                    std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5 if len(returns) > 1 else 0
                    
                    # Annual risk-free rate (e.g., 2%)
                    risk_free_rate = 0.02 
                    # Convert to daily rate
                    daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
                    
                    # Calculate Sharpe ratio
                    sharpe_ratio = ((avg_return - daily_rf_rate) / std_return) * np.sqrt(252) if std_return > 0 else 0.0
                    
                    # Calculate Sortino ratio (downside risk)
                    negative_returns = [r for r in returns if r < 0]
                    downside_std = (sum(r ** 2 for r in negative_returns) / len(negative_returns)) ** 0.5 if negative_returns else 0
                    sortino_ratio = ((avg_return - daily_rf_rate) / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0
                else:
                    sharpe_ratio = 0.0
                    sortino_ratio = 0.0
                
                # Log results
                logger.info(
                    f"Backtest abgeschlossen:\n"
                    f"  Rendite: {return_percentage:.2f}%\n"
                    f"  Win-Rate: {win_rate:.2f}%\n"
                    f"  Sharpe: {sharpe_ratio:.2f}\n"
                    f"  Max Drawdown: {max_drawdown:.2f}%\n"
                    f"  Sortino: {sortino_ratio}"
                )
                
                # Return detailed results
                return {
                    'initial_balance': initial_balance,
                    'final_balance': equity_curve[-1][1] if equity_curve else initial_balance,
                    'return_percentage': return_percentage,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'equity_curve': equity_curve,
                    'trades': [trade for symbol_data in self.backtest_data.values() for trade in symbol_data['trades']]
                }
                
            except Exception as e:
                logger.error(f"Fehler während der Trading-Simulation: {e}")
                logger.error(traceback.format_exc())
                return self._get_empty_backtest_results()
            
        except Exception as e:
            logger.error(f"Kritischer Fehler beim Backtest: {e}")
            logger.error(traceback.format_exc())
            return self._get_empty_backtest_results()
    
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
