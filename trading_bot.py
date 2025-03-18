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
                    
                    # Setze den Index auf datetime
                    df.index = pd.to_datetime(df.index)
                    
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
            
            # Simuliere Trading
            try:
                date_range = pd.date_range(start=start_date, end=end_date or pd.Timestamp.now())
                for date in date_range:
                    date_str = date.strftime('%Y-%m-%d')
                    
                    for symbol, df in market_data.items():
                        if date_str not in df.index:
                            continue
                            
                        # Hole Daten für das aktuelle Datum
                        try:
                            # Konvertiere date_str zu Timestamp
                            date_timestamp = pd.Timestamp(date_str)
                            
                            # Prüfe ob Datum im Index
                            if date_timestamp not in df.index:
                                continue
                            
                            # Extrahiere die Series für das aktuelle Datum
                            current_data = df.loc[date_timestamp]
                            
                            # Prüfe ob wir eine Series haben
                            if isinstance(current_data, pd.DataFrame):
                                if len(current_data) == 1:
                                    current_data = current_data.iloc[0]
                                else:
                                    logger.error(f"Mehrere Einträge für {symbol} am {date_str}")
                                    continue
                            
                            # Prüfe auf fehlende Werte
                            if current_data.isnull().any():
                                logger.warning(f"Fehlende Werte für {symbol} am {date_str}")
                                continue
                            
                            # Find active position for this symbol if it exists
                            current_position = None
                            for trade in trades:
                                if trade.symbol == symbol:
                                    current_position = trade
                                    break
                            
                            # Update current price for the position
                            if current_position:
                                current_position.current_price = float(current_data['close'])
                            
                            # Prüfe auf Handelssignale
                            position_update = self.strategy.evaluate_position(
                                position=current_position,
                                current_data=pd.DataFrame([current_data])
                            )
                            
                            # Prüfe ob Position geschlossen werden soll
                            # Unpack the tuple returned by evaluate_position
                            status, reason = position_update
                            
                            # Prüfe ob Status SELL ist
                            if status == PositionStatus.SELL and current_position:
                                # Simuliere Verkauf
                                try:
                                    price = float(current_data['close'])
                                    
                                    # Calculate profit/loss
                                    revenue = current_position.quantity * price
                                    cost = current_position.quantity * current_position.entry_price
                                    pnl = revenue - cost
                                    
                                    # Add back the revenue to balance
                                    balance += revenue
                                    
                                    # Aktualisiere Gewinn/Verlust-Statistik
                                    if pnl > 0:
                                        winning_trades += 1
                                    else:
                                        losing_trades += 1
                                    
                                    # Close the position
                                    current_position.close(price, reason)
                                    
                                    # Record the closed trade for analysis
                                    closed_trade = {
                                        'symbol': current_position.symbol,
                                        'type': 'BUY',
                                        'entry_price': current_position.entry_price,
                                        'exit_price': price,
                                        'size': current_position.quantity,
                                        'entry_date': current_position.entry_time,
                                        'exit_date': date_str,
                                        'cost': cost,
                                        'pnl': pnl,
                                        'pnl_percentage': (pnl / cost) * 100,
                                        'close_reason': reason
                                    }
                                    
                                    # Remove the position from active trades
                                    trades.remove(current_position)
                                    
                                    logger.info(
                                        f"Verkauf: {symbol} zu {price:.2f} USDT, "
                                        f"PnL: {pnl:.2f} USDT ({(pnl / cost * 100):.2f}%), "
                                        f"Grund: {reason}"
                                    )
                                    
                                    if balance > max_balance:
                                        max_balance = balance
                                except Exception as e:
                                    logger.error(f"Fehler beim Verkaufssignal für {symbol}: {e}")
                            
                            # Prüfe auf Kaufsignal wenn keine Position offen ist
                            elif not current_position:
                                should_buy, buy_strength = self.strategy.should_buy(pd.DataFrame([current_data]))
                                if should_buy:
                                    # Simuliere Kauf
                                    try:
                                        price = float(current_data['close'])
                                        position_size = (balance * self.risk_percentage / 100) / price
                                        cost = position_size * price
                                        
                                        if cost <= balance:
                                            # Subtract cost from balance
                                            balance -= cost
                                            
                                            # Create a new Position object
                                            new_position = Position(
                                                symbol=symbol,
                                                entry_price=price,
                                                quantity=position_size,
                                                entry_time=date_timestamp
                                            )
                                            
                                            # Add attributes specific to our backtest
                                            new_position.cost = cost
                                            new_position.signal_strength = buy_strength
                                            
                                            # Add to active trades
                                            trades.append(new_position)
                                            
                                            logger.info(
                                                f"Kauf: {symbol} zu {price:.2f} USDT, "
                                                f"Größe: {position_size:.8f}, "
                                                f"Signal-Stärke: {buy_strength:.1f}"
                                            )
                                    except Exception as e:
                                        logger.error(f"Fehler beim Kaufsignal für {symbol}: {e}")
                        except Exception as e:
                            logger.error(f"Fehler bei der Verarbeitung von {symbol} am {date_str}: {e}")
                    
                    # Aktualisiere Equity-Kurve
                    equity_curve.append((date_str, balance))
                
            except Exception as e:
                logger.error(f"Fehler während der Trading-Simulation: {e}")
            
            # Berechne erweiterte Performance-Metriken
            try:
                # Basis-Metriken
                total_trades: int = winning_trades + losing_trades
                win_rate: float = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
                return_percentage: float = ((balance - initial_balance) / initial_balance) * 100
                
                # Drawdown-Analyse
                drawdowns: List[float] = []
                peak: float = initial_balance
                for _, current_balance in equity_curve:
                    if current_balance > peak:
                        peak = current_balance
                    drawdown = ((peak - current_balance) / peak) * 100
                    drawdowns.append(drawdown)
                
                max_drawdown: float = max(drawdowns) if drawdowns else 0.0
                avg_drawdown: float = sum(drawdowns) / len(drawdowns) if drawdowns else 0.0
                
                # Rendite-Analyse
                returns: List[float] = []
                for i in range(1, len(equity_curve)):
                    prev_balance = equity_curve[i-1][1]
                    curr_balance = equity_curve[i][1]
                    if prev_balance > 0:  # Verhindere Division durch Null
                        daily_return = (curr_balance - prev_balance) / prev_balance
                        returns.append(daily_return)
                
                returns_series = pd.Series(returns)
                if len(returns_series) > 1:
                    returns_std: float = float(returns_series.std() or 0.0)
                    avg_return: float = float(returns_series.mean() or 0.0)
                    risk_free_rate: float = 0.02  # 2% jährliche risikofreie Rendite
                    daily_rf_rate: float = (1 + risk_free_rate) ** (1/252) - 1
                    
                    # Risiko-Metriken
                    sharpe_ratio: float = ((avg_return - daily_rf_rate) / returns_std) * np.sqrt(252) if returns_std != 0 else 0.0
                    
                    # Downside-Risiko
                    negative_returns = returns_series[returns_series < 0]
                    downside_std: float = float(negative_returns.std() or 0.0)
                    sortino_ratio: float = ((avg_return - daily_rf_rate) / downside_std) * np.sqrt(252) if downside_std != 0 else 0.0
                    
                    # Calmar Ratio
                    calmar_ratio: float = abs(return_percentage / max_drawdown) if max_drawdown != 0 else 0.0
                else:
                    returns_std = avg_return = sharpe_ratio = sortino_ratio = calmar_ratio = 0.0
                
                # Handelsanalyse
                trade_metrics = self._analyze_trades(trades)
                
                results = {
                    'initial_balance': initial_balance,
                    'final_balance': balance,
                    'return_percentage': return_percentage,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    
                    # Drawdown-Metriken
                    'max_drawdown': max_drawdown,
                    'avg_drawdown': avg_drawdown,
                    
                    # Risiko-Metriken
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'calmar_ratio': calmar_ratio,
                    'volatility': returns_std * np.sqrt(252),  # Annualisierte Volatilität
                    'avg_return': avg_return * 252,  # Annualisierte Rendite
                    
                    # Handelsmetriken
                    **trade_metrics,
                    
                    # Rohdaten
                    'trades': trades,
                    'equity_curve': equity_curve
                }
                
                # Logge detaillierte Performance
                logger.info(
                    f"Backtest abgeschlossen:\n"
                    f"  Rendite: {return_percentage:.2f}%\n"
                    f"  Win-Rate: {win_rate:.2f}%\n"
                    f"  Sharpe: {sharpe_ratio:.2f}\n"
                    f"  Max Drawdown: {max_drawdown:.2f}%\n"
                    f"  Sortino: {sortino_ratio:.2f}"
                )
                
                return results
                
            except Exception as e:
                logger.error(f"Fehler bei der Berechnung der Performance-Metriken: {e}")
                return self._get_empty_backtest_results()
            
        except Exception as e:
            logger.error(f"Kritischer Fehler beim Backtest: {e}")
            return self._get_empty_backtest_results()
    
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
    
    def _calculate_max_drawdown(self, equity_curve: List[Tuple[str, float]]) -> float:
        """
        Berechnet den maximalen Drawdown aus der Equity-Kurve.
        
        Args:
            equity_curve: Liste von (Datum, Wert) Tupeln
            
        Returns:
            Maximaler Drawdown in Prozent
        """
        try:
            values = [v for _, v in equity_curve]
            peak = values[0]
            max_dd = 0
            
            for value in values[1:]:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            
            return max_dd
        except Exception as e:
            logger.error(f"Fehler bei der Drawdown-Berechnung: {e}")
            return 0.0
    
    def _analyze_trades(self, trades: List[Union[Dict[str, Any], 'Position']]) -> Dict[str, Union[float, int]]:
        """Analysiert die Handelshistorie und berechnet wichtige Metriken.
        
        Args:
            trades: Liste von Trade-Objekten (Position oder Dictionary) mit Details zu jedem Trade
            
        Returns:
            Dictionary mit Handelsmetriken
        """
        try:
            if not trades:
                return {
                    'avg_trade_duration': 0.0,
                    'avg_profit_per_trade': 0.0,
                    'profit_factor': 0.0,
                    'max_consecutive_wins': 0,
                    'max_consecutive_losses': 0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'risk_reward_ratio': 0.0
                }
            
            # Trade-Dauer Analyse
            durations: List[float] = []
            profits: List[float] = []
            losses: List[float] = []
            consecutive_wins: int = 0
            consecutive_losses: int = 0
            max_consecutive_wins: int = 0
            max_consecutive_losses: int = 0
            
            for trade in trades:
                try:
                    # Check if trade is a Position object or dictionary
                    is_position = hasattr(trade, 'entry_time') and hasattr(trade, 'unrealized_pnl')
                    
                    # Berechne Trade-Dauer
                    if is_position:
                        if trade.close_time is not None and trade.entry_time is not None:
                            duration = (trade.close_time - trade.entry_time).total_seconds() / 86400  # Tage
                            durations.append(duration)
                        pnl = trade.unrealized_pnl
                    else:
                        # Handle as dictionary
                        if all(key in trade for key in ['exit_date', 'entry_date']):
                            duration = (pd.Timestamp(trade['exit_date']) - 
                                      pd.Timestamp(trade['entry_date'])).total_seconds() / 86400  # Tage
                            durations.append(duration)
                        pnl = float(trade.get('pnl', 0.0))
                    
                    # Analysiere PnL
                    if pnl > 0:
                        profits.append(pnl)
                        consecutive_wins += 1
                        consecutive_losses = 0
                        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    elif pnl < 0:
                        losses.append(abs(pnl))  # Absolute Werte für Verluste
                        consecutive_losses += 1
                        consecutive_wins = 0
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                
                except Exception as e:
                    logger.warning(f"Fehler bei der Analyse eines Trades: {e}")
                    continue
            
            # Berechne Metriken mit Null-Division-Schutz
            avg_trade_duration = float(sum(durations) / len(durations)) if durations else 0.0
            avg_profit_per_trade = float(sum(profits) / len(trades)) if trades else 0.0
            
            total_profit = sum(profits)
            total_loss = sum(losses)
            profit_factor = float(total_profit / total_loss) if total_loss > 0 else float('inf')
            
            largest_win = float(max(profits)) if profits else 0.0
            largest_loss = float(max(losses)) if losses else 0.0
            avg_win = float(sum(profits) / len(profits)) if profits else 0.0
            avg_loss = float(sum(losses) / len(losses)) if losses else 0.0
            
            # Risiko-Rendite-Verhältnis
            risk_reward_ratio = float(avg_win / avg_loss) if avg_loss > 0 else float('inf')
            
            return {
                'avg_trade_duration': avg_trade_duration,
                'avg_profit_per_trade': avg_profit_per_trade,
                'profit_factor': profit_factor,
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'risk_reward_ratio': risk_reward_ratio
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Trade-Analyse: {e}")
            return {
                'avg_trade_duration': 0.0,
                'avg_profit_per_trade': 0.0,
                'profit_factor': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'risk_reward_ratio': 0.0
            }
    
    def _calculate_sharpe_ratio(self, equity_curve: List[Tuple[str, float]], risk_free_rate: float = 0.01) -> float:
        """
        Berechnet das Sharpe Ratio aus der Equity-Kurve.
        
        Args:
            equity_curve: Liste von (Datum, Wert) Tupeln
            risk_free_rate: Risikofreier Zinssatz (Standard: 1%)
            
        Returns:
            Sharpe Ratio
        """
        try:
            values = np.array([v for _, v in equity_curve])
            returns = np.diff(values) / values[:-1]
            
            if len(returns) < 2:
                return 0.0
                
            # Annualisiere Werte (angenommen tägliche Daten)
            avg_return = np.mean(returns) * 252
            std_return = np.std(returns) * np.sqrt(252)
            
            if std_return == 0:
                return 0.0
                
            sharpe = (avg_return - risk_free_rate) / std_return
            return sharpe
            
        except Exception as e:
            logger.error(f"Fehler bei der Sharpe Ratio Berechnung: {e}")
            return 0.0

    def _process_data(self, symbol: str, data: pd.DataFrame, is_historical: bool = False) -> None:
        """
        Verarbeitet die Daten für ein Symbol, generiert Signale und führt Trades aus.
        
        Args:
            symbol: Das zu verarbeitende Symbol
            data: Die zu verarbeitenden Daten
            is_historical: Ob es sich um historische Daten handelt
        """
        try:
            if data.empty:
                logger.warning(f"Keine Daten für {symbol} vorhanden")
                return
                
            # Füge technische Indikatoren hinzu
            data_with_indicators = self.indicators.add_indicators(data.copy())
            
            # Generiere Handelssignale mit der Strategie
            data_with_signals = self.strategy.generate_signals(data_with_indicators)
            
            # Hole die letzten n Datenpunkte für die Verarbeitung
            current_data = data_with_signals.iloc[-min(len(data_with_signals), 10):]
            
            # Bei Backtest verarbeite alle Daten
            if is_historical:
                self._process_historical_data(symbol, data_with_signals)
                return
                
            # Im Live-Modus prüfe auf Signals und handle entsprechend
            self._check_for_signals(symbol, current_data)
            
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung von {symbol}: {e}")
            logger.debug(traceback.format_exc())

    def _check_for_signals(self, symbol: str, current_data: pd.DataFrame) -> None:
        """
        Prüft auf Handelssignale und führt entsprechende Aktionen aus.
        
        Args:
            symbol: Das zu prüfende Symbol
            current_data: Die aktuellen Daten mit Signalen
        """
        try:
            if current_data.empty:
                logger.warning(f"Keine aktuellen Daten für Signal-Prüfung bei {symbol}")
                return
                
            # Hole aktuelle Positionen für dieses Symbol
            current_positions = self.position_manager.get_positions(symbol)
            trades = current_positions if current_positions else []
            
            # Wenn keine offene Position besteht, prüfe auf Kaufsignal
            if not trades:
                # Prüfe auf Kaufsignal
                should_buy, signal_strength = self.strategy.should_buy(current_data, trades)
                
                if should_buy:
                    # Berechne Positionsgröße
                    latest_data = current_data.iloc[-1]
                    position_size = self._calculate_position_size(symbol, latest_data, signal_strength)
                    
                    if position_size > 0:
                        # Führe Kauf aus
                        order_result = self.executor.execute_order(
                            symbol=symbol,
                            side='buy',
                            quantity=position_size,
                            is_test=self.testnet
                        )
                        
                        if order_result and 'orderId' in order_result:
                            logger.info(f"Kaufauftrag für {symbol} erfolgreich ausgeführt: {order_result}")
                        else:
                            logger.warning(f"Fehler beim Kaufauftrag für {symbol}: {order_result}")
            
            # Wenn eine offene Position besteht, prüfe auf Verkaufssignal oder aktualisiere Position
            else:
                # Hole Position für dieses Symbol
                for trade in trades:
                    # Prüfe Positionsstatus
                    status, reason = self.strategy.evaluate_position(
                        position=trade,
                        current_data=current_data
                    )
                    
                    # Wenn Position geschlossen werden soll
                    if status == PositionStatus.SELL:
                        logger.info(f"Verkaufssignal für {symbol}: {reason}")
                        
                        # Führe Verkauf aus
                        order_result = self.executor.execute_order(
                            symbol=symbol,
                            side='sell',
                            quantity=trade.get('quantity', 0),
                            is_test=self.testnet
                        )
                        
                        if order_result and 'orderId' in order_result:
                            logger.info(f"Verkaufsauftrag für {symbol} erfolgreich ausgeführt: {order_result}")
                            # Entferne Position nach erfolgreichem Verkauf
                            self.position_manager.remove_position(symbol, trade.get('id'))
                        else:
                            logger.warning(f"Fehler beim Verkaufsauftrag für {symbol}: {order_result}")
                    
                    elif status == PositionStatus.REDUCE:
                        logger.info(f"Reduziere Position für {symbol}: {reason}")
                        # Implementiere hier Logik zum teilweisen Verkauf (z.B. 50% der Position)
                        reduce_quantity = trade.get('quantity', 0) * 0.5
                        
                        if reduce_quantity > 0:
                            order_result = self.executor.execute_order(
                                symbol=symbol,
                                side='sell',
                                quantity=reduce_quantity,
                                is_test=self.testnet
                            )
                            
                            if order_result and 'orderId' in order_result:
                                logger.info(f"Reduktionsauftrag für {symbol} erfolgreich ausgeführt: {order_result}")
                                # Aktualisiere Position nach erfolgreicher Reduzierung
                                trade['quantity'] -= reduce_quantity
                                self.position_manager.update_position(symbol, trade)
                            else:
                                logger.warning(f"Fehler beim Reduktionsauftrag für {symbol}: {order_result}")
                    
                    else:  # HOLD
                        # Aktualisiere Position (z.B. höchster Preis für Trailing-Stop)
                        latest_data = current_data.iloc[-1]
                        trade['current_price'] = float(latest_data['close'])
                        
                        # Berechne aktuellen P&L
                        if 'entry_price' in trade:
                            entry_price = float(trade['entry_price'])
                            current_price = float(latest_data['close'])
                            trade['unrealized_pnl'] = (current_price - entry_price) / entry_price * 100
                            
                        self.position_manager.update_position(symbol, trade)
                        logger.debug(f"Position für {symbol} aktualisiert: {trade}")
        
        except Exception as e:
            logger.error(f"Fehler bei der Signal-Prüfung für {symbol}: {e}")
            logger.debug(traceback.format_exc())

    def _process_historical_data(self, symbol: str, data_with_signals: pd.DataFrame) -> None:
        """
        Processes historical data as part of the backtest.
        
        Args:
            symbol: Trading symbol
            data_with_signals: Historical data with trading signals
        """
        if data_with_signals.empty:
            logger.error(f"No historical data available for {symbol}")
            return
            
        # Make a copy to avoid modifying the original
        data = data_with_signals.copy()
        
        try:
            # Reset index to avoid duplicate index issues
            data = data.reset_index()
            
            # Ensure index is datetime and sort by it
            if 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
                data = data.set_index('datetime', drop=False)
            elif 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date', drop=False)
            
            # Ensure index is unique
            if not data.index.is_unique:
                logger.warning(f"Duplicate index values found in data for {symbol}, keeping first occurrence")
                data = data[~data.index.duplicated(keep='first')]
                
            # Sort by index to ensure chronological order
            data = data.sort_index()
            
            # Keep track of positions, pending orders, etc. for this symbol in the backtest
            self.positions[symbol] = None
            self.pending_orders[symbol] = []
            self.trade_history[symbol] = []
            self.equity_curve = [(str(data.index[0]), self.initial_balance)]
            self.last_balance = self.initial_balance
            
            # Set up backtest metrics tracking
            self.backtest_metrics[symbol] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0
            }
            
            # Group data by date to simulate daily trading
            date_groups = data.groupby(data.index.date)
            
            # Process data day by day for more realistic backtesting
            for date, day_data in date_groups:
                try:
                    day_date_str = str(date)
                    
                    # Process each candle in the day
                    for idx, candle in day_data.iterrows():
                        current_datetime = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)
                        
                        # Process buy signals (only if not already in a position)
                        if candle['buy_signal'] > 0 and (self.positions[symbol] is None or len(self.positions[symbol]) < self.max_positions):
                            # Check if we should buy
                            should_buy, signal_strength = self.strategy.should_buy(day_data.loc[:idx], self.positions[symbol])
                            
                            if should_buy:
                                # Calculate position size
                                size = self._calculate_position_size(symbol, candle, signal_strength)
                                
                                if size > 0:
                                    # Create a new position
                                    price = candle['close']
                                    position_value = size * price
                                    
                                    # Check if we have enough balance
                                    if position_value <= self.last_balance:
                                        # Create position
                                        position = {
                                            'symbol': symbol,
                                            'entry_price': price,
                                            'quantity': size,
                                            'entry_time': current_datetime,
                                            'unrealized_pnl': 0.0,
                                            'unrealized_pnl_pct': 0.0,
                                            'highest_price': price,
                                            'lowest_price': price,
                                            'current_price': price,
                                            'stop_loss': price * (1 - self.strategy.stop_loss_pct / 100),
                                            'take_profit': price * (1 + self.strategy.take_profit_pct / 100),
                                            'trailing_stop': price * (1 - self.strategy.trailing_stop_pct / 100),
                                            'status': 'open',
                                            'exit_reason': None,
                                            'signal_strength': signal_strength
                                        }
                                        
                                        # Update the trailing stop based on ATR if available
                                        if 'atr' in candle and candle['atr'] > 0:
                                            atr_stop_distance = candle['atr'] * 1.5
                                            atr_based_stop = price - atr_stop_distance
                                            position['stop_loss'] = max(position['stop_loss'], atr_based_stop)
                                            position['trailing_stop'] = position['stop_loss']
                                        
                                        # Add position to list
                                        if self.positions[symbol] is None:
                                            self.positions[symbol] = [position]
                                        else:
                                            self.positions[symbol].append(position)
                                            
                                        # Update balance
                                        self.last_balance -= position_value
                                        
                                        # Log the trade
                                        logger.info(f"Kauf: {symbol} zu {price:.2f} USDT, Größe: {size:.8f}, Signal-Stärke: {signal_strength:.1f}")
                                    else:
                                        logger.warning(f"Nicht genügend Guthaben für Kauf von {symbol}: {self.last_balance:.2f} USDT < {position_value:.2f} USDT")
                        
                        # Update existing positions
                        if self.positions[symbol]:
                            updated_positions = []
                            for position in self.positions[symbol]:
                                # Update current price and P&L
                                position['current_price'] = candle['close']
                                
                                # Track highest and lowest prices
                                if position['current_price'] > position['highest_price']:
                                    position['highest_price'] = position['current_price']
                                    
                                    # Update trailing stop if price moves up (for long positions)
                                    new_trailing_stop = position['current_price'] * (1 - self.strategy.trailing_stop_pct / 100)
                                    if new_trailing_stop > position['trailing_stop']:
                                        position['trailing_stop'] = new_trailing_stop
                                
                                if position['current_price'] < position['lowest_price']:
                                    position['lowest_price'] = position['current_price']
                                
                                # Calculate unrealized P&L
                                price_diff = position['current_price'] - position['entry_price']
                                position['unrealized_pnl'] = price_diff * position['quantity']
                                position['unrealized_pnl_pct'] = (price_diff / position['entry_price']) * 100
                                
                                # Add unrealized_pnl_pct to candle data for sell signal generation
                                candle_with_pnl = candle.copy()
                                candle_with_pnl['unrealized_pnl_pct'] = position['unrealized_pnl_pct']
                                
                                # Check if we should sell based on:
                                # 1. Sell signal
                                # 2. Stop loss hit
                                # 3. Take profit hit
                                # 4. Trailing stop hit
                                # 5. Max drawdown hit
                                
                                sell_reason = None
                                
                                # 1. Check sell signal
                                if candle['sell_signal'] > 0:
                                    should_sell, signal_strength = self.strategy.should_sell(day_data.loc[:idx].copy(), position)
                                    if should_sell:
                                        sell_reason = "sell_signal"
                                
                                # 2. Check stop loss
                                elif position['current_price'] <= position['stop_loss']:
                                    sell_reason = "stop_loss"
                                
                                # 3. Check take profit
                                elif position['current_price'] >= position['take_profit']:
                                    sell_reason = "take_profit"
                                
                                # 4. Check trailing stop
                                elif position['current_price'] <= position['trailing_stop']:
                                    sell_reason = "trailing_stop"
                                
                                # 5. Check max drawdown
                                elif position['unrealized_pnl_pct'] <= -self.strategy.max_drawdown_pct:
                                    sell_reason = "max_drawdown"
                                
                                # If we have a reason to sell, close the position
                                if sell_reason:
                                    # Close position
                                    close_price = position['current_price']
                                    position_value = position['quantity'] * close_price
                                    
                                    # Update balance
                                    self.last_balance += position_value
                                    
                                    # Calculate final P&L
                                    pnl = position['unrealized_pnl']
                                    pnl_pct = position['unrealized_pnl_pct']
                                    
                                    # Mark position as closed
                                    position['status'] = 'closed'
                                    position['exit_reason'] = sell_reason
                                    position['exit_time'] = current_datetime
                                    position['exit_price'] = close_price
                                    position['realized_pnl'] = pnl
                                    position['realized_pnl_pct'] = pnl_pct
                                    
                                    # Add to trade history
                                    self.trade_history[symbol].append(position)
                                    
                                    # Update metrics
                                    self.backtest_metrics[symbol]['trades'] += 1
                                    if pnl > 0:
                                        self.backtest_metrics[symbol]['wins'] += 1
                                        self.backtest_metrics[symbol]['total_profit'] += pnl
                                    else:
                                        self.backtest_metrics[symbol]['losses'] += 1
                                        self.backtest_metrics[symbol]['total_loss'] += abs(pnl)
                                    
                                    # Log the trade
                                    logger.info(f"Verkauf: {symbol} zu {close_price:.2f} USDT, PnL: {pnl:.2f} USDT ({pnl_pct:.2f}%), Grund: {sell_reason}")
                                else:
                                    # Keep position open
                                    updated_positions.append(position)
                            
                            # Update positions list (removing closed positions)
                            self.positions[symbol] = updated_positions if updated_positions else None
                        
                        # Record equity at this point
                        self.equity_curve.append((str(current_datetime), self.last_balance + self._calculate_open_positions_value()))
                    
                except Exception as e:
                    logger.error(f"Fehler bei der Verarbeitung von {symbol} am {day_date_str}: {e}")
            
            # Calculate final backtest metrics
            self._calculate_backtest_metrics(symbol)
            
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung der historischen Daten für {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _calculate_backtest_metrics(self, symbol: str) -> None:
        """
        Berechnet die Performance-Metriken für die Backtest-Daten eines Symbols.
        
        Args:
            symbol: Das Symbol, für das Metriken berechnet werden sollen
        """
        try:
            if symbol not in self.backtest_data:
                logger.warning(f"Keine Backtest-Daten für {symbol} vorhanden")
                return
                
            backtest_data = self.backtest_data[symbol]
            trades = backtest_data['trades']
            equity_curve = backtest_data['equity_curve']
            
            # Wenn keine Trades vorhanden sind, setze Default-Werte
            if not trades:
                backtest_data['results'] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'avg_profit': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'total_return': (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100 if equity_curve else 0.0
                }
                return
                
            # Berechne Basis-Metriken
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            total_trades = len(trades)
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0
            
            # Profit- und Verlust-Analyse
            total_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0.0
            total_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0.0
            
            avg_profit = total_profit / win_count if win_count > 0 else 0.0
            avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Drawdown-Analyse
            drawdowns = []
            peak = equity_curve[0]
            for balance in equity_curve:
                if balance > peak:
                    peak = balance
                drawdown = (peak - balance) / peak * 100
                drawdowns.append(drawdown)
            
            max_drawdown = max(drawdowns) if drawdowns else 0.0
            
            # Return-Berechnung
            initial_balance = equity_curve[0]
            final_balance = equity_curve[-1]
            total_return = (final_balance - initial_balance) / initial_balance * 100
            
            # Risiko-Kennzahlen
            if len(equity_curve) > 1:
                # Berechne tägliche Returns
                returns = []
                for i in range(1, len(equity_curve)):
                    prev = equity_curve[i-1]
                    curr = equity_curve[i]
                    if prev > 0:  # Verhindere Division durch Null
                        daily_return = (curr - prev) / prev
                        returns.append(daily_return)
                
                # Risiko-Metriken
                if returns:
                    returns_mean = sum(returns) / len(returns)
                    returns_std = (sum((r - returns_mean) ** 2 for r in returns) / len(returns)) ** 0.5
                    
                    # Sharpe Ratio (annualisiert)
                    risk_free_rate = 0.02 / 252  # Tägliche risikofreie Rendite (2% p.a.)
                    sharpe_ratio = ((returns_mean - risk_free_rate) / returns_std) * (252 ** 0.5) if returns_std > 0 else 0.0
                    
                    # Sortino Ratio (annualisiert)
                    negative_returns = [r for r in returns if r < 0]
                    if negative_returns:
                        downside_std = (sum(r ** 2 for r in negative_returns) / len(negative_returns)) ** 0.5
                        sortino_ratio = ((returns_mean - risk_free_rate) / downside_std) * (252 ** 0.5) if downside_std > 0 else 0.0
                    else:
                        sortino_ratio = float('inf')
                else:
                    sharpe_ratio = 0.0
                    sortino_ratio = 0.0
            else:
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
            
            # Speichere Ergebnisse
            backtest_data['results'] = {
                'total_trades': total_trades,
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'total_return': total_return
            }
            
            logger.info(
                f"Backtest-Metriken für {symbol}:\n"
                f"  Rendite: {total_return:.2f}%\n"
                f"  Win-Rate: {win_rate:.2f}%\n"
                f"  Profit-Faktor: {profit_factor:.2f}\n"
                f"  Max Drawdown: {max_drawdown:.2f}%\n"
                f"  Sharpe: {sharpe_ratio:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung der Backtest-Metriken für {symbol}: {e}")
            logger.debug(traceback.format_exc())

    def _calculate_position_size(self, symbol: str, data: pd.Series, signal_strength: float) -> float:
        """
        Calculates the optimal position size for a trade based on risk parameters and market conditions.
        
        Args:
            symbol: Trading symbol
            data: Current market data point
            signal_strength: Strength of the trading signal (0.0-1.0)
            
        Returns:
            Quantity to trade
        """
        try:
            # Get account balance
            balance = self.executor.get_account_balance()
            if balance <= 0:
                logger.warning(f"Account balance is too low for trading: {balance} USDT")
                return 0.0
                
            # Get current price
            current_price = data['close']
            if current_price <= 0:
                logger.warning(f"Invalid price for {symbol}: {current_price}")
                return 0.0
                
            # Calculate maximum position size based on account balance
            max_position_value = balance / self.max_positions
                
            # Apply risk percentage to determine risk amount
            # Adjust risk based on signal strength - stronger signals get more allocation
            adjusted_risk = self.risk_percentage * (0.8 + (signal_strength * 0.4))
            risk_amount = (adjusted_risk / 100) * balance
                
            # Get ATR for volatility-based position sizing
            atr = data.get('atr', 0)
            if atr <= 0:
                atr = current_price * 0.01  # Default to 1% if ATR not available
                
            # Calculate position size based on ATR (volatility-adjusted position sizing)
            if self.strategy.volatility_adjustment and atr > 0:
                # More conservative stops for higher volatility
                stop_distance = atr * 1.5
                if stop_distance > 0:
                    # Position size = risk amount / stop distance
                    position_size = risk_amount / stop_distance
                    
                    # Convert to quantity
                    quantity = position_size / current_price
                else:
                    # Fallback to simpler calculation
                    quantity = risk_amount / current_price
            else:
                # Simple position sizing using fixed risk percentage
                quantity = risk_amount / current_price
                
            # Limit position value to maximum allowed per position
            if quantity * current_price > max_position_value:
                quantity = max_position_value / current_price
                
            # Apply signal strength adjustment
            # Stronger signals get closer to full allocation, weaker signals get reduced
            quantity *= (0.7 + (signal_strength * 0.5))
                
            # Round to appropriate precision based on symbol filters
            symbol_info = self.executor.get_symbol_filters(symbol)
            if symbol_info and 'LOT_SIZE' in symbol_info:
                step_size = symbol_info['LOT_SIZE'].get('stepSize', 0.00001)
                quantity = self.executor.round_step_size(quantity, float(step_size))
                
            logger.debug(f"Calculated position size for {symbol}: {quantity} (value: {quantity * current_price:.2f} USDT)")
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    # Add a new method to dynamically adjust stop losses based on market conditions
    def _adjust_stop_levels(self, position, current_data):
        """
        Dynamically adjusts stop loss and take profit levels based on market conditions.
        
        Args:
            position: The current position
            current_data: Current market data
            
        Returns:
            Updated position with adjusted stop levels
        """
        try:
            if position is None or current_data is None:
                return position
                
            # Get ATR for volatility-based adjustments
            atr = current_data.get('atr', 0)
            if atr <= 0:
                atr = current_data['close'] * 0.01  # Default to 1% if ATR not available
                
            # Set initial stop loss if not set
            if position.stop_loss_level is None:
                # For long positions, stop is below entry
                if position.direction == 'long':
                    stop_distance = max(atr * 1.5, position.entry_price * (self.strategy.stop_loss_pct / 100))
                    position.stop_loss_level = position.entry_price - stop_distance
                else:  # For short positions, stop is above entry
                    stop_distance = max(atr * 1.5, position.entry_price * (self.strategy.stop_loss_pct / 100))
                    position.stop_loss_level = position.entry_price + stop_distance
                    
            # Set take profit if not set
            if position.take_profit_level is None:
                # Dynamic take profit based on volatility
                # Higher volatility = higher take profit targets
                volatility_factor = atr / current_data['close']
                adjusted_take_profit = self.strategy.take_profit_pct * (1 + volatility_factor * 10)
                
                # For long positions, take profit is above entry
                if position.direction == 'long':
                    position.take_profit_level = position.entry_price * (1 + adjusted_take_profit / 100)
                else:  # For short positions, take profit is below entry
                    position.take_profit_level = position.entry_price * (1 - adjusted_take_profit / 100)
                    
            # Update trailing stop if needed
            if position.trailing_stop_level is None:
                position.trailing_stop_level = position.stop_loss_level
                
            # Adjust trailing stop based on price movement
            if position.direction == 'long' and current_data['close'] > position.entry_price:
                # Calculate new trailing stop level
                new_stop = current_data['close'] * (1 - self.strategy.trailing_stop_pct / 100)
                
                # Only move stop up, never down
                if new_stop > position.trailing_stop_level:
                    position.trailing_stop_level = new_stop
                    logger.debug(f"Adjusted trailing stop to {position.trailing_stop_level} for {position.symbol}")
                    
            elif position.direction == 'short' and current_data['close'] < position.entry_price:
                # Calculate new trailing stop level for shorts
                new_stop = current_data['close'] * (1 + self.strategy.trailing_stop_pct / 100)
                
                # Only move stop down, never up
                if new_stop < position.trailing_stop_level:
                    position.trailing_stop_level = new_stop
                    logger.debug(f"Adjusted trailing stop to {position.trailing_stop_level} for {position.symbol}")
                    
            return position
                
        except Exception as e:
            logger.error(f"Error adjusting stop levels: {e}")
            return position
