"""
Hauptmodul des Trading Bots.
Führt alle Komponenten zusammen und implementiert die Hauptlogik.
"""
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import schedule
from typing import Dict, List, Optional, Union, Tuple, Any
from binance.client import Client
from binance.exceptions import BinanceAPIException
import importlib

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.logger import setup_logger
from data.data_collector import DataCollector
from data.indicators import TechnicalIndicators
from strategies.multi_indicator_strategy import MultiIndicatorStrategy
from utils.order_executor import OrderExecutor
from models.prediction_model import PredictionModel
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
                            
                            # Prüfe auf Handelssignale
                            position_update = self.strategy.evaluate_position(
                                position=trades[-1] if trades else None,
                                current_data=current_data
                            )
                            
                            # Prüfe ob Position geschlossen werden soll
                            if position_update.get('should_close', False):
                                # Simuliere Verkauf
                                try:
                                    price = float(current_data['close'])
                                    for trade in trades[:]:  # Kopie der Liste für sichere Iteration
                                        if trade['symbol'] == symbol:
                                            revenue = trade['size'] * price
                                            pnl = revenue - trade['cost']
                                            balance += revenue
                                            
                                            # Aktualisiere Gewinn/Verlust-Statistik
                                            if pnl > 0:
                                                winning_trades += 1
                                            else:
                                                losing_trades += 1
                                            
                                            trade.update({
                                                'exit_price': price,
                                                'exit_date': date_str,
                                                'pnl': pnl,
                                                'pnl_percentage': (pnl / trade['cost']) * 100,
                                                'close_reason': position_update.get('close_reason', 'signal')
                                            })
                                            
                                            trades.remove(trade)
                                            logger.info(
                                                f"Verkauf: {symbol} zu {price:.2f} USDT, "
                                                f"PnL: {pnl:.2f} USDT ({(pnl / trade['cost'] * 100):.2f}%), "
                                                f"Grund: {position_update.get('close_reason', 'signal')}"
                                            )
                                            
                                            if balance > max_balance:
                                                max_balance = balance
                                except Exception as e:
                                    logger.error(f"Fehler beim Verkaufssignal für {symbol}: {e}")
                            
                            # Prüfe auf Kaufsignal wenn keine Position offen ist
                            elif not trades:
                                should_buy, buy_strength = self.strategy.should_buy(pd.DataFrame([current_data]))
                                if should_buy:
                                    # Simuliere Kauf
                                    try:
                                        price = float(current_data['close'])
                                        position_size = (balance * self.risk_percentage / 100) / price
                                        cost = position_size * price
                                        
                                        if cost <= balance:
                                            balance -= cost
                                            trades.append({
                                                'symbol': symbol,
                                                'type': 'BUY',
                                                'entry_price': price,
                                                'size': position_size,
                                                'entry_date': date_str,
                                                'cost': cost,
                                                'signal_strength': buy_strength
                                            })
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
    
    def _analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Union[float, int]]:
        """Analysiert die Handelshistorie und berechnet wichtige Metriken.
        
        Args:
            trades: Liste von Trade-Dictionaries mit Details zu jedem Trade
            
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
                    # Berechne Trade-Dauer
                    if all(key in trade for key in ['exit_date', 'entry_date']):
                        duration = (pd.Timestamp(trade['exit_date']) - 
                                  pd.Timestamp(trade['entry_date'])).total_seconds() / 86400  # Tage
                        durations.append(duration)
                    
                    # Analysiere PnL
                    pnl = float(trade.get('pnl', 0.0))
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
