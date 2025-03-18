"""
Strategy Optimizer für Binance Trading Bot.
Dieses Skript testet verschiedene Parametereinstellungen für DOGEBTC-Trading.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import itertools
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import requests
import yfinance as yf
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Eigene Module importieren
from config.config import BINANCE_CONFIG, DATA_DIRECTORY, LOG_DIRECTORY
from data.data_collector import DataCollector
from data.indicators import TechnicalIndicators
from strategy.base_strategy import BaseStrategy

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIRECTORY, 'strategy_optimizer.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('strategy_optimizer')

class StrategyOptimizer:
    """
    Klasse zum Optimieren von Handelsstrategien durch Backtesting mit verschiedenen Parametern.
    """
    
    def __init__(
        self, 
        symbol: str = "DOGEBTC", 
        timeframe: str = "1h",
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        initial_capital: float = 1.0,  # In BTC
        commission_rate: float = 0.001,  # 0.1% Binance Standardgebühr
        data_source: str = "binance"  # 'binance' oder 'yahoo'
    ):
        """
        Initialisiert den Strategy Optimizer.
        
        Args:
            symbol: Das zu testende Handelspaar
            timeframe: Zeitrahmen für die Daten (z.B. "1h", "4h", "1d")
            start_date: Startdatum für Backtesting (YYYY-MM-DD)
            end_date: Enddatum für Backtesting (YYYY-MM-DD), None für aktuelles Datum
            initial_capital: Anfangskapital für das Backtesting
            commission_rate: Gebührensatz für Trades
            data_source: Datenquelle für historische Daten ("binance" oder "yahoo")
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime("%Y-%m-%d")
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.data_source = data_source.lower()
        
        # Datenquelle initialisieren
        if self.data_source == "binance":
            # Binance Client erstellen
            try:
                self.client = Client(
                    BINANCE_CONFIG["API_KEY"],
                    BINANCE_CONFIG["API_SECRET"],
                    testnet=BINANCE_CONFIG["USE_TESTNET"]
                )
                # Daten-Collector initialisieren
                self.data_collector = DataCollector(client=self.client)
                logger.info(f"Binance {'Testnet' if BINANCE_CONFIG['USE_TESTNET'] else 'Live'} API initialisiert")
            except Exception as e:
                logger.error(f"Fehler bei der Initialisierung der Binance API: {e}")
                raise
        elif self.data_source == "yahoo":
            logger.info("Yahoo Finance als Datenquelle initialisiert")
        else:
            raise ValueError(f"Ungültige Datenquelle: {data_source}. Erlaubte Werte: 'binance', 'yahoo'")
        
        # Ergebnisse speichern
        self.results = []
        
        logger.info(f"Strategy Optimizer für {symbol} initialisiert. Zeitraum: {start_date} bis {end_date}")
    
    def load_historical_data(self) -> pd.DataFrame:
        """
        Lädt historische Daten für das angegebene Symbol und den Zeitraum.
        Unterstützt sowohl Binance als auch Yahoo Finance als Datenquellen.
        
        Returns:
            DataFrame mit historischen Daten (OHLCV-Format)
        """
        try:
            # Prüfen, ob Daten bereits lokal gespeichert sind
            data_file = os.path.join(
                DATA_DIRECTORY, 
                f"{self.symbol}_{self.timeframe}_{self.start_date}_{self.end_date}_{self.data_source}.csv"
            )
            
            if os.path.exists(data_file):
                logger.info(f"Lade gespeicherte Daten aus {data_file}")
                df = pd.read_csv(data_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            
            # Daten je nach Quelle abrufen
            if self.data_source == "binance":
                logger.info(f"Lade historische Daten für {self.symbol} von Binance {'Testnet' if BINANCE_CONFIG['USE_TESTNET'] else 'Live'}...")
                # Stelle sicher, dass self.data_collector existiert
                if not hasattr(self, 'data_collector') or self.data_collector is None:
                    raise ValueError("Binance DataCollector wurde nicht korrekt initialisiert")
                    
                df = self.data_collector.get_historical_data(
                    symbol=self.symbol,
                    interval=self.timeframe,
                    start_str=self.start_date,
                    end_str=self.end_date
                )
            
            elif self.data_source == "yahoo":
                logger.info(f"Lade historische Daten für {self.symbol} von Yahoo Finance...")
                # Yahoo Finance verwendet andere Symbole für Kryptowährungen
                # DOGEBTC wird zu DOGE-BTC
                yahoo_symbol = self.symbol.replace('BTC', '-BTC')
                
                try:
                    # Versuche zuerst das Symbol direkt
                    df = yf.download(
                        yahoo_symbol,
                        start=self.start_date,
                        end=self.end_date,
                        interval=self._convert_timeframe_to_yahoo(self.timeframe)
                    )
                    
                    # Wenn keine Daten zurückgegeben werden, versuche alternatives Format
                    if df.empty:
                        logger.warning(f"Keine Daten für {yahoo_symbol} gefunden, versuche alternatives Format...")
                        # Versuche DOGE-BTC=X Format
                        yahoo_symbol = f"{yahoo_symbol}=X"
                        df = yf.download(
                            yahoo_symbol,
                            start=self.start_date,
                            end=self.end_date,
                            interval=self._convert_timeframe_to_yahoo(self.timeframe)
                        )
                    
                    # Standardisiere die Spaltennamen, um mit Binance-Format kompatibel zu sein
                    df.columns = [col.lower() for col in df.columns]
                    if 'adj close' in df.columns:
                        df = df.rename(columns={'adj close': 'close'})
                    
                    # Füge Volumen hinzu, falls es fehlt
                    if 'volume' not in df.columns:
                        df['volume'] = 0
                        
                    # Stelle sicher, dass der Index als 'timestamp' bezeichnet wird
                    df = df.reset_index()
                    df = df.rename(columns={'Date': 'timestamp', 'date': 'timestamp', 'Datetime': 'timestamp', 'datetime': 'timestamp'})
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                    
                except Exception as e:
                    logger.error(f"Fehler beim Abrufen von Yahoo Finance Daten: {e}")
                    # Versuche als Fallback, die Daten über die Binance API zu holen
                    logger.info(f"Versuche Fallback über Binance API...")
                    client = Client(BINANCE_CONFIG["API_KEY"], BINANCE_CONFIG["API_SECRET"], testnet=BINANCE_CONFIG["USE_TESTNET"])
                    data_collector = DataCollector(client=client)
                    df = data_collector.get_historical_data(
                        symbol=self.symbol,
                        interval=self.timeframe,
                        start_str=self.start_date,
                        end_str=self.end_date
                    )
            else:
                raise ValueError(f"Ungültige Datenquelle: {self.data_source}")
            
            # Prüfe, ob Daten erfolgreich geladen wurden
            if df is None or df.empty:
                raise ValueError(f"Keine Daten für {self.symbol} im Zeitraum {self.start_date} bis {self.end_date} gefunden")
            
            # Daten lokal speichern
            df.to_csv(data_file)
            logger.info(f"Daten in {data_file} gespeichert")
            
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der historischen Daten: {e}")
            raise
    
    def _convert_timeframe_to_yahoo(self, timeframe: str) -> str:
        """
        Konvertiert Binance-Zeitrahmen in Yahoo Finance-Zeitrahmen.
        
        Args:
            timeframe: Binance-Zeitrahmen (z.B. "1h", "4h", "1d")
            
        Returns:
            Yahoo Finance-Zeitrahmen
        """
        # Mapping von Binance zu Yahoo Finance Zeitrahmen
        mapping = {
            "1m": "1m",
            "3m": "2m",  # Yahoo hat kein 3m, nächstbestes ist 2m
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "2h": "1h",  # Yahoo hat kein 2h, nächstbestes ist 1h
            "4h": "4h",
            "6h": "4h",  # Yahoo hat kein 6h, nächstbestes ist 4h
            "8h": "4h",  # Yahoo hat kein 8h, nächstbestes ist 4h
            "12h": "4h", # Yahoo hat kein 12h, nächstbestes ist 4h
            "1d": "1d",
            "3d": "1d",  # Yahoo hat kein 3d, nächstbestes ist 1d
            "1w": "1wk",
            "1M": "1mo"
        }
        
        if timeframe not in mapping:
            logger.warning(f"Unbekannter Zeitrahmen: {timeframe}, verwende '1d' als Standard")
            return "1d"
            
        return mapping[timeframe]
    
    def generate_parameter_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generiert alle möglichen Kombinationen der Parameter.
        
        Args:
            param_grid: Dictionary mit Parameternamen und möglichen Werten
            
        Returns:
            Liste von Dictionaries mit Parameterkombinationen
        """
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        result = []
        for combo in combinations:
            result.append(dict(zip(keys, combo)))
        
        logger.info(f"Generierte {len(result)} Parameterkombinationen")
        return result
    
    def backtest_strategy(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt einen Backtest mit den angegebenen Parametern durch.
        
        Args:
            df: DataFrame mit historischen Daten
            params: Dictionary mit Strategieparametern
            
        Returns:
            Dictionary mit Backtesting-Ergebnissen
        """
        # Kopie der Daten erstellen, um Originaldaten nicht zu verändern
        data = df.copy()
        
        # Indikatoren hinzufügen
        data = self._add_indicators_with_params(data, params)
        
        # Handelssignale generieren
        data = self._generate_signals(data, params)
        
        # Backtesting durchführen
        results = self._run_backtest(data, params)
        
        return results
    
    def _add_indicators_with_params(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Fügt technische Indikatoren mit den angegebenen Parametern hinzu.
        
        Args:
            df: DataFrame mit historischen Daten
            params: Dictionary mit Indikatorparametern
            
        Returns:
            DataFrame mit hinzugefügten Indikatoren
        """
        # Validiere Eingabedaten, um NoneType-Fehler zu vermeiden
        if df is None or df.empty:
            logger.warning("DataFrame ist leer oder None, keine Indikatoren werden hinzugefügt")
            return pd.DataFrame()
            
        # Sichere Kopie erstellen
        data = df.copy()
        
        # Stelle sicher, dass alle erforderlichen Spalten vorhanden sind
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Fehlende Spalten im DataFrame: {missing_columns}")
            return data
        
        # Stelle sicher, dass keine NaN-Werte in wichtigen Spalten vorhanden sind
        for col in required_columns:
            if data[col].isnull().any():
                logger.warning(f"NaN-Werte in {col}-Spalte gefunden, werden durch Vorwärtsfüllen ersetzt")
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        # TechnicalIndicators-Instanz erstellen
        indicators_config = {}
        # Übertrage Standardkonfiguration aus INDICATORS_CONFIG
        if hasattr(self, 'config') and 'indicators' in self.config:
            indicators_config = self.config['indicators']
            
        # Erstelle die TechnicalIndicators-Instanz
        indicators = TechnicalIndicators(indicators_config)
        
        # Indikatoren einzeln hinzufügen mit expliziter Fehlerbehandlung
        # Verwende die add_all_indicators-Methode, wenn keine spezifischen Parameter übergeben wurden
        if not params or all(not key.startswith(('rsi_', 'macd_', 'bb_', 'stoch_', 'atr_')) for key in params.keys()):
            try:
                logger.info("Füge alle Standardindikatoren hinzu")
                data = indicators.add_all_indicators(data)
                return data
            except Exception as e:
                logger.error(f"Fehler beim Hinzufügen aller Indikatoren: {e}")
                # Fahre mit einzelnen Indikatoren fort
        
        # RSI hinzufügen
        if any(key.startswith('rsi_') for key in params.keys()):
            try:
                # Extrahiere den Periodenwert oder verwende Standardwert
                rsi_period = params.get('rsi_period', 14)  # Standardwert 14
                # Stelle sicher, dass der Wert ein Integer ist
                if rsi_period is not None and isinstance(rsi_period, (int, float)):
                    rsi_period = int(rsi_period)
                    data = indicators.add_rsi(data)
                    logger.info(f"RSI mit Periode {rsi_period} hinzugefügt")
            except Exception as e:
                logger.error(f"Fehler beim Hinzufügen des RSI: {e}")
        
        # MACD hinzufügen
        if any(key.startswith('macd_') for key in params.keys()):
            try:
                # Extrahiere Parameter oder verwende Standardwerte
                macd_fast = params.get('macd_fast', 12)  # Standardwert 12
                macd_slow = params.get('macd_slow', 26)  # Standardwert 26
                macd_signal = params.get('macd_signal', 9)  # Standardwert 9
                
                # Stelle sicher, dass die Werte Integer sind
                if all(param is not None and isinstance(param, (int, float)) 
                       for param in [macd_fast, macd_slow, macd_signal]):
                    data = indicators.add_macd(data)
                    logger.info(f"MACD mit Parametern fast={macd_fast}, slow={macd_slow}, signal={macd_signal} hinzugefügt")
            except Exception as e:
                logger.error(f"Fehler beim Hinzufügen des MACD: {e}")
        
        # Bollinger Bands hinzufügen
        if any(key.startswith('bb_') for key in params.keys()):
            try:
                # Extrahiere Parameter oder verwende Standardwerte
                bb_period = params.get('bb_period', 20)  # Standardwert 20
                bb_std = params.get('bb_std', 2.0)  # Standardwert 2.0
                
                # Stelle sicher, dass die Werte korrekte Typen haben
                if bb_period is not None and isinstance(bb_period, (int, float)) and \
                   bb_std is not None and isinstance(bb_std, (int, float)):
                    data = indicators.add_bollinger_bands(data)
                    logger.info(f"Bollinger Bands mit Periode {bb_period} und StdDev {bb_std} hinzugefügt")
            except Exception as e:
                logger.error(f"Fehler beim Hinzufügen der Bollinger Bands: {e}")
        
        # Stochastic Oscillator hinzufügen
        if any(key.startswith('stoch_') for key in params.keys()):
            try:
                # Extrahiere Parameter oder verwende Standardwerte
                stoch_k = params.get('stoch_k', 14)  # Standardwert 14
                stoch_d = params.get('stoch_d', 3)   # Standardwert 3
                stoch_smooth = params.get('stoch_smooth', 3)  # Standardwert 3
                
                # Stelle sicher, dass die Werte Integer sind
                if all(param is not None and isinstance(param, (int, float)) 
                       for param in [stoch_k, stoch_d, stoch_smooth]):
                    data = indicators.add_stochastic(data)
                    logger.info(f"Stochastic mit K={stoch_k}, D={stoch_d}, Smooth={stoch_smooth} hinzugefügt")
            except Exception as e:
                logger.error(f"Fehler beim Hinzufügen des Stochastic Oscillator: {e}")
        
        # ATR hinzufügen
        if any(key.startswith('atr_') for key in params.keys()):
            try:
                # Extrahiere den Periodenwert oder verwende Standardwert
                atr_period = params.get('atr_period', 14)  # Standardwert 14
                
                # Stelle sicher, dass der Wert ein Integer ist
                if atr_period is not None and isinstance(atr_period, (int, float)):
                    data = indicators.add_atr(data)
                    logger.info(f"ATR mit Periode {atr_period} hinzugefügt")
            except Exception as e:
                logger.error(f"Fehler beim Hinzufügen des ATR: {e}")
        
        return data
    
    def _generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf den Indikatoren und Parametern.
        
        Args:
            df: DataFrame mit Indikatoren
            params: Dictionary mit Signalparametern
            
        Returns:
            DataFrame mit Handelssignalen
        """
        data = df.copy()
        
        # Initialisiere Signal-Spalte
        data['signal'] = 0
        
        # RSI-Signale
        if 'rsi_oversold' in params and 'rsi_overbought' in params and 'rsi' in data.columns:
            # Kaufsignal, wenn RSI unter oversold ist
            data.loc[data['rsi'] < params['rsi_oversold'], 'rsi_buy_signal'] = 1
            # Verkaufssignal, wenn RSI über overbought ist
            data.loc[data['rsi'] > params['rsi_overbought'], 'rsi_sell_signal'] = -1
        
        # MACD-Signale
        if all(col in data.columns for col in ['macd', 'macd_signal']):
            # Kaufsignal, wenn MACD über Signal-Linie kreuzt
            data['macd_cross'] = 0
            data.loc[(data['macd'] > data['macd_signal']) & 
                     (data['macd'].shift(1) <= data['macd_signal'].shift(1)), 'macd_buy_signal'] = 1
            # Verkaufssignal, wenn MACD unter Signal-Linie kreuzt
            data.loc[(data['macd'] < data['macd_signal']) & 
                     (data['macd'].shift(1) >= data['macd_signal'].shift(1)), 'macd_sell_signal'] = -1
        
        # Bollinger Bands Signale
        if all(col in data.columns for col in ['bb_lower', 'bb_upper', 'close']):
            # Kaufsignal, wenn Preis unter unterem Band
            data.loc[data['close'] < data['bb_lower'], 'bb_buy_signal'] = 1
            # Verkaufssignal, wenn Preis über oberem Band
            data.loc[data['close'] > data['bb_upper'], 'bb_sell_signal'] = -1
        
        # Stochastic Signale
        if all(col in data.columns for col in ['stoch_k', 'stoch_d']):
            # Kaufsignal, wenn Stochastic K über D kreuzt und beide unter 20
            data.loc[(data['stoch_k'] > data['stoch_d']) & 
                     (data['stoch_k'].shift(1) <= data['stoch_d'].shift(1)) &
                     (data['stoch_k'] < 20), 'stoch_buy_signal'] = 1
            # Verkaufssignal, wenn Stochastic K unter D kreuzt und beide über 80
            data.loc[(data['stoch_k'] < data['stoch_d']) & 
                     (data['stoch_k'].shift(1) >= data['stoch_d'].shift(1)) &
                     (data['stoch_k'] > 80), 'stoch_sell_signal'] = -1
        
        # Kombiniere Signale basierend auf der Strategie
        signal_columns = [col for col in data.columns if col.endswith('_buy_signal') or col.endswith('_sell_signal')]
        
        if signal_columns:
            # Gewichtete Summe der Signale basierend auf den Parametern
            for col in signal_columns:
                weight_param = f"{col.split('_')[0]}_weight"
                if weight_param in params:
                    data['signal'] += data[col].fillna(0) * params[weight_param]
                else:
                    data['signal'] += data[col].fillna(0)
        
        # Signalstärke normalisieren
        data['signal_strength'] = data['signal'].abs()
        
        # Diskrete Signale (-1, 0, 1)
        data['signal'] = np.sign(data['signal'])
        
        return data
    
    def _run_backtest(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt den eigentlichen Backtest durch.
        
        Args:
            df: DataFrame mit Signalen
            params: Dictionary mit Strategieparametern
            
        Returns:
            Dictionary mit Backtesting-Ergebnissen
        """
        data = df.copy()
        
        # Initialisiere Portfolio-Werte
        capital = self.initial_capital
        btc_amount = 0
        trades = []
        
        # Parameter für Positionsgrößenberechnung
        risk_per_trade = params.get('risk_per_trade', 0.02)  # 2% Risiko pro Trade
        
        for i in range(1, len(data)):
            current_row = data.iloc[i]
            previous_row = data.iloc[i-1]
            
            # Kaufsignal
            if previous_row['signal'] > 0 and btc_amount == 0:
                # Berechne Positionsgröße basierend auf Risiko
                position_size = capital * risk_per_trade
                
                # Berechne Anzahl der zu kaufenden DOGE
                price = current_row['open']
                doge_amount = position_size / price
                
                # Berechne Gebühren
                fee = position_size * self.commission_rate
                
                # Führe Kauf durch
                btc_amount = position_size - fee
                capital -= position_size
                
                trades.append({
                    'timestamp': current_row.name,
                    'type': 'buy',
                    'price': price,
                    'amount': doge_amount,
                    'value': position_size,
                    'fee': fee
                })
            
            # Verkaufssignal
            elif previous_row['signal'] < 0 and btc_amount > 0:
                # Berechne Verkaufswert
                price = current_row['open']
                position_value = btc_amount * price
                
                # Berechne Gebühren
                fee = position_value * self.commission_rate
                
                # Führe Verkauf durch
                capital += position_value - fee
                btc_amount = 0
                
                trades.append({
                    'timestamp': current_row.name,
                    'type': 'sell',
                    'price': price,
                    'amount': doge_amount,
                    'value': position_value,
                    'fee': fee
                })
        
        # Berechne Endergebnis
        final_value = capital
        if btc_amount > 0:
            # Wenn noch DOGE gehalten werden, zum letzten Preis umrechnen
            final_value += btc_amount * data.iloc[-1]['close']
        
        # Berechne Performance-Metriken
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Berechne Drawdown
        portfolio_values = self._calculate_portfolio_values(data, trades)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # Berechne Sharpe Ratio
        daily_returns = portfolio_values.pct_change().dropna()
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        
        # Berechne Win/Loss Ratio
        win_trades = [t for t in trades if t['type'] == 'sell' and t['value'] > t.get('buy_value', 0)]
        loss_trades = [t for t in trades if t['type'] == 'sell' and t['value'] <= t.get('buy_value', 0)]
        win_ratio = len(win_trades) / len(trades) * 100 if trades else 0
        
        return {
            'params': params,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_ratio': win_ratio,
            'num_trades': len(trades),
            'final_value': final_value,
            'trades': trades
        }
    
    def _calculate_portfolio_values(self, df: pd.DataFrame, trades: List[Dict[str, Any]]) -> pd.Series:
        """
        Berechnet den Portfolio-Wert für jeden Zeitpunkt.
        
        Args:
            df: DataFrame mit Preisdaten
            trades: Liste der durchgeführten Trades
            
        Returns:
            Series mit Portfolio-Werten
        """
        portfolio_values = pd.Series(index=df.index, dtype=float)
        portfolio_values.iloc[0] = self.initial_capital
        
        capital = self.initial_capital
        btc_amount = 0
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            
            # Aktualisiere Portfolio-Wert
            portfolio_value = capital
            if btc_amount > 0:
                portfolio_value += btc_amount * row['close']
            
            portfolio_values.iloc[i] = portfolio_value
            
            # Aktualisiere Kapital und BTC-Menge basierend auf Trades
            for trade in trades:
                if trade['timestamp'] == row.name:
                    if trade['type'] == 'buy':
                        capital -= trade['value']
                        btc_amount = trade['amount']
                    elif trade['type'] == 'sell':
                        capital += trade['value'] - trade['fee']
                        btc_amount = 0
        
        return portfolio_values
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """
        Berechnet den maximalen Drawdown.
        
        Args:
            portfolio_values: Series mit Portfolio-Werten
            
        Returns:
            Maximaler Drawdown in Prozent
        """
        # Berechne kumulative maximale Werte
        running_max = portfolio_values.cummax()
        
        # Berechne Drawdown
        drawdown = (portfolio_values - running_max) / running_max * 100
        
        # Maximaler Drawdown
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Berechnet das Sharpe Ratio.
        
        Args:
            returns: Series mit täglichen Renditen
            risk_free_rate: Risikofreier Zinssatz (annualisiert)
            
        Returns:
            Sharpe Ratio (annualisiert)
        """
        # Täglicher risikofreier Zinssatz
        daily_risk_free = risk_free_rate / 252
        
        # Excess Returns
        excess_returns = returns - daily_risk_free
        
        # Sharpe Ratio
        if excess_returns.std() == 0:
            return 0
        
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        return sharpe
    
    def optimize(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Optimiert die Strategie durch Testen aller Parameterkombinationen.
        
        Args:
            param_grid: Dictionary mit Parameternamen und möglichen Werten
            
        Returns:
            Liste der Ergebnisse, sortiert nach Performance
        """
        try:
            # Lade historische Daten
            df = self.load_historical_data()
            
            # Generiere Parameterkombinationen
            param_combinations = self.generate_parameter_combinations(param_grid)
            
            # Führe Backtests durch
            logger.info(f"Starte Optimierung mit {len(param_combinations)} Parameterkombinationen...")
            
            results = []
            for params in tqdm(param_combinations, desc="Backtesting"):
                result = self.backtest_strategy(df, params)
                results.append(result)
            
            # Sortiere Ergebnisse nach Gesamt-Return
            results.sort(key=lambda x: x['total_return'], reverse=True)
            
            # Speichere Ergebnisse
            self.results = results
            
            # Speichere Top-Ergebnisse in JSON-Datei
            top_results = results[:10]
            results_file = os.path.join(
                DATA_DIRECTORY, 
                f"optimization_results_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(results_file, 'w') as f:
                json.dump(top_results, f, default=str, indent=4)
            
            logger.info(f"Optimierung abgeschlossen. Top-Ergebnisse in {results_file} gespeichert.")
            
            return results
            
        except Exception as e:
            logger.error(f"Fehler bei der Optimierung: {e}")
            raise
    
    def plot_results(self, top_n: int = 3) -> None:
        """
        Plottet die Ergebnisse der Top-N-Strategien.
        
        Args:
            top_n: Anzahl der zu plottenden Top-Strategien
        """
        if not self.results:
            logger.warning("Keine Ergebnisse zum Plotten vorhanden.")
            return
        
        # Wähle Top-N-Strategien
        top_results = self.results[:top_n]
        
        # Lade historische Daten
        df = self.load_historical_data()
        
        # Erstelle Plot
        plt.figure(figsize=(12, 8))
        
        # Plotte Preisdaten
        ax1 = plt.subplot(211)
        ax1.plot(df.index, df['close'], label='DOGEBTC Preis')
        ax1.set_title(f'DOGEBTC Preis und Portfolio-Werte ({self.start_date} bis {self.end_date})')
        ax1.set_ylabel('Preis (BTC)')
        ax1.legend()
        
        # Plotte Portfolio-Werte
        ax2 = plt.subplot(212, sharex=ax1)
        
        for i, result in enumerate(top_results):
            # Berechne Portfolio-Werte
            portfolio_values = self._calculate_portfolio_values(df, result['trades'])
            
            # Plotte Portfolio-Werte
            ax2.plot(portfolio_values.index, portfolio_values, 
                     label=f"Strategie {i+1} (Return: {result['total_return']:.2f}%)")
        
        # Plotte Buy-and-Hold-Strategie
        initial_price = df.iloc[0]['close']
        final_price = df.iloc[-1]['close']
        buy_hold_return = (final_price - initial_price) / initial_price * 100
        
        buy_hold = pd.Series(index=df.index)
        buy_hold = df['close'] / df.iloc[0]['close'] * self.initial_capital
        
        ax2.plot(buy_hold.index, buy_hold, 
                 label=f"Buy & Hold (Return: {buy_hold_return:.2f}%)", 
                 linestyle='--')
        
        ax2.set_ylabel('Portfolio-Wert (BTC)')
        ax2.set_xlabel('Datum')
        ax2.legend()
        
        # Speichere Plot
        plot_file = os.path.join(
            DATA_DIRECTORY, 
            f"optimization_plot_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(plot_file)
        
        logger.info(f"Plot in {plot_file} gespeichert.")
        plt.close()


if __name__ == "__main__":
    # Parameter-Grid für DOGEBTC
    param_grid = {
        # RSI-Parameter
        'rsi_period': [7, 14, 21],
        'rsi_oversold': [20, 25, 30],
        'rsi_overbought': [70, 75, 80],
        'rsi_weight': [0.5, 1.0, 1.5],
        
        # MACD-Parameter
        'macd_fast': [8, 12, 16],
        'macd_slow': [21, 26, 30],
        'macd_signal': [7, 9, 11],
        'macd_weight': [0.5, 1.0, 1.5],
        
        # Bollinger Bands Parameter
        'bb_period': [15, 20, 25],
        'bb_std': [1.8, 2.0, 2.2],
        'bb_weight': [0.5, 1.0, 1.5],
        
        # Stochastic Parameter
        'stoch_k': [9, 14, 18],
        'stoch_d': [3, 5, 7],
        'stoch_smooth': [2, 3, 4],
        'stoch_weight': [0.5, 1.0, 1.5],
        
        # ATR für Volatilitätsanpassung
        'atr_period': [10, 14, 18],
        
        # Risikomanagement
        'risk_per_trade': [0.01, 0.02, 0.03]
    }
    
    # Initialisiere Optimizer
    optimizer = StrategyOptimizer(
        symbol="DOGEBTC",
        timeframe="1h",
        start_date="2023-01-01",
        end_date=None,  # Aktuelles Datum
        initial_capital=1.0,  # 1 BTC
        commission_rate=0.001  # 0.1% Binance Standardgebühr
    )
    
    # Führe Optimierung durch
    results = optimizer.optimize(param_grid)
    
    # Plotte Ergebnisse
    optimizer.plot_results(top_n=3)
    
    # Gib beste Parameter aus
    best_params = results[0]['params']
    print("\nBeste Parameter für DOGEBTC-Trading:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    print(f"\nGesamtrendite: {results[0]['total_return']:.2f}%")
    print(f"Maximaler Drawdown: {results[0]['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {results[0]['sharpe_ratio']:.2f}")
    print(f"Win Ratio: {results[0]['win_ratio']:.2f}%")
    print(f"Anzahl Trades: {results[0]['num_trades']}")
