"""
Technische Indikatoren Modul für den Trading Bot.
Berechnet verschiedene technische Indikatoren für die Marktanalyse.
"""
import os
import pandas as pd
import numpy as np
import pandas_ta as pta
from typing import Dict, List, Optional, Union, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from config.config import INDICATORS_CONFIG, LOG_LEVEL, LOG_FILE, DATA_DIRECTORY

# Logger einrichten
logger = setup_logger(os.path.join(DATA_DIRECTORY, LOG_FILE), LOG_LEVEL)

class TechnicalIndicators:
    """
    Klasse zur Berechnung technischer Indikatoren für Marktdaten.
    """
    
    def __init__(self, config: Dict = INDICATORS_CONFIG):
        """
        Initialisiert die TechnicalIndicators-Klasse.
        
        Args:
            config: Konfigurationswörterbuch für Indikatoren-Parameter
        """
        self.config = config
        logger.info("Technische Indikatoren initialisiert")
    
    def _safe_fillna(self, series: pd.Series, default_value: float = 0) -> pd.Series:
        """
        Sicheres Auffüllen von NaN-Werten mit Forward-Fill und Back-Fill.
        
        Args:
            series: Pandas Series mit möglichen NaN-Werten
            default_value: Standardwert für verbleibende NaN-Werte
            
        Returns:
            Aufgefüllte Pandas Series
        """
        return series.ffill().bfill().fillna(default_value)
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt alle konfigurierten technischen Indikatoren zum DataFrame hinzu.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            DataFrame mit hinzugefügten Indikatoren
        """
        try:
            # Erstelle Kopie, um das Original nicht zu verändern
            df_with_indicators = df.copy()
            
            # Füge verschiedene Indikatoren hinzu
            df_with_indicators = self.add_moving_averages(df_with_indicators)
            df_with_indicators = self.add_rsi(df_with_indicators)
            df_with_indicators = self.add_macd(df_with_indicators)
            df_with_indicators = self.add_bollinger_bands(df_with_indicators)
            df_with_indicators = self.add_atr(df_with_indicators)
            df_with_indicators = self.add_stochastic(df_with_indicators)
            df_with_indicators = self.add_volume_indicators(df_with_indicators)
            df_with_indicators = self.add_volatility(df_with_indicators)
            
            # Entferne NaN-Zeilen aus Berechnungen
            df_with_indicators.dropna(inplace=True)
            
            logger.info(f"Alle technischen Indikatoren hinzugefügt, {len(df_with_indicators)} gültige Datenpunkte")
            return df_with_indicators
            
        except Exception as e:
            logger.error(f"Fehler beim Berechnen der technischen Indikatoren: {e}")
            return df
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt Simple Moving Averages (SMA) und Exponential Moving Averages (EMA) hinzu.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            DataFrame mit hinzugefügten Moving Averages
        """
        try:
            # Validiere Eingabedaten
            if df.empty:
                logger.warning("DataFrame ist leer, überspringe Moving Averages")
                return df
            
            if 'close' not in df.columns:
                logger.error("'close' Spalte nicht im DataFrame gefunden")
                return df
            
            # Stelle sicher, dass Close keine None-Werte enthält
            close_series = df['close'].copy()
            if close_series.isnull().any():
                logger.warning("Null-Werte in close-Daten gefunden, führe Bereinigung durch")
                close_series = close_series.ffill().bfill()
            
            # Konfigurationswerte mit Fehlerbehandlung
            sma_config = self.config.get('SMA', {'short': 10, 'medium': 20, 'long': 50})
            ema_config = self.config.get('EMA', {'short': 9, 'medium': 21, 'long': 55})
            
            # Simple Moving Averages mit Validierung
            for period_name, length in sma_config.items():
                try:
                    sma = pta.sma(close_series, length=length)
                    if sma is None or sma.isnull().all():
                        logger.warning(f"SMA-{period_name} konnte nicht berechnet werden")
                        continue
                    df[f'sma_{period_name}'] = self._safe_fillna(sma, close_series.mean())
                except Exception as e:
                    logger.error(f"Fehler bei SMA-{period_name} Berechnung: {e}")
            
            # Exponential Moving Averages mit Validierung
            for period_name, length in ema_config.items():
                try:
                    ema = pta.ema(close_series, length=length)
                    if ema is None or ema.isnull().all():
                        logger.warning(f"EMA-{period_name} konnte nicht berechnet werden, verwende SMA")
                        ema = pta.sma(close_series, length=length)
                    df[f'ema_{period_name}'] = self._safe_fillna(ema, close_series.mean())
                except Exception as e:
                    logger.error(f"Fehler bei EMA-{period_name} Berechnung: {e}")
            
            # Überprüfe finale Datenqualität
            ma_columns = [col for col in df.columns if col.startswith(('sma_', 'ema_'))]
            nan_counts = {col: df[col].isnull().sum() for col in ma_columns}
            if any(nan_counts.values()):
                logger.warning(f"NaN-Werte in Moving Averages gefunden: {nan_counts}")
            
            logger.debug("Moving Averages erfolgreich hinzugefügt")
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Berechnen der Moving Averages: {e}")
            return df
    
    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt Relative Strength Index (RSI) hinzu.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            DataFrame mit hinzugefügtem RSI
        """
        try:
            # Validiere Eingabedaten
            if df.empty:
                logger.warning("DataFrame ist leer, überspringe RSI-Berechnung")
                return df
            
            if 'close' not in df.columns:
                logger.error("'close' Spalte nicht im DataFrame gefunden")
                return df
            
            # Stelle sicher, dass Close keine None-Werte enthält
            close_series = df['close'].copy()
            if close_series.isnull().any():
                logger.warning("Null-Werte in close-Daten gefunden, führe Bereinigung durch")
                close_series = close_series.ffill().bfill()
            
            # Konfigurationswerte mit Standardwerten
            rsi_config = self.config.get('RSI', {
                'period': 14,
                'overbought': 70,
                'oversold': 30
            })
            
            try:
                # Berechne RSI mit Validierung
                rsi = pta.rsi(close_series, length=rsi_config['period'])
                if rsi is None or rsi.isnull().all():
                    logger.warning("RSI konnte nicht berechnet werden, verwende alternative Methode")
                    # Alternative RSI-Berechnung
                    delta = close_series.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_config['period']).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_config['period']).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                
                # Validiere RSI-Werte
                rsi = self._safe_fillna(rsi, 50)  # 50 als neutraler Wert
                rsi = rsi.clip(0, 100)  # Beschränke Werte auf 0-100
                
                df['rsi'] = rsi
                
                # Füge RSI-Signale hinzu mit Validierung
                df['rsi_overbought'] = df['rsi'] > rsi_config['overbought']
                df['rsi_oversold'] = df['rsi'] < rsi_config['oversold']
                
                # Berechne RSI-Momentum
                df['rsi_momentum'] = df['rsi'].diff()
                
                logger.debug("RSI erfolgreich hinzugefügt")
                
            except Exception as e:
                logger.error(f"Fehler bei RSI-Berechnung: {e}")
                # Setze Standardwerte im Fehlerfall
                df['rsi'] = 50
                df['rsi_overbought'] = False
                df['rsi_oversold'] = False
                df['rsi_momentum'] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen des RSI: {e}")
            return df
    
    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt Moving Average Convergence Divergence (MACD) hinzu.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            DataFrame mit hinzugefügtem MACD
        """
        try:
            if df.empty or 'close' not in df.columns:
                return df
            
            close_series = df['close'].copy()
            close_series = self._safe_fillna(close_series)
            
            config = self.config.get('MACD', {'fast': 12, 'slow': 26, 'signal': 9})
            
            # Berechne Fast EMA
            fast_ema = pta.ema(close_series, length=config['fast'])
            fast_ema = self._safe_fillna(fast_ema, close_series.mean())
            
            # Berechne Slow EMA
            slow_ema = pta.ema(close_series, length=config['slow'])
            slow_ema = self._safe_fillna(slow_ema, close_series.mean())
            
            # MACD Line
            df['macd'] = fast_ema - slow_ema
            df['macd'] = self._safe_fillna(df['macd'], 0)
            
            # Signal Line
            signal_line = pta.ema(df['macd'], length=config['signal'])
            df['macd_signal'] = self._safe_fillna(signal_line, 0)
            
            # MACD Histogram
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # MACD Crossover Signals
            macd_shift = self._safe_fillna(df['macd'].shift(1), 0)
            signal_shift = self._safe_fillna(df['macd_signal'].shift(1), 0)
            
            df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (macd_shift <= signal_shift)
            df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (macd_shift >= signal_shift)
            
            return df
        except Exception as e:
            logger.error(f"Fehler beim Berechnen des MACD: {e}")
            return df
    
    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt Bollinger Bands hinzu.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            DataFrame mit hinzugefügten Bollinger Bands
        """
        try:
            if df.empty or 'close' not in df.columns:
                return df
            
            close_series = df['close'].copy()
            close_series = self._safe_fillna(close_series)
            
            bbands_config = self.config.get('BBANDS', {'period': 20, 'std_dev': 2.0})
            
            sma = close_series.rolling(window=bbands_config['period']).mean()
            std = close_series.rolling(window=bbands_config['period']).std()
            
            df['bb_upper'] = sma + (std * bbands_config['std_dev'])
            df['bb_middle'] = sma
            df['bb_lower'] = sma - (std * bbands_config['std_dev'])
            
            # Fill NaN values
            bb_columns = ['bb_upper', 'bb_middle', 'bb_lower']
            for col in bb_columns:
                df[col] = self._safe_fillna(df[col], close_series.mean())
            
            # Calculate additional BB indicators
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_width'] = self._safe_fillna(df['bb_width'], 0)
            
            df['bb_pct_b'] = (close_series - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_pct_b'] = self._safe_fillna(df['bb_pct_b'], 0.5)
            
            # Calculate crosses using safe shift values
            upper_shift = self._safe_fillna(df['bb_upper'].shift(1), df['bb_upper'])
            lower_shift = self._safe_fillna(df['bb_lower'].shift(1), df['bb_lower'])
            close_shift = self._safe_fillna(close_series.shift(1), close_series)
            
            df['bb_upper_cross'] = (close_series > df['bb_upper']) & (close_shift <= upper_shift)
            df['bb_lower_cross'] = (close_series < df['bb_lower']) & (close_shift >= lower_shift)
            
            return df
        except Exception as e:
            logger.error(f"Fehler beim Berechnen der Bollinger Bands: {e}")
            return df
    
    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt Average True Range (ATR) hinzu.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            DataFrame mit hinzugefügtem ATR
        """
        try:
            if df.empty:
                return df
            
            required_columns = ['high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                return df
            
            # Fill NaN values in required columns
            for col in required_columns:
                df[col] = self._safe_fillna(df[col])
            
            atr_config = self.config.get('ATR', {'period': 14})
            
            # Calculate True Range
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - df['close'].shift(1)).abs()
            tr3 = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            df['atr'] = tr.rolling(window=atr_config['period']).mean()
            df['atr'] = self._safe_fillna(df['atr'])
            
            # Additional ATR indicators
            df['atr_pct'] = (df['atr'] / df['close'] * 100)
            df['atr_pct'] = self._safe_fillna(df['atr_pct'])
            
            df['atr_increasing'] = df['atr'].diff() > 0
            
            atr_pct_ma = df['atr_pct'].rolling(window=atr_config['period']).mean()
            df['atr_high'] = df['atr_pct'] > self._safe_fillna(atr_pct_ma)
            
            return df
        except Exception as e:
            logger.error(f"Fehler beim Berechnen des ATR: {e}")
            return df
    
    def add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt Stochastic Oscillator hinzu.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            DataFrame mit hinzugefügtem Stochastic Oscillator
        """
        try:
            if df.empty:
                return df
            
            required_columns = ['high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                return df
            
            # Fill NaN values in required columns
            for col in required_columns:
                df[col] = self._safe_fillna(df[col])
            
            stoch_config = self.config.get('STOCH', {
                'k_period': 14,
                'd_period': 3,
                'slowing': 3,
                'overbought': 80,
                'oversold': 20
            })
            
            # Calculate %K
            low_min = df['low'].rolling(window=stoch_config['k_period']).min()
            high_max = df['high'].rolling(window=stoch_config['k_period']).max()
            
            k_raw = 100 * (df['close'] - low_min) / (high_max - low_min)
            k = k_raw.rolling(window=stoch_config['slowing']).mean() if stoch_config['slowing'] > 1 else k_raw
            
            # Calculate %D
            d = k.rolling(window=stoch_config['d_period']).mean()
            
            # Add to DataFrame with safe filling
            df['stoch_k'] = self._safe_fillna(k, 50)
            df['stoch_d'] = self._safe_fillna(d, 50)
            
            # Add signals using safe shift values
            k_shift = self._safe_fillna(df['stoch_k'].shift(1), 50)
            d_shift = self._safe_fillna(df['stoch_d'].shift(1), 50)
            
            df['stoch_overbought'] = df['stoch_k'] > stoch_config['overbought']
            df['stoch_oversold'] = df['stoch_k'] < stoch_config['oversold']
            df['stoch_cross_up'] = (df['stoch_k'] > df['stoch_d']) & (k_shift <= d_shift)
            df['stoch_cross_down'] = (df['stoch_k'] < df['stoch_d']) & (k_shift >= d_shift)
            
            return df
        except Exception as e:
            logger.error(f"Fehler beim Berechnen des Stochastic Oscillator: {e}")
            return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt volumenbasierte Indikatoren hinzu.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            DataFrame mit hinzugefügten Volumen-Indikatoren
        """
        try:
            # On-Balance Volume (OBV) mit pandas_ta
            df['obv'] = pta.obv(close=df['close'], volume=df['volume'])
            
            # Volume Moving Average mit pandas_ta
            df['volume_sma'] = pta.sma(df['volume'], length=20)
            
            # Volume Ratio (aktuelles Volumen / durchschnittliches Volumen)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            logger.debug("Volumen-Indikatoren hinzugefügt")
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Berechnen der Volumen-Indikatoren: {e}")
            return df
    
    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt Volatilitätsberechnungen hinzu.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            DataFrame mit hinzugefügten Volatilitätsberechnungen
        """
        try:
            # Berechne prozentuale Preisänderungen
            df['returns'] = df['close'].pct_change()
            
            # Volatilität (Standardabweichung der Returns)
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(20)
            
            # True Range in Prozent mit pandas_ta - robuster gegen None-Werte
            tr = pta.true_range(df['high'], df['low'], df['close'])
            # Sicherstellen, dass NaN-Werte durch 0 ersetzt werden
            tr = tr.fillna(0)
            # Verwende einen einfacheren Ansatz für die Berechnung
            df['tr_pct'] = 100 * tr / df['close'].replace({0: float('nan')}).fillna(1)
            
            logger.debug("Volatilitätsberechnungen hinzugefügt")
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Berechnen der Volatilität: {e}")
            return df
