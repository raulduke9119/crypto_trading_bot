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
                    df[f'sma_{period_name}'] = sma.fillna(method='ffill').fillna(close_series.mean())
                except Exception as e:
                    logger.error(f"Fehler bei SMA-{period_name} Berechnung: {e}")
            
            # Exponential Moving Averages mit Validierung
            for period_name, length in ema_config.items():
                try:
                    ema = pta.ema(close_series, length=length)
                    if ema is None or ema.isnull().all():
                        logger.warning(f"EMA-{period_name} konnte nicht berechnet werden, verwende SMA")
                        ema = pta.sma(close_series, length=length)
                    df[f'ema_{period_name}'] = ema.fillna(method='ffill').fillna(close_series.mean())
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
                rsi = rsi.fillna(method='ffill').fillna(50)  # 50 als neutraler Wert
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
            # Validiere Eingabedaten
            if df.empty:
                logger.warning("DataFrame ist leer, überspringe MACD-Berechnung")
                return df
            
            if 'close' not in df.columns:
                logger.error("'close' Spalte nicht im DataFrame gefunden")
                return df
            
            # Konfigurationswerte mit Standardwerten
            config = self.config.get('MACD', {})
            fast_length = config.get('fast', 12)
            slow_length = config.get('slow', 26)
            signal_length = config.get('signal', 9)
            
            # Stelle sicher, dass das Close nicht None enthält
            close_series = df['close'].copy()
            if close_series.isnull().any():
                logger.warning("Null-Werte in close-Daten gefunden, fülle mit forward-fill auf")
                close_series = close_series.ffill().fillna(0)
            
            # Berechne EMAs mit expliziter Null-Wert-Behandlung
            if len(close_series) < max(fast_length, slow_length):
                logger.warning(f"Zu wenig Datenpunkte für EMA-Berechnung: {len(close_series)} < {max(fast_length, slow_length)}")
                return df

            # Berechne Fast EMA
            fast_ema = pta.ema(close_series, length=fast_length)
            if fast_ema is None or fast_ema.isnull().all():
                logger.warning("Fast EMA konnte nicht berechnet werden, verwende SMA")
                fast_ema = pta.sma(close_series, length=fast_length)
                if fast_ema is None or fast_ema.isnull().all():
                    logger.error("Auch SMA konnte nicht berechnet werden")
                    return df
            fast_ema = fast_ema.fillna(method='ffill').fillna(method='bfill').fillna(close_series.mean())
            
            # Berechne Slow EMA
            slow_ema = pta.ema(close_series, length=slow_length)
            if slow_ema is None or slow_ema.isnull().all():
                logger.warning("Slow EMA konnte nicht berechnet werden, verwende SMA")
                slow_ema = pta.sma(close_series, length=slow_length)
                if slow_ema is None or slow_ema.isnull().all():
                    logger.error("Auch SMA konnte nicht berechnet werden")
                    return df
            slow_ema = slow_ema.fillna(method='ffill').fillna(method='bfill').fillna(close_series.mean())
            
            # MACD-Linie berechnen
            df['macd'] = fast_ema - slow_ema
            
            # Signal-Linie berechnen
            macd_series = df['macd'].fillna(method='ffill').fillna(0)
            signal_line = pta.ema(macd_series, length=signal_length)
            if signal_line is None or signal_line.isnull().all():
                logger.error("Signal-Linie konnte nicht berechnet werden")
                return df
            df['macd_signal'] = signal_line.fillna(method='ffill').fillna(0)
            
            # Histogramm berechnen
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Signale berechnen mit sicherer Null-Wert-Behandlung
            df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & \
                                 (df['macd'].shift(1).fillna(0) <= df['macd_signal'].shift(1).fillna(0))
            df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & \
                                   (df['macd'].shift(1).fillna(0) >= df['macd_signal'].shift(1).fillna(0))
            
            # Überprüfe finale Datenqualität
            macd_columns = ['macd', 'macd_signal', 'macd_hist']
            nan_counts = {col: df[col].isnull().sum() for col in macd_columns}
            if any(nan_counts.values()):
                logger.warning(f"NaN-Werte in MACD-Indikatoren gefunden: {nan_counts}")
                # Fülle verbleibende NaN-Werte
                df[macd_columns] = df[macd_columns].fillna(0)
            
            logger.debug("MACD erfolgreich hinzugefügt")
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Berechnen des MACD: {e}")
            # Stelle sicher, dass alle MACD-Spalten existieren
            for col in ['macd', 'macd_signal', 'macd_hist', 'macd_cross_up', 'macd_cross_down']:
                if col not in df.columns:
                    df[col] = 0
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
            # Validiere Eingabedaten
            if df.empty:
                logger.warning("DataFrame ist leer, überspringe Bollinger Bands")
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
            bbands_config = self.config.get('BBANDS', {'period': 20, 'std_dev': 2.0})
            period = bbands_config.get('period', 20)
            std_dev = bbands_config.get('std_dev', 2.0)
            
            # Prüfe, ob genügend Datenpunkte vorhanden sind
            if len(close_series) < period:
                logger.warning(f"Zu wenig Datenpunkte für Bollinger Bands: {len(close_series)} < {period}")
                return df
            
            # Berechne Bollinger Bands manuell, um mehr Kontrolle zu haben
            sma = close_series.rolling(window=period).mean()
            std = close_series.rolling(window=period).std()
            
            # Füge Bollinger Bands zum DataFrame hinzu
            df['bb_upper'] = sma + (std * std_dev)
            df['bb_middle'] = sma
            df['bb_lower'] = sma - (std * std_dev)
            
            # Berechne Bollinger Band Width und %B mit Null-Wert-Prüfung
            df['bb_width'] = df.apply(
                lambda row: (row['bb_upper'] - row['bb_lower']) / row['bb_middle'] 
                if pd.notnull(row['bb_middle']) and row['bb_middle'] != 0 else 0, 
                axis=1
            )
            
            # Berechne %B mit sicherer Division
            df['bb_pct_b'] = df.apply(
                lambda row: (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower']) 
                if pd.notnull(row['bb_upper']) and pd.notnull(row['bb_lower']) and row['bb_upper'] != row['bb_lower'] 
                else 0.5, 
                axis=1
            )
            
            # Füge Bollinger Band Signale hinzu
            df['bb_upper_cross'] = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
            df['bb_lower_cross'] = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
            
            # Fülle verbleibende NaN-Werte
            bb_columns = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct_b']
            df[bb_columns] = df[bb_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            logger.debug("Bollinger Bands hinzugefügt")
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Berechnen der Bollinger Bands: {e}")
            # Stelle sicher, dass alle BB-Spalten existieren
            for col in ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct_b', 'bb_upper_cross', 'bb_lower_cross']:
                if col not in df.columns:
                    df[col] = 0
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
            # Validiere Eingabedaten
            if df.empty:
                logger.warning("DataFrame ist leer, überspringe ATR-Berechnung")
                return df
            
            required_columns = ['high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"'{col}' Spalte nicht im DataFrame gefunden")
                    return df
            
            # Stelle sicher, dass keine None-Werte in den Eingabedaten sind
            for col in required_columns:
                if df[col].isnull().any():
                    logger.warning(f"Null-Werte in {col}-Daten gefunden, führe Bereinigung durch")
                    df[col] = df[col].ffill().bfill()
            
            # Konfigurationswerte mit Fehlerbehandlung
            atr_config = self.config.get('ATR', {'period': 14})
            period = atr_config.get('period', 14)
            
            # Prüfe, ob genügend Datenpunkte vorhanden sind
            if len(df) < period:
                logger.warning(f"Zu wenig Datenpunkte für ATR: {len(df)} < {period}")
                return df
            
            # Berechne True Range manuell
            high = df['high']
            low = df['low']
            close = df['close']
            prev_close = close.shift(1)
            
            # Berechne die drei Komponenten des True Range
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            
            # True Range ist das Maximum der drei Komponenten
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Berechne ATR als gleitenden Durchschnitt des True Range
            df['atr'] = tr.rolling(window=period).mean()
            
            # Berechne ATR in Prozent des Preises mit Null-Wert-Prüfung
            df['atr_pct'] = df.apply(
                lambda row: (row['atr'] / row['close'] * 100) 
                if pd.notnull(row['atr']) and pd.notnull(row['close']) and row['close'] != 0 
                else 0, 
                axis=1
            )
            
            # Füge ATR-basierte Signale hinzu
            df['atr_increasing'] = df['atr'].diff() > 0
            df['atr_high'] = df['atr_pct'] > df['atr_pct'].rolling(window=period).mean()
            
            # Fülle verbleibende NaN-Werte
            atr_columns = ['atr', 'atr_pct', 'atr_increasing', 'atr_high']
            df[atr_columns] = df[atr_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            logger.debug("ATR hinzugefügt")
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Berechnen des ATR: {e}")
            # Stelle sicher, dass alle ATR-Spalten existieren
            for col in ['atr', 'atr_pct', 'atr_increasing', 'atr_high']:
                if col not in df.columns:
                    df[col] = 0
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
            # Validiere Eingabedaten
            if df.empty:
                logger.warning("DataFrame ist leer, überspringe Stochastic-Berechnung")
                return df
            
            required_columns = ['high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"'{col}' Spalte nicht im DataFrame gefunden")
                    return df
            
            # Stelle sicher, dass keine None-Werte in den Eingabedaten sind
            for col in required_columns:
                if df[col].isnull().any():
                    logger.warning(f"Null-Werte in {col}-Daten gefunden, führe Bereinigung durch")
                    df[col] = df[col].ffill().bfill()
            
            # Konfigurationswerte mit Fehlerbehandlung
            stoch_config = self.config.get('STOCH', {'k_period': 14, 'd_period': 3, 'slowing': 3})
            k_period = stoch_config.get('k_period', 14)
            d_period = stoch_config.get('d_period', 3)
            slowing = stoch_config.get('slowing', 3)
            overbought = stoch_config.get('overbought', 80)
            oversold = stoch_config.get('oversold', 20)
            
            # Prüfe, ob genügend Datenpunkte vorhanden sind
            if len(df) < k_period:
                logger.warning(f"Zu wenig Datenpunkte für Stochastic: {len(df)} < {k_period}")
                return df
            
            # Berechne Stochastic Oscillator manuell
            # Berechne %K
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            
            # Sichere Division mit Null-Wert-Prüfung
            k_raw = df.apply(
                lambda row: 100 * (row['close'] - low_min[row.name]) / (high_max[row.name] - low_min[row.name]) 
                if pd.notnull(high_max[row.name]) and pd.notnull(low_min[row.name]) and (high_max[row.name] - low_min[row.name]) != 0 
                else 50, 
                axis=1
            )
            
            # Berechne %K mit Glättung
            k = k_raw.rolling(window=slowing).mean() if slowing > 1 else k_raw
            
            # Berechne %D (Signal-Linie)
            d = k.rolling(window=d_period).mean()
            
            # Füge zum DataFrame hinzu
            df['stoch_k'] = k
            df['stoch_d'] = d
            
            # Füge Stochastic-Signale hinzu mit Null-Wert-Prüfung
            df['stoch_overbought'] = df['stoch_k'] > overbought
            df['stoch_oversold'] = df['stoch_k'] < oversold
            
            # Berechne Kreuzungen mit sicherer Null-Wert-Behandlung
            k_shift = df['stoch_k'].shift(1).fillna(50)
            d_shift = df['stoch_d'].shift(1).fillna(50)
            
            df['stoch_cross_up'] = (df['stoch_k'] > df['stoch_d']) & (k_shift <= d_shift)
            df['stoch_cross_down'] = (df['stoch_k'] < df['stoch_d']) & (k_shift >= d_shift)
            
            # Fülle verbleibende NaN-Werte
            stoch_columns = ['stoch_k', 'stoch_d', 'stoch_overbought', 'stoch_oversold', 'stoch_cross_up', 'stoch_cross_down']
            for col in stoch_columns:
                if col in df.columns:
                    if col in ['stoch_k', 'stoch_d']:
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(50)
                    elif col in ['stoch_overbought', 'stoch_oversold', 'stoch_cross_up', 'stoch_cross_down']:
                        df[col] = df[col].fillna(False)
            
            logger.debug("Stochastic Oscillator hinzugefügt")
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Berechnen des Stochastic Oscillator: {e}")
            # Stelle sicher, dass alle Stochastic-Spalten existieren
            for col in ['stoch_k', 'stoch_d', 'stoch_overbought', 'stoch_oversold', 'stoch_cross_up', 'stoch_cross_down']:
                if col not in df.columns:
                    if col in ['stoch_k', 'stoch_d']:
                        df[col] = 50
                    else:
                        df[col] = False
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
