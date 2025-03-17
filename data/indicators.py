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
            # Bollinger Bands mit pandas_ta berechnen
            bbands = pta.bbands(
                df['close'], 
                length=self.config['BBANDS']['period'],
                std=self.config['BBANDS']['std_dev']
            )
            
            # Bollinger Bands-Spalten extrahieren und zum Haupt-DataFrame hinzufügen
            df['bb_upper'] = bbands['BBU_' + str(self.config['BBANDS']['period']) + '_' + str(self.config['BBANDS']['std_dev'])]
            df['bb_middle'] = bbands['BBM_' + str(self.config['BBANDS']['period']) + '_' + str(self.config['BBANDS']['std_dev'])]
            df['bb_lower'] = bbands['BBL_' + str(self.config['BBANDS']['period']) + '_' + str(self.config['BBANDS']['std_dev'])]
            
            # Berechne Bollinger Band Width und %B
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            # Prüfe, ob bb_upper und bb_lower identisch sind (Division durch Null vermeiden)
            df['bb_pct_b'] = df.apply(
                lambda row: (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower']) 
                if row['bb_upper'] != row['bb_lower'] else 0.5, axis=1
            )
            
            logger.debug("Bollinger Bands hinzugefügt")
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
            # ATR mit pandas_ta berechnen
            df['atr'] = pta.atr(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                length=self.config['ATR']['period']
            )
            
            # Berechne ATR in Prozent des Preises
            df['atr_pct'] = df['atr'] / df['close'] * 100
            
            logger.debug("ATR hinzugefügt")
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
            # Stochastic mit pandas_ta berechnen
            stoch = pta.stoch(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                k=self.config['STOCH']['k_period'],
                d=self.config['STOCH']['d_period'],
                smooth_k=self.config['STOCH']['slowing']
            )
            
            # Extrahiere die K- und D-Linien
            k_name = f"STOCHk_{self.config['STOCH']['k_period']}_{self.config['STOCH']['d_period']}_{self.config['STOCH']['slowing']}"
            d_name = f"STOCHd_{self.config['STOCH']['k_period']}_{self.config['STOCH']['d_period']}_{self.config['STOCH']['slowing']}"
            
            if k_name in stoch.columns and d_name in stoch.columns:
                df['stoch_k'] = stoch[k_name]
                df['stoch_d'] = stoch[d_name]
            else:
                # Fallback für den Fall, dass die Spaltennamen anders sind
                df['stoch_k'] = stoch.iloc[:, 0]
                df['stoch_d'] = stoch.iloc[:, 1]
            
            # Füge Stochastic-Signale hinzu
            df['stoch_cross_up'] = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
            df['stoch_cross_down'] = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
            
            logger.debug("Stochastic Oscillator hinzugefügt")
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
