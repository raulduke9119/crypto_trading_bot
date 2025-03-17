"""
Unit-Tests für das Technical Indicators Modul.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np

# Füge Root-Verzeichnis zum Pfad hinzu, um relative Imports zu ermöglichen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.indicators import TechnicalIndicators
from config.config import INDICATORS_CONFIG

class TestTechnicalIndicators(unittest.TestCase):
    """
    Testklasse für die TechnicalIndicators-Klasse.
    """
    
    def setUp(self):
        """
        Testdaten und Indicator-Instanz vorbereiten.
        """
        # Erstelle Testdaten (ein DataFrame mit OHLCV-Daten)
        self.test_data = pd.DataFrame({
            'open': [100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                    112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
                    122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 127.0, 126.0, 125.0],
            'high': [103.0, 105.0, 107.0, 106.0, 108.0, 110.0, 111.0, 112.0, 113.0, 114.0,
                    115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0,
                    125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 129.0, 128.0, 127.0],
            'low':  [98.0,  100.0, 102.0, 101.0, 103.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                    110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
                    120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 124.0, 123.0, 122.0],
            'close': [101.0, 104.0, 105.0, 102.0, 107.0, 109.0, 110.0, 111.0, 112.0, 113.0,
                    114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0,
                    124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 125.0, 124.0, 123.0],
            'volume': [1000, 1200, 1300, 1100, 1400, 1500, 1600, 1700, 1800, 1900,
                     2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
                     3000, 3100, 3200, 3300, 3400, 3500, 3600, 3000, 2900, 2800]
        })
        
        # Setze einen Index, um Zeilen leichter identifizieren zu können
        self.test_data.index = pd.date_range(start='2023-01-01', periods=len(self.test_data), freq='D')
        
        # Initialisiere TechnicalIndicators mit der Standard-Konfiguration
        self.indicators = TechnicalIndicators(config=INDICATORS_CONFIG)
    
    def test_moving_averages(self):
        """
        Teste die Berechnung von Moving Averages.
        """
        # Füge Moving Averages hinzu
        df_with_ma = self.indicators.add_moving_averages(self.test_data.copy())
        
        # Überprüfe, ob die Moving Averages korrekt hinzugefügt wurden
        self.assertIn('sma_short', df_with_ma.columns)
        self.assertIn('ema_short', df_with_ma.columns)
        
        # None-Vergleiche vermeiden - überprüfe explizit auf NaN
        # Im Falle einer SMA mit einer Periode von z.B. 10 sollten die ersten 9 Werte NaN sein
        first_non_nan_index = INDICATORS_CONFIG['SMA']['short'] - 1
        
        # Überprüfe, ob die NaN-Werte korrekt gesetzt wurden
        self.assertTrue(df_with_ma['sma_short'].iloc[:first_non_nan_index].isna().all())
        
        # Überprüfe, ob der SMA-Wert nach den NaN-Werten berechnet wurde
        self.assertFalse(pd.isna(df_with_ma['sma_short'].iloc[first_non_nan_index + 5]))
    
    def test_rsi(self):
        """
        Teste die Berechnung des RSI.
        """
        # Füge RSI hinzu
        df_with_rsi = self.indicators.add_rsi(self.test_data.copy())
        
        # Überprüfe, ob RSI korrekt hinzugefügt wurde
        self.assertIn('rsi', df_with_rsi.columns)
        
        # Überprüfe die Grenzen des RSI (0-100)
        non_nan_rsi = df_with_rsi['rsi'].dropna()
        if len(non_nan_rsi) > 0:  # Stelle sicher, dass es nicht-NaN RSI-Werte gibt
            self.assertTrue(all(0 <= val <= 100 for val in non_nan_rsi))
    
    def test_macd(self):
        """
        Teste die Berechnung des MACD.
        """
        # Füge nur MACD hinzu, um das Problem zu isolieren
        try:
            df_with_macd = self.indicators.add_macd(self.test_data.copy())
            
            # Überprüfe, ob die MACD-Spalten vorhanden sind
            macd_columns = ['macd', 'macd_signal', 'macd_hist']
            for col in macd_columns:
                self.assertIn(col, df_with_macd.columns)
                # Überprüfe auf None-Werte
                self.assertFalse(df_with_macd[col].isna().any(), f"MACD-Spalte {col} enthält NaN-Werte")
                
            print("MACD wurde erfolgreich berechnet!")
            return df_with_macd
            
        except Exception as e:
            self.fail(f"Fehler beim Berechnen des MACD: {e}")
    
    def test_all_indicators(self):
        """
        Teste das Hinzufügen aller Indikatoren.
        """
        # Teste zuerst den MACD separat
        self.test_macd()
        
        # Jetzt füge alle Indikatoren hinzu
        try:
            df_with_indicators = self.indicators.add_all_indicators(self.test_data.copy())
            
            # Überprüfe, ob mindestens ein paar der erwarteten Spalten vorhanden sind
            expected_columns = ['sma_short', 'rsi', 'macd', 'bb_upper', 'atr']
            for col in expected_columns:
                self.assertIn(col, df_with_indicators.columns, f"Spalte {col} fehlt im DataFrame")
            
            # Teste, ob keine NaN-Werte im resultierenden DataFrame sind
            # (add_all_indicators sollte dropna() aufrufen)
            self.assertFalse(df_with_indicators.isnull().any().any(), "DataFrame enthält NaN-Werte")
            
            # Erfolgreicher Test
            print("Alle Indikatoren wurden erfolgreich hinzugefügt!")
            
        except Exception as e:
            self.fail(f"Fehler beim Hinzufügen aller Indikatoren: {e}")

if __name__ == '__main__':
    unittest.main()
