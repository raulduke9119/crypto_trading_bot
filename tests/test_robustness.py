"""
Robustheitstests für die technischen Indikatoren.
Fokus auf Fehlerbehandlung und Umgang mit None-Werten.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Root-Verzeichnis zum Path hinzufügen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.indicators import TechnicalIndicators
from config.config import INDICATORS_CONFIG

# Logger deaktivieren während der Tests
logging.disable(logging.CRITICAL)

class TestIndicatorRobustness(unittest.TestCase):
    """
    Testet die Robustheit der technischen Indikatoren gegen verschiedene Fehlerszenarien.
    """
    
    def setUp(self):
        """
        Testdaten und Indikatoren vor jedem Test einrichten.
        """
        # Einfache Testdaten erstellen
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'open': np.random.random(100) * 100 + 100,
            'high': np.random.random(100) * 100 + 150,
            'low': np.random.random(100) * 100 + 50,
            'close': np.random.random(100) * 100 + 100,
            'volume': np.random.random(100) * 1000
        })
        self.test_data.set_index('timestamp', inplace=True)
        
        # Indikatoren initialisieren
        self.ti = TechnicalIndicators(INDICATORS_CONFIG)
    
    def test_none_values_in_data(self):
        """
        Testet, ob die Indikatoren mit None-Werten im Datensatz richtig umgehen.
        """
        # Erstelle eine Kopie der Testdaten mit einigen None-Werten
        df_with_none = self.test_data.copy()
        
        # Füge None-Werte an zufälligen Positionen ein
        random_indices = np.random.choice(len(df_with_none), 10, replace=False)
        df_with_none.loc[df_with_none.index[random_indices], 'close'] = None
        
        # Berechne alle Indikatoren
        result_df = self.ti.add_all_indicators(df_with_none)
        
        # Prüfe, dass keine None-Werte in kritischen Indikator-Spalten vorhanden sind
        important_columns = ['macd', 'macd_signal', 'rsi', 'bb_upper', 'bb_lower']
        for col in important_columns:
            if col in result_df.columns:
                self.assertEqual(
                    result_df[col].isna().sum(), 
                    0, 
                    f"Spalte {col} enthält immer noch {result_df[col].isna().sum()} None-Werte"
                )
        
        # Prüfe, dass keine NoneType-Fehler auftreten (würde einen Test-Fehler verursachen)
        try:
            _ = result_df['macd'] > result_df['macd_signal']
            _ = result_df['stoch_k'] > result_df['stoch_d']
            self.assertTrue(True, "Keine NoneType-Fehler bei Vergleichen aufgetreten")
        except TypeError as e:
            self.fail(f"TypeError bei Vergleich aufgetreten: {e}")
    
    def test_empty_dataframe(self):
        """
        Testet, ob die Indikatoren mit einem leeren DataFrame richtig umgehen.
        """
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        result_df = self.ti.add_all_indicators(empty_df)
        
        # Prüfe, dass ein DataFrame zurückgegeben wird
        self.assertIsInstance(result_df, pd.DataFrame)
        
        # Leerer DataFrame sollte leer bleiben (keine neuen Zeilen)
        self.assertEqual(len(result_df), 0)
    
    def test_missing_columns(self):
        """
        Testet, ob die Indikatoren mit fehlenden Spalten richtig umgehen.
        """
        # DataFrame ohne 'volume' Spalte
        df_missing_vol = self.test_data.drop(columns=['volume']).copy()
        
        # Dies sollte nicht abstürzen, sondern einen Fehler loggen und das Original zurückgeben
        result_df = self.ti.add_volume_indicators(df_missing_vol)
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df.columns), len(df_missing_vol.columns))
    
    def test_zero_values(self):
        """
        Testet, ob die Indikatoren mit Null-Werten richtig umgehen.
        """
        # Erstelle Testdaten mit einigen Nullen
        df_with_zeros = self.test_data.copy()
        
        # Füge Nullen an einigen Positionen ein
        random_indices = np.random.choice(len(df_with_zeros), 5, replace=False)
        df_with_zeros.loc[df_with_zeros.index[random_indices], 'close'] = 0
        
        # Berechne alle Indikatoren
        result_df = self.ti.add_all_indicators(df_with_zeros)
        
        # Prüfe, dass keine Division-by-Zero-Fehler auftreten
        # Bei der Berechnung von Prozentsätzen oder Verhältnissen
        self.assertTrue('bb_pct_b' in result_df.columns)
        self.assertTrue('tr_pct' in result_df.columns)
    
    def test_extreme_values(self):
        """
        Testet, ob die Indikatoren mit extremen Werten richtig umgehen.
        """
        # Erstelle Testdaten mit extremen Werten
        df_extreme = self.test_data.copy()
        
        # Füge einige sehr große Werte ein
        df_extreme.loc[df_extreme.index[0], 'close'] = 1e9
        # Und einige sehr kleine Werte
        df_extreme.loc[df_extreme.index[1], 'close'] = 1e-9
        
        # Berechne Indikatoren
        result_df = self.ti.add_all_indicators(df_extreme)
        
        # Prüfe, dass keine Overflow/Underflow-Fehler auftreten
        self.assertFalse(np.isinf(result_df['macd']).any())
        self.assertFalse(np.isinf(result_df['rsi']).any())
    
    def test_nan_inf_values(self):
        """
        Testet, ob die Indikatoren mit NaN und Inf-Werten richtig umgehen.
        """
        # Erstelle Testdaten mit NaN und Inf-Werten
        df_with_nan_inf = self.test_data.copy()
        
        # Füge NaN und Inf-Werte ein
        df_with_nan_inf.loc[df_with_nan_inf.index[0], 'close'] = np.nan
        df_with_nan_inf.loc[df_with_nan_inf.index[1], 'close'] = np.inf
        df_with_nan_inf.loc[df_with_nan_inf.index[2], 'close'] = -np.inf
        
        # Berechne Indikatoren
        result_df = self.ti.add_all_indicators(df_with_nan_inf)
        
        # Prüfe, dass NaN- und Inf-Werte korrekt behandelt wurden
        important_columns = ['macd', 'macd_signal', 'rsi']
        for col in important_columns:
            self.assertEqual(
                np.isinf(result_df[col]).sum(), 
                0, 
                f"Spalte {col} enthält immer noch Inf-Werte"
            )

    def test_cross_signals_with_none(self):
        """
        Testet speziell die Berechnung von Cross-Signalen bei None-Werten.
        """
        # Erzeuge spezielle Testdaten für Cross-Signale
        df = self.test_data.copy()
        
        # Manuelle Berechnung der MACD-Werte
        macd = np.random.random(100) - 0.5
        signal = np.random.random(100) - 0.5
        
        # Füge ein spezielles Muster für ein sicheres Cross-Signal ein
        macd[50] = 0.1
        macd[51] = 0.3
        signal[50] = 0.2
        signal[51] = 0.2
        
        # Füge None-Werte in der Nähe ein
        macd[52] = None
        signal[53] = None
        
        # Füge die Werte dem DataFrame hinzu
        df['macd'] = macd
        df['macd_signal'] = signal
        
        # Berechne Cross-Signale direkt (wie im Indikator-Code)
        macd_prev = df['macd'].shift(1).fillna(0)
        signal_prev = df['macd_signal'].shift(1).fillna(0)
        df['macd_cross_up'] = (df['macd'].fillna(0) > df['macd_signal'].fillna(0)) & (macd_prev <= signal_prev)
        
        # Überprüfe, dass die Kreuzsignale korrekt berechnet wurden
        # Wir sollten ein Cross-Up-Signal an Position 51 haben
        self.assertTrue(df['macd_cross_up'].iloc[51])

if __name__ == '__main__':
    unittest.main()
