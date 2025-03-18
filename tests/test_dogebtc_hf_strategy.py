"""
Unit-Tests für die DOGEBTC High-Frequency Trading Strategie.
Testet alle wichtigen Komponenten der Strategie.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Füge Projekt-Root zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.dogebtc_hf_strategy import DogebtcHFStrategy
from data.indicators import TechnicalIndicators
from utils.logger import setup_logger

class TestDogebtcHFStrategy(unittest.TestCase):
    """Test-Suite für die DOGEBTC HF-Strategie."""
    
    @classmethod
    def setUpClass(cls):
        """Einmalige Setup-Routine für die gesamte Test-Klasse."""
        # Erstelle Test-DataFrame
        dates = pd.date_range(start='2025-01-01', periods=100, freq='5min')
        cls.test_data = pd.DataFrame({
            'open': np.random.normal(100, 10, 100),
            'high': np.random.normal(105, 10, 100),
            'low': np.random.normal(95, 10, 100),
            'close': np.random.normal(100, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)
        
        # Stelle sicher, dass high/low konsistent sind
        cls.test_data['high'] = cls.test_data[['open', 'close']].max(axis=1) + abs(np.random.normal(2, 0.5, 100))
        cls.test_data['low'] = cls.test_data[['open', 'close']].min(axis=1) - abs(np.random.normal(2, 0.5, 100))
        
        # Initialisiere Strategie
        cls.strategy = DogebtcHFStrategy(
            risk_percent=1.0,
            max_positions=2,
            use_ml_predictions=False,
            adjust_for_volatility=True
        )
        
        # Füge Indikatoren hinzu
        ti = TechnicalIndicators()
        cls.test_data = ti.add_all_indicators(cls.test_data)
    
    def test_strategy_initialization(self):
        """Testet die korrekte Initialisierung der Strategie."""
        self.assertEqual(self.strategy.risk_percent, 1.0)
        self.assertEqual(self.strategy.max_positions, 2)
        self.assertEqual(self.strategy.rsi_period, 7)
        self.assertEqual(self.strategy.macd_fast, 6)
        self.assertEqual(self.strategy.macd_slow, 22)
    
    def test_signal_generation(self):
        """Testet die Signalgenerierung der Strategie."""
        # Generiere Signale
        result_df = self.strategy.generate_signals(self.test_data.copy())
        
        # Prüfe, ob alle erwarteten Spalten vorhanden sind
        required_columns = ['buy_signal', 'sell_signal', 'buy_strength', 'sell_strength']
        for col in required_columns:
            self.assertIn(col, result_df.columns)
        
        # Prüfe Signaltypen
        self.assertTrue(result_df['buy_signal'].dtype == bool)
        self.assertTrue(result_df['sell_signal'].dtype == bool)
        
        # Prüfe, dass nicht gleichzeitig Kauf- und Verkaufssignal
        self.assertTrue(not any(result_df['buy_signal'] & result_df['sell_signal']))
    
    def test_position_size_calculation(self):
        """Testet die Berechnung der Positionsgrößen."""
        # Test mit verschiedenen Kapitalbeträgen
        test_capitals = [0.1, 1.0, 10.0]
        
        for capital in test_capitals:
            position_size = self.strategy.calculate_position_size(
                self.test_data.iloc[-10:],
                capital
            )
            
            # Position sollte nie größer als Kapital sein
            self.assertLessEqual(position_size, capital)
            
            # Position sollte positiv sein
            self.assertGreater(position_size, 0)
    
    def test_should_buy_conditions(self):
        """Testet die Kaufsignal-Bedingungen."""
        # Generiere Signale
        data = self.strategy.generate_signals(self.test_data.copy())
        
        # Teste verschiedene Szenarien
        for i in range(len(data)-1):
            current_data = data.iloc[i:i+2]
            should_buy, strength = self.strategy.should_buy(current_data, None)
            
            # Prüfe Rückgabetypen
            self.assertIsInstance(should_buy, bool)
            self.assertIsInstance(strength, float)
            
            # Prüfe Signalstärke-Bereich
            self.assertGreaterEqual(strength, 0.0)
            self.assertLessEqual(strength, 10.0)
    
    def test_should_sell_conditions(self):
        """Testet die Verkaufssignal-Bedingungen."""
        # Erstelle Test-Position
        test_position = {
            'price': 100.0,
            'amount': 1.0,
            'highest_price': 105.0
        }
        
        # Generiere Signale
        data = self.strategy.generate_signals(self.test_data.copy())
        
        # Teste verschiedene Szenarien
        for i in range(len(data)-1):
            current_data = data.iloc[i:i+2]
            should_sell, strength = self.strategy.should_sell(current_data, test_position)
            
            # Prüfe Rückgabetypen
            self.assertIsInstance(should_sell, bool)
            self.assertIsInstance(strength, float)
            
            # Prüfe Signalstärke-Bereich
            self.assertGreaterEqual(strength, 0.0)
            self.assertLessEqual(strength, 10.0)
    
    def test_position_update(self):
        """Testet die Aktualisierung von Positionen."""
        # Erstelle Test-Position
        initial_position = {
            'price': 100.0,
            'amount': 1.0,
            'highest_price': 100.0
        }
        
        # Teste mit steigenden Preisen
        rising_data = self.test_data.copy()
        rising_data['close'] = 110.0
        
        updated_position = self.strategy.update_position(
            initial_position.copy(),
            rising_data.iloc[-2:]
        )
        
        # Prüfe, ob highest_price aktualisiert wurde
        self.assertEqual(updated_position['highest_price'], 110.0)
        
        # Teste mit fallenden Preisen
        falling_data = self.test_data.copy()
        falling_data['close'] = 90.0
        
        updated_position = self.strategy.update_position(
            initial_position.copy(),
            falling_data.iloc[-2:]
        )
        
        # Prüfe, ob highest_price nicht reduziert wurde
        self.assertEqual(updated_position['highest_price'], 100.0)
    
    def test_risk_management(self):
        """Testet die Risikomanagement-Funktionen."""
        # Teste Trailing-Stop
        position = {
            'price': 100.0,
            'amount': 1.0,
            'highest_price': 110.0
        }
        
        # Erstelle Daten mit Preis unter Trailing-Stop
        data = self.test_data.copy()
        data['close'] = 107.0  # Mehr als 1.5% unter Höchstpreis
        
        # Sollte Verkaufssignal generieren
        should_sell, strength = self.strategy.should_sell(data.iloc[-2:], position)
        self.assertTrue(should_sell)
        self.assertGreater(strength, 7.0)  # Sollte hohe Signalstärke haben
        
        # Teste Max-Drawdown
        position = {
            'price': 100.0,
            'amount': 1.0,
            'highest_price': 100.0
        }
        
        # Erstelle Daten mit Preis unter Max-Drawdown
        data = self.test_data.copy()
        data['close'] = 94.0  # Mehr als 5% unter Einstiegspreis
        
        # Sollte Verkaufssignal generieren
        should_sell, strength = self.strategy.should_sell(data.iloc[-2:], position)
        self.assertTrue(should_sell)
        self.assertGreater(strength, 8.0)  # Sollte sehr hohe Signalstärke haben

if __name__ == '__main__':
    unittest.main()
