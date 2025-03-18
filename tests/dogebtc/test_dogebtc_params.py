"""
DOGEBTC Parameter-Tester - Testet verschiedene optimale Parameter für DOGEBTC-Trading.
Basierend auf Recherchen und API-Dokumentation.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

# Eigene Module importieren
from config.config import BINANCE_CONFIG, DATA_DIRECTORY, LOG_DIRECTORY
from data.data_collector import DataCollector
from data.indicators import TechnicalIndicators
from trading_bot import TradingBot

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIRECTORY, 'dogebtc_params_test.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('dogebtc_params_test')

def test_parameter_set(params, symbol="DOGEBTC", start_date=None, end_date=None, initial_balance=1000):
    """
    Testet einen Parametersatz für DOGEBTC.
    
    Args:
        params: Dictionary mit Parametern
        symbol: Trading-Symbol
        start_date: Startdatum für den Test
        end_date: Enddatum für den Test
        initial_balance: Anfangskapital
        
    Returns:
        Dictionary mit Testergebnissen
    """
    # Standardwerte für Datum setzen, wenn nicht angegeben
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Parameter für den Backtest
    param_str = " ".join([f"--{k}={v}" for k, v in params.items()])
    
    # Backtest-Befehl erstellen
    backtest_cmd = f"python run_bot.py --mode backtest --symbols {symbol} --start-date {start_date} --end-date {end_date} --initial-balance {initial_balance} {param_str}"
    
    logger.info(f"Führe Backtest mit Parametern aus: {params}")
    logger.info(f"Befehl: {backtest_cmd}")
    
    # Backtest ausführen
    os.system(backtest_cmd)
    
    # Ergebnisse auslesen (in einer realen Implementierung würden wir die Ergebnisse aus der Ausgabe extrahieren)
    # Hier simulieren wir die Ergebnisse
    results = {
        'params': params,
        'command': backtest_cmd
    }
    
    return results

def run_parameter_tests():
    """
    Führt Tests mit verschiedenen Parametersätzen durch.
    """
    # Optimale Parameter basierend auf Recherche
    parameter_sets = [
        # Test 1: Standardparameter
        {
            'bb_period': 20,
            'bb_std': 2.0,
            'bb_weight': 1.0,
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3,
            'stoch_weight': 1.0,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'risk_per_trade': 0.01
        },
        # Test 2: Angepasste Bollinger Bands
        {
            'bb_period': 22,
            'bb_std': 2.5,
            'bb_weight': 1.5,
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3,
            'stoch_weight': 1.0,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'risk_per_trade': 0.01
        },
        # Test 3: Angepasster Stochastic Oscillator
        {
            'bb_period': 20,
            'bb_std': 2.0,
            'bb_weight': 1.0,
            'stoch_k': 14,
            'stoch_d': 5,
            'stoch_smooth': 3,
            'stoch_weight': 1.5,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'risk_per_trade': 0.01
        },
        # Test 4: Angepasster ATR
        {
            'bb_period': 20,
            'bb_std': 2.0,
            'bb_weight': 1.0,
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3,
            'stoch_weight': 1.0,
            'atr_period': 14,
            'atr_multiplier': 2.5,
            'risk_per_trade': 0.01
        },
        # Test 5: Kombinierte Anpassungen
        {
            'bb_period': 22,
            'bb_std': 2.2,
            'bb_weight': 1.2,
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3,
            'stoch_weight': 1.2,
            'atr_period': 14,
            'atr_multiplier': 2.2,
            'risk_per_trade': 0.015
        },
        # Test 6: Aggressivere Einstellungen
        {
            'bb_period': 20,
            'bb_std': 2.0,
            'bb_weight': 1.5,
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3,
            'stoch_weight': 1.5,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'risk_per_trade': 0.02
        },
        # Test 7: Optimierte Hochfrequenz-Handelsstrategie
        {
            'bb_period': 10,           # Kürzere Bollinger-Band-Periode für empfindlichere Signale
            'bb_std': 1.8,              # Engere Bollinger Bänder
            'bb_weight': 1.8,           # Höhere Gewichtung
            'stoch_k': 5,               # Sehr kurzer Stochastik für schnelle Signale
            'stoch_d': 2,               # Kurzer Stochastik-Glättungsfaktor
            'stoch_smooth': 2,          # Minimale Glättung für reaktionsschnellere Signale
            'stoch_weight': 2.0,        # Höhere Gewichtung für Stochastik
            'atr_period': 7,            # Kürzerer ATR-Zeitraum
            'atr_multiplier': 1.5,      # Weniger konservative Stops für schnellere Trades
            'risk_per_trade': 0.008,    # Reduziertes Risiko pro Trade, da mehr Trades
            'rsi_period': 2,            # Ultraschneller RSI
            'rsi_overbought': 80,       # RSI-Überkauft-Schwelle
            'rsi_oversold': 20,         # RSI-Überverkauft-Schwelle
            'use_high_frequency': True  # Aktiviere Hochfrequenz-Modus
        },
        # Test 8: Ultraschnelle Scalping-Strategie
        {
            'bb_period': 5,             # Extrem kurze Bollinger-Band-Periode
            'bb_std': 1.5,              # Noch engere Bollinger Bänder
            'bb_weight': 2.0,           # Maximale Gewichtung
            'stoch_k': 3,               # Extrem kurzer Stochastik
            'stoch_d': 2,               # Minimale Stochastik-Glättung
            'stoch_smooth': 1,          # Keine Glättung
            'stoch_weight': 2.5,        # Sehr hohe Gewichtung
            'atr_period': 5,            # Extrem kurzer ATR
            'atr_multiplier': 1.2,      # Enge Stops für schnelle Reaktion
            'risk_per_trade': 0.005,    # Minimales Risiko pro Trade
            'rsi_period': 1,            # 1-Perioden RSI (extrem reaktionsschnell)
            'rsi_overbought': 75,       # Angepasste RSI-Überkauft-Schwelle für mehr Signale
            'rsi_oversold': 25,         # Angepasste RSI-Überverkauft-Schwelle für mehr Signale
            'use_high_frequency': True,  # Aktiviere Hochfrequenz-Modus
            'signal_threshold': 10      # Niedrigere Schwelle für Signalgenerierung
        }
    ]
    
    # Zeitraum für Tests festlegen
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # Tests durchführen
    results = []
    for params in parameter_sets:
        result = test_parameter_set(
            params=params,
            symbol="DOGEBTC",
            start_date=start_date,
            end_date=end_date,
            initial_balance=1000
        )
        results.append(result)
        
    # Ergebnisse anzeigen
    logger.info("=== Testergebnisse ===")
    for i, result in enumerate(results):
        logger.info(f"Test {i+1}:")
        logger.info(f"Parameter: {result['params']}")
        logger.info(f"Befehl: {result['command']}")
        logger.info("-" * 50)
    
    return results

if __name__ == "__main__":
    logger.info("Starte DOGEBTC Parameter-Tests")
    results = run_parameter_tests()
    logger.info("Tests abgeschlossen")
