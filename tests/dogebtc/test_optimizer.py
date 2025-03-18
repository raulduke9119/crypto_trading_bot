#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test-Skript für den Strategy Optimizer.
Testet die Strategie-Optimierung mit Binance und Yahoo Finance als Datenquellen.
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

from strategy_optimizer import StrategyOptimizer
from config.config import BINANCE_CONFIG

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_optimizer')

def test_optimizer_with_data_source(data_source: str = "binance"):
    """
    Testet den Strategy Optimizer mit der angegebenen Datenquelle.
    
    Args:
        data_source: Datenquelle ("binance" oder "yahoo")
    """
    logger.info(f"Teste Strategy Optimizer mit Datenquelle: {data_source}")
    
    # Zeitraum für den Test (letzten 90 Tage)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    
    # Strategy Optimizer initialisieren
    optimizer = StrategyOptimizer(
        symbol="DOGEBTC",
        timeframe="1d",
        start_date=start_date,
        end_date=end_date,
        initial_capital=1.0,
        commission_rate=0.001,
        data_source=data_source
    )
    
    try:
        # Historische Daten laden
        logger.info(f"Lade historische Daten von {data_source}...")
        df = optimizer.load_historical_data()
        
        if df is not None and not df.empty:
            logger.info(f"Daten erfolgreich geladen: {len(df)} Einträge")
            logger.info(f"Erster Eintrag: {df.index[0]}")
            logger.info(f"Letzter Eintrag: {df.index[-1]}")
            logger.info(f"Spalten: {df.columns.tolist()}")
            
            # Einfache Parameter für den Test
            test_params = {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'bb_std': 2.0,
                'stoch_k': 14,
                'stoch_d': 3,
                'stoch_smooth': 3,
                'atr_period': 14
            }
            
            # Indikatoren hinzufügen
            logger.info("Füge technische Indikatoren hinzu...")
            df_with_indicators = optimizer._add_indicators_with_params(df, test_params)
            
            if df_with_indicators is not None:
                logger.info(f"Indikatoren erfolgreich hinzugefügt. Neue Spalten: {[col for col in df_with_indicators.columns if col not in df.columns]}")
                return True
            else:
                logger.error("Fehler beim Hinzufügen der Indikatoren")
                return False
        else:
            logger.error("Keine Daten geladen oder Daten sind leer")
            return False
    
    except Exception as e:
        logger.error(f"Fehler beim Testen des Optimizers mit {data_source}: {e}")
        return False

def main():
    """
    Hauptfunktion zum Testen des Strategy Optimizers mit beiden Datenquellen.
    """
    # Teste mit Binance
    binance_result = test_optimizer_with_data_source("binance")
    logger.info(f"Test mit Binance: {'Erfolgreich' if binance_result else 'Fehlgeschlagen'}")
    
    # Teste mit Yahoo Finance
    yahoo_result = test_optimizer_with_data_source("yahoo")
    logger.info(f"Test mit Yahoo Finance: {'Erfolgreich' if yahoo_result else 'Fehlgeschlagen'}")
    
    # Zusammenfassung
    if binance_result and yahoo_result:
        logger.info("Beide Datenquellen funktionieren korrekt!")
    elif binance_result:
        logger.info("Nur Binance funktioniert korrekt.")
    elif yahoo_result:
        logger.info("Nur Yahoo Finance funktioniert korrekt.")
    else:
        logger.error("Beide Datenquellen sind fehlgeschlagen.")

if __name__ == "__main__":
    main()
