#!/usr/bin/env python
"""
Startskript für den Binance Trading Bot.
Implementiert einen sicheren Testmodus und Live/Backtest-Modi.
"""
import os
import sys
import time
import logging
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from trading_bot import TradingBot
from config.config import (
    API_KEY, API_SECRET, USE_TESTNET, TRADING_SYMBOLS,
    LOG_LEVEL, LOG_FILE
)

# Logging-Konfiguration
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parst Befehlszeilenargumente.
    """
    parser = argparse.ArgumentParser(description='Binance Trading Bot')
    
    # Allgemeine Parameter
    parser.add_argument('--mode', '-m', choices=['live', 'backtest'], default='backtest',
                        help='Betriebsmodus: live oder backtest')
    parser.add_argument('--symbols', '-s', nargs='+', default=TRADING_SYMBOLS,
                        help='Trading-Symbole (z.B. BTCUSDT ETHUSDT)')
    parser.add_argument('--interval', '-i', type=int, default=15,
                        help='Update-Intervall in Minuten (nur im Live-Modus)')
    
    # Backtest-Parameter
    parser.add_argument('--start-date', type=str, default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                        help='Startdatum für Backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Enddatum für Backtest (YYYY-MM-DD)')
    parser.add_argument('--initial-balance', type=float, default=1000.0,
                        help='Anfangskapital für Backtest')
    
    # ML-Parameter
    parser.add_argument('--use-ml', action='store_true',
                        help='ML-Vorhersagen verwenden')
    
    args = parser.parse_args()
    return args

def run_live_mode(args):
    """
    Startet den Bot im Live-Modus.
    """
    print(f"Starte Trading Bot im Live-Modus mit Symbolen: {args.symbols}")
    
    bot = TradingBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        use_testnet=USE_TESTNET,
        symbols=args.symbols,
        use_ml=args.use_ml
    )
    
    # Teste Verbindung
    if not bot.test_connection():
        print("Verbindungstest fehlgeschlagen. Bitte API-Schlüssel überprüfen.")
        return
    
    # Starte Bot
    bot.run(interval_minutes=args.interval)
    
def run_backtest_mode(args):
    """
    Startet den Bot im Backtest-Modus.
    """
    print(f"Starte Backtest von {args.start_date} bis {args.end_date or 'heute'}")
    print(f"Symbole: {args.symbols}, Anfangskapital: ${args.initial_balance}")
    
    bot = TradingBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        use_testnet=USE_TESTNET,  # Konfiguration aus config.py verwenden
        symbols=args.symbols,
        use_ml=args.use_ml
    )
    
    # Führe Backtest aus
    results = bot.backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance
    )
    
    # Zeige Ergebnisse
    print("\n=== Backtest-Ergebnisse ===")
    
    if results:
        print(f"Startkapital: ${args.initial_balance:.2f}")
        print(f"Endkapital: ${results['final_balance']:.2f}")
        print(f"Rendite: {results['return_percentage']:.2f}%")
        print(f"Anzahl Trades: {results['total_trades']}")
        print(f"Gewinnrate: {results['win_rate']:.2f}%")
        print(f"Maximaler Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
        # Equity-Kurve als ASCII-Art darstellen (vereinfacht)
        if results['equity_curve']:
            values = [e[1] for e in results['equity_curve']]
            min_val = min(values)
            max_val = max(values)
            
            if max_val > min_val:
                scale = 20 / (max_val - min_val)
                print("\nEquity-Kurve:")
                for value in values[::max(1, len(values) // 50)]:  # Reduziere auf ca. 50 Punkte
                    bars = int((value - min_val) * scale)
                    print(f"${value:.2f} " + "#" * bars)
    else:
        print("Keine Backtest-Ergebnisse verfügbar.")

def main():
    """
    Hauptfunktion.
    """
    # Parse Argumente
    args = parse_arguments()
    
    # Überprüfe API-Schlüssel
    if not API_KEY or not API_SECRET:
        print("Fehler: API-Schlüssel nicht gefunden. Bitte in config.py konfigurieren.")
        return
    
    # Starte im entsprechenden Modus
    if args.mode == 'live':
        run_live_mode(args)
    else:
        run_backtest_mode(args)

def run_test_mode(symbols: List[str] = ['BTCUSDT']) -> None:
    """Führt den Bot im Testmodus aus."""
    try:
        # Initialisiere Bot mit minimaler Konfiguration
        bot = TradingBot(
            api_key=API_KEY,
            api_secret=API_SECRET,
            use_testnet=True,  # Immer TestNet im Testmodus
            symbols=symbols,
            risk_percentage=1.0,  # Minimales Risiko
            max_positions=1,      # Nur eine Position
            use_ml=False         # ML deaktiviert für Test
        )
        
        # Teste Verbindung
        if not bot.test_connection():
            logging.error("Verbindungstest fehlgeschlagen!")
            return
            
        # Hole aktuelle Marktdaten
        market_data = bot.update_market_data(days_back=1)  # Nur 1 Tag
        if not market_data:
            logging.error("Keine Marktdaten verfügbar!")
            return
            
        logging.info("Testmodus erfolgreich durchgeführt!")
        
    except Exception as e:
        logging.error(f"Fehler im Testmodus: {e}")
        raise

if __name__ == "__main__":
    try:
        # Parse und verarbeite Argumente
        args = parse_arguments()
        
        # Führe den gewählten Modus aus
        if args.mode == "live":
            run_live_mode(args)
        else:
            run_backtest_mode(args)
            
    except KeyboardInterrupt:
        logger.info("\nProgramm durch Benutzer beendet.")
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}", exc_info=True)
        raise
