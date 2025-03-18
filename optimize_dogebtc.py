"""
DOGEBTC Optimizer - Findet optimale Parameter für DOGEBTC-Trading.
Fokus auf Bollinger Bands, Stochastic Oscillator und ATR.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Any

# Eigene Module importieren
from config.config import BINANCE_CONFIG, DATA_DIRECTORY, LOG_DIRECTORY
from data.data_collector import DataCollector
from data.indicators import TechnicalIndicators
from strategy_optimizer import StrategyOptimizer

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIRECTORY, 'dogebtc_optimizer.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('dogebtc_optimizer')

def optimize_for_timeframe(timeframe: str, days_back: int = 90):
    """
    Optimiert die Strategie für einen bestimmten Zeitrahmen.
    
    Args:
        timeframe: Zeitrahmen für die Analyse ('1h', '4h', '1d', etc.)
        days_back: Anzahl der Tage für den Backtest
    
    Returns:
        Dict mit den besten Parametern und Ergebnissen
    """
    # Berechne Start- und Enddatum
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    logger.info(f"Optimiere DOGEBTC für Zeitrahmen {timeframe} von {start_date} bis {end_date}")
    
    # Parameter-Grid für DOGEBTC mit Fokus auf Bollinger Bands, Stochastic und ATR
    # Optimierte Parameter basierend auf Recherche und Backtesting
    param_grid = {
        # Bollinger Bands Parameter - Optimale Werte für DOGEBTC
        'bb_period': [20, 22],  # Standard: 20
        'bb_std': [2.0, 2.2, 2.5],  # Standard: 2.0
        'bb_weight': [1.0, 1.2, 1.5],
        
        # Stochastic Parameter - Optimale Werte für DOGEBTC
        'stoch_k': [14],  # Standard: 14
        'stoch_d': [3, 5],  # Standard: 3
        'stoch_smooth': [3],
        'stoch_weight': [1.0, 1.2, 1.5],
        
        # ATR für Volatilitätsanpassung - Optimale Werte für DOGEBTC
        'atr_period': [14],  # Standard: 14
        'atr_multiplier': [2.0, 2.2, 2.5],
        
        # RSI und MACD mit festen Werten (da Fokus auf BB, Stoch, ATR)
        'rsi_period': [14],
        'rsi_oversold': [30],
        'rsi_overbought': [70],
        'rsi_weight': [1.0],
        
        'macd_fast': [12],
        'macd_slow': [26],
        'macd_signal': [9],
        'macd_weight': [1.0],
        
        # Risikomanagement - Verschiedene Risikoniveaus testen
        'risk_per_trade': [0.01, 0.015, 0.02]
    }
    
    # Initialisiere Optimizer
    optimizer = StrategyOptimizer(
        symbol="DOGEBTC",
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=1.0,  # 1 BTC
        commission_rate=0.001,  # 0.1% Binance Standardgebühr
        data_source="binance"
    )
    
    # Führe Optimierung durch
    # Wir reduzieren die Anzahl der Parameter-Kombinationen durch ein kleineres Grid
    results = optimizer.optimize(param_grid)
    
    # Plotte Ergebnisse
    optimizer.plot_results(top_n=5)
    # Hinweis: Die plot_results-Methode unterstützt keinen save_path-Parameter
    # Die Ergebnisse werden in der Konsole angezeigt
    
    # Speichere die besten Parameter
    best_params = results[0]['params']
    best_results = {
        'timeframe': timeframe,
        'params': best_params,
        'total_return': results[0]['total_return'],
        'max_drawdown': results[0]['max_drawdown'],
        'sharpe_ratio': results[0]['sharpe_ratio'],
        'win_ratio': results[0]['win_ratio'],
        'num_trades': results[0]['num_trades']
    }
    
    # Speichere die Ergebnisse in einer JSON-Datei
    results_file = os.path.join(LOG_DIRECTORY, f'dogebtc_best_params_{timeframe}.json')
    with open(results_file, 'w') as f:
        json.dump(best_results, f, indent=4)
    
    logger.info(f"Beste Parameter für {timeframe} in {results_file} gespeichert")
    
    return best_results

def run_backtest_with_best_params(best_params: Dict[str, Any], timeframe: str, days: int = 30):
    """
    Führt einen Backtest mit den besten gefundenen Parametern durch.
    
    Args:
        best_params: Die besten Parameter aus der Optimierung
        timeframe: Zeitrahmen für den Backtest
        days: Anzahl der Tage für den Backtest
    """
    # Extrahiere Parameter für den Backtest
    params_str = " ".join([f"--{k}={v}" for k, v in best_params['params'].items()])
    
    # Berechne Start- und Enddatum
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Führe Backtest aus
    logger.info(f"Führe Backtest mit besten Parametern für {timeframe} durch")
    
    # Hier würde normalerweise der Backtest-Befehl ausgeführt werden
    backtest_cmd = f"python run_bot.py --mode backtest --symbols DOGEBTC --interval {timeframe} --start-date {start_date} --end-date {end_date} --initial-balance 1000 {params_str}"
    
    logger.info(f"Backtest-Befehl: {backtest_cmd}")
    
    # In einer realen Implementierung würde hier der Befehl ausgeführt werden
    # os.system(backtest_cmd)
    
    return backtest_cmd

if __name__ == "__main__":
    # Zeitrahmen, die wir optimieren wollen
    # Wir testen verschiedene Zeitrahmen, um die besten Parameter zu finden
    timeframes = ['1h', '4h', '1d']
    
    # Anzahl der Tage für den Backtest
    days_back = 90  # 90 Tage für ausreichend Daten
    
    # Kommandozeilenargumente verarbeiten
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] in ['1h', '4h', '1d', '1w']:
            timeframes = [sys.argv[1]]
            logger.info(f"Optimiere nur für Zeitrahmen {timeframes[0]}")
        
        if len(sys.argv) > 2 and sys.argv[2].isdigit():
            days_back = int(sys.argv[2])
            logger.info(f"Verwende {days_back} Tage für den Backtest")
    
    # Speichere die besten Parameter für jeden Zeitrahmen
    best_results = {}
    
    print(f"=== DOGEBTC Optimierung ===")
    print(f"Zeitrahmen: {', '.join(timeframes)}")
    print(f"Zeitraum: {days_back} Tage")
    print(f"Optimierte Parameter: Bollinger Bands, Stochastic Oscillator, ATR")
    print("=" * 50)
    
    for tf in timeframes:
        try:
            # Optimiere für jeden Zeitrahmen
            print(f"\nOptimiere für Zeitrahmen {tf}...")
            result = optimize_for_timeframe(tf, days_back)
            
            if result:
                best_results[tf] = result
                
                # Zeige die besten Parameter an
                print(f"\nBeste Parameter für DOGEBTC ({tf}):")
                for param, value in result['params'].items():
                    print(f"{param}: {value}")
                
                print(f"\nGesamtrendite: {result['total_return']:.2f}%")
                print(f"Maximaler Drawdown: {result['max_drawdown']:.2f}%")
                print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                print(f"Win Ratio: {result['win_ratio']:.2f}%")
                print(f"Anzahl Trades: {result['num_trades']}")
                
                # Generiere Backtest-Befehl
                backtest_cmd = run_backtest_with_best_params(result, tf)
                print(f"\nBacktest-Befehl für manuelle Ausführung:")
                print(backtest_cmd)
            else:
                print(f"Keine Ergebnisse für Zeitrahmen {tf}")
            
        except Exception as e:
            logger.error(f"Fehler bei der Optimierung für {tf}: {e}")
            print(f"Fehler bei der Optimierung für {tf}: {e}")
    
    # Vergleiche die Ergebnisse der verschiedenen Zeitrahmen
    if best_results:
        print("\n=== Vergleich der Zeitrahmen ===")
        for tf, result in best_results.items():
            print(f"{tf}: Rendite={result['total_return']:.2f}%, Sharpe={result['sharpe_ratio']:.2f}, Win={result['win_ratio']:.2f}%")
        
        # Finde den besten Zeitrahmen basierend auf Sharpe Ratio
        best_tf = max(best_results.items(), key=lambda x: x[1]['sharpe_ratio'])[0]
        print(f"\nBester Zeitrahmen basierend auf Sharpe Ratio: {best_tf}")
        
        # Speichere alle Ergebnisse in einer JSON-Datei
        all_results_file = os.path.join(LOG_DIRECTORY, 'dogebtc_all_results.json')
        with open(all_results_file, 'w') as f:
            json.dump(best_results, f, indent=4)
        print(f"\nAlle Ergebnisse wurden in {all_results_file} gespeichert")
    else:
        print("\nKeine erfolgreichen Optimierungen durchgeführt")
