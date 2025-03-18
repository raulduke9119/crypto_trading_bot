"""
Test-Script für die DogebtcHFStrategy (High-Frequency-Strategie).
Führt einen Backtest durch und analysiert die Ergebnisse.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any

# Eigene Module importieren
from config.config import BINANCE_CONFIG, DATA_DIRECTORY, LOG_DIRECTORY
from data.data_collector import DataCollector
from data.indicators import TechnicalIndicators
from strategies.dogebtc_hf_strategy import DogebtcHFStrategy
from trading_bot import TradingBot
from binance.client import Client

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIRECTORY, 'dogebtc_hf_test.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('dogebtc_hf_test')

def backtest_dogebtc_hf_strategy(days_back: int = 30, timeframe: str = "5m", 
                               initial_capital: float = 1.0, 
                               risk_percentage: float = 1.0,
                               plot_results: bool = True) -> Dict[str, Any]:
    """
    Führt einen Backtest für die DOGEBTC HF-Strategie durch.
    
    Args:
        days_back: Anzahl der Tage für den Backtest
        timeframe: Zeitintervall für die Analyse
        initial_capital: Anfangskapital in BTC
        risk_percentage: Risikoprozentsatz pro Trade
        plot_results: Ob Ergebnisse geplottet werden sollen
        
    Returns:
        Dict mit den Backtest-Ergebnissen
    """
    logger.info(f"Starte Backtest für DOGEBTC HF-Strategie mit {timeframe} Intervall über {days_back} Tage")
    
    # Zeitraum berechnen
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Binance-Client initialisieren
    client = Client(BINANCE_CONFIG.get('api_key', ''), BINANCE_CONFIG.get('api_secret', ''))
    
    # Daten laden
    data_collector = DataCollector(client)
    data = data_collector.fetch_historical_data("DOGEBTC", timeframe, start_date, end_date)
    
    if data.empty:
        logger.error("Keine Daten für DOGEBTC gefunden!")
        return {"error": "Keine Daten verfügbar"}
    
    logger.info(f"{len(data)} Datenpunkte für DOGEBTC geladen, Zeitraum: {data.index[0]} bis {data.index[-1]}")
    
    # DOGEBTC HF-Strategie initialisieren
    strategy = DogebtcHFStrategy(
        risk_percent=risk_percentage,
        max_positions=1,
        use_ml_predictions=False,
        adjust_for_volatility=True,
        trailing_stop_percent=1.5,
        max_drawdown_percent=5.0
    )
    
    # Indikatoren hinzufügen und Signale generieren
    ti = TechnicalIndicators()
    
    # Grundlegende Indikatoren
    data = ti.add_all_indicators(data)
    
    # ATR für die Volatilitätsanpassung hinzufügen
    if 'atr' not in data.columns:
        data = ti.add_atr(data, period=14)
        data['atr_pct'] = (data['atr'] / data['close']) * 100
    
    # Signale mit unserer Strategie generieren
    data = strategy.generate_signals(data)
    
    # Backtest durchführen
    results = run_backtest(
        data=data,
        strategy=strategy,
        initial_capital=initial_capital
    )
    
    # Ergebnisse anzeigen
    print_backtest_results(results)
    
    # Optional: Ergebnisse plotten
    if plot_results:
        plot_backtest_results(data, results)
    
    return results

def run_backtest(data: pd.DataFrame, strategy: DogebtcHFStrategy, 
                initial_capital: float = 1.0) -> Dict[str, Any]:
    """
    Führt den eigentlichen Backtest durch.
    
    Args:
        data: DataFrame mit Marktdaten und Signalen
        strategy: Die zu testende Strategie
        initial_capital: Anfangskapital
        
    Returns:
        Dict mit Backtest-Ergebnissen
    """
    # Initialisierung
    equity = [initial_capital]
    positions = []
    trades = []
    current_position = None
    
    # Durch alle Datenpunkte iterieren
    for i in range(1, len(data)):
        current_data = data.iloc[i-1:i+1]
        current_price = current_data['close'].iloc[-1]
        
        # Position aktualisieren, falls vorhanden
        if current_position:
            current_position = strategy.update_position(current_position, current_data)
            
            # Verkaufssignal auswerten
            should_sell, signal_strength = strategy.should_sell(current_data, current_position)
            
            if should_sell:
                # Position schließen (Verkauf)
                sell_price = current_price
                profit_pct = ((sell_price / current_position['price']) - 1) * 100
                profit = current_position['amount'] * profit_pct / 100
                
                # Kapital aktualisieren
                equity.append(equity[-1] + profit)
                
                # Trade dokumentieren
                trade = {
                    'entry_date': current_position['date'],
                    'exit_date': data.index[i],
                    'entry_price': current_position['price'],
                    'exit_price': sell_price,
                    'profit_pct': profit_pct,
                    'profit': profit,
                    'signal_strength': signal_strength
                }
                trades.append(trade)
                
                logger.info(f"VERKAUF: {data.index[i]}, Preis: {sell_price:.8f}, "
                           f"Gewinn: {profit_pct:.2f}%, Signal: {signal_strength:.1f}")
                
                # Position zurücksetzen
                current_position = None
            else:
                # Position halten, Equity aktualisieren
                unrealized_profit_pct = ((current_price / current_position['price']) - 1) * 100
                unrealized_profit = current_position['amount'] * unrealized_profit_pct / 100
                equity.append(equity[-1] + unrealized_profit - equity[-2] + equity[-3])
        else:
            # Kein Position, prüfen ob kaufen
            should_buy, signal_strength = strategy.should_buy(current_data, None)
            
            if should_buy:
                # Position eröffnen (Kauf)
                position_size = strategy.calculate_position_size(current_data, equity[-1])
                
                # Neue Position
                current_position = {
                    'date': data.index[i],
                    'price': current_price,
                    'amount': position_size,
                    'highest_price': current_price
                }
                positions.append(current_position)
                
                logger.info(f"KAUF: {data.index[i]}, Preis: {current_price:.8f}, "
                          f"Betrag: {position_size:.8f}, Signal: {signal_strength:.1f}")
                
                # Equity bleibt unverändert beim Kauf
                equity.append(equity[-1])
            else:
                # Keine Aktion, Equity bleibt gleich
                equity.append(equity[-1])
    
    # Ergebnisse zusammenstellen
    equity_series = pd.Series(equity, index=data.index)
    returns = equity_series.pct_change().dropna()
    
    # Metriken berechnen
    total_return_pct = ((equity[-1] / initial_capital) - 1) * 100
    successful_trades = sum(1 for t in trades if t['profit_pct'] > 0)
    win_rate = (successful_trades / len(trades)) * 100 if trades else 0
    
    # Drawdown berechnen
    rollmax = equity_series.cummax()
    drawdown = (equity_series / rollmax - 1) * 100
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio (annualisiert, angenommen: tägliche Renditen)
    risk_free_rate = 0.02  # 2% p.a.
    samples_per_year = 365 * 24 * 12 if timeframe == "5m" else 365  # Für 5-Minuten-Daten
    sharpe_ratio = ((returns.mean() * samples_per_year) - (risk_free_rate / samples_per_year)) / (returns.std() * np.sqrt(samples_per_year)) if returns.std() > 0 else 0
    
    return {
        'equity': equity_series,
        'trades': trades,
        'total_return_pct': total_return_pct,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': len(trades)
    }

def print_backtest_results(results: Dict[str, Any]) -> None:
    """
    Gibt die Ergebnisse des Backtests aus.
    
    Args:
        results: Ergebnisse des Backtests
    """
    logger.info("=" * 50)
    logger.info("DOGEBTC HF-STRATEGIE BACKTEST ERGEBNISSE")
    logger.info("=" * 50)
    logger.info(f"Gesamtrendite: {results['total_return_pct']:.2f}%")
    logger.info(f"Anzahl Trades: {results['num_trades']}")
    logger.info(f"Gewinnrate: {results['win_rate']:.2f}%")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info("=" * 50)
    
    if results['trades']:
        avg_profit = sum(t['profit_pct'] for t in results['trades']) / len(results['trades'])
        avg_win = sum(t['profit_pct'] for t in results['trades'] if t['profit_pct'] > 0) / sum(1 for t in results['trades'] if t['profit_pct'] > 0) if sum(1 for t in results['trades'] if t['profit_pct'] > 0) > 0 else 0
        avg_loss = sum(t['profit_pct'] for t in results['trades'] if t['profit_pct'] <= 0) / sum(1 for t in results['trades'] if t['profit_pct'] <= 0) if sum(1 for t in results['trades'] if t['profit_pct'] <= 0) > 0 else 0
        
        logger.info(f"Durchschnittl. Gewinn pro Trade: {avg_profit:.2f}%")
        logger.info(f"Durchschnittl. gewinnender Trade: {avg_win:.2f}%")
        logger.info(f"Durchschnittl. verlierender Trade: {avg_loss:.2f}%")
        
        if avg_loss != 0:
            risk_reward = abs(avg_win / avg_loss)
            logger.info(f"Risk-Reward-Ratio: {risk_reward:.2f}")
    
    logger.info("=" * 50)

def plot_backtest_results(data: pd.DataFrame, results: Dict[str, Any]) -> None:
    """
    Plottet die Ergebnisse des Backtests.
    
    Args:
        data: DataFrame mit Marktdaten
        results: Ergebnisse des Backtests
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Preis und Signale
    ax1.set_title("DOGEBTC Preisverlauf mit Kauf- und Verkaufssignalen")
    ax1.plot(data.index, data['close'], label='DOGEBTC Preis', alpha=0.7)
    
    # Kauf- und Verkaufssignale einzeichnen
    buy_signals = data[data['buy_signal'] == True]
    sell_signals = data[data['sell_signal'] == True]
    
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Kaufsignal')
    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Verkaufssignal')
    
    # Tatsächliche Trades einzeichnen
    for trade in results['trades']:
        ax1.plot([trade['entry_date'], trade['exit_date']], 
                [trade['entry_price'], trade['exit_price']], 
                color='blue' if trade['profit_pct'] > 0 else 'orange', 
                linewidth=2)
    
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Equity Curve
    ax2.set_title("Equity Curve")
    ax2.plot(results['equity'].index, results['equity'], label='Equity', color='blue')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # Speichere Plot als Datei
    plot_file = os.path.join(LOG_DIRECTORY, 'dogebtc_hf_backtest_plot.png')
    plt.savefig(plot_file)
    logger.info(f"Plot gespeichert unter: {plot_file}")
    
    # Schließe Plot
    plt.close()

if __name__ == "__main__":
    # Parameter für Backtest
    DAYS_BACK = 30  # Daten von 30 Tagen
    TIMEFRAME = "5m"  # 5-Minuten-Intervall
    INITIAL_CAPITAL = 1.0  # 1 BTC
    RISK_PERCENTAGE = 1.0  # 1% Risiko pro Trade
    
    # Führe Backtest durch
    results = backtest_dogebtc_hf_strategy(
        days_back=DAYS_BACK,
        timeframe=TIMEFRAME,
        initial_capital=INITIAL_CAPITAL,
        risk_percentage=RISK_PERCENTAGE,
        plot_results=True
    )
    
    # Optional: Verschiedene Parameter testen
    logger.info("\nTeste verschiedene Parameter-Kombinationen...")
    
    risk_percentages = [0.5, 1.0, 1.5]
    timeframes = ["5m", "15m", "30m"]
    
    best_result = None
    best_return = -float('inf')
    
    for rp in risk_percentages:
        for tf in timeframes:
            logger.info(f"\nBacktest mit: Risiko={rp}%, Timeframe={tf}")
            result = backtest_dogebtc_hf_strategy(
                days_back=DAYS_BACK,
                timeframe=tf,
                initial_capital=INITIAL_CAPITAL,
                risk_percentage=rp,
                plot_results=False
            )
            
            if result['total_return_pct'] > best_return:
                best_return = result['total_return_pct']
                best_result = {
                    'risk_percentage': rp,
                    'timeframe': tf,
                    'return': best_return,
                    'win_rate': result['win_rate'],
                    'trades': result['num_trades']
                }
    
    if best_result:
        logger.info("\n" + "=" * 50)
        logger.info("BESTE PARAMETER-KOMBINATION:")
        logger.info("=" * 50)
        logger.info(f"Risiko: {best_result['risk_percentage']}%")
        logger.info(f"Timeframe: {best_result['timeframe']}")
        logger.info(f"Rendite: {best_result['return']:.2f}%")
        logger.info(f"Gewinnrate: {best_result['win_rate']:.2f}%")
        logger.info(f"Anzahl Trades: {best_result['trades']}")
        logger.info("=" * 50)
