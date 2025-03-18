#!/usr/bin/env python
"""
Demonstrationsprogramm für die technischen Indikatoren des Trading Bots.
Lädt echte Daten von Binance und berechnet alle Indikatoren.
"""
import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
import matplotlib.pyplot as plt

# Füge Root-Verzeichnis zum Pfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import BINANCE_CONFIG, INDICATORS_CONFIG
from data.indicators import TechnicalIndicators
from utils.logger import setup_logger

# Logger initialisieren - mit korrektem Log-Datei-Pfad
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'indicator_demo.log')
logger = setup_logger(log_file=log_file, log_level='INFO')

def get_binance_client():
    """
    Erstellt einen Binance Client mit den API-Schlüsseln aus der Konfiguration.
    """
    try:
        # Verwende Testnet, falls konfiguriert
        api_key = BINANCE_CONFIG.get('API_KEY', '')
        api_secret = BINANCE_CONFIG.get('API_SECRET', '')
        testnet = BINANCE_CONFIG.get('USE_TESTNET', True)
        
        if testnet:
            client = Client(api_key, api_secret, testnet=True)
            logger.info("Verbindung zum Binance Testnet hergestellt")
        else:
            client = Client(api_key, api_secret)
            logger.info("Verbindung zum Binance Mainnet hergestellt")
            
        return client
    except Exception as e:
        logger.error(f"Fehler beim Verbinden mit Binance: {e}")
        return None

def get_historical_data(client, symbol='BTCUSDT', interval='1h', limit=500):
    """
    Lädt historische Kline/Candlestick-Daten von Binance.
    
    Args:
        client: Binance Client
        symbol: Handelspaar (z.B. 'BTCUSDT')
        interval: Zeitintervall ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
        limit: Anzahl der Datenpunkte (max. 1000)
        
    Returns:
        DataFrame mit OHLCV-Daten
    """
    try:
        # Lade Kline-Daten
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        # Konvertiere zu DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Konvertiere Datentypen
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Setze Timestamp als Index
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"{len(df)} Datenpunkte für {symbol} mit Intervall {interval} geladen")
        return df
    except Exception as e:
        logger.error(f"Fehler beim Laden der historischen Daten: {e}")
        return None

def plot_indicators(df, symbol, interval):
    """
    Plottet die wichtigsten Indikatoren für visuelle Analyse.
    
    Args:
        df: DataFrame mit Preis- und Indikatordaten
        symbol: Handelspaar-Symbol
        interval: Zeitintervall
    """
    # Aktualisierte Stilanpassung für neuere Matplotlib-Versionen
    try:
        # Versuche zuerst den aktuellen Stil zu setzen
        import seaborn as sns
        sns.set_style('darkgrid')
    except (ImportError, AttributeError):
        # Fallback für ältere Versionen
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            # Allgemeiner Fallback
            plt.style.use('default')
    
    # Erstelle Subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    fig.suptitle(f'Trading Bot Indikatoren - {symbol} ({interval})', fontsize=16)
    
    # Plot 1: Preischart mit SMA, EMA und Bollinger Bands
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Schlusskurs', color='black', alpha=0.75)
    ax1.plot(df.index, df['sma_short'], label=f'SMA ({INDICATORS_CONFIG["SMA"]["short"]})', color='blue', alpha=0.6)
    ax1.plot(df.index, df['ema_short'], label=f'EMA ({INDICATORS_CONFIG["EMA"]["short"]})', color='red', alpha=0.6)
    ax1.plot(df.index, df['bb_upper'], label='BB Upper', color='green', alpha=0.3)
    ax1.plot(df.index, df['bb_middle'], label='BB Middle', color='green', alpha=0.5)
    ax1.plot(df.index, df['bb_lower'], label='BB Lower', color='green', alpha=0.3)
    ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], color='green', alpha=0.05)
    ax1.set_title('Preis mit Moving Averages und Bollinger Bands')
    ax1.set_ylabel('Preis')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Plot 2: MACD
    ax2 = axes[1]
    ax2.plot(df.index, df['macd'], label='MACD', color='blue')
    ax2.plot(df.index, df['macd_signal'], label='Signal', color='red')
    ax2.bar(df.index, df['macd_hist'], label='Histogramm', color='green', alpha=0.5, width=0.01)
    # Markiere Cross-Punkte
    ax2.scatter(df.index[df['macd_cross_up']], df.loc[df['macd_cross_up'], 'macd'], 
                color='green', marker='^', s=100, label='Bullish Cross')
    ax2.scatter(df.index[df['macd_cross_down']], df.loc[df['macd_cross_down'], 'macd'], 
                color='red', marker='v', s=100, label='Bearish Cross')
    ax2.set_title('MACD')
    ax2.set_ylabel('Wert')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # Plot 3: RSI
    ax3 = axes[2]
    ax3.plot(df.index, df['rsi'], label='RSI', color='purple')
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    ax3.set_title('Relative Strength Index (RSI)')
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    # Plot 4: Volumen und ATR
    ax4 = axes[3]
    ax4.bar(df.index, df['volume'], label='Volumen', color='blue', alpha=0.3, width=0.01)
    ax4_2 = ax4.twinx()
    ax4_2.plot(df.index, df['atr'], label='ATR', color='red')
    ax4.set_title('Volumen und Average True Range (ATR)')
    ax4.set_ylabel('Volumen')
    ax4_2.set_ylabel('ATR')
    ax4.legend(loc='upper left')
    ax4_2.legend(loc='upper right')
    ax4.grid(True)
    
    # X-Achsen-Formatierung
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Zeige Plot an
    plt.show()

def save_to_csv(df, symbol, interval):
    """
    Speichert die Daten mit Indikatoren in eine CSV-Datei.
    
    Args:
        df: DataFrame mit Preis- und Indikatordaten
        symbol: Handelspaar-Symbol
        interval: Zeitintervall
    """
    # Erstelle Ordner, falls nicht vorhanden
    os.makedirs('output', exist_ok=True)
    
    # Erzeuge Dateinamen
    filename = f"output/{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Speichere als CSV
    df.to_csv(filename)
    logger.info(f"Daten in {filename} gespeichert")
    return filename

def main():
    """
    Hauptfunktion der Demo.
    """
    # Verbinde mit Binance
    client = get_binance_client()
    if not client:
        logger.error("Konnte nicht mit Binance verbinden. Bitte API-Schlüssel überprüfen.")
        return
    
    # Benutzereingaben für Symbol und Intervall
    print("\n=== Trading Bot Indikatoren Demo ===")
    symbol = input("Bitte gib das Handelspaar ein (z.B. BTCUSDT, ETHUSDT): ").upper() or "BTCUSDT"
    interval_options = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    
    print("Verfügbare Zeitintervalle:")
    for i, option in enumerate(interval_options):
        print(f"{i+1}. {option}")
    
    interval_choice = input(f"Wähle ein Zeitintervall (1-{len(interval_options)}): ")
    try:
        interval_idx = int(interval_choice) - 1
        interval = interval_options[interval_idx]
    except (ValueError, IndexError):
        interval = '1h'
        print(f"Ungültige Eingabe, verwende Standardintervall: {interval}")
    
    limit = input("Anzahl der zu ladenden Datenpunkte (max. 1000): ")
    try:
        limit = int(limit)
        if limit < 1 or limit > 1000:
            limit = 500
            print(f"Ungültige Eingabe, verwende Standardlimit: {limit}")
    except ValueError:
        limit = 500
        print(f"Ungültige Eingabe, verwende Standardlimit: {limit}")
    
    # Lade historische Daten
    print(f"\nLade historische Daten für {symbol} mit Intervall {interval}...")
    df = get_historical_data(client, symbol, interval, limit)
    if df is None or len(df) == 0:
        logger.error("Keine Daten gefunden. Bitte überprüfe das Symbol und Intervall.")
        return
    
    # Initialisiere Technical Indicators
    ti = TechnicalIndicators(INDICATORS_CONFIG)
    
    # Berechne alle Indikatoren
    print("Berechne technische Indikatoren...")
    df_with_indicators = ti.add_all_indicators(df)
    
    # Zeige Ergebnisse an
    print(f"\nErfolg! {len(df_with_indicators)} Datenpunkte mit {len(df_with_indicators.columns)} Indikatoren berechnet.")
    print("\nVerfügbare Indikatoren:")
    
    # Gruppiere Spalten nach Indikatortyp für bessere Übersichtlichkeit
    column_groups = {
        'Preis': ['open', 'high', 'low', 'close', 'volume'],
        'Moving Averages': [col for col in df_with_indicators.columns if 'sma' in col or 'ema' in col],
        'Oscillatoren': [col for col in df_with_indicators.columns if 'rsi' in col or 'stoch' in col],
        'MACD': [col for col in df_with_indicators.columns if 'macd' in col],
        'Bollinger Bands': [col for col in df_with_indicators.columns if 'bb_' in col],
        'Volatilität': [col for col in df_with_indicators.columns if 'atr' in col or 'volatility' in col or 'tr_' in col],
        'Volumen': [col for col in df_with_indicators.columns if 'volume' in col or 'obv' in col],
        'Sonstige': []
    }
    
    # Finde Spalten, die in keine Gruppe passen
    all_grouped_cols = [col for group in column_groups.values() for col in group]
    column_groups['Sonstige'] = [col for col in df_with_indicators.columns if col not in all_grouped_cols]
    
    # Zeige Gruppen an
    for group_name, columns in column_groups.items():
        if columns:
            print(f"\n{group_name}:")
            for col in columns:
                print(f"  - {col}")
    
    # Speichere Daten als CSV
    save_option = input("\nMöchtest du die Daten als CSV speichern? (j/n): ").lower()
    if save_option.startswith('j'):
        filename = save_to_csv(df_with_indicators, symbol, interval)
        print(f"Daten wurden in {filename} gespeichert.")
    
    # Plotte Daten
    plot_option = input("\nMöchtest du die Indikatoren visualisieren? (j/n): ").lower()
    if plot_option.startswith('j'):
        print("Erstelle Visualisierung...")
        plot_indicators(df_with_indicators, symbol, interval)
    
    print("\nDemo abgeschlossen. Vielen Dank!")

if __name__ == "__main__":
    main()
