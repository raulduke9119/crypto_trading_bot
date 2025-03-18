#!/usr/bin/env python
"""
Einfaches Startskript für den Binance Trading Bot.
Bietet eine benutzerfreundliche Oberfläche zum Starten des Bots mit verschiedenen Konfigurationen.
"""
import os
import sys
import argparse
import logging
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Füge das Root-Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_bot import TradingBot
from config.config import TRADING_SYMBOLS, DEFAULT_TIMEFRAME, LOG_LEVEL, LOG_FILE
from utils.logger import setup_logger

# Logger einrichten
logger = setup_logger(LOG_FILE, LOG_LEVEL)

def setup_argparse():
    """
    Konfiguriert den Argument-Parser für benutzerfreundliche Optionen.
    """
    parser = argparse.ArgumentParser(
        description='Binance Trading Bot mit optimierter Multi-Indikator-Strategie',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Hauptbetriebsmodus
    mode_group = parser.add_argument_group('Betriebsmodus')
    mode_group.add_argument(
        '--mode', '-m',
        choices=['backtest', 'live', 'paper'],
        default='paper',
        help='Betriebsmodus: backtest = Historische Daten, live = Echtgeld-Trading, paper = Simuliertes Trading'
    )
    
    # Trading-Konfiguration
    trading_group = parser.add_argument_group('Trading-Konfiguration')
    trading_group.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=['BTCUSDT'],
        help='Trading-Symbole (z.B. BTCUSDT ETHUSDT SOLUSDT)'
    )
    trading_group.add_argument(
        '--timeframe', '-t',
        default=DEFAULT_TIMEFRAME,
        help='Zeitintervall (z.B. 5m, 15m, 1h, 4h, 1d)'
    )
    trading_group.add_argument(
        '--interval', '-i',
        type=int,
        default=15,
        help='Update-Intervall in Minuten (nur im Live/Paper-Modus)'
    )
    
    # Risikomanagement
    risk_group = parser.add_argument_group('Risikomanagement')
    risk_group.add_argument(
        '--risk', '-r',
        type=float,
        default=1.0,
        help='Risikoprozentsatz pro Trade (1.0 = 1%% des Kapitals)'
    )
    risk_group.add_argument(
        '--max-positions', '-p',
        type=int,
        default=2,
        help='Maximale Anzahl gleichzeitiger Positionen'
    )
    risk_group.add_argument(
        '--trailing-stop',
        type=float,
        default=1.5,
        help='Trailing-Stop-Prozentsatz für Gewinnmitnahme'
    )
    risk_group.add_argument(
        '--max-drawdown',
        type=float,
        default=5.0,
        help='Maximaler erlaubter Drawdown in Prozent'
    )
    
    # Backtest-Konfiguration
    backtest_group = parser.add_argument_group('Backtest-Konfiguration')
    backtest_group.add_argument(
        '--start-date',
        default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        help='Startdatum für Backtest (YYYY-MM-DD)'
    )
    backtest_group.add_argument(
        '--end-date',
        default=None,
        help='Enddatum für Backtest (YYYY-MM-DD, None = heute)'
    )
    backtest_group.add_argument(
        '--initial-balance',
        type=float,
        default=1000.0,
        help='Anfangskapital für Backtest in USDT'
    )
    
    # Erweiterte Optionen
    advanced_group = parser.add_argument_group('Erweiterte Optionen')
    advanced_group.add_argument(
        '--use-ml',
        action='store_true',
        help='ML-Vorhersagen aktivieren'
    )
    advanced_group.add_argument(
        '--strategy',
        choices=['multi_indicator', 'dogebtc_hf'],
        default='multi_indicator',
        help='Zu verwendende Trading-Strategie'
    )
    advanced_group.add_argument(
        '--pattern',
        type=str,
        default='default_pattern.json',
        help='Zu verwendendes Trading-Pattern (aus dem patterns/-Verzeichnis)'
    )
    advanced_group.add_argument(
        '--testnet',
        action='store_true',
        help='Binance Testnet verwenden (für Live/Paper-Trading)'
    )
    advanced_group.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Log-Level für die Protokollierung'
    )
    
    return parser

def display_header():
    """
    Zeigt einen schönen ASCII-Header für den Trading Bot an.
    """
    header = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗ ██╗███╗   ██╗ █████╗ ███╗   ██╗ ██████╗███████╗   ║
║   ██╔══██╗██║████╗  ██║██╔══██╗████╗  ██║██╔════╝██╔════╝   ║
║   ██████╔╝██║██╔██╗ ██║███████║██╔██╗ ██║██║     █████╗     ║
║   ██╔══██╗██║██║╚██╗██║██╔══██║██║╚██╗██║██║     ██╔══╝     ║
║   ██████╔╝██║██║ ╚████║██║  ██║██║ ╚████║╚██████╗███████╗   ║
║   ╚═════╝ ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝   ║
║                                                              ║
║                  TRADING BOT v1.0.0                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(header)

def display_settings(args):
    """
    Zeigt die aktuellen Einstellungen übersichtlich an.
    
    Args:
        args: Geparste Befehlszeilenargumente
    """
    settings = f"""
╔══════════════════════════════════════════════════════════════╗
║ TRADING BOT EINSTELLUNGEN                                    ║
╠══════════════════════════════════════════════════════════════╣
║ Betriebsmodus:      {args.mode:<42} ║
║ Trading-Symbole:    {', '.join(args.symbols):<42} ║
║ Zeitintervall:      {args.timeframe:<42} ║
║ Update-Intervall:   {args.interval} Minuten{' ' * 31} ║
║ Risikoprozentsatz:  {args.risk}%{' ' * 39} ║
║ Max. Positionen:    {args.max_positions}{' ' * 41} ║
║ Trailing-Stop:      {args.trailing_stop}%{' ' * 39} ║
║ Max. Drawdown:      {args.max_drawdown}%{' ' * 39} ║
║ ML-Vorhersagen:     {'Aktiviert' if args.use_ml else 'Deaktiviert':<42} ║
║ Strategie:          {args.strategy:<42} ║
║ Pattern:            {args.pattern:<42} ║
║ Testnet:            {'Ja' if args.testnet else 'Nein':<42} ║
╚══════════════════════════════════════════════════════════════╝
"""
    if args.mode == 'backtest':
        backtest_settings = f"""
╔══════════════════════════════════════════════════════════════╗
║ BACKTEST EINSTELLUNGEN                                       ║
╠══════════════════════════════════════════════════════════════╣
║ Startdatum:         {args.start_date:<42} ║
║ Enddatum:           {args.end_date if args.end_date else 'Heute':<42} ║
║ Anfangskapital:     {args.initial_balance} USDT{' ' * 34} ║
╚══════════════════════════════════════════════════════════════╝
"""
        settings += backtest_settings
    
    print(settings)

def display_warning():
    """
    Zeigt eine Warnung für den Live-Trading-Modus an.
    """
    warning = """
╔══════════════════════════════════════════════════════════════╗
║                        ⚠️ WARNUNG ⚠️                          ║
╠══════════════════════════════════════════════════════════════╣
║ Du hast den LIVE-TRADING-MODUS aktiviert.                    ║
║ In diesem Modus wird mit echtem Geld gehandelt!              ║
║                                                              ║
║ Bitte stelle sicher, dass du:                                ║
║ - Die Handelsstrategie verstehst                             ║
║ - Die Risiken des Krypto-Handels kennst                      ║
║ - Nie mehr Geld einsetzt, als du verlieren kannst            ║
║                                                              ║
║ Der Autor dieses Bots übernimmt keine Verantwortung für      ║
║ eventuelle finanzielle Verluste.                             ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(warning)
    
    while True:
        response = input("Möchtest du wirklich fortfahren? (ja/nein): ").lower()
        if response in ["ja", "j", "yes", "y"]:
            break
        elif response in ["nein", "n", "no"]:
            print("Live-Trading abgebrochen. Beende Programm.")
            sys.exit(0)
        else:
            print("Bitte antworte mit 'ja' oder 'nein'.")

def run_live_trading(args):
    """
    Führt den Bot im Live- oder Paper-Trading-Modus aus.
    
    Args:
        args: Geparste Befehlszeilenargumente
    """
    # Initialisiere den Trading Bot
    logger.info(f"Starte Trading Bot im {'PAPER' if args.mode == 'paper' else 'LIVE'}-Modus")
    logger.info(f"Trading-Symbole: {', '.join(args.symbols)}")
    
    # Konfiguriere den Bot
    bot = TradingBot(
        use_testnet=args.testnet,
        symbols=args.symbols,
        timeframe=args.timeframe,
        risk_percentage=args.risk,
        max_positions=args.max_positions,
        use_ml=args.use_ml,
        pattern_name=args.pattern
    )
    
    # Teste die Verbindung
    if not bot.test_connection():
        logger.error("Verbindungstest fehlgeschlagen! Bitte API-Schlüssel überprüfen.")
        return
    
    # Starte den Trading-Loop
    logger.info(f"Trading-Loop gestartet (Update alle {args.interval} Minuten)")
    
    try:
        # Initialer Daten-Update
        market_data = bot.update_market_data()
        if not market_data:
            logger.error("Konnte keine Marktdaten abrufen. Beende Programm.")
            return
        
        # Trading-Loop
        while True:
            # Update Marktdaten
            market_data = bot.update_market_data()
            
            # Verarbeite Trading-Signale
            for symbol, df in market_data.items():
                if df is None or df.empty:
                    logger.warning(f"Keine Daten für {symbol}")
                    continue
                
                # Prüfe auf neue Signale (letzte Zeile)
                last_row = df.iloc[-1]
                if last_row.get('buy_signal', False):
                    strength = last_row.get('buy_strength', 1.0)
                    logger.info(f"BUY Signal für {symbol}: Stärke {strength}")
                    
                    # Führe Order aus (im Paper-Modus wird dies simuliert)
                    if args.mode == 'live':
                        bot.order_executor.execute_buy_order(symbol, strength)
                
                if last_row.get('sell_signal', False):
                    strength = last_row.get('sell_strength', 1.0)
                    logger.info(f"SELL Signal für {symbol}: Stärke {strength}")
                    
                    # Führe Order aus (im Paper-Modus wird dies simuliert)
                    if args.mode == 'live':
                        bot.order_executor.execute_sell_order(symbol, strength)
            
            # Warte bis zum nächsten Update
            logger.info(f"Warte {args.interval} Minuten bis zum nächsten Update...")
            time.sleep(args.interval * 60)  # Warte in Sekunden
            
    except KeyboardInterrupt:
        logger.info("Trading-Loop durch Benutzer beendet (STRG+C)")
    except Exception as e:
        logger.error(f"Fehler im Trading-Loop: {e}", exc_info=True)

def run_backtest(args):
    """
    Führt einen Backtest durch.
    
    Args:
        args: Geparste Befehlszeilenargumente
    """
    logger.info(f"Starte Backtest von {args.start_date} bis {args.end_date or 'heute'}")
    logger.info(f"Trading-Symbole: {', '.join(args.symbols)}, Anfangskapital: {args.initial_balance} USDT")
    
    # Initialisiere den Trading Bot
    bot = TradingBot(
        use_testnet=args.testnet,
        symbols=args.symbols,
        timeframe=args.timeframe,
        risk_percentage=args.risk,
        max_positions=args.max_positions,
        initial_capital=args.initial_balance,
        use_ml=args.use_ml,
        pattern_name=args.pattern
    )
    
    # Führe Backtest aus
    results = bot.backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance
    )
    
    # Zeige Ergebnisse
    if results:
        print("\n=== Backtest-Ergebnisse ===")
        print(f"Startkapital: ${args.initial_balance:.2f}")
        print(f"Endkapital: ${results['final_balance']:.2f}")
        print(f"Rendite: {results['return_percentage']:.2f}%")
        print(f"Anzahl Trades: {results['total_trades']}")
        print(f"Gewinnrate: {results['win_rate']:.2f}%")
        print(f"Maximaler Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    else:
        logger.error("Backtest lieferte keine Ergebnisse.")

def main():
    """
    Hauptfunktion für den Start des Trading Bots.
    """
    # Lade Umgebungsvariablen
    load_dotenv()
    
    # Zeige Header
    display_header()
    
    # Parse Befehlszeilenargumente
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Passe Log-Level an
    if args.log_level:
        logging.getLogger().setLevel(args.log_level)
    
    # Zeige Einstellungen
    display_settings(args)
    
    # Zeige Warnung für Live-Trading
    if args.mode == 'live':
        display_warning()
    
    # Starte den Bot im gewählten Modus
    if args.mode == 'backtest':
        run_backtest(args)
    else:  # 'live' oder 'paper'
        run_live_trading(args)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer beendet (STRG+C).")
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}", exc_info=True)
        sys.exit(1) 