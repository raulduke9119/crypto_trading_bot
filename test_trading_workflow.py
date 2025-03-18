#!/usr/bin/env python
"""
End-to-End Test für den Binance Trading Bot.
Testet den kompletten Trading-Workflow von der API-Verbindung bis zur Order-Ausführung.
"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Füge das Root-Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import BINANCE_CONFIG, USE_TESTNET, TRADING_SYMBOLS, DEFAULT_TIMEFRAME
from trading_bot import TradingBot
from data.data_collector import DataCollector
from data.indicators import TechnicalIndicators
from strategies.multi_indicator_strategy import MultiIndicatorStrategy
from utils.order_executor import OrderExecutor
from utils.logger import setup_logger

# Stelle sicher, dass wir immer im Testnet arbeiten
os.environ['USE_TESTNET'] = 'True'

# Logger einrichten
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'trading_workflow_test.log')
logger = setup_logger(log_file=log_file, log_level='INFO')

class TradingWorkflowTest:
    """
    Führt einen End-to-End-Test des Trading-Workflows durch.
    """
    
    def __init__(self, symbol='BTCUSDT', timeframe='5m'):
        """
        Initialisiert den Trading-Workflow-Test.
        
        Args:
            symbol: Trading-Paar für den Test
            timeframe: Zeitintervall für den Test
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.trading_bot = None
        self.data_collector = None
        self.indicators = None
        self.strategy = None
        self.order_executor = None
        
        logger.info(f"Trading-Workflow-Test initialisiert für {symbol} ({timeframe})")
    
    def test_api_connection(self):
        """
        Testet die Verbindung zur Binance API.
        """
        logger.info("=== Test 1: API-Verbindung ===")
        try:
            # Erstelle TradingBot-Instanz
            self.trading_bot = TradingBot(
                use_testnet=True,
                symbols=[self.symbol],
                timeframe=self.timeframe
            )
            
            # Teste Server-Zeit
            server_time = self.trading_bot.client.get_server_time()
            server_time_str = datetime.fromtimestamp(server_time['serverTime'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Binance Server-Zeit: {server_time_str}")
            
            # Teste Konto-Informationen
            account_info = self.trading_bot.client.get_account()
            balances = [b for b in account_info['balances'] if float(b['free']) > 0]
            logger.info(f"Anzahl der Assets mit Guthaben: {len(balances)}")
            
            logger.info("✓ API-Verbindung erfolgreich hergestellt")
            self.data_collector = self.trading_bot.data_collector
            self.order_executor = self.trading_bot.order_executor
            return True
            
        except Exception as e:
            logger.error(f"✗ API-Verbindung fehlgeschlagen: {e}")
            return False
    
    def test_data_collection(self):
        """
        Testet das Sammeln von historischen Daten.
        """
        logger.info("\n=== Test 2: Datensammlung ===")
        try:
            # Historische Daten abrufen
            df = self.data_collector.get_historical_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                start_str="1 day ago UTC"
            )
            
            if df is None or df.empty:
                logger.error("✗ Keine Daten erhalten")
                return False
            
            logger.info(f"✓ {len(df)} Datenpunkte für {self.symbol} erhalten")
            logger.info(f"Zeitraum: {df.index[0]} bis {df.index[-1]}")
            
            # Zeige einige Statistiken
            logger.info(f"Durchschnittlicher Schlusskurs: {df['close'].mean():.2f}")
            logger.info(f"Durchschnittliches Volumen: {df['volume'].mean():.2f}")
            
            self.historical_data = df
            return True
            
        except Exception as e:
            logger.error(f"✗ Datensammlung fehlgeschlagen: {e}")
            return False
    
    def test_indicators(self):
        """
        Testet die Berechnung technischer Indikatoren.
        """
        logger.info("\n=== Test 3: Technische Indikatoren ===")
        try:
            # Initialisiere TechnicalIndicators
            self.indicators = TechnicalIndicators()
            
            # Berechne alle Indikatoren
            df_with_indicators = self.indicators.add_all_indicators(self.historical_data)
            
            if df_with_indicators is None or df_with_indicators.empty:
                logger.error("✗ Keine Indikatoren berechnet")
                return False
            
            # Identifiziere neue Spalten (Indikatoren)
            new_columns = set(df_with_indicators.columns) - set(self.historical_data.columns)
            logger.info(f"✓ {len(new_columns)} Indikatoren berechnet:")
            
            # Zeige die ersten 10 Indikatoren
            for col in sorted(list(new_columns))[:10]:
                logger.info(f"  - {col}")
            
            if len(new_columns) > 10:
                logger.info(f"  - ... und {len(new_columns) - 10} weitere")
            
            self.data_with_indicators = df_with_indicators
            return True
            
        except Exception as e:
            logger.error(f"✗ Indikatorberechnung fehlgeschlagen: {e}")
            return False
    
    def test_strategy(self):
        """
        Testet die Handelsstrategie.
        """
        logger.info("\n=== Test 4: Handelsstrategie ===")
        try:
            # Initialisiere MultiIndicatorStrategy
            self.strategy = MultiIndicatorStrategy(
                risk_percentage=1.0,
                max_positions=2,
                use_ml_predictions=False
            )
            
            # Generiere Handelssignale
            df_with_signals = self.strategy.generate_signals(self.data_with_indicators)
            
            if df_with_signals is None or df_with_signals.empty:
                logger.error("✗ Keine Signale generiert")
                return False
            
            # Suche nach Kauf- und Verkaufssignalen
            buy_signals = df_with_signals.loc[df_with_signals['buy_signal'] == True]
            sell_signals = df_with_signals.loc[df_with_signals['sell_signal'] == True]
            
            logger.info(f"✓ {len(buy_signals)} Kaufsignale und {len(sell_signals)} Verkaufssignale generiert")
            
            # Zeige Details zu den letzten Signalen
            if not buy_signals.empty:
                last_buy = buy_signals.iloc[-1]
                logger.info(f"Letztes Kaufsignal: {last_buy.name}, Stärke: {last_buy.get('buy_strength', 'N/A')}")
            
            if not sell_signals.empty:
                last_sell = sell_signals.iloc[-1]
                logger.info(f"Letztes Verkaufssignal: {last_sell.name}, Stärke: {last_sell.get('sell_strength', 'N/A')}")
            
            self.data_with_signals = df_with_signals
            return True
            
        except Exception as e:
            logger.error(f"✗ Strategietest fehlgeschlagen: {e}")
            return False
    
    def test_order_execution(self, execute_orders=False):
        """
        Testet die Order-Ausführung (ohne tatsächliche Orders zu platzieren).
        
        Args:
            execute_orders: Ob tatsächlich Orders platziert werden sollen (Standard: False)
        """
        logger.info("\n=== Test 5: Order-Ausführung ===")
        try:
            # Hole Kontoguthaben
            usdt_balance = self.order_executor.get_account_balance(asset='USDT')
            btc_balance = self.order_executor.get_account_balance(asset='BTC')
            
            logger.info(f"Aktuelles Guthaben: {usdt_balance:.2f} USDT, {btc_balance:.8f} BTC")
            
            # Simuliere eine Market-Order (ohne tatsächlich zu handeln)
            symbol_info = self.order_executor.get_symbol_info(self.symbol)
            filters = self.order_executor.get_symbol_filters(self.symbol)
            
            logger.info(f"Symbol-Info: {self.symbol}")
            logger.info(f"  - Minimale Ordergröße: {filters['LOT_SIZE']['min_qty']}")
            logger.info(f"  - Schrittgröße: {filters['LOT_SIZE']['step_size']}")
            logger.info(f"  - Minimales Notional: {filters['MIN_NOTIONAL']['min_notional']}")
            
            # Simuliere eine kleine Order
            quantity = 0.001  # Sehr kleine Menge für BTC
            price = float(self.trading_bot.client.get_symbol_ticker(symbol=self.symbol)['price'])
            
            logger.info(f"Aktueller Preis für {self.symbol}: {price}")
            logger.info(f"Simulierte Ordergröße: {quantity} BTC (Wert: {quantity * price:.2f} USDT)")
            
            rounded_quantity = self.order_executor.round_step_size(quantity, float(filters['LOT_SIZE']['step_size']))
            logger.info(f"Gerundete Ordergröße: {rounded_quantity}")
            
            # Option zum tatsächlichen Platzieren einer Order
            if execute_orders and usdt_balance >= 15.0:  # Mindestens 15 USDT Guthaben
                logger.info("Platziere echte Test-Orders (kleine Größe)...")
                
                # Platziere Limit-Buy-Order weit unter Marktpreis
                buy_price = price * 0.9  # 10% unter Marktpreis
                buy_order = self.order_executor.place_limit_order(
                    symbol=self.symbol, 
                    side='BUY', 
                    quantity=rounded_quantity, 
                    price=buy_price
                )
                
                if buy_order:
                    order_id = buy_order['orderId']
                    logger.info(f"✓ Buy-Limit-Order platziert: ID {order_id}")
                    
                    # Storniere die Order sofort wieder
                    time.sleep(2)  # Warte kurz
                    cancelled = self.order_executor.cancel_order(self.symbol, order_id)
                    logger.info(f"✓ Order storniert: {cancelled}")
            else:
                logger.info("Echter Handel deaktiviert oder unzureichendes Guthaben")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Order-Ausführungstest fehlgeschlagen: {e}")
            return False
    
    def run_all_tests(self, execute_orders=False):
        """
        Führt alle Tests nacheinander aus.
        
        Args:
            execute_orders: Ob tatsächlich Orders platziert werden sollen
            
        Returns:
            Dictionary mit Testergebnissen
        """
        results = {}
        
        logger.info("=== Trading-Workflow-Kompletttest ===")
        logger.info(f"Symbol: {self.symbol}, Timeframe: {self.timeframe}")
        logger.info(f"Testnet: {USE_TESTNET}")
        logger.info("Start: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Test 1: API-Verbindung
        results['api_connection'] = self.test_api_connection()
        if not results['api_connection']:
            logger.error("API-Verbindung fehlgeschlagen, breche weitere Tests ab")
            return results
        
        # Test 2: Datensammlung
        results['data_collection'] = self.test_data_collection()
        if not results['data_collection']:
            logger.error("Datensammlung fehlgeschlagen, breche weitere Tests ab")
            return results
        
        # Test 3: Indikatoren
        results['indicators'] = self.test_indicators()
        if not results['indicators']:
            logger.error("Indikatorberechnung fehlgeschlagen, breche weitere Tests ab")
            return results
        
        # Test 4: Strategie
        results['strategy'] = self.test_strategy()
        if not results['strategy']:
            logger.error("Strategietest fehlgeschlagen, breche weitere Tests ab")
            return results
        
        # Test 5: Order-Ausführung
        results['order_execution'] = self.test_order_execution(execute_orders)
        
        # Zusammenfassung
        logger.info("\n=== Zusammenfassung ===")
        all_passed = all(results.values())
        
        for test_name, passed in results.items():
            status = "✓ Bestanden" if passed else "✗ Fehlgeschlagen"
            logger.info(f"{test_name}: {status}")
        
        if all_passed:
            logger.info("Alle Tests erfolgreich bestanden!")
        else:
            logger.warning("Einige Tests sind fehlgeschlagen!")
        
        logger.info("Ende: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return results

def main():
    """
    Hauptfunktion zum Ausführen des Trading-Workflow-Tests.
    """
    # Lade Umgebungsvariablen
    load_dotenv()
    
    # Test-Parameter
    symbol = 'BTCUSDT'  # Standard-Symbol für Tests
    timeframe = '5m'    # 5-Minuten-Intervall
    execute_orders = False  # Keine echten Orders platzieren
    
    # Parse Befehlszeilenargumente für flexible Tests
    import argparse
    parser = argparse.ArgumentParser(description='Trading-Workflow-Test')
    parser.add_argument('--symbol', type=str, default=symbol, help='Trading-Symbol (z.B. BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default=timeframe, help='Zeitintervall (z.B. 5m, 1h)')
    parser.add_argument('--execute-orders', action='store_true', help='Echte Orders platzieren (Vorsicht!)')
    
    args = parser.parse_args()
    
    # Erstelle und führe den Test aus
    workflow_test = TradingWorkflowTest(
        symbol=args.symbol,
        timeframe=args.timeframe
    )
    
    results = workflow_test.run_all_tests(execute_orders=args.execute_orders)
    
    # Gib Exit-Code basierend auf Testergebnissen zurück
    sys.exit(0 if all(results.values()) else 1)

if __name__ == "__main__":
    main() 