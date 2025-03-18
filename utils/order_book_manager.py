"""
Order Book Manager-Modul für den Trading Bot.
Verantwortlich für die Analyse der Markttiefe und Liquidität.
"""
import os
import time
import pandas as pd
import numpy as np
import threading
from typing import Dict, List, Optional, Union, Tuple, Any
from binance.client import Client
from binance.exceptions import BinanceAPIException

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from config.config import DATA_DIRECTORY, LOG_LEVEL, LOG_FILE

# Logger einrichten
logger = setup_logger(os.path.join(DATA_DIRECTORY, LOG_FILE), LOG_LEVEL)

class OrderBookManager:
    """
    Klasse zum Verwalten und Analysieren der Markttiefe (Order Book) für Trading-Entscheidungen.
    Ermöglicht eine verbesserte Preisfindung und Positionsgrößenberechnung basierend auf der
    verfügbaren Liquidität.
    """
    
    def __init__(self, client: Client, symbols: List[str], update_interval: int = 5, depth: int = 100):
        """
        Initialisiert den OrderBookManager.
        
        Args:
            client: Binance Client-Instanz
            symbols: Liste der zu überwachenden Trading-Paare
            update_interval: Intervall in Sekunden für Order Book-Updates
            depth: Tiefe des Order Books (Anzahl der Preislevels)
        """
        self.client = client
        self.symbols = symbols
        self.update_interval = update_interval
        self.depth = depth
        self.running = False
        
        # Initialize order books for each symbol
        self.order_books = {}
        for symbol in symbols:
            self.order_books[symbol] = {
                'bids': {},  # price -> quantity
                'asks': {},  # price -> quantity
                'last_update_id': 0,
                'timestamp': None
            }
            
        # Cache für Support/Resistance-Levels
        self.support_resistance_cache = {}
        self.cache_expiry = 60  # Cache-Ablaufzeit in Sekunden
        self.cache_last_update = {}
        
    def start(self):
        """Startet den Order Book-Aktualisierungsprozess in einem separaten Thread."""
        if self.running:
            logger.warning("OrderBookManager läuft bereits")
            return
            
        self.running = True
        threading.Thread(target=self._order_book_loop, daemon=True).start()
        logger.info(f"OrderBookManager gestartet für {len(self.symbols)} Symbole mit Aktualisierungsintervall {self.update_interval}s")
        
    def stop(self):
        """Stoppt den Order Book-Aktualisierungsprozess."""
        self.running = False
        logger.info("OrderBookManager gestoppt")
        
    def _order_book_loop(self):
        """Hauptschleife zur regelmäßigen Aktualisierung der Order Books."""
        while self.running:
            for symbol in self.symbols:
                try:
                    # Hole Order Book-Snapshot von Binance
                    depth = self.client.get_order_book(symbol=symbol, limit=self.depth)
                    
                    # Aktualisiere lokale Kopie des Order Books
                    self.order_books[symbol]['bids'] = {float(price): float(qty) for price, qty in depth['bids']}
                    self.order_books[symbol]['asks'] = {float(price): float(qty) for price, qty in depth['asks']}
                    self.order_books[symbol]['last_update_id'] = depth['lastUpdateId']
                    self.order_books[symbol]['timestamp'] = time.time()
                    
                    # Invalidiere Cache nach Aktualisierung
                    if symbol in self.support_resistance_cache:
                        del self.support_resistance_cache[symbol]
                        
                except BinanceAPIException as e:
                    logger.error(f"Binance API-Fehler beim Aktualisieren des Order Books für {symbol}: {e}")
                except Exception as e:
                    logger.error(f"Fehler beim Aktualisieren des Order Books für {symbol}: {e}")
                    
                # Kurze Pause zwischen Symbolen, um Rate Limits zu vermeiden
                time.sleep(0.2)
                
            # Warte bis zum nächsten Update-Intervall
            time.sleep(self.update_interval)
    
    def get_order_book(self, symbol: str) -> Dict[str, Any]:
        """
        Gibt das aktuelle Order Book für ein Symbol zurück.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            
        Returns:
            Dictionary mit Bids, Asks und Metadaten oder leeres Dictionary bei Fehler
        """
        if symbol not in self.symbols or symbol not in self.order_books:
            logger.warning(f"Symbol {symbol} nicht in der Überwachungsliste")
            return {'bids': {}, 'asks': {}, 'last_update_id': 0, 'timestamp': None}
            
        return self.order_books[symbol]
        
    def get_market_depth(self, symbol: str, side: str = 'both', levels: int = 5) -> pd.DataFrame:
        """
        Gibt die Markttiefe als DataFrame zurück.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            side: 'bids', 'asks' oder 'both'
            levels: Anzahl der Preislevels, die zurückgegeben werden sollen
            
        Returns:
            DataFrame mit Preis- und Mengeninformationen
        """
        if symbol not in self.order_books:
            logger.warning(f"Symbol {symbol} nicht in der Überwachungsliste")
            return pd.DataFrame()
            
        order_book = self.order_books[symbol]
        
        if side == 'bids' or side == 'both':
            bids = sorted(order_book['bids'].items(), reverse=True)[:levels]
            bids_df = pd.DataFrame(bids, columns=['price', 'quantity'])
            bids_df['side'] = 'bid'
            
        if side == 'asks' or side == 'both':
            asks = sorted(order_book['asks'].items())[:levels]
            asks_df = pd.DataFrame(asks, columns=['price', 'quantity'])
            asks_df['side'] = 'ask'
            
        if side == 'both':
            result = pd.concat([bids_df, asks_df], ignore_index=True)
        elif side == 'bids':
            result = bids_df
        else:
            result = asks_df
            
        return result
        
    def get_best_bid(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Gibt den besten Bid-Preis und die zugehörige Menge zurück.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            
        Returns:
            Tuple mit (Preis, Menge) oder (None, None) bei Fehler
        """
        if symbol not in self.order_books:
            logger.warning(f"Symbol {symbol} nicht in der Überwachungsliste")
            return None, None
            
        bids = self.order_books[symbol]['bids']
        if not bids:
            return None, None
            
        best_price = max(bids.keys())
        return best_price, bids[best_price]
        
    def get_best_ask(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Gibt den besten Ask-Preis und die zugehörige Menge zurück.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            
        Returns:
            Tuple mit (Preis, Menge) oder (None, None) bei Fehler
        """
        if symbol not in self.order_books:
            logger.warning(f"Symbol {symbol} nicht in der Überwachungsliste")
            return None, None
            
        asks = self.order_books[symbol]['asks']
        if not asks:
            return None, None
            
        best_price = min(asks.keys())
        return best_price, asks[best_price]
        
    def get_spread(self, symbol: str) -> Optional[float]:
        """
        Berechnet den aktuellen Spread für ein Symbol.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            
        Returns:
            Spread als Prozentsatz oder None bei Fehler
        """
        best_bid, _ = self.get_best_bid(symbol)
        best_ask, _ = self.get_best_ask(symbol)
        
        if best_bid is None or best_ask is None:
            return None
            
        spread = (best_ask - best_bid) / best_bid * 100
        return spread
        
    def get_optimal_entry_price(self, symbol: str, side: str, quantity: float) -> Optional[float]:
        """
        Berechnet den optimalen Einstiegspreis basierend auf der Markttiefe und gewünschten Menge.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            side: 'BUY' oder 'SELL'
            quantity: Gewünschte Menge
            
        Returns:
            Optimaler Einstiegspreis unter Berücksichtigung der Liquidität
        """
        if symbol not in self.order_books:
            logger.warning(f"Symbol {symbol} nicht in der Überwachungsliste")
            return None
            
        order_book = self.order_books[symbol]
        
        if side.upper() == 'BUY':
            # Für Käufe schauen wir auf die Ask-Preise
            sorted_asks = sorted(order_book['asks'].items())
            cumulative_qty = 0
            weighted_price = 0
            
            for price, qty in sorted_asks:
                if cumulative_qty >= quantity:
                    break
                    
                # Wie viel können wir zu diesem Preis kaufen?
                available_qty = min(qty, quantity - cumulative_qty)
                weighted_price += price * (available_qty / quantity)
                cumulative_qty += available_qty
                
            if cumulative_qty < quantity:
                logger.warning(f"Nicht genügend Liquidität im Order Book für {symbol} beim Kauf von {quantity}")
                # Verwende den letzten Preis, falls nicht genug Liquidität vorhanden ist
                return sorted_asks[-1][0] if sorted_asks else None
                
            return weighted_price
            
        else:  # SELL
            # Für Verkäufe schauen wir auf die Bid-Preise
            sorted_bids = sorted(order_book['bids'].items(), reverse=True)
            cumulative_qty = 0
            weighted_price = 0
            
            for price, qty in sorted_bids:
                if cumulative_qty >= quantity:
                    break
                    
                # Wie viel können wir zu diesem Preis verkaufen?
                available_qty = min(qty, quantity - cumulative_qty)
                weighted_price += price * (available_qty / quantity)
                cumulative_qty += available_qty
                
            if cumulative_qty < quantity:
                logger.warning(f"Nicht genügend Liquidität im Order Book für {symbol} beim Verkauf von {quantity}")
                # Verwende den letzten Preis, falls nicht genug Liquidität vorhanden ist
                return sorted_bids[-1][0] if sorted_bids else None
                
            return weighted_price
            
    def get_support_resistance_levels(self, symbol: str, num_levels: int = 3) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Identifiziert Support- und Resistance-Levels basierend auf Volumenkonzentrationen im Order Book.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            num_levels: Anzahl der zurückzugebenden Levels
            
        Returns:
            Tuple mit (Support-Levels, Resistance-Levels) als Liste von (Preis, Volumen)-Tuples
        """
        # Prüfen ob wir aktuelle Daten im Cache haben
        current_time = time.time()
        if (symbol in self.support_resistance_cache and 
            symbol in self.cache_last_update and 
            current_time - self.cache_last_update[symbol] < self.cache_expiry):
            return self.support_resistance_cache[symbol]
            
        if symbol not in self.order_books:
            logger.warning(f"Symbol {symbol} nicht in der Überwachungsliste")
            return [], []
            
        order_book = self.order_books[symbol]
        
        # Gruppenbreite festlegen (abhängig vom Preis)
        # Für hochpreisige Assets wie BTC verwenden wir größere Gruppen
        bid_prices = list(order_book['bids'].keys())
        ask_prices = list(order_book['asks'].keys())
        
        if not bid_prices or not ask_prices:
            return [], []
            
        # Dynamische Bestimmung der optimalen Gruppengröße basierend auf dem Preisbereich
        avg_price = (sum(bid_prices) / len(bid_prices) + sum(ask_prices) / len(ask_prices)) / 2
        
        if avg_price > 10000:  # BTC-Bereich
            price_step = 100.0
        elif avg_price > 1000:  # ETH-Bereich
            price_step = 10.0
        elif avg_price > 100:
            price_step = 1.0
        elif avg_price > 10:
            price_step = 0.1
        elif avg_price > 1:
            price_step = 0.01
        else:
            price_step = 0.001
            
        # Support-Levels (Bids)
        support_levels = {}
        for price, qty in order_book['bids'].items():
            price_level = int(price / price_step) * price_step
            if price_level not in support_levels:
                support_levels[price_level] = 0
            support_levels[price_level] += qty
            
        # Resistance-Levels (Asks)
        resistance_levels = {}
        for price, qty in order_book['asks'].items():
            price_level = int(price / price_step) * price_step
            if price_level not in resistance_levels:
                resistance_levels[price_level] = 0
            resistance_levels[price_level] += qty
            
        # Sortieren nach Volumen, um die stärksten Levels zu finden
        sorted_support = sorted(support_levels.items(), key=lambda x: x[1], reverse=True)[:num_levels]
        sorted_resistance = sorted(resistance_levels.items(), key=lambda x: x[1], reverse=True)[:num_levels]
        
        # Ergebnis im Cache speichern
        self.support_resistance_cache[symbol] = (sorted_support, sorted_resistance)
        self.cache_last_update[symbol] = current_time
        
        return sorted_support, sorted_resistance
        
    def calculate_slippage(self, symbol: str, side: str, quantity: float) -> float:
        """
        Berechnet den erwarteten Slippage für einen Trade mit bestimmter Größe.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            side: 'BUY' oder 'SELL'
            quantity: Zu handelnde Menge
            
        Returns:
            Erwarteter Slippage als Prozentsatz
        """
        optimal_price = self.get_optimal_entry_price(symbol, side, quantity)
        
        if optimal_price is None:
            return 0.0
            
        if side.upper() == 'BUY':
            best_price, _ = self.get_best_ask(symbol)
            if best_price is None:
                return 0.0
            slippage = (optimal_price - best_price) / best_price * 100
        else:  # SELL
            best_price, _ = self.get_best_bid(symbol)
            if best_price is None:
                return 0.0
            slippage = (best_price - optimal_price) / best_price * 100
            
        return max(0.0, slippage)  # Slippage kann nicht negativ sein
        
    def get_optimal_take_profit_levels(self, symbol: str, entry_price: float, side: str = 'BUY', num_levels: int = 3) -> List[float]:
        """
        Berechnet optimale Take-Profit-Levels basierend auf der Marktstruktur.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            entry_price: Einstiegspreis
            side: 'BUY' (Long) oder 'SELL' (Short)
            num_levels: Anzahl der zurückzugebenden Levels
            
        Returns:
            Liste mit optimalen Take-Profit-Preisen
        """
        support_levels, resistance_levels = self.get_support_resistance_levels(symbol, num_levels * 2)
        
        if side.upper() == 'BUY':
            # For long positions, look at resistance levels above entry price
            tp_levels = [price for price, _ in resistance_levels if price > entry_price]
            # If we don't have enough levels, add some based on percentage increases
            while len(tp_levels) < num_levels:
                next_level = entry_price * (1 + 0.01 * (len(tp_levels) + 1))
                tp_levels.append(next_level)
        else:
            # For short positions, look at support levels below entry price
            tp_levels = [price for price, _ in support_levels if price < entry_price]
            # If we don't have enough levels, add some based on percentage decreases
            while len(tp_levels) < num_levels:
                next_level = entry_price * (1 - 0.01 * (len(tp_levels) + 1))
                tp_levels.append(next_level)
                
        # Ensure we return exactly num_levels sorted levels
        return sorted(tp_levels)[:num_levels]
        
    def get_optimal_stop_loss_level(self, symbol: str, entry_price: float, side: str = 'BUY') -> float:
        """
        Berechnet ein optimales Stop-Loss-Level basierend auf der Marktstruktur.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            entry_price: Einstiegspreis
            side: 'BUY' (Long) oder 'SELL' (Short)
            
        Returns:
            Optimaler Stop-Loss-Preis
        """
        support_levels, resistance_levels = self.get_support_resistance_levels(symbol, 5)
        
        if side.upper() == 'BUY':
            # For long positions, find nearest support level below entry price
            support_prices = [price for price, _ in support_levels if price < entry_price]
            if support_prices:
                # Use the strongest support level below entry price
                return max(support_prices)
            else:
                # If no support level found, use a default percentage
                return entry_price * 0.97  # 3% below entry
        else:
            # For short positions, find nearest resistance level above entry price
            resistance_prices = [price for price, _ in resistance_levels if price > entry_price]
            if resistance_prices:
                # Use the strongest resistance level above entry price
                return min(resistance_prices)
            else:
                # If no resistance level found, use a default percentage
                return entry_price * 1.03  # 3% above entry
                
    def is_enough_liquidity(self, symbol: str, side: str, quantity: float, max_slippage: float = 0.5) -> bool:
        """
        Prüft, ob genügend Liquidität für einen Trade vorhanden ist.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            side: 'BUY' oder 'SELL'
            quantity: Zu handelnde Menge
            max_slippage: Maximaler akzeptabler Slippage in Prozent
            
        Returns:
            True, wenn genügend Liquidität vorhanden ist, sonst False
        """
        slippage = self.calculate_slippage(symbol, side, quantity)
        return slippage <= max_slippage 