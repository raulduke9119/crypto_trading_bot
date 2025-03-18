"""
Order-Executor-Modul für den Trading Bot.
Verantwortlich für die Ausführung von Handelsaufträgen auf Binance.
"""
import os
import time
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from config.config import LOG_LEVEL, LOG_FILE, DATA_DIRECTORY

# Logger einrichten
logger = setup_logger(os.path.join(DATA_DIRECTORY, LOG_FILE), LOG_LEVEL)

class OrderExecutor:
    """
    Klasse zur Ausführung von Handelsaufträgen auf Binance.
    """
    
    def __init__(self, client: Client):
        """
        Initialisiert den OrderExecutor.
        
        Args:
            client: Binance Client-Instanz
        """
        self.client = client
        logger.info("OrderExecutor initialisiert")
    
    def get_account_balance(self, asset: str = 'USDT') -> float:
        """
        Gibt das Guthaben eines bestimmten Assets zurück.
        
        Args:
            asset: Asset-Symbol (Standard: USDT)
            
        Returns:
            Float-Guthaben
        """
        try:
            account_info = self.client.get_account()
            balances = account_info['balances']
            
            for balance in balances:
                if balance['asset'] == asset:
                    return float(balance['free'])
            
            logger.warning(f"Asset {asset} nicht im Konto gefunden")
            return 0
            
        except BinanceAPIException as e:
            logger.error(f"Fehler beim Abrufen des Kontoguthabens: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Abrufen des Kontoguthabens: {e}")
            return 0
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Gibt Informationen über ein Symbol zurück.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            
        Returns:
            Symbol-Informationen oder None bei Fehler
        """
        try:
            symbol_info = self.client.get_symbol_info(symbol)
            return symbol_info
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Symbol-Informationen für {symbol}: {e}")
            return None
    
    def get_symbol_filters(self, symbol: str) -> Dict[str, Dict[str, float]]:
        """
        Extrahiert die wichtigsten Filter für ein Symbol.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            
        Returns:
            Dictionary mit Filtern
        """
        filters = {
            'LOT_SIZE': {'min_qty': 0, 'max_qty': 0, 'step_size': 0},
            'PRICE_FILTER': {'min_price': 0, 'max_price': 0, 'tick_size': 0},
            'MIN_NOTIONAL': {'min_notional': 0}
        }
        
        try:
            symbol_info = self.get_symbol_info(symbol)
            
            if not symbol_info:
                return filters
            
            for filter_dict in symbol_info['filters']:
                filter_type = filter_dict['filterType']
                
                if filter_type == 'LOT_SIZE':
                    filters['LOT_SIZE']['min_qty'] = float(filter_dict['minQty'])
                    filters['LOT_SIZE']['max_qty'] = float(filter_dict['maxQty'])
                    filters['LOT_SIZE']['step_size'] = float(filter_dict['stepSize'])
                
                elif filter_type == 'PRICE_FILTER':
                    filters['PRICE_FILTER']['min_price'] = float(filter_dict['minPrice'])
                    filters['PRICE_FILTER']['max_price'] = float(filter_dict['maxPrice'])
                    filters['PRICE_FILTER']['tick_size'] = float(filter_dict['tickSize'])
                
                elif filter_type == 'MIN_NOTIONAL':
                    filters['MIN_NOTIONAL']['min_notional'] = float(filter_dict['minNotional'])
            
            return filters
            
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren der Symbol-Filter für {symbol}: {e}")
            return filters
    
    def round_step_size(self, quantity: float, step_size: float) -> float:
        """
        Rundet eine Menge auf die nächste gültige Schrittgröße.
        
        Args:
            quantity: Zu rundende Menge
            step_size: Schrittgröße
            
        Returns:
            Gerundete Menge
        """
        if step_size == 0:
            return quantity
        
        precision = int(round(-1 * np.log10(step_size)))
        return np.round(quantity - (quantity % step_size), precision)
    
    def place_market_order(self, 
                          symbol: str, 
                          side: str, 
                          quantity: float) -> Optional[Dict[str, Any]]:
        """
        Platziert eine Market-Order.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            side: 'BUY' oder 'SELL'
            quantity: Zu kaufende/verkaufende Menge
            
        Returns:
            Order-Informationen oder None bei Fehler
        """
        try:
            # Hole Symbol-Filter
            filters = self.get_symbol_filters(symbol)
            
            # Runde Menge auf gültige Schrittgröße
            step_size = filters['LOT_SIZE']['step_size']
            min_qty = filters['LOT_SIZE']['min_qty']
            
            if quantity < min_qty:
                logger.warning(f"Menge {quantity} ist kleiner als Mindestmenge {min_qty} für {symbol}")
                return None
            
            quantity = self.round_step_size(quantity, step_size)
            
            # Platziere Order
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            logger.info(f"{side} Market-Order platziert: {quantity} {symbol}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Fehler beim Platzieren der Market-Order: {e}")
            return None
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Platzieren der Market-Order: {e}")
            return None
    
    def place_limit_order(self, 
                         symbol: str, 
                         side: str, 
                         quantity: float, 
                         price: float) -> Optional[Dict[str, Any]]:
        """
        Platziert eine Limit-Order.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            side: 'BUY' oder 'SELL'
            quantity: Zu kaufende/verkaufende Menge
            price: Limit-Preis
            
        Returns:
            Order-Informationen oder None bei Fehler
        """
        try:
            # Hole Symbol-Filter
            filters = self.get_symbol_filters(symbol)
            
            # Runde Menge und Preis auf gültige Schrittgrößen
            step_size = filters['LOT_SIZE']['step_size']
            tick_size = filters['PRICE_FILTER']['tick_size']
            min_qty = filters['LOT_SIZE']['min_qty']
            min_notional = filters['MIN_NOTIONAL']['min_notional']
            
            if quantity < min_qty:
                logger.warning(f"Menge {quantity} ist kleiner als Mindestmenge {min_qty} für {symbol}")
                return None
            
            if quantity * price < min_notional:
                logger.warning(f"Ordervolumen {quantity * price} ist kleiner als Mindestvolumen {min_notional} für {symbol}")
                return None
            
            quantity = self.round_step_size(quantity, step_size)
            
            # Runde Preis auf gültige Tickgröße
            precision = int(round(-1 * np.log10(tick_size)))
            price = np.round(price, precision)
            
            # Platziere Order
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=quantity,
                price=str(price)
            )
            
            logger.info(f"{side} Limit-Order platziert: {quantity} {symbol} zu {price}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Fehler beim Platzieren der Limit-Order: {e}")
            return None
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Platzieren der Limit-Order: {e}")
            return None
    
    def place_stop_loss_order(self, 
                             symbol: str, 
                             quantity: float, 
                             stop_price: float) -> Optional[Dict[str, Any]]:
        """
        Platziert eine Stop-Loss-Order.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            quantity: Zu verkaufende Menge
            stop_price: Stop-Preis
            
        Returns:
            Order-Informationen oder None bei Fehler
        """
        try:
            # Hole Symbol-Filter
            filters = self.get_symbol_filters(symbol)
            
            # Runde Menge und Preis auf gültige Schrittgrößen
            step_size = filters['LOT_SIZE']['step_size']
            tick_size = filters['PRICE_FILTER']['tick_size']
            
            quantity = self.round_step_size(quantity, step_size)
            
            # Runde Preis auf gültige Tickgröße
            precision = int(round(-1 * np.log10(tick_size)))
            stop_price = np.round(stop_price, precision)
            
            # Limit-Preis leicht unter Stop-Preis
            limit_price = np.round(stop_price * 0.99, precision)
            
            # Platziere Order
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_STOP_LOSS_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=quantity,
                stopPrice=str(stop_price),
                price=str(limit_price)
            )
            
            logger.info(f"Stop-Loss-Order platziert: {quantity} {symbol} bei {stop_price}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Fehler beim Platzieren der Stop-Loss-Order: {e}")
            return None
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Platzieren der Stop-Loss-Order: {e}")
            return None
    
    def place_take_profit_order(self, 
                               symbol: str, 
                               quantity: float, 
                               take_profit_price: float) -> Optional[Dict[str, Any]]:
        """
        Platziert eine Take-Profit-Order.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            quantity: Zu verkaufende Menge
            take_profit_price: Take-Profit-Preis
            
        Returns:
            Order-Informationen oder None bei Fehler
        """
        try:
            # Hole Symbol-Filter
            filters = self.get_symbol_filters(symbol)
            
            # Runde Menge und Preis auf gültige Schrittgrößen
            step_size = filters['LOT_SIZE']['step_size']
            tick_size = filters['PRICE_FILTER']['tick_size']
            
            quantity = self.round_step_size(quantity, step_size)
            
            # Runde Preis auf gültige Tickgröße
            precision = int(round(-1 * np.log10(tick_size)))
            take_profit_price = np.round(take_profit_price, precision)
            
            # Platziere Order
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=quantity,
                price=str(take_profit_price)
            )
            
            logger.info(f"Take-Profit-Order platziert: {quantity} {symbol} bei {take_profit_price}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Fehler beim Platzieren der Take-Profit-Order: {e}")
            return None
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Platzieren der Take-Profit-Order: {e}")
            return None
    
    def check_order_status(self, symbol: str, order_id: int) -> Optional[str]:
        """
        Prüft den Status einer Order.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            order_id: Order-ID
            
        Returns:
            Order-Status oder None bei Fehler
        """
        try:
            order = self.client.get_order(
                symbol=symbol,
                orderId=order_id
            )
            
            status = order['status']
            logger.debug(f"Order-Status für {order_id}: {status}")
            return status
            
        except BinanceAPIException as e:
            logger.error(f"Fehler beim Prüfen des Order-Status: {e}")
            return None
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Prüfen des Order-Status: {e}")
            return None
    
    def cancel_order(self, symbol: str, order_id: int) -> bool:
        """
        Storniert eine Order.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            order_id: Order-ID
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            self.client.cancel_order(
                symbol=symbol,
                orderId=order_id
            )
            
            logger.info(f"Order {order_id} für {symbol} storniert")
            return True
            
        except BinanceAPIException as e:
            logger.error(f"Fehler beim Stornieren der Order: {e}")
            return False
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Stornieren der Order: {e}")
            return False
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Gibt alle offenen Orders zurück.
        
        Args:
            symbol: Trading-Paar (optional, für alle Symbole wenn None)
            
        Returns:
            Liste offener Orders
        """
        try:
            if symbol:
                orders = self.client.get_open_orders(symbol=symbol)
            else:
                orders = self.client.get_open_orders()
            
            logger.debug(f"{len(orders)} offene Orders gefunden")
            return orders
            
        except BinanceAPIException as e:
            logger.error(f"Fehler beim Abrufen offener Orders: {e}")
            return []
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Abrufen offener Orders: {e}")
            return []
    
    def get_order_history(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Ruft die Order-Historie für ein Symbol ab.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            limit: Maximale Anzahl von Ergebnissen
            
        Returns:
            Liste mit Order-Historien-Einträgen
        """
        try:
            orders = self.client.get_all_orders(symbol=symbol, limit=limit)
            return orders
        except BinanceAPIException as e:
            logger.error(f"Fehler beim Abrufen der Order-Historie für {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Abrufen der Order-Historie: {e}")
            return []
    
    def execute_buy_order(self, symbol: str, strength: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Führt eine Kauforder basierend auf einem Trading-Signal aus.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            strength: Signalstärke (1.0-5.0), beeinflusst die Ordergröße
            
        Returns:
            Order-Informationen oder None bei Fehler
        """
        try:
            # Überprüfe, ob das Symbol ein USDT-Paar ist
            base_asset = symbol.replace('USDT', '')
            
            # Hole aktuellen Kontostand und aktuellen Preis
            usdt_balance = self.get_account_balance('USDT')
            current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            
            # Berechne die Ordergröße basierend auf Kontostand, Signalstärke und Preis
            # Stärkere Signale führen zu größeren Orders (bis zu 5% des Kontostands)
            order_size_percentage = min(1.0, strength) / 100  # Bei Stärke 1.0 = 1%, bei Stärke 5.0 = 5%
            order_value = usdt_balance * order_size_percentage
            
            # Hole Symbol-Filter
            filters = self.get_symbol_filters(symbol)
            min_notional = filters['MIN_NOTIONAL']['min_notional']
            min_qty = filters['LOT_SIZE']['min_qty']
            step_size = filters['LOT_SIZE']['step_size']
            
            # Stelle sicher, dass die Order über dem Mindestvolumen liegt
            if order_value < min_notional:
                if min_notional > usdt_balance * 0.1:  # Wenn Mindestvolumen > 10% des Kontostands
                    logger.warning(f"Nicht genügend USDT für Mindestordervolumen: {min_notional} USDT erforderlich")
                    return None
                order_value = min_notional  # Setze auf Mindestvolumen
            
            # Berechne die Menge basierend auf USDT-Wert und aktuellem Preis
            quantity = order_value / current_price
            
            # Stelle sicher, dass die Menge über der Mindestmenge liegt
            if quantity < min_qty:
                logger.warning(f"Berechnete Menge {quantity} ist unter Mindestmenge {min_qty}")
                quantity = min_qty
            
            # Runde auf gültige Schrittgröße
            quantity = self.round_step_size(quantity, step_size)
            
            logger.info(f"Führe BUY für {symbol} aus: {quantity} (Wert: {quantity * current_price:.2f} USDT)")
            
            # Platziere die Market-Order
            order = self.place_market_order(symbol=symbol, side='BUY', quantity=quantity)
            
            if order:
                logger.info(f"BUY-Order erfolgreich ausgeführt: {order['orderId']}")
                
                # Berechne Stop-Loss-Preis (2-5% unter Kaufpreis, abhängig von Volatilität)
                stop_loss_percentage = 0.03  # 3% Stop-Loss
                stop_price = current_price * (1 - stop_loss_percentage)
                
                # Platziere Stop-Loss-Order
                stop_order = self.place_stop_loss_order(
                    symbol=symbol,
                    quantity=quantity,
                    stop_price=stop_price
                )
                
                if stop_order:
                    logger.info(f"Stop-Loss für {symbol} bei {stop_price} gesetzt")
                
                return order
            else:
                logger.error(f"Konnte BUY-Order für {symbol} nicht platzieren")
                return None
                
        except BinanceAPIException as e:
            logger.error(f"Binance API-Fehler beim Ausführen der BUY-Order: {e}")
            return None
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Ausführen der BUY-Order: {e}")
            return None
    
    def execute_sell_order(self, symbol: str, strength: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Führt eine Verkaufsorder basierend auf einem Trading-Signal aus.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            strength: Signalstärke (1.0-5.0), beeinflusst die Ordergröße
            
        Returns:
            Order-Informationen oder None bei Fehler
        """
        try:
            # Extrahiere das Basis-Asset aus dem Symbol
            base_asset = symbol.replace('USDT', '')
            
            # Hole aktuellen Asset-Kontostand
            asset_balance = self.get_account_balance(base_asset)
            
            if asset_balance <= 0:
                logger.warning(f"Kein {base_asset}-Guthaben zum Verkaufen vorhanden")
                return None
            
            # Bestimme die Verkaufsmenge basierend auf der Signalstärke
            # Stärkere Signale führen zu größeren Verkäufen
            sell_percentage = min(strength / 5.0, 1.0)  # Bei Stärke 5.0 = 100%, bei Stärke 1.0 = 20%
            quantity = asset_balance * sell_percentage
            
            # Hole Symbol-Filter
            filters = self.get_symbol_filters(symbol)
            min_qty = filters['LOT_SIZE']['min_qty']
            step_size = filters['LOT_SIZE']['step_size']
            
            # Stelle sicher, dass die Menge über der Mindestmenge liegt
            if quantity < min_qty:
                if min_qty > asset_balance:
                    logger.warning(f"Asset-Guthaben {asset_balance} ist unter Mindestmenge {min_qty}")
                    return None
                quantity = min_qty
            
            # Runde auf gültige Schrittgröße
            quantity = self.round_step_size(quantity, step_size)
            
            # Stelle sicher, dass wir nicht mehr verkaufen als wir haben
            quantity = min(quantity, asset_balance)
            
            if quantity <= 0:
                logger.warning(f"Berechnete Verkaufsmenge ist null oder negativ")
                return None
            
            # Hole aktuellen Preis für Log-Zwecke
            current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            logger.info(f"Führe SELL für {symbol} aus: {quantity} (Wert: {quantity * current_price:.2f} USDT)")
            
            # Platziere die Market-Order
            order = self.place_market_order(symbol=symbol, side='SELL', quantity=quantity)
            
            if order:
                logger.info(f"SELL-Order erfolgreich ausgeführt: {order['orderId']}")
                return order
            else:
                logger.error(f"Konnte SELL-Order für {symbol} nicht platzieren")
                return None
                
        except BinanceAPIException as e:
            logger.error(f"Binance API-Fehler beim Ausführen der SELL-Order: {e}")
            return None
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Ausführen der SELL-Order: {e}")
            return None
