"""
Basis-Strategie-Modul für den Trading Bot.
Definiert die Grundstruktur für alle Trading-Strategien.
"""
import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from config.config import LOG_LEVEL, LOG_FILE, DATA_DIRECTORY

# Logger einrichten
logger = setup_logger(os.path.join(DATA_DIRECTORY, LOG_FILE), LOG_LEVEL)

class BaseStrategy(ABC):
    """
    Abstrakte Basisklasse für alle Trading-Strategien.
    """
    
    def __init__(self, risk_percentage: float = 2.0, max_positions: int = 3):
        """
        Initialisiert die Basisstrategie.
        
        Args:
            risk_percentage: Prozentsatz des Kapitals, der pro Trade riskiert werden soll
            max_positions: Maximale Anzahl gleichzeitiger Positionen
        """
        self.risk_percentage = risk_percentage
        self.max_positions = max_positions
        self.current_positions = {}
        logger.info(f"Basisstrategie initialisiert: Risiko={risk_percentage}%, Max Positionen={max_positions}")
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generiert Kauf-/Verkaufssignale basierend auf technischen Indikatoren.
        Muss von abgeleiteten Klassen implementiert werden.
        
        Args:
            df: DataFrame mit Preis- und Indikatordaten
            
        Returns:
            DataFrame mit hinzugefügten Signalspalten
        """
        pass
    
    def calculate_position_size(self, 
                               price: float, 
                               stop_loss_price: float, 
                               account_balance: float) -> float:
        """
        Berechnet die Positionsgröße basierend auf dem Risikoprozentsatz.
        
        Args:
            price: Aktueller Preis des Assets
            stop_loss_price: Stop-Loss-Preis
            account_balance: Gesamtes Kontoguthaben
            
        Returns:
            Zu kaufende/verkaufende Menge
        """
        try:
            # Berechne das Dollarrisiko (wie viel Geld wir riskieren wollen)
            dollar_risk = account_balance * (self.risk_percentage / 100)
            
            # Berechne das Preisrisiko (Differenz zwischen Einstieg und Stop-Loss)
            price_risk = abs(price - stop_loss_price)
            
            # Stelle sicher, dass price_risk nicht Null ist
            if price_risk == 0 or price_risk < 0.000001:
                price_risk = price * 0.01  # Standard 1%, wenn kein Preisrisiko
            
            # Berechne Positionsgröße
            position_size = dollar_risk / price_risk
            
            # Berechne Positionsgröße in Einheiten des Basisassets
            quantity = position_size / price
            
            logger.debug(f"Positionsgröße berechnet: {quantity} Einheiten bei Preis {price}")
            return quantity
            
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung der Positionsgröße: {e}")
            return 0
    
    def get_stop_loss_price(self, 
                           df: pd.DataFrame, 
                           current_price: float, 
                           is_long: bool) -> float:
        """
        Berechnet den Stop-Loss-Preis basierend auf ATR oder Support-/Resistance-Levels.
        
        Args:
            df: DataFrame mit Indikatoren
            current_price: Aktueller Asset-Preis
            is_long: Boolean, True für Long-Positionen, False für Short
            
        Returns:
            Stop-Loss-Preis
        """
        try:
            # Hole den aktuellsten ATR-Wert
            atr = df['atr'].iloc[-1]
            
            # Setze Stop-Loss-Distanz als 2x ATR
            atr_multiplier = 2.0
            
            if is_long:
                # Für Long-Positionen, setze Stop-Loss unter dem aktuellen Preis
                stop_loss = current_price - (atr * atr_multiplier)
            else:
                # Für Short-Positionen, setze Stop-Loss über dem aktuellen Preis
                stop_loss = current_price + (atr * atr_multiplier)
            
            logger.debug(f"Stop-Loss berechnet: {stop_loss} für {'Long' if is_long else 'Short'}-Position")
            return stop_loss
            
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung des Stop-Loss: {e}")
            # Standard 5% vom aktuellen Preis
            return current_price * 0.95 if is_long else current_price * 1.05
    
    def calculate_take_profit(self, 
                             entry_price: float, 
                             stop_loss_price: float, 
                             risk_reward_ratio: float = 2.0) -> float:
        """
        Berechnet den Take-Profit-Preis basierend auf dem Risiko-Ertrags-Verhältnis.
        
        Args:
            entry_price: Eintrittspreis
            stop_loss_price: Stop-Loss-Preis
            risk_reward_ratio: Gewünschtes Risiko-Ertrags-Verhältnis
            
        Returns:
            Take-Profit-Preis
        """
        try:
            # Berechne Risiko (Differenz zwischen Einstieg und Stop-Loss)
            risk = abs(entry_price - stop_loss_price)
            
            # Berechne Take-Profit basierend auf Risiko-Ertrags-Verhältnis
            if entry_price > stop_loss_price:  # Long-Position
                take_profit = entry_price + (risk * risk_reward_ratio)
            else:  # Short-Position
                take_profit = entry_price - (risk * risk_reward_ratio)
            
            logger.debug(f"Take-Profit berechnet: {take_profit} mit RRR {risk_reward_ratio}")
            return take_profit
            
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung des Take-Profit: {e}")
            # Standard 10% vom Eintrittspreis für Long, -10% für Short
            return entry_price * 1.1 if entry_price > stop_loss_price else entry_price * 0.9
