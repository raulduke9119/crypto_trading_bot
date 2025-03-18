"""
Multi-Indikator-Strategie für den Trading Bot.
Kombiniert mehrere technische Indikatoren für Handelsentscheidungen.
"""
import os
import pandas as pd
import numpy as np
import importlib.util
import operator
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.base_strategy import BaseStrategy
from utils.logger import setup_logger
from utils.pattern_loader import PatternLoader
from config.config import LOG_LEVEL, LOG_FILE, DATA_DIRECTORY

# Logger einrichten
logger = setup_logger(os.path.join(DATA_DIRECTORY, LOG_FILE), LOG_LEVEL)

class MultiIndicatorStrategy(BaseStrategy):
    """
    Strategie, die mehrere technische Indikatoren kombiniert, um Handelssignale zu generieren.
    Unterstützt flexible Trading-Pattern über die PatternLoader-Komponente.
    """
    
    def __init__(self, 
                risk_percentage: float = 1.0,  # Reduziertes Risiko pro Trade
                max_positions: int = 2,        # Weniger gleichzeitige Positionen
                use_ml_predictions: bool = False,
                volatility_adjustment: bool = True,
                trailing_stop_pct: float = 1.5,
                max_drawdown_pct: float = 5.0,
                pattern_name: str = "default_pattern.json"):
        """
        Initialisiert die Multi-Indikator-Strategie mit verbessertem Risikomanagement.
        
        Args:
            risk_percentage: Prozentsatz des Kapitals, der pro Trade riskiert werden soll
            max_positions: Maximale Anzahl gleichzeitiger Positionen
            use_ml_predictions: Ob ML-Vorhersagen in die Strategie einbezogen werden sollen
            volatility_adjustment: Ob Volatilitätsanpassung aktiviert werden soll
            trailing_stop_pct: Prozentsatz für Trailing-Stop-Loss
            max_drawdown_pct: Maximaler erlaubter Drawdown in Prozent
            pattern_name: Name der zu verwendenden Pattern-Datei (in patterns/)
        """
        super().__init__(risk_percentage, max_positions)
        self.use_ml_predictions = use_ml_predictions
        self.volatility_adjustment = volatility_adjustment
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.current_drawdown = 0.0
        self.peak_value = 0.0
        
        # Optimierte Parameter basierend auf Backtesting-Recherche
        self.rsi_short_period = 2         # Kurzer RSI für Mean-Reversion (Forschungsergebnis)
        self.rsi_buy_threshold = 15       # RSI-Kaufsignal bei niedrigem Wert
        self.rsi_sell_threshold = 85      # RSI-Verkaufssignal bei hohem Wert
        self.macd_min_strength = 0.1      # Mindesthöhe des MACD-Histogramms
        self.bb_squeeze_factor = 1.5      # Faktor für BB-Squeeze-Erkennung
        self.volume_confirm_ratio = 1.2   # Volumen-Bestätigungsverhältnis
        
        # Pattern Loader initialisieren und Pattern laden
        self.pattern_loader = PatternLoader()
        self.pattern_name = pattern_name
        
        # Versuche, das angegebene Pattern zu laden oder Fallback zum Standard-Pattern
        self.pattern = self.pattern_loader.load_pattern(pattern_name)
        if self.pattern is None:
            logger.warning(f"Pattern '{pattern_name}' konnte nicht geladen werden. Verwende Standard-Pattern")
            self.pattern = self.pattern_loader.load_default_pattern()
            self.pattern_name = "default_pattern.json"
        
        # Schwellenwerte aus dem Pattern extrahieren
        self.signal_threshold = self.pattern.get("signal_threshold", 4.0)
        
        # Operatormapping für Bedingungsprüfungen
        self.operator_map = {
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne
        }
        
        logger.info(f"Multi-Indikator-Strategie initialisiert: "
                   f"ML-Vorhersagen={use_ml_predictions}, "
                   f"Volatilitätsanpassung={volatility_adjustment}, "
                   f"Trailing-Stop={trailing_stop_pct}%, "
                   f"Max-Drawdown={max_drawdown_pct}%, "
                   f"Pattern={self.pattern_name}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf dem geladenen Trading-Pattern.
        
        Args:
            df: DataFrame mit technischen Indikatoren
            
        Returns:
            DataFrame mit hinzugefügten Signalspalten
        """
        try:
            # Prüfe, ob der DataFrame leer ist
            if df.empty:
                logger.warning("DataFrame ist leer, keine Signale generiert")
                return df
            
            # Erstelle Kopie des DataFrames
            data = df.copy()
            
            # Initialisiere Signalspalten
            data['buy_signal'] = False
            data['sell_signal'] = False
            data['buy_strength'] = 0.0
            data['sell_strength'] = 0.0
            
            # Vorverarbeitung für Indikatorvergleiche (z.B. MACD_hist_prev)
            data = self._preprocess_dataframe(data)
            
            # Wende das Pattern an, um die Signalstärken zu berechnen
            self._apply_pattern_conditions(data)
            
            # Generiere finale Kauf- und Verkaufssignale basierend auf der Signalstärke
            data['buy_signal'] = data['buy_strength'] >= self.signal_threshold
            data['sell_signal'] = data['sell_strength'] >= self.signal_threshold
            
            # Keine Kaufsignale, wenn gleichzeitig auch ein Verkaufssignal vorliegt
            data.loc[data['sell_signal'], 'buy_signal'] = False
            
            # Logge Anzahl der generierten Signale
            num_signals = data['buy_signal'].sum() + data['sell_signal'].sum()
            logger.info(f"Hochfrequenz-Handelssignale generiert: {num_signals} Signale.")
            
            return data
            
        except Exception as e:
            logger.error(f"Fehler bei der Signalgenerierung: {e}")
            return df
    
    def _preprocess_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Bereitet den DataFrame für die Pattern-Auswertung vor, indem zusätzliche
        verschobene Spalten für Vergleiche eingefügt werden.
        
        Args:
            data: Der zu verarbeitende DataFrame
            
        Returns:
            DataFrame mit hinzugefügten Spalten
        """
        # Erstelle eine Kopie, um Fragmentierung zu vermeiden
        data_copy = data.copy()
        
        # Dictionary zum Sammeln aller neuen Spalten
        new_columns = {}
        
        # Füge verschobene Versionen wichtiger Indikatoren hinzu
        for col in data.columns:
            # Überspringe bereits vorhandene verschobene Spalten
            if col.endswith('_prev'):
                continue
                
            # Erstelle verschobene Versionen der Spalten
            new_columns[f'{col}_prev'] = data_copy[col].shift(1)
            new_columns[f'{col}_prev2'] = data_copy[col].shift(2)
        
        # Berechne MA-Versionen von Spalten, die häufig benötigt werden
        key_columns = ['volume', 'atr', 'rsi', 'macd_hist', 'bb_width']
        for col in key_columns:
            if col in data_copy.columns:
                # MA mit verschiedenen Zeitfenstern
                for period in [5, 10, 20]:
                    new_columns[f'{col}_ma{period}'] = data_copy[col].rolling(window=period).mean()
        
        # Berechne Min/Max über Zeitfenster
        for col in ['high', 'low', 'close']:
            if col in data_copy.columns:
                # Min/Max mit verschiedenen Zeitfenstern
                for period in [10, 20, 50]:
                    new_columns[f'{col}_min{period}'] = data_copy[col].rolling(window=period).min()
                    new_columns[f'{col}_max{period}'] = data_copy[col].rolling(window=period).max()
        
        # Füge alle neuen Spalten auf einmal hinzu
        if new_columns:
            result = pd.concat([data_copy, pd.DataFrame(new_columns, index=data_copy.index)], axis=1)
            return result
        
        return data_copy
    
    def _apply_pattern_conditions(self, data: pd.DataFrame) -> None:
        """
        Wendet die Bedingungen aus dem Pattern an, um Kauf- und Verkaufssignalstärken zu berechnen.
        
        Args:
            data: Der DataFrame, auf den die Pattern-Bedingungen angewendet werden sollen
        """
        try:
            # --- 1. Kaufbedingungen anwenden ---
            if "buy_conditions" in self.pattern:
                buy_strength = pd.Series(0.0, index=data.index)
                
                # Gesamtgewicht für die Normalisierung
                total_buy_weight = sum(cond.get("weight", 1.0) for cond in self.pattern["buy_conditions"])
                
                # Jede Kaufbedingung auswerten
                for condition_group in self.pattern["buy_conditions"]:
                    conditions = condition_group.get("conditions", [])
                    weight = condition_group.get("weight", 1.0)
                    
                    # Erstelle eine Maske, die True ist, wenn alle Unterbedingungen erfüllt sind
                    mask = pd.Series(True, index=data.index)
                    
                    # Jede einzelne Bedingung prüfen
                    for subcond in conditions:
                        submask = self._evaluate_condition(data, subcond)
                        mask = mask & submask
                    
                    # Addiere das Gewicht zur Signalstärke, wo die Bedingung erfüllt ist
                    buy_strength = buy_strength + mask.astype(float) * weight
                
                # Normalisiere die Signalstärke (0-10 Skala)
                data['buy_strength'] = (buy_strength / total_buy_weight * 10).round(2)
            
            # --- 2. Verkaufsbedingungen anwenden ---
            if "sell_conditions" in self.pattern:
                sell_strength = pd.Series(0.0, index=data.index)
                
                # Gesamtgewicht für die Normalisierung
                total_sell_weight = sum(cond.get("weight", 1.0) for cond in self.pattern["sell_conditions"])
                
                # Jede Verkaufsbedingung auswerten
                for condition_group in self.pattern["sell_conditions"]:
                    conditions = condition_group.get("conditions", [])
                    weight = condition_group.get("weight", 1.0)
                    
                    # Erstelle eine Maske, die True ist, wenn alle Unterbedingungen erfüllt sind
                    mask = pd.Series(True, index=data.index)
                    
                    # Jede einzelne Bedingung prüfen
                    for subcond in conditions:
                        submask = self._evaluate_condition(data, subcond)
                        mask = mask & submask
                    
                    # Addiere das Gewicht zur Signalstärke, wo die Bedingung erfüllt ist
                    sell_strength = sell_strength + mask.astype(float) * weight
                
                # Normalisiere die Signalstärke (0-10 Skala)
                data['sell_strength'] = (sell_strength / total_sell_weight * 10).round(2)
                
        except Exception as e:
            logger.error(f"Fehler beim Anwenden der Pattern-Bedingungen: {e}")
    
    def _evaluate_condition(self, data: pd.DataFrame, condition: Dict[str, Any]) -> pd.Series:
        """
        Wertet eine einzelne Bedingung aus dem Pattern aus.
        
        Args:
            data: Der DataFrame mit den Indikatoren
            condition: Die zu prüfende Bedingung
            
        Returns:
            Eine boolesche Series, die angibt, wo die Bedingung erfüllt ist
        """
        try:
            indicator = condition["indicator"]
            op = self.operator_map[condition["operator"]]
            
            # Prüfe, ob der Indikator im DataFrame vorhanden ist
            if indicator not in data.columns:
                logger.warning(f"Indikator '{indicator}' nicht im DataFrame vorhanden")
                return pd.Series(False, index=data.index)
            
            left_operand = data[indicator]
            
            # Bestimme den rechten Operanden basierend auf den verschiedenen Bedingungstypen
            if "value" in condition:
                # Direkter Wertvergleich (z.B. RSI < 30)
                right_operand = condition["value"]
                
            elif "value_indicator" in condition:
                # Vergleich mit einem anderen Indikator (z.B. close < bb_upper)
                value_indicator = condition["value_indicator"]
                if value_indicator not in data.columns:
                    logger.warning(f"Vergleichsindikator '{value_indicator}' nicht im DataFrame vorhanden")
                    return pd.Series(False, index=data.index)
                right_operand = data[value_indicator]
                
            elif "indicator_prev" in condition:
                # Vergleich mit verschobener Version des gleichen Indikators (z.B. close > close_prev)
                prev_indicator = condition["indicator_prev"]
                if prev_indicator not in data.columns and f"{prev_indicator}_prev" in data.columns:
                    prev_indicator = f"{prev_indicator}_prev"
                elif prev_indicator not in data.columns:
                    logger.warning(f"Vorheriger Indikator '{prev_indicator}' nicht im DataFrame vorhanden")
                    return pd.Series(False, index=data.index)
                right_operand = data[prev_indicator]
                
            elif "indicator_prev2" in condition:
                # Vergleich mit zweifach verschobener Version (z.B. close > close_prev2)
                prev2_indicator = condition["indicator_prev2"]
                if prev2_indicator not in data.columns and f"{prev2_indicator}_prev2" in data.columns:
                    prev2_indicator = f"{prev2_indicator}_prev2"
                elif prev2_indicator not in data.columns:
                    logger.warning(f"Vorheriger Indikator (2) '{prev2_indicator}' nicht im DataFrame vorhanden")
                    return pd.Series(False, index=data.index)
                right_operand = data[prev2_indicator]
                
            elif "indicator_ma" in condition:
                # Vergleich mit gleitendem Durchschnitt (z.B. volume > volume_ma10)
                ma_indicator = condition["indicator_ma"]
                ma_period = condition.get("ma_period", 20)
                
                ma_col = f"{ma_indicator}_ma{ma_period}"
                if ma_col not in data.columns:
                    # Berechne MA, wenn noch nicht vorhanden
                    data[ma_col] = data[ma_indicator].rolling(window=ma_period).mean()
                
                right_operand = data[ma_col]
                
                # Optionaler Faktor für den Vergleich (z.B. volume > volume_ma10 * 1.5)
                if "factor" in condition:
                    right_operand = right_operand * condition["factor"]
                
            elif "indicator_max" in condition:
                # Vergleich mit Maximum über einen Zeitraum (z.B. close > high_max20)
                max_indicator = condition["indicator_max"]
                period = condition.get("period", 20)
                offset = condition.get("offset", 0)
                
                max_col = f"{max_indicator}_max{period}"
                if max_col not in data.columns:
                    # Berechne Max, wenn noch nicht vorhanden
                    data[max_col] = data[max_indicator].rolling(window=period).max().shift(offset)
                
                right_operand = data[max_col]
                
                # Optionaler Faktor für den Vergleich
                if "factor" in condition:
                    right_operand = right_operand * condition["factor"]
                
            elif "indicator_min" in condition:
                # Vergleich mit Minimum über einen Zeitraum (z.B. close < low_min20)
                min_indicator = condition["indicator_min"]
                period = condition.get("period", 20)
                offset = condition.get("offset", 0)
                
                min_col = f"{min_indicator}_min{period}"
                if min_col not in data.columns:
                    # Berechne Min, wenn noch nicht vorhanden
                    data[min_col] = data[min_indicator].rolling(window=period).min().shift(offset)
                
                right_operand = data[min_col]
                
                # Optionaler Faktor für den Vergleich
                if "factor" in condition:
                    right_operand = right_operand * condition["factor"]
                
            else:
                logger.warning(f"Unbekannter Bedingungstyp: {condition}")
                return pd.Series(False, index=data.index)
            
            # Führe den Vergleich durch und gib das Ergebnis zurück
            return op(left_operand, right_operand)
            
        except Exception as e:
            logger.error(f"Fehler bei der Auswertung einer Bedingung: {e} - {condition}")
            return pd.Series(False, index=data.index)
    
    def should_buy(self, df: pd.DataFrame, position: Optional[Dict] = None) -> Tuple[bool, float]:
        """
        Entscheidet, ob gekauft werden soll, basierend auf den generierten Signalen.
        
        Args:
            df: DataFrame mit Signalen
            position: Aktuell gehaltene Position (falls vorhanden)
            
        Returns:
            Tuple aus (Kaufen-ja/nein, Signalstärke)
        """
        try:
            # Keine Käufe, wenn bereits eine Position besteht
            if position is not None:
                return False, 0.0
            
            # Bei leerem DataFrame nicht kaufen
            if df.empty:
                return False, 0.0
            
            # Verwende den letzten Datenpunkt
            last_data = df.iloc[-1]
            
            # Prüfe, ob ein Kaufsignal vorliegt
            should_buy = bool(last_data.get('buy_signal', False))
            
            # Signalstärke zwischen 0 und 10
            signal_strength = float(last_data.get('buy_strength', 0.0))
            
            return should_buy, signal_strength
            
        except Exception as e:
            logger.error(f"Fehler bei der Kaufentscheidung: {e}")
            return False, 0.0
    
    def should_sell(self, df: pd.DataFrame, position: Optional[Dict] = None) -> Tuple[bool, float]:
        """
        Entscheidet, ob verkauft werden soll, basierend auf den generierten Signalen.
        
        Args:
            df: DataFrame mit Signalen
            position: Aktuell gehaltene Position
            
        Returns:
            Tuple aus (Verkaufen-ja/nein, Signalstärke)
        """
        try:
            # Keine Position zum Verkaufen
            if position is None:
                return False, 0.0
            
            # Bei leerem DataFrame nicht verkaufen
            if df.empty:
                return False, 0.0
            
            # Verwende den letzten Datenpunkt
            last_data = df.iloc[-1]
            
            # Prüfe, ob ein Verkaufssignal vorliegt
            should_sell = bool(last_data.get('sell_signal', False))
            
            # Signalstärke zwischen 0 und 10
            signal_strength = float(last_data.get('sell_strength', 0.0))
            
            # Trailing Stop-Loss für Gewinnmitnahme
            if position and 'entry_price' in position and 'current_price' in position:
                entry_price = position['entry_price']
                current_price = position['current_price']
                
                # Wenn Position im Gewinn ist
                if current_price > entry_price:
                    # Berechne maximalen Gewinn
                    if 'highest_price' in position:
                        highest_price = position['highest_price']
                        
                        # Wenn Preis vom Höchststand um x% gefallen ist
                        trailing_stop_price = highest_price * (1 - self.trailing_stop_pct / 100)
                        if current_price <= trailing_stop_price:
                            logger.info(f"Trailing-Stop ausgelöst: {current_price} <= {trailing_stop_price}")
                            should_sell = True
                            signal_strength = 10.0  # Höchste Priorität
            
            # Schutz vor zu großem Drawdown
            if position and 'entry_price' in position and 'current_price' in position:
                entry_price = position['entry_price']
                current_price = position['current_price']
                
                # Berechne aktuellen Verlust in Prozent
                loss_pct = (entry_price - current_price) / entry_price * 100
                
                # Wenn Verlust größer als erlaubter Drawdown ist
                if loss_pct >= self.max_drawdown_pct:
                    logger.info(f"Max-Drawdown-Schutz ausgelöst: {loss_pct:.2f}% Verlust")
                    should_sell = True
                    signal_strength = 10.0  # Höchste Priorität
            
            return should_sell, signal_strength
            
        except Exception as e:
            logger.error(f"Fehler bei der Verkaufsentscheidung: {e}")
            return False, 0.0

    def evaluate_position(self, position: Dict, current_data: pd.Series) -> Dict:
        """
        Evaluiert eine bestehende Position und aktualisiert deren Status.
        
        Args:
            position: Dictionary mit Positionsdaten
            current_data: Aktuelle Marktdaten als pandas Series
            
        Returns:
            Aktualisiertes Positions-Dictionary
        """
        try:
            # Wenn keine Position existiert, gib ein leeres Ergebnis zurück
            if not position:
                return {'should_close': False}

            if not current_data.any():
                return {'should_close': False}

            # Aktualisiere Positionswerte
            current_price = float(current_data['close'])
            position['current_price'] = current_price
            
            # Aktualisiere Höchstpreis für Trailing-Stop
            if 'highest_price' not in position:
                position['highest_price'] = current_price
            elif current_price > position['highest_price']:
                position['highest_price'] = current_price
            
            # Berechne aktuelle P&L
            entry_price = float(position['entry_price'])
            quantity = float(position['size'])
            position['unrealized_pnl'] = (current_price - entry_price) * quantity
            position['unrealized_pnl_pct'] = (current_price - entry_price) / entry_price * 100
            
            # Prüfe Trailing-Stop
            if position['highest_price'] > entry_price:
                trailing_stop_price = position['highest_price'] * (1 - self.trailing_stop_pct / 100)
                position['trailing_stop'] = trailing_stop_price
                
                if current_price <= trailing_stop_price:
                    position['should_close'] = True
                    position['close_reason'] = 'trailing_stop'
                    return position
            
            # Prüfe maximalen Drawdown
            if position['unrealized_pnl_pct'] <= -self.max_drawdown_pct:
                position['should_close'] = True
                position['close_reason'] = 'max_drawdown'
                return position
            
            # Prüfe Verkaufssignal
            should_sell, sell_strength = self.should_sell(pd.DataFrame([current_data]), position)
            if should_sell:
                position['should_close'] = True
                position['close_reason'] = 'sell_signal'
                position['signal_strength'] = sell_strength
                return position
            
            # Position bleibt offen
            position['should_close'] = False
            return position
            
        except Exception as e:
            logger.error(f"Fehler bei der Positionsevaluierung: {e}")
            return {'should_close': False, 'error': str(e)}
