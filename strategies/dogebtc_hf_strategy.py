"""
Spezialisierte High-Frequency Strategie für DOGEBTC mit 5-Minuten-Intervall.
Optimiert basierend auf spezifischen Eigenschaften und Volatilitätsmustern von DOGE.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime

from strategies.base_strategy import BaseStrategy
from data.indicators import TechnicalIndicators

logger = logging.getLogger('trading_bot')

class DogebtcHFStrategy(BaseStrategy):
    """
    DOGEBTC High-Frequency Trading Strategie optimiert für 5-Minuten-Intervalle.
    Nutzt speziell angepasste Parameter basierend auf DOGEBTC-spezifischen Eigenschaften
    und kombiniert mehrere technische Indikatoren für präzisere Kauf- und Verkaufssignale.
    """

    def __init__(self, 
                 risk_percent: float = 1.0, 
                 max_positions: int = 2,
                 use_ml_predictions: bool = False,
                 adjust_for_volatility: bool = True,
                 trailing_stop_percent: float = 1.5,
                 max_drawdown_percent: float = 5.0,
                 **kwargs):
        """
        Initialisiert die DOGEBTC HF-Strategie mit optimierten Parametern.
        
        Args:
            risk_percent: Risiko pro Trade in Prozent
            max_positions: Maximale Anzahl gleichzeitiger Positionen
            use_ml_predictions: ML-Vorhersagen verwenden (wenn verfügbar)
            adjust_for_volatility: Position-Größe an Volatilität anpassen
            trailing_stop_percent: Trailing-Stop-Prozentsatz
            max_drawdown_percent: Maximaler Drawdown für Positionen
        """
        super().__init__(risk_percent, max_positions)
        
        # Allgemeine Strategie-Einstellungen
        self.use_ml_predictions = use_ml_predictions
        self.adjust_for_volatility = adjust_for_volatility
        self.trailing_stop_percent = trailing_stop_percent
        self.max_drawdown_percent = max_drawdown_percent
        
        # DOGEBTC-spezifische optimierte Parameter (basierend auf Perplexity-Empfehlungen)
        # RSI-Parameter
        self.rsi_period = 7
        self.rsi_overbought = 72
        self.rsi_oversold = 28
        self.rsi_ultra_period = 2  # Sehr kurzperiodiger RSI für Mean-Reversion
        
        # MACD-Parameter 
        self.macd_fast = 6
        self.macd_slow = 13
        self.macd_signal = 4
        
        # Bollinger-Band-Parameter
        self.bb_period = 12
        self.bb_std = 2.2
        
        # Stochastik-Parameter
        self.stoch_k = 5 
        self.stoch_d = 3
        self.stoch_smooth = 2
        
        # Volumen-Parameter
        self.volume_spike_threshold = 1.8
        
        # Trading-Parameter
        self.pattern_recognition = True
        self.mean_reversion_mode = True
        
        logger.info(f"DOGEBTC HF-Strategie initialisiert: Risiko={risk_percent}%, Volatilitätsanpassung={adjust_for_volatility}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generiert Trading-Signale basierend auf mehreren Indikatoren,
        speziell optimiert für DOGEBTC.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            DataFrame mit hinzugefügten Signalspalten
        """
        try:
            # Validiere Eingabedaten
            if df.empty:
                logger.warning("DataFrame ist leer, keine Signale generiert")
                return df
            
            # Stelle sicher, dass wir mit einer Kopie arbeiten
            data = df.copy()
            
            # --- 1. Stelle sicher, dass alle benötigten Indikatoren vorhanden sind ---
            
            # Initialisiere TechnicalIndicators, falls fehlende Indikatoren berechnet werden müssen
            indicators = TechnicalIndicators()
            
            # 1.1 RSI überprüfen und berechnen wenn nötig
            if 'rsi' not in data.columns:
                data = indicators.add_rsi(data)
                
            # Füge Ultra-Short RSI hinzu (für Mean-Reversion)
            if 'rsi_ultra' not in data.columns:
                # Erstelle eine temporäre RSI-Berechnung mit kurzem Zeitfenster
                close_series = data['close'].copy()
                delta = close_series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_ultra_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_ultra_period).mean()
                rs = gain / loss.replace(0, float('nan'))  # Vermeide Division durch Null
                data['rsi_ultra'] = 100 - (100 / (1 + rs))
                data['rsi_ultra'] = data['rsi_ultra'].fillna(50)  # Neutrale RSI-Werte für NaN
            
            # 1.2 MACD überprüfen und berechnen wenn nötig
            macd_columns = ['macd', 'macd_signal', 'macd_hist']
            if not all(col in data.columns for col in macd_columns):
                data = indicators.add_macd(data)
            
            # 1.3 Bollinger Bands überprüfen und berechnen wenn nötig
            bb_columns = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width']
            if not all(col in data.columns for col in bb_columns):
                data = indicators.add_bollinger_bands(data)
            
            # 1.5 Stochastischer Oszillator für DOGEBTC optimiert
            stoch_columns = ['stoch_k', 'stoch_d']
            if not all(col in data.columns for col in stoch_columns):
                data = indicators.add_stochastic(data)
            
            # --- 2. DOGEBTC-spezifische Signale generieren ---
            
            # Initialisiere Signal-Spalten
            data['buy_signal'] = False
            data['sell_signal'] = False
            data['buy_strength'] = 0.0
            data['sell_strength'] = 0.0
            
            # 2.1 Volume-basierte Signalfilter (kritisch für DOGEBTC)
            data['volume_sma_5'] = data['volume'].rolling(window=5).mean()
            data['volume_ratio_5'] = data['volume'] / data['volume_sma_5'].replace(0, 0.001)
            
            # Volume-Spike-Indikatoren
            volume_spike = data['volume_ratio_5'] > self.volume_spike_threshold
            volume_trend_up = (data['volume'] > data['volume'].shift(1)) & \
                             (data['volume'].shift(1) > data['volume'].shift(2))
            
            # 2.2 Muster definieren
            
            # RSI-Muster (optimiert für DOGEBTC)
            rsi_oversold = data['rsi_ultra'] < self.rsi_oversold
            rsi_overbought = data['rsi_ultra'] > self.rsi_overbought
            rsi_recovering = (data['rsi_ultra'] > data['rsi_ultra'].shift(1)) & (data['rsi_ultra'].shift(1) < self.rsi_oversold)
            rsi_falling = (data['rsi_ultra'] < data['rsi_ultra'].shift(1)) & (data['rsi_ultra'].shift(1) > self.rsi_overbought)
            
            # MACD-Muster
            macd_cross_up = (data['macd_hist'] > 0) & (data['macd_hist'].shift(1) <= 0)
            macd_cross_down = (data['macd_hist'] < 0) & (data['macd_hist'].shift(1) >= 0)
            macd_turning_up = (data['macd_hist'] > data['macd_hist'].shift(1)) & (data['macd_hist'].shift(1) > data['macd_hist'].shift(2))
            macd_turning_down = (data['macd_hist'] < data['macd_hist'].shift(1)) & (data['macd_hist'].shift(1) < data['macd_hist'].shift(2))
            
            # Bollinger Bands Muster
            bb_lower_touch = data['close'] <= data['bb_lower']
            bb_upper_touch = data['close'] >= data['bb_upper']
            bb_squeeze = data['bb_width'] < data['bb_width'].rolling(window=20).mean() * 0.8
            
            # Stochastik-Muster
            stoch_oversold = data['stoch_k'] < 20
            stoch_overbought = data['stoch_k'] > 80
            stoch_cross_up = (data['stoch_k'] > data['stoch_d']) & (data['stoch_k'].shift(1) <= data['stoch_d'].shift(1))
            stoch_cross_down = (data['stoch_k'] < data['stoch_d']) & (data['stoch_k'].shift(1) >= data['stoch_d'].shift(1))
            
            # 2.3 DOGEBTC-spezifische Signallogik
            
            # Kaufsignale - DOGEBTC-spezifisch optimiert
            buy_conditions = [
                # RSI-Signale
                (rsi_oversold & volume_spike, 3.0),  # Starkes Signal mit Volumenbestätigung
                (rsi_recovering & volume_trend_up, 2.5),  # Erholt sich mit Volumentrend
                
                # MACD-Signale
                (macd_cross_up & (data['rsi_ultra'] < 50) & volume_spike, 3.0),  # MACD-Kreuzung mit RSI & Volumen
                (macd_turning_up & (data['close'] > data['bb_middle']) & volume_trend_up, 2.0),  # MACD-Wende mit BB
                
                # Bollinger Bands-Signale
                (bb_lower_touch & (data['rsi_ultra'] < 40) & volume_spike, 3.0),  # BB-Berührung mit RSI & Volumen
                (bb_squeeze & macd_cross_up & volume_trend_up, 2.5),  # BB-Squeeze mit MACD & Volumen
                
                # Stochastik-Signale
                (stoch_oversold & stoch_cross_up & (data['close'] > data['bb_lower']), 2.0),  # Stoch-Kreuzung mit BB
                
                # Kombinierte Signale
                ((data['close'] > data['bb_lower']) & rsi_oversold & stoch_cross_up & volume_spike, 3.5)  # Multi-Indikator
            ]
            
            # Verkaufssignale - DOGEBTC-spezifisch optimiert
            sell_conditions = [
                # RSI-Signale
                (rsi_overbought & volume_spike, 3.0),  # Starkes Signal mit Volumenbestätigung
                (rsi_falling & volume_trend_up, 2.5),  # Fällt mit Volumentrend
                
                # MACD-Signale
                (macd_cross_down & (data['rsi_ultra'] > 50) & volume_spike, 3.0),  # MACD-Kreuzung mit RSI & Volumen
                (macd_turning_down & (data['close'] < data['bb_middle']) & volume_trend_up, 2.0),  # MACD-Wende mit BB
                
                # Bollinger Bands-Signale
                (bb_upper_touch & (data['rsi_ultra'] > 60) & volume_spike, 3.0),  # BB-Berührung mit RSI & Volumen
                (data['close'] > data['bb_upper'] * 1.01, 2.0),  # Übertreibung über BB
                
                # Stochastik-Signale
                (stoch_overbought & stoch_cross_down & (data['close'] < data['bb_upper']), 2.0),  # Stoch-Kreuzung mit BB
                
                # Kombinierte Signale
                ((data['close'] < data['bb_upper']) & rsi_overbought & stoch_cross_down & volume_spike, 3.5)  # Multi-Indikator
            ]
            
            # 2.4 Signal-Stärken berechnen
            buy_strength = pd.Series(0.0, index=data.index)
            sell_strength = pd.Series(0.0, index=data.index)
            
            for condition, weight in buy_conditions:
                buy_strength += condition.astype(float) * weight
            
            for condition, weight in sell_conditions:
                sell_strength += condition.astype(float) * weight
            
            # Normalisieren
            max_buy_weight = sum(weight for _, weight in buy_conditions)
            max_sell_weight = sum(weight for _, weight in sell_conditions)
            
            data['buy_strength'] = (buy_strength / max_buy_weight * 10).round(2)
            data['sell_strength'] = (sell_strength / max_sell_weight * 10).round(2)
            
            # 2.5 Generiere finale Signale basierend auf Schwellenwerten
            signal_threshold = 4.0  # Höherer Schwellenwert für präzisere Signale
            
            # Kaufsignale mit höherer Präzision
            data['buy_signal'] = data['buy_strength'] >= signal_threshold
            
            # Verkaufssignale mit höherer Präzision
            data['sell_signal'] = data['sell_strength'] >= signal_threshold
            
            # 2.6 Verkaufssignale haben Priorität über Kaufsignale
            data.loc[data['sell_signal'], 'buy_signal'] = False
            
            logger.debug(f"DOGEBTC Signale generiert: {data['buy_signal'].sum()} Kaufsignale, {data['sell_signal'].sum()} Verkaufssignale")
            return data
            
        except Exception as e:
            logger.error(f"Fehler bei der Signalgenerierung: {e}")
            return df
    
    def should_buy(self, data: pd.DataFrame, position: Optional[Dict] = None) -> Tuple[bool, float]:
        """
        Entscheidet, ob gekauft werden soll, basierend auf den generierten Signalen.
        
        Args:
            data: DataFrame mit technischen Indikatoren und Signalen
            position: Aktuelle Position (falls vorhanden)
            
        Returns:
            Tuple aus (Kaufen-ja/nein, Signalstärke)
        """
        if data.empty:
            return False, 0.0
        
        # Letzten Datenpunkt verwenden
        last_row = data.iloc[-1]
        
        # Prüfe, ob ein Kaufsignal vorliegt
        should_buy = last_row.get('buy_signal', False)
        signal_strength = last_row.get('buy_strength', 0.0)
        
        # Keine wiederholten Käufe, wenn bereits eine Position besteht
        if position is not None:
            should_buy = False
        
        return should_buy, signal_strength
    
    def should_sell(self, data: pd.DataFrame, position: Optional[Dict] = None) -> Tuple[bool, float]:
        """
        Entscheidet, ob verkauft werden soll, basierend auf den generierten Signalen.
        
        Args:
            data: DataFrame mit technischen Indikatoren und Signalen
            position: Aktuelle Position (falls vorhanden)
            
        Returns:
            Tuple aus (Verkaufen-ja/nein, Signalstärke)
        """
        if data.empty or position is None:
            return False, 0.0
        
        # Letzten Datenpunkt verwenden
        last_row = data.iloc[-1]
        
        # Prüfe, ob ein Verkaufssignal vorliegt
        should_sell = last_row.get('sell_signal', False)
        signal_strength = last_row.get('sell_strength', 0.0)
        
        # Trailing-Stop-Loss-Logik für Gewinnmitnahme und Verlustbegrenzung
        if position and 'entry_price' in position:
            entry_price = position['entry_price']
            current_price = last_row['close']
            
            # Berechne prozentuale Veränderung seit Einstieg
            pct_change = (current_price - entry_price) / entry_price * 100
            
            # Trailing-Stop-Loss: Verkaufe, wenn der Preis um x% vom Höchststand gefallen ist
            if 'highest_price' in position and position['highest_price'] > entry_price:
                trailing_stop_price = position['highest_price'] * (1 - self.trailing_stop_percent / 100)
                if current_price <= trailing_stop_price:
                    logger.info(f"Trailing-Stop ausgelöst: Aktuell {current_price}, Stop bei {trailing_stop_price:.2f}")
                    should_sell = True
                    signal_strength = 10.0  # Höchste Priorität für Stop-Loss
            
            # Maximaler Drawdown-Schutz
            if pct_change < -self.max_drawdown_percent:
                logger.info(f"Max-Drawdown-Schutz ausgelöst: {pct_change:.2f}% Verlust")
                should_sell = True
                signal_strength = 10.0  # Höchste Priorität für Drawdown-Schutz
        
        return should_sell, signal_strength
    
    def calculate_position_size(self, 
                               balance: float, 
                               price: float, 
                               risk_pct: Optional[float] = None, 
                               volatility: Optional[float] = None) -> float:
        """
        Berechnet die optimale Positionsgröße basierend auf Risiko und optionaler Volatilitätsanpassung.
        
        Args:
            balance: Verfügbares Kapital
            price: Aktueller Preis
            risk_pct: Risikoprozentsatz (überschreibt den Standard)
            volatility: Volatilität für Positionsgrößenanpassung
            
        Returns:
            Positionsgröße (Anzahl Coins)
        """
        # Verwende den angegebenen Risikoprozentsatz oder den Standardwert
        risk_percentage = risk_pct if risk_pct is not None else self.risk_percentage
        
        # Basisberechnung der Positionsgröße
        position_size = (balance * risk_percentage / 100) / price
        
        # Volatilitätsanpassung, wenn aktiviert und Volatilität angegeben
        if self.adjust_for_volatility and volatility is not None and volatility > 0:
            # Anpassungsfaktor: Bei höherer Volatilität kleinere Position
            volatility_factor = 1 / (1 + volatility)
            position_size *= volatility_factor
            logger.debug(f"Positionsgröße volatilitätsangepasst: Faktor {volatility_factor:.2f}")
        
        return position_size
    
    def update_position(self, position: Dict, current_data: pd.Series) -> Dict:
        """
        Aktualisiert die Positionsdaten für Trailing-Stop und andere Mechanismen.
        
        Args:
            position: Aktuelle Positionsdaten
            current_data: Aktuelle Marktdaten
            
        Returns:
            Aktualisierte Positionsdaten
        """
        if not position:
            return position
        
        # Aktualisiere höchsten Preis für Trailing-Stop-Loss
        current_price = current_data['close']
        if 'highest_price' not in position or current_price > position['highest_price']:
            position['highest_price'] = current_price
            
        return position
