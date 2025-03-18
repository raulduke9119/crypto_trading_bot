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
        self.rsi_period = 7  # Kurzperioden-RSI für schnellere Reaktion auf DOGEBTC-Volatilität
        self.rsi_ultra_period = 2  # Ultra-Kurzperiode für schnelle HF-Signale
        self.rsi_oversold = 30  # Überverkauft-Schwellenwert
        self.rsi_overbought = 70  # Überkauft-Schwellenwert
        
        # MACD-Parameter
        self.macd_fast = 6  # Schneller EMA (5-7 empfohlen für 5m)
        self.macd_slow = 22  # Langsamer EMA (20-25 empfohlen für 5m)
        self.macd_signal = 4  # Signal-Linie (3-5 empfohlen für 5m)
        
        # Bollinger Bands-Parameter
        self.bb_period = 20  # 20 ist empfohlen für 5m
        self.bb_std = 1.8  # Standardabweichung zwischen 1.5-2.0 empfohlen für HFT
        
        # Stochastik-Parameter für DOGEBTC
        self.stoch_k = 8  # %K Periode (kürzer für HFT)
        self.stoch_d = 3  # %D Periode
        self.stoch_smooth = 3  # Glättungsperiode
        
        # Volume Filter Parameter
        self.volume_spike_threshold = 1.5  # Volumenanstieg-Schwellenwert (50% über Durchschnitt)
        self.volume_trend_threshold = 1.2  # 20% Anstieg als Trend
        
        logger.info(f"DOGEBTC HF-Strategie initialisiert: RSI={self.rsi_period}/{self.rsi_ultra_period}, "
                   f"MACD={self.macd_fast}/{self.macd_slow}/{self.macd_signal}, BB={self.bb_period}/{self.bb_std}")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generiert Kauf- und Verkaufssignale basierend auf technischen Indikatoren
        optimiert für DOGEBTC im 5-Minuten-Intervall.
        
        Args:
            df: DataFrame mit OHLCV-Daten und technischen Indikatoren
            
        Returns:
            DataFrame mit hinzugefügten Signal-Spalten
        """
        try:
            if df.empty:
                logger.warning("Leeres DataFrame, keine Signale generiert")
                return df
            
            # Kopiere DataFrame, um Original nicht zu verändern
            data = df.copy()
            
            # Stelle sicher, dass alle benötigten Indikatoren vorhanden sind
            required_columns = ['close', 'high', 'low', 'volume']
            if not all(col in data.columns for col in required_columns):
                logger.warning(f"Fehlende erforderliche Spalten in Daten. Verfügbar: {data.columns}")
                return df
            
            # --- 1. DOGEBTC-spezifische Indikatoren hinzufügen ---
            
            # 1.1 Optimierter RSI (Kurzperiode für HF)
            if 'rsi' not in data.columns:
                ti = TechnicalIndicators()
                data = ti.add_rsi(data, self.rsi_period)
            
            # 1.2 Ultra-kurzer RSI für schnellere Reaktion
            data = TechnicalIndicators.add_rsi(data, period=self.rsi_ultra_period, column_name='rsi_ultra')
            
            # 1.3 Optimierter MACD mit angepassten Parametern für DOGEBTC
            data = TechnicalIndicators.add_macd(
                data, 
                fast_period=self.macd_fast, 
                slow_period=self.macd_slow, 
                signal_period=self.macd_signal
            )
            
            # 1.4 Optimierte Bollinger Bands
            bb_columns = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct_b']
            if not all(col in data.columns for col in bb_columns):
                data = TechnicalIndicators.add_bollinger_bands(data, period=self.bb_period, std_dev=self.bb_std)
            
            # 1.5 Stochastischer Oszillator für DOGEBTC optimiert
            stoch_columns = ['stoch_k', 'stoch_d']
            if not all(col in data.columns for col in stoch_columns):
                data = TechnicalIndicators.add_stochastic(
                    data, 
                    k_period=self.stoch_k, 
                    d_period=self.stoch_d, 
                    smooth_k=self.stoch_smooth
                )
            
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
            
            # 2.5 Schwellenwerte verwenden (DOGEBTC-spezifisch)
            buy_threshold = 3.0  # Optimiert für mehr Signale
            sell_threshold = 3.5  # Etwas höher für besseren Schutz
            
            # Finale Signale setzen
            data['buy_signal'] = (data['buy_strength'] >= buy_threshold) & \
                                 (data['buy_strength'] > data['sell_strength'] * 1.2)
            
            data['sell_signal'] = (data['sell_strength'] >= sell_threshold) & \
                                  (data['sell_strength'] > data['buy_strength'] * 1.2)
            
            # 2.6 Zusätzliche direkte DOGEBTC-spezifische Signale
            direct_buy = (
                (bb_lower_touch & (data['rsi_ultra'] < 30) & volume_spike) |
                (macd_cross_up & (data['close'] > data['bb_lower']) & (data['rsi_ultra'] < 40)) |
                (stoch_oversold & stoch_cross_up & (data['rsi_ultra'] < 40) & volume_trend_up)
            )
            
            direct_sell = (
                (bb_upper_touch & (data['rsi_ultra'] > 70) & volume_spike) |
                (macd_cross_down & (data['close'] < data['bb_upper']) & (data['rsi_ultra'] > 60)) |
                (stoch_overbought & stoch_cross_down & (data['rsi_ultra'] > 60) & volume_trend_up)
            )
            
            # Kombiniere generierte Signale mit direkten Signalen
            data['buy_signal'] = data['buy_signal'] | direct_buy
            data['sell_signal'] = data['sell_signal'] | direct_sell
            
            return data
            
        except Exception as e:
            logger.error(f"Fehler bei der Signalgenerierung in DOGEBTC HF-Strategie: {e}")
            # Rückgabe des ursprünglichen DataFrame bei Fehler
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
    
    def should_sell(self, data: pd.DataFrame, position: Optional[Dict]) -> Tuple[bool, float]:
        """
        Entscheidet, ob verkauft werden soll, basierend auf den generierten Signalen
        oder Schutzmaßnahmen wie Trailing-Stop oder Max-Drawdown.
        
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
        
        # Basis-Verkaufssignal
        should_sell = last_row.get('sell_signal', False)
        signal_strength = last_row.get('sell_strength', 0.0)
        
        # Erweiterte Schutzmaßnahmen
        
        # 1. Trailing-Stop-Loss (für Gewinnmitnahme)
        if 'price' in position and 'highest_price' in position:
            current_price = last_row['close']
            entry_price = position['price']
            highest_price = position['highest_price']
            
            # Wenn Preis um x% unter Höchstpreis gefallen ist
            trailing_stop_triggered = (
                current_price < highest_price * (1 - self.trailing_stop_percent / 100) and
                current_price > entry_price  # Nur wenn Position im Gewinn ist
            )
            
            if trailing_stop_triggered:
                logger.info(f"Trailing-Stop ausgelöst: Aktuell={current_price}, "
                           f"Höchst={highest_price}, Differenz={((current_price/highest_price)-1)*100:.2f}%")
                should_sell = True
                signal_strength = max(signal_strength, 8.0)  # Hohe Signal-Priorität
        
        # 2. Maximaler Drawdown-Schutz (für Verlustbegrenzung)
        if 'price' in position:
            current_price = last_row['close']
            entry_price = position['price']
            
            # Wenn Preis um x% unter Einstiegspreis gefallen ist
            max_drawdown_triggered = (
                current_price < entry_price * (1 - self.max_drawdown_percent / 100)
            )
            
            if max_drawdown_triggered:
                logger.info(f"Max-Drawdown ausgelöst: Aktuell={current_price}, "
                           f"Einstieg={entry_price}, Differenz={((current_price/entry_price)-1)*100:.2f}%")
                should_sell = True
                signal_strength = max(signal_strength, 9.0)  # Höchste Signal-Priorität
        
        return should_sell, signal_strength
    
    def calculate_position_size(self, data: pd.DataFrame, capital: float) -> float:
        """
        Berechnet die optimale Positionsgröße basierend auf Risiko und Volatilität.
        
        Args:
            data: DataFrame mit technischen Indikatoren
            capital: Verfügbares Kapital
            
        Returns:
            Optimale Positionsgröße in Prozent des Kapitals
        """
        if data.empty:
            return capital * (self.risk_percent / 100)
        
        position_size = capital * (self.risk_percent / 100)
        
        # Volatilitätsanpassung wenn gewünscht
        if self.adjust_for_volatility and 'atr_pct' in data.columns:
            latest_volatility = data['atr_pct'].iloc[-1]
            
            # Vergleichen mit 20-Perioden-Durchschnitt
            avg_volatility = data['atr_pct'].rolling(window=20).mean().iloc[-1]
            
            if not np.isnan(latest_volatility) and not np.isnan(avg_volatility) and avg_volatility > 0:
                # Reduziert Position bei höherer Volatilität, erhöht bei niedrigerer
                volatility_ratio = latest_volatility / avg_volatility
                
                # Begrenzen auf 0.5x - 1.5x des Standard-Risikos
                volatility_factor = max(0.5, min(1.5, 1 / volatility_ratio))
                position_size *= volatility_factor
                
                logger.debug(f"Volatilitätsanpassung: Faktor={volatility_factor:.2f}, "
                            f"Aktuelle ATR={latest_volatility:.2f}%, Durchschnitt={avg_volatility:.2f}%")
        
        return position_size

    def update_position(self, position: Dict, current_data: pd.DataFrame) -> Dict:
        """
        Aktualisiert die Positionsinformationen mit den neuesten Daten.
        
        Args:
            position: Aktuelle Position
            current_data: Aktuelle Marktdaten
            
        Returns:
            Aktualisierte Position
        """
        if current_data.empty:
            return position
        
        current_price = current_data['close'].iloc[-1]
        
        # Aktualisiere Höchstpreis für Trailing-Stop
        if 'highest_price' in position:
            position['highest_price'] = max(position['highest_price'], current_price)
        else:
            position['highest_price'] = current_price
        
        # Aktualisiere aktuelle Kennzahlen
        position['current_price'] = current_price
        
        if 'price' in position and position['price'] > 0:
            position['current_pnl_pct'] = ((current_price / position['price']) - 1) * 100
        
        return position
