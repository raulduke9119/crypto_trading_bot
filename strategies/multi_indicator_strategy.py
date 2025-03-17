"""
Multi-Indikator-Strategie für den Trading Bot.
Kombiniert mehrere technische Indikatoren für Handelsentscheidungen.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.base_strategy import BaseStrategy
from utils.logger import setup_logger
from config.config import LOG_LEVEL, LOG_FILE, DATA_DIRECTORY

# Logger einrichten
logger = setup_logger(os.path.join(DATA_DIRECTORY, LOG_FILE), LOG_LEVEL)

class MultiIndicatorStrategy(BaseStrategy):
    """
    Strategie, die mehrere technische Indikatoren kombiniert, um Handelssignale zu generieren.
    """
    
    def __init__(self, 
                risk_percentage: float = 1.0,  # Reduziertes Risiko pro Trade
                max_positions: int = 2,        # Weniger gleichzeitige Positionen
                use_ml_predictions: bool = False,
                volatility_adjustment: bool = True,
                trailing_stop_pct: float = 1.5,
                max_drawdown_pct: float = 5.0):
        """
        Initialisiert die Multi-Indikator-Strategie mit verbessertem Risikomanagement.
        
        Args:
            risk_percentage: Prozentsatz des Kapitals, der pro Trade riskiert werden soll
            max_positions: Maximale Anzahl gleichzeitiger Positionen
            use_ml_predictions: Ob ML-Vorhersagen in die Strategie einbezogen werden sollen
            volatility_adjustment: Ob Volatilitätsanpassung aktiviert werden soll
            trailing_stop_pct: Prozentsatz für Trailing-Stop-Loss
            max_drawdown_pct: Maximaler erlaubter Drawdown in Prozent
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
        
        logger.info(f"Multi-Indikator-Strategie initialisiert: "
                   f"ML-Vorhersagen={use_ml_predictions}, "
                   f"Volatilitätsanpassung={volatility_adjustment}, "
                   f"Trailing-Stop={trailing_stop_pct}%, "
                   f"Max-Drawdown={max_drawdown_pct}%, "
                   f"RSI-Kurzperiode={self.rsi_short_period}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generiert Kauf-/Verkaufssignale basierend auf mehreren technischen Indikatoren.
        Implementiert eine profitable Kombination aus kurzen RSI, MACD-Histogramm und Bollinger Bands.
        
        Args:
            df: DataFrame mit Preis- und Indikatordaten
            
        Returns:
            DataFrame mit hinzugefügten Signalspalten
        """
        try:
            # Erstelle eine Kopie des DataFrames
            data = df.copy()
            
            # Initialisiere Signalspalten
            data['signal'] = 0  # 1 für Kauf, -1 für Verkauf, 0 für Halten
            data['signal_strength'] = 0  # Signalstärke von 0 bis 100
            
            # ======= Optimierte Strategie-Logik basierend auf Backtesting-Recherche =======
            
            # 1. Kurzfristiger RSI (2-Perioden) für schnelle Mean-Reversion
            # 2. MACD-Histogramm für Trendbestätigung mit 4-Tages-Abfall
            # 3. Bollinger-Band-Ausbrüche als primäre Kauf-/Verkaufssignale
            # 4. Volumenbestätigung für alle Signale
            # 5. Überdurchschnittliches Volumen als Signalverstärker
            
            # Validiere erforderliche Spalten und setze sichere Standardwerte ein
            required_columns = ['macd_hist', 'rsi', 'close', 'sma_medium', 'stoch_k', 
                              'stoch_d', 'bb_upper', 'bb_lower', 'bb_middle', 'volume', 'ema_short']
            
            # Sicherstellen, dass alle erforderlichen Spalten existieren
            for col in required_columns:
                if col not in data.columns:
                    logger.error(f"Erforderliche Spalte {col} nicht gefunden")
                    return df
            
            # Sichere Behandlung von None/NaN-Werten für alle kritischen Berechnungen
            for col in required_columns:
                if data[col].isnull().any():
                    logger.warning(f"Null-Werte in {col} gefunden, führe Bereinigung durch")
                    # Verwende explizite Methoden statt method=ffill wegen FutureWarning
                    data[col] = data[col].ffill().bfill()
                    
            # Berechne kurzfristigen RSI (2-Perioden) für schnellere Mean-Reversion-Signale
            # Basierend auf Forschungsergebnissen arbeitet 2-Perioden RSI besser für kurzfristige Signale
            if 'close' in data.columns:
                try:
                    delta = data['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=self.rsi_short_period).mean()
                    avg_loss = loss.rolling(window=self.rsi_short_period).mean()
                    rs = avg_gain / avg_loss.replace(0, 1e-10)  # Vermeidet Division durch Null
                    data['rsi_short'] = 100 - (100 / (1 + rs))
                except Exception as e:
                    logger.error(f"Fehler bei RSI-Kurzberechnung: {e}")
                    data['rsi_short'] = data['rsi']  # Fallback auf Standard-RSI
            
            # Berechne optimierte Kaufbedingungen basierend auf Backtesting-Ergebnissen
            try:
                # === Strategie 1: Kurzer RSI (2 Perioden) mit überverkauftem Zustand ===
                # Research zeigt, dass 2-Perioden RSI bei Werten unter 15 starke Kaufsignale gibt
                rsi_short_oversold = data['rsi_short'] < self.rsi_buy_threshold  # RSI(2) < 15
                
                # === Strategie 2: MACD-Histogramm mit 4-Tage-Abfall + Umkehrung ===
                # MACD-Histogramm muss 4 Tage in Folge gefallen sein und der 4. Tag unter Null
                macd_hist_falling = True
                for i in range(1, 5):
                    if i < len(data) and i > 0:
                        # Prüfe, ob das MACD-Histogramm in den letzten 4 Tagen gefallen ist
                        if i < 4:
                            macd_hist_falling &= data['macd_hist'].shift(i-1) < data['macd_hist'].shift(i)
                        # Der 4. Tag muss unter Null sein
                        elif i == 4:
                            macd_hist_falling &= data['macd_hist'].shift(i-1) < 0
                
                # Aktuelle Umkehr - MACD-Histogramm steigt wieder
                macd_reversal = data['macd_hist'] > data['macd_hist'].shift(1)
                macd_buy_signal = macd_hist_falling & macd_reversal
                
                # === Strategie 3: Bollinger Band Ausbruch und Mean-Reversion ===
                # Kauf wenn Preis unter unteres Bollinger Band fällt und dann darüber schließt
                bollinger_oversold = data['close'].shift(1) < data['bb_lower'].shift(1) & \
                                    data['close'] > data['bb_lower']
                
                # Zusätzlich: Bollinger Band Squeeze für volatile Breakout-Phasen erkennen
                # Squeeze = wenn die Bänder enger werden als normal (multipliziert mit Faktor)
                bb_bandwidth = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
                bb_bandwidth_avg = bb_bandwidth.rolling(window=20).mean()
                bb_squeeze_condition = bb_bandwidth < (bb_bandwidth_avg / self.bb_squeeze_factor)
                
                # === Strategie 4: Volumenvalidierung ===
                # Überdurchschnittliches Volumen als Bestätigung
                # Berechne Volumen-Ratio (aktuelles Volumen / durchschnittliches Volumen)
                if 'volume' in data.columns:
                    data['volume_sma'] = data['volume'].rolling(window=20).mean()
                    data['volume_ratio'] = data['volume'] / data['volume_sma']
                    high_volume = data['volume_ratio'] > self.volume_confirm_ratio
                else:
                    high_volume = pd.Series(True, index=data.index)  # Fallback wenn kein Volumen
                
                # Trend-Validierung für zusätzliche Bestätigung
                uptrend = data['close'] > data['sma_medium']
                
            except Exception as e:
                logger.error(f"Fehler bei Kaufsignal-Berechnung: {e}")
                return df
                
            # Berechne optimierte Verkaufsbedingungen basierend auf Backtesting-Ergebnissen
            try:
                # === Strategie 1: Kurzer RSI (2 Perioden) mit überkauftem Zustand ===
                # Research zeigt, dass 2-Perioden RSI bei Werten über 85 starke Verkaufssignale gibt
                rsi_short_overbought = data['rsi_short'] > self.rsi_sell_threshold  # RSI(2) > 85
                
                # === Strategie 2: MACD-Histogramm mit Umkehrung nach oben ===
                # MACD-Histogramm mit Trendumkehr für Verkaufssignale
                macd_high_reversal = (data['macd_hist'].shift(1) > data['macd_hist'].shift(2)) & \
                                  (data['macd_hist'] < data['macd_hist'].shift(1)) & \
                                  (data['macd_hist'] > 0)  # Histogramm noch positiv aber fallend
                
                # === Strategie 3: Bollinger Band Ausbruch und Mean-Reversion ===
                # Verkauf wenn Preis über oberes Bollinger Band steigt und dann darunter schließt
                bollinger_overbought = (data['close'].shift(1) > data['bb_upper'].shift(1)) & \
                                      (data['close'] < data['bb_upper'])
                
                # Zusätzlich: Trailing-Stop basierend auf ATR oder Bollinger-Untergrenze
                # Berechne Trailing-Stop als aktueller Preis minus X% oder BB Untergrenze
                if 'close' in data.columns:
                    trailing_stop = data['close'] * (1 - self.trailing_stop_pct/100)
                    # Für bereits gekaufte Positionen, wenn der Preis unter den Trailing-Stop fällt
                    trailing_stop_triggered = data['close'] < trailing_stop
                else:
                    trailing_stop_triggered = pd.Series(False, index=data.index)
                
                # === Strategie 4: Volumenvalidierung ===
                # Hoher Volumenanstieg bei Preisruckgang = möglicher Ausverkauf
                if 'volume' in data.columns and 'volume_ratio' in data.columns:
                    volume_spike = (data['volume_ratio'] > 1.5) & (data['close'] < data['close'].shift(1))
                else:
                    volume_spike = pd.Series(False, index=data.index)
                
                # Abwärtstrend als zusätzliche Bestätigung für Verkaufssignale
                downtrend = data['close'] < data['sma_medium']
                price_dropping = data['close'] < data['close'].shift(1)
                
            except Exception as e:
                logger.error(f"Fehler bei Verkaufssignal-Berechnung: {e}")
                return df
            
            # Kombiniere optimierte Kaufstrategien mit gewichteter Bewertung
            # 1. RSI(2) Strategie (Hohe Erfolgsrate in Backtests)
            # 2. MACD-Histogramm 4-Tage-Abfall + Umkehr (Hoher Profit-Faktor laut Tests)
            # 3. Bollinger Band Ausbruch (Effektiv für Mean-Reversion)
            # 4. Volumenvalidierung (Reduziert Fehlsignale)
            
            # Gewichtung basierend auf Backtesting-Ergebnissen
            # Die Forschung zeigt, dass RSI(2) + MACD die höchsten Erfolgsraten haben
            buy_conditions = {
                'rsi_short': (rsi_short_oversold, 3.0),           # Höchstes Gewicht (Backtesting-Erfolg)
                'macd_pattern': (macd_buy_signal, 2.5),           # Starkes Gewicht (guter Profit-Faktor)
                'bollinger': (bollinger_oversold, 1.5),           # Mittleres Gewicht
                'volume': (high_volume, 1.0),                     # Bestätigendes Gewicht
                'trend': (uptrend, 0.5),                          # Niedriges Gewicht
                'squeeze': (bb_squeeze_condition, 0.5)            # Niedriges Gewicht
            }
            
            # Volatilitätsanpassung für Signalschwellen - wichtig für Kryptowährungen
            # Bei hoher Volatilität strengere Kriterien anlegen
            if self.volatility_adjustment:
                if 'volatility' in data.columns:
                    vol_factor = 1 + (data['volatility'] - data['volatility'].rolling(20).mean()) / \
                                data['volatility'].rolling(20).std().replace(0, 1e-5)
                    vol_threshold = 0.5 + (vol_factor * 0.1).clip(-0.1, 0.2)
                elif 'atr' in data.columns and 'close' in data.columns:
                    # Alternative: Normalisiertes ATR verwenden wenn 'volatility' nicht verfügbar
                    vol_factor = data['atr'] / (data['close'].rolling(20).mean() * 0.02)
                    vol_threshold = 0.5 + (vol_factor * 0.1).clip(-0.1, 0.2)
                else:
                    vol_threshold = 0.5  # Standard ohne Volatilitätsdaten
            else:
                vol_threshold = 0.5  # Fester Wert wenn Volatililtätsanpassung deaktiviert
            
            # Berechne gewichtete Kaufsignalstärke mit sicherer Typkonvertierung
            buy_score = sum((cond.fillna(False)).astype(float) * weight 
                         for cond, weight in buy_conditions.values())
            total_buy_weight = sum(weight for _, weight in buy_conditions.values())
            buy_score_normalized = buy_score / total_buy_weight if total_buy_weight > 0 else 0
            
            # Hauptkaufsignal mit primären und sekundären Bedingungen
            # Primär: RSI(2) unter 15 ODER MACD-Histogramm-Muster - einer von beiden muss erfüllt sein
            primary_buy_condition = rsi_short_oversold | macd_buy_signal
            
            # Sekundär: Gewichtete Gesamtpunktzahl über volatilitätsangepasstem Schwellenwert
            secondary_buy_condition = buy_score_normalized >= vol_threshold
            
            # Gesamtes Kaufsignal benötigt primäre UND sekundäre Bedingungen
            buy_signal = primary_buy_condition & secondary_buy_condition
            
            # Kombiniere optimierte Verkaufsstrategien mit gewichteter Bewertung
            # 1. RSI(2) Strategie - Verkauf bei RSI > 85 (Backtesting-Erfolg)
            # 2. MACD-Histogramm Umkehr nach oben (bessere Exit-Timing)
            # 3. Bollinger Band Ausbruch über oberes Band + Umkehr (Effektiv für Mean-Reversion)
            # 4. Trailing-Stop basierend auf Einstiegspreis (Gewinnabsicherung)
            # 5. Volumenvalidierung mit Spike-Erkennung (Reduziert Fehlsignale)
            
            # Gewichtung basierend auf Backtesting-Ergebnissen
            sell_conditions = {
                'rsi_short': (rsi_short_overbought, 3.0),        # Höchstes Gewicht (Backtesting-Erfolg)
                'macd_pattern': (macd_high_reversal, 2.0),       # Hohes Gewicht
                'bollinger': (bollinger_overbought, 1.5),        # Mittleres Gewicht
                'trailing_stop': (trailing_stop_triggered, 3.5),  # Höchstes Gewicht (Risikomanagement!)
                'volume': (volume_spike, 1.0),                   # Bestätigendes Gewicht
                'trend': (downtrend & price_dropping, 0.5),      # Niedriges Gewicht
                'momentum': (data['close'].pct_change(3) < -0.02, 1.5)  # Starker Abwärtsdruck
            }
            
            # Berechne gewichtete Verkaufssignalstärke mit Drawdown-Schutz
            sell_score = sum(cond.astype(float) * weight for cond, weight in sell_conditions.values())
            total_sell_weight = sum(weight for _, weight in sell_conditions.values())
            
            # Aktualisiere Drawdown-Tracking
            current_value = data['close'].iloc[-1]
            self.peak_value = max(self.peak_value, current_value)
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value * 100
            
            # Verkaufssignal mit Drawdown-Schutz
            sell_signal = (sell_score >= (total_sell_weight * vol_threshold)) | \
                         (self.current_drawdown >= self.max_drawdown_pct)
            
            # Speichere Scores für spätere Analyse
            data['buy_score'] = buy_score / total_buy_weight * 100
            data['sell_score'] = sell_score / total_sell_weight * 100
            
            # Wende Signale an
            data.loc[buy_signal, 'signal'] = 1
            data.loc[sell_signal, 'signal'] = -1
            
            # Berechne Signalstärke mit Risikoanpassung
            try:
                strength_components = []
                
                # 1. RSI-Momentum und Trendstärke
                # Sicherstellen, dass rsi-Spalte existiert und keine NaN enthält
                if 'rsi' in data.columns and not data['rsi'].isna().all():
                    data['rsi'] = data['rsi'].fillna(50)  # Neutrale Ersetzung für NaN
                    rsi_momentum = data['rsi'].diff(3).fillna(0)
                    rsi_strength = ((50 - data['rsi']).abs() / 50) * \
                                  (1 + rsi_momentum.abs() / 30).clip(upper=2)
                    strength_components.append(rsi_strength)
                else:
                    # Fallback, wenn RSI nicht verfügbar ist
                    strength_components.append(pd.Series(1.0, index=data.index))
                
                # 2. MACD-Signalstärke und Divergenz
                if 'macd_hist' in data.columns and not data['macd_hist'].isna().all():
                    data['macd_hist'] = data['macd_hist'].fillna(0)  # Neutrale Ersetzung für NaN
                    close_mean = data['close'].mean()
                    if close_mean > 0:  # Vermeide Division durch 0
                        macd_strength = data['macd_hist'].abs() / close_mean * 100
                        price_momentum = data['close'].pct_change(3).fillna(0)
                        # Sichere Variante von np.sign mit NaN-Handling
                        macd_sign = data['macd_hist'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                        momentum_sign = price_momentum.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                        macd_divergence = (macd_sign != momentum_sign).astype(float) * 0.5
                        macd_score = macd_strength * (1 - macd_divergence)
                        strength_components.append(macd_score)
                    else:
                        strength_components.append(pd.Series(1.0, index=data.index))
                else:
                    # Fallback, wenn MACD nicht verfügbar ist
                    strength_components.append(pd.Series(1.0, index=data.index))
                
                # 3. Bollinger Band Position und Squeeze
                all_bb_cols = all(col in data.columns for col in ['bb_middle', 'bb_upper', 'bb_lower'])
                if all_bb_cols and not data[['bb_middle', 'bb_upper', 'bb_lower']].isna().all().any():
                    # Ersetze NaN-Werte in Bollinger-Band-Spalten
                    for col in ['bb_middle', 'bb_upper', 'bb_lower']:
                        data[col] = data[col].fillna(data['close'])
                    
                    # Verhindere Division durch Null
                    bb_range = data['bb_upper'] - data['bb_lower']
                    bb_range = bb_range.replace(0, 1e-5)  # Klein, aber nicht Null
                    
                    bb_position = (data['close'] - data['bb_middle']) / bb_range
                    # Begrenze extreme Werte
                    bb_position = bb_position.clip(-3, 3)  
                    
                    # Sichere Überprüfung von bb_squeeze (was immer das war im vorherigen Code)
                    bb_squeeze_factor = 1.5 if 'bb_squeeze' in locals() and bb_squeeze else 1.0
                    bb_strength = bb_position.abs() * bb_squeeze_factor
                    strength_components.append(bb_strength)
                else:
                    # Fallback, wenn BB nicht verfügbar ist
                    strength_components.append(pd.Series(1.0, index=data.index))
                
                # 4. Trendstärke und Konsistenz
                if all(col in data.columns for col in ['ema_short', 'sma_medium']):
                    # Sichere Ersetzung für NaN-Werte
                    data['ema_short'] = data['ema_short'].fillna(data['close'])
                    data['sma_medium'] = data['sma_medium'].fillna(data['close'])
                    
                    # Sichere Berechnung des MA-Spreads
                    ma_spread = (data['ema_short'] - data['sma_medium']).abs() / data['close'].replace(0, 1e-5) * 100
                    ma_spread = ma_spread.fillna(0).clip(0, 50)  # Begrenze extreme Werte
                    
                    # Sichere Berechnung der Trendkonsistenz
                    with np.errstate(invalid='ignore'):  # Ignoriere NumPy-Warnungen für NaN
                        trend_direction = (data['close'] > data['sma_medium']).astype(float).fillna(0.5)
                        trend_consistency = trend_direction.rolling(5, min_periods=1).mean().fillna(0.5)
                    
                    trend_score = ma_spread * trend_consistency
                    strength_components.append(trend_score)
                else:
                    # Fallback, wenn MA-Daten nicht verfügbar sind
                    strength_components.append(pd.Series(1.0, index=data.index))
                
                # 5. Volumen-Profil und Anomalien
                if 'volume_ratio' in data.columns:
                    # Sichere Verarbeitung des Volumen-Verhältnisses
                    volume_ratio = data['volume_ratio'].fillna(1.0).clip(upper=5) 
                    volume_trend = volume_ratio.diff(3).fillna(0).abs()
                    
                    # Begrenze die Werte, um Ausreißer zu vermeiden
                    volume_score = (volume_ratio * (1 + volume_trend)).clip(upper=3) / 3
                    strength_components.append(volume_score)
                else:
                    # Fallback, wenn Volume-Ratio nicht verfügbar ist
                    strength_components.append(pd.Series(1.0, index=data.index))
                
                # 6. Volatiltätsanpassung (wenn verfügbar)
                if 'volatility' in data.columns and not data['volatility'].isna().all():
                    # Sichere Berechnung der Volatilitätsanpassung
                    vol_mean = data['volatility'].mean()
                    vol_std = data['volatility'].std()
                    
                    # Vermeide Division durch Null und sichere gegen NaN
                    if vol_std > 0:
                        vol_adjustment = 1 + (data['volatility'] - vol_mean) / vol_std
                        vol_adjustment = vol_adjustment.fillna(1.0).clip(0.5, 2.0)  # Begrenze Extremwerte
                        strength_components = [comp * vol_adjustment for comp in strength_components]
                    # Kein else-Fall nötig, da wir die Komponenten unverändert lassen, wenn vol_std == 0
                
            except Exception as e:
                logger.error(f"Fehler bei Signalstärke-Berechnung: {e}")
                return df
            
            # Kombiniere alle Stärkekomponenten mit sicherer Aggregation
            if strength_components:
                # Ersetze NaN durch 0 in allen Komponenten
                clean_components = [comp.fillna(0) for comp in strength_components]
                data['signal_strength'] = sum(clean_components) / len(clean_components)
                
                # Normalisiere Stärke sicher auf 0-100 Skala
                max_strength = data['signal_strength'].max()
                if max_strength > 0:
                    data['signal_strength'] = data['signal_strength'] / max_strength * 100
                else:
                    data['signal_strength'] = 0  # Default, wenn kein Maximum gefunden wurde
            else:
                # Fallback, wenn keine Stärkekomponenten verfügbar sind
                data['signal_strength'] = 0
            
            # Sichere Berechnung der Konfidenzwerte
            buy_count_safe = 0 if buy_count is None else buy_count
            sell_count_safe = 0 if sell_count is None else sell_count
            buy_conditions_len = max(1, len(buy_conditions))  # Vermeide Division durch Null
            sell_conditions_len = max(1, len(sell_conditions)) # Vermeide Division durch Null
            
            data['buy_confidence'] = buy_count_safe / buy_conditions_len * 100
            data['sell_confidence'] = sell_count_safe / sell_conditions_len * 100
            
            # Stelle sicher, dass alle Signal-Spalten keine NaN-Werte enthalten
            for col in ['signal', 'signal_strength', 'buy_confidence', 'sell_confidence']:
                if col in data.columns:
                    data[col] = data[col].fillna(0)
            
            logger.info(f"Signale generiert: {len(data[data['signal'] != 0])} Signale gefunden")
            # Berechne finale Statistiken für Debugging und Analyse
            signal_counts = data['signal'].value_counts()
            buy_count = signal_counts.get(1, 0)
            sell_count = signal_counts.get(-1, 0)
            hold_count = signal_counts.get(0, 0)
            total_signals = buy_count + sell_count
            
            # Detailliertes Logging
            logger.info(f"Signal-Statistik: {buy_count} Kaufsignale, {sell_count} Verkaufssignale, "
                       f"{hold_count} Halten-Signale, Signalrate: {total_signals/len(data)*100:.2f}%")
            
            if 'buy_confidence' in data.columns and not data['buy_confidence'].isna().all():
                avg_buy_conf = data.loc[data['signal'] == 1, 'buy_confidence'].mean()
                logger.info(f"Durchschnittliche Kaufkonfidenz: {avg_buy_conf:.2f}%")
                
            if 'sell_confidence' in data.columns and not data['sell_confidence'].isna().all():
                avg_sell_conf = data.loc[data['signal'] == -1, 'sell_confidence'].mean()
                logger.info(f"Durchschnittliche Verkaufskonfidenz: {avg_sell_conf:.2f}%")
            
            return data
            
        except Exception as e:
            logger.error(f"Fehler beim Generieren von Handelssignalen: {e}")
            import traceback
            logger.error(traceback.format_exc())  # Vollständiger Stack-Trace für besseres Debugging
            return df
    
    def adjust_signals_with_ml(self, df: pd.DataFrame, ml_prediction: float) -> pd.DataFrame:
        """
        Passt die generierten Signale basierend auf ML-Vorhersagen und Marktbedingungen an.
        
        Args:
            df: DataFrame mit Signalen
            ml_prediction: ML-Vorhersage für zukünftige Preisbewegung (in Prozent)
            
        Returns:
            DataFrame mit angepassten Signalen
        """
        if not self.use_ml_predictions or ml_prediction is None:
            return df
        
        try:
            # Erstelle eine Kopie des DataFrames
            data = df.copy()
            
            # Berechne Marktstimmung und Volatilität
            market_trend = data['close'].pct_change(12).mean() * 100  # 12-Perioden-Trend
            volatility = data['close'].pct_change().std() * 100     # Aktuelle Volatilität
            avg_volatility = data['close'].pct_change().rolling(20).std().mean() * 100  # Durchschnittliche Volatilität
            
            # Volatilitätsbasierte Signalstärkenanpassung
            vol_ratio = volatility / avg_volatility
            vol_factor = np.clip(1 / vol_ratio, 0.5, 1.5)  # Reduziere Signalstärke bei hoher Volatilität
            
            # Für Kaufsignale
            buy_mask = data['signal'] > 0
            if ml_prediction > 0:
                # Positive Vorhersage verstärkt Kaufsignale, aber berücksichtigt Marktbedingungen
                signal_boost = (1 + abs(ml_prediction)) * vol_factor
                if market_trend < 0:  # Vorsichtiger bei negativem Markttrend
                    signal_boost *= 0.8
                data.loc[buy_mask, 'signal_strength'] *= signal_boost
            else:
                # Negative Vorhersage schwächt Kaufsignale ab
                signal_reduction = max(0.2, 1 - abs(ml_prediction) * vol_factor)
                data.loc[buy_mask, 'signal_strength'] *= signal_reduction
            
            # Für Verkaufssignale
            sell_mask = data['signal'] < 0
            if ml_prediction < 0:
                # Negative Vorhersage verstärkt Verkaufssignale
                signal_boost = (1 + abs(ml_prediction)) * vol_factor
                if market_trend > 0:  # Vorsichtiger bei positivem Markttrend
                    signal_boost *= 0.8
                data.loc[sell_mask, 'signal_strength'] *= signal_boost
            else:
                # Positive Vorhersage schwächt Verkaufssignale ab
                signal_reduction = max(0.2, 1 - abs(ml_prediction) * vol_factor)
                data.loc[sell_mask, 'signal_strength'] *= signal_reduction
            
            # Trailing-Stop Anpassung
            if self.trailing_stop_pct > 0:
                for i in range(len(data)):
                    if data.iloc[i]['signal'] > 0:  # Für Long-Positionen
                        trailing_stop = data.iloc[i]['close'] * (1 - self.trailing_stop_pct / 100)
                        if i > 0 and data.iloc[i-1]['close'] < trailing_stop:
                            data.iloc[i, data.columns.get_loc('signal')] = -1  # Verkaufssignal
                    
            # Füge ML-Vorhersage und Marktbedingungen zum DataFrame hinzu
            data['ml_prediction'] = ml_prediction
            data['market_trend'] = market_trend
            data['volatility_ratio'] = vol_ratio
            
            # Füge ML-Vorhersage zum DataFrame hinzu
            data['ml_prediction'] = ml_prediction
            
            logger.info(f"Signale mit ML-Vorhersage angepasst: Vorhersage={ml_prediction:.2%}")
            return data
            
        except Exception as e:
            logger.error(f"Fehler beim Anpassen der Signale mit ML: {e}")
            return df
    
    def evaluate_position(self, row: pd.Series) -> str:
        """
        Evaluiert eine einzelne Zeile von Marktdaten und gibt ein Handelssignal zurück.
        
        Args:
            row: Pandas Series mit technischen Indikatoren
            
        Returns:
            'BUY', 'SELL' oder 'HOLD'
        """
        try:
            # Prüfe, ob row eine gültige Series ist
            if not isinstance(row, pd.Series):
                logger.error(f"Ungültiger Datentyp für row: {type(row)}")
                return 'HOLD'
            
            # Prüfe, ob alle erforderlichen Indikatoren vorhanden sind
            required_indicators = [
                'macd', 'macd_signal', 'macd_hist',
                'rsi', 'sma_medium', 'stoch_k', 'stoch_d',
                'bb_pct_b', 'close'
            ]
            
            # Erstelle Dictionary für Indikatorwerte
            indicator_values = {}
            
            for indicator in required_indicators:
                # Prüfe ob Indikator existiert
                if indicator not in row.index:
                    logger.warning(f"Indikator {indicator} fehlt in den Daten")
                    return 'HOLD'
                
                # Hole Wert und prüfe auf None/NaN
                value = row[indicator]
                if value is None or pd.isna(value):
                    logger.warning(f"Indikator {indicator} ist None/NaN")
                    return 'HOLD'
                
                # Konvertiere zu Float
                try:
                    indicator_values[indicator] = float(value)
                except (ValueError, TypeError) as e:
                    logger.error(f"Konvertierungsfehler für {indicator}: {e}")
                    return 'HOLD'
            
            # Extrahiere Werte für bessere Lesbarkeit
            macd_hist = indicator_values['macd_hist']
            rsi = indicator_values['rsi']
            close = indicator_values['close']
            sma_medium = indicator_values['sma_medium']
            stoch_k = indicator_values['stoch_k']
            stoch_d = indicator_values['stoch_d']
            bb_pct_b = indicator_values['bb_pct_b']
            
            # Kaufbedingungen prüfen
            buy_conditions = [
                macd_hist > 0,  # MACD-Histogramm ist positiv
                40 < rsi < 70,   # RSI im gesunden Bereich
                close > sma_medium,  # Preis über SMA
                stoch_k > stoch_d,  # Stochastic K über D
                bb_pct_b > 0.5   # Preis über BB-Mittellinie
            ]
            
            # Verkaufbedingungen prüfen
            sell_conditions = [
                macd_hist < 0,  # MACD-Histogramm ist negativ
                rsi > 70 or rsi < 30,  # RSI überkauft/überverkauft
                close < sma_medium,  # Preis unter SMA
                stoch_k < stoch_d,  # Stochastic K unter D
                bb_pct_b < 0.5   # Preis unter BB-Mittellinie
            ]
            
            # Prüfe ob alle Bedingungen gültige Booleans sind
            for i, condition in enumerate(buy_conditions):
                if not isinstance(condition, bool):
                    logger.error(f"Ungültige Kaufbedingung {i}: {condition}")
                    return 'HOLD'
            
            for i, condition in enumerate(sell_conditions):
                if not isinstance(condition, bool):
                    logger.error(f"Ungültige Verkaufbedingung {i}: {condition}")
                    return 'HOLD'
            
            # Zähle erfüllte Bedingungen
            buy_count = sum(buy_conditions)
            sell_count = sum(sell_conditions)
            
            # Entscheidungslogik
            if buy_count >= 3:  # Mindestens 3 von 5 Kaufbedingungen erfüllt
                logger.debug(
                    f"BUY Signal für {close:.2f}: "
                    f"MACD={macd_hist:.2f}, RSI={rsi:.2f}, "
                    f"StochK/D={stoch_k:.2f}/{stoch_d:.2f}, BB%={bb_pct_b:.2f}"
                )
                return 'BUY'
            elif sell_count >= 3:  # Mindestens 3 von 5 Verkaufbedingungen erfüllt
                logger.debug(
                    f"SELL Signal für {close:.2f}: "
                    f"MACD={macd_hist:.2f}, RSI={rsi:.2f}, "
                    f"StochK/D={stoch_k:.2f}/{stoch_d:.2f}, BB%={bb_pct_b:.2f}"
                )
                return 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Fehler bei der Positionsevaluierung: {e}")
            return 'HOLD'
    
    def rank_opportunities(self, 
                          market_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, int, float]]:
        """
        Rankt potenzielle Handelsmöglichkeiten basierend auf Signalen und Stärke.
        
        Args:
            market_data: Dictionary mit Symbol als Schlüssel und DataFrame als Wert
            
        Returns:
            Liste von (Symbol, Signal, Stärke)-Tupeln, sortiert nach Stärke
        """
        try:
            opportunities = []
            
            for symbol, df in market_data.items():
                # Hole den neuesten Datenpunkt
                latest = df.iloc[-1]
                
                # Prüfe, ob es ein Handelssignal gibt
                signal = latest['signal']
                
                if signal != 0:  # Wenn es ein Kauf- oder Verkaufssignal gibt
                    strength = latest['signal_strength']
                    opportunities.append((symbol, int(signal), float(strength)))
            
            # Sortiere nach Stärke in absteigender Reihenfolge
            opportunities.sort(key=lambda x: x[2], reverse=True)
            
            logger.info(f"Handelsmöglichkeiten gerankt: {len(opportunities)} gefunden")
            return opportunities
            
        except Exception as e:
            logger.error(f"Fehler beim Ranken der Handelsmöglichkeiten: {e}")
            return []
