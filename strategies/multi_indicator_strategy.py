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
        Generiert Kauf-/Verkaufssignale für hochfrequenten Handel mit kleinen Gewinnen.
        Implementiert deterministische Muster und Scalping-Strategien für häufige Trades.
        
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
            
            # ======= Hoch-Frequenz Scalping-Strategie für kleine, häufige Trades =======
            
            # 1. Ultraschneller RSI (1-2 Perioden) für sofortige Umkehrsignale
            # 2. Mikroausbrüche aus kurzfristigen Price Channels
            # 3. Hochfrequenz-Bollinger-Band-Touches für Mean-Reversion
            # 4. Kurzzeitiges Momentum für schnelle Ein- und Ausstiege
            # 5. Deterministische Muster bei Kerzenumschwüngen
            
            # Validiere erforderliche Spalten und setze sichere Standardwerte ein
            required_columns = ['macd_hist', 'rsi', 'close', 'sma_medium', 'stoch_k', 
                              'stoch_d', 'bb_upper', 'bb_lower', 'bb_middle', 'volume', 'ema_short', 'high', 'low']
            
            # Sicherstellen, dass alle erforderlichen Spalten existieren
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                # Wenn high/low fehlen, aber close vorhanden ist, diese aus close erzeugen
                if 'close' in data.columns:
                    if 'high' in missing_columns:
                        data['high'] = data['close']
                    if 'low' in missing_columns:
                        data['low'] = data['close']
                    missing_columns = [col for col in missing_columns if col not in ['high', 'low']]
                
                if missing_columns:
                    logger.error(f"Erforderliche Spalten fehlen: {missing_columns}")
                    return df
            
            # Sichere Behandlung von None/NaN-Werten für alle kritischen Berechnungen
            for col in [c for c in required_columns if c in data.columns]:
                if data[col].isnull().any():
                    # Der Benutzer möchte die Warnungen ignorieren, also verwenden wir fillna mit method
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                    
            # ======= HOCHFREQUENZ-INDIKATOREN FÜR SCALPING =======
            
            # 1. Ultraschneller RSI (1-Periode) für sofortige Marktumkehrungen
            try:
                # Berechne EMA-basierten RSI mit sehr kurzer Periode (1) für maximale Empfindlichkeit
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                # Verwende EMA für noch schnellere Reaktion
                alpha = 0.5  # Hoher Alpha-Wert für schnellere Reaktion
                avg_gain = gain.ewm(alpha=alpha).mean()
                avg_loss = loss.ewm(alpha=alpha).mean()
                
                rs = avg_gain / avg_loss.replace(0, 1e-10)  # Vermeidet Division durch Null
                data['rsi_ultra'] = 100 - (100 / (1 + rs))
                
                # Berechne auch 2-Perioden RSI als Backup-Indikator
                avg_gain2 = gain.rolling(window=2).mean()
                avg_loss2 = loss.rolling(window=2).mean()
                rs2 = avg_gain2 / avg_loss2.replace(0, 1e-10)
                data['rsi_short'] = 100 - (100 / (1 + rs2))
                
                # RSI Divergence (Preisdivergenz) - wichtiger technischer Indikator
                price_higher = data['close'] > data['close'].shift(1)
                rsi_lower = data['rsi_ultra'] < data['rsi_ultra'].shift(1)
                bearish_divergence = price_higher & rsi_lower
                
                price_lower = data['close'] < data['close'].shift(1)
                rsi_higher = data['rsi_ultra'] > data['rsi_ultra'].shift(1)
                bullish_divergence = price_lower & rsi_higher
                
            except Exception as e:
                logger.error(f"Fehler bei Ultraschnell-RSI-Berechnung: {e}")
                data['rsi_ultra'] = data['rsi']
                data['rsi_short'] = data['rsi']
                bearish_divergence = pd.Series(False, index=data.index)
                bullish_divergence = pd.Series(False, index=data.index)
            
            # ======= DETERMINISTISCHE HANDELSMUSTER FÜR MICRO-SCALPING =======
            try:
                # === 2. KURZFRISTIGE PREISKANÄLE FÜR MIKROAUSBRÜCHE ===
                # Erstelle sehr enge Preiskanäle für Mikroausbrüche (3-5 Perioden)
                data['micro_highest_3'] = data['high'].rolling(window=3).max()
                data['micro_lowest_3'] = data['low'].rolling(window=3).min()
                data['micro_highest_5'] = data['high'].rolling(window=5).max()
                data['micro_lowest_5'] = data['low'].rolling(window=5).min()
                
                # Micro-Breakouts (sehr kurzfristige Ausbrüche für häufige Signals)
                micro_breakout_up = data['close'] > data['micro_highest_3'].shift(1)
                micro_breakout_down = data['close'] < data['micro_lowest_3'].shift(1)
                
                # Micro-Bounces (Abpraller an den Kanalgrenzwerten)
                micro_bounce_up = (data['low'] <= data['micro_lowest_5'].shift(1)) & \
                                  (data['close'] > data['low'] * 1.001)  # Mindestens 0,1% Abprall
                micro_bounce_down = (data['high'] >= data['micro_highest_5'].shift(1)) & \
                                    (data['close'] < data['high'] * 0.999)  # Mindestens 0,1% Abprall
                
                # === 3. MICRO-BOLLINGER-BAND EVENTS ===
                # Bollinger Band Touch Events (keine vollständigen Ausbrüche nötig)
                # Verwende kürzere Bollinger Perioden (10 statt 20) für mehr Signale
                if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
                    # BB-Touch statt Ausbruch (genauere Einstiege)
                    bb_touch_upper = (data['high'] >= data['bb_upper'] * 0.998) & \
                                      (data['close'] < data['bb_upper'] * 0.995)  # Touch und Rückzug
                    bb_touch_lower = (data['low'] <= data['bb_lower'] * 1.002) & \
                                      (data['close'] > data['bb_lower'] * 1.005)  # Touch und Abprall
                    
                    # Berechne engere Bollinger Bänder für noch mehr Signale
                    data['bb_middle_10'] = data['close'].rolling(window=10).mean()
                    data['bb_std_10'] = data['close'].rolling(window=10).std()
                    data['bb_upper_10'] = data['bb_middle_10'] + 1.5 * data['bb_std_10']
                    data['bb_lower_10'] = data['bb_middle_10'] - 1.5 * data['bb_std_10']
                    
                    # Ultra-kurzfristige BB-Berührungen
                    bb_touch_upper_10 = (data['high'] >= data['bb_upper_10']) & \
                                         (data['close'] < data['bb_upper_10'])
                    bb_touch_lower_10 = (data['low'] <= data['bb_lower_10']) & \
                                         (data['close'] > data['bb_lower_10'])
                else:
                    bb_touch_upper = pd.Series(False, index=data.index)
                    bb_touch_lower = pd.Series(False, index=data.index)
                    bb_touch_upper_10 = pd.Series(False, index=data.index)
                    bb_touch_lower_10 = pd.Series(False, index=data.index)
                
                # === 4. KURZZEITIGES MOMENTUM FÜR SCHNELLE TRADES ===
                # Berechne Micro-Momentum (2-Perioden)
                close_diff_pct = data['close'].pct_change()
                data['momentum_2'] = close_diff_pct + close_diff_pct.shift(1)
                strong_momentum_up = data['momentum_2'] > 0.005  # >0.5% in 2 Perioden
                strong_momentum_down = data['momentum_2'] < -0.005  # <-0.5% in 2 Perioden
                
                # Momentum-Umkehr (Umkehr nach starkem Move)
                momentum_reversal_up = (close_diff_pct.shift(1) < -0.003) & (close_diff_pct > 0)
                momentum_reversal_down = (close_diff_pct.shift(1) > 0.003) & (close_diff_pct < 0)
                
                # === 5. KERZENMUSTER-ERKENNUNG ===
                # Erkennung von Hammer/Shooting Stars für Umkehrsignale
                body_size = abs(data['close'] - data['open']) if 'open' in data.columns else \
                            abs(data['close'] - data['close'].shift(1)) * 0.5
                
                upper_wick = data['high'] - np.maximum(data['close'], data['open'] if 'open' in data.columns \
                                                    else data['close'])
                lower_wick = np.minimum(data['close'], data['open'] if 'open' in data.columns \
                                       else data['close']) - data['low']
                
                # Hammer-Muster (Bullish)
                hammer = (lower_wick > (body_size * 2)) & (upper_wick < (body_size * 0.5))
                
                # Shooting Star-Muster (Bearish)
                shooting_star = (upper_wick > (body_size * 2)) & (lower_wick < (body_size * 0.5))
                
                # === 6. ERWEITERTE VOLUMENANALYSE FÜR HOCHFREQUENZHANDEL ===
                if 'volume' in data.columns:
                    try:
                        # Mehrere Zeitfenster für Volumen-Analyse (für unterschiedliche Trading-Horizonte)
                        data['volume_sma_3'] = data['volume'].rolling(window=3).mean()  # Ultra-kurzfristig
                        data['volume_sma_5'] = data['volume'].rolling(window=5).mean()  # Kurzfristig
                        data['volume_sma_10'] = data['volume'].rolling(window=10).mean()  # Mittelfristig
                        
                        # Verschiedene Volumen-Verhältnisse berechnen
                        data['volume_ratio_3'] = data['volume'] / data['volume_sma_3'].replace(0, 0.001)  # Verhältnis zum ultraschnellen Durchschnitt
                        data['volume_ratio_5'] = data['volume'] / data['volume_sma_5'].replace(0, 0.001)  # Verhältnis zum kurzfristigen Durchschnitt
                        data['volume_ratio_10'] = data['volume'] / data['volume_sma_10'].replace(0, 0.001)  # Verhältnis zum mittelfristigen Durchschnitt
                        
                        # 1. Erweiterte Volumen-Spike-Erkennung für 5m HF-Trading (basierend auf Perplexity-Empfehlungen)
                        extreme_volume_spike = data['volume_ratio_5'] > 1.7  # Extremer Anstieg (erhöht von 1.6)
                        sudden_volume_spike = data['volume_ratio_5'] > 1.4  # Starker Anstieg (erhöht von 1.3)
                        moderate_volume_increase = data['volume_ratio_5'] > 1.2  # Moderater Anstieg (erhöht von 1.1)
                        
                        # 2. Volumentrend-Analyse für 5m HF-Trading (optimiert nach Perplexity-Empfehlungen)
                        volume_increasing = data['volume'] > data['volume'].shift(1) * 1.12  # 12% Anstieg zum Vortag (erhöht von 8%)
                        volume_trend_up = (data['volume'] > data['volume'].shift(1)) & \
                                         (data['volume'].shift(1) > data['volume'].shift(2))  # Anstieg über 2 Perioden
                        
                        # 3. Volumen-Preisbestätigung (kritisch für Signalqualität)
                        vol_confirms_price_up = (data['close'] > data['close'].shift(1)) & \
                                               (data['volume'] > data['volume_sma_5'])  # Preis steigt mit überdurchschnittlichem Volumen
                        vol_confirms_price_down = (data['close'] < data['close'].shift(1)) & \
                                                 (data['volume'] > data['volume_sma_5'])  # Preis fällt mit überdurchschnittlichem Volumen
                        
                        # 4. Volumen-Divergenz (fortgeschrittenes Signal)
                        # Positive Divergenz (Kaufsignal): Preis fällt, aber Volumen steigt
                        vol_positive_divergence = (data['close'] < data['close'].shift(1)) & \
                                                 (data['volume'] > data['volume'].shift(1) * 1.4)  # Volumen 40% höher bei fallendem Preis
                        
                        # Negative Divergenz (Verkaufssignal): Preis steigt, aber Volumen fällt
                        vol_negative_divergence = (data['close'] > data['close'].shift(1)) & \
                                                 (data['volume'] < data['volume'].shift(1) * 0.7)  # Volumen 30% niedriger bei steigendem Preis
                    except Exception as e:
                        logger.error(f"Fehler bei der erweiterten Volumenanalyse: {e}")
                        # Fallback zu einfachen Werten bei Fehler
                        sudden_volume_spike = data['volume'] > data['volume'].rolling(window=5).mean() * 1.8
                        volume_increasing = data['volume'] > data['volume'].shift(1) * 1.3
                        
                        # Standardwerte für neue Variablen
                        extreme_volume_spike = pd.Series(False, index=data.index)
                        moderate_volume_increase = pd.Series(False, index=data.index)
                        volume_trend_up = pd.Series(False, index=data.index)
                        vol_confirms_price_up = pd.Series(False, index=data.index) 
                        vol_confirms_price_down = pd.Series(False, index=data.index)
                        vol_positive_divergence = pd.Series(False, index=data.index)
                        vol_negative_divergence = pd.Series(False, index=data.index)
                else:
                    # Wenn keine Volumendaten verfügbar sind
                    sudden_volume_spike = pd.Series(False, index=data.index)
                    volume_increasing = pd.Series(False, index=data.index)
                    extreme_volume_spike = pd.Series(False, index=data.index)
                    moderate_volume_increase = pd.Series(False, index=data.index)
                    volume_trend_up = pd.Series(False, index=data.index)
                    vol_confirms_price_up = pd.Series(False, index=data.index)
                    vol_confirms_price_down = pd.Series(False, index=data.index)
                    vol_positive_divergence = pd.Series(False, index=data.index)
                    vol_negative_divergence = pd.Series(False, index=data.index)
                
                # === 7. STOCHASTIK FÜR MICRO-REVERSALS ===
                # Kurzzeitorientierte Stochastik-Kreuze (schneller als traditionelle 14,3,3)
                if 'stoch_k' in data.columns and 'stoch_d' in data.columns:
                    # Stochastik-Kreuzung nach oben
                    stoch_cross_up = (data['stoch_k'] > data['stoch_d']) & \
                                     (data['stoch_k'].shift(1) <= data['stoch_d'].shift(1))
                    
                    # Stochastik-Kreuzung nach unten
                    stoch_cross_down = (data['stoch_k'] < data['stoch_d']) & \
                                       (data['stoch_k'].shift(1) >= data['stoch_d'].shift(1))
                    
                    # Überverkaufter/überkaufter Bereich mit Umkehr
                    stoch_oversold_reversal = (data['stoch_k'].shift(1) < 20) & (data['stoch_k'] > data['stoch_k'].shift(1))
                    stoch_overbought_reversal = (data['stoch_k'].shift(1) > 80) & (data['stoch_k'] < data['stoch_k'].shift(1))
                else:
                    stoch_cross_up = pd.Series(False, index=data.index)
                    stoch_cross_down = pd.Series(False, index=data.index)
                    stoch_oversold_reversal = pd.Series(False, index=data.index)
                    stoch_overbought_reversal = pd.Series(False, index=data.index)            
            except Exception as e:
                logger.error(f"Fehler bei der Berechnung der Scalping-Indikatoren: {e}")
                return df
                
            # ======= SIGNALBERECHNUNG FÜR HOCHFREQUENZ-STRATEGIE =======
            try:
                # Prüfe, ob rsi_ultra bereits berechnet wurde, ansonsten berechne es jetzt
                if 'rsi_ultra' not in data.columns:
                    # Ultra-schneller RSI (2-Perioden) für schnelle Signale
                    # Berechne direkt hier, ohne externe Funktion
                    delta = data['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    
                    # Für 2-Perioden RSI
                    avg_gain = gain.rolling(window=2).mean()
                    avg_loss = loss.rolling(window=2).mean()
                    rs = avg_gain / avg_loss.replace(0, 1e-10)  # Vermeidet Division durch Null
                    data['rsi_ultra'] = 100 - (100 / (1 + rs))
                
                # Erstelle eine Kombination von Signalen für häufigere Trades                
                # === 1. KAUFSIGNALE FÜR HOCHFREQUENZ-HANDEL (OPTIMIERT MIT VOLUMENFILTER) ===
                buy_conditions = {
                    # A. PRIMÄRE SIGNALE (HAUPTGEWICHT)
                    
                    # Ultra-RSI-Signale mit Volumenfilter (sehr empfindlich aber präzise)
                    'rsi_ultra_oversold': (data['rsi_ultra'] < 20, 2.5),  # Gewichtung erhöht für starke Überverkauft-Signale
                    'rsi_bounce': ((data['rsi_ultra'].shift(1) < 30) & 
                                   (data['rsi_ultra'] > data['rsi_ultra'].shift(1)) & 
                                   moderate_volume_increase, 2.0),  # Volumenfilter hinzugefügt
                    
                    # Divergenz mit Volumenfilter - fortgeschrittenes Pattern
                    'bullish_divergence': (bullish_divergence & vol_confirms_price_up, 3.0),  # Nur mit Volumenfilter
                    'vol_price_divergence_up': (vol_positive_divergence, 2.5),  # Neues Volumen-Divergenz-Signal
                    
                    # B. MIKRO-TREND-SIGNALE (MITTLERES GEWICHT)
                    
                    # Micro-Channel Signale mit Volumenfilter
                    'micro_bounce_up': (micro_bounce_up & moderate_volume_increase, 2.0),  # Volumenfilter hinzugefügt
                    'micro_breakout_up': (micro_breakout_up & (volume_increasing | extreme_volume_spike), 2.2),  # Stärkerer Volumenfilter
                    
                    # Bollinger-Band-Signale präzisiert durch Volumen
                    'bb_touch_lower': (bb_touch_lower & volume_trend_up, 1.8),  # Volumentrend-Bestätigung
                    'bb_touch_lower_10': (bb_touch_lower_10 & volume_trend_up, 1.5),  # Volumentrend-Bestätigung
                    
                    # Momentum-Signale mit Volumenfilter (höhere Qualität)
                    'momentum_reversal_up': (momentum_reversal_up & vol_confirms_price_up, 2.0),  # Mit Volumenfilter
                    'strong_momentum_up': (strong_momentum_up & extreme_volume_spike, 1.8) if 'strong_momentum_up' in locals() else (pd.Series(False, index=data.index), 1.8),
                    
                    # C. BESTÄTIGENDE SIGNALE (GERINGERES GEWICHT, ABER WICHTIG FÜR QUALITÄT)
                    
                    # Kerzenmuster mit Volumenfilter
                    'hammer_pattern': (hammer & volume_increasing, 1.5),  # Volumenfilter hinzugefügt
                    
                    # Stochastik-Signale mit Volumenfilter
                    'stoch_cross_up': (stoch_cross_up & moderate_volume_increase, 1.2),  # Volumenfilter hinzugefügt
                    'stoch_oversold_reversal': (stoch_oversold_reversal & (volume_increasing | sudden_volume_spike), 1.7),  # Volumenfilter
                    
                    # Direkte Volumen-Signale (neuer Abschnitt)
                    'extreme_volume_spike_up': (extreme_volume_spike & (data['close'] > data['close'].shift(1)), 1.8),  # Starker Volumenanstieg
                    'volume_trend_acceleration_up': (volume_trend_up & (data['close'] > data['close'].shift(1)), 1.0)  # Bestätigter Volumentrend
                }
                
                # === 2. VERKAUFSSIGNALE FÜR HOCHFREQUENZ-HANDEL (OPTIMIERT MIT VOLUMENFILTER) ===
                sell_conditions: Dict[str, Tuple[pd.Series, float]] = {
                    # A. PRIMÄRE SIGNALE (HAUPTGEWICHT)
                    
                    # Ultra-RSI-Signale mit Volumenfilter (sehr empfindlich aber präzise)
                    'rsi_ultra_overbought': (data['rsi_ultra'] > 80, 2.5),  # Gewichtung erhöht für starke Überkauft-Signale
                    'rsi_drop': ((data['rsi_ultra'].shift(1) > 70) & 
                                (data['rsi_ultra'] < data['rsi_ultra'].shift(1)) & 
                                moderate_volume_increase, 2.0),  # Volumenfilter hinzugefügt
                    
                    # Divergenz mit Volumenfilter - fortgeschrittenes Pattern
                    'bearish_divergence': (bearish_divergence & vol_confirms_price_down, 3.0),  # Nur mit Volumenfilter
                    'vol_price_divergence': (vol_negative_divergence, 2.5),  # Neues Volumen-Divergenz-Signal
                    
                    # B. MIKRO-TREND-SIGNALE (MITTLERES GEWICHT)
                    
                    # Micro-Channel Signale mit Volumenfilter
                    'micro_bounce_down': (micro_bounce_down & moderate_volume_increase, 2.0),  # Volumenfilter hinzugefügt
                    'micro_breakout_down': (micro_breakout_down & (volume_increasing | extreme_volume_spike), 2.2),  # Stärkerer Volumenfilter
                    
                    # Bollinger-Band-Signale präzisiert durch Volumen
                    'bb_touch_upper': (bb_touch_upper & volume_trend_up, 1.8),  # Volumentrend-Bestätigung
                    'bb_touch_upper_10': (bb_touch_upper_10 & volume_trend_up, 1.5),  # Volumentrend-Bestätigung
                    
                    # Momentum-Signale mit Volumenfilter (höhere Qualität)
                    'momentum_reversal_down': (momentum_reversal_down & vol_confirms_price_down, 2.0),  # Mit Volumenfilter
                    'strong_momentum_down': (strong_momentum_down & extreme_volume_spike, 1.8),  # Mit Volumenfilter
                    
                    # C. BESTÄTIGENDE SIGNALE (GERINGERES GEWICHT, ABER WICHTIG FÜR QUALITÄT)
                    
                    # Kerzenmuster mit Volumenfilter
                    'shooting_star': (shooting_star & volume_increasing, 1.5),  # Volumenfilter hinzugefügt
                    
                    # Stochastik-Signale mit Volumenfilter
                    'stoch_cross_down': (stoch_cross_down & moderate_volume_increase, 1.2),  # Volumenfilter hinzugefügt
                    'stoch_overbought_reversal': (stoch_overbought_reversal & (volume_increasing | sudden_volume_spike), 1.7),  # Volumenfilter
                    
                    # Direkte Volumen-Signale (neuer Abschnitt)
                    'extreme_volume_spike_down': (extreme_volume_spike & (data['close'] < data['close'].shift(1)), 1.8),  # Starker Volumenanstieg
                    'volume_trend_acceleration': (volume_trend_up & (data['close'] < data['close'].shift(1)), 1.0),  # Bestätigter Volumentrend
                }
                
                # Berechne Kauf-Score - deutlich reduzierte Schwelle für mehr Signale
                buy_score = pd.Series(0, index=data.index)
                for cond, weight in buy_conditions.values():
                    buy_score += (cond.fillna(False)).astype(float) * weight
                
                total_buy_weight = sum(weight for _, weight in buy_conditions.values())
                normalized_buy_score = 100 * buy_score / total_buy_weight
                
                # Berechne Verkauf-Score - deutlich reduzierte Schwelle für mehr Signale
                sell_score = pd.Series(0, index=data.index)
                for cond, weight in sell_conditions.values():
                    sell_score += (cond.fillna(False)).astype(float) * weight
                
                total_sell_weight = sum(weight for _, weight in sell_conditions.values())
                normalized_sell_score = 100 * sell_score / total_sell_weight
                
                # Optimierte Schwellenwerte für bessere Signalqualität im 5m HF-Trading
                # Basierend auf Backtest-Ergebnissen und Perplexity-Empfehlungen
                buy_threshold = 6.5  # Von 5 auf 6.5 erhöht für höhere Signalqualität
                sell_threshold = 8  # Von 7 auf 8 erhöht für besseren Schutz gegen falsche Verkaufssignale
                
                # === OPTIMIERTE SIGNALGENERIERUNG MIT VOLUMENFILTER ===
                # Speichere Scores im DataFrame für Debugging und Feinabstimmung
                data['buy_score'] = normalized_buy_score
                data['sell_score'] = normalized_sell_score
                
                # Kaufsignal: Buy-Score über Schwellenwert mit optimierten Bedingungen für 5m HF-Trading
                # Volumenfilter angepasst für bessere Signalqualität
                buy_signal = (normalized_buy_score > buy_threshold) & \
                             (normalized_buy_score > normalized_sell_score * 1.2) & \
                             (moderate_volume_increase | volume_increasing | vol_confirms_price_up)  # Optimierte Volumenfilter
                
                # Verkaufssignal: Sell-Score über Schwellenwert mit verbessertem Schutz für 5m HF-Trading
                # Stärkerer Volumenfilter für höhere Verkaufssignal-Qualität
                sell_signal = (normalized_sell_score > sell_threshold) & \
                              (normalized_sell_score > normalized_buy_score * 1.4) & \
                              (sudden_volume_spike | vol_confirms_price_down | vol_negative_divergence)  # Verbesserter Volumenfilter für robustere Signale
                
                # Optimierte direkte Kaufsignale für 5m HF-Trading mit Bollinger+MACD+RSI Kombinationen
                # Basierend auf Perplexity-Empfehlungen für DOGEBTC
                direct_buy_signal = (
                    ((micro_bounce_up & (volume_increasing | data['volume'] > data['volume_sma_5'])) | 
                     # Verbesserte Bollinger Band + RSI Kombination
                     (bb_touch_lower & data['rsi_ultra'] < 40 & (sudden_volume_spike | volume_increasing)) | 
                     # Optimierte RSI-Schwellenwerte für DOGEBTC
                     ((data['rsi_ultra'] < 30) & (sudden_volume_spike | volume_increasing)) |  
                     # Hammer-Pattern mit stärkerem Volumenfilter
                     (hammer & (sudden_volume_spike | volume_increasing)) |
                     # MACD Bullish Crossover mit Volumenbestätigung
                     ((data['macd_hist'] > 0) & (data['macd_hist'] > data['macd_hist'].shift(1)) & (data['volume'] > data['volume_sma_3'])) |
                     # Stärkerer Mikro-Momentum-Filter
                     ((data['macd_hist'] > data['macd_hist'].shift(1)) & 
                      (data['macd_hist'].shift(1) > data['macd_hist'].shift(2)) & 
                      (data['close'] > data['sma_short'])))
                )
                
                # Optimierte direkte Verkaufssignale mit höheren Schutzmaßnahmen für 5m HF-Trading
                direct_sell_signal = (
                    ((micro_bounce_down & sudden_volume_spike) | 
                     # Verbesserte Bollinger Band + RSI Kombination
                     (bb_touch_upper & data['rsi_ultra'] > 60 & sudden_volume_spike) | 
                     # Optimierter RSI-Filter für DOGEBTC
                     ((data['rsi_ultra'] > 70) & (sudden_volume_spike | vol_confirms_price_down)) |  
                     # Shooting Star mit stärkerer Volumenbestätigung
                     (shooting_star & (vol_confirms_price_down | sudden_volume_spike)) | 
                     ((data['macd_hist'] < 0) & (data['macd_hist'] < data['macd_hist'].shift(1)) & moderate_volume_increase) |  # Volumenfilter gelockert
                     ((data['macd_hist'] < data['macd_hist'].shift(1)) & (data['macd_hist'].shift(1) < data['macd_hist'].shift(2)) & moderate_volume_increase) |  # Mikro-Momentum
                     (vol_negative_divergence & (data['close'] > data['sma_short'])))  # Zusätzlicher Schutz
                )
                
                # Initialisiere oder setze Signal-Spalten zurück
                data['signal'] = 0  # 0 = keine Aktion, 1 = Kauf, -1 = Verkauf
                data['signal_strength'] = 0.0  # Stärke des Signals (für Positionsgrößenanpassung)
                data['signal_quality'] = 0  # Neue Spalte für Signalqualität (0-10)
                
                # Setze Kaufsignal (1) mit Signalstärke und -qualität
                buy_mask = buy_signal | direct_buy_signal
                data.loc[buy_mask, 'signal'] = 1
                data.loc[buy_mask, 'signal_strength'] = normalized_buy_score / 100  # Signalstärke (0-1)
                
                # Berechne Signalqualität für Kaufsignale (0-10 Skala)
                # Höhere Qualität, wenn Volumen und Preis übereinstimmen
                buy_quality = np.zeros(len(data))
                buy_quality = np.where(extreme_volume_spike, buy_quality + 3, buy_quality)  # +3 für extremes Volumen
                buy_quality = np.where(vol_confirms_price_up, buy_quality + 2, buy_quality)  # +2 für Volumen-Preis-Bestätigung
                buy_quality = np.where(vol_positive_divergence, buy_quality + 2, buy_quality)  # +2 für positive Divergenz
                buy_quality = np.where(volume_trend_up, buy_quality + 1, buy_quality)  # +1 für steigenden Volumentrend
                buy_quality = np.where(moderate_volume_increase, buy_quality + 1, buy_quality)  # +1 für moderaten Volumenanstieg
                buy_quality = np.where(buy_quality > 5, 5, buy_quality)  # Auf Max 5 begrenzen
                buy_quality = buy_quality + 5  # Basisscore von 5 hinzufügen
                
                data.loc[buy_mask, 'signal_quality'] = buy_quality[buy_mask]
                
                # Setze Verkaufssignal (-1) mit Signalstärke und -qualität
                sell_mask = sell_signal | direct_sell_signal
                data.loc[sell_mask, 'signal'] = -1
                data.loc[sell_mask, 'signal_strength'] = normalized_sell_score / 100  # Signalstärke (0-1)
                
                # Berechne Signalqualität für Verkaufssignale (0-10 Skala)
                # Höhere Qualität, wenn Volumen und Preis übereinstimmen
                sell_quality = np.zeros(len(data))
                sell_quality = np.where(extreme_volume_spike, sell_quality + 3, sell_quality)  # +3 für extremes Volumen
                sell_quality = np.where(vol_confirms_price_down, sell_quality + 2, sell_quality)  # +2 für Volumen-Preis-Bestätigung
                sell_quality = np.where(vol_negative_divergence, sell_quality + 2, sell_quality)  # +2 für negative Divergenz
                sell_quality = np.where(volume_trend_up, sell_quality + 1, sell_quality)  # +1 für steigenden Volumentrend
                sell_quality = np.where(moderate_volume_increase, sell_quality + 1, sell_quality)  # +1 für moderaten Volumenanstieg
                sell_quality = np.where(sell_quality > 5, 5, sell_quality)  # Auf Max 5 begrenzen
                sell_quality = sell_quality + 5  # Basisscore von 5 hinzufügen
                
                data.loc[sell_mask, 'signal_quality'] = sell_quality[sell_mask]
                
                # Bei Signalkonflikten: Entscheide basierend auf Signal-Qualität und -Stärke
                conflict_mask = (buy_mask & sell_mask)
                if conflict_mask.any():
                    # Berechne gewichtete Bewertung (Qualität * Stärke)
                    for idx in data.index[conflict_mask]:
                        buy_weight = data.loc[idx, 'signal_quality'] * data.loc[idx, 'signal_strength'] if buy_mask[idx] else 0
                        sell_weight = data.loc[idx, 'signal_quality'] * data.loc[idx, 'signal_strength'] if sell_mask[idx] else 0
                        
                        # Wähle das stärkere gewichtete Signal
                        data.loc[idx, 'signal'] = 1 if buy_weight > sell_weight else -1
                
                # Intelligente Signalfilterung basierend auf Qualität und Abstand
                # Dies verhindert zu häufiges Trading, erlaubt aber qualitativ hochwertige Signale
                last_signal_idx = -np.inf
                min_periods_between_signals = 1  # Basis-Mindestabstand für HFT
                quality_factor = 0.2  # Qualitative Signale können den Mindestabstand reduzieren
                
                for i in range(len(data)):
                    if data.iloc[i]['signal'] != 0:  # Wenn ein Signal vorhanden ist
                        # Signalqualität (0-10) und Stärke (0-1) abrufen
                        signal_quality = data.iloc[i]['signal_quality']
                        signal_strength = data.iloc[i]['signal_strength']
                        
                        # Dynamischer Mindestabstand basierend auf Signalqualität
                        # Hochwertige Signale dürfen näher am letzten Signal liegen
                        dynamic_min_periods = max(0, min_periods_between_signals - (signal_quality * quality_factor))
                        
                        # Prüfe, ob das Signal zu nah am letzten Signal liegt
                        if i - last_signal_idx <= dynamic_min_periods:
                            # Signal zu nah am letzten - entscheide welches zu behalten
                            if last_signal_idx >= 0:  # Falls es ein vorheriges Signal gibt
                                prev_quality = data.iloc[last_signal_idx]['signal_quality']
                                prev_strength = data.iloc[last_signal_idx]['signal_strength']
                                
                                # Qualität und Stärke kombiniert vergleichen (gewichtete Bewertung)
                                current_score = signal_quality * 0.7 + signal_strength * 30 * 0.3  # Qualität hat mehr Gewicht
                                prev_score = prev_quality * 0.7 + prev_strength * 30 * 0.3
                                
                                if current_score > prev_score * 1.2:  # Neues Signal ist signifikant besser (20% Schwelle)
                                    # Lösche altes Signal, behalte das neue
                                    data.iloc[last_signal_idx, data.columns.get_loc('signal')] = 0
                                    data.iloc[last_signal_idx, data.columns.get_loc('signal_strength')] = 0
                                    data.iloc[last_signal_idx, data.columns.get_loc('signal_quality')] = 0
                                    last_signal_idx = i  # Update last signal index
                                else:
                                    # Behalte altes Signal, lösche das neue
                                    data.iloc[i, data.columns.get_loc('signal')] = 0
                                    data.iloc[i, data.columns.get_loc('signal_strength')] = 0
                                    data.iloc[i, data.columns.get_loc('signal_quality')] = 0
                            else:
                                # Kein vorheriges Signal, behalte das aktuelle
                                last_signal_idx = i
                        else:
                            # Genug Abstand zum letzten Signal, behalte es
                            last_signal_idx = i
                
                logger.info(f"Hochfrequenz-Handelssignale generiert: {data['signal'].abs().sum()} Signale.")
                return data
                
            except Exception as e:
                logger.error(f"Fehler bei der Signalberechnung für Hochfrequenzhandel: {e}")
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
