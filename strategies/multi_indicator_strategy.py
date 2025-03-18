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
import warnings
from enum import Enum

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.base_strategy import BaseStrategy
from utils.logger import setup_logger
from utils.pattern_loader import PatternLoader
from config.config import LOG_LEVEL, LOG_FILE, DATA_DIRECTORY

# Logger einrichten
logger = setup_logger(os.path.join(DATA_DIRECTORY, LOG_FILE), LOG_LEVEL)

# Suppress warnings from pandas operations
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class PositionStatus(Enum):
    """Enum for position status recommendations"""
    HOLD = "hold"
    SELL = "sell"
    REDUCE = "reduce"

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
        
        # Extrahiere erweiterte Risikomanagement-Parameter aus dem Pattern, falls vorhanden
        self.risk_params = self.pattern.get("risk_params", {})
        if self.risk_params:
            self.take_profit_pct = self.risk_params.get("take_profit_pct", 2.0)
            self.stop_loss_pct = self.risk_params.get("stop_loss_pct", 1.0)
            self.max_trade_duration = self.risk_params.get("max_trade_duration", 48)
            self.position_sizing_method = self.risk_params.get("position_sizing", "fixed")
        else:
            self.take_profit_pct = 2.0
            self.stop_loss_pct = 1.0
            self.max_trade_duration = 48
            self.position_sizing_method = "fixed"
        
        # Überprüfe, ob wir die vom Pattern angegebenen Risikomanagement-Parameter verwenden sollen
        if self.risk_params:
            # Benutze die vom Pattern angegebenen Werte
            if "trailing_stop_pct" in self.risk_params:
                self.trailing_stop_pct = self.risk_params.get("trailing_stop_pct")
            if "max_drawdown_pct" in self.risk_params:
                self.max_drawdown_pct = self.risk_params.get("max_drawdown_pct")
        
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
                   f"Trailing-Stop={self.trailing_stop_pct}%, "
                   f"Max-Drawdown={self.max_drawdown_pct}%, "
                   f"Pattern={self.pattern_name}")
    
    def generate_signals(self, df):
        """
        Generates trading signals based on the loaded trading pattern.
        
        Args:
            df: DataFrame with OHLCV data and technical indicators
            
        Returns:
            DataFrame with buy_signal and sell_signal columns added
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to generate_signals")
            return df
        
        # Preprocess DataFrame to prepare for indicator comparisons
        df_with_indicators = self._preprocess_dataframe(df)
        
        # Apply pattern conditions to calculate buy and sell strengths
        df_with_signals = self._apply_pattern_conditions(df_with_indicators)
        
        # Generate final buy/sell signals based on signal threshold
        df_with_signals['buy_signal'] = (df_with_signals['buy_strength'] >= self.signal_threshold / 10).astype(int)
        df_with_signals['sell_signal'] = (df_with_signals['sell_strength'] >= self.signal_threshold / 10).astype(int)
        
        # Log information about generated signals
        num_buy_signals = df_with_signals['buy_signal'].sum()
        num_sell_signals = df_with_signals['sell_signal'].sum()
        logger.info(f"Generated {num_buy_signals} buy and {num_sell_signals} sell signals from {len(df_with_signals)} data points")
        
        return df_with_signals
    
    def _preprocess_dataframe(self, data):
        """
        Prepares DataFrame with shifted and calculated indicators needed for pattern evaluation.
        Optimizes performance by minimizing DataFrame fragmentation.
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            
        Returns:
            DataFrame with additional indicators for pattern evaluation
        """
        if data.empty:
            return data
            
        # Create a copy to avoid modifying the original dataframe
        data_copy = data.copy()
        
        # Create dictionary to hold all new columns
        new_columns = {}
        
        # Add shifted versions of key indicators for comparison
        for indicator in ['close', 'high', 'low', 'open', 'volume', 'rsi', 'macd_hist']:
            if indicator in data_copy.columns:
                # Previous value (t-1)
                new_columns[f'{indicator}_prev'] = data_copy[indicator].shift(1)
                # Previous value (t-2) for some indicators
                if indicator in ['close', 'high', 'low', 'rsi', 'macd_hist']:
                    new_columns[f'{indicator}_prev2'] = data_copy[indicator].shift(2)
        
        # Calculate SMA values that might be needed
        for period in [5, 10, 20, 50]:
            if 'close' in data_copy.columns:
                new_columns[f'sma_{period}'] = data_copy['close'].rolling(window=period).mean()
            if 'volume' in data_copy.columns:
                new_columns[f'volume_sma_{period}'] = data_copy['volume'].rolling(window=period).mean()
        
        # Add previous SMA values for comparison
        if 'close' in data_copy.columns:
            sma5 = data_copy['close'].rolling(window=5).mean()
            new_columns['sma_5'] = sma5
            new_columns['sma_5_prev'] = sma5.shift(1)
        
        # Min/Max values over periods - calculate all at once to prevent fragmentation
        for period in [10, 20]:
            if 'high' in data_copy.columns:
                new_columns[f'high_max{period}'] = data_copy['high'].rolling(window=period).max()
            if 'low' in data_copy.columns:
                new_columns[f'low_min{period}'] = data_copy['low'].rolling(window=period).min()
        
        # Add Bollinger Band width if needed
        if all(col in data_copy.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            new_columns['bb_width'] = (data_copy['bb_upper'] - data_copy['bb_lower']) / data_copy['bb_middle']
            new_columns['bb_width_prev'] = new_columns['bb_width'].shift(1)
        
        # Add missing EMAs if needed for patterns
        for period in [5, 10, 12, 26]:
            if f'ema_{period}' not in data_copy.columns and 'close' in data_copy.columns:
                new_columns[f'ema_{period}'] = data_copy['close'].ewm(span=period, adjust=False).mean()
                new_columns[f'ema_{period}_prev'] = new_columns[f'ema_{period}'].shift(1)
        
        # Add ADX if not present but needed
        if 'adx' not in data_copy.columns and all(col in data_copy.columns for col in ['high', 'low', 'close']):
            try:
                import pandas_ta as ta
                new_columns['adx'] = ta.adx(data_copy['high'], data_copy['low'], data_copy['close'], length=14)['ADX_14']
            except Exception as e:
                logger.warning(f"Could not calculate ADX: {e}")
        
        # Add unrealized PnL placeholder column (to be filled during evaluation)
        new_columns['unrealized_pnl_pct'] = pd.Series(0.0, index=data_copy.index)
        
        # Add all new columns to the dataframe at once
        data_copy = pd.concat([data_copy, pd.DataFrame(new_columns, index=data_copy.index)], axis=1)
        
        return data_copy
    
    def _apply_pattern_conditions(self, data):
        """
        Applies the buy and sell conditions from the loaded pattern to calculate
        strengths based on weighted conditions.
        
        Args:
            data: DataFrame with preprocessed indicators
            
        Returns:
            DataFrame with buy_strength and sell_strength columns
        """
        if not data.empty:
            buy_weights_sum = 0
            sell_weights_sum = 0
            
            # Initialize buy and sell strength columns
            data['buy_strength'] = 0.0
            data['sell_strength'] = 0.0
            
            # Process buy conditions
            for condition in self.pattern.get('buy_conditions', []):
                weight = condition.get('weight', 1.0)
                buy_weights_sum += weight
                
                # Check if condition is in the new simplified format
                if 'indicator' in condition and 'conditions' not in condition:
                    condition_result = self._evaluate_simple_condition(data, condition)
                # Check if this is a composite condition (with sub-conditions)
                elif 'conditions' in condition:
                    sub_results = []
                    for sub_condition in condition['conditions']:
                        sub_weight = sub_condition.get('weight', 1.0)
                        # Apply the sub condition result
                        sub_result = self._evaluate_simple_condition(data, sub_condition)
                        sub_results.append(sub_result * sub_weight)
                    
                    # Calculate weighted average of sub-conditions
                    sub_weights_sum = sum(sub_condition.get('weight', 1.0) for sub_condition in condition['conditions'])
                    if sub_weights_sum > 0:
                        condition_result = sum(sub_results) / sub_weights_sum
                    else:
                        condition_result = 0
                else:
                    logger.warning(f"Unsupported condition format in buy conditions: {condition}")
                    continue
                    
                # Apply the weight to the condition result and add to strength
                data['buy_strength'] += condition_result * weight
            
            # Process sell conditions
            for condition in self.pattern.get('sell_conditions', []):
                weight = condition.get('weight', 1.0)
                sell_weights_sum += weight
                
                # Check if condition is in the new simplified format
                if 'indicator' in condition and 'conditions' not in condition:
                    condition_result = self._evaluate_simple_condition(data, condition)
                # Check if this is a composite condition (with sub-conditions)
                elif 'conditions' in condition:
                    sub_results = []
                    for sub_condition in condition['conditions']:
                        sub_weight = sub_condition.get('weight', 1.0)
                        # Apply the sub condition result
                        sub_result = self._evaluate_simple_condition(data, sub_condition)
                        sub_results.append(sub_result * sub_weight)
                    
                    # Calculate weighted average of sub-conditions
                    sub_weights_sum = sum(sub_condition.get('weight', 1.0) for sub_condition in condition['conditions'])
                    if sub_weights_sum > 0:
                        condition_result = sum(sub_results) / sub_weights_sum
                    else:
                        condition_result = 0
                else:
                    logger.warning(f"Unsupported condition format in sell conditions: {condition}")
                    continue
                
                # Apply the weight to the condition result and add to strength
                data['sell_strength'] += condition_result * weight
            
            # Normalize buy and sell strengths based on total weights
            if buy_weights_sum > 0:
                data['buy_strength'] = data['buy_strength'] / buy_weights_sum
            if sell_weights_sum > 0:
                data['sell_strength'] = data['sell_strength'] / sell_weights_sum
                
            logger.debug(f"Applied pattern conditions - Buy strength max: {data['buy_strength'].max()}, "
                        f"Sell strength max: {data['sell_strength'].max()}")
        
        return data
    
    def _evaluate_simple_condition(self, data, condition):
        """
        Evaluates a simple condition against the dataframe.
        
        Args:
            data: DataFrame with indicators
            condition: Condition dictionary with indicator, operator, and value/value_indicator
            
        Returns:
            Series with boolean results (1 for True, 0 for False)
        """
        # Get the indicator name and check if it's in the dataframe
        indicator = condition.get('indicator')
        if indicator not in data.columns:
            logger.warning(f"Indicator '{indicator}' not found in DataFrame")
            return pd.Series(0, index=data.index)
        
        # Get the operator function
        op_str = condition.get('operator')
        if op_str not in self.operator_map:
            logger.warning(f"Operator '{op_str}' not supported")
            return pd.Series(0, index=data.index)
        
        op_func = self.operator_map[op_str]
        
        # Determine the comparison value
        if 'value' in condition:
            # Direct value comparison (e.g., RSI < 30)
            value = condition.get('value')
            result = op_func(data[indicator], value)
        elif 'indicator_compare' in condition:
            # Compare with another indicator (e.g., close > sma_20)
            compare_indicator = condition.get('indicator_compare')
            if compare_indicator not in data.columns:
                logger.warning(f"Comparison indicator '{compare_indicator}' not found in DataFrame")
                return pd.Series(0, index=data.index)
            
            # Apply optional factor (e.g., volume > volume_sma_20 * 1.5)
            factor = condition.get('factor', 1.0)
            result = op_func(data[indicator], data[compare_indicator] * factor)
        else:
            logger.warning(f"Condition must have either 'value' or 'indicator_compare': {condition}")
            return pd.Series(0, index=data.index)
        
        # Convert boolean result to 1.0 or 0.0
        return result.astype(float)
    
    def should_buy(self, df, position=None):
        """
        Determines whether to buy based on generated signals and current market conditions.
        
        Args:
            df: DataFrame with signals
            position: Optional current position information
            
        Returns:
            tuple: (buy_decision, signal_strength)
        """
        if df.empty:
            return False, 0.0
            
        # Get the latest data point
        latest = df.iloc[-1]
        
        # Check if we already have the maximum number of positions
        if position is not None and len(position) >= self.max_positions:
            return False, 0.0
            
        # Check for buy signal
        if latest.get('buy_signal', 0) > 0:
            # Get signal strength as confidence indicator
            signal_strength = latest.get('buy_strength', 0.0)
            
            # Additional conditions for buy confirmation
            if self.volatility_adjustment:
                # Check if volatility is acceptable
                atr = latest.get('atr', 0)
                close = latest.get('close', 0)
                
                if close > 0 and atr > 0:
                    # Calculate volatility as percentage of price
                    volatility_pct = (atr / close) * 100
                    
                    # If volatility is too high, reduce signal strength
                    if volatility_pct > 5.0:  # 5% volatility threshold
                        signal_strength *= 0.5
                    
            # Machine learning confirmation if enabled
            if self.use_ml_predictions and 'ml_prediction' in latest:
                ml_prediction = latest.get('ml_prediction', 0)
                
                # If ML predicts negative, reduce signal strength
                if ml_prediction < 0:
                    signal_strength *= 0.5
                # If ML strongly predicts positive, boost signal strength
                elif ml_prediction > 0.5:
                    signal_strength *= 1.5
            
            return signal_strength >= self.signal_threshold / 10, signal_strength
            
        return False, 0.0
        
    def should_sell(self, df, position=None):
        """
        Determines whether to sell based on generated signals and current market conditions.
        
        Args:
            df: DataFrame with signals
            position: Optional current position information
            
        Returns:
            tuple: (sell_decision, signal_strength)
        """
        if df.empty:
            return False, 0.0
            
        # Get the latest data point
        latest = df.iloc[-1]
        
        # If no position, nothing to sell
        if position is None:
            return False, 0.0
            
        # Check for sell signal
        if latest.get('sell_signal', 0) > 0:
            # Get signal strength as confidence indicator
            signal_strength = latest.get('sell_strength', 0.0)
            
            # Additional conditions or risk management rules can be added here
            
            # Machine learning confirmation if enabled
            if self.use_ml_predictions and 'ml_prediction' in latest:
                ml_prediction = latest.get('ml_prediction', 0)
                
                # If ML predicts positive, reduce sell signal strength
                if ml_prediction > 0:
                    signal_strength *= 0.7
                # If ML strongly predicts negative, boost sell signal
                elif ml_prediction < -0.5:
                    signal_strength *= 1.5
            
            return signal_strength >= self.signal_threshold / 10, signal_strength
            
        return False, 0.0

    def evaluate_position(self, position, current_data):
        """
        Evaluates an existing position and updates its status based on current market data.
        
        Args:
            position: The position to evaluate
            current_data: Current market data (Series or DataFrame)
            
        Returns:
            tuple: (recommendation, reason)
                recommendation: One of HOLD, SELL, REDUCE
                reason: String explaining the recommendation
        """
        if position is None or current_data is None:
            return PositionStatus.HOLD, "Insufficient data for evaluation"
        
        # Ensure current_data is a Series (not a DataFrame)
        if isinstance(current_data, pd.DataFrame):
            if current_data.empty:
                return PositionStatus.HOLD, "Empty data for evaluation"
            if len(current_data) == 1:
                latest = current_data.iloc[0]
            else:
                logger.warning(f"Multiple rows in current_data for position evaluation. Using first row.")
                latest = current_data.iloc[0]
        else:
            # current_data is already a Series
            latest = current_data
        
        symbol = position.symbol
        
        # Calculate current unrealized profit/loss as a percentage
        if position.current_price is None or position.current_price == 0:
            position.current_price = latest['close']
        
        unrealized_pnl_pct = ((latest['close'] - position.entry_price) / position.entry_price * 100) 
        if hasattr(position, 'direction') and position.direction == 'short':
            unrealized_pnl_pct = -unrealized_pnl_pct
        
        # Calculate time in position (in hours)
        hours_in_position = 0
        if position.entry_time is not None:
            current_time = pd.Timestamp.now()
            hours_in_position = (current_time - position.entry_time).total_seconds() / 3600
        
        # Take profit check using the pattern's take_profit_pct
        if unrealized_pnl_pct >= self.take_profit_pct:
            return PositionStatus.SELL, f"Take profit target reached: {unrealized_pnl_pct:.2f}% >= {self.take_profit_pct:.2f}%"
        
        # Stop loss check using the pattern's stop_loss_pct
        if unrealized_pnl_pct <= -self.stop_loss_pct:
            return PositionStatus.SELL, f"Stop loss triggered: {unrealized_pnl_pct:.2f}% <= -{self.stop_loss_pct:.2f}%"
        
        # Maximum trade duration check
        if hours_in_position >= self.max_trade_duration:
            return PositionStatus.SELL, f"Maximum trade duration reached: {hours_in_position:.1f} hours >= {self.max_trade_duration} hours"
        
        # Get trailing stop level if it exists
        trailing_stop_level = getattr(position, 'trailing_stop_level', None)
        highest_price = getattr(position, 'highest_price', position.entry_price)
        
        # Update highest price if current price is higher
        if latest['close'] > highest_price:
            position.highest_price = latest['close']
            
            # Calculate new trailing stop level
            if self.trailing_stop_pct > 0:
                position.trailing_stop_level = position.highest_price * (1 - self.trailing_stop_pct / 100)
                logger.debug(f"Updated trailing stop for {symbol} to {position.trailing_stop_level}")
                
        # Check if trailing stop is triggered
        if (trailing_stop_level is not None and 
            position.trailing_stop_level is not None and 
            latest['close'] < position.trailing_stop_level):
            return PositionStatus.SELL, f"Trailing stop triggered at {position.trailing_stop_level}"
        
        # Check for sell signals based on our indicators
        if isinstance(latest, pd.Series) and 'sell_signal' in latest:
            if latest.get('sell_signal', 0) > 0:
                return PositionStatus.SELL, "Sell signal generated by strategy"
            
        # Default to holding the position
        return PositionStatus.HOLD, "No exit criteria met"
