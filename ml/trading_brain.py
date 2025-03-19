"""
Trading Brain - Unified ML module that combines multiple prediction components
to make intelligent trading decisions for the Binance Trading Bot.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

from ml.price_direction_predictor import PriceDirectionPredictor
from ml.support_resistance_detector import SupportResistanceDetector
from ml.market_regime_classifier import MarketRegimeClassifier, MarketRegime

class TradingBrain:
    """
    Unified ML decision engine that combines multiple prediction components:
    1. Price Direction Predictor - Using LSTM to predict short-term price movements
    2. Support/Resistance Detector - Identifying key price levels
    3. Market Regime Classifier - Adapting strategy to current market conditions
    
    This class serves as the primary interface for ML-based decision making
    in the trading bot.
    """
    
    def __init__(self, symbol: str, timeframe: str = '1h', 
                data_dir: str = 'data', model_dir: str = 'ml/models'):
        """
        Initialize the trading brain with all required ML components.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            data_dir: Directory for data storage
            model_dir: Directory for ML models
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Initialize individual components
        self.price_predictor = PriceDirectionPredictor(symbol, timeframe, model_dir)
        self.sr_detector = SupportResistanceDetector(symbol, timeframe, os.path.join(data_dir, 'zones'))
        self.regime_classifier = MarketRegimeClassifier(symbol, timeframe, model_dir)
        
        # Trading state and history
        self.last_prediction = None
        self.prediction_history = []
        self.current_regime = MarketRegime.UNKNOWN
        self.current_zones = {'support': [], 'resistance': []}
        
        # Configuration
        self.min_prediction_confidence = 0.60
        self.min_zone_strength = 0.5
        self.use_adaptive_parameters = True
        
        # Logger setup
        self.logger = logging.getLogger('trading_brain')
        
        # Memory of recent predictions (for tracking performance)
        self.prediction_memory = []
        
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all ML models using historical data.
        
        Args:
            df: DataFrame with OHLCV and indicator data
            
        Returns:
            Dict with training results from each component
        """
        self.logger.info(f"Training ML models for {self.symbol} {self.timeframe}")
        
        results = {}
        
        # Train price direction predictor
        if len(df) >= 1000:  # Need substantial data for LSTM
            try:
                price_results = self.price_predictor.train(df, epochs=50, batch_size=32)
                results['price_predictor'] = {
                    'status': 'success',
                    'metrics': price_results
                }
            except Exception as e:
                self.logger.error(f"Error training price predictor: {e}")
                results['price_predictor'] = {
                    'status': 'error',
                    'message': str(e)
                }
        else:
            results['price_predictor'] = {
                'status': 'skipped',
                'message': f"Insufficient data: {len(df)} rows (need at least 1000)"
            }
            
        # Train market regime classifier
        try:
            regime_results = self.regime_classifier.train(df, n_clusters=6)
            results['regime_classifier'] = {
                'status': 'success',
                'metrics': regime_results
            }
        except Exception as e:
            self.logger.error(f"Error training regime classifier: {e}")
            results['regime_classifier'] = {
                'status': 'error',
                'message': str(e)
            }
            
        # The SR detector doesn't require training (rule-based)
        results['sr_detector'] = {
            'status': 'ready',
            'message': 'Support/resistance detector is rule-based and doesn\'t require training'
        }
        
        return results
        
    def analyze(self, df: pd.DataFrame, order_book: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis using all ML components.
        
        Args:
            df: DataFrame with OHLCV and indicator data
            order_book: Optional order book data for enhanced S/R detection
            
        Returns:
            Dict with analysis results and trading recommendations
        """
        # Ensure we have enough data
        if len(df) < 50:
            return {
                'status': 'error',
                'message': f"Insufficient data: {len(df)} rows (need at least 50)",
                'action': 'none',
                'confidence': 0.0
            }
            
        current_price = df['close'].iloc[-1]
        
        # Step 1: Classify market regime
        regime, regime_confidence = self.regime_classifier.classify_regime(df)
        self.current_regime = regime
        
        # Step 2: Detect support and resistance zones
        zones = self.sr_detector.detect_zones(df, order_book)
        self.current_zones = zones
        
        # Get nearest zones context
        zones_context = self.sr_detector.calculate_zone_context(current_price)
        
        # Step 3: Predict price direction
        try:
            direction_probability, direction, direction_details = self.price_predictor.predict(df)
        except Exception as e:
            self.logger.warning(f"Error predicting price direction: {e}")
            direction_probability = 0.5
            direction = "neutral"
            direction_details = {}
            
        # Step 4: Generate combined signal and recommendations
        signal, confidence, trade_params = self._generate_trading_signal(
            current_price, direction, direction_probability,
            regime, regime_confidence, zones_context
        )
        
        # Record prediction
        prediction_record = {
            'timestamp': datetime.now(),
            'price': current_price,
            'regime': regime.value,
            'direction': direction,
            'signal': signal,
            'confidence': confidence
        }
        
        self.last_prediction = prediction_record
        self.prediction_history.append(prediction_record)
        
        # Limit history length
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
            
        # Build detailed response
        analysis = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'current_price': float(current_price),
            
            # Trading recommendation
            'action': signal,
            'confidence': float(confidence),
            'trade_params': trade_params,
            
            # Component details
            'price_prediction': {
                'direction': direction,
                'probability': float(direction_probability),
                'details': direction_details
            },
            'market_regime': {
                'regime': regime.value,
                'confidence': float(regime_confidence),
                'parameters': self.regime_classifier.optimize_strategy_parameters(regime)
            },
            'support_resistance': {
                'zones': {
                    'support': [{'price': float(p), 'strength': float(s)} for p, s in zones['support']],
                    'resistance': [{'price': float(p), 'strength': float(s)} for p, s in zones['resistance']]
                },
                'context': zones_context
            }
        }
        
        return analysis
        
    def _generate_trading_signal(self, current_price: float, 
                              direction: str, direction_probability: float,
                              regime: MarketRegime, regime_confidence: float,
                              zones_context: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
        """
        Generate a unified trading signal by combining all ML components.
        
        Args:
            current_price: Current price
            direction: Predicted price direction
            direction_probability: Confidence in direction prediction
            regime: Current market regime
            regime_confidence: Confidence in regime classification
            zones_context: Support/resistance zone context
            
        Returns:
            Tuple of (signal, confidence, trade_parameters)
        """
        # Default signal
        signal = "none"
        confidence = 0.0
        
        # Get regime-optimized parameters
        params = self.regime_classifier.optimize_strategy_parameters(regime)
        
        # Base confidence calculation from direction prediction
        base_confidence = direction_probability
        
        # Adjust confidence based on regime and zone context
        if direction == "buy":
            # Adjustments for buy signals
            zone_factor = zones_context.get('buy_zone_strength', 0.5)
            
            # Strong confidence if in support zone or far from resistance
            if zones_context.get('in_support_zone', False):
                zone_factor += 0.2
            elif zones_context.get('strong_support', False):
                zone_factor += 0.15
                
            # Reduce confidence if near strong resistance
            if zones_context.get('strong_resistance', False):
                zone_factor -= 0.2
                
            # Consider regime - uptrend regimes favor buys
            if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.BREAKOUT]:
                regime_factor = 1.2
            elif regime == MarketRegime.WEAK_UPTREND:
                regime_factor = 1.1
            elif regime == MarketRegime.RANGING:
                regime_factor = 0.9
            elif regime in [MarketRegime.WEAK_DOWNTREND, MarketRegime.STRONG_DOWNTREND]:
                regime_factor = 0.6
            elif regime == MarketRegime.VOLATILE:
                regime_factor = 0.8
            elif regime == MarketRegime.REVERSAL and zones_context.get('support_proximity', 1.0) < 0.02:
                # Potential bullish reversal near support
                regime_factor = 1.1
            else:
                regime_factor = 0.9
                
            # Calculate final confidence
            confidence = base_confidence * zone_factor * regime_factor
            confidence = min(0.95, confidence)  # Cap at 0.95
            
            # Determine signal based on threshold
            if confidence >= self.min_prediction_confidence:
                signal = "buy"
            
        elif direction == "sell":
            # Adjustments for sell signals
            zone_factor = zones_context.get('sell_zone_strength', 0.5)
            
            # Strong confidence if in resistance zone or far from support
            if zones_context.get('in_resistance_zone', False):
                zone_factor += 0.2
            elif zones_context.get('strong_resistance', False):
                zone_factor += 0.15
                
            # Reduce confidence if near strong support
            if zones_context.get('strong_support', False):
                zone_factor -= 0.2
                
            # Consider regime - downtrend regimes favor sells
            if regime in [MarketRegime.STRONG_DOWNTREND]:
                regime_factor = 1.2
            elif regime == MarketRegime.WEAK_DOWNTREND:
                regime_factor = 1.1
            elif regime == MarketRegime.RANGING:
                regime_factor = 0.9
            elif regime in [MarketRegime.WEAK_UPTREND, MarketRegime.STRONG_UPTREND]:
                regime_factor = 0.6
            elif regime == MarketRegime.VOLATILE:
                regime_factor = 0.8
            elif regime == MarketRegime.REVERSAL and zones_context.get('resistance_proximity', 1.0) < 0.02:
                # Potential bearish reversal near resistance
                regime_factor = 1.1
            else:
                regime_factor = 0.9
                
            # Calculate final confidence
            confidence = base_confidence * zone_factor * regime_factor
            confidence = min(0.95, confidence)  # Cap at 0.95
            
            # Determine signal based on threshold
            if confidence >= self.min_prediction_confidence:
                signal = "sell"
                
        # Build trade parameters
        trade_params = {
            'signal_threshold': params['signal_threshold'],
            'risk_percentage': params['risk_percentage'],
            'confidence': float(confidence),
            'regime': regime.value,
            'position_size_factor': params['position_size_factor']
        }
        
        # Add S/R based stop-loss and take-profit
        if signal == "buy" and zones_context.get('nearest_support') is not None:
            price, strength = zones_context['nearest_support']
            if strength > self.min_zone_strength:
                # Set stop loss below support
                trade_params['stop_loss'] = price * 0.995
                
            # Optimal take profit levels
            if zones_context.get('nearest_resistance') is not None:
                res_price, res_strength = zones_context['nearest_resistance']
                if res_strength > self.min_zone_strength:
                    # Set take profit near resistance
                    trade_params['take_profit'] = res_price * 0.995
                    
        elif signal == "sell" and zones_context.get('nearest_resistance') is not None:
            price, strength = zones_context['nearest_resistance']
            if strength > self.min_zone_strength:
                # Set stop loss above resistance
                trade_params['stop_loss'] = price * 1.005
                
            # Optimal take profit levels
            if zones_context.get('nearest_support') is not None:
                sup_price, sup_strength = zones_context['nearest_support']
                if sup_strength > self.min_zone_strength:
                    # Set take profit near support
                    trade_params['take_profit'] = sup_price * 1.005
                    
        # Default stop loss if not set
        if signal in ["buy", "sell"] and 'stop_loss' not in trade_params:
            if signal == "buy":
                trade_params['stop_loss'] = current_price * (1 - params['stop_loss_pct'] / 100)
            else:
                trade_params['stop_loss'] = current_price * (1 + params['stop_loss_pct'] / 100)
                
        # Default take profit if not set
        if signal in ["buy", "sell"] and 'take_profit' not in trade_params:
            if signal == "buy":
                trade_params['take_profit'] = current_price * (1 + params['take_profit_pct'] / 100)
            else:
                trade_params['take_profit'] = current_price * (1 - params['take_profit_pct'] / 100)
                
        return signal, confidence, trade_params
        
    def evaluate_position(self, position: Dict[str, Any], current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate an existing position and recommend action.
        
        Args:
            position: Dictionary with position details
            current_data: DataFrame with recent market data
            
        Returns:
            Dict with position evaluation and recommendations
        """
        if len(current_data) < 20:
            return {
                'action': 'hold',
                'confidence': 0.5,
                'reason': 'Insufficient data for evaluation'
            }
            
        # Current price and position info
        current_price = current_data['close'].iloc[-1]
        entry_price = position.get('entry_price', current_price)
        position_type = position.get('direction', 'long')
        
        # Run full analysis
        analysis = self.analyze(current_data)
        
        # Check if we should exit based on ML predictions
        action = "hold"
        confidence = 0.5
        reason = "Default hold"
        
        # Calculate unrealized P&L
        if position_type == 'long':
            unrealized_pnl_pct = (current_price / entry_price - 1) * 100
        else:
            unrealized_pnl_pct = (entry_price / current_price - 1) * 100
            
        # Get current regime
        regime = self.current_regime
        regime_params = self.regime_classifier.optimize_strategy_parameters(regime)
        
        # Get zone context
        zones_context = self.sr_detector.calculate_zone_context(current_price)
        
        # Check for exit signals
        if position_type == 'long':
            # Long position, check for sell signals
            if analysis['action'] == 'sell' and analysis['confidence'] > 0.7:
                action = "exit"
                confidence = analysis['confidence']
                reason = "Strong sell signal"
            elif unrealized_pnl_pct > regime_params['take_profit_pct']:
                action = "exit"
                confidence = 0.8
                reason = f"Take profit reached: {unrealized_pnl_pct:.2f}%"
            elif unrealized_pnl_pct < -regime_params['stop_loss_pct']:
                action = "exit"
                confidence = 0.9
                reason = f"Stop loss reached: {unrealized_pnl_pct:.2f}%"
            elif zones_context.get('in_resistance_zone', False):
                action = "reduce"
                confidence = 0.7
                reason = "Price at resistance zone"
            elif regime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.WEAK_DOWNTREND]:
                action = "reduce"
                confidence = 0.65
                reason = f"Market regime changed to {regime.value}"
                
        else:
            # Short position, check for buy signals
            if analysis['action'] == 'buy' and analysis['confidence'] > 0.7:
                action = "exit"
                confidence = analysis['confidence']
                reason = "Strong buy signal"
            elif unrealized_pnl_pct > regime_params['take_profit_pct']:
                action = "exit"
                confidence = 0.8
                reason = f"Take profit reached: {unrealized_pnl_pct:.2f}%"
            elif unrealized_pnl_pct < -regime_params['stop_loss_pct']:
                action = "exit"
                confidence = 0.9
                reason = f"Stop loss reached: {unrealized_pnl_pct:.2f}%"
            elif zones_context.get('in_support_zone', False):
                action = "reduce"
                confidence = 0.7
                reason = "Price at support zone"
            elif regime in [MarketRegime.STRONG_UPTREND, MarketRegime.WEAK_UPTREND]:
                action = "reduce"
                confidence = 0.65
                reason = f"Market regime changed to {regime.value}"
                
        # Check for trailing stop
        if action == "hold" and position.get('highest_price') and position_type == 'long':
            highest_price = position['highest_price']
            trailing_stop_pct = regime_params['trailing_stop_pct']
            trailing_stop_level = highest_price * (1 - trailing_stop_pct / 100)
            
            if current_price < trailing_stop_level and unrealized_pnl_pct > 0:
                action = "exit"
                confidence = 0.75
                reason = f"Trailing stop triggered at {trailing_stop_level:.2f}"
                
        elif action == "hold" and position.get('lowest_price') and position_type == 'short':
            lowest_price = position['lowest_price']
            trailing_stop_pct = regime_params['trailing_stop_pct']
            trailing_stop_level = lowest_price * (1 + trailing_stop_pct / 100)
            
            if current_price > trailing_stop_level and unrealized_pnl_pct > 0:
                action = "exit"
                confidence = 0.75
                reason = f"Trailing stop triggered at {trailing_stop_level:.2f}"
                
        return {
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'regime': regime.value,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'zones_context': zones_context
        }
        
    def optimize_exit_parameters(self, df: pd.DataFrame, position_type: str = 'long') -> Dict[str, float]:
        """
        Optimize exit parameters (stop-loss, take-profit) based on current market conditions.
        
        Args:
            df: DataFrame with market data
            position_type: 'long' or 'short'
            
        Returns:
            Dict with optimized exit parameters
        """
        # Analyze current regime and support/resistance
        current_price = df['close'].iloc[-1]
        regime, _ = self.regime_classifier.classify_regime(df)
        regime_params = self.regime_classifier.optimize_strategy_parameters(regime)
        
        # Detect support/resistance zones
        zones = self.sr_detector.detect_zones(df)
        
        if position_type == 'long':
            # For long positions, find optimal stop loss and take profit
            stop_loss = self.sr_detector.get_optimal_stop_loss(current_price, 'BUY')
            if stop_loss is None:
                # Use regime-based stop loss if no clear S/R levels
                stop_loss = current_price * (1 - regime_params['stop_loss_pct'] / 100)
                
            # Get take profit levels
            take_profit_levels = self.sr_detector.get_optimal_take_profit(current_price, 'BUY')
            if take_profit_levels:
                # Use nearest strong resistance as take profit
                take_profit = take_profit_levels[0][0]
            else:
                # Use regime-based take profit if no clear S/R levels
                take_profit = current_price * (1 + regime_params['take_profit_pct'] / 100)
                
        else:
            # For short positions, find optimal stop loss and take profit
            stop_loss = self.sr_detector.get_optimal_stop_loss(current_price, 'SELL')
            if stop_loss is None:
                # Use regime-based stop loss if no clear S/R levels
                stop_loss = current_price * (1 + regime_params['stop_loss_pct'] / 100)
                
            # Get take profit levels
            take_profit_levels = self.sr_detector.get_optimal_take_profit(current_price, 'SELL')
            if take_profit_levels:
                # Use nearest strong support as take profit
                take_profit = take_profit_levels[0][0]
            else:
                # Use regime-based take profit if no clear S/R levels
                take_profit = current_price * (1 - regime_params['take_profit_pct'] / 100)
                
        # Calculate risk-reward ratio
        if position_type == 'long':
            reward = take_profit - current_price
            risk = current_price - stop_loss
        else:
            reward = current_price - take_profit
            risk = stop_loss - current_price
            
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Final parameters
        return {
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'trailing_stop_pct': float(regime_params['trailing_stop_pct']),
            'risk_reward_ratio': float(risk_reward_ratio),
            'regime': regime.value
        }
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics for ML predictions.
        
        Returns:
            Dict with performance metrics
        """
        # Default metrics
        metrics = {
            'accuracy': 0.0,
            'buy_accuracy': 0.0,
            'sell_accuracy': 0.0,
            'total_predictions': len(self.prediction_memory),
            'correct_predictions': 0,
            'regime_distribution': {}
        }
        
        if not self.prediction_memory:
            return metrics
            
        # Calculate basic metrics
        correct_count = sum(1 for p in self.prediction_memory if p.get('correct', False))
        metrics['accuracy'] = correct_count / len(self.prediction_memory) if self.prediction_memory else 0
        
        # Buy/sell specific accuracy
        buy_predictions = [p for p in self.prediction_memory if p['signal'] == 'buy']
        sell_predictions = [p for p in self.prediction_memory if p['signal'] == 'sell']
        
        buy_correct = sum(1 for p in buy_predictions if p.get('correct', False))
        sell_correct = sum(1 for p in sell_predictions if p.get('correct', False))
        
        metrics['buy_accuracy'] = buy_correct / len(buy_predictions) if buy_predictions else 0
        metrics['sell_accuracy'] = sell_correct / len(sell_predictions) if sell_predictions else 0
        
        # Regime distribution
        regime_counts = {}
        for p in self.prediction_memory:
            regime = p.get('regime', 'unknown')
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
        metrics['regime_distribution'] = {
            regime: count / len(self.prediction_memory)
            for regime, count in regime_counts.items()
        }
        
        return metrics
        
    def update_prediction_performance(self, prediction_id: str, 
                                   correct: bool, pnl: float) -> None:
        """
        Update performance record for a specific prediction.
        
        Args:
            prediction_id: Unique identifier for the prediction
            correct: Whether the prediction was correct
            pnl: Profit/loss from the trade
        """
        # Find prediction in history
        for prediction in self.prediction_memory:
            if prediction.get('id') == prediction_id:
                prediction['correct'] = correct
                prediction['pnl'] = pnl
                prediction['evaluated'] = True
                break 