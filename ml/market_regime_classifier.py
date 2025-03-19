"""
Market Regime Classifier for Binance Trading Bot.
Identifies market conditions (trending, ranging, volatile) to optimize trading strategy parameters.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from enum import Enum
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

class MarketRegime(Enum):
    """Enum for different market regimes"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"

class MarketRegimeClassifier:
    """
    Classifies market conditions into distinct regimes to adapt trading strategies.
    
    This classifier uses a combination of technical indicators, volatility measures,
    and pattern recognition to determine the current market regime and suggest
    optimal trading parameters.
    """
    
    def __init__(self, symbol: str, timeframe: str = '1h', model_dir: str = 'ml/models'):
        """
        Initialize the market regime classifier.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            model_dir: Directory for saving/loading models
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_dir = model_dir
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Model file paths
        self.model_file = os.path.join(model_dir, f'regime_classifier_{symbol}_{timeframe}.joblib')
        self.scaler_file = os.path.join(model_dir, f'regime_scaler_{symbol}_{timeframe}.joblib')
        
        # Initialize classifier components
        self.kmeans = None  # Unsupervised model for regime clustering
        self.scaler = None  # Feature scaler
        
        # Configure parameters for regime detection
        self.lookback_period = 20  # Period for trend analysis
        self.volatility_window = 10  # Window for volatility calculation
        self.regime_change_threshold = 0.7  # Confidence threshold for regime change
        
        # Current regime state
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history = []  # Store recent regime classifications
        self.regime_confidence = 0.0
        
        # Load model if available
        self._load_model()
        
        # Logger setup
        self.logger = logging.getLogger('market_regime_classifier')
        
    def _load_model(self) -> bool:
        """
        Load the saved classifier model if it exists.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
                self.kmeans = joblib.load(self.model_file)
                self.scaler = joblib.load(self.scaler_file)
                self.logger.info(f"Loaded regime classifier model for {self.symbol} {self.timeframe}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error loading regime classifier model: {e}")
            return False
            
    def _save_model(self) -> None:
        """Save the current model to disk."""
        try:
            if self.kmeans is not None and self.scaler is not None:
                joblib.dump(self.kmeans, self.model_file)
                joblib.dump(self.scaler, self.scaler_file)
                self.logger.info(f"Saved regime classifier model for {self.symbol} {self.timeframe}")
        except Exception as e:
            self.logger.error(f"Error saving regime classifier model: {e}")
            
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for regime classification from price data.
        
        Args:
            df: DataFrame with OHLCV and indicator data
            
        Returns:
            DataFrame with extracted features for regime classification
        """
        # Create a copy to avoid modifying the original
        features = pd.DataFrame(index=df.index)
        
        # Price trends
        features['price_trend'] = df['close'].pct_change(self.lookback_period)
        
        # Moving average relationships
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            features['ma_relationship'] = df['sma_20'] / df['sma_50'] - 1
        else:
            # Calculate if not present
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            features['ma_relationship'] = df['sma_20'] / df['sma_50'] - 1
        
        # Volatility measures
        if 'atr' in df.columns:
            features['normalized_atr'] = df['atr'] / df['close']
        else:
            # Simple volatility measure using standard deviation
            features['normalized_atr'] = df['close'].rolling(window=14).std() / df['close']
        
        # Bollinger band width if available
        if all(x in df.columns for x in ['bb_upper', 'bb_lower', 'bb_middle']):
            features['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volume trend
        features['volume_trend'] = df['volume'].pct_change(self.lookback_period)
        
        # RSI extremes and trends
        if 'rsi' in df.columns:
            features['rsi_level'] = (df['rsi'] - 50) / 50  # Normalize around zero
            features['rsi_trend'] = df['rsi'].diff(5)
        
        # MACD if available
        if all(x in df.columns for x in ['macd', 'macd_signal']):
            features['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Handle missing values
        features = features.replace([np.inf, -np.inf], np.nan).dropna()
        
        return features
        
    def train(self, df: pd.DataFrame, n_clusters: int = 6) -> Dict[str, Any]:
        """
        Train the classifier using historical data.
        
        Args:
            df: DataFrame with OHLCV and indicator data
            n_clusters: Number of market regimes to identify
            
        Returns:
            Dict with training results and metrics
        """
        self.logger.info(f"Training regime classifier on {len(df)} data points")
        
        # Extract features for classification
        features_df = self._extract_features(df)
        
        # Normalize features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features_df.values)
        
        # Train KMeans model
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(scaled_features)
        
        # Add cluster labels to original data for analysis
        cluster_labels = self.kmeans.labels_
        df_labeled = df.loc[features_df.index].copy()
        df_labeled['regime_cluster'] = cluster_labels
        
        # Analyze clusters to understand their characteristics
        cluster_analysis = self._analyze_clusters(df_labeled)
        
        # Map clusters to market regimes
        regime_mapping = self._map_clusters_to_regimes(df_labeled, cluster_analysis)
        
        # Save the trained model
        self._save_model()
        
        return {
            'n_clusters': n_clusters,
            'samples_trained': len(features_df),
            'cluster_analysis': cluster_analysis,
            'regime_mapping': regime_mapping
        }
        
    def _analyze_clusters(self, df_labeled: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        Analyze characteristics of each cluster to understand different regimes.
        
        Args:
            df_labeled: DataFrame with cluster labels
            
        Returns:
            Dict with cluster analysis metrics
        """
        analysis = {}
        
        for cluster_id in df_labeled['regime_cluster'].unique():
            cluster_data = df_labeled[df_labeled['regime_cluster'] == cluster_id]
            
            # Calculate various metrics for each cluster
            ret_mean = cluster_data['close'].pct_change().mean()
            ret_std = cluster_data['close'].pct_change().std()
            
            # Calculate win rate (percentage of positive returns)
            win_rate = (cluster_data['close'].pct_change() > 0).mean()
            
            # Average trend metrics
            if 'sma_20' in cluster_data.columns and 'sma_50' in cluster_data.columns:
                ma_ratio = (cluster_data['sma_20'] / cluster_data['sma_50']).mean()
            else:
                ma_ratio = np.nan
            
            # Volatility
            if 'atr' in cluster_data.columns:
                volatility = (cluster_data['atr'] / cluster_data['close']).mean()
            else:
                volatility = cluster_data['close'].pct_change().rolling(10).std().mean()
            
            # RSI metrics if available
            if 'rsi' in cluster_data.columns:
                avg_rsi = cluster_data['rsi'].mean()
            else:
                avg_rsi = np.nan
            
            analysis[int(cluster_id)] = {
                'return_mean': float(ret_mean),
                'return_std': float(ret_std),
                'win_rate': float(win_rate),
                'ma_ratio': float(ma_ratio),
                'volatility': float(volatility),
                'avg_rsi': float(avg_rsi),
                'sample_count': len(cluster_data)
            }
            
        return analysis
        
    def _map_clusters_to_regimes(self, df_labeled: pd.DataFrame, 
                              cluster_analysis: Dict[int, Dict[str, float]]) -> Dict[int, MarketRegime]:
        """
        Map numeric clusters to named market regimes.
        
        Args:
            df_labeled: DataFrame with cluster labels
            cluster_analysis: Analysis of cluster characteristics
            
        Returns:
            Mapping from cluster IDs to market regimes
        """
        regime_mapping = {}
        
        for cluster_id, metrics in cluster_analysis.items():
            ret_mean = metrics['return_mean']
            volatility = metrics['volatility']
            ma_ratio = metrics['ma_ratio'] 
            
            # Assign regimes based on cluster characteristics
            if ret_mean > 0.001 and ma_ratio > 1.01:
                # Strong positive returns with MA confirmation
                regime = MarketRegime.STRONG_UPTREND
            elif ret_mean > 0.0005 and ma_ratio > 1.0:
                # Modest positive returns
                regime = MarketRegime.WEAK_UPTREND
            elif ret_mean < -0.001 and ma_ratio < 0.99:
                # Strong negative returns with MA confirmation
                regime = MarketRegime.STRONG_DOWNTREND
            elif ret_mean < -0.0005 and ma_ratio < 1.0:
                # Modest negative returns
                regime = MarketRegime.WEAK_DOWNTREND
            elif volatility > 0.015:
                # High volatility without clear direction
                regime = MarketRegime.VOLATILE
            else:
                # Low volatility, no clear direction
                regime = MarketRegime.RANGING
                
            regime_mapping[cluster_id] = regime
            
        return regime_mapping
        
    def classify_regime(self, df: pd.DataFrame, window_size: int = 10) -> Tuple[MarketRegime, float]:
        """
        Classify the current market regime.
        
        Args:
            df: DataFrame with OHLCV and indicator data
            window_size: Window of recent data to use for classification
            
        Returns:
            Tuple of (market_regime, confidence)
        """
        if self.kmeans is None or self.scaler is None:
            self.logger.warning("Regime classifier not trained. Using heuristic classification.")
            return self._heuristic_classification(df, window_size)
            
        # Get recent data
        recent_data = df.iloc[-window_size:].copy()
        
        # Extract features
        features_df = self._extract_features(recent_data)
        
        if features_df.empty:
            return MarketRegime.UNKNOWN, 0.0
            
        # Scale features
        scaled_features = self.scaler.transform(features_df.values)
        
        # Predict clusters
        cluster_predictions = self.kmeans.predict(scaled_features)
        
        # Get most common cluster
        unique_clusters, counts = np.unique(cluster_predictions, return_counts=True)
        most_common_cluster = unique_clusters[np.argmax(counts)]
        confidence = np.max(counts) / len(cluster_predictions)
        
        # Check if we need to detect special regimes not covered by clustering
        special_regime, special_confidence = self._detect_special_regimes(recent_data)
        
        if special_confidence > confidence:
            regime = special_regime
            confidence = special_confidence
        else:
            # Map cluster to regime
            try:
                # Try to get from mapping (if previously trained with mapping)
                # This would need to be loaded separately
                regime = self._map_cluster_to_regime(most_common_cluster, recent_data)
            except:
                # Fallback to heuristic classification
                regime, fallback_confidence = self._heuristic_classification(df, window_size)
                if fallback_confidence > confidence:
                    confidence = fallback_confidence
        
        # Update current regime state
        self.current_regime = regime
        self.regime_confidence = confidence
        
        # Add to history (limit to last 10 classifications)
        self.regime_history.append((regime, confidence, datetime.now()))
        if len(self.regime_history) > 10:
            self.regime_history = self.regime_history[-10:]
            
        return regime, confidence
        
    def _map_cluster_to_regime(self, cluster_id: int, recent_data: pd.DataFrame) -> MarketRegime:
        """
        Map a cluster ID to a market regime using recent data context.
        
        Args:
            cluster_id: The cluster ID from KMeans
            recent_data: Recent price data for context
            
        Returns:
            MarketRegime corresponding to the cluster
        """
        # Calculate key metrics for this data
        ret_mean = recent_data['close'].pct_change().mean()
        volatility = recent_data['close'].pct_change().std()
        
        # Moving average relationship
        if 'sma_20' in recent_data.columns and 'sma_50' in recent_data.columns:
            ma_ratio = (recent_data['sma_20'] / recent_data['sma_50']).iloc[-1]
        else:
            sma20 = recent_data['close'].rolling(20).mean()
            sma50 = recent_data['close'].rolling(50).mean()
            ma_ratio = sma20.iloc[-1] / sma50.iloc[-1] if not sma50.iloc[-1] == 0 else 1.0
            
        # RSI level if available
        if 'rsi' in recent_data.columns:
            rsi_level = recent_data['rsi'].iloc[-1]
        else:
            rsi_level = 50  # Neutral default
            
        # Determine regime based on metrics
        if ret_mean > 0.001 and ma_ratio > 1.01:
            return MarketRegime.STRONG_UPTREND
        elif ret_mean > 0.0005 and ma_ratio > 1.0:
            return MarketRegime.WEAK_UPTREND
        elif ret_mean < -0.001 and ma_ratio < 0.99:
            return MarketRegime.STRONG_DOWNTREND
        elif ret_mean < -0.0005 and ma_ratio < 1.0:
            return MarketRegime.WEAK_DOWNTREND
        elif volatility > 0.015:
            return MarketRegime.VOLATILE
        elif rsi_level > 70 and ma_ratio > 1.02:
            return MarketRegime.REVERSAL  # Potential reversal from overbought
        elif rsi_level < 30 and ma_ratio < 0.98:
            return MarketRegime.REVERSAL  # Potential reversal from oversold
        else:
            return MarketRegime.RANGING
            
    def _heuristic_classification(self, df: pd.DataFrame, window_size: int = 10) -> Tuple[MarketRegime, float]:
        """
        Use heuristics to classify market regime when no model is available.
        
        Args:
            df: DataFrame with OHLCV data
            window_size: Window of recent data to use
            
        Returns:
            Tuple of (market_regime, confidence)
        """
        if len(df) < window_size + 20:  # Need some history
            return MarketRegime.UNKNOWN, 0.0
            
        # Get recent data
        recent = df.iloc[-window_size:].copy()
        
        # Calculate some basic metrics
        returns = recent['close'].pct_change().dropna()
        volatility = returns.std()
        trend = (recent['close'].iloc[-1] / recent['close'].iloc[0]) - 1
        
        # Moving averages
        if 'sma_20' not in recent.columns or 'sma_50' not in recent.columns:
            sma20 = df['close'].rolling(20).mean()
            sma50 = df['close'].rolling(50).mean()
        else:
            sma20 = recent['sma_20']
            sma50 = recent['sma_50']
            
        ma_trend = (sma20.iloc[-1] / sma50.iloc[-1]) - 1
        
        # RSI if available
        if 'rsi' in recent.columns:
            rsi = recent['rsi'].iloc[-1]
            rsi_trend = recent['rsi'].diff(3).iloc[-1]
        else:
            # Simple RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1]))
            rsi_trend = 0
            
        # Breakout detection
        if 'bb_upper' in recent.columns and 'bb_lower' in recent.columns:
            # Check if price is breaking out of Bollinger Bands
            upper_break = recent['close'].iloc[-1] > recent['bb_upper'].iloc[-1]
            lower_break = recent['close'].iloc[-1] < recent['bb_lower'].iloc[-1]
            breakout = upper_break or lower_break
        else:
            breakout = False
            
        # Volume trend if available
        if 'volume' in recent.columns:
            vol_trend = (recent['volume'].iloc[-5:].mean() / 
                         recent['volume'].iloc[:5].mean()) - 1
        else:
            vol_trend = 0
            
        # Classify based on heuristics
        confidence = 0.6  # Base confidence level for heuristics
        
        if trend > 0.03 and ma_trend > 0.01 and vol_trend > 0.1:
            regime = MarketRegime.STRONG_UPTREND
            confidence = 0.8
        elif trend > 0.01 and ma_trend > 0:
            regime = MarketRegime.WEAK_UPTREND
            confidence = 0.7
        elif trend < -0.03 and ma_trend < -0.01 and vol_trend > 0.1:
            regime = MarketRegime.STRONG_DOWNTREND
            confidence = 0.8
        elif trend < -0.01 and ma_trend < 0:
            regime = MarketRegime.WEAK_DOWNTREND
            confidence = 0.7
        elif volatility > 0.02 and abs(trend) < 0.01:
            regime = MarketRegime.RANGING
            confidence = 0.6
        elif volatility > 0.03:
            regime = MarketRegime.VOLATILE
            confidence = 0.7
        elif breakout and vol_trend > 0.3:
            regime = MarketRegime.BREAKOUT
            confidence = 0.75
        elif (rsi > 70 and rsi_trend < 0) or (rsi < 30 and rsi_trend > 0):
            regime = MarketRegime.REVERSAL
            confidence = 0.65
        else:
            regime = MarketRegime.RANGING
            confidence = 0.5
            
        return regime, confidence
        
    def _detect_special_regimes(self, df: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Detect special market regimes that may not be captured by clustering.
        
        Args:
            df: DataFrame with OHLCV and indicator data
            
        Returns:
            Tuple of (detected_special_regime, confidence)
        """
        # Default return values
        special_regime = MarketRegime.UNKNOWN
        confidence = 0.0
        
        # Breakout detection
        if self._detect_breakout(df):
            special_regime = MarketRegime.BREAKOUT
            confidence = 0.75
            
        # Reversal detection
        reversal_confidence = self._detect_reversal(df)
        if reversal_confidence > confidence:
            special_regime = MarketRegime.REVERSAL
            confidence = reversal_confidence
            
        return special_regime, confidence
        
    def _detect_breakout(self, df: pd.DataFrame) -> float:
        """
        Detect breakout patterns.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Confidence score for breakout (0.0-1.0)
        """
        if len(df) < 10:
            return 0.0
            
        # Check if Bollinger Bands are available
        if all(band in df.columns for band in ['bb_upper', 'bb_lower', 'bb_middle']):
            # Price breaking outside bands
            upper_break = df['close'].iloc[-1] > df['bb_upper'].iloc[-1]
            lower_break = df['close'].iloc[-1] < df['bb_lower'].iloc[-1]
            
            # Check if bands were narrow (squeeze) before breakout
            if 'bb_width' in df.columns:
                bb_squeeze = df['bb_width'].iloc[-3] < df['bb_width'].iloc[-10:].mean() * 0.8
            else:
                bb_width = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                bb_squeeze = bb_width.iloc[-3] < bb_width.iloc[-10:].mean() * 0.8
                
            # Volume confirmation if available
            if 'volume' in df.columns:
                vol_confirm = df['volume'].iloc[-1] > df['volume'].iloc[-5:].mean() * 1.5
            else:
                vol_confirm = False
                
            # Calculate confidence
            if (upper_break or lower_break) and bb_squeeze:
                confidence = 0.7
                if vol_confirm:
                    confidence = 0.85
                return confidence
                
        return 0.0
        
    def _detect_reversal(self, df: pd.DataFrame) -> float:
        """
        Detect potential market reversals.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Confidence score for reversal (0.0-1.0)
        """
        if len(df) < 20:
            return 0.0
            
        # Get recent price trend
        recent_trend = df['close'].iloc[-5:].pct_change().mean()
        
        # Check RSI divergence if available
        if 'rsi' in df.columns:
            rsi = df['rsi']
            price = df['close']
            
            # Bearish divergence: price higher but RSI lower
            if (price.iloc[-1] > price.iloc[-5] and 
                rsi.iloc[-1] < rsi.iloc[-5] and 
                rsi.iloc[-1] > 70):
                return 0.75
                
            # Bullish divergence: price lower but RSI higher
            if (price.iloc[-1] < price.iloc[-5] and 
                rsi.iloc[-1] > rsi.iloc[-5] and 
                rsi.iloc[-1] < 30):
                return 0.75
                
        # Check for candlestick reversal patterns
        # (simplified version - a full implementation would check more patterns)
        if recent_trend > 0.02:  # In an uptrend
            # Check for bearish engulfing or shooting star
            if (df['open'].iloc[-1] > df['close'].iloc[-1] and  # Bearish candle
                df['open'].iloc[-1] > df['high'].iloc[-2] and  # Opens above previous high
                df['close'].iloc[-1] < df['open'].iloc[-2]):  # Closes below previous open
                return 0.7
                
        elif recent_trend < -0.02:  # In a downtrend
            # Check for bullish engulfing or hammer
            if (df['open'].iloc[-1] < df['close'].iloc[-1] and  # Bullish candle
                df['open'].iloc[-1] < df['low'].iloc[-2] and  # Opens below previous low
                df['close'].iloc[-1] > df['open'].iloc[-2]):  # Closes above previous open
                return 0.7
                
        return 0.0
        
    def optimize_strategy_parameters(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Get optimized trading parameters for the current market regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Dict with optimized parameter values
        """
        # Base parameter set
        params = {
            'signal_threshold': 3.0,
            'risk_percentage': 1.0,
            'stop_loss_pct': 1.0,
            'take_profit_pct': 2.0,
            'trailing_stop_pct': 1.0,
            'position_size_factor': 1.0,
            'use_dynamic_exits': True
        }
        
        # Adjust parameters based on market regime
        if regime == MarketRegime.STRONG_UPTREND:
            params.update({
                'signal_threshold': 2.5,  # Lower threshold to enter more trades
                'risk_percentage': 1.2,   # Increase risk allocation
                'stop_loss_pct': 1.2,     # Wider stop loss
                'take_profit_pct': 2.5,   # Higher take profit
                'trailing_stop_pct': 1.5, # Wider trailing stop
                'position_size_factor': 1.2  # Increase position size
            })
        elif regime == MarketRegime.WEAK_UPTREND:
            params.update({
                'signal_threshold': 2.8,
                'risk_percentage': 1.0,
                'stop_loss_pct': 1.0,
                'take_profit_pct': 2.0,
                'trailing_stop_pct': 1.2,
                'position_size_factor': 1.0
            })
        elif regime == MarketRegime.RANGING:
            params.update({
                'signal_threshold': 3.5,  # Higher threshold to avoid false signals
                'risk_percentage': 0.8,   # Reduce risk
                'stop_loss_pct': 0.8,     # Tighter stop loss
                'take_profit_pct': 1.5,   # Lower take profit
                'trailing_stop_pct': 0.8, # Tighter trailing stop
                'position_size_factor': 0.8  # Smaller positions
            })
        elif regime == MarketRegime.WEAK_DOWNTREND:
            params.update({
                'signal_threshold': 4.0,  # Much higher threshold for entry
                'risk_percentage': 0.6,   # Reduce risk significantly
                'stop_loss_pct': 0.7,     # Tighter stop loss
                'take_profit_pct': 1.8,   # Adjust take profit
                'trailing_stop_pct': 0.7, # Tighter trailing stop
                'position_size_factor': 0.7  # Smaller positions
            })
        elif regime == MarketRegime.STRONG_DOWNTREND:
            params.update({
                'signal_threshold': 4.5,  # Very high threshold for entry
                'risk_percentage': 0.5,   # Minimum risk
                'stop_loss_pct': 0.6,     # Tight stop loss
                'take_profit_pct': 1.5,   # Adjust take profit
                'trailing_stop_pct': 0.5, # Tight trailing stop
                'position_size_factor': 0.5,  # Minimum position size
                'use_dynamic_exits': False  # Fixed exits in strong downtrends
            })
        elif regime == MarketRegime.VOLATILE:
            params.update({
                'signal_threshold': 3.8,  # Higher threshold to avoid noise
                'risk_percentage': 0.7,   # Reduced risk
                'stop_loss_pct': 1.3,     # Wider stop loss to handle volatility
                'take_profit_pct': 2.2,   # Higher take profit to compensate
                'trailing_stop_pct': 1.3, # Wider trailing stop
                'position_size_factor': 0.7  # Smaller positions
            })
        elif regime == MarketRegime.BREAKOUT:
            params.update({
                'signal_threshold': 2.5,  # Lower threshold to catch breakouts
                'risk_percentage': 1.1,   # Slightly higher risk
                'stop_loss_pct': 1.0,     # Standard stop loss
                'take_profit_pct': 2.3,   # Higher take profit
                'trailing_stop_pct': 1.2, # Wider trailing stop
                'position_size_factor': 1.1  # Slightly larger positions
            })
        elif regime == MarketRegime.REVERSAL:
            params.update({
                'signal_threshold': 3.0,
                'risk_percentage': 0.9,
                'stop_loss_pct': 0.9,
                'take_profit_pct': 2.0,
                'trailing_stop_pct': 0.9,
                'position_size_factor': 0.9
            })
            
        return params
        
    def get_regime_info(self) -> Dict[str, Any]:
        """
        Get information about the current market regime and history.
        
        Returns:
            Dict with regime information
        """
        # Get current regime info
        regime_info = {
            'current_regime': self.current_regime.value,
            'confidence': self.regime_confidence,
            'parameters': self.optimize_strategy_parameters(self.current_regime),
            'history': [
                {
                    'regime': r.value,
                    'confidence': c,
                    'timestamp': t.isoformat()
                }
                for r, c, t in self.regime_history
            ]
        }
        
        return regime_info
        
    def adjust_signal_strength(self, base_signal_strength: float, regime: Optional[MarketRegime] = None) -> float:
        """
        Adjust signal strength based on market regime.
        
        Args:
            base_signal_strength: Original signal strength (0.0-1.0)
            regime: Market regime to use (defaults to current regime)
            
        Returns:
            Adjusted signal strength
        """
        if regime is None:
            regime = self.current_regime
            
        # Adjustment factors for different regimes
        regime_factors = {
            MarketRegime.STRONG_UPTREND: 1.2,
            MarketRegime.WEAK_UPTREND: 1.1,
            MarketRegime.RANGING: 0.8,
            MarketRegime.WEAK_DOWNTREND: 0.7,
            MarketRegime.STRONG_DOWNTREND: 0.5,
            MarketRegime.VOLATILE: 0.75,
            MarketRegime.BREAKOUT: 1.15,
            MarketRegime.REVERSAL: 0.9,
            MarketRegime.UNKNOWN: 1.0
        }
        
        # Apply adjustment
        factor = regime_factors.get(regime, 1.0)
        adjusted_strength = base_signal_strength * factor
        
        # Ensure it stays within valid range
        return max(0.0, min(1.0, adjusted_strength)) 