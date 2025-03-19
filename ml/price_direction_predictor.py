"""
Price Direction Predictor Model for Binance Trading Bot.
Implements an LSTM neural network to predict short-term price movement direction
based on technical indicators and historical price data.
"""
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Ensure TensorFlow logging is not too verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Constants
SEQUENCE_LENGTH = 20  # Number of time steps to look back
PREDICTION_HORIZON = 6  # Number of candles to predict ahead
FEATURE_COLUMNS = [
    'close', 'high', 'low', 'volume',  # Price and volume
    'rsi', 'macd', 'macd_signal', 'macd_hist',  # Momentum
    'adx', 'plus_di', 'minus_di',  # Trend strength
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',  # Volatility
    'ema_12', 'ema_26', 'sma_50', 'sma_200',  # Moving averages
]

class PriceDirectionPredictor:
    """
    Neural network model that predicts the price movement direction for a given symbol.
    Uses LSTM architecture to capture temporal dependencies in the market data.
    """
    
    def __init__(self, symbol: str, timeframe: str = '1h', model_dir: str = 'ml/models'):
        """
        Initialize the price direction predictor model.
        
        Args:
            symbol: Trading symbol to predict (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            model_dir: Directory to save/load model files
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, f'price_direction_{symbol}_{timeframe}.h5')
        self.scaler_path = os.path.join(model_dir, f'scaler_{symbol}_{timeframe}.pkl')
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
        # Load existing model if available
        self._load_model()
        
        # Logger setup
        self.logger = logging.getLogger('price_direction_predictor')
        
    def _load_model(self) -> bool:
        """
        Load pre-trained model from disk if available.
        
        Returns:
            bool: True if model was loaded, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                
                # Load scaler
                if os.path.exists(self.scaler_path):
                    import joblib
                    self.scaler = joblib.load(self.scaler_path)
                
                self.is_trained = True
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build LSTM neural network model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
        """
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25, activation='relu'),
            Dense(units=1, activation='sigmoid')  # Binary classification (up/down)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and normalize data for training or prediction.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            Tuple of X (feature sequences) and y (target labels) arrays
        """
        # Select and validate features
        available_columns = [col for col in FEATURE_COLUMNS if col in df.columns]
        if len(available_columns) < 5:
            raise ValueError(f"Insufficient features. Found only {available_columns}")
            
        # Create target: 1 if price goes up after prediction horizon, 0 otherwise
        df['target'] = (df['close'].shift(-PREDICTION_HORIZON) > df['close']).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        # Extract features and target
        features = df[available_columns].values
        targets = df['target'].values
        
        # Scale features
        if not self.is_trained:
            # Only fit scaler during training
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)
            
        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1):
            X.append(features_scaled[i:i+SEQUENCE_LENGTH])
            y.append(targets[i+SEQUENCE_LENGTH-1])
            
        return np.array(X), np.array(y)
        
    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the model using historical data with indicators.
        
        Args:
            df: DataFrame with OHLCV data and all required indicators
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Dictionary with training metrics
        """
        try:
            # Prepare data
            X, y = self._prepare_data(df)
            
            # Skip training if not enough data
            if len(X) < 100:
                raise ValueError(f"Insufficient data for training: {len(X)} sequences")
                
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Build model if not loaded
            if self.model is None:
                self._build_model(input_shape=(X.shape[1], X.shape[2]))
                
            # Early stopping to prevent overfitting
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=1
            )
            
            # Evaluate model
            evaluation = self.model.evaluate(X_test, y_test, verbose=0)
            metrics = {
                'loss': evaluation[0],
                'accuracy': evaluation[1],
                'training_samples': len(X_train),
                'testing_samples': len(X_test)
            }
            
            # Save model and scaler
            self._save_model()
            self.is_trained = True
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
            
    def _save_model(self) -> None:
        """Save trained model and scaler to disk."""
        if self.model is not None:
            self.model.save(self.model_path)
            
            # Save scaler
            import joblib
            joblib.dump(self.scaler, self.scaler_path)
            
    def predict(self, df: pd.DataFrame) -> Tuple[float, str, Dict[str, float]]:
        """
        Predict price movement direction using the most recent data.
        
        Args:
            df: DataFrame with recent OHLCV data and all required indicators
            
        Returns:
            Tuple containing:
                - Probability of price moving up (0-1)
                - Signal direction ('buy', 'sell', or 'neutral')
                - Dictionary with additional prediction metrics
        """
        if not self.is_trained or self.model is None:
            return 0.5, 'neutral', {'error': 'Model not trained'}
            
        try:
            # Prepare data (only need the last sequence)
            X, _ = self._prepare_data(df)
            
            if len(X) == 0:
                return 0.5, 'neutral', {'error': 'Insufficient data for prediction'}
                
            # Get the most recent sequence
            latest_sequence = X[-1:]
            
            # Make prediction
            prediction = self.model.predict(latest_sequence, verbose=0)[0][0]
            
            # Interpret prediction
            if prediction > 0.6:
                signal = 'buy'
            elif prediction < 0.4:
                signal = 'sell'
            else:
                signal = 'neutral'
                
            # Calculate recent accuracy (if possible)
            accuracy = 0.0
            if len(X) > 20:
                recent_preds = self.model.predict(X[-20:], verbose=0).flatten()
                if 'target' in df.columns:
                    recent_targets = df['target'].iloc[-20:].values
                    if len(recent_targets) >= 20:
                        binary_preds = (recent_preds > 0.5).astype(int)
                        accuracy = np.mean(binary_preds == recent_targets[:len(binary_preds)])
                
            metrics = {
                'probability': float(prediction),
                'confidence': abs(prediction - 0.5) * 2,  # Scale 0-1: 0=neutral, 1=confident
                'recent_accuracy': float(accuracy),
                'timestamp': datetime.now().isoformat()
            }
            
            return float(prediction), signal, metrics
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return 0.5, 'neutral', {'error': str(e)}
            
    def calculate_signal_strength(self, prediction_probability: float) -> float:
        """
        Calculate signal strength based on prediction probability.
        
        Args:
            prediction_probability: Model's predicted probability (0-1)
            
        Returns:
            Signal strength between 0-1
        """
        # Convert probability to signal strength (0.5 = neutral, 0/1 = strong)
        return min(1.0, abs(prediction_probability - 0.5) * 4)  # Amplify small differences 