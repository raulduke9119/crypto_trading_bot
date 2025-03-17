"""
Vorhersagemodell-Modul für den Trading Bot.
Implementiert ein neuronales Netz zur Vorhersage von Preisbewegungen.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from typing import Dict, List, Optional, Union, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from config.config import ML_FEATURES, ML_PREDICTION_HORIZON, ML_TRAIN_TEST_SPLIT, MODEL_DIRECTORY, LOG_LEVEL, LOG_FILE

# Logger einrichten
logger = setup_logger(os.path.join(MODEL_DIRECTORY, LOG_FILE), LOG_LEVEL)

class PredictionModel:
    """
    Klasse für die Preisvorhersage mit maschinellem Lernen.
    """
    
    def __init__(self, symbol: str, model_dir: str = MODEL_DIRECTORY):
        """
        Initialisiert das Vorhersagemodell.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            model_dir: Verzeichnis zum Speichern/Laden des Modells
        """
        self.symbol = symbol
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, f"{symbol}_prediction_model.h5")
        self.scaler_path = os.path.join(model_dir, f"{symbol}_scaler.pkl")
        
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        
        # Erstelle Modellverzeichnis, falls es nicht existiert
        os.makedirs(model_dir, exist_ok=True)
        
        # Versuche, ein bestehendes Modell zu laden
        self._load_model()
        
        logger.info(f"Vorhersagemodell für {symbol} initialisiert")
    
    def _load_model(self) -> bool:
        """
        Lädt ein gespeichertes Modell, falls verfügbar.
        
        Returns:
            True, wenn das Modell erfolgreich geladen wurde, sonst False
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = load_model(self.model_path)
                scalers = joblib.load(self.scaler_path)
                self.scaler_X = scalers['scaler_X']
                self.scaler_y = scalers['scaler_y']
                logger.info(f"Bestehendes Vorhersagemodell für {self.symbol} geladen")
                return True
            return False
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells für {self.symbol}: {e}")
            return False
    
    def _save_model(self) -> bool:
        """
        Speichert das aktuelle Modell.
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            if self.model is not None:
                self.model.save(self.model_path)
                scalers = {'scaler_X': self.scaler_X, 'scaler_y': self.scaler_y}
                joblib.dump(scalers, self.scaler_path)
                logger.info(f"Vorhersagemodell für {self.symbol} gespeichert")
                return True
            return False
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Modells für {self.symbol}: {e}")
            return False
    
    def _prepare_features(self, 
                         df: pd.DataFrame, 
                         prediction_horizon: int = ML_PREDICTION_HORIZON,
                         features: List[str] = ML_FEATURES) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bereitet Features für das Modell vor.
        
        Args:
            df: DataFrame mit Indikatoren
            prediction_horizon: Anzahl der Zeiteinheiten für die Vorhersage
            features: Liste der zu verwendenden Features
            
        Returns:
            X: Feature-Matrix
            y: Zielwerte
        """
        try:
            # Erstelle eine Kopie der Daten
            data = df.copy()
            
            # Erstelle die Zielvariable: zukünftige Preisänderung
            data['future_return'] = data['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
            
            # Filtere gültige Zeilen
            data = data.dropna()
            
            # Prüfe, ob alle angeforderten Features im DataFrame vorhanden sind
            available_features = [f for f in features if f in data.columns]
            if len(available_features) < len(features):
                missing = set(features) - set(available_features)
                logger.warning(f"Fehlende Features: {missing}")
            
            # Bereite Eingaben und Ziele vor
            X = data[available_features].values
            y = data['future_return'].values.reshape(-1, 1)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Fehler beim Vorbereiten der Features: {e}")
            return np.array([]), np.array([])
    
    def _create_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Erstellt ein LSTM-Modell für Zeitreihenvorhersage.
        
        Args:
            input_shape: Form der Eingabedaten (Sequenzlänge, Anzahl Features)
            
        Returns:
            Konfiguriertes LSTM-Modell
        """
        model = Sequential([
            LSTM(units=64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=32),
            Dropout(0.2),
            Dense(units=16, activation='relu'),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _create_dense_model(self, input_dim: int) -> Sequential:
        """
        Erstellt ein einfaches Dense-Modell für Regressionsaufgaben.
        
        Args:
            input_dim: Anzahl der Eingabefeatures
            
        Returns:
            Konfiguriertes Dense-Modell
        """
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _prepare_sequences(self, 
                          X: np.ndarray, 
                          y: np.ndarray, 
                          seq_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bereitet Sequenzen für LSTM-Modelle vor.
        
        Args:
            X: Feature-Matrix
            y: Zielwerte
            seq_length: Länge der Sequenzen
            
        Returns:
            X_seq: Sequenzierte Feature-Matrix
            y_seq: Entsprechende Zielwerte
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, 
             df: pd.DataFrame, 
             use_lstm: bool = False,
             seq_length: int = 10,
             epochs: int = 100,
             batch_size: int = 32,
             validation_split: float = 0.1) -> bool:
        """
        Trainiert das Modell.
        
        Args:
            df: DataFrame mit Indikatoren
            use_lstm: Ob ein LSTM-Modell verwendet werden soll
            seq_length: Länge der Sequenzen für LSTM
            epochs: Anzahl der Trainingsepochen
            batch_size: Batch-Größe für das Training
            validation_split: Anteil der Daten für Validierung
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Bereite Features und Ziele vor
            X, y = self._prepare_features(df)
            
            if X.size == 0 or y.size == 0:
                logger.error("Keine gültigen Daten für das Training")
                return False
            
            # Teile die Daten
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ML_TRAIN_TEST_SPLIT, random_state=42
            )
            
            # Skaliere die Daten
            X_train = self.scaler_X.fit_transform(X_train)
            X_test = self.scaler_X.transform(X_test)
            
            y_train = self.scaler_y.fit_transform(y_train)
            y_test = self.scaler_y.transform(y_test)
            
            # Erstelle und trainiere das Modell
            if use_lstm:
                # Bereite Sequenzen für LSTM vor
                X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train, seq_length)
                X_test_seq, y_test_seq = self._prepare_sequences(X_test, y_test, seq_length)
                
                # Erstelle LSTM-Modell
                self.model = self._create_lstm_model((seq_length, X_train.shape[1]))
                
                # Definiere Callbacks
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ModelCheckpoint(self.model_path, save_best_only=True)
                ]
                
                # Trainiere das Modell
                history = self.model.fit(
                    X_train_seq, y_train_seq,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Evaluiere das Modell
                y_pred = self.model.predict(X_test_seq)
                
            else:
                # Erstelle Dense-Modell
                self.model = self._create_dense_model(X_train.shape[1])
                
                # Definiere Callbacks
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ModelCheckpoint(self.model_path, save_best_only=True)
                ]
                
                # Trainiere das Modell
                history = self.model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Evaluiere das Modell
                y_pred = self.model.predict(X_test)
            
            # Transformiere zurück für Metriken
            y_test_orig = self.scaler_y.inverse_transform(y_test)
            y_pred_orig = self.scaler_y.inverse_transform(y_pred)
            
            # Berechne Metriken
            mse = mean_squared_error(y_test_orig, y_pred_orig)
            r2 = r2_score(y_test_orig, y_pred_orig)
            
            logger.info(f"Modell für {self.symbol} trainiert - MSE: {mse:.6f}, R²: {r2:.4f}")
            
            # Speichere das Modell
            self._save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Trainieren des Modells für {self.symbol}: {e}")
            return False
    
    def predict(self, df: pd.DataFrame, use_lstm: bool = False, seq_length: int = 10) -> Optional[float]:
        """
        Macht Vorhersagen mit dem Modell.
        
        Args:
            df: DataFrame mit Indikatoren
            use_lstm: Ob ein LSTM-Modell verwendet wurde
            seq_length: Länge der Sequenzen für LSTM
            
        Returns:
            Vorhersage: Prozentuale Preisänderung oder None bei Fehler
        """
        try:
            if self.model is None:
                logger.warning(f"Kein Modell verfügbar für {self.symbol}")
                return None
            
            # Bereite Features vor
            features = [f for f in ML_FEATURES if f in df.columns]
            
            # Hole die neuesten Daten
            X = df[features].iloc[-seq_length:].values if use_lstm else df[features].iloc[-1:].values
            
            # Skaliere die Daten
            X = self.scaler_X.transform(X)
            
            # Mache Vorhersage
            if use_lstm:
                # Reshape für LSTM
                X = X.reshape(1, seq_length, X.shape[1])
                pred = self.model.predict(X)
            else:
                pred = self.model.predict(X)
            
            # Inverse-Skaliere die Vorhersage
            pred = self.scaler_y.inverse_transform(pred)
            
            logger.info(f"Vorhersage für {self.symbol}: {pred[0][0]:.2%}")
            return float(pred[0][0])
            
        except Exception as e:
            logger.error(f"Fehler bei der Vorhersage für {self.symbol}: {e}")
            return None
