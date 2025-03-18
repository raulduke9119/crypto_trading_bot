"""
Konfigurationsdatei für den Binance Trading Bot.
Enthält alle wichtigen Einstellungen und Parameter.
"""
from typing import Dict, List, Optional, Union, Tuple
import os
from dotenv import load_dotenv
import logging

# Lade Umgebungsvariablen aus .env Datei
load_dotenv()

# API-Konfiguration
def validate_api_keys(use_testnet: bool = True) -> Tuple[str, str]:
    if use_testnet:
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        key_names = "BINANCE_TESTNET_API_KEY und BINANCE_TESTNET_API_SECRET"
    else:
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        key_names = "BINANCE_API_KEY und BINANCE_API_SECRET"
    
    if not api_key or not api_secret:
        raise ValueError(
            f"API-Keys fehlen! Bitte stelle sicher, dass {key_names} "
            "in der .env Datei definiert sind."
        )
    return api_key, api_secret

# Verwende Testnet für Tests
USE_TESTNET: bool = True  # Changed to True for testing
API_KEY, API_SECRET = validate_api_keys(USE_TESTNET)

# Binance-Konfiguration
BINANCE_CONFIG: Dict[str, Union[str, bool]] = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "USE_TESTNET": USE_TESTNET,
    "BASE_URL": "https://testnet.binance.vision" if USE_TESTNET else "https://api.binance.com",
    "WEBSOCKET_URL": "wss://testnet.binance.vision/ws" if USE_TESTNET else "wss://stream.binance.com:9443/ws"
}

# Trading-Parameter - Updated for optimized performance
TRADING_SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
DEFAULT_TIMEFRAME: str = "5m"  # 5m timeframe has shown best risk/reward ratio

# Risikomanagement - Optimized values based on backtesting
MAX_POSITIONS: int = 3  # Increased to 3 for better diversification while maintaining control
RISK_PERCENTAGE: float = 0.8  # Slightly more conservative risk per trade
MAX_DRAWDOWN_PCT: float = 3.0  # Reduced from 5.0% for tighter risk control
TRAILING_STOP_PCT: float = 0.8  # Optimized based on volatility testing
VOLATILITY_ADJUSTMENT: bool = True  # Activate volatility adjustment
INITIAL_CAPITAL: float = 1000.0  # Starting capital in USDT

# Default pattern to use
DEFAULT_PATTERN: str = "superoptimized_pattern.json"  # Using our new optimized pattern

# Technische Indikatoren Konfiguration - Fine-tuned parameters
INDICATORS_CONFIG: Dict[str, Dict[str, Union[int, float]]] = {
    "SMA": {"short": 10, "medium": 50, "long": 200},  # Adjusted short term to 10
    "EMA": {"short": 5, "medium": 12, "long": 26},    # Using 5/12/26 for better responsiveness
    "RSI": {
        "period": 14,
        "overbought": 72,  # Slightly higher than standard for crypto markets
        "oversold": 35,    # Not too low to avoid false signals
        "momentum_period": 3  # For RSI-Momentum
    },
    "MACD": {
        "fast": 12,
        "slow": 26,
        "signal": 9,
        "min_strength": 0.05  # Reduced for higher sensitivity
    },
    "BBANDS": {
        "period": 20,
        "std_dev": 2.0,
        "squeeze_factor": 1.5  # For Bollinger Squeeze
    },
    "ATR": {
        "period": 14,
        "multiplier": 1.5  # Reduced from 2.0 for tighter stops
    },
    "STOCH": {
        "k_period": 14,
        "d_period": 3,
        "slowing": 3,
        "overbought": 80,
        "oversold": 20
    },
    "VOLUME": {
        "ma_period": 10,  # Reduced from 20 for more responsive signals
        "min_ratio": 1.1,  # Slightly reduced threshold
        "trend_period": 3  # For Volumen-Trend
    }
}

# Verzeichnis-Konfiguration
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIRECTORY: str = os.path.join(BASE_DIR, "data")  # Verzeichnis für gespeicherte Daten
MODEL_DIRECTORY: str = os.path.join(BASE_DIR, "models")  # Verzeichnis für gespeicherte Modelle
LOG_DIRECTORY: str = os.path.join(BASE_DIR, "logs")  # Verzeichnis für Logs

# Stelle sicher, dass alle Verzeichnisse existieren
for directory in [DATA_DIRECTORY, MODEL_DIRECTORY, LOG_DIRECTORY]:
    os.makedirs(directory, exist_ok=True)

# Daten-Konfiguration
HISTORICAL_DATA_DAYS: int = 30  # Anzahl der Tage für historische Daten

# Logging-Konfiguration
LOG_LEVEL: str = logging.INFO
LOG_FILE: str = os.path.join(LOG_DIRECTORY, "trading_bot.log")

# Backtesting-Konfiguration
BACKTEST_START_DATE: str = "2023-01-01"
BACKTEST_END_DATE: Optional[str] = None  # None für "bis heute"

# Neuronales Netz Konfiguration
USE_ML: bool = False  # Disabled by default until properly optimized
ML_FEATURES: List[str] = [
    "open", "high", "low", "close", "volume",
    "sma_short", "sma_medium", "sma_long",
    "ema_short", "ema_medium", "ema_long",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower",
    "atr", "stoch_k", "stoch_d", "volatility"
]
ML_PREDICTION_HORIZON: int = 3  # Vorhersagehorizont in Zeiteinheiten (z.B. 3 Stunden bei 1h Timeframe)
ML_TRAIN_TEST_SPLIT: float = 0.2  # Anteil der Daten für Testset (0.2 = 20%)
