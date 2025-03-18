"""
Breakout Pattern für fortgeschrittene Handelssignale.
Dieses Pattern sucht nach Ausbrüchen aus Konsolidierungsphasen.
"""

def get_pattern():
    """
    Returns the pattern configuration.
    
    This is required for pattern loader to recognize this as a valid pattern.
    """
    return {
        "name": "Breakout Trading Pattern",
        "description": "Erkennt Ausbrüche aus Konsolidierungsphasen mit Volumenbestätigung",
        "buy_conditions": [
            {
                "name": "Price Breakout Up",
                "conditions": [
                    {"indicator": "close", "operator": ">", "indicator_max": "high", "period": 20, "offset": 1},
                    {"indicator": "volume", "operator": ">", "indicator_ma": "volume", "ma_period": 10, "factor": 1.5}
                ],
                "weight": 3.5
            },
            {
                "name": "Narrow Range Breakout",
                "conditions": [
                    {"indicator": "atr", "operator": "<", "indicator_ma": "atr", "ma_period": 10, "factor": 0.8, "offset": 1},
                    {"indicator": "close", "operator": ">", "indicator_prev": "close", "factor": 1.02}
                ],
                "weight": 2.0
            },
            {
                "name": "Support Bounce",
                "conditions": [
                    {"indicator": "low", "operator": ">", "indicator_min": "low", "period": 10, "offset": 1, "factor": 0.99},
                    {"indicator": "close", "operator": ">", "indicator_prev": "close"}
                ],
                "weight": 2.5
            }
        ],
        "sell_conditions": [
            {
                "name": "Price Breakout Down",
                "conditions": [
                    {"indicator": "close", "operator": "<", "indicator_min": "low", "period": 20, "offset": 1},
                    {"indicator": "volume", "operator": ">", "indicator_ma": "volume", "ma_period": 10, "factor": 1.5}
                ],
                "weight": 3.5
            },
            {
                "name": "Failed Breakout",
                "conditions": [
                    {"indicator": "high", "operator": ">", "indicator_max": "high", "period": 20, "offset": 1},
                    {"indicator": "close", "operator": "<", "indicator_prev": "close"}
                ],
                "weight": 3.0
            },
            {
                "name": "Resistance Rejection",
                "conditions": [
                    {"indicator": "high", "operator": "<", "indicator_max": "high", "period": 10, "offset": 1, "factor": 1.01},
                    {"indicator": "close", "operator": "<", "indicator_prev": "close"}
                ],
                "weight": 2.5
            }
        ],
        "signal_threshold": 3.5,
        
        # Zusätzliche spezifische Parameter für dieses Pattern
        "lookback_period": 20,
        "consolidation_threshold": 0.03,  # 3% Schwankungsbreite für Konsolidierung
        "breakout_volume_factor": 1.5     # Mindestvolumen für gültige Ausbrüche
    }

# Diese Funktion könnte genutzt werden, um erweiterte Berechnung zu implementieren,
# die über die Standard-Implementierung hinausgehen
def calculate_custom_indicators(df):
    """
    Calculate custom indicators specific to this pattern.
    This function is optional and can be used for complex calculations.
    
    Args:
        df: DataFrame with OHLCV and standard indicators
        
    Returns:
        DataFrame with added custom indicators
    """
    # Beispiel: Berechne die Konsolidierungsphase
    pattern = get_pattern()
    lookback = pattern["lookback_period"]
    
    # Berechne Konsolidierungswert (niedriger = stärkere Konsolidierung)
    df['consolidation'] = (
        df['high'].rolling(lookback).max() - 
        df['low'].rolling(lookback).min()
    ) / df['close']
    
    # Berechne relative Volumen-Stärke
    df['volume_strength'] = df['volume'] / df['volume'].rolling(10).mean()
    
    # Konsolidierungsbreakout-Indikator
    df['is_consolidating'] = df['consolidation'] < pattern["consolidation_threshold"]
    
    return df 