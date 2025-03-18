"""
Pattern Loader Utility für den Trading Bot.
Ermöglicht das dynamische Laden und Verwenden verschiedener Trading-Pattern.
"""
import os
import sys
import json
import importlib.util
from typing import Dict, List, Any, Optional, Callable
import logging

# Configure the path to include parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from config.config import LOG_LEVEL, LOG_FILE, DATA_DIRECTORY

# Setup logger
logger = setup_logger(os.path.join(DATA_DIRECTORY, LOG_FILE), LOG_LEVEL)

class PatternLoader:
    """
    Utility-Klasse zum Laden und Verwalten von Trading-Pattern.
    """
    
    def __init__(self, patterns_dir: str = "patterns"):
        """
        Initialisiert den Pattern Loader.
        
        Args:
            patterns_dir: Verzeichnis, in dem die Muster gespeichert sind
        """
        self.patterns_dir = patterns_dir
        self.loaded_patterns = {}
        self.default_pattern = None
        
        # Stelle sicher, dass das Patterns-Verzeichnis existiert
        if not os.path.exists(patterns_dir):
            os.makedirs(patterns_dir)
            logger.info(f"Patterns-Verzeichnis erstellt: {patterns_dir}")
        
        logger.info(f"Pattern Loader initialisiert. Verzeichnis: {patterns_dir}")
    
    def get_available_patterns(self) -> List[str]:
        """
        Gibt eine Liste aller verfügbaren Pattern-Dateien zurück.
        
        Returns:
            Liste der Namen verfügbarer Pattern-Dateien
        """
        try:
            pattern_files = []
            
            # Durchsuche Verzeichnis nach Python- und JSON-Dateien
            for file in os.listdir(self.patterns_dir):
                if file.endswith('.py') or file.endswith('.json'):
                    pattern_files.append(file)
            
            return pattern_files
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der verfügbaren Patterns: {e}")
            return []
    
    def load_pattern(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        """
        Lädt ein Pattern aus einer Datei.
        
        Args:
            pattern_name: Name der Pattern-Datei (mit oder ohne Erweiterung)
            
        Returns:
            Das geladene Pattern als Dictionary oder None bei Fehler
        """
        try:
            # Wenn das Pattern bereits geladen wurde, gib es zurück
            if pattern_name in self.loaded_patterns:
                return self.loaded_patterns[pattern_name]
            
            # Stelle sicher, dass wir eine korrekte Dateiendung haben
            if not (pattern_name.endswith('.py') or pattern_name.endswith('.json')):
                if os.path.exists(os.path.join(self.patterns_dir, f"{pattern_name}.py")):
                    pattern_name = f"{pattern_name}.py"
                elif os.path.exists(os.path.join(self.patterns_dir, f"{pattern_name}.json")):
                    pattern_name = f"{pattern_name}.json"
                else:
                    logger.error(f"Pattern-Datei nicht gefunden: {pattern_name}")
                    return None
            
            pattern_path = os.path.join(self.patterns_dir, pattern_name)
            
            # Prüfe, ob die Datei existiert
            if not os.path.exists(pattern_path):
                logger.error(f"Pattern-Datei nicht gefunden: {pattern_path}")
                return None
            
            # Lade je nach Dateiformat
            if pattern_name.endswith('.json'):
                # JSON-Format
                with open(pattern_path, 'r') as file:
                    pattern = json.load(file)
                    
                    # Validiere das Pattern
                    if not self._validate_json_pattern(pattern):
                        logger.error(f"Ungültiges Pattern-Format: {pattern_name}")
                        return None
                    
                    self.loaded_patterns[pattern_name] = pattern
                    logger.info(f"Pattern geladen (JSON): {pattern_name}")
                    return pattern
            
            elif pattern_name.endswith('.py'):
                # Python-Modul
                module_name = os.path.splitext(pattern_name)[0]
                
                try:
                    # Lade das Modul dynamisch
                    spec = importlib.util.spec_from_file_location(module_name, pattern_path)
                    if spec is None or spec.loader is None:
                        logger.error(f"Konnte Modul-Spezifikation nicht laden: {pattern_name}")
                        return None
                    
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Prüfe, ob das Modul die erforderliche Funktion hat
                    if not hasattr(module, 'get_pattern'):
                        logger.error(f"Modul hat keine get_pattern-Funktion: {pattern_name}")
                        return None
                    
                    pattern = module.get_pattern()
                    
                    # Validiere das Pattern
                    if not self._validate_python_pattern(pattern):
                        logger.error(f"Ungültiges Pattern-Format aus Python-Modul: {pattern_name}")
                        return None
                    
                    self.loaded_patterns[pattern_name] = pattern
                    logger.info(f"Pattern geladen (Python): {pattern_name}")
                    return pattern
                    
                except Exception as e:
                    logger.error(f"Fehler beim Laden des Python-Moduls {pattern_name}: {e}")
                    return None
            
            else:
                logger.error(f"Nicht unterstütztes Dateiformat: {pattern_name}")
                return None
                
        except Exception as e:
            logger.error(f"Fehler beim Laden des Patterns {pattern_name}: {e}")
            return None
    
    def load_default_pattern(self) -> Dict[str, Any]:
        """
        Lädt das Standard-Pattern oder erstellt eines, wenn keines existiert.
        
        Returns:
            Das Standard-Pattern als Dictionary
        """
        if self.default_pattern:
            return self.default_pattern
        
        # Prüfe, ob ein Standard-Pattern existiert
        default_pattern_path = os.path.join(self.patterns_dir, "default_pattern.json")
        
        if os.path.exists(default_pattern_path):
            try:
                with open(default_pattern_path, 'r') as file:
                    default_pattern = json.load(file)
                    self.default_pattern = default_pattern
                    logger.info("Standard-Pattern geladen")
                    return default_pattern
            except Exception as e:
                logger.error(f"Fehler beim Laden des Standard-Patterns: {e}")
        
        # Erstelle ein Standard-Pattern
        default_pattern = self._create_default_pattern()
        self.default_pattern = default_pattern
        
        # Speichere es für zukünftige Verwendung
        try:
            with open(default_pattern_path, 'w') as file:
                json.dump(default_pattern, file, indent=4)
                logger.info("Standard-Pattern erstellt und gespeichert")
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Standard-Patterns: {e}")
        
        return default_pattern
    
    def _create_default_pattern(self) -> Dict[str, Any]:
        """
        Erstellt ein Standard-Pattern mit grundlegenden Trading-Regeln.
        
        Returns:
            Ein Standard-Pattern als Dictionary
        """
        default_pattern = {
            "name": "Default Pattern",
            "description": "Standard-Pattern mit grundlegenden Trading-Regeln",
            "buy_conditions": [
                {
                    "name": "RSI Oversold",
                    "conditions": [
                        {"indicator": "rsi", "operator": "<", "value": 30}
                    ],
                    "weight": 3.0
                },
                {
                    "name": "MACD Crossover",
                    "conditions": [
                        {"indicator": "macd_hist", "operator": ">", "value": 0},
                        {"indicator": "macd_hist_prev", "operator": "<=", "value": 0}
                    ],
                    "weight": 2.0
                },
                {
                    "name": "Bollinger Bottom",
                    "conditions": [
                        {"indicator": "close", "operator": "<", "value_indicator": "bb_lower"}
                    ],
                    "weight": 2.5
                }
            ],
            "sell_conditions": [
                {
                    "name": "RSI Overbought",
                    "conditions": [
                        {"indicator": "rsi", "operator": ">", "value": 70}
                    ],
                    "weight": 3.0
                },
                {
                    "name": "MACD Crossover Down",
                    "conditions": [
                        {"indicator": "macd_hist", "operator": "<", "value": 0},
                        {"indicator": "macd_hist_prev", "operator": ">=", "value": 0}
                    ],
                    "weight": 2.0
                },
                {
                    "name": "Bollinger Top",
                    "conditions": [
                        {"indicator": "close", "operator": ">", "value_indicator": "bb_upper"}
                    ],
                    "weight": 2.5
                }
            ],
            "signal_threshold": 4.0  # Mindestens 4.0 Punkte für ein Signal
        }
        
        return default_pattern
    
    def _validate_json_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Validiert ein JSON-Pattern.
        
        Args:
            pattern: Das zu validierende Pattern
            
        Returns:
            True wenn valide, False sonst
        """
        try:
            # Überprüfe erforderliche Schlüssel
            required_keys = ["name", "buy_conditions", "sell_conditions", "signal_threshold"]
            if not all(key in pattern for key in required_keys):
                logger.error(f"Pattern fehlen erforderliche Schlüssel: {required_keys}")
                return False
                
            # Validiere Buy-Conditions
            for condition in pattern["buy_conditions"]:
                if not all(key in condition for key in ["name", "conditions", "weight"]):
                    logger.error(f"Buy-Condition fehlen erforderliche Schlüssel: name, conditions, weight")
                    return False
                    
                # Validiere einzelne Bedingungen
                for subcond in condition["conditions"]:
                    if not all(key in subcond for key in ["indicator", "operator"]):
                        logger.error(f"Condition fehlen erforderliche Schlüssel: indicator, operator")
                        return False
                    
                    # Überprüfe, ob entweder "value" oder "value_indicator" vorhanden ist
                    if "value" not in subcond and "value_indicator" not in subcond:
                        logger.error(f"Condition braucht entweder 'value' oder 'value_indicator'")
                        return False
            
            # Validiere Sell-Conditions (ähnlich wie bei Buy)
            for condition in pattern["sell_conditions"]:
                if not all(key in condition for key in ["name", "conditions", "weight"]):
                    logger.error(f"Sell-Condition fehlen erforderliche Schlüssel: name, conditions, weight")
                    return False
                    
                for subcond in condition["conditions"]:
                    if not all(key in subcond for key in ["indicator", "operator"]):
                        logger.error(f"Condition fehlen erforderliche Schlüssel: indicator, operator")
                        return False
                    
                    if "value" not in subcond and "value_indicator" not in subcond:
                        logger.error(f"Condition braucht entweder 'value' oder 'value_indicator'")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der Validierung des Patterns: {e}")
            return False
    
    def _validate_python_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Validiert ein aus Python geladenes Pattern.
        
        Args:
            pattern: Das zu validierende Pattern
            
        Returns:
            True wenn valide, False sonst
        """
        # Die Validierung ist ähnlich wie bei JSON-Patterns
        return self._validate_json_pattern(pattern) 