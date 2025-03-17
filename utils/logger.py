"""
Logger-Modul für den Trading Bot.
Stellt Logging-Funktionalität für alle Komponenten bereit.
"""
import logging
import os
from datetime import datetime
from typing import Optional, Union

def setup_logger(log_file: Optional[str] = None, log_level: Union[str, int] = "INFO") -> logging.Logger:
    """
    Konfiguriert und gibt einen Logger zurück.
    
    Args:
        log_file: Pfad zur Log-Datei. Wenn None, wird nur auf die Konsole geloggt.
        log_level: Log-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Konfigurierter Logger
    """
    # Erstelle Logs-Verzeichnis, falls es nicht existiert
    if log_file and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Konvertiere Log-Level in numerischen Wert
    if isinstance(log_level, str):
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Ungültiges Log-Level: {log_level}")
    elif isinstance(log_level, int):
        numeric_level = log_level
    else:
        raise TypeError(f"Log-Level muss str oder int sein, nicht {type(log_level)}")
    
    # Konfiguriere Logger
    logger = logging.getLogger("trading_bot")
    logger.setLevel(numeric_level)
    
    # Entferne bestehende Handler, um doppelte Logs zu vermeiden
    if logger.handlers:
        logger.handlers.clear()
    
    # Erstelle Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Erstelle und konfiguriere Console-Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Erstelle und konfiguriere File-Handler, falls log_file angegeben
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
