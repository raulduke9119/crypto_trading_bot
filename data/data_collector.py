"""
Datensammler-Modul für den Trading Bot.
Verantwortlich für das Abrufen und Speichern von Marktdaten von Binance.
"""
import os
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from config.config import DATA_DIRECTORY, LOG_LEVEL, LOG_FILE

# Logger einrichten
logger = setup_logger(os.path.join(DATA_DIRECTORY, LOG_FILE), LOG_LEVEL)

class DataCollector:
    """
    Klasse zum Sammeln und Verarbeiten von Marktdaten von Binance.
    """
    
    def __init__(self, client: Optional[Client] = None, data_dir: str = DATA_DIRECTORY) -> None:
        """
        Initialisiert den DataCollector mit verbesserter Fehlerbehandlung und Rate-Limiting.
        
        Args:
            client: Binance Client-Instanz (für Trading-Operationen)
            data_dir: Verzeichnis zum Speichern der Daten
        """
        # Trading-Client (mit API-Keys)
        self.client = client
        
        # Liste von verfügbaren API-Endpunkten für Fallback
        self.base_urls = [
            'https://api.binance.com',
            'https://api1.binance.com',
            'https://api2.binance.com',
            'https://api3.binance.com',
            'https://api4.binance.com'
        ]
        self.current_url_index = 0
        
        # Rate-Limiting-Parameter
        self.request_weight = 0
        self.last_request_time = time.time()
        self.weight_reset_window = 60  # 1 Minute
        self.max_weight_per_minute = 1200
        
        # Separater Client für historische Daten (ohne API-Keys)
        self.historical_client = Client(None, None, {'verify': True, 'timeout': 20})
        self.historical_client.API_URL = self.base_urls[0] + '/api'
        
        self.data_dir = data_dir
        
        # Erstelle Datenverzeichnis, falls es nicht existiert
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            logger.info(f"Datenverzeichnis erstellt: {self.data_dir}")
        
        # Initialisiere Zeitsynchronisation
        self._sync_time()
    
    def _sync_time(self) -> None:
        """Synchronisiert die lokale Zeit mit dem Binance-Server."""
        try:
            server_time = self.historical_client.get_server_time()
            self.time_offset = server_time['serverTime'] - int(time.time() * 1000)
            logger.debug(f"Zeitsynchronisation durchgeführt, Offset: {self.time_offset}ms")
        except Exception as e:
            logger.error(f"Fehler bei der Zeitsynchronisation: {e}")
            self.time_offset = 0

    def _check_rate_limit(self) -> None:
        """Überprüft und handhabt Rate-Limiting."""
        current_time = time.time()
        time_passed = current_time - self.last_request_time
        
        # Reset Weight nach Ablauf des Zeitfensters
        if time_passed >= self.weight_reset_window:
            self.request_weight = 0
            self.last_request_time = current_time
        
        # Warte wenn nötig
        if self.request_weight >= self.max_weight_per_minute:
            wait_time = self.weight_reset_window - time_passed
            if wait_time > 0:
                logger.warning(f"Rate-Limit erreicht, warte {wait_time:.2f}s")
                time.sleep(wait_time)
                self.request_weight = 0
                self.last_request_time = time.time()

    def _handle_api_error(self, e: Exception, retry_count: int = 0) -> bool:
        """Behandelt API-Fehler und wechselt ggf. die Base-URL.

        Args:
            e: Die aufgetretene Exception
            retry_count: Anzahl der bisherigen Versuche

        Returns:
            bool: True wenn weiterer Versuch möglich, False sonst
        """
        if retry_count >= len(self.base_urls):
            logger.error(f"Alle API-Endpunkte erschöpft nach {retry_count} Versuchen")
            return False

        if isinstance(e, BinanceAPIException):
            if e.code == -1003:  # LIMIT_EXCEEDED
                logger.warning("Rate limit überschritten, warte 30 Sekunden...")
                time.sleep(30)
                return True
            elif e.code == -1021:  # INVALID_TIMESTAMP
                logger.warning("Zeitstempel-Fehler, synchronisiere...")
                self._sync_time()
                return True

        # Wechsle zur nächsten Base-URL
        self.current_url_index = (self.current_url_index + 1) % len(self.base_urls)
        new_url = self.base_urls[self.current_url_index]
        self.historical_client.API_URL = f"{new_url}/api"
        logger.info(f"Wechsel zu API-Endpunkt: {new_url}")
        return True

    def get_historical_klines(self, 
                             symbol: str, 
                             interval: str, 
                             start_str: str, 
                             end_str: Optional[str] = None,
                             max_retries: int = 3) -> pd.DataFrame:
        """
        Lädt historische Kerzendaten (OHLCV) von Binance mit Retry-Mechanismus und Chunking.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            interval: Zeitintervall (z.B. '1h', '1d')
            start_str: Startdatum als String (z.B. '1 Jan 2023' oder '5 days ago')
            end_str: Enddatum als String (optional)
            max_retries: Maximale Anzahl von Wiederholungsversuchen
            
        Returns:
            DataFrame mit OHLCV-Daten
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Lade historische Daten für {symbol} ({interval}) von {start_str} bis {end_str or 'jetzt'} (Versuch {attempt + 1}/{max_retries})")
                
                # Konvertiere Datum zu Timestamp
                try:
                    # Verarbeite relative Zeitangaben
                    if isinstance(start_str, str):
                        if 'ago' in start_str:
                            num = int(start_str.split()[0])
                            unit = start_str.split()[1]
                            if unit in ['day', 'days']:
                                start_time = pd.Timestamp.now() - pd.Timedelta(days=num)
                            elif unit in ['month', 'months']:
                                start_time = pd.Timestamp.now() - pd.DateOffset(months=num)
                            elif unit in ['week', 'weeks']:
                                start_time = pd.Timestamp.now() - pd.Timedelta(weeks=num)
                            else:
                                start_time = pd.Timestamp.now() - pd.Timedelta(days=30)  # Standard: 30 Tage
                        else:
                            start_time = pd.Timestamp(start_str)
                    
                    if end_str:
                        if 'now' in end_str.lower():
                            end_time = pd.Timestamp.now()
                        elif 'ago' in end_str:
                            num = int(end_str.split()[0])
                            unit = end_str.split()[1]
                            if unit in ['day', 'days']:
                                end_time = pd.Timestamp.now() - pd.Timedelta(days=num)
                            elif unit in ['month', 'months']:
                                end_time = pd.Timestamp.now() - pd.DateOffset(months=num)
                            elif unit in ['week', 'weeks']:
                                end_time = pd.Timestamp.now() - pd.Timedelta(weeks=num)
                            else:
                                end_time = pd.Timestamp.now()
                        else:
                            end_time = pd.Timestamp(end_str)
                    else:
                        end_time = pd.Timestamp.now()
                    
                    # Konvertiere zu Millisekunden-Timestamp
                    start_ts = int(start_time.timestamp() * 1000)
                    end_ts = int(end_time.timestamp() * 1000)
                    
                    logger.debug(f"Konvertierte Zeitstempel - Start: {start_time}, Ende: {end_time}")
                    
                except ValueError as e:
                    logger.error(f"Fehler bei der Datums-Konvertierung: {e}")
                    continue
                
                # Lade Daten von Binance mit Chunking für große Zeiträume
                klines = []
                chunk_size = timedelta(days=30)  # Binance-Limit pro Request
                
                current_start = start_time
                
                while current_start < end_time:
                    chunk_end = min(current_start + chunk_size, end_time)
                    
                    # Verwende den historischen Client ohne API-Keys
                    chunk_klines = self.historical_client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=str(current_start),
                        end_str=str(chunk_end)
                    )
                    
                    if chunk_klines:
                        klines.extend(chunk_klines)
                        logger.debug(f"Geladen: {len(chunk_klines)} Klines von {current_start} bis {chunk_end}")
                    
                    current_start = chunk_end
                    time.sleep(0.5)  # Rate-Limiting vermeiden
                
                if not klines:
                    logger.warning(f"Keine Daten gefunden für {symbol} im angegebenen Zeitraum")
                    return pd.DataFrame()
                
                # Erstelle DataFrame mit korrekten Spaltennamen
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Konvertiere Timestamp zu Datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Konvertiere String-Werte zu Floats und handle NaN-Werte
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Entferne Duplikate und sortiere nach Zeitstempel
                df = df[~df.index.duplicated(keep='first')]
                df.sort_index(inplace=True)
                
                # Validiere Datenqualität
                if df[numeric_columns].isnull().any().any():
                    logger.warning(f"Null-Werte in {symbol} Daten gefunden, führe Bereinigung durch")
                    
                    # Fülle fehlende Werte mit vorwärts/rückwärts Interpolation
                    df[numeric_columns] = df[numeric_columns].ffill().bfill()
                    
                    # Prüfe ob noch Null-Werte existieren
                    remaining_nulls = df[numeric_columns].isnull().sum().sum()
                    if remaining_nulls > 0:
                        logger.error(f"Nach Bereinigung verbleiben {remaining_nulls} Null-Werte in {symbol} Daten")
                        return pd.DataFrame()
                
                logger.info(f"Erfolgreich {len(df)} Datenpunkte für {symbol} geladen")
                return df
                
            except BinanceAPIException as e:
                if 'Invalid symbol' in str(e):
                    logger.error(f"Ungültiges Symbol {symbol}: {e}")
                    return pd.DataFrame()
                elif attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Binance API Fehler für {symbol}, warte {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Maximale Anzahl von Versuchen erreicht für {symbol}: {e}")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Unerwarteter Fehler beim Laden historischer Daten für {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return pd.DataFrame()
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Gibt den aktuellen Preis eines Symbols zurück.
        
        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            
        Returns:
            Aktueller Preis oder None bei Fehler
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            logger.debug(f"Aktueller Preis für {symbol}: {price}")
            return price
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des aktuellen Preises für {symbol}: {e}")
            return None
    
    def get_multiple_symbols_data(self, 
                                 symbols: List[str], 
                                 interval: str, 
                                 days_back: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Lädt historische Daten für mehrere Symbole.
        
        Args:
            symbols: Liste von Trading-Paaren
            interval: Zeitintervall
            days_back: Anzahl der Tage zurück
            
        Returns:
            Dictionary mit Symbol als Schlüssel und DataFrame als Wert
        """
        start_str = f"{days_back} days ago"
        all_data = {}
        
        for symbol in symbols:
            try:
                df = self.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=start_str
                )
                
                if not df.empty:
                    all_data[symbol] = df
                
                # Warte kurz, um Rate-Limits zu vermeiden
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Fehler beim Laden von Daten für {symbol}: {e}")
        
        return all_data
    
    def save_data(self, symbol: str, interval: str, df: pd.DataFrame) -> bool:
        """
        Speichert DataFrame als CSV-Datei.
        
        Args:
            symbol: Trading-Paar
            interval: Zeitintervall
            df: DataFrame mit Daten
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Erstelle Dateinamen
            filename = f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            # Speichere DataFrame
            df.to_csv(filepath)
            logger.info(f"Daten gespeichert: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Daten für {symbol}: {e}")
            return False
    
    def load_data(self, symbol: str, interval: str, date_str: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Lädt gespeicherte Daten aus CSV-Datei.
        
        Args:
            symbol: Trading-Paar
            interval: Zeitintervall
            date_str: Datumsstring im Format 'YYYYMMDD' (optional, sonst neueste Datei)
            
        Returns:
            DataFrame mit Daten oder None bei Fehler
        """
        try:
            # Finde passende Dateien
            pattern = f"{symbol}_{interval}"
            if date_str:
                pattern += f"_{date_str}"
            
            matching_files = [f for f in os.listdir(self.data_dir) if f.startswith(pattern) and f.endswith('.csv')]
            
            if not matching_files:
                logger.warning(f"Keine gespeicherten Daten gefunden für {pattern}")
                return None
            
            # Wähle neueste Datei, wenn kein spezifisches Datum angegeben wurde
            if not date_str:
                matching_files.sort(reverse=True)
            
            # Lade Daten
            filepath = os.path.join(self.data_dir, matching_files[0])
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Daten geladen: {filepath}")
            
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Daten für {symbol}: {e}")
            return None
