"""
Performance Tracker-Modul für den Trading Bot.
Verfolgt Handelsleistung und berechnet Kennzahlen für risikoadaptive Positionsgrößenberechnung.
"""
import os
import time
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from config.config import (
    DATA_DIRECTORY, LOG_LEVEL, LOG_FILE, RISK_PERCENTAGE
)

# Logger einrichten
logger = setup_logger(os.path.join(DATA_DIRECTORY, LOG_FILE), LOG_LEVEL)

class PerformanceTracker:
    """
    Klasse zum Verfolgen und Analysieren der Trading-Performance.
    Ermöglicht adaptive Positionsgrößenberechnung basierend auf historischer Performance.
    """
    
    def __init__(self, initial_capital: float = 1000.0, history_file: Optional[str] = None):
        """
        Initialisiert den PerformanceTracker.
        
        Args:
            initial_capital: Anfangskapital für Performance-Tracking
            history_file: Optional - Pfad zur Datei mit historischen Trades
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.history_file = history_file or os.path.join(DATA_DIRECTORY, "trade_history.json")
        
        # Handelshistorie und Performance-Metriken
        self.trades = []
        self.symbols_performance = {}  # Performance-Metriken pro Symbol
        self.global_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'expected_value': 0.0,
            'kelly_percentage': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_streak': 0,
            'loss_streak': 0,
            'current_streak': 0
        }
        
        # Equity-Kurve für Drawdown-Berechnung
        self.equity_curve = [(datetime.now(), initial_capital)]
        self.peak_capital = initial_capital
        
        # Lade historische Daten, falls vorhanden
        self._load_history()
        
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """
        Fügt einen abgeschlossenen Trade zur Handelshistorie hinzu und aktualisiert Metriken.
        
        Args:
            trade: Dictionary mit Trade-Informationen (symbol, entry_price, exit_price, etc.)
        """
        if not isinstance(trade, dict):
            logger.error("Trade muss ein Dictionary sein")
            return
            
        # Stelle sicher, dass alle erforderlichen Felder vorhanden sind
        required_fields = ['symbol', 'entry_time', 'exit_time', 'entry_price', 
                          'exit_price', 'quantity', 'pnl', 'pnl_pct']
        
        for field in required_fields:
            if field not in trade:
                logger.error(f"Fehlender Pflichtparameter im Trade: {field}")
                return
                
        # Füge zusätzliche Metadaten hinzu
        trade['id'] = f"{trade['symbol']}_{int(datetime.timestamp(datetime.now()))}"
        trade['timestamp'] = datetime.now()
        
        # Füge Trade zur Handelshistorie hinzu
        self.trades.append(trade)
        
        # Aktualisiere Kapital
        self.current_capital += trade['pnl']
        
        # Aktualisiere Equity-Kurve
        self.equity_curve.append((datetime.now(), self.current_capital))
        
        # Aktualisiere Peak-Kapital und Drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        else:
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
            self.global_metrics['current_drawdown'] = current_drawdown
            self.global_metrics['max_drawdown'] = max(self.global_metrics['max_drawdown'], current_drawdown)
        
        # Aktualisiere symbolspezifische Metriken
        symbol = trade['symbol']
        if symbol not in self.symbols_performance:
            self.symbols_performance[symbol] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'profit': 0.0,
                'loss': 0.0,
                'win_streak': 0,
                'loss_streak': 0,
                'current_streak': 0
            }
        
        perf = self.symbols_performance[symbol]
        perf['trades'] += 1
        
        # Aktualisiere Win/Loss-Zähler
        if trade['pnl'] > 0:
            perf['wins'] += 1
            perf['profit'] += trade['pnl']
            self.global_metrics['winning_trades'] += 1
            self.global_metrics['largest_win'] = max(self.global_metrics['largest_win'], trade['pnl'])
            
            # Aktualisiere Streaks
            if perf['current_streak'] >= 0:
                perf['current_streak'] += 1
            else:
                perf['current_streak'] = 1
                
            if self.global_metrics['current_streak'] >= 0:
                self.global_metrics['current_streak'] += 1
            else:
                self.global_metrics['current_streak'] = 1
        else:
            perf['losses'] += 1
            perf['loss'] += abs(trade['pnl'])
            self.global_metrics['losing_trades'] += 1
            self.global_metrics['largest_loss'] = min(self.global_metrics['largest_loss'], trade['pnl'])
            
            # Aktualisiere Streaks
            if perf['current_streak'] <= 0:
                perf['current_streak'] -= 1
            else:
                perf['current_streak'] = -1
                
            if self.global_metrics['current_streak'] <= 0:
                self.global_metrics['current_streak'] -= 1
            else:
                self.global_metrics['current_streak'] = -1
        
        # Aktualisiere Max-Streaks
        perf['win_streak'] = max(perf['win_streak'], perf['current_streak'] if perf['current_streak'] > 0 else 0)
        perf['loss_streak'] = min(perf['loss_streak'], perf['current_streak'] if perf['current_streak'] < 0 else 0)
        
        self.global_metrics['win_streak'] = max(self.global_metrics['win_streak'], 
                                              self.global_metrics['current_streak'] if self.global_metrics['current_streak'] > 0 else 0)
        self.global_metrics['loss_streak'] = min(self.global_metrics['loss_streak'], 
                                               self.global_metrics['current_streak'] if self.global_metrics['current_streak'] < 0 else 0)
        
        # Aktualisiere globale Metriken
        self.global_metrics['total_trades'] += 1
        
        # Berechne alle Metriken neu
        self._calculate_metrics()
        
        # Speichere aktualisierte Handelshistorie
        self._save_history()
        
        logger.info(f"Trade hinzugefügt: {symbol} - PnL: {trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")
        
    def _calculate_metrics(self) -> None:
        """Berechnet alle Performance-Metriken basierend auf der aktuellen Handelshistorie."""
        if not self.trades:
            return
            
        # Berechne globale Metriken
        total_profit = sum(max(0, trade['pnl']) for trade in self.trades)
        total_loss = sum(abs(min(0, trade['pnl'])) for trade in self.trades)
        
        # Win-Rate
        self.global_metrics['win_rate'] = self.global_metrics['winning_trades'] / self.global_metrics['total_trades'] if self.global_metrics['total_trades'] > 0 else 0
        
        # Durchschnittlicher Gewinn/Verlust
        self.global_metrics['average_win'] = total_profit / self.global_metrics['winning_trades'] if self.global_metrics['winning_trades'] > 0 else 0
        self.global_metrics['average_loss'] = total_loss / self.global_metrics['losing_trades'] if self.global_metrics['losing_trades'] > 0 else 0
        
        # Profit-Faktor und Erwartungswert
        self.global_metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
        self.global_metrics['expected_value'] = (self.global_metrics['win_rate'] * self.global_metrics['average_win'] - 
                                              (1 - self.global_metrics['win_rate']) * self.global_metrics['average_loss'])
        
        # Berechne Kelly-Kriterium
        win_rate = self.global_metrics['win_rate']
        win_loss_ratio = self.global_metrics['average_win'] / self.global_metrics['average_loss'] if self.global_metrics['average_loss'] > 0 else 1
        
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio) if win_loss_ratio > 0 else 0
        self.global_metrics['kelly_percentage'] = max(0, kelly)  # Kelly kann nicht negativ sein
        
        # Berechne Risiko-adjustierte Metriken (Sharpe, Sortino)
        if len(self.trades) > 1:
            # Daily returns für Sharpe/Sortino
            returns = [trade['pnl_pct'] / 100 for trade in self.trades]  # Konvertiere zu Dezimal
            avg_return = np.mean(returns)
            std_dev = np.std(returns)
            
            # Nur negative Returns für Sortino
            neg_returns = [r for r in returns if r < 0]
            downside_dev = np.std(neg_returns) if neg_returns else 0
            
            # Annualisierte Sharpe/Sortino (angenommen durchschnittlich 252 Handelstage)
            self.global_metrics['sharpe_ratio'] = (avg_return / std_dev) * np.sqrt(252) if std_dev > 0 else 0
            self.global_metrics['sortino_ratio'] = (avg_return / downside_dev) * np.sqrt(252) if downside_dev > 0 else 0
    
    def _load_history(self) -> None:
        """Lädt Handelshistorie aus Datei, falls vorhanden."""
        if not self.history_file or not os.path.exists(self.history_file):
            return
            
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, dict):
                if 'trades' in data and isinstance(data['trades'], list):
                    self.trades = data['trades']
                    
                    # Konvertiere String-Datumsangaben zurück in datetime
                    for trade in self.trades:
                        if 'entry_time' in trade and isinstance(trade['entry_time'], str):
                            trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])
                        if 'exit_time' in trade and isinstance(trade['exit_time'], str):
                            trade['exit_time'] = datetime.fromisoformat(trade['exit_time'])
                        if 'timestamp' in trade and isinstance(trade['timestamp'], str):
                            trade['timestamp'] = datetime.fromisoformat(trade['timestamp'])
                            
                if 'initial_capital' in data:
                    self.initial_capital = data['initial_capital']
                    
                if 'current_capital' in data:
                    self.current_capital = data['current_capital']
                    
                if 'equity_curve' in data and isinstance(data['equity_curve'], list):
                    self.equity_curve = [(datetime.fromisoformat(date), value) for date, value in data['equity_curve']]
                    
                if 'symbols_performance' in data:
                    self.symbols_performance = data['symbols_performance']
                    
                if 'global_metrics' in data:
                    self.global_metrics = data['global_metrics']
                    
                # Berechne Metriken neu, um sicherzustellen, dass alles aktuell ist
                self._calculate_metrics()
                
                logger.info(f"Handelshistorie geladen: {len(self.trades)} Trades")
                
        except Exception as e:
            logger.error(f"Fehler beim Laden der Handelshistorie: {e}")
    
    def _save_history(self) -> None:
        """Speichert aktuelle Handelshistorie in Datei."""
        if not self.history_file:
            return
            
        try:
            # Erstelle Verzeichnis, falls es nicht existiert
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            # Konvertiere datetime-Objekte zu Strings für JSON-Serialisierung
            serializable_trades = []
            for trade in self.trades:
                trade_copy = trade.copy()
                
                if 'entry_time' in trade_copy and isinstance(trade_copy['entry_time'], datetime):
                    trade_copy['entry_time'] = trade_copy['entry_time'].isoformat()
                    
                if 'exit_time' in trade_copy and isinstance(trade_copy['exit_time'], datetime):
                    trade_copy['exit_time'] = trade_copy['exit_time'].isoformat()
                    
                if 'timestamp' in trade_copy and isinstance(trade_copy['timestamp'], datetime):
                    trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
                    
                serializable_trades.append(trade_copy)
                
            # Serialisiere Equity-Kurve
            serializable_equity = [(date.isoformat(), value) for date, value in self.equity_curve]
            
            data = {
                'trades': serializable_trades,
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'equity_curve': serializable_equity,
                'symbols_performance': self.symbols_performance,
                'global_metrics': self.global_metrics
            }
            
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Handelshistorie gespeichert in {self.history_file}")
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Handelshistorie: {e}")
    
    def get_metrics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Gibt Performance-Metriken für ein Symbol oder global zurück.
        
        Args:
            symbol: Optional - Trading-Paar für symbolspezifische Metriken
            
        Returns:
            Dictionary mit Performance-Metriken
        """
        if symbol:
            if symbol not in self.symbols_performance:
                logger.warning(f"Keine Performance-Daten für Symbol {symbol}")
                return {}
                
            perf = self.symbols_performance[symbol]
            
            # Berechne symbolspezifische Metriken
            win_rate = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
            avg_win = perf['profit'] / perf['wins'] if perf['wins'] > 0 else 0
            avg_loss = perf['loss'] / perf['losses'] if perf['losses'] > 0 else 0
            profit_factor = perf['profit'] / perf['loss'] if perf['loss'] > 0 else float('inf')
            
            # Berechne Kelly nur für dieses Symbol
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
            kelly = win_rate - ((1 - win_rate) / win_loss_ratio) if win_loss_ratio > 0 else 0
            kelly_pct = max(0, kelly)
            
            return {
                'symbol': symbol,
                'trades': perf['trades'],
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'win_streak': perf['win_streak'],
                'loss_streak': perf['loss_streak'],
                'current_streak': perf['current_streak'],
                'kelly_percentage': kelly_pct,
                'expectancy': (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            }
        else:
            # Gib globale Metriken zurück
            return self.global_metrics
    
    def calculate_optimal_position_size(self, 
                                      symbol: str, 
                                      account_balance: float, 
                                      risk_factor: float = 0.5,
                                      min_trades: int = 10) -> float:
        """
        Berechnet die optimale Positionsgröße basierend auf Kelly-Kriterium und historischer Performance.
        
        Args:
            symbol: Trading-Paar
            account_balance: Aktuelles Kontoguthaben
            risk_factor: Risikofaktor (typischerweise 0.5 für Half-Kelly oder 0.25 für Quarter-Kelly)
            min_trades: Mindestanzahl von Trades für verlässliche Kelly-Berechnung
            
        Returns:
            Optimale Positionsgröße als Prozentsatz des Kontoguthabens
        """
        # Prüfe, ob wir genügend Daten für dieses Symbol haben
        if symbol in self.symbols_performance:
            perf = self.symbols_performance[symbol]
            
            # Wenn wir genügend Trades für verlässliche Statistiken haben
            if perf['trades'] >= min_trades:
                # Berechne Win-Rate und Win/Loss-Verhältnis
                win_rate = perf['wins'] / perf['trades']
                avg_win = perf['profit'] / perf['wins'] if perf['wins'] > 0 else 0
                avg_loss = perf['loss'] / perf['losses'] if perf['losses'] > 0 else 0
                win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
                
                # Berechne Kelly-Prozentsatz
                kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
                kelly_pct = max(0, kelly)  # Kelly kann nicht negativ sein
                
                # Anpassung für aktuelle Streak (reduziere Größe bei Verlustserie)
                streak_factor = 1.0
                if perf['current_streak'] < 0:
                    # Reduziere Positionsgröße nach Verlusten
                    streak_factor = max(0.5, 1.0 - (abs(perf['current_streak']) / 10))
                
                # Anwendung des Risikofaktors und der Streak-Anpassung
                adjusted_kelly = kelly_pct * risk_factor * streak_factor
                
                # Begrenze auf sinnvollen Bereich (max 20% des Kontoguthabens)
                return min(adjusted_kelly, 0.2)
            
        # Fallback auf Standard-Risikoprozentsatz, wenn nicht genügend Daten vorliegen
        return RISK_PERCENTAGE / 100  # Konvertiere von Prozent zu Dezimal
        
    def get_equity_curve(self) -> List[Tuple[datetime, float]]:
        """
        Gibt die Equity-Kurve für Visualisierung und Analyse zurück.
        
        Returns:
            Liste von (Zeitstempel, Kapital)-Tupeln
        """
        return self.equity_curve
        
    def get_symbol_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Gibt alle Trades für ein bestimmtes Symbol zurück.
        
        Args:
            symbol: Trading-Paar
            
        Returns:
            Liste von Trade-Dictionaries
        """
        return [trade for trade in self.trades if trade['symbol'] == symbol]
        
    def get_recent_trades(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Gibt die letzten N Trades zurück.
        
        Args:
            count: Anzahl der zurückzugebenden Trades
            
        Returns:
            Liste der letzten N Trades
        """
        sorted_trades = sorted(self.trades, key=lambda t: t.get('exit_time', datetime.now()), reverse=True)
        return sorted_trades[:count]
        
    def calculate_drawdown(self) -> Tuple[float, float]:
        """
        Berechnet den aktuellen und maximalen Drawdown.
        
        Returns:
            Tuple mit (aktueller Drawdown in %, maximaler Drawdown in %)
        """
        if not self.equity_curve:
            return 0.0, 0.0
            
        # Extrahiere nur die Kapitalwerte
        equity_values = [value for _, value in self.equity_curve]
        
        # Berechne Running Maximum
        running_max = np.maximum.accumulate(equity_values)
        
        # Berechne prozentuale Drawdowns
        drawdowns = (running_max - equity_values) / running_max * 100
        
        # Aktueller und maximaler Drawdown
        current_drawdown = drawdowns[-1] if drawdowns.size > 0 else 0.0
        max_drawdown = np.max(drawdowns) if drawdowns.size > 0 else 0.0
        
        return current_drawdown, max_drawdown
        
    def reset(self) -> None:
        """Setzt den PerformanceTracker auf Anfangszustand zurück."""
        self.trades = []
        self.symbols_performance = {}
        self.current_capital = self.initial_capital
        self.equity_curve = [(datetime.now(), self.initial_capital)]
        self.peak_capital = self.initial_capital
        
        # Zurücksetzen der globalen Metriken
        for key in self.global_metrics:
            self.global_metrics[key] = 0.0
            
        # Speichere den zurückgesetzten Zustand
        self._save_history()
        
        logger.info("PerformanceTracker zurückgesetzt")
        
    def get_performance_summary(self) -> str:
        """
        Erstellt eine formatierte Zusammenfassung der Performance-Metriken.
        
        Returns:
            Formatierte Zusammenfassung als String
        """
        if not self.trades:
            return "Keine Handelshistorie verfügbar."
            
        current_drawdown, max_drawdown = self.calculate_drawdown()
        
        summary = [
            "=== Performance-Zusammenfassung ===",
            f"Anfangskapital: ${self.initial_capital:.2f}",
            f"Aktuelles Kapital: ${self.current_capital:.2f}",
            f"Gesamtrendite: {((self.current_capital - self.initial_capital) / self.initial_capital * 100):.2f}%",
            f"Anzahl Trades: {self.global_metrics['total_trades']}",
            f"Gewinnrate: {self.global_metrics['win_rate'] * 100:.2f}%",
            f"Profit-Faktor: {self.global_metrics['profit_factor']:.2f}",
            f"Durchschn. Gewinn: ${self.global_metrics['average_win']:.2f}",
            f"Durchschn. Verlust: ${self.global_metrics['average_loss']:.2f}",
            f"Kelly-Prozentsatz: {self.global_metrics['kelly_percentage'] * 100:.2f}%",
            f"Aktueller Drawdown: {current_drawdown:.2f}%",
            f"Maximaler Drawdown: {max_drawdown:.2f}%",
            f"Sharpe-Ratio: {self.global_metrics['sharpe_ratio']:.2f}",
            f"Sortino-Ratio: {self.global_metrics['sortino_ratio']:.2f}"
        ]
        
        return "\n".join(summary) 