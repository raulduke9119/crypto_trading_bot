# Binance Trading Bot

Dieser optimierte Trading Bot für Binance ermöglicht automatisiertes Kryptowährungshandel mit einer robusten Multi-Indikator-Strategie und fortschrittlichem Risikomanagement.

![Version](https://img.shields.io/badge/version-1.2.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## Übersicht

Der Bot kombiniert mehrere technische Indikatoren, um profitable Handelssignale zu generieren und nutzt eine modulare Architektur für einfache Erweiterbarkeit:

- **Datensammlung**: Abrufen historischer und Echtzeit-Marktdaten von Binance
- **Technische Indikatoren**: Fortschrittliche Berechnung von SMA, EMA, RSI, MACD und Bollinger Bands
- **Handelsstrategien**: Optimierte Multi-Indikator-Strategie mit adaptiver Signalgewichtung
- **Maschinelles Lernen** (optional): KI-gestützte Vorhersagemodelle für Preisbewegungen
- **Order-Ausführung**: Intelligente Platzierung und Verwaltung von Handelsaufträgen
- **Backtesting**: Umfassende Evaluation von Strategien mit historischen Daten

## Optimierte Handelsstrategie

Die aktuelle Version implementiert eine verbesserte Multi-Indikator-Strategie:

- **Kurzfristiger RSI (2-Perioden)** für präzise Mean-Reversion-Signale
- **MACD-Histogramm-Analyse** zur Trendbestimmung und Früherkennung von Umkehrungen
- **Bollinger-Band-Signale** für Ausbruchs- und Squeeze-Erkennung
- **Volumenvalidierung** mit anomalie-basierter Erkennung wichtiger Marktbewegungen
- **Trailing-Stop-Mechanismus** für optimierte Gewinnmitnahme und Verlustbegrenzung
- **Adaptive Volatiltätsanpassung** der Signalstärke für verschiedene Marktbedingungen

## Hauptfunktionen

- Parallel-Trading für mehrere Kryptowährungspaare mit konfigurierbaren Intervallen
- Umfassendes Risikomanagement mit Positionsgrößenbegrenzung und Drawdown-Schutz
- Hochleistungsfähiges Backtesting-Framework mit detaillierter Performance-Analyse
- Robuste Fehlerbehandlung für zuverlässigen 24/7-Betrieb
- Umfangreiche Logging- und Performance-Überwachungsfunktionen
- Optional: KI-gestützte Signalverstärkung und Marktanalyse

## Getting Started

### Voraussetzungen

- Python 3.8 oder höher
- Pip (Python Package Manager)
- Binance-Konto mit API-Schlüsseln

### Installation

1. Repository klonen:

```bash
git clone https://github.com/yourusername/binance-trading-bot.git
cd binance-trading-bot
```

1. Virtual Environment erstellen und aktivieren:

```bash
python -m venv venv

# Unter Linux/Mac
source venv/bin/activate

# Unter Windows
venv\Scripts\activate
```

1. Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

1. Umgebungsvariablen konfigurieren:
   - Kopiere die `.env.example` Datei zu `.env`
   - Trage deine Binance API-Schlüssel und andere Konfigurationen ein

```bash
cp .env.example .env
# Anschließend editieren mit deinem bevorzugten Texteditor
```

### Konfiguration

Die wichtigsten Einstellungen können in der `config/config.py` angepasst werden:

```python
# Trading-Paare konfigurieren
TRADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

# Risikomanagement-Parameter
MAX_POSITIONS = 2        # Maximale gleichzeitige Positionen
RISK_PERCENTAGE = 1.0    # Risiko pro Trade in Prozent
MAX_DRAWDOWN_PCT = 5.0   # Maximaler Drawdown in Prozent

# Strategie-Parameter
RSI_BUY_THRESHOLD = 15   # RSI-Kaufsignal-Schwelle
RSI_SELL_THRESHOLD = 85  # RSI-Verkaufssignal-Schwelle
```

### Trading-Bot starten

1. **Backtesting** - Strategie mit historischen Daten testen:

```bash
python backtest.py
```

2. **Live-Trading** - Echtzeit-Trading auf Binance:

```bash
python trading_bot.py
```

3. **Paper-Trading** - Simulation ohne echtes Geld:

```bash
python trading_bot.py --paper-trading
```

### Git Setup

Für den ersten Commit und Push:

```bash
# Git-Repository initialisieren (falls nicht geklont)
git init

# Änderungen hinzufügen
git add .

# Commit erstellen
git commit -m "Initial commit: Optimized multi-indicator strategy with improved error handling"

# Remote hinzufügen (falls nicht geklont)
git remote add origin https://github.com/yourusername/binance-trading-bot.git

# Änderungen pushen
git push -u origin main
```

## Erweiterte Funktionen

### Kryptowährungspaare und Parameter

Der Bot kann mit verschiedenen Parametern konfiguriert werden:

```bash
python trading_bot.py --symbols BTCUSDT ETHUSDT SOLUSDT --interval 15 --risk 1.5
```

### Verfügbare Parameter

- `--symbols`: Liste der Trading-Paare (z.B. `BTCUSDT ETHUSDT`)
- `--interval`: Update-Intervall in Minuten (Standard: 5)
- `--risk`: Risikoprozentsatz pro Trade (Standard: 1.0)
- `--max-positions`: Maximale gleichzeitige Positionen (Standard: 2)
- `--mode`: Betriebsmodus `live`, `paper` oder `backtest` (Standard: `paper`)
- `--start-date`: Startdatum für Backtest (Format: YYYY-MM-DD)
- `--end-date`: Enddatum für Backtest (optional)
- `--initial-balance`: Anfangskapital für Backtest (Standard: 1000.0 USDT)
- `--use-ml`: ML-Vorhersagen aktivieren (Flag)

## Strategie-Optimierung

Zur Optimierung der Strategie-Parameter kann das Optimierungsskript verwendet werden:

```bash
python optimize_strategy.py --symbol BTCUSDT --period 90d
```

Dies führt eine Grid-Search für die besten Parameter durch und speichert die Ergebnisse.

## Projektstruktur

```
binance_trading_bot/
├── config/
│   └── config.py         # Konfigurationsparameter
├── data/
│   ├── data_collector.py # Datenabruf von Binance
│   └── indicators.py     # Technische Indikatoren
├── strategies/
│   ├── base_strategy.py    # Basis-Strategieklasse
│   └── multi_indicator_strategy.py # Multi-Indikator-Strategie
├── utils/
│   ├── logger.py         # Logging-Funktionalität
│   └── order_executor.py # Order-Ausführung
├── trading_bot.py        # Hauptanwendung
├── backtest.py           # Backtesting-Tool
├── optimize_strategy.py  # Strategie-Optimierer
├── requirements.txt      # Abhängigkeiten
└── README.md             # Dokumentation
```

## Sicherheitshinweise

- **WICHTIG**: Verwende zunächst den Paper-Trading-Modus, um die Funktionalität ohne finanzielles Risiko zu testen
- Speichere API-Schlüssel ausschließlich in der `.env`-Datei (niemals im Code)
- Beschränke die API-Berechtigungen in deinem Binance-Konto auf das Nötigste

## Robuste Fehlerbehandlung

Der Bot implementiert umfassende Fehlerbehandlung, um auch bei instabilen Marktbedingungen zuverlässig zu funktionieren:

- **Safe Type Handling**: Verhindert Abstürze bei None-Werten oder NaN-Daten
- **Verbesserte Logging**: Detailliertes Logging mit Stack-Traces für bessere Fehleranalyse
- **Automatische Wiederherstellung**: Automatische Wiederaufnahme nach Verbindungsproblemen
- **Daten-Validierung**: Umfangreiche Prüfung auf fehlende oder ungültige Daten


## Risikomanagement

Der Bot verwendet folgende Risikomanagement-Techniken:

- Prozentuales Risiko pro Trade (konfigurierbar)
- Stop-Loss-Orders für alle Positionen
- Take-Profit-Orders für Gewinnmitnahme
- Automatische Positionsgrößenberechnung basierend auf Volatilität
- Begrenzung der maximalen Anzahl gleichzeitiger Positionen

## Fehlerbehandlung

Der Bot behandelt potenzielle `None`-Werte in Vergleichsoperationen, um den Fehler `TypeError: '>' not supported between instances of 'NoneType' and 'int'` zu vermeiden:

- Explizite Prüfung vor Vergleichen: `if value is not None and value > x`
- Standardwerte für fehlende Daten: `int(value or default_value)`
- Validierung aller API-Rückgaben vor der Verwendung

## Haftungsausschluss

Dieser Bot ist nur für Bildungs- und Forschungszwecke gedacht. Der Handel mit Kryptowährungen birgt erhebliche finanzielle Risiken. Verwende diesen Bot niemals mit mehr Kapital, als du bereit bist zu verlieren.

## Weiterentwicklung

Potenzielle Verbesserungen:
- Implementierung weiterer Strategien
- Optimierung der ML-Modelle
- Web-Interface zur Überwachung
- Integration weiterer Börsen
- Strategie-Optimierung mit genetischen Algorithmen
