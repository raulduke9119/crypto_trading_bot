#!/usr/bin/env python
"""
Test-Runner für den Binance Trading Bot.
Führt alle Tests aus und prüft die Funktionalität aller Komponenten.
"""
import os
import sys
import unittest
import argparse
import importlib.util
from typing import List, Dict, Any, Optional, Tuple

# Farbcodes für Terminal-Ausgabe
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def check_imports() -> Tuple[bool, List[str]]:
    """
    Überprüft, ob alle erforderlichen Module importiert werden können.
    
    Returns:
        Tuple[bool, List[str]]: (Erfolg, Liste der fehlgeschlagenen Importe)
    """
    required_modules = [
        "pandas",
        "numpy",
        {"name": "python-binance", "import_name": "binance"},
        "matplotlib",
        {"name": "python-dotenv", "import_name": "dotenv"},
        "ta",
    ]
    
    # Optional, aber empfohlen
    optional_modules = [
        "ta-lib",
        "tensorflow",
        "scikit-learn",
        "pytest",
        "mypy",
    ]
    
    failed_imports = []
    optional_failed = []
    
    print(f"{BLUE}Prüfe erforderliche Module...{RESET}")
    for module in required_modules:
        try:
            # Prüfe, ob es ein Modul mit separatem Importnamen ist
            if isinstance(module, dict):
                import_name = module["import_name"]
                display_name = module["name"]
            else:
                # Wandle Paketname in Importname um
                import_name = module.replace("-", "_")
                display_name = module
            
            __import__(import_name)
            print(f"  {GREEN}✓{RESET} {display_name}")
        except ImportError:
            if isinstance(module, dict):
                print(f"  {RED}✗{RESET} {module['name']}")
                failed_imports.append(module['name'])
            else:
                print(f"  {RED}✗{RESET} {module}")
                failed_imports.append(module)
    
    print(f"\n{BLUE}Prüfe optionale Module...{RESET}")
    for module in optional_modules:
        try:
            # Wandle Paketname in Importname um
            import_name = module.replace("-", "_")
            __import__(import_name)
            print(f"  {GREEN}✓{RESET} {module}")
        except ImportError:
            print(f"  {YELLOW}?{RESET} {module} (optional)")
            optional_failed.append(module)
    
    if failed_imports:
        print(f"\n{RED}Fehlende erforderliche Module:{RESET}")
        for module in failed_imports:
            print(f"  {RED}•{RESET} {module}")
        print(f"\nBitte installiere diese Module mit: pip install {' '.join(failed_imports)}")
    
    if optional_failed:
        print(f"\n{YELLOW}Fehlende optionale Module:{RESET}")
        for module in optional_failed:
            print(f"  {YELLOW}•{RESET} {module}")
        print(f"\nDiese Module sind optional, werden aber für einige Funktionen empfohlen.")
    
    return len(failed_imports) == 0, failed_imports


def check_directory_structure() -> bool:
    """
    Überprüft, ob die Verzeichnisstruktur korrekt ist.
    
    Returns:
        bool: True wenn alle erforderlichen Verzeichnisse existieren
    """
    required_dirs = [
        "config",
        "data",
        "strategies",
        "tests",
        "utils",
    ]
    
    print(f"\n{BLUE}Prüfe Verzeichnisstruktur...{RESET}")
    missing_dirs = []
    
    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"  {GREEN}✓{RESET} {directory}/")
        else:
            print(f"  {RED}✗{RESET} {directory}/")
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"\n{RED}Fehlende Verzeichnisse:{RESET}")
        for directory in missing_dirs:
            print(f"  {RED}•{RESET} {directory}/")
        print("\nBitte erstelle diese Verzeichnisse manuell.")
        return False
    
    return True


def check_key_files() -> bool:
    """
    Überprüft, ob alle wichtigen Dateien vorhanden sind.
    
    Returns:
        bool: True wenn alle erforderlichen Dateien existieren
    """
    required_files = [
        "trading_bot.py",
        "config/config.py",
        "strategies/base_strategy.py",
        "strategies/multi_indicator_strategy.py",
        "strategies/dogebtc_hf_strategy.py",
        "data/data_collector.py",
        "data/indicators.py",
        "tests/test_dogebtc_hf_strategy.py",
    ]
    
    print(f"\n{BLUE}Prüfe wichtige Dateien...{RESET}")
    missing_files = []
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"  {GREEN}✓{RESET} {file_path}")
        else:
            print(f"  {RED}✗{RESET} {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n{RED}Fehlende Dateien:{RESET}")
        for file_path in missing_files:
            print(f"  {RED}•{RESET} {file_path}")
        print("\nBitte erstelle oder lade diese Dateien von deinem Quellcode-Repository.")
        return False
    
    return True


def check_environment_variables() -> bool:
    """
    Überprüft, ob die erforderlichen Umgebungsvariablen gesetzt sind.
    
    Returns:
        bool: True wenn .env vorhanden oder Umgebungsvariablen gesetzt sind
    """
    print(f"\n{BLUE}Prüfe Umgebungsvariablen...{RESET}")
    
    # Prüfe, ob .env existiert
    if os.path.isfile(".env"):
        print(f"  {GREEN}✓{RESET} .env Datei gefunden")
        try:
            import dotenv
            dotenv.load_dotenv()
            print(f"  {GREEN}✓{RESET} Umgebungsvariablen aus .env geladen")
        except ImportError:
            print(f"  {YELLOW}!{RESET} python-dotenv nicht installiert, kann Variablen nicht laden")
    else:
        print(f"  {YELLOW}!{RESET} Keine .env Datei gefunden")
        print(f"  {YELLOW}!{RESET} Prüfe stattdessen direkte Umgebungsvariablen...")
    
    # Liste der benötigten Umgebungsvariablen
    env_vars = [
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "BINANCE_TESTNET_API_KEY",
        "BINANCE_TESTNET_API_SECRET",
    ]
    
    # Überprüfe, ob Umgebungsvariablen gesetzt sind
    missing_vars = []
    for var in env_vars:
        if var in os.environ:
            value = os.environ[var]
            masked_value = value[:3] + "*" * (len(value) - 3) if value else ""
            print(f"  {GREEN}✓{RESET} {var}={masked_value}")
        else:
            print(f"  {YELLOW}?{RESET} {var} nicht gesetzt")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n{YELLOW}Fehlende Umgebungsvariablen:{RESET}")
        for var in missing_vars:
            print(f"  {YELLOW}•{RESET} {var}")
        print("\nBitte setze diese Variablen in der .env Datei oder direkt im System.")
        return False
    
    return True


def run_unit_tests(verbose: bool = False) -> bool:
    """
    Führt alle Unit-Tests aus.
    
    Args:
        verbose: Wenn True, werden ausführliche Testausgaben angezeigt.
        
    Returns:
        bool: True wenn alle Tests bestanden wurden
    """
    print(f"\n{BLUE}Führe Unit-Tests aus...{RESET}")
    
    test_loader = unittest.TestLoader()
    test_runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    
    # Suche nach Tests in tests/-Verzeichnis
    try:
        test_suite = test_loader.discover("tests", pattern="test_*.py")
        results = test_runner.run(test_suite)
        
        if results.wasSuccessful():
            print(f"\n{GREEN}Alle Unit-Tests bestanden!{RESET}")
            return True
        else:
            print(f"\n{RED}Einige Unit-Tests sind fehlgeschlagen!{RESET}")
            return False
    except Exception as e:
        print(f"{RED}Fehler beim Ausführen der Tests: {e}{RESET}")
        return False


def test_data_collector() -> bool:
    """
    Testet den Datensammler.
    
    Returns:
        bool: True wenn der Test erfolgreich war
    """
    print(f"\n{BLUE}Teste Datensammler...{RESET}")
    
    try:
        # Importiere den Datensammler
        spec = importlib.util.spec_from_file_location(
            "data_collector", "data/data_collector.py"
        )
        data_collector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_collector_module)
        
        # Hole Testdaten (minimaler Zeitraum)
        data_collector = data_collector_module.DataCollector()
        data = data_collector.get_historical_data(
            symbol="BTCUSDT",
            interval="1h",
            start_str="1 day ago"
        )
        
        if data is not None and not data.empty:
            print(f"  {GREEN}✓{RESET} Datensammler konnte historische Daten abrufen")
            print(f"      Symbol: BTCUSDT, Intervall: 1h")
            print(f"      Datenpunkte: {len(data)}")
            print(f"      Erste Zeile:\n{data.head(1)}")
            return True
        else:
            print(f"  {RED}✗{RESET} Datensammler konnte keine historischen Daten abrufen")
            return False
    except Exception as e:
        print(f"  {RED}✗{RESET} Fehler beim Testen des Datensammlers: {e}")
        return False


def test_strategy() -> bool:
    """
    Testet die Basis-Strategie.
    
    Returns:
        bool: True wenn der Test erfolgreich war
    """
    print(f"\n{BLUE}Teste Strategien...{RESET}")
    
    try:
        # Importiere Module
        import pandas as pd
        
        # Erstelle Test-DataFrame
        df = pd.DataFrame({
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [1000.0, 1100.0, 1200.0],
        })
        
        # Importiere die DogebtcHFStrategy
        spec = importlib.util.spec_from_file_location(
            "dogebtc_hf_strategy", "strategies/dogebtc_hf_strategy.py"
        )
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)
        
        # Erstelle Strategie-Instanz
        strategy = strategy_module.DogebtcHFStrategy(risk_percent=1.0)
        
        # Lade Indikatoren
        spec = importlib.util.spec_from_file_location(
            "indicators", "data/indicators.py"
        )
        indicators_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(indicators_module)
        
        # Füge Indikatoren hinzu
        ti = indicators_module.TechnicalIndicators()
        df_with_indicators = ti.add_all_indicators(df)
        
        # Generiere Signale
        try:
            result = strategy.generate_signals(df_with_indicators)
            print(f"  {GREEN}✓{RESET} DogebtcHFStrategy konnte Signale generieren")
            return True
        except Exception as e:
            print(f"  {RED}✗{RESET} Fehler bei der Signalgenerierung: {e}")
            return False
        
    except Exception as e:
        print(f"  {RED}✗{RESET} Fehler beim Testen der Strategie: {e}")
        return False


def main() -> None:
    """Hauptfunktion."""
    parser = argparse.ArgumentParser(description="Test-Runner für den Binance Trading Bot")
    parser.add_argument("-v", "--verbose", action="store_true", help="Ausführliche Ausgabe")
    parser.add_argument("--skip-imports", action="store_true", help="Import-Prüfung überspringen")
    parser.add_argument("--skip-env", action="store_true", help="Umgebungsvariablen-Prüfung überspringen")
    parser.add_argument("--skip-unit-tests", action="store_true", help="Unit-Tests überspringen")
    parser.add_argument("--skip-data-collector", action="store_true", help="Datensammler-Test überspringen")
    parser.add_argument("--skip-strategy", action="store_true", help="Strategie-Test überspringen")
    args = parser.parse_args()
    
    print(f"{BLUE}=" * 80 + RESET)
    print(f"{BLUE}BINANCE TRADING BOT TEST-RUNNER{RESET}")
    print(f"{BLUE}=" * 80 + RESET)
    
    # Tests ausführen
    results = {}
    
    if not args.skip_imports:
        imports_ok, _ = check_imports()
        results["Modul-Importe"] = imports_ok
    
    structure_ok = check_directory_structure()
    results["Verzeichnisstruktur"] = structure_ok
    
    files_ok = check_key_files()
    results["Schlüsseldateien"] = files_ok
    
    if not args.skip_env:
        env_ok = check_environment_variables()
        results["Umgebungsvariablen"] = env_ok
    
    if not args.skip_unit_tests:
        tests_ok = run_unit_tests(args.verbose)
        results["Unit-Tests"] = tests_ok
    
    if not args.skip_data_collector:
        data_collector_ok = test_data_collector()
        results["Datensammler"] = data_collector_ok
    
    if not args.skip_strategy:
        strategy_ok = test_strategy()
        results["Strategie"] = strategy_ok
    
    # Zusammenfassung anzeigen
    print(f"\n{BLUE}=" * 80 + RESET)
    print(f"{BLUE}TESTERGEBNISSE ZUSAMMENFASSUNG{RESET}")
    print(f"{BLUE}=" * 80 + RESET)
    
    all_passed = True
    for test_name, passed in results.items():
        status = f"{GREEN}BESTANDEN{RESET}" if passed else f"{RED}FEHLGESCHLAGEN{RESET}"
        print(f"{test_name:20}: {status}")
        all_passed = all_passed and passed
    
    print(f"\n{BLUE}=" * 80 + RESET)
    if all_passed:
        print(f"{GREEN}ALLE TESTS BESTANDEN! Der Trading Bot ist bereit.{RESET}")
    else:
        print(f"{RED}EINIGE TESTS SIND FEHLGESCHLAGEN. Bitte behebe die Probleme.{RESET}")
    print(f"{BLUE}=" * 80 + RESET)


if __name__ == "__main__":
    main()
