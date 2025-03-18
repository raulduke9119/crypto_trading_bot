"""
Setup-Script für den Binance Trading Bot.
Installiert alle benötigten Pakete und konfiguriert das Projekt.
"""
import os
import sys
import platform
import subprocess
from setuptools import setup, find_packages
from typing import List, Tuple, Dict, Optional, Any

# Projektmetadaten
VERSION = "1.2.0"
AUTHOR = "Trading Team"
DESCRIPTION = "Optimierter Trading Bot für Binance mit Multi-Indikator-Strategie"
URL = "https://github.com/yourusername/binance-trading-bot"
LICENSE = "MIT"

# Debug-Modus
DEBUG = "--debug" in sys.argv
if DEBUG:
    sys.argv.remove("--debug")


def read_requirements() -> List[str]:
    """Liest die requirements.txt und gibt sie als Liste zurück."""
    requirements = []
    try:
        with open("requirements.txt", "r") as req_file:
            for line in req_file:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    except Exception as e:
        print(f"Fehler beim Lesen der requirements.txt: {e}")
        print("Verwende Standard-Abhängigkeiten...")
        requirements = [
            "python-binance>=1.0.16",
            "pandas>=1.5.3",
            "numpy>=1.24.3",
            "matplotlib>=3.7.1",
            "python-dotenv>=1.0.0",
            "ta>=0.10.2",
        ]
    return requirements


def setup_directories() -> None:
    """Erstellt benötigte Verzeichnisse, falls sie nicht existieren."""
    directories = ["logs", "data", "output", "models"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Verzeichnis '{directory}' überprüft/erstellt")


def create_env_file() -> None:
    """Erstellt eine .env-Beispieldatei, wenn keine vorhanden ist."""
    if not os.path.exists(".env"):
        with open(".env.example", "w") as env_file:
            env_file.write("""# Binance API Konfiguration (LIVE)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Binance Testnet API Konfiguration (für Tests)
BINANCE_TESTNET_API_KEY=your_testnet_api_key_here
BINANCE_TESTNET_API_SECRET=your_testnet_api_secret_here

# App-Konfiguration
LOG_LEVEL=INFO
USE_TESTNET=True
RISK_PERCENTAGE=1.0
""")
        print("HINWEIS: .env.example wurde erstellt. Bitte in .env umbenennen und mit deinen API-Keys konfigurieren.")


def check_python_version() -> bool:
    """Überprüft, ob die Python-Version kompatibel ist."""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version.major < required_version[0] or (
        current_version.major == required_version[0] and 
        current_version.minor < required_version[1]
    ):
        print(f"WARNUNG: Python {required_version[0]}.{required_version[1]}+ wird empfohlen. "
              f"Du verwendest Python {current_version.major}.{current_version.minor}")
        return False
    return True


def install_talib() -> bool:
    """Installiert TA-Lib basierend auf dem Betriebssystem."""
    system = platform.system().lower()
    
    if DEBUG:
        print(f"Debug: Betriebssystem erkannt als {system}")
    
    try:
        if system == "linux":
            subprocess.check_call(
                ["pip", "install", "--no-cache-dir", "numpy==1.24.3"]
            )
            subprocess.check_call(
                ["apt-get", "update"]
            )
            subprocess.check_call(
                ["apt-get", "install", "-y", "build-essential", "libta-lib0", "libta-lib-dev"]
            )
            subprocess.check_call(
                ["pip", "install", "--no-cache-dir", "TA-Lib==0.4.28"]
            )
        elif system == "darwin":  # macOS
            subprocess.check_call(
                ["pip", "install", "--no-cache-dir", "numpy==1.24.3"]
            )
            subprocess.check_call(
                ["brew", "install", "ta-lib"]
            )
            subprocess.check_call(
                ["pip", "install", "--no-cache-dir", "TA-Lib==0.4.28"]
            )
        elif system == "windows":
            print("HINWEIS: Unter Windows musst du TA-Lib manuell installieren.")
            print("Anleitung: https://github.com/mrjbq7/ta-lib#windows")
            return False
        
        return True
    except Exception as e:
        print(f"Fehler bei der Installation von TA-Lib: {e}")
        print("HINWEIS: Du musst TA-Lib manuell installieren.")
        return False


def run_tests() -> bool:
    """Führt alle Tests aus."""
    try:
        print("Führe Tests aus...")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Alle Tests erfolgreich bestanden!")
            return True
        else:
            print("❌ Einige Tests sind fehlgeschlagen:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Fehler beim Ausführen der Tests: {e}")
        return False


def main() -> None:
    """Hauptfunktion für das Setup."""
    print("=" * 80)
    print("BINANCE TRADING BOT SETUP")
    print("=" * 80)
    
    # Überprüfe Python-Version
    python_ok = check_python_version()
    if not python_ok:
        print("Die Python-Version ist nicht optimal, aber das Setup wird fortgesetzt...")
    
    # Erstelle Verzeichnisse
    setup_directories()
    
    # Erstelle .env.example-Datei
    create_env_file()
    
    # Installiere TA-Lib (optional)
    print("\nMöchtest du versuchen, TA-Lib automatisch zu installieren? (y/n)")
    choice = input().lower()
    talib_installed = False
    
    if choice.startswith("y"):
        print("Installiere TA-Lib...")
        talib_installed = install_talib()
    else:
        print("TA-Lib-Installation übersprungen. Bitte manuell installieren, wenn nötig.")
    
    # Führe pip install aus
    try:
        requirements = read_requirements()
        if not talib_installed:
            # Entferne TA-Lib aus requirements, wenn nicht installiert
            requirements = [req for req in requirements if "ta-lib" not in req.lower()]
        
        print("\nInstalliere Abhängigkeiten...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", "."]
        )
        
        print("✅ Abhängigkeiten erfolgreich installiert!")
    except Exception as e:
        print(f"❌ Fehler bei der Installation: {e}")
    
    # Frage, ob Tests ausgeführt werden sollen
    print("\nMöchtest du die Tests ausführen? (y/n)")
    choice = input().lower()
    
    if choice.startswith("y"):
        tests_ok = run_tests()
        if tests_ok:
            print("\n✅ Das Setup wurde erfolgreich abgeschlossen!")
        else:
            print("\n⚠️ Setup abgeschlossen, aber einige Tests schlugen fehl.")
    else:
        print("\n✅ Setup abgeschlossen! Tests wurden übersprungen.")
    
    print("\n" + "=" * 80)
    print("NÄCHSTE SCHRITTE:")
    print("1. Erstelle eine .env-Datei mit deinen Binance API-Keys")
    print("2. Passe die Konfiguration in config/config.py an")
    print("3. Starte den Bot mit 'python trading_bot.py'")
    print("=" * 80)


if __name__ == "__main__":
    # Führe setup() aus, wenn als eigenständiges Skript aufgerufen
    if "--install" in sys.argv:
        setup(
            name="binance_trading_bot",
            version=VERSION,
            description=DESCRIPTION,
            author=AUTHOR,
            url=URL,
            license=LICENSE,
            packages=find_packages(),
            install_requires=read_requirements(),
            python_requires=">=3.8",
            classifiers=[
                "Development Status :: 4 - Beta",
                "Intended Audience :: Financial and Insurance Industry",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10",
                "Topic :: Office/Business :: Financial :: Investment",
            ],
        )
    else:
        main()
