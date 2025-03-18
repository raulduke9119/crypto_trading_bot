#!/bin/bash
# Installations-Script für den Binance Trading Bot

# Farbdefinitionen
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   BINANCE TRADING BOT - INSTALLATIONSSCRIPT    ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Prüfe Python-Installation
echo -e "\n${BLUE}Prüfe Python-Installation...${NC}"
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
    echo -e "${GREEN}✓${NC} Python 3 gefunden!"
    python3 --version
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
    echo -e "${GREEN}✓${NC} Python gefunden!"
    python --version
else
    echo -e "${RED}✗${NC} Python wurde nicht gefunden. Bitte installiere Python 3.8 oder höher."
    exit 1
fi

# Prüfe Python-Version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_VERSION_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_VERSION_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_VERSION_MAJOR" -lt 3 ] || ([ "$PYTHON_VERSION_MAJOR" -eq 3 ] && [ "$PYTHON_VERSION_MINOR" -lt 8 ]); then
    echo -e "${YELLOW}⚠${NC} Python $PYTHON_VERSION gefunden. Empfohlen ist Python 3.8 oder höher."
    read -p "Möchtest du trotzdem fortfahren? (j/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Jj]$ ]]; then
        echo -e "${RED}Installation abgebrochen.${NC}"
        exit 1
    fi
fi

# Erstelle Virtual Environment
echo -e "\n${BLUE}Erstelle Virtual Environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠${NC} Virtual Environment existiert bereits. Möchtest du es neu erstellen?"
    read -p "Neu erstellen? (j/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Jj]$ ]]; then
        echo -e "${YELLOW}Lösche bestehendes Virtual Environment...${NC}"
        rm -rf venv
        $PYTHON_CMD -m venv venv
        echo -e "${GREEN}✓${NC} Neues Virtual Environment erstellt"
    else
        echo -e "${GREEN}✓${NC} Bestehendes Virtual Environment wird verwendet"
    fi
else
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✓${NC} Virtual Environment erstellt"
fi

# Aktiviere Virtual Environment
echo -e "\n${BLUE}Aktiviere Virtual Environment...${NC}"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi
echo -e "${GREEN}✓${NC} Virtual Environment aktiviert"

# Aktualisiere pip, setuptools, wheel
echo -e "\n${BLUE}Aktualisiere grundlegende Pakete...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓${NC} Grundlegende Pakete aktualisiert"

# Prüfe auf Betriebssystem für TA-Lib
echo -e "\n${BLUE}Prüfe Betriebssystem für TA-Lib Installation...${NC}"
OS="$(uname)"
INSTALL_TALIB=true

case "$OS" in
    Linux*)
        echo -e "${GREEN}✓${NC} Linux erkannt"
        echo -e "${YELLOW}⚠${NC} Die Installation von TA-Lib erfordert möglicherweise Root-Rechte."
        read -p "Möchtest du versuchen, TA-Lib zu installieren? (j/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Jj]$ ]]; then
            echo -e "${BLUE}Installiere TA-Lib-Abhängigkeiten...${NC}"
            if command -v apt-get &>/dev/null; then
                sudo apt-get update
                sudo apt-get install -y build-essential libta-lib0 libta-lib-dev
            elif command -v yum &>/dev/null; then
                sudo yum install -y gcc gcc-c++ make ta-lib-devel
            else
                echo -e "${YELLOW}⚠${NC} Konnte Paketmanager nicht erkennen. Bitte installiere TA-Lib manuell."
                INSTALL_TALIB=false
            fi
        else
            INSTALL_TALIB=false
        fi
        ;;
    Darwin*)
        echo -e "${GREEN}✓${NC} macOS erkannt"
        if command -v brew &>/dev/null; then
            echo -e "${BLUE}Installiere TA-Lib mit Homebrew...${NC}"
            brew install ta-lib
        else
            echo -e "${YELLOW}⚠${NC} Homebrew nicht gefunden. Bitte installiere TA-Lib manuell."
            INSTALL_TALIB=false
        fi
        ;;
    MINGW*|MSYS*)
        echo -e "${GREEN}✓${NC} Windows erkannt"
        echo -e "${YELLOW}⚠${NC} TA-Lib-Installation auf Windows ist kompliziert."
        echo -e "Bitte folge den Anweisungen unter: https://github.com/mrjbq7/ta-lib#windows"
        INSTALL_TALIB=false
        ;;
    *)
        echo -e "${YELLOW}⚠${NC} Unbekanntes Betriebssystem: $OS"
        echo -e "Bitte installiere TA-Lib manuell."
        INSTALL_TALIB=false
        ;;
esac

# Installiere Abhängigkeiten
echo -e "\n${BLUE}Installiere Python-Abhängigkeiten...${NC}"

# Installiere zuerst NumPy
pip install numpy==1.24.3

# Installiere TA-Lib, wenn möglich
if [ "$INSTALL_TALIB" = true ]; then
    echo -e "${BLUE}Installiere TA-Lib...${NC}"
    pip install TA-Lib==0.4.28
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} TA-Lib erfolgreich installiert"
    else
        echo -e "${RED}✗${NC} TA-Lib-Installation fehlgeschlagen. Installiere andere Abhängigkeiten trotzdem weiter."
        # Modifiziere requirements.txt temporär, um TA-Lib zu überspringen
        sed -i.bak '/ta-lib/d' requirements.txt
    fi
else
    # Modifiziere requirements.txt temporär, um TA-Lib zu überspringen
    sed -i.bak '/ta-lib/d' requirements.txt
fi

# Installiere restliche Abhängigkeiten
echo -e "${BLUE}Installiere restliche Abhängigkeiten...${NC}"
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Alle Abhängigkeiten erfolgreich installiert"
else
    echo -e "${RED}✗${NC} Es gab Fehler bei der Installation der Abhängigkeiten."
    echo -e "Versuche, die grundlegenden Abhängigkeiten einzeln zu installieren..."
    
    # Versuche grundlegende Abhängigkeiten einzeln zu installieren
    pip install python-binance==1.0.16
    pip install pandas==1.5.3
    pip install matplotlib==3.7.1
    pip install python-dotenv==1.0.0
    pip install ta==0.10.2
    
    echo -e "${YELLOW}⚠${NC} Bitte überprüfe die obigen Fehler und installiere fehlende Pakete manuell."
fi

# Stelle sicher, dass die requirements.txt wieder im Originalzustand ist
if [ -f "requirements.txt.bak" ]; then
    mv requirements.txt.bak requirements.txt
fi

# Erstelle Umgebungsdatei, wenn sie nicht existiert
echo -e "\n${BLUE}Prüfe auf .env-Datei...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠${NC} Keine .env-Datei gefunden. Erstelle eine Beispieldatei..."
    cat > .env.example << EOF
# Binance API Konfiguration (LIVE)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Binance Testnet API Konfiguration (für Tests)
BINANCE_TESTNET_API_KEY=your_testnet_api_key_here
BINANCE_TESTNET_API_SECRET=your_testnet_api_secret_here

# App-Konfiguration
LOG_LEVEL=INFO
USE_TESTNET=True
RISK_PERCENTAGE=1.0
EOF
    echo -e "${GREEN}✓${NC} .env.example erstellt. Bitte kopiere zu .env und füge deine API-Keys ein."
fi

# Erstelle erforderliche Verzeichnisse
echo -e "\n${BLUE}Erstelle erforderliche Verzeichnisse...${NC}"
mkdir -p logs data output models
echo -e "${GREEN}✓${NC} Verzeichnisse erstellt/überprüft"

# Führe Tests aus
echo -e "\n${BLUE}Möchtest du die Tests ausführen, um die Installation zu überprüfen?${NC}"
read -p "Tests ausführen? (j/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Jj]$ ]]; then
    echo -e "${BLUE}Führe Tests aus...${NC}"
    $PYTHON_CMD run_tests.py --skip-env
else
    echo -e "${YELLOW}Tests übersprungen.${NC}"
fi

# Installation abgeschlossen
echo -e "\n${BLUE}=================================================${NC}"
echo -e "${GREEN}   INSTALLATION ABGESCHLOSSEN!   ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "\nNächste Schritte:"
echo -e "1. ${YELLOW}Kopiere .env.example zu .env und füge deine Binance API-Keys ein${NC}"
echo -e "2. ${YELLOW}Passe die Konfiguration in config/config.py an deine Bedürfnisse an${NC}"
echo -e "3. ${YELLOW}Starte den Bot mit: python trading_bot.py${NC}"
echo -e "\nViel Erfolg mit deinem Trading Bot!"
