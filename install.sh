#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   BINANCE TRADING BOT - INSTALLATIONSSCRIPT    ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Determine whether to use conda or venv
if command -v conda &> /dev/null; then
  echo -e "${GREEN}Conda detected. Using conda for environment setup...${NC}"
  USE_CONDA=true
else
  echo -e "${GREEN}Using Python virtual environment (venv)...${NC}"
  USE_CONDA=false
fi

if [ "$USE_CONDA" = true ]; then
  # Create conda environment if it doesn't exist
  if ! conda info --envs | grep -q "crypto_trading"; then
    echo -e "${GREEN}Creating conda environment 'crypto_trading'...${NC}"
    conda create -y -n crypto_trading python=3.9
  else
    echo -e "${GREEN}Conda environment 'crypto_trading' already exists.${NC}"
  fi

  # Activate the environment
  eval "$(conda shell.bash hook)"
  conda activate crypto_trading
else
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
fi

# Install pip dependencies
echo -e "${GREEN}Installing required Python packages...${NC}"
pip install --upgrade pip setuptools wheel

# Install numpy with compatible version for TensorFlow
echo -e "${GREEN}Installing numpy with compatible version...${NC}"
pip install "numpy<1.24.0"

# Install TensorFlow without specifying version to get the latest compatible version
echo -e "${GREEN}Installing TensorFlow...${NC}"
pip install tensorflow

# Install other required packages
echo -e "${GREEN}Installing other required packages...${NC}"
pip install pandas matplotlib scikit-learn ccxt python-binance 
pip install pandas-ta plotly websocket-client python-dotenv joblib

# Check if CUDA is installed and set up properly
if command -v nvidia-smi &> /dev/null; then
  echo -e "${GREEN}CUDA detected. Setting up TensorFlow with GPU support...${NC}"
  
  # Set environment variables to suppress warnings and optimize TensorFlow
  if ! grep -q "TF_CPP_MIN_LOG_LEVEL" ~/.bashrc; then
    echo -e "${GREEN}Adding TensorFlow environment variables to ~/.bashrc...${NC}"
    cat >> ~/.bashrc << EOF

# TensorFlow environment variables
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
EOF
  fi
  
  # One-time fix for current session
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  export TF_CPP_MIN_LOG_LEVEL=2
  export TF_ENABLE_ONEDNN_OPTS=0
  export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
  
  # Create a .env.tensorflow file for easy environment setup
  echo -e "${GREEN}Creating .env.tensorflow file for easy environment setup...${NC}"
  cat > .env.tensorflow << EOF
# TensorFlow environment variables to suppress warnings
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
EOF
  chmod +x .env.tensorflow
else
  echo -e "${YELLOW}CUDA not detected. TensorFlow will run in CPU mode.${NC}"
  
  # Still add some environment variables to suppress CPU warnings
  if ! grep -q "TF_CPP_MIN_LOG_LEVEL" ~/.bashrc; then
    echo -e "${GREEN}Adding TensorFlow environment variables to ~/.bashrc...${NC}"
    cat >> ~/.bashrc << EOF

# TensorFlow environment variables
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
EOF
  fi
  
  # One-time fix for current session
  export TF_CPP_MIN_LOG_LEVEL=2
  export TF_ENABLE_ONEDNN_OPTS=0
  
  # Create a .env.tensorflow file for easy environment setup
  echo -e "${GREEN}Creating .env.tensorflow file for easy environment setup...${NC}"
  cat > .env.tensorflow << EOF
# TensorFlow environment variables to suppress warnings
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
EOF
  chmod +x .env.tensorflow
fi

# Create a startup script wrapper for the trading bot
echo -e "${GREEN}Creating startup script for the trading bot...${NC}"
cat > start_trading_bot.sh << 'EOF'
#!/bin/bash

# Source TensorFlow environment variables
if [ -f .env.tensorflow ]; then
  source .env.tensorflow
  echo "TensorFlow environment variables set"
fi

# Run the trading bot
python trading_bot.py "$@"
EOF
chmod +x start_trading_bot.sh

# Prüfe auf Betriebssystem für TA-Lib
echo -e "\n${BLUE}Prüfe Betriebssystem für TA-Lib Installation...${NC}"
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS="Linux";;
    Darwin*)    OS="MacOS";;
    CYGWIN*)    OS="Windows";;
    MINGW*)     OS="Windows";;
    *)          OS="Unknown";;
esac

if [ "$OS" = "Linux" ] || [ "$OS" = "MacOS" ]; then
    echo -e "${GREEN}✓${NC} $OS erkannt"
else
    echo -e "${RED}✗${NC} Dein Betriebssystem ($OS) wird möglicherweise nicht unterstützt."
    echo -e "${YELLOW}⚠${NC} Versuche trotzdem fortzufahren..."
fi

echo -e "${YELLOW}⚠${NC} Die Installation von TA-Lib erfordert möglicherweise Root-Rechte."
read -p "Möchtest du versuchen, TA-Lib zu installieren? (j/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Jj]$ ]]; then
    INSTALL_TALIB=true
    
    # Installation von TA-Lib
    echo -e "${BLUE}Installiere TA-Lib...${NC}"
    
    # Tools to download ta-lib
    DOWNLOAD_CMD=""
    if command -v wget &> /dev/null; then
        DOWNLOAD_CMD="wget -q"
    elif command -v curl &> /dev/null; then
        DOWNLOAD_CMD="curl -s -o ta-lib-0.4.0-src.tar.gz"
    else
        echo -e "${RED}✗${NC} Weder wget noch curl gefunden. Bitte installiere eines dieser Programme."
        echo -e "${YELLOW}Überspringe TA-Lib-Installation.${NC}"
        INSTALL_TALIB=false
    fi
    
    if [ "$INSTALL_TALIB" = true ]; then
        # Create temporary directory
        mkdir -p tmp
        cd tmp
        
        # Download and extract TA-Lib source
        echo -e "${GREEN}Downloading TA-Lib source...${NC}"
        if [[ $DOWNLOAD_CMD == wget* ]]; then
            wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
        else
            curl -s -o ta-lib-0.4.0-src.tar.gz http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
        fi
        
        echo -e "${GREEN}Extracting TA-Lib source...${NC}"
        tar -xzf ta-lib-0.4.0-src.tar.gz
        cd ta-lib/
        
        # Configure, make and install
        echo -e "${GREEN}Configuring and building TA-Lib...${NC}"
        ./configure --prefix=/usr/local
        make
        
        # Install (may require sudo)
        echo -e "${YELLOW}Installing TA-Lib system-wide (may require password)...${NC}"
        sudo make install
        
        # Update shared library cache and set environment variables
        if [[ "$OS" == "Linux" ]]; then
            echo -e "${GREEN}Updating shared library cache...${NC}"
            sudo ldconfig
            
            # Add TA-Lib library path to environment if not already there
            if ! grep -q "/usr/local/lib" /etc/ld.so.conf.d/*.conf; then
                echo -e "${GREEN}Adding TA-Lib to library path...${NC}"
                echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/talib.conf
                sudo ldconfig
            fi
        fi
        
        # Export library path for current session
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        
        # Clean up
        cd ../..
        rm -rf tmp
        
        # Install Python wrapper with explicit pointers to the libraries
        echo -e "${GREEN}Installing Python TA-Lib wrapper...${NC}"
        pip install --no-cache-dir --no-binary :all: TA-Lib
        
        echo -e "${GREEN}✓${NC} TA-Lib installation completed!"
    fi
else
    INSTALL_TALIB=false
    echo -e "${YELLOW}TA-Lib installation skipped.${NC}"
    echo -e "${YELLOW}You can later install it manually by following instructions at:${NC}"
    echo -e "${YELLOW}https://github.com/mrjbq7/ta-lib#installation${NC}"
fi

# Process requirements.txt with compatibility fixes
if [ -f "requirements.txt" ]; then
    echo -e "${BLUE}Installiere kompatible Abhängigkeiten aus requirements.txt...${NC}"
    
    # Create a temporary file without specific versions for problematic packages
    grep -v "^numpy==" requirements.txt | grep -v "^tensorflow==" | grep -v "^scikit-learn==" | grep -v "^ta-lib==" > requirements_compatible.txt
    
    # Install from modified requirements
    pip install -r requirements_compatible.txt
    
    # Clean up
    rm requirements_compatible.txt
    
    echo -e "${GREEN}✓${NC} Abhängigkeiten installiert"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
  echo -e "${GREEN}Creating sample .env file...${NC}"
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

# Trading parameters
TRADING_PAIR=BTCUSDT
TIMEFRAME=1h
TRADE_AMOUNT=0.001
STOP_LOSS_PERCENT=2
TAKE_PROFIT_PERCENT=4

# Model parameters
LOOKBACK_PERIOD=30
EOF
  echo -e "${RED}Please edit the .env.example file, rename it to .env, and add your API keys and parameters.${NC}"
else
  echo -e "${GREEN}.env file already exists.${NC}"
fi

# Erstelle erforderliche Verzeichnisse
echo -e "\n${BLUE}Erstelle erforderliche Verzeichnisse...${NC}"
mkdir -p logs data output models
echo -e "${GREEN}✓${NC} Verzeichnisse erstellt/überprüft"

# Test the installation
echo -e "${GREEN}Testing the installation...${NC}"
# Suppress TensorFlow warnings for the test
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU Available:', bool(tf.config.list_physical_devices('GPU')))"
python -c "import numpy as np; print('NumPy version:', np.__version__)"
python -c "import pandas as pd; print('Pandas version:', pd.__version__)"

# Display TA-Lib status
if [ "$INSTALL_TALIB" = true ]; then
    echo -e "${GREEN}Testing TA-Lib installation...${NC}"
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib python -c "import talib; print('TA-Lib version:', talib.__version__)" 2>/dev/null && echo -e "${GREEN}✓${NC} TA-Lib successfully installed!" || echo -e "${RED}✗${NC} TA-Lib installation failed. Try installing it manually: https://github.com/mrjbq7/ta-lib#installation"
fi

# Führe Tests aus
echo -e "\n${BLUE}Möchtest du die Tests ausführen, um die Installation zu überprüfen?${NC}"
read -p "Tests ausführen? (j/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Jj]$ ]]; then
    echo -e "${BLUE}Führe Tests aus...${NC}"
    if [ -f "run_tests.py" ]; then
        # Run tests with TensorFlow warnings suppressed
        TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 python run_tests.py --skip-env
    else
        echo -e "${RED}run_tests.py nicht gefunden. Tests können nicht ausgeführt werden.${NC}"
    fi
else
    echo -e "${YELLOW}Tests übersprungen.${NC}"
fi

# Installation abgeschlossen
echo -e "\n${BLUE}=================================================${NC}"
echo -e "${GREEN}   INSTALLATION ABGESCHLOSSEN!   ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "\nNächste Schritte:"
echo -e "1. ${YELLOW}Kopiere .env.example zu .env und füge deine Binance API-Keys ein${NC}"
echo -e "2. ${YELLOW}Passe die Konfiguration an deine Bedürfnisse an${NC}"
echo -e "3. ${YELLOW}Starte den Bot mit: ./start_trading_bot.sh${NC}"
echo -e "   ${GREEN}(Dies verwendet die optimierten TensorFlow-Einstellungen)${NC}"
echo -e "\nViel Erfolg mit deinem Trading Bot!"
