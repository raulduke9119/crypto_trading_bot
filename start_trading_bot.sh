#!/bin/bash

# Source TensorFlow environment variables
if [ -f .env.tensorflow ]; then
  source .env.tensorflow
  echo "TensorFlow environment variables set"
fi

# Run the trading bot
python trading_bot.py "$@"
