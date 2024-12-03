"""
Configuration settings for the trading environment and model training
"""

import os

# Trading parameters
TRADING_PARAMS = {
    'symbol': 'USDJPY',
    'timeframe': '5',  # 5-minute timeframe
    'initial_balance': 10000,
    'lot_size': 0.1,
    'max_positions': 1,
    'stop_loss_pips': 30,
    'take_profit_pips': 60,
}

# Training parameters
TRAINING_PARAMS = {
    'total_timesteps': 100000,
    'learning_rate': 0.0003,
    'batch_size': 64,
    'n_steps': 2048,
    'n_epochs': 10,
    'gamma': 0.99,
    'reward_threshold': 1000,  # Stop training if mean reward reaches this threshold
    'eval_freq': 10000,  # Evaluate model every n steps
    'verbose': 1
}

# Data parameters
SYMBOL = TRADING_PARAMS['symbol']
TIMEFRAME = TRADING_PARAMS['timeframe']
TRAINING_YEAR = 2023
TESTING_YEAR = 2024

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', f'ppo_{SYMBOL}_{TIMEFRAME}m.zip')

# Create directories if they don't exist
for path in [DATA_PATH, os.path.dirname(MODEL_PATH)]:
    os.makedirs(path, exist_ok=True)

# Backtesting parameters
BACKTEST_PARAMS = {
    'initial_balance': 10000,
    'commission': 0.0001,  # 0.01% commission per trade
    'slippage': 0.0001,    # 0.01% slippage per trade
}
