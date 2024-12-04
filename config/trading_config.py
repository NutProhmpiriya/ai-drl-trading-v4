"""
Configuration settings for the trading environment and model training
"""

import os
import MetaTrader5 as mt5

# Trading parameters
TRADING_PARAMS = {
    'symbol': 'USDJPY',
    'timeframe': '5',  # 5-minute timeframe
    'initial_balance': 10000,
    'lot_size': 0.1,
    'leverage': 100,  # เพิ่มเลเวอเรจ 1:100
    'margin_requirement': 0.01,  # 1% margin requirement (100:1 leverage)
    'max_positions': 1,
    'stop_loss_pips': 50,  # เพิ่ม stop loss เพื่อให้มีพื้นที่ในการทำกำไรมากขึ้น
    'take_profit_pips': 100,  # เพิ่ม take profit เพื่อให้คุ้มค่ากับความเสี่ยง
}

# Training parameters
TRAINING_PARAMS = {
    'total_timesteps': 1,  # ใช้เป็นตัวคูณกับจำนวนข้อมูล
    'learning_rate': 1e-5,  # ลด learning rate
    'batch_size': 128,  # เพิ่ม batch size
    'n_steps': 2048,
    'n_epochs': 10,
    'gamma': 0.99,
    'ent_coef': 0.01,  # เพิ่ม entropy coefficient
    'reward_threshold': 1000,  # Stop training if mean reward reaches this threshold
    'eval_freq': 10000,  # Evaluate model every n steps
    'verbose': 1
}

# Data parameters
SYMBOL = "USDJPY"
TIMEFRAME = mt5.TIMEFRAME_M5  # 5 minute timeframe

# Training period (2023)
TRAIN_START = "2023-01-01"
TRAIN_END = "2023-12-31"

# Testing period (2024)
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"

# Backtesting periods 
BACKTEST_PERIODS = [
    ("2024-01-01", "2024-12-31")
]

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
