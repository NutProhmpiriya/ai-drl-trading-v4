"""
Configuration settings for the trading environment and model training
"""

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
    'gamma': 0.99,
    'policy': 'MlpPolicy',
    'verbose': 1
}

# Data parameters
DATA_PARAMS = {
    'lookback_window': 100,  # Number of candles to look back
    'features': [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'EMA9', 'EMA21', 'EMA50', 'RSI', 'ATR', 'OBV'
    ],
}

# Paths
PATHS = {
    'data_dir': 'data',
    'models_dir': 'models',
    'logs_dir': 'logs',
    'results_dir': 'results'
}

# Backtesting parameters
BACKTEST_PARAMS = {
    'initial_balance': 10000,
    'commission': 0.0001,  # 0.01% commission per trade
    'slippage': 0.0001,    # 0.01% slippage per trade
}
