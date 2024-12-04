import os
import logging
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from rl_env.forex_env import ForexEnv
from utils.data_processor import get_or_cache_data
from config.trading_config import (
    SYMBOL, TIMEFRAME,
    TRAIN_START, TRAIN_END,
    MODEL_PATH, DATA_PATH,
    TRADING_PARAMS
)
import MetaTrader5 as mt5
import pandas as pd
import ta
import time
import torch as th

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_env(df, params, log_dir=None):
    """Create and wrap the trading environment"""
    def _init():
        env = ForexEnv(
            df=df,
            initial_balance=params['initial_balance'],
            lot_size=params['lot_size'],
            max_positions=params['max_positions'],
            stop_loss_pips=params['stop_loss_pips'],
            take_profit_pips=params['take_profit_pips']
        )
        if log_dir:
            env = Monitor(env, log_dir)
        return env
    return _init

def load_data():
    """Load and prepare training data"""
    # Initialize MT5 connection
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        raise RuntimeError("Failed to initialize MT5")
        
    # Download data from MT5
    logger.info(f"Downloading data for {SYMBOL} from {TRAIN_START} to {TRAIN_END}")
    rates = mt5.copy_rates_range(
        SYMBOL,
        mt5.TIMEFRAME_M5,
        pd.Timestamp(TRAIN_START),
        pd.Timestamp(TRAIN_END)
    )
    
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data found for {SYMBOL} from {TRAIN_START} to {TRAIN_END}")
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Add technical indicators
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    logger.info(f"Loaded {len(df)} training samples from {df.index[0]} to {df.index[-1]}")
    return df

def train_model():
    """Train the model using PPO"""
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        tensorboard_dir = os.path.join(os.path.dirname(MODEL_PATH), 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)

        # Get training data
        logger.info(f"Loading training data for {SYMBOL} from {TRAIN_START} to {TRAIN_END}")
        train_data = load_data()
        if train_data is None:
            logger.error("Failed to load training data")
            return

        # Calculate total timesteps based on data size
        n_samples = len(train_data)
        total_timesteps = n_samples * 1  # Use a fixed multiplier
        logger.info(f"Training for {total_timesteps} timesteps (100000x {n_samples} samples)")

        # Create and configure the environment
        env = ForexEnv(
            df=train_data,
            initial_balance=float(TRADING_PARAMS['initial_balance']),
            lot_size=float(TRADING_PARAMS['lot_size']),
            max_positions=int(TRADING_PARAMS['max_positions']),
            stop_loss_pips=float(TRADING_PARAMS['stop_loss_pips']),
            take_profit_pips=float(TRADING_PARAMS['take_profit_pips']),
            leverage=float(TRADING_PARAMS['leverage']),
            margin_requirement=float(TRADING_PARAMS['margin_requirement'])
        )

        logger.info("\nTraining Configuration:")
        logger.info(f"Training period: {TRAIN_START} to {TRAIN_END}")
        logger.info(f"Number of training samples: {n_samples:,}")
        logger.info(f"Total training steps: {total_timesteps:,} steps")
        logger.info("\nStarting training in 3 seconds...")
        time.sleep(3)

        # Create the model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                activation_fn=th.nn.ReLU,
            ),
            tensorboard_log=tensorboard_dir,
            verbose=1
        )

        # Train the model
        model.learn(total_timesteps=total_timesteps)

        # Save the trained model
        model_save_path = os.path.join(os.path.dirname(MODEL_PATH), 'final_model.zip')
        model.save(model_save_path)
        logger.info(f"\nModel saved to {model_save_path}")

    except Exception as e:
        logger.error(f"Error during setup: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
