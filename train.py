import os
import logging
import MetaTrader5 as mt5
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from rl_env.forex_env import ForexEnv
from utils.data_processor import get_or_cache_data
from config.trading_config import (
    SYMBOL, TIMEFRAME, TRAINING_YEAR,
    MODEL_PATH, DATA_PATH,
    TRAINING_PARAMS, TRADING_PARAMS
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    """Train the trading model using data from the specified year"""
    try:
        # Initialize MT5 connection
        if not mt5.initialize():
            logger.error("Failed to initialize MetaTrader5")
            return
        
        logger.info(f"Fetching training data for {SYMBOL} {TRAINING_YEAR}")
        df = get_or_cache_data(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            year=TRAINING_YEAR,
            data_dir=DATA_PATH
        )
        
        if df is None or df.empty:
            logger.error("Failed to get training data")
            return
        
        logger.info(f"Creating Forex environment with {len(df)} data points")
        env = ForexEnv(
            df=df,
            initial_balance=TRADING_PARAMS['initial_balance'],
            lot_size=TRADING_PARAMS['lot_size'],
            max_positions=TRADING_PARAMS['max_positions'],
            stop_loss_pips=TRADING_PARAMS['stop_loss_pips'],
            take_profit_pips=TRADING_PARAMS['take_profit_pips']
        )
        
        # Validate the environment
        logger.info("Validating environment...")
        check_env(env)
        
        # Create callback for model evaluation
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=TRAINING_PARAMS['reward_threshold'],
            verbose=1
        )
        eval_callback = EvalCallback(
            env,
            best_model_save_path=os.path.dirname(MODEL_PATH),
            log_path=os.path.dirname(MODEL_PATH),
            eval_freq=TRAINING_PARAMS['eval_freq'],
            deterministic=True,
            render=False,
            callback_after_eval=stop_callback
        )
        
        # Initialize and train the model
        logger.info("Initializing PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=TRAINING_PARAMS['learning_rate'],
            n_steps=TRAINING_PARAMS['n_steps'],
            batch_size=TRAINING_PARAMS['batch_size'],
            n_epochs=TRAINING_PARAMS['n_epochs'],
            gamma=TRAINING_PARAMS['gamma'],
            tensorboard_log=os.path.join(os.path.dirname(MODEL_PATH), 'tensorboard')
        )
        
        logger.info("Starting model training...")
        model.learn(
            total_timesteps=TRAINING_PARAMS['total_timesteps'],
            callback=eval_callback
        )
        
        # Save the final model
        model_filename = f"{os.path.splitext(MODEL_PATH)[0]}_{TRAINING_YEAR}.zip"
        model.save(model_filename)
        logger.info(f"Model saved to {model_filename}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    train_model()
