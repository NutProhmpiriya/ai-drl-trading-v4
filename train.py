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
    SYMBOL,
    TIMEFRAME,
    TRAINING_YEAR,
    TESTING_YEAR,
    DATA_PATH,
    MODEL_PATH,
    TRADING_PARAMS,
    TRAINING_PARAMS
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch as th

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

def train_model():
    """Train the trading model using data from the specified year"""
    try:
        # Create directories if they don't exist
        model_dir = os.path.dirname(MODEL_PATH)
        tensorboard_dir = os.path.join(model_dir, 'tensorboard')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)

        # Get training data
        logger.info(f"Loading training data for {SYMBOL} {TRAINING_YEAR}")
        train_data = get_or_cache_data(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            year=TRAINING_YEAR,
            data_dir=DATA_PATH
        )
        if train_data is None:
            logger.error("Failed to load training data")
            return

        # Get test data
        logger.info(f"Loading test data for {SYMBOL} {TESTING_YEAR}")
        test_data = get_or_cache_data(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            year=TESTING_YEAR,
            data_dir=DATA_PATH
        )
        if test_data is None:
            logger.error("Failed to load test data")
            return

        # Create environments
        train_log_dir = os.path.join(tensorboard_dir, 'train')
        eval_log_dir = os.path.join(tensorboard_dir, 'eval')
        os.makedirs(train_log_dir, exist_ok=True)
        os.makedirs(eval_log_dir, exist_ok=True)

        env = DummyVecEnv([make_env(train_data, TRADING_PARAMS, log_dir=os.path.join(model_dir, 'logs'))])
        eval_env = DummyVecEnv([make_env(test_data, TRADING_PARAMS)])

        # Create the model with updated hyperparameters
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-5,  # Reduced learning rate
            n_steps=2048,  # Increased batch size
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,  # Increased entropy coefficient
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                activation_fn=th.nn.ReLU,
                optimizer_kwargs=dict(eps=1e-5),
            ),
            verbose=1,
            tensorboard_log=tensorboard_dir
        )

        # Create callback for evaluation
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=os.path.join(model_dir, 'logs'),
            eval_freq=10000,
            deterministic=True,
            render=False
        )

        # Train the model
        logger.info("Starting training...")
        try:
            model.learn(
                total_timesteps=TRAINING_PARAMS['total_timesteps'],
                callback=eval_callback,
                progress_bar=True
            )
            
            # Save the final model
            final_model_path = os.path.join(model_dir, "final_model")
            model.save(final_model_path)
            logger.info(f"Final model saved to {final_model_path}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error during setup: {str(e)}")
    finally:
        # Clean up environments
        if 'env' in locals():
            env.close()
        if 'eval_env' in locals():
            eval_env.close()

if __name__ == "__main__":
    train_model()
