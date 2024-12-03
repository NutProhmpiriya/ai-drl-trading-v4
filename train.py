import os
import sys
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from config import TRADING_PARAMS, TRAINING_PARAMS, PATHS
from rl_env.forex_env import ForexEnv
from utils.data_processor import DataProcessor

def train_model():
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Get training data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Get last 30 days of data
    df = data_processor.fetch_data(
        symbol=TRADING_PARAMS['symbol'],
        timeframe=TRADING_PARAMS['timeframe'],
        start_date=start_date,
        end_date=end_date
    )
    
    if df is None:
        print("Failed to fetch data")
        return
    
    # Create and configure the environment
    env = ForexEnv(
        df=df,
        initial_balance=TRADING_PARAMS['initial_balance'],
        lot_size=TRADING_PARAMS['lot_size'],
        max_positions=TRADING_PARAMS['max_positions'],
        stop_loss_pips=TRADING_PARAMS['stop_loss_pips'],
        take_profit_pips=TRADING_PARAMS['take_profit_pips']
    )
    
    # Create models directory if it doesn't exist
    os.makedirs(PATHS['models_dir'], exist_ok=True)
    
    # Initialize the model
    model = PPO(
        policy=TRAINING_PARAMS['policy'],
        env=env,
        learning_rate=TRAINING_PARAMS['learning_rate'],
        batch_size=TRAINING_PARAMS['batch_size'],
        n_steps=TRAINING_PARAMS['n_steps'],
        gamma=TRAINING_PARAMS['gamma'],
        verbose=TRAINING_PARAMS['verbose']
    )
    
    # Train the model
    model.learn(total_timesteps=TRAINING_PARAMS['total_timesteps'])
    
    # Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        PATHS['models_dir'],
        f"ppo_forex_{TRADING_PARAMS['symbol']}_{TRADING_PARAMS['timeframe']}m_{timestamp}.zip"
    )
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
