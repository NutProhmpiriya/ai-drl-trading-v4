import os
import sys
from datetime import datetime, timedelta
import numpy as np
from stable_baselines3 import PPO
from config import TRADING_PARAMS, PATHS
from rl_env.forex_env import ForexEnv
from utils.data_processor import DataProcessor

def test_model(model_path: str):
    """Test a trained model on recent data"""
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Get test data (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    df = data_processor.fetch_data(
        symbol=TRADING_PARAMS['symbol'],
        timeframe=TRADING_PARAMS['timeframe'],
        start_date=start_date,
        end_date=end_date
    )
    
    if df is None:
        print("Failed to fetch test data")
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
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Test the model
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    
    # Print results
    print(f"\nTest Results:")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Balance: {env.balance:.2f}")
    print(f"Total Trades: {env.total_trades}")
    if env.total_trades > 0:
        win_rate = (env.winning_trades / env.total_trades) * 100
        print(f"Win Rate: {win_rate:.2f}%")
    
    return total_reward, env.balance, env.total_trades

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_model.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    
    test_model(model_path)
