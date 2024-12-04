import os
import logging
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
from stable_baselines3 import PPO
from rl_env.forex_env import ForexEnv
from utils.data_processor import get_or_cache_data
from config.trading_config import (
    SYMBOL, TIMEFRAME, TESTING_YEAR,
    MODEL_PATH, DATA_PATH,
    TRADING_PARAMS
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    """Test the trained model using data from the specified year"""
    try:
        # Initialize MT5 connection
        if not mt5.initialize():
            logger.error("Failed to initialize MetaTrader5")
            return
        
        # Load the trained model
        model_filename = "models/final_model.zip"
        if not os.path.exists(model_filename):
            logger.error(f"Model file not found: {model_filename}")
            return
        
        logger.info(f"Loading model from {model_filename}")
        model = PPO.load(model_filename)
        
        logger.info(f"Fetching test data for {SYMBOL} {TESTING_YEAR}")
        df = get_or_cache_data(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            year=TESTING_YEAR,
            data_dir=DATA_PATH
        )
        
        if df is None or df.empty:
            logger.error("Failed to get test data")
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
        
        # Run test episodes
        logger.info("Starting model testing...")
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        trades = []
        
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if info['trade_executed']:
                trades.append({
                    'type': info['trade_type'],
                    'price': info['execution_price'],
                    'profit': info['trade_profit']
                })
        
        # Calculate performance metrics
        win_rate = (env.winning_trades / env.total_trades * 100) if env.total_trades > 0 else 0
        final_balance = env.balance
        profit_percentage = ((final_balance - TRADING_PARAMS['initial_balance']) / 
                           TRADING_PARAMS['initial_balance'] * 100)
        
        # Log results
        logger.info("\nTesting Results:")
        logger.info(f"Total Trades: {env.total_trades}")
        logger.info(f"Winning Trades: {env.winning_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Initial Balance: ${TRADING_PARAMS['initial_balance']:.2f}")
        logger.info(f"Final Balance: ${final_balance:.2f}")
        logger.info(f"Profit/Loss: ${final_balance - TRADING_PARAMS['initial_balance']:.2f}")
        logger.info(f"Profit Percentage: {profit_percentage:.2f}%")
        
        # Save detailed trade history
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            results_dir = os.path.join(os.path.dirname(MODEL_PATH), 'results')
            os.makedirs(results_dir, exist_ok=True)
            trades_file = os.path.join(results_dir, f'trades_{SYMBOL}_{TESTING_YEAR}.csv')
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"\nDetailed trade history saved to {trades_file}")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    test_model()
