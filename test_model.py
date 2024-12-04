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
    SYMBOL, TIMEFRAME,
    TEST_START, TEST_END,
    MODEL_PATH, DATA_PATH,
    TRADING_PARAMS, BACKTEST_PERIODS
)
import json
import ta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data():
    """Load and prepare test data"""
    # Initialize MT5 connection
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        raise RuntimeError("Failed to initialize MT5")
        
    # Download data from MT5
    logger.info(f"Downloading data for {SYMBOL} from {TEST_START} to {TEST_END}")
    rates = mt5.copy_rates_range(
        SYMBOL,
        mt5.TIMEFRAME_M5,
        pd.Timestamp(TEST_START),
        pd.Timestamp(TEST_END)
    )
    
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data found for {SYMBOL} from {TEST_START} to {TEST_END}")
    
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
    
    logger.info(f"Loaded {len(df)} testing samples from {df.index[0]} to {df.index[-1]}")
    return df

def load_data_for_period(start_date, end_date):
    """Load data for a specific time period"""
    logger.info(f"Loading data for {SYMBOL} from {start_date} to {end_date}")
    
    # Convert dates to datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Download and prepare data
    df = get_or_cache_data(
        symbol=SYMBOL,
        year=start.year,  
        timeframe=TIMEFRAME,
        data_dir=DATA_PATH
    )
    if df is None or len(df) == 0:
        raise ValueError(f"No data available for period {start_date} to {end_date}")
    
    # Filter data for the specified period
    df = df[(df.index >= start) & (df.index <= end)]
    logger.info(f"Loaded {len(df)} samples from {df.index[0]} to {df.index[-1]}")
    return df

def test_model():
    """Test the trained model"""
    try:
        # Load the trained model
        model_path = os.path.join(os.path.dirname(MODEL_PATH), 'final_model.zip')
        logger.info(f"Loading model from {model_path}")
        model = PPO.load(model_path)
        
        # Load testing data
        logger.info(f"Loading testing data for {SYMBOL} from {TEST_START} to {TEST_END}")
        df = load_data_for_period(TEST_START, TEST_END)
        
        # Create environment
        logger.info(f"Creating Forex environment with {len(df)} data points")
        env = ForexEnv(
            df=df,
            initial_balance=float(TRADING_PARAMS['initial_balance']),
            lot_size=float(TRADING_PARAMS['lot_size']),
            max_positions=int(TRADING_PARAMS['max_positions']),
            stop_loss_pips=float(TRADING_PARAMS['stop_loss_pips']),
            take_profit_pips=float(TRADING_PARAMS['take_profit_pips']),
            leverage=float(TRADING_PARAMS['leverage']),
            margin_requirement=float(TRADING_PARAMS['margin_requirement'])
        )
        
        # Test the model
        logger.info("Starting model testing...")
        obs, _ = env.reset()  # Get observation and info
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            step_result = env.step(action)
            obs = step_result[0]  # observation
            reward = step_result[1]  # reward
            done = step_result[2]  # done
            truncated = step_result[3] if len(step_result) > 3 else False  # truncated (if available)
        
        # Get trading results
        trades_df = pd.DataFrame(env.trades_history)
        
        # Calculate statistics
        if len(trades_df) > 0:
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            total_trades = len(trades_df)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate profit/loss
            total_profit = trades_df['profit'].sum()
            profit_percentage = (total_profit / float(TRADING_PARAMS['initial_balance'])) * 100
            
            # Log results
            logger.info("\nTesting Results:")
            logger.info(f"Total Trades: {total_trades}")
            logger.info(f"Winning Trades: {winning_trades}")
            logger.info(f"Win Rate: {win_rate:.2f}%")
            logger.info(f"Initial Balance: ${float(TRADING_PARAMS['initial_balance']):,.2f}")
            logger.info(f"Final Balance: ${(float(TRADING_PARAMS['initial_balance']) + total_profit):,.2f}")
            logger.info(f"Profit/Loss: ${total_profit:,.2f}")
            logger.info(f"Profit Percentage: {profit_percentage:.2f}%")
            
            # Calculate additional statistics
            stats = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'initial_balance': float(TRADING_PARAMS['initial_balance']),
                'final_balance': float(TRADING_PARAMS['initial_balance']) + total_profit,
                'total_profit': total_profit,
                'profit_percentage': profit_percentage,
                'profit_factor': (trades_df[trades_df['profit'] > 0]['profit'].sum() / abs(trades_df[trades_df['profit'] < 0]['profit'].sum())) if len(trades_df[trades_df['profit'] < 0]) > 0 else float('inf'),
                'avg_win': trades_df[trades_df['profit'] > 0]['profit'].mean() if len(trades_df[trades_df['profit'] > 0]) > 0 else 0,
                'avg_loss': trades_df[trades_df['profit'] < 0]['profit'].mean() if len(trades_df[trades_df['profit'] < 0]) > 0 else 0,
                'max_drawdown': trades_df['balance'].cummax().max() - trades_df['balance'].min(),
                'max_drawdown_percentage': ((trades_df['balance'].cummax().max() - trades_df['balance'].min()) / trades_df['balance'].cummax().max() * 100) if len(trades_df) > 0 else 0,
                'sharpe_ratio': (trades_df['profit'].mean() / trades_df['profit'].std() * np.sqrt(252)) if len(trades_df) > 1 else 0,
            }
            
            # Save test report
            save_backtest_report(TEST_START, TEST_END, trades_df, stats)
            
            # Log additional statistics
            logger.info("\nAdditional Statistics:")
            logger.info(f"Profit Factor: {stats['profit_factor']}")
            logger.info(f"Average Win: ${stats['avg_win']:.2f}")
            logger.info(f"Average Loss: ${stats['avg_loss']:.2f}")
            logger.info(f"Maximum Drawdown: ${stats['max_drawdown']:.2f} ({stats['max_drawdown_percentage']:.2f}%)")
            logger.info(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
            
        else:
            logger.info("No trades executed during testing period")
            
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

def save_backtest_report(start_date: str, end_date: str, trades_df: pd.DataFrame, stats: dict):
    """Save backtest report to file"""
    # Get project directory (current file directory)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(project_dir, 'backtest_report')
    os.makedirs(report_dir, exist_ok=True)
    
    # Save trades history
    trades_file = os.path.join(report_dir, f'trades_{SYMBOL}_{start_date}_{end_date}.csv')
    trades_df.to_csv(trades_file)
    logger.info(f"\nDetailed trade history saved to {trades_file}")
    
    # Save statistics
    stats_file = os.path.join(report_dir, f'stats_{SYMBOL}_{start_date}_{end_date}.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Trading statistics saved to {stats_file}")

def run_backtest():
    """Run backtesting on historical periods"""
    try:
        # Load the trained model
        model_path = os.path.join(os.path.dirname(MODEL_PATH), 'final_model.zip')
        logger.info(f"Loading model from {model_path}")
        model = PPO.load(model_path)
        
        # Run tests for each backtest period
        for start_date, end_date in BACKTEST_PERIODS:
            logger.info(f"\nBacktesting period: {start_date} to {end_date}")
            
            # Load data for this period
            df = load_data_for_period(start_date, end_date)
            
            # Create environment
            env = ForexEnv(
                df=df,
                initial_balance=float(TRADING_PARAMS['initial_balance']),
                lot_size=float(TRADING_PARAMS['lot_size']),
                max_positions=int(TRADING_PARAMS['max_positions']),
                stop_loss_pips=float(TRADING_PARAMS['stop_loss_pips']),
                take_profit_pips=float(TRADING_PARAMS['take_profit_pips']),
                leverage=float(TRADING_PARAMS['leverage']),
                margin_requirement=float(TRADING_PARAMS['margin_requirement'])
            )
            
            # Test the model
            logger.info("Starting model testing...")
            obs, _ = env.reset()  # Get observation and info
            done = False
            truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs)
                step_result = env.step(action)
                obs = step_result[0]  # observation
                reward = step_result[1]  # reward
                done = step_result[2]  # done
                truncated = step_result[3] if len(step_result) > 3 else False  # truncated (if available)
            
            # Get trading results
            trades_df = pd.DataFrame(env.trades_history)
            
            # Calculate statistics
            if len(trades_df) > 0:
                winning_trades = len(trades_df[trades_df['profit'] > 0])
                total_trades = len(trades_df)
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                # Calculate profit/loss
                total_profit = trades_df['profit'].sum()
                profit_percentage = (total_profit / float(TRADING_PARAMS['initial_balance'])) * 100
                
                # Log results
                logger.info("\nBacktesting Results:")
                logger.info(f"Total Trades: {total_trades}")
                logger.info(f"Winning Trades: {winning_trades}")
                logger.info(f"Win Rate: {win_rate:.2f}%")
                logger.info(f"Initial Balance: ${float(TRADING_PARAMS['initial_balance']):,.2f}")
                logger.info(f"Final Balance: ${(float(TRADING_PARAMS['initial_balance']) + total_profit):,.2f}")
                logger.info(f"Profit/Loss: ${total_profit:,.2f}")
                logger.info(f"Profit Percentage: {profit_percentage:.2f}%")
                
                # Calculate additional statistics
                stats = {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': win_rate,
                    'initial_balance': float(TRADING_PARAMS['initial_balance']),
                    'final_balance': float(TRADING_PARAMS['initial_balance']) + total_profit,
                    'total_profit': total_profit,
                    'profit_percentage': profit_percentage,
                    'profit_factor': (trades_df[trades_df['profit'] > 0]['profit'].sum() / abs(trades_df[trades_df['profit'] < 0]['profit'].sum())) if len(trades_df[trades_df['profit'] < 0]) > 0 else float('inf'),
                    'avg_win': trades_df[trades_df['profit'] > 0]['profit'].mean() if len(trades_df[trades_df['profit'] > 0]) > 0 else 0,
                    'avg_loss': trades_df[trades_df['profit'] < 0]['profit'].mean() if len(trades_df[trades_df['profit'] < 0]) > 0 else 0,
                    'max_drawdown': trades_df['balance'].cummax().max() - trades_df['balance'].min(),
                    'max_drawdown_percentage': ((trades_df['balance'].cummax().max() - trades_df['balance'].min()) / trades_df['balance'].cummax().max() * 100) if len(trades_df) > 0 else 0,
                    'sharpe_ratio': (trades_df['profit'].mean() / trades_df['profit'].std() * np.sqrt(252)) if len(trades_df) > 1 else 0,
                }
                
                # Save backtest report
                save_backtest_report(start_date, end_date, trades_df, stats)
                
                # Log additional statistics
                logger.info("\nAdditional Statistics:")
                logger.info(f"Profit Factor: {stats['profit_factor']}")
                logger.info(f"Average Win: ${stats['avg_win']:.2f}")
                logger.info(f"Average Loss: ${stats['avg_loss']:.2f}")
                logger.info(f"Maximum Drawdown: ${stats['max_drawdown']:.2f} ({stats['max_drawdown_percentage']:.2f}%)")
                logger.info(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
            
            else:
                logger.info("\nNo trades executed during backtesting period")
                
    except Exception as e:
        logger.error(f"Error during backtesting: {str(e)}")
        raise

if __name__ == "__main__":
    # Run normal testing
    test_model()
    
    # Run backtesting
    logger.info("\nStarting backtesting...")
    run_backtest()
