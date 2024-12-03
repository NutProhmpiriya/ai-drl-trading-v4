import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from config import TRADING_PARAMS, BACKTEST_PARAMS, PATHS
from rl_env.forex_env import ForexEnv

class BackTester:
    def __init__(self, model_path: str):
        """Initialize backtester with model and parameters"""
        self.model = PPO.load(model_path)
        self.initial_balance = BACKTEST_PARAMS['initial_balance']
        self.commission = BACKTEST_PARAMS['commission']
        self.slippage = BACKTEST_PARAMS['slippage']
    
    def run_backtest(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Run backtest on historical data"""
        
        # Create environment
        env = ForexEnv(
            df=df,
            initial_balance=self.initial_balance,
            lot_size=TRADING_PARAMS['lot_size'],
            max_positions=TRADING_PARAMS['max_positions'],
            stop_loss_pips=TRADING_PARAMS['stop_loss_pips'],
            take_profit_pips=TRADING_PARAMS['take_profit_pips']
        )
        
        # Run simulation
        obs = env.reset()
        done = False
        trades = []
        
        while not done:
            action, _states = self.model.predict(obs)
            obs, reward, done, info = env.step(action)
            
            if info.get('trade_executed'):
                trade_info = {
                    'timestamp': df.index[env.current_step],
                    'action': 'buy' if action == 1 else 'sell',
                    'price': info['execution_price'],
                    'profit': info['trade_profit'],
                    'balance': env.balance
                }
                trades.append(trade_info)
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate statistics
        stats = self._calculate_statistics(trades_df)
        
        return trades_df, stats
    
    def _calculate_statistics(self, trades_df: pd.DataFrame) -> dict:
        """Calculate backtest statistics"""
        
        if trades_df.empty:
            return {
                'Total Trades': 0,
                'Win Rate': 0,
                'Total Profit': 0,
                'Average Profit': 0,
                'Max Drawdown': 0,
                'Sharpe Ratio': 0
            }
        
        # Basic statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit statistics
        total_profit = trades_df['profit'].sum()
        avg_profit = trades_df['profit'].mean()
        
        # Calculate drawdown
        peak = trades_df['balance'].expanding().max()
        drawdown = (trades_df['balance'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        returns = trades_df['profit'] / self.initial_balance
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
        
        return {
            'Total Trades': total_trades,
            'Win Rate': win_rate,
            'Total Profit': total_profit,
            'Average Profit': avg_profit,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio
        }
    
    def plot_results(self, df: pd.DataFrame, trades_df: pd.DataFrame, save_path: str = None):
        """Plot backtest results"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot price
        ax1.plot(df.index, df['Close'], label='Price', alpha=0.7)
        
        # Plot trades
        if not trades_df.empty:
            buy_points = trades_df[trades_df['action'] == 'buy']
            sell_points = trades_df[trades_df['action'] == 'sell']
            
            ax1.scatter(buy_points['timestamp'], buy_points['price'],
                       marker='^', color='green', label='Buy', alpha=0.7)
            ax1.scatter(sell_points['timestamp'], sell_points['price'],
                       marker='v', color='red', label='Sell', alpha=0.7)
        
        ax1.set_title('Price and Trades')
        ax1.legend()
        
        # Plot balance
        if not trades_df.empty:
            ax2.plot(trades_df['timestamp'], trades_df['balance'], label='Balance')
        ax2.set_title('Account Balance')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
