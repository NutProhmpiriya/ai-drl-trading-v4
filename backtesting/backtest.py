import pandas as pd
import numpy as np
from datetime import datetime
import mplfinance as mpf
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from stable_baselines3 import PPO
from utils.data_processor import DataProcessor
from rl_env.forex_env import ForexEnv

class BackTester:
    def __init__(self, model_path: str, initial_balance: float = 10000.0):
        """Initialize backtester with trained model"""
        self.model = PPO.load(model_path)
        self.initial_balance = initial_balance
        self.trades: List[Dict] = []
        self.daily_returns: List[float] = []
        self.equity_curve: List[float] = []
        
    def run_backtest(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Run backtest on historical data"""
        env = ForexEnv(df, self.initial_balance)
        obs, _ = env.reset()
        done = False
        current_balance = self.initial_balance
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            
            # Record trade if position was opened or closed
            if env.current_position or (not env.current_position and len(self.trades) > 0 
                and not self.trades[-1].get('exit_time')):
                self._record_trade(env, df.index[env.current_step])
            
            # Record daily returns and equity
            if len(self.trades) > 0:
                current_balance = self.initial_balance + sum(t.get('pnl', 0) for t in self.trades)
            self.equity_curve.append(current_balance)
            
        return self._generate_results()
    
    def _record_trade(self, env: ForexEnv, current_time: datetime):
        """Record trade details"""
        if env.current_position and (len(self.trades) == 0 or self.trades[-1].get('exit_time')):
            # New trade opened
            self.trades.append({
                'entry_time': current_time,
                'type': env.current_position['type'],
                'entry_price': env.current_position['entry_price'],
                'size': env.current_position['size'],
                'sl': env.current_position['sl'],
                'tp': env.current_position['tp']
            })
        elif not env.current_position and len(self.trades) > 0 and not self.trades[-1].get('exit_time'):
            # Trade closed
            last_trade = self.trades[-1]
            exit_price = env.current_price
            
            if last_trade['type'] == 'buy':
                pnl = (exit_price - last_trade['entry_price']) * last_trade['size']
            else:
                pnl = (last_trade['entry_price'] - exit_price) * last_trade['size']
            
            last_trade.update({
                'exit_time': current_time,
                'exit_price': exit_price,
                'pnl': pnl
            })
    
    def _generate_results(self) -> Tuple[pd.DataFrame, Dict]:
        """Generate backtest results and statistics"""
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
            profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                              trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else float('inf')
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Calculate returns and drawdown
        equity_curve = pd.Series(self.equity_curve)
        returns = equity_curve.pct_change()
        drawdown = (equity_curve - equity_curve.cummax()) / equity_curve.cummax()
        
        stats = {
            'Total Trades': total_trades,
            'Win Rate': win_rate,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Profit Factor': profit_factor,
            'Total Return': (equity_curve.iloc[-1] - self.initial_balance) / self.initial_balance,
            'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0,
            'Max Drawdown': drawdown.min(),
            'Buy Trades': len(trades_df[trades_df['type'] == 'buy']),
            'Sell Trades': len(trades_df[trades_df['type'] == 'sell']),
            'Buy Win Rate': len(trades_df[(trades_df['type'] == 'buy') & (trades_df['pnl'] > 0)]) / 
                          len(trades_df[trades_df['type'] == 'buy']) if len(trades_df[trades_df['type'] == 'buy']) > 0 else 0,
            'Sell Win Rate': len(trades_df[(trades_df['type'] == 'sell') & (trades_df['pnl'] > 0)]) / 
                           len(trades_df[trades_df['type'] == 'sell']) if len(trades_df[trades_df['type'] == 'sell']) > 0 else 0
        }
        
        return trades_df, stats
    
    def plot_results(self, df: pd.DataFrame, trades_df: pd.DataFrame, save_path: str = None):
        """Plot backtest results with candlestick chart and indicators"""
        # Prepare candlestick data
        df_plot = df.copy()
        df_plot.index.name = 'Date'
        
        # Prepare trade markers
        buy_signals = []
        sell_signals = []
        tp_markers = []
        sl_markers = []
        
        for _, trade in trades_df.iterrows():
            if trade['type'] == 'buy':
                buy_signals.append((trade['entry_time'], trade['entry_price']))
                if trade.get('exit_time'):
                    if trade['pnl'] > 0:
                        tp_markers.append((trade['exit_time'], trade['exit_price']))
                    else:
                        sl_markers.append((trade['exit_time'], trade['exit_price']))
            else:
                sell_signals.append((trade['entry_time'], trade['entry_price']))
                if trade.get('exit_time'):
                    if trade['pnl'] > 0:
                        tp_markers.append((trade['exit_time'], trade['exit_price']))
                    else:
                        sl_markers.append((trade['exit_time'], trade['exit_price']))
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot candlestick chart
        mpf.plot(df_plot, type='candle', style='charles',
                title=f'Backtest Results\nInitial Balance: ${self.initial_balance:,.2f}',
                ax=ax1, volume=False)
        
        # Add EMAs
        ax1.plot(df_plot.index, df_plot['EMA9'], label='EMA9', alpha=0.7)
        ax1.plot(df_plot.index, df_plot['EMA21'], label='EMA21', alpha=0.7)
        ax1.plot(df_plot.index, df_plot['EMA50'], label='EMA50', alpha=0.7)
        
        # Add trade markers
        for time, price in buy_signals:
            ax1.plot(time, price, '^', color='g', markersize=10, label='Buy' if 'Buy' not in ax1.get_legend_handles_labels()[1] else '')
        for time, price in sell_signals:
            ax1.plot(time, price, 'v', color='r', markersize=10, label='Sell' if 'Sell' not in ax1.get_legend_handles_labels()[1] else '')
        for time, price in tp_markers:
            ax1.plot(time, price, 's', color='blue', markersize=8, label='TP' if 'TP' not in ax1.get_legend_handles_labels()[1] else '')
        for time, price in sl_markers:
            ax1.plot(time, price, 's', color='black', markersize=8, label='SL' if 'SL' not in ax1.get_legend_handles_labels()[1] else '')
        
        ax1.legend()
        
        # Plot RSI
        df_plot['RSI'].plot(ax=ax2, label='RSI')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.legend()
        
        # Plot equity curve
        pd.Series(self.equity_curve, index=df_plot.index).plot(ax=ax3, label='Equity')
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
