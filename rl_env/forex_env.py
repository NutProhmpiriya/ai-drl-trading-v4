import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Tuple, Any

class ForexEnv(gym.Env):
    """Custom Forex Trading Environment that follows gymnasium interface"""
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0,
                 lot_size: float = 0.1, max_positions: int = 1,
                 stop_loss_pips: float = 30, take_profit_pips: float = 60):
        super().__init__()

        # Convert column names to lowercase for consistency
        self.df = df.copy()
        self.df.columns = self.df.columns.str.lower()
        
        # Add additional features
        self.df['spread'] = self.df['ask'] - self.df['bid']
        self.df['volume_ma'] = self.df['tick_volume'].rolling(window=20).mean()
        self.df['volatility'] = self.df['close'].pct_change().rolling(window=20).std()
        
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.max_positions = max_positions
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        
        self.current_step = 0
        self.current_position = None
        self.daily_loss = 0
        self.last_trade_day = None
        self.balance = initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # Observation space: [EMAs, RSI, ATR, OBV, Volume, Spread, Volatility, Position, Balance, Daily PnL]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(13,),  # Added Volume, Spread, Volatility
            dtype=np.float32
        )
    
    def reset(self, *, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_position = None
        self.balance = self.initial_balance
        self.daily_loss = 0
        self.last_trade_day = None
        self.total_trades = 0
        self.winning_trades = 0
        
        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'position': None,
            'total_trades': 0,
            'winning_trades': 0
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment"""
        self.current_step += 1
        
        # Get current price data
        current_price = self.df['close'].iloc[self.current_step]
        
        # Initialize step info
        info = {
            'trade_executed': False,
            'trade_type': None,
            'execution_price': None,
            'trade_profit': 0,
            'balance': self.balance,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades
        }
        
        # Check for stop loss or take profit if position exists
        if self.current_position is not None:
            entry_price = self.current_position['entry_price']
            position_type = self.current_position['type']
            
            if position_type == 'buy':
                profit_pips = (current_price - entry_price) * 100
            else:  # sell
                profit_pips = (entry_price - current_price) * 100
            
            # Check stop loss
            if profit_pips <= -self.stop_loss_pips:
                reward = -self.stop_loss_pips
                self.balance += reward * self.lot_size * 100  # Convert pips to cash
                self.current_position = None
                self.total_trades += 1
                info['trade_executed'] = True
                info['trade_type'] = 'stop_loss'
                info['execution_price'] = current_price
                info['trade_profit'] = reward * self.lot_size * 100
                
                return self._get_observation(), reward, False, False, info
            
            # Check take profit
            if profit_pips >= self.take_profit_pips:
                reward = self.take_profit_pips
                self.balance += reward * self.lot_size * 100  # Convert pips to cash
                self.current_position = None
                self.total_trades += 1
                self.winning_trades += 1
                info['trade_executed'] = True
                info['trade_type'] = 'take_profit'
                info['execution_price'] = current_price
                info['trade_profit'] = reward * self.lot_size * 100
                
                return self._get_observation(), reward, False, False, info
        
        # Process the action
        if action != 0 and self.current_position is None:  # Open new position
            if action == 1:  # Buy
                self.current_position = {
                    'type': 'buy',
                    'entry_price': current_price,
                    'entry_step': self.current_step
                }
                info['trade_executed'] = True
                info['trade_type'] = 'buy'
                info['execution_price'] = current_price
            else:  # Sell
                self.current_position = {
                    'type': 'sell',
                    'entry_price': current_price,
                    'entry_step': self.current_step
                }
                info['trade_executed'] = True
                info['trade_type'] = 'sell'
                info['execution_price'] = current_price
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 2 or self.balance <= 0
        
        return self._get_observation(), reward, done, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        current_price = self.df.iloc[self.current_step]
        
        # Technical indicators
        ema_fast = current_price['ema9']
        ema_medium = current_price['ema21']
        ema_slow = current_price['ema50']
        rsi = current_price['rsi']
        atr = current_price['atr']
        obv = current_price['obv']
        
        # New features
        volume = current_price['tick_volume'] / self.df['tick_volume'].max()  # Normalized volume
        volume_ma = current_price['volume_ma'] / self.df['volume_ma'].max()  # Normalized volume MA
        spread = current_price['spread'] / current_price['close']  # Normalized spread
        volatility = current_price['volatility']
        
        # Position and account info
        position = 1 if self.current_position is not None and self.current_position['type'] == 'buy' else (-1 if self.current_position is not None and self.current_position['type'] == 'sell' else 0)
        balance = self.balance / self.initial_balance  # Normalized balance
        daily_pnl = self.daily_loss / self.initial_balance  # Normalized daily PnL
        
        return np.array([
            ema_fast, ema_medium, ema_slow,
            rsi, atr, obv,
            volume, volume_ma, spread, volatility,
            position, balance, daily_pnl
        ], dtype=np.float32)
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on action and state"""
        reward = 0
        
        # Base reward from profit/loss
        if self.current_position:
            profit = self._get_unrealized_profit()
            reward = profit / self.initial_balance  # Normalize profit
            
            # Penalize for holding losing positions
            if profit < 0:
                reward *= 1.5  # Increase penalty for losses
            
            # Reward for taking profits at good times
            if profit > 0 and self.df.iloc[self.current_step]['rsi'] > 70:
                reward *= 1.2  # Bonus for selling in overbought
            elif profit > 0 and self.df.iloc[self.current_step]['rsi'] < 30:
                reward *= 1.2  # Bonus for buying in oversold
        
        # Penalize for trading in high spread conditions
        current_spread = self.df.iloc[self.current_step]['spread']
        if action != 0 and current_spread > self.df['spread'].mean() * 1.5:
            reward *= 0.8
        
        # Reward for trading with the trend
        if action == 1:  # Buy
            if self.df.iloc[self.current_step]['ema9'] > self.df.iloc[self.current_step]['ema21']:
                reward *= 1.1
        elif action == 2:  # Sell
            if self.df.iloc[self.current_step]['ema9'] < self.df.iloc[self.current_step]['ema21']:
                reward *= 1.1
        
        # Penalize for overtrading
        if action != 0 and self.total_trades > 20:  # Assuming 20 trades per day is excessive
            reward *= 0.9
        
        return reward
    
    def _get_unrealized_profit(self) -> float:
        """Get unrealized profit"""
        if self.current_position is None:
            return 0
        
        entry_price = self.current_position['entry_price']
        current_price = self.df['close'].iloc[self.current_step]
        
        if self.current_position['type'] == 'buy':
            return (current_price - entry_price) * 100
        else:
            return (entry_price - current_price) * 100
    
    def render(self, mode: str = 'human'):
        """Render the environment to the screen"""
        pass
    
    def close(self):
        """Close the environment"""
        pass
