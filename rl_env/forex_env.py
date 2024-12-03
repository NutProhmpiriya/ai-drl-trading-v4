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

        self.df = df
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
        
        # Observation space: [EMAs, RSI, ATR, OBV, Position, Balance, Daily PnL]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,),  # 3 EMAs + RSI + ATR + OBV + Position + Balance + Daily PnL + Current Price
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
        
        obs = self._next_observation()
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
        current_price = self.df['Close'].iloc[self.current_step]
        
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
                
            # Check take profit
            elif profit_pips >= self.take_profit_pips:
                reward = self.take_profit_pips
                self.balance += reward * self.lot_size * 100  # Convert pips to cash
                self.current_position = None
                self.total_trades += 1
                self.winning_trades += 1
                info['trade_executed'] = True
                info['trade_type'] = 'take_profit'
                info['execution_price'] = current_price
                info['trade_profit'] = reward * self.lot_size * 100
        
        # Process new action if no position or position was just closed
        if self.current_position is None and action != 0:  # 0 is hold
            if action == 1:  # Buy
                self.current_position = {
                    'type': 'buy',
                    'entry_price': current_price
                }
                info['trade_executed'] = True
                info['trade_type'] = 'buy'
                info['execution_price'] = current_price
            elif action == 2:  # Sell
                self.current_position = {
                    'type': 'sell',
                    'entry_price': current_price
                }
                info['trade_executed'] = True
                info['trade_type'] = 'sell'
                info['execution_price'] = current_price
        
        # Calculate reward
        reward = 0
        if self.current_position is not None:
            if self.current_position['type'] == 'buy':
                reward = (current_price - self.current_position['entry_price']) * 100
            else:  # sell
                reward = (self.current_position['entry_price'] - current_price) * 100
        
        # Update daily loss
        current_day = self.df.index[self.current_step].date()
        if self.last_trade_day != current_day:
            self.daily_loss = 0
            self.last_trade_day = current_day
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        truncated = False  # For gymnasium v1.0.0
        
        # Update info with final state
        info.update({
            'balance': self.balance,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades
        })
        
        # Get next observation
        obs = self._next_observation()
        
        return obs, reward, done, truncated, info
    
    def _next_observation(self) -> np.ndarray:
        """Get the next observation"""
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Normalize price-based features
        ema9 = self.df['EMA9'].iloc[self.current_step] / current_price - 1
        ema21 = self.df['EMA21'].iloc[self.current_step] / current_price - 1
        ema50 = self.df['EMA50'].iloc[self.current_step] / current_price - 1
        
        # Get other indicators
        rsi = self.df['RSI'].iloc[self.current_step] / 100  # Normalize to 0-1
        atr = self.df['ATR'].iloc[self.current_step] / current_price  # Normalize by price
        obv = self.df['OBV'].iloc[self.current_step]
        
        # Normalize OBV using recent window
        obv_window = self.df['OBV'].iloc[max(0, self.current_step-20):self.current_step+1]
        obv_min = obv_window.min()
        obv_max = obv_window.max()
        if obv_max - obv_min != 0:
            obv_norm = (obv - obv_min) / (obv_max - obv_min)
        else:
            obv_norm = 0
        
        # Position encoding: -1 for sell, 0 for no position, 1 for buy
        position = 0
        if self.current_position is not None:
            position = 1 if self.current_position['type'] == 'buy' else -1
        
        # Normalize balance change from initial
        balance_change = (self.balance - self.initial_balance) / self.initial_balance
        
        # Normalize daily PnL
        daily_pnl = self.daily_loss / self.initial_balance
        
        # Current price change
        price_change = 0
        if self.current_step > 0:
            prev_price = self.df['Close'].iloc[self.current_step-1]
            price_change = (current_price - prev_price) / prev_price
        
        return np.array([
            ema9, ema21, ema50,
            rsi, atr, obv_norm,
            position,
            balance_change,
            daily_pnl,
            price_change
        ], dtype=np.float32)
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment to the screen"""
        pass
    
    def close(self) -> None:
        """Close the environment"""
        pass
