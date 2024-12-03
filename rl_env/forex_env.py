import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Tuple

class ForexEnv(gym.Env):
    """Custom Forex Trading Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        super(ForexEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        self.current_position = None
        self.daily_loss = 0
        self.last_trade_day = None
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # Observation space: [EMAs, RSI, ATR, OBV, Position, Balance, Daily PnL]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,),  # 3 EMAs + RSI + ATR + OBV + Position + Balance + Daily PnL + Current Price
            dtype=np.float32
        )

    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on action and position"""
        reward = 0
        
        # Penalize if daily loss limit is hit
        if self.daily_loss <= -0.01 * self.initial_balance:
            reward -= 2.0  # Strong penalty for hitting daily loss limit
            
        if self.current_position:
            # Calculate PnL
            if self.current_position['type'] == 'buy':
                reward = (self.current_price - self.current_position['entry_price']) * self.current_position['size']
            else:
                reward = (self.current_position['entry_price'] - self.current_price) * self.current_position['size']
                
            # Scale reward based on remaining daily loss capacity
            loss_ratio = abs(self.daily_loss) / (0.01 * self.initial_balance)
            if loss_ratio > 0.5:  # If we've lost more than half of daily limit
                reward *= (1 - loss_ratio)  # Reduce reward as we get closer to daily loss limit
                
        return reward

    def _get_observation(self) -> np.array:
        """Get current state observation"""
        current_data = self.df.iloc[self.current_step]
        
        obs = np.array([
            current_data['EMA9'],
            current_data['EMA21'],
            current_data['EMA50'],
            current_data['RSI'],
            current_data['ATR'],
            current_data['OBV'],
            1.0 if self.current_position and self.current_position['type'] == 'buy' else 
            (-1.0 if self.current_position and self.current_position['type'] == 'sell' else 0.0),
            self.balance,
            self.daily_loss,
            current_data['Close']
        ], dtype=np.float32)
        
        return obs

    def _update_position(self):
        """Update current position based on SL/TP and trailing stop"""
        if not self.current_position:
            return

        current_data = self.df.iloc[self.current_step]
        
        # Check SL/TP
        if self.current_position['type'] == 'buy':
            # Update trailing stop if price moves in favorable direction
            if current_data['Close'] > self.current_position['entry_price']:
                new_sl = current_data['Close'] - current_data['ATR']
                if new_sl > self.current_position['sl']:
                    self.current_position['sl'] = new_sl
            
            # Check if SL or TP hit
            if current_data['Low'] <= self.current_position['sl']:
                self._close_position(self.current_position['sl'])
            elif current_data['High'] >= self.current_position['tp']:
                self._close_position(self.current_position['tp'])
        
        else:  # sell position
            # Update trailing stop if price moves in favorable direction
            if current_data['Close'] < self.current_position['entry_price']:
                new_sl = current_data['Close'] + current_data['ATR']
                if new_sl < self.current_position['sl']:
                    self.current_position['sl'] = new_sl
            
            # Check if SL or TP hit
            if current_data['High'] >= self.current_position['sl']:
                self._close_position(self.current_position['sl'])
            elif current_data['Low'] <= self.current_position['tp']:
                self._close_position(self.current_position['tp'])

    def _close_position(self, price: float):
        """Close current position and update balance"""
        if not self.current_position:
            return

        pnl = 0
        if self.current_position['type'] == 'buy':
            pnl = (price - self.current_position['entry_price']) * self.current_position['size']
        else:
            pnl = (self.current_position['entry_price'] - price) * self.current_position['size']

        self.balance += pnl
        self.daily_loss += min(pnl, 0)
        self.current_position = None

    def step(self, action: int) -> Tuple[np.array, float, bool, bool, Dict]:
        """Execute one time step within the environment"""
        self.current_step += 1
        if self.current_step >= len(self.df):
            return self._get_observation(), 0, True, False, {}

        current_data = self.df.iloc[self.current_step]
        self.current_price = current_data['Close']

        # Check if it's a new trading day
        current_day = pd.Timestamp(current_data.name).date()
        if self.last_trade_day != current_day:
            self.daily_loss = 0
            self.last_trade_day = current_day

        # Update existing position
        self._update_position()

        # Process new action
        reward = 0
        if action != 0 and not self.current_position and self.daily_loss > -0.01 * self.initial_balance:
            position_size = 0.01  # Calculate based on risk
            entry_price = current_data['Close']
            atr = current_data['ATR']

            if action == 1:  # Buy
                sl = entry_price - atr
                tp = entry_price + (1.5 * atr)
                self.current_position = {
                    'type': 'buy',
                    'entry_price': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'size': position_size
                }
            elif action == 2:  # Sell
                sl = entry_price + atr
                tp = entry_price - (1.5 * atr)
                self.current_position = {
                    'type': 'sell',
                    'entry_price': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'size': position_size
                }

        reward = self._calculate_reward(action)
        done = self.current_step >= len(self.df) - 1 or self.daily_loss <= -0.01 * self.initial_balance

        return self._get_observation(), reward, done, False, {}

    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.current_position = None
        self.daily_loss = 0
        self.last_trade_day = None
        return self._get_observation(), {}
