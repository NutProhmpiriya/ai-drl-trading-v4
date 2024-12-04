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
                 stop_loss_pips: float = 30, take_profit_pips: float = 60,
                 leverage: float = 100, margin_requirement: float = 0.01,
                 spread: float = 0.0002):
        super().__init__()

        # Convert column names to lowercase for consistency
        self.df = df.copy()
        self.df.columns = self.df.columns.str.lower()
        
        # Calculate bid/ask from close price
        self.df['ask'] = self.df['close'] + (spread / 2)
        self.df['bid'] = self.df['close'] - (spread / 2)
        
        # Add additional features
        self.df['spread'] = self.df['ask'] - self.df['bid']
        self.df['volume_ma'] = self.df['tick_volume'].rolling(window=20).mean()
        
        # Drop any NaN values
        self.df.dropna(inplace=True)
        
        # Store parameters
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.max_positions = max_positions
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.leverage = leverage
        self.margin_requirement = margin_requirement
        
        # Initialize state
        self.current_step = 0
        self.balance = initial_balance
        self.positions = []
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_loss = 0
        self.last_trade_day = None
        self.used_margin = 0
        self.trades_history = []  # Initialize trades history list
        self.equity_curve = []  # Track equity curve
        
        # Calculate observation space size
        self.obs_shape = (
            4 +  # Price features (OHLC)
            5 +  # Technical indicators (RSI, ATR, EMA9, EMA21, EMA50)
            2 +  # Market features (Volume, Spread)
            5    # Position features (Position, Profit, Balance, Free Margin, Position Utilization)
        )
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_shape,),
            dtype=np.float32
        )
    
    def reset(self, *, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = []
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_loss = 0
        self.last_trade_day = None
        self.used_margin = 0
        self.trades_history = []  # Reset trades history list
        self.equity_curve = []  # Reset equity curve
        
        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'position': None,
            'total_trades': 0,
            'winning_trades': 0,
            'used_margin': self.used_margin,
            'free_margin': self.balance - self.used_margin
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment"""
        self.current_step += 1
        
        # Get current price data
        current_price = self.df['close'].iloc[self.current_step]
        current_date = pd.to_datetime(self.df.index[self.current_step]).date()
        
        # Reset daily loss if new day
        if self.last_trade_day is None or current_date != self.last_trade_day:
            self.daily_loss = 0
            self.last_trade_day = current_date
        
        # Initialize step info
        info = {
            'trade_executed': False,
            'trade_type': None,
            'execution_price': None,
            'trade_profit': 0,
            'balance': self.balance,
            'used_margin': self.used_margin,
            'free_margin': self.balance - self.used_margin,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'daily_loss': self.daily_loss
        }
        
        # Calculate required margin for new position
        position_size = self.lot_size * 100000  # Convert lot size to units
        required_margin = position_size * self.margin_requirement
        
        # Check if we have enough free margin for new position
        if action != 0 and not self.positions:
            if required_margin > (self.balance - self.used_margin):
                # Not enough margin, force hold
                action = 0
                info['trade_type'] = 'margin_call'
        
        # Check if daily loss limit is reached
        if self.daily_loss <= -self.initial_balance * 0.01:  # 1% max daily loss
            # Close any open position
            if self.positions:
                profit_pips = (current_price - self.positions[0]['entry_price']) * 100
                self.balance += profit_pips * self.lot_size * 100
                trade_data = {
                    'timestamp': self.df.index[self.current_step],
                    'type': 'daily_loss_limit',
                    'entry_price': self.positions[0]['entry_price'],
                    'execution_price': current_price,
                    'profit': profit_pips * self.lot_size * 100,
                    'balance': self.balance,
                    'position_size': self.lot_size,
                    'duration': self.current_step - self.positions[0]['entry_step'],
                    'market_price': self.df['close'].iloc[self.current_step],
                    'equity': self.balance + self._calculate_unrealized_pnl(),
                    'drawdown': self._calculate_drawdown(),
                    'drawdown_pct': self._calculate_drawdown_percentage()
                }
                self.trades_history.append(trade_data)
                self.positions = []
                self.total_trades += 1
                if profit_pips > 0:
                    self.winning_trades += 1
                
                info['trade_executed'] = True
                info['trade_type'] = 'daily_loss_limit'
                info['execution_price'] = current_price
                info['trade_profit'] = profit_pips * self.lot_size * 100
            
            return self._get_observation(), 0, True, False, info
        
        # Check for stop loss or take profit if position exists
        if self.positions:
            entry_price = self.positions[0]['entry_price']
            
            profit_pips = (current_price - entry_price) * 100
            
            # Check stop loss
            if profit_pips <= -self.stop_loss_pips:
                reward = -self.stop_loss_pips
                self.balance += reward * self.lot_size * 100  # Convert pips to cash
                trade_data = {
                    'timestamp': self.df.index[self.current_step],
                    'type': 'stop_loss',
                    'entry_price': entry_price,
                    'execution_price': current_price,
                    'profit': reward * self.lot_size * 100,
                    'balance': self.balance,
                    'position_size': self.lot_size,
                    'duration': self.current_step - self.positions[0]['entry_step'],
                    'market_price': self.df['close'].iloc[self.current_step],
                    'equity': self.balance + self._calculate_unrealized_pnl(),
                    'drawdown': self._calculate_drawdown(),
                    'drawdown_pct': self._calculate_drawdown_percentage()
                }
                self.trades_history.append(trade_data)
                self.positions = []
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
                trade_data = {
                    'timestamp': self.df.index[self.current_step],
                    'type': 'take_profit',
                    'entry_price': entry_price,
                    'execution_price': current_price,
                    'profit': reward * self.lot_size * 100,
                    'balance': self.balance,
                    'position_size': self.lot_size,
                    'duration': self.current_step - self.positions[0]['entry_step'],
                    'market_price': self.df['close'].iloc[self.current_step],
                    'equity': self.balance + self._calculate_unrealized_pnl(),
                    'drawdown': self._calculate_drawdown(),
                    'drawdown_pct': self._calculate_drawdown_percentage()
                }
                self.trades_history.append(trade_data)
                self.positions = []
                self.total_trades += 1
                self.winning_trades += 1
                info['trade_executed'] = True
                info['trade_type'] = 'take_profit'
                info['execution_price'] = current_price
                info['trade_profit'] = reward * self.lot_size * 100
                
                return self._get_observation(), reward, False, False, info
        
        # Process the action
        if action != 0 and not self.positions:  # Open new position
            if action == 1:  # Buy
                self.positions.append({
                    'type': 'buy',
                    'entry_price': current_price,
                    'entry_step': self.current_step
                })
                self.used_margin += required_margin
                info['trade_executed'] = True
                info['trade_type'] = 'buy'
                info['execution_price'] = current_price
            else:  # Sell
                self.positions.append({
                    'type': 'sell',
                    'entry_price': current_price,
                    'entry_step': self.current_step
                })
                self.used_margin += required_margin
                info['trade_executed'] = True
                info['trade_type'] = 'sell'
                info['execution_price'] = current_price
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 2 or self.balance <= 0
        
        return self._get_observation(), reward, done, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (state)"""
        current_step_data = self.df.iloc[self.current_step]
        
        # Price features
        price_features = [
            current_step_data['close'],
            current_step_data['high'],
            current_step_data['low'],
            current_step_data['open']
        ]
        
        # Technical indicators
        tech_features = [
            current_step_data['rsi'],
            current_step_data['atr'],
            current_step_data['ema9'],
            current_step_data['ema21'],
            current_step_data['ema50']
        ]
        
        # Volume and spread
        market_features = [
            current_step_data['tick_volume'] / self.df['tick_volume'].mean(),  # Normalized volume
            current_step_data['spread'] / self.df['spread'].mean()  # Normalized spread
        ]
        
        # Position features
        position_features = [
            1 if self.positions and self.positions[0]['type'] == 'buy' else (-1 if self.positions and self.positions[0]['type'] == 'sell' else 0),
            self._get_unrealized_profit(),
            self.balance / self.initial_balance,  # Normalized balance
            (self.balance - self.used_margin) / self.initial_balance,  # Normalized free margin
            self.total_trades / self.max_positions  # Position utilization
        ]
        
        return np.array(price_features + tech_features + market_features + position_features)
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on action and state"""
        reward = 0
        
        # Base reward from profit/loss
        if self.positions:
            profit = self._get_unrealized_profit()
            reward = profit / self.initial_balance  # Normalize profit
            
            # Update daily loss
            self.daily_loss += profit * self.lot_size * 100
            
            # Penalize for holding losing positions
            if profit < 0:
                reward *= 1.5  # Increase penalty for losses
            
            # Extra penalty if approaching daily loss limit
            if self.daily_loss < -self.initial_balance * 0.01 * 0.7:  # Warning at 70% of max daily loss
                reward *= 1.3
            
            # Extra penalty if approaching margin call (less than 30% free margin)
            free_margin_ratio = (self.balance - self.used_margin) / self.balance
            if free_margin_ratio < 0.3:
                reward *= 1.3
        
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
        if not self.positions:
            return 0
        
        entry_price = self.positions[0]['entry_price']
        current_price = self.df['close'].iloc[self.current_step]
        
        if self.positions[0]['type'] == 'buy':
            return (current_price - entry_price) * 100
        else:
            return (entry_price - current_price) * 100
    
    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized profit and loss"""
        if not self.positions:
            return 0
        
        entry_price = self.positions[0]['entry_price']
        current_price = self.df['close'].iloc[self.current_step]
        
        if self.positions[0]['type'] == 'buy':
            return (current_price - entry_price) * self.lot_size * 100
        else:
            return (entry_price - current_price) * self.lot_size * 100
    
    def _calculate_drawdown(self) -> float:
        """Calculate drawdown"""
        if not self.equity_curve:
            return 0
        
        max_equity = max(self.equity_curve)
        current_equity = self.equity_curve[-1]
        
        return max_equity - current_equity
    
    def _calculate_drawdown_percentage(self) -> float:
        """Calculate drawdown percentage"""
        if not self.equity_curve:
            return 0
        
        max_equity = max(self.equity_curve)
        current_equity = self.equity_curve[-1]
        
        return ((max_equity - current_equity) / max_equity) * 100
    
    def render(self, mode: str = 'human'):
        """Render the environment to the screen"""
        pass
    
    def close(self):
        """Close the environment"""
        pass
