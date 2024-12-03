import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
from typing import Dict, Optional
from stable_baselines3 import PPO
from utils.data_processor import DataProcessor

class LiveTrader:
    def __init__(self, model_path: str, symbol: str = "USDJPY", timeframe: str = "5",
                 risk_per_trade: float = 0.01, max_daily_loss: float = 0.01):
        """Initialize live trader"""
        # Trading parameters
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss = max_daily_loss
        
        # Initialize MT5 connection
        if not mt5.initialize():
            print("MT5 initialization failed")
            mt5.shutdown()
            raise Exception("MT5 initialization failed")
            
        # Subscribe to market data
        if not mt5.market_book_add(self.symbol):
            print(f"Failed to subscribe to market data for {self.symbol}")
            mt5.shutdown()
            raise Exception(f"Market data subscription failed for {self.symbol}")
            
        # Initialize data buffer for indicators
        self.price_buffer = []
        self.buffer_size = 100  # Store enough data for indicators
        
        # Load the trained model
        self.model = PPO.load(model_path)
        self.data_processor = DataProcessor()
        
        # Trading state
        self.current_position: Optional[Dict] = None
        self.daily_loss = 0
        self.last_trade_day = None
        self.last_update_time = None
        
        # Get account info
        self.account_info = mt5.account_info()
        if not self.account_info:
            raise Exception("Failed to get account info")
            
        print(f"Connected to MT5: {self.account_info.server}")
        print(f"Balance: {self.account_info.balance}")
        print(f"Equity: {self.account_info.equity}")
        
    def _update_price_buffer(self) -> bool:
        """Update price buffer with latest market data"""
        book = mt5.market_book_get(self.symbol)
        if not book:
            return False
            
        # Calculate OHLCV from order book
        best_ask = min(item.price for item in book if item.type == mt5.BOOK_TYPE_SELL)
        best_bid = max(item.price for item in book if item.type == mt5.BOOK_TYPE_BUY)
        current_price = (best_ask + best_bid) / 2
        current_time = datetime.now()
        
        # Initialize or update timeframe candle
        if not self.last_update_time:
            self.last_update_time = current_time
            self.current_candle = {
                'open': current_price,
                'high': current_price,
                'low': current_price,
                'close': current_price,
                'volume': 0
            }
        else:
            # Update current candle
            self.current_candle['high'] = max(self.current_candle['high'], current_price)
            self.current_candle['low'] = min(self.current_candle['low'], current_price)
            self.current_candle['close'] = current_price
            self.current_candle['volume'] += sum(item.volume for item in book)
            
            # Check if it's time for a new candle
            timeframe_minutes = int(self.timeframe)
            if (current_time - self.last_update_time).total_seconds() >= timeframe_minutes * 60:
                # Add completed candle to buffer
                self.price_buffer.append(self.current_candle)
                if len(self.price_buffer) > self.buffer_size:
                    self.price_buffer.pop(0)
                    
                # Start new candle
                self.current_candle = {
                    'open': current_price,
                    'high': current_price,
                    'low': current_price,
                    'close': current_price,
                    'volume': 0
                }
                self.last_update_time = current_time
                
        return True
        
    def _calculate_indicators(self) -> Optional[pd.DataFrame]:
        """Calculate indicators from price buffer"""
        if len(self.price_buffer) < 50:  # Need enough data for indicators
            return None
            
        df = pd.DataFrame(self.price_buffer)
        
        # Calculate indicators
        df['EMA9'] = self.data_processor._calculate_ema(df['close'], 9)
        df['EMA21'] = self.data_processor._calculate_ema(df['close'], 21)
        df['EMA50'] = self.data_processor._calculate_ema(df['close'], 50)
        df['RSI'] = self.data_processor._calculate_rsi(df['close'], 14)
        df['ATR'] = self.data_processor._calculate_atr(df['high'], df['low'], df['close'], 14)
        df['OBV'] = self.data_processor._calculate_obv(df['close'], df['volume'])
        
        return df
    
    def calculate_position_size(self, stop_loss_pips: float) -> float:
        """Calculate position size based on risk per trade"""
        account_balance = mt5.account_info().balance
        risk_amount = account_balance * self.risk_per_trade
        
        # Get symbol info
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            raise Exception(f"Failed to get symbol info for {self.symbol}")
        
        pip_value = symbol_info.trade_tick_value * (stop_loss_pips / symbol_info.trade_tick_size)
        position_size = risk_amount / pip_value
        
        # Round to nearest 0.01 lot
        position_size = round(position_size / 0.01) * 0.01
        
        return max(min(position_size, symbol_info.volume_max), symbol_info.volume_min)
    
    def place_order(self, order_type: str, entry_price: float, sl_price: float, tp_price: float) -> bool:
        """Place a market order with SL and TP"""
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            print(f"Failed to get symbol info for {self.symbol}")
            return False
            
        # Calculate position size
        stop_loss_pips = abs(entry_price - sl_price) / symbol_info.point
        position_size = self.calculate_position_size(stop_loss_pips)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position_size,
            "type": mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": 234000,
            "comment": "DRL Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: {result.comment}")
            return False
            
        self.current_position = {
            "ticket": result.order,
            "type": order_type,
            "entry_price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "size": position_size
        }
        return True
    
    def modify_sl_tp(self, ticket: int, sl_price: float, tp_price: float) -> bool:
        """Modify SL/TP of an existing position"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "sl": sl_price,
            "tp": tp_price,
            "position": ticket
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Modify order failed: {result.comment}")
            return False
        
        if self.current_position:
            self.current_position.update({"sl": sl_price, "tp": tp_price})
        return True
    
    def check_position(self) -> None:
        """Check current position status and update trailing stop"""
        if not self.current_position:
            return
            
        position = mt5.positions_get(ticket=self.current_position["ticket"])
        if not position:  # Position closed
            self._handle_position_closed()
            return
            
        position = position[0]
        current_price = mt5.symbol_info_tick(self.symbol).ask if self.current_position["type"] == "buy" else mt5.symbol_info_tick(self.symbol).bid
        
        # Update trailing stop
        if self.current_position["type"] == "buy" and current_price > position.price_open:
            new_sl = current_price - (position.price_open - position.sl)
            if new_sl > position.sl:
                self.modify_sl_tp(position.ticket, new_sl, position.tp)
        elif self.current_position["type"] == "sell" and current_price < position.price_open:
            new_sl = current_price + (position.sl - position.price_open)
            if new_sl < position.sl:
                self.modify_sl_tp(position.ticket, new_sl, position.tp)
    
    def _handle_position_closed(self) -> None:
        """Handle position closure and update daily loss"""
        if not self.current_position:
            return
            
        # Get closed position history
        from_date = datetime.now() - timedelta(minutes=1)
        history_deals = mt5.history_deals_get(from_date, datetime.now(), ticket=self.current_position["ticket"])
        
        if history_deals:
            deal = history_deals[0]
            profit = deal.profit
            
            # Update daily loss if trade was a loss
            if profit < 0:
                current_day = datetime.now().date()
                if self.last_trade_day != current_day:
                    self.daily_loss = 0
                    self.last_trade_day = current_day
                self.daily_loss += abs(profit)
        
        self.current_position = None
    
    def run(self, check_interval: float = 0.1) -> None:
        """Run the live trading loop"""
        print(f"Starting live trading for {self.symbol}")
        
        while True:
            try:
                # Check if it's weekend
                if datetime.now().weekday() >= 5:
                    print("Weekend - waiting for market open")
                    time.sleep(3600)  # Sleep for 1 hour
                    continue
                
                # Update price buffer
                if not self._update_price_buffer():
                    print("Failed to update market data")
                    time.sleep(1)
                    continue
                
                # Calculate indicators
                df = self._calculate_indicators()
                if df is None:
                    print("Waiting for enough data to calculate indicators...")
                    time.sleep(1)
                    continue
                
                # Check current position
                self.check_position()
                
                # Skip if we already have a position or hit daily loss limit
                if self.current_position or self.daily_loss >= self.max_daily_loss * self.account_info.balance:
                    time.sleep(check_interval)
                    continue
                
                # Prepare observation
                last_row = df.iloc[-1]
                obs = np.array([
                    last_row['EMA9'],
                    last_row['EMA21'],
                    last_row['EMA50'],
                    last_row['RSI'],
                    last_row['ATR'],
                    last_row['OBV'],
                    0,  # No position
                    self.account_info.balance,
                    self.daily_loss,
                    last_row['close']
                ], dtype=np.float32)
                
                # Get model prediction
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Place order if model suggests buy or sell
                if action > 0:
                    current_price = mt5.symbol_info_tick(self.symbol).ask if action == 1 else mt5.symbol_info_tick(self.symbol).bid
                    atr = last_row['ATR']
                    
                    if action == 1:  # Buy
                        sl_price = current_price - atr
                        tp_price = current_price + (1.5 * atr)
                        self.place_order("buy", current_price, sl_price, tp_price)
                    else:  # Sell
                        sl_price = current_price + atr
                        tp_price = current_price - (1.5 * atr)
                        self.place_order("sell", current_price, sl_price, tp_price)
                
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"Error in trading loop: {str(e)}")
                time.sleep(1)
    
    def __del__(self):
        """Cleanup"""
        mt5.shutdown()
