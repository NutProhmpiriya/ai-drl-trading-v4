import os
import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        if not mt5.initialize():
            print("Initialize() failed")
            mt5.shutdown()
        
        # Create data directory if it doesn't exist
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def get_or_cache_data(self, symbol: str, timeframe: str, year: int) -> Optional[pd.DataFrame]:
        """
        Get data from cache if available, otherwise fetch from MT5 and cache it
        """
        # Define cache file path
        cache_file = os.path.join(self.data_dir, f"{symbol}_{timeframe}m_{year}_data.csv")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            logger.info(f"Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            return df
        
        # Fetch new data
        logger.info(f"Fetching new data for {symbol} {year}")
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31) if year < datetime.now().year else datetime.now()
        
        df = self.fetch_data(symbol, timeframe, start_date, end_date)
        if df is not None:
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Cache the data
            logger.info(f"Caching data to {cache_file}")
            df.to_csv(cache_file)
        
        return df

    def fetch_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch data from MetaTrader5
        """
        try:
            # Convert timeframe string to MT5 timeframe
            timeframe_map = {
                '1': mt5.TIMEFRAME_M1,
                '5': mt5.TIMEFRAME_M5,
                '15': mt5.TIMEFRAME_M15,
                '30': mt5.TIMEFRAME_M30,
                '60': mt5.TIMEFRAME_H1,
                '240': mt5.TIMEFRAME_H4,
                'D': mt5.TIMEFRAME_D1,
            }
            mt5_timeframe = timeframe_map.get(timeframe)
            if mt5_timeframe is None:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            # Fetch the data
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            if rates is None or len(rates) == 0:
                logger.error(f"Failed to fetch data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame
        """
        # Calculate EMAs
        df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Calculate OBV
        df['OBV'] = (np.sign(df['close'].diff()) * df['tick_volume']).fillna(0).cumsum()
        
        # Forward fill any NaN values
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df

    def __del__(self):
        """Cleanup MT5 connection"""
        mt5.shutdown()
