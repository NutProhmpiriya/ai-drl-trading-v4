import os
import time
import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Rate limiting parameters
REQUEST_DELAY = 1.0  # Delay between requests in seconds
MAX_RETRIES = 3     # Maximum number of retries for failed requests
RETRY_DELAY = 60    # Delay between retries in seconds

def get_or_cache_data(symbol: str, timeframe: str, year: int, data_dir: str) -> Optional[pd.DataFrame]:
    """
    Get data from cache if available, otherwise fetch from MT5 and cache it
    """
    # Create cache directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define cache file path
    cache_file = os.path.join(data_dir, f"{symbol}_{timeframe}m_{year}_data.csv")
    
    # Try to load from cache
    if os.path.exists(cache_file):
        logger.info(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df
    
    # Fetch new data with retries
    logger.info(f"Fetching new data for {symbol} {year}")
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31) if year < datetime.now().year else datetime.now()
    
    for retry in range(MAX_RETRIES):
        try:
            df = fetch_data(symbol, timeframe, start_date, end_date)
            if df is not None:
                # Add technical indicators
                df = add_technical_indicators(df)
                
                # Cache the data
                logger.info(f"Caching data to {cache_file}")
                df.to_csv(cache_file)
                return df
            
        except Exception as e:
            logger.warning(f"Attempt {retry + 1}/{MAX_RETRIES} failed: {str(e)}")
            if retry < MAX_RETRIES - 1:
                logger.info(f"Waiting {RETRY_DELAY} seconds before retrying...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("All retry attempts failed")
                return None
    
    return None

def fetch_data(symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """
    Fetch data from MetaTrader5 with rate limiting
    """
    try:
        # Initialize MT5 if not already initialized
        if not mt5.initialize():
            logger.error("Failed to initialize MetaTrader5")
            return None
            
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
        
        # Add rate limiting delay
        time.sleep(REQUEST_DELAY)
        
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
    finally:
        mt5.shutdown()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame
    """
    try:
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
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        return df  # Return original dataframe if calculation fails
