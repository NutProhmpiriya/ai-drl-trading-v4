import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pytz

class DataProcessor:
    def __init__(self):
        if not mt5.initialize():
            print("Initialize() failed")
            mt5.shutdown()
        
        # Create data directory if it doesn't exist
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from MT5 and calculate technical indicators"""
        
        # Try to load from cache first
        df = self._load_from_cache(symbol, timeframe, start_date, end_date)
        if df is not None:
            print(f"Loaded data from cache for {symbol} {timeframe}m from {start_date.date()} to {end_date.date()}")
            return df
        
        # Convert timeframe string to MT5 timeframe
        timeframe_dict = {
            '1': mt5.TIMEFRAME_M1,
            '5': mt5.TIMEFRAME_M5,
            '15': mt5.TIMEFRAME_M15,
            '30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        
        # Get timezone info
        timezone = pytz.timezone("Etc/UTC")
        start_date = timezone.localize(start_date)
        end_date = timezone.localize(end_date)
        
        # Fetch OHLCV data
        rates = mt5.copy_rates_range(symbol, timeframe_dict.get(timeframe, mt5.TIMEFRAME_M5),
                                   start_date, end_date)
        
        if rates is None:
            print("Failed to fetch data from MT5")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Remove weekend data
        df = df[df.index.dayofweek < 5]
        
        # Calculate technical indicators
        df = self._calculate_indicators(df)
        
        # Save to cache
        self._save_to_cache(df, symbol, timeframe, start_date, end_date)
        print(f"Fetched and cached data for {symbol} {timeframe}m from {start_date.date()} to {end_date.date()}")
        
        return df
    
    def _get_cache_filename(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> str:
        """Generate cache filename based on parameters"""
        return os.path.join(self.data_dir, 
                          f"{symbol}_{timeframe}m_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")
    
    def _save_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str, 
                      start_date: datetime, end_date: datetime):
        """Save data to cache file"""
        cache_file = self._get_cache_filename(symbol, timeframe, start_date, end_date)
        df.to_csv(cache_file)
    
    def _load_from_cache(self, symbol: str, timeframe: str, 
                        start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Try to load data from cache file"""
        cache_file = self._get_cache_filename(symbol, timeframe, start_date, end_date)
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
        return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        
        # Calculate EMAs
        df['EMA9'] = self._calculate_ema(df['close'], 9)
        df['EMA21'] = self._calculate_ema(df['close'], 21)
        df['EMA50'] = self._calculate_ema(df['close'], 50)
        
        # Calculate RSI
        df['RSI'] = self._calculate_rsi(df['close'], 14)
        
        # Calculate ATR
        df['ATR'] = self._calculate_atr(df['high'], df['low'], df['close'], 14)
        
        # Calculate OBV
        df['OBV'] = self._calculate_obv(df['close'], df['tick_volume'])
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        # Rename columns to match environment expectations
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)
        
        return df
    
    @staticmethod
    def _calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range"""
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        return tr['tr'].rolling(period).mean()
    
    @staticmethod
    def _calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def __del__(self):
        """Cleanup MT5 connection"""
        mt5.shutdown()
