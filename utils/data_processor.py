import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

class DataProcessor:
    def __init__(self):
        if not mt5.initialize():
            print("Initialize() failed")
            mt5.shutdown()

    def fetch_data(self, symbol: str, timeframe: int, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from MT5 and calculate technical indicators"""
        
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
        
        return df
    
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
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
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
