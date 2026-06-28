import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import ta
from datetime import datetime, timedelta
from typing import Dict, Optional, List

class MarketData:
    def __init__(self, config: Dict):
        """Initialize market data handler"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._data_cache = {}
        self.cache_timeout = 60  # Cache timeout in seconds
        
    def get_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get market data for symbol and timeframe with technical indicators"""
        try:
            # Convert timeframe string to MT5 timeframe
            tf_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = tf_map.get(timeframe)
            if mt5_timeframe is None:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return pd.DataFrame()
                
            # Get number of bars from config
            num_bars = self.config.get('market_data', {}).get('bars', 1000)
            
            # Get rates from MT5
            rates = mt5.copy_rates_from(symbol, mt5_timeframe, datetime.now(), num_bars)
            if rates is None:
                self.logger.error(f"Failed to get rates for {symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return pd.DataFrame()
            
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        try:
            if df.empty:
                return df
                
            # Get indicator settings from config
            indicator_config = self.config.get('technical_indicators', {})
            
            # Moving Averages
            sma_periods = indicator_config.get('sma_periods', [20, 50, 200])
            for period in sma_periods:
                df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
            
            ema_periods = indicator_config.get('ema_periods', [9, 21])
            for period in ema_periods:
                df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
            
            # RSI
            rsi_period = indicator_config.get('rsi_period', 14)
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi()
            
            # MACD
            macd = ta.trend.MACD(
                df['close'],
                window_slow=indicator_config.get('macd_slow', 26),
                window_fast=indicator_config.get('macd_fast', 12),
                window_sign=indicator_config.get('macd_signal', 9)
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bb_period = indicator_config.get('bb_period', 20)
            bb_std = indicator_config.get('bb_std', 2)
            bollinger = ta.volatility.BollingerBands(df['close'], window=bb_period, window_dev=bb_std)
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            
            # ATR
            atr_period = indicator_config.get('atr_period', 14)
            df['atr'] = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=atr_period
            ).average_true_range()
            
            # Stochastic
            stoch_period = indicator_config.get('stoch_period', 14)
            stoch = ta.momentum.StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=stoch_period
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            return df
            
    def get_tick_data(self, symbol: str, num_ticks: int = 1000) -> pd.DataFrame:
        """Get tick data for symbol"""
        try:
            # Get ticks from MT5
            ticks = mt5.copy_ticks_from(symbol, datetime.now(), num_ticks, mt5.COPY_TICKS_ALL)
            if ticks is None:
                self.logger.error(f"Failed to get ticks for {symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting tick data: {str(e)}")
            return pd.DataFrame()
            
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                self.logger.error(f"Failed to get symbol info for {symbol}")
                return None
                
            return {
                'symbol': symbol,
                'bid': info.bid,
                'ask': info.ask,
                'point': info.point,
                'digits': info.digits,
                'spread': info.spread,
                'trade_mode': info.trade_mode,
                'volume_min': info.volume_min,
                'volume_max': info.volume_max,
                'volume_step': info.volume_step
            }
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {str(e)}")
            return None
