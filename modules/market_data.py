import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import logging
from typing import Dict, Optional, List, Union

class MarketDataFetcher:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        
    def get_current_price(self) -> Optional[Dict]:
        """Get current bid/ask prices"""
        symbol_info = mt5.symbol_info_tick(self.symbol)
        if symbol_info is None:
            return None
        return {
            'bid': symbol_info.bid,
            'ask': symbol_info.ask,
            'spread': symbol_info.ask - symbol_info.bid
        }
        
    def get_historical_data(self, num_bars: int = 100) -> Optional[pd.DataFrame]:
        """Fetch historical price data and calculate indicators"""
        timeframe = self.timeframe_map.get(self.timeframe, mt5.TIMEFRAME_M5)
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, num_bars)
        
        if rates is None:
            logging.error(f"Failed to fetch historical data for {self.symbol}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Add technical indicators
        self._add_indicators(df)
        return df
        
    def _add_indicators(self, df: pd.DataFrame) -> None:
        """Calculate and add technical indicators to the dataframe"""
        # Moving Averages
        sma = SMAIndicator(close=df['close'], window=20)
        ema = EMAIndicator(close=df['close'], window=20)
        df['SMA20'] = sma.sma_indicator()
        df['EMA20'] = ema.ema_indicator()
        
        # RSI
        rsi = RSIIndicator(close=df['close'], window=14)
        df['RSI'] = rsi.rsi()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        
    def get_order_book(self, depth: int = 10) -> Optional[Dict]:
        """Get market depth/order book data"""
        book = mt5.market_book_get(self.symbol)
        if book is None:
            return None
            
        orders = {
            'bids': [],
            'asks': []
        }
        
        for order in book[:depth]:
            if order.type == mt5.BOOK_TYPE_SELL:
                orders['asks'].append({'price': order.price, 'volume': order.volume})
            else:
                orders['bids'].append({'price': order.price, 'volume': order.volume})
                
        return orders
        
    def get_market_info(self) -> Optional[Dict]:
        """Get symbol specification and trading conditions"""
        info = mt5.symbol_info(self.symbol)
        if info is None:
            return None
            
        return {
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'trade_contract_size': info.trade_contract_size,
            'point': info.point,
            'tick_size': info.trade_tick_size,
            'tick_value': info.trade_tick_value
        }
