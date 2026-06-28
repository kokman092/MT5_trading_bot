import logging
import pandas as pd
import MetaTrader5 as mt5
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class MarketData:
    def __init__(self, config: Dict):
        """Initialize market data manager"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._data_cache = {}
        self._last_update = {}
        
    def get_market_data(self, symbol: str, timeframe: str, lookback: int = 100) -> Optional[pd.DataFrame]:
        """Get market data for a symbol"""
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
                return None
            
            # For crypto, use a longer lookback period due to higher volatility
            if symbol in ['BTCUSD', 'ETHUSD', 'XRPUSD', 'DOTUSD', 'ADAUSD', 'SOLUSD']:
                lookback = int(lookback * 1.5)  # 50% more data for better analysis
                
            # Check cache
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self._data_cache:
                last_update = self._last_update.get(cache_key, datetime.min)
                # More frequent updates for crypto due to 24/7 trading
                update_interval = 0.5 if symbol in ['BTCUSD', 'ETHUSD', 'XRPUSD', 'DOTUSD', 'ADAUSD', 'SOLUSD'] else 1
                if datetime.now() - last_update < timedelta(minutes=update_interval):
                    return self._data_cache[cache_key]
                    
            # Get rates from MT5
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, lookback)
            if rates is None:
                self.logger.error(f"Failed to get rates for {symbol}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Cache data
            self._data_cache[cache_key] = df
            self._last_update[cache_key] = datetime.now()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return None
            
    def get_tick_data(self, symbol: str) -> Optional[Dict]:
        """Get latest tick data for a symbol"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.error(f"Failed to get tick data for {symbol}")
                return None
                
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': datetime.fromtimestamp(tick.time),
                'spread': tick.ask - tick.bid
            }
            
        except Exception as e:
            self.logger.error(f"Error getting tick data: {str(e)}")
            return None
            
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                self.logger.error(f"Failed to get symbol info for {symbol}")
                return None
                
            return {
                'symbol': symbol,
                'digits': info.digits,
                'point': info.point,
                'tick_size': info.trade_tick_size,
                'contract_size': info.trade_contract_size,
                'volume_min': info.volume_min,
                'volume_max': info.volume_max,
                'volume_step': info.volume_step
            }
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {str(e)}")
            return None
            
    def is_market_open(self, symbol: str) -> bool:
        """Check if market is open for trading"""
        try:
            # Crypto markets are always open
            if symbol in ['BTCUSD', 'ETHUSD', 'XRPUSD', 'DOTUSD', 'ADAUSD', 'SOLUSD']:
                return True
                
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Failed to get symbol info for {symbol}")
                return False
                
            return bool(symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED)
            
        except Exception as e:
            self.logger.error(f"Error checking market status for {symbol}: {str(e)}")
            return False
            
    def clean_cache(self):
        """Remove old cache entries"""
        try:
            current_time = datetime.now()
            keys_to_remove = []
            
            for key, last_update in self._last_update.items():
                if current_time - last_update > timedelta(minutes=5):
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del self._data_cache[key]
                del self._last_update[key]
                
        except Exception as e:
            self.logger.error(f"Error cleaning cache: {str(e)}")
            
    def get_active_symbols(self) -> List[str]:
        """Get list of active trading symbols"""
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                self.logger.error("Failed to get symbols")
                return []
                
            # Filter for configured symbols
            configured_symbols = self.config.get('symbols', [])
            active_symbols = [s.name for s in symbols if s.visible and s.name in configured_symbols]
            
            return active_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting active symbols: {str(e)}")
            return []
