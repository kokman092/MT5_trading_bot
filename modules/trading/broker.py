import asyncio
import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta
import time
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from functools import wraps

# Add retry decorator for MT5 operations
def mt5_operation_with_retry(max_retries=3, retry_delay=1.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    logger = logging.getLogger(__name__)
                    logger.warning(f"MT5 operation failed: {str(e)}. Retry {retries}/{max_retries}")
                    if retries == max_retries:
                        raise
                    await asyncio.sleep(retry_delay * retries)  # Exponential backoff
            return None
        return wrapper
    return decorator

class MT5Broker:
    _instance = None
    _connection_lock = asyncio.Lock()
    _data_cache = {}
    _cache_expiry = {}
    
    def __new__(cls, config: Dict):
        """Singleton pattern to ensure only one broker instance"""
        if cls._instance is None:
            cls._instance = super(MT5Broker, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
        
    def __init__(self, config: Dict):
        """Initialize MT5 broker connection"""
        # Skip if already initialized
        if hasattr(self, 'config') and self.config == config and self.initialized:
            return
            
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.last_connection_check = datetime.now()
        self.connection_check_interval = timedelta(minutes=5)
        self._initialize()
        
    async def ensure_connection(self):
        """Ensure MT5 connection is active and reconnect if needed"""
        async with self._connection_lock:
            current_time = datetime.now()
            
            # Check connection periodically
            if (current_time - self.last_connection_check > self.connection_check_interval or 
                not self.initialized or not mt5.terminal_info()):
                self.logger.info("Checking MT5 connection status")
                
                # If not connected, try to reconnect
                if not mt5.terminal_info():
                    self.logger.warning("MT5 connection lost. Attempting to reconnect...")
                    mt5.shutdown()
                    self._initialize()
                
                self.last_connection_check = current_time

    def _initialize(self) -> None:
        """Initialize connection to MT5"""
        try:
            # Fix for MT5 connection issues - ensure clean shutdown first
            mt5.shutdown()
            
            # Only pass path if it is explicitly configured and not None
            mt5_path = self.config.get('mt5_path', None)
            timeout_val = self.config.get('mt5_account', {}).get('timeout', 60000)
            
            if mt5_path:
                success = mt5.initialize(path=str(mt5_path), timeout=timeout_val)
            else:
                success = mt5.initialize(timeout=timeout_val)
                
            if not success:
                raise Exception(f"Failed to initialize MT5: {mt5.last_error()}")
                
            # Login to MT5 account
            account_config = self.config.get('mt5_account', {})
            if not mt5.login(
                login=int(account_config.get('login')),
                password=account_config.get('password'),
                server=account_config.get('server')
            ):
                raise Exception(f"Failed to login: {mt5.last_error()}")
                
            self.initialized = True
            self.logger.info("Successfully initialized MT5 broker connection")
            
        except Exception as e:
            self.logger.error(f"Error initializing MT5 broker: {str(e)}")
            raise

    def __del__(self):
        """Cleanup MT5 connection"""
        if hasattr(self, 'initialized') and self.initialized:
            mt5.shutdown()

    @mt5_operation_with_retry()
    async def get_account_info(self) -> Dict:
        """Get current account information"""
        await self.ensure_connection()
            
        account_info = mt5.account_info()
        if account_info is None:
            raise Exception(f"Failed to get account info: {mt5.last_error()}")
            
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'margin_level': account_info.margin_level,
            'leverage': account_info.leverage
        }
        
    def get_positions(self) -> List[Dict]:
        """Get current open positions"""
        if not self.initialized:
            raise Exception("Broker not initialized")
            
        positions = mt5.positions_get()
        if positions is None:
            return []
            
        return [{
            'ticket': pos.ticket,
            'symbol': pos.symbol,
            'type': 'buy' if pos.type == mt5.POSITION_TYPE_BUY else 'sell',
            'volume': pos.volume,
            'open_price': pos.price_open,
            'sl': pos.sl,
            'tp': pos.tp,
            'profit': pos.profit,
            'swap': pos.swap,
            'magic': pos.magic
        } for pos in positions]
        
    def _get_filling_mode(self, symbol: str) -> int:
        """Get the supported filling mode for a symbol dynamically"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return mt5.ORDER_FILLING_IOC
            
        filling_mode = symbol_info.filling_mode
        # 1 = SYMBOL_FILLING_FOK, 2 = SYMBOL_FILLING_IOC
        if filling_mode & 1:
            return mt5.ORDER_FILLING_FOK
        elif filling_mode & 2:
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURN

    def place_order(self, order_params: Dict) -> Optional[int]:
        """Place a new order"""
        if not self.initialized:
            raise Exception("Broker not initialized")
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order_params['symbol'],
            "volume": float(order_params['volume']),
            "type": mt5.ORDER_TYPE_BUY if order_params['type'].lower() == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": order_params.get('price', 0.0),
            "sl": order_params.get('sl', 0.0),
            "tp": order_params.get('tp', 0.0),
            "deviation": order_params.get('deviation', 10),
            "magic": order_params.get('magic', 123456),
            "comment": order_params.get('comment', ""),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(order_params['symbol']),
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order failed: {result.comment}")
            return None
            
        return result.order
        
    def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> bool:
        """Modify an existing position"""
        if not self.initialized:
            raise Exception("Broker not initialized")
            
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
            
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position[0].symbol,
            "sl": sl if sl is not None else position[0].sl,
            "tp": tp if tp is not None else position[0].tp,
            "position": ticket
        }
        
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE
        
    def close_position(self, ticket: int) -> bool:
        """Close an existing position"""
        if not self.initialized:
            raise Exception("Broker not initialized")
            
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position[0].symbol,
            "volume": position[0].volume,
            "type": mt5.ORDER_TYPE_SELL if position[0].type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": 0.0,
            "deviation": 10,
            "magic": 123456,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(position[0].symbol),
        }
        
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE
        
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        if not self.initialized:
            raise Exception("Broker not initialized")
            
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
            
        return {
            'bid': info.bid,
            'ask': info.ask,
            'spread': info.spread,
            'digits': info.digits,
            'min_volume': info.volume_min,
            'max_volume': info.volume_max,
            'volume_step': info.volume_step,
            'trade_mode': info.trade_mode
        }
        
    @mt5_operation_with_retry()
    async def get_historical_data(self, symbol: str, timeframe: str, bars: int = 1000, start_time=None, end_time=None) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data with caching"""
        await self.ensure_connection()
        
        # Convert timeframe string to MT5 timeframe
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        
        tf = timeframe_map.get(timeframe)
        if tf is None:
            self.logger.error(f"Invalid timeframe: {timeframe}")
            return None
            
        # Create cache key
        cache_key = f"{symbol}_{timeframe}_{bars}_{start_time}_{end_time}"
        
        # Check if we have recent data in cache
        if cache_key in self._data_cache and cache_key in self._cache_expiry:
            if datetime.now() < self._cache_expiry[cache_key]:
                return self._data_cache[cache_key]
                
        # Fetch data from MT5
        if start_time and end_time:
            rates = mt5.copy_rates_range(symbol, tf, start_time, end_time)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            
        if rates is None or len(rates) == 0:
            self.logger.warning(f"No historical data for {symbol} {timeframe}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Cache the data - expires after timeframe-appropriate interval
        expiry_minutes = max(int(timeframe.replace('M', '').replace('H', '')) * 60 if 'H' in timeframe else int(timeframe.replace('M', '')), 5)
        self._data_cache[cache_key] = df
        self._cache_expiry[cache_key] = datetime.now() + timedelta(minutes=expiry_minutes)
        
        return df

    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get current market data for symbol"""
        try:
            # Get symbol info
            symbol_info = await asyncio.to_thread(mt5.symbol_info, symbol)
            if not symbol_info:
                return None

            # Get recent OHLCV data
            rates = await asyncio.to_thread(
                mt5.copy_rates_from_pos,
                symbol,
                mt5.TIMEFRAME_M1,
                0,
                100
            )

            if rates is None:
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            return {
                'symbol': symbol,
                'bid': symbol_info.bid,
                'ask': symbol_info.ask,
                'spread': symbol_info.spread,
                'volume': symbol_info.volume,
                'time': datetime.now(),
                'historical_data': df,
                'trade_mode': symbol_info.trade_mode,
                'session_deals': symbol_info.session_deals,
                'session_volume': symbol_info.session_volume
            }

        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None

    async def calculate_margin(self, symbol: str, volume: float, price: float) -> float:
        """Calculate required margin for position"""
        try:
            margin_info = await asyncio.to_thread(
                mt5.order_calc_margin,
                mt5.ORDER_TYPE_BUY,
                symbol,
                volume,
                price
            )
            
            return margin_info if margin_info else float('inf')

        except Exception as e:
            self.logger.error(f"Error calculating margin: {str(e)}")
            return float('inf')

    async def get_total_exposure(self) -> float:
        """Get total account exposure"""
        try:
            positions = await asyncio.to_thread(mt5.positions_get)
            if positions is None:
                return 0.0

            total_exposure = sum(pos.volume for pos in positions)
            account_info = await self.get_account_info()
            
            return total_exposure / account_info['equity'] if account_info else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating total exposure: {str(e)}")
            return 0.0

    async def get_symbol_exposure(self, symbol: str) -> float:
        """Get exposure for specific symbol"""
        try:
            positions = await asyncio.to_thread(mt5.positions_get, symbol=symbol)
            if positions is None:
                return 0.0

            return sum(pos.volume for pos in positions)

        except Exception as e:
            self.logger.error(f"Error getting symbol exposure: {str(e)}")
            return 0.0

    async def get_account_value(self) -> float:
        """Get current account value"""
        try:
            account_info = await self.get_account_info()
            return account_info['equity'] if account_info else 0.0

        except Exception as e:
            self.logger.error(f"Error getting account value: {str(e)}")
            return 0.0

    async def execute_trade(self, trade_params: Dict) -> Dict:
        """Execute trade with given parameters"""
        try:
            symbol_info = await asyncio.to_thread(mt5.symbol_info, trade_params['symbol'])
            if not symbol_info:
                return {'success': False, 'error': 'Symbol not found'}

            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": trade_params['symbol'],
                "volume": trade_params['volume'],
                "type": mt5.ORDER_TYPE_BUY if trade_params['direction'] == 'long' else mt5.ORDER_TYPE_SELL,
                "price": symbol_info.ask if trade_params['direction'] == 'long' else symbol_info.bid,
                "sl": trade_params['stop_loss'],
                "tp": trade_params['take_profit'],
                "deviation": self.config.get('order_deviation', 20),
                "magic": self.config.get('magic_number', 234000),
                "comment": trade_params.get('comment', 'python trade'),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._get_filling_mode(trade_params['symbol']),
            }

            # Execute trade
            result = await asyncio.to_thread(mt5.order_send, request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error': f"Order failed: {result.comment}",
                    'retcode': result.retcode
                }

            return {
                'success': True,
                'trade_id': result.order,
                'entry_price': result.price,
                'volume': result.volume,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Trade execution error: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def close_trade(self, trade_id: int) -> Dict:
        """Close an existing trade"""
        try:
            position = await asyncio.to_thread(mt5.positions_get, ticket=trade_id)
            if not position:
                return {'success': False, 'error': 'Position not found'}

            position = position[0]
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": trade_id,
                "price": await asyncio.to_thread(mt5.symbol_info_tick, position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else await asyncio.to_thread(mt5.symbol_info_tick, position.symbol).ask,
                "deviation": self.config.get('order_deviation', 20),
                "magic": position.magic,
                "comment": "python close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._get_filling_mode(position.symbol),
            }

            # Execute close
            result = await asyncio.to_thread(mt5.order_send, request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error': f"Close failed: {result.comment}",
                    'retcode': result.retcode
                }

            return {
                'success': True,
                'trade_id': trade_id,
                'exit_price': result.price,
                'profit': result.profit,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Trade closing error: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def update_trade_history(self, trade_record: Dict):
        """Update trade history database"""
        try:
            # Implement trade history storage
            # This is a placeholder - implement according to your storage solution
            pass

        except Exception as e:
            self.logger.error(f"Trade history update error: {str(e)}") 