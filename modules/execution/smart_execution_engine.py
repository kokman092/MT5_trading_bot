import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import MetaTrader5 as mt5
from collections import deque

class SmartExecutionEngine:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Execution parameters
        self.max_spread = config.get('MAX_SPREAD', 0.0003)
        self.min_volume = config.get('MIN_VOLUME', 0.01)
        self.max_slippage = config.get('MAX_SLIPPAGE', 0.0002)
        self.retry_attempts = config.get('RETRY_ATTEMPTS', 3)
        self.retry_delay = config.get('RETRY_DELAY', 1)  # seconds
        
        # TWAP/VWAP parameters
        self.twap_interval = config.get('TWAP_INTERVAL', 300)  # seconds
        self.vwap_lookback = config.get('VWAP_LOOKBACK', 20)  # periods
        
        # Execution analytics
        self.execution_history = []
        self.market_impact = {}
        self.spread_history = {}
        
        # Order book analytics
        self.order_book_cache = {}
        self.volume_profile = {}
        
        # Load history
        self.load_history()
    
    def execute_order(self, symbol, order_type, volume, price=None, sl=None, tp=None,
                     execution_algo='smart'):
        """Execute order using specified algorithm"""
        try:
            # Validate market conditions
            if not self.validate_market_conditions(symbol):
                return None
            
            # Choose execution algorithm
            if execution_algo == 'twap':
                return self.execute_twap(symbol, order_type, volume, price, sl, tp)
            elif execution_algo == 'vwap':
                return self.execute_vwap(symbol, order_type, volume, price, sl, tp)
            elif execution_algo == 'smart':
                return self.execute_smart(symbol, order_type, volume, price, sl, tp)
            else:
                return self.execute_market(symbol, order_type, volume, price, sl, tp)
            
        except Exception as e:
            self.logger.error(f"Error executing order: {str(e)}")
            return None
    
    def validate_market_conditions(self, symbol):
        """Validate current market conditions"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Failed to get symbol info for {symbol}")
                return False
            
            # Check spread
            current_spread = symbol_info.ask - symbol_info.bid
            if current_spread > self.max_spread:
                self.logger.warning(f"Spread too high: {current_spread}")
                return False
            
            # Check trading hours
            if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                self.logger.warning("Market is closed")
                return False
            
            # Update spread history
            if symbol not in self.spread_history:
                self.spread_history[symbol] = deque(maxlen=1000)
            self.spread_history[symbol].append({
                'timestamp': datetime.now().isoformat(),
                'spread': current_spread
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating market conditions: {str(e)}")
            return False
    
    def execute_smart(self, symbol, order_type, volume, price=None, sl=None, tp=None):
        """Smart order execution with adaptive logic"""
        try:
            # Analyze order book
            book = self.analyze_order_book(symbol)
            if not book:
                return self.execute_market(symbol, order_type, volume, price, sl, tp)
            
            # Determine optimal execution strategy
            if book['is_liquid'] and book['spread'] <= self.max_spread:
                if volume > book['avg_trade_size'] * 3:
                    # Large order - use TWAP/VWAP
                    if book['volume_profile'] == 'increasing':
                        return self.execute_vwap(symbol, order_type, volume, price, sl, tp)
                    else:
                        return self.execute_twap(symbol, order_type, volume, price, sl, tp)
                else:
                    # Small order - use intelligent market order
                    return self.execute_market(symbol, order_type, volume, price, sl, tp)
            else:
                # Poor liquidity - use careful TWAP
                return self.execute_twap(symbol, order_type, volume, price, sl, tp)
            
        except Exception as e:
            self.logger.error(f"Error executing smart order: {str(e)}")
            return None
    
    def execute_twap(self, symbol, order_type, volume, price=None, sl=None, tp=None):
        """Execute order using Time-Weighted Average Price"""
        try:
            # Calculate sub-orders
            interval_count = 5
            sub_volume = volume / interval_count
            
            orders = []
            for i in range(interval_count):
                # Wait for interval
                if i > 0:
                    mt5.sleep(self.twap_interval / interval_count)
                
                # Execute sub-order
                result = self.execute_market(symbol, order_type, sub_volume, price, sl, tp)
                if result:
                    orders.append(result)
                else:
                    self.logger.warning(f"Failed to execute TWAP sub-order {i+1}")
            
            # Record execution
            self.record_execution('twap', symbol, order_type, volume, orders)
            
            return orders if orders else None
            
        except Exception as e:
            self.logger.error(f"Error executing TWAP order: {str(e)}")
            return None
    
    def execute_vwap(self, symbol, order_type, volume, price=None, sl=None, tp=None):
        """Execute order using Volume-Weighted Average Price"""
        try:
            # Get volume profile
            volume_profile = self.get_volume_profile(symbol)
            if not volume_profile:
                return self.execute_twap(symbol, order_type, volume, price, sl, tp)
            
            # Calculate volume-weighted sub-orders
            total_volume = sum(v for _, v in volume_profile)
            sub_orders = []
            remaining_volume = volume
            
            for period_volume in volume_profile:
                sub_volume = min(remaining_volume, volume * period_volume[1] / total_volume)
                if sub_volume >= self.min_volume:
                    sub_orders.append(sub_volume)
                    remaining_volume -= sub_volume
            
            # Execute sub-orders
            orders = []
            for sub_volume in sub_orders:
                result = self.execute_market(symbol, order_type, sub_volume, price, sl, tp)
                if result:
                    orders.append(result)
                mt5.sleep(self.twap_interval / len(sub_orders))
            
            # Record execution
            self.record_execution('vwap', symbol, order_type, volume, orders)
            
            return orders if orders else None
            
        except Exception as e:
            self.logger.error(f"Error executing VWAP order: {str(e)}")
            return None
    
    def execute_market(self, symbol, order_type, volume, price=None, sl=None, tp=None):
        """Execute market order with smart retry logic"""
        try:
            for attempt in range(self.retry_attempts):
                # Prepare order request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL,
                    "price": price if price else mt5.symbol_info_tick(symbol).ask if order_type == "buy" else mt5.symbol_info_tick(symbol).bid,
                    "deviation": int(self.max_slippage * 100000),
                    "magic": 234000,
                    "comment": f"smart_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                if sl:
                    request["sl"] = sl
                if tp:
                    request["tp"] = tp
                
                # Send order
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Record execution
                    self.record_execution('market', symbol, order_type, volume, [result])
                    return result
                
                self.logger.warning(f"Order attempt {attempt + 1} failed: {result.comment}")
                mt5.sleep(self.retry_delay)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error executing market order: {str(e)}")
            return None
    
    def analyze_order_book(self, symbol):
        """Analyze order book for liquidity and execution strategy"""
        try:
            # Get order book
            book = mt5.market_book_get(symbol)
            if not book:
                return None
            
            # Calculate metrics
            bids = [b for b in book if b.type == mt5.BOOK_TYPE_SELL]
            asks = [a for a in book if a.type == mt5.BOOK_TYPE_BUY]
            
            total_bid_volume = sum(b.volume for b in bids)
            total_ask_volume = sum(a.volume for a in asks)
            
            analysis = {
                'spread': asks[0].price - bids[0].price if asks and bids else float('inf'),
                'bid_volume': total_bid_volume,
                'ask_volume': total_ask_volume,
                'is_liquid': total_bid_volume > 0 and total_ask_volume > 0,
                'avg_trade_size': np.mean([b.volume for b in book]),
                'volume_profile': 'increasing' if total_ask_volume > total_bid_volume else 'decreasing'
            }
            
            # Cache analysis
            self.order_book_cache[symbol] = {
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing order book: {str(e)}")
            return None
    
    def get_volume_profile(self, symbol, timeframe=mt5.TIMEFRAME_M1):
        """Get volume profile for VWAP calculation"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, self.vwap_lookback)
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            df['hour'] = pd.to_datetime(df['time'], unit='s').dt.hour
            
            # Calculate volume profile
            volume_profile = df.groupby('hour')['tick_volume'].mean()
            return list(volume_profile.items())
            
        except Exception as e:
            self.logger.error(f"Error getting volume profile: {str(e)}")
            return None
    
    def record_execution(self, algo, symbol, order_type, volume, results):
        """Record execution details for analysis"""
        try:
            execution = {
                'timestamp': datetime.now().isoformat(),
                'algorithm': algo,
                'symbol': symbol,
                'order_type': order_type,
                'volume': volume,
                'results': [
                    {
                        'order_id': r.order,
                        'price': r.price,
                        'volume': r.volume,
                        'time': r.time
                    } for r in results
                ]
            }
            
            self.execution_history.append(execution)
            
            # Calculate market impact
            if symbol not in self.market_impact:
                self.market_impact[symbol] = []
            
            # Get post-trade price movement
            post_trade_ticks = mt5.copy_ticks_from(
                symbol,
                datetime.now(),
                100,
                mt5.COPY_TICKS_ALL
            )
            
            if post_trade_ticks is not None:
                df = pd.DataFrame(post_trade_ticks)
                price_impact = (df['ask'].mean() - results[-1].price) if order_type == 'buy' else (results[-1].price - df['bid'].mean())
                
                self.market_impact[symbol].append({
                    'timestamp': datetime.now().isoformat(),
                    'volume': volume,
                    'impact': price_impact
                })
            
            self.save_history()
            
        except Exception as e:
            self.logger.error(f"Error recording execution: {str(e)}")
    
    def get_execution_analytics(self):
        """Get execution analytics and insights"""
        try:
            if not self.execution_history:
                return None
            
            analytics = {}
            for symbol in set(e['symbol'] for e in self.execution_history):
                symbol_executions = [e for e in self.execution_history if e['symbol'] == symbol]
                
                # Calculate metrics
                analytics[symbol] = {
                    'total_executions': len(symbol_executions),
                    'avg_market_impact': np.mean([m['impact'] for m in self.market_impact.get(symbol, [])]),
                    'avg_spread': np.mean([s['spread'] for s in self.spread_history.get(symbol, [])]),
                    'algo_performance': self.calculate_algo_performance(symbol_executions)
                }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error getting execution analytics: {str(e)}")
            return None
    
    def calculate_algo_performance(self, executions):
        """Calculate performance metrics for each execution algorithm"""
        try:
            algo_stats = {}
            for algo in ['market', 'twap', 'vwap', 'smart']:
                algo_executions = [e for e in executions if e['algorithm'] == algo]
                if not algo_executions:
                    continue
                
                # Calculate metrics
                volumes = [e['volume'] for e in algo_executions]
                prices = [[r['price'] for r in e['results']] for e in algo_executions]
                
                algo_stats[algo] = {
                    'usage_count': len(algo_executions),
                    'avg_volume': np.mean(volumes),
                    'price_improvement': np.mean([max(p) - min(p) for p in prices if p]),
                    'success_rate': len([e for e in algo_executions if e['results']]) / len(algo_executions)
                }
            
            return algo_stats
            
        except Exception as e:
            self.logger.error(f"Error calculating algo performance: {str(e)}")
            return None
    
    def save_history(self):
        """Save execution history"""
        try:
            history_data = {
                'execution_history': self.execution_history,
                'market_impact': self.market_impact,
                'spread_history': {
                    symbol: list(spreads) for symbol, spreads in self.spread_history.items()
                }
            }
            
            with open('execution_history.json', 'w') as f:
                json.dump(history_data, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving history: {str(e)}")
    
    def load_history(self):
        """Load execution history"""
        try:
            with open('execution_history.json', 'r') as f:
                history_data = json.load(f)
                self.execution_history = history_data['execution_history']
                self.market_impact = history_data['market_impact']
                self.spread_history = {
                    symbol: deque(spreads, maxlen=1000)
                    for symbol, spreads in history_data['spread_history'].items()
                }
                
        except FileNotFoundError:
            self.logger.info("No execution history file found. Starting fresh.")
        except Exception as e:
            self.logger.error(f"Error loading history: {str(e)}")
