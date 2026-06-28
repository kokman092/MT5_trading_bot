from typing import Dict, List, Optional, Tuple, Deque
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
from collections import deque
import MetaTrader5 as mt5
from ..deployment.error_handler import ErrorHandler

class HFTTrader:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('hft_trader')
        
        # Initialize data structures
        self.order_book_cache = {}
        self.price_cache = {}
        self.trade_cache = {}
        self.signals = {}
        
        # Performance tracking
        self.execution_times = deque(maxlen=1000)
        self.signal_performance = {}
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize trading parameters"""
        self.params = {
            'min_spread': self.config.get('HFT_MIN_SPREAD', 0.0001),
            'max_spread': self.config.get('HFT_MAX_SPREAD', 0.0005),
            'min_volume': self.config.get('HFT_MIN_VOLUME', 1.0),
            'max_position_size': self.config.get('HFT_MAX_POSITION', 0.1),
            'profit_target': self.config.get('HFT_PROFIT_TARGET', 0.0002),
            'stop_loss': self.config.get('HFT_STOP_LOSS', 0.0001),
            'order_timeout': self.config.get('HFT_ORDER_TIMEOUT', 0.5),
            'book_depth': self.config.get('HFT_BOOK_DEPTH', 10)
        }
        
    async def analyze_order_book(self, symbol: str) -> Dict:
        """Analyze order book for trading opportunities"""
        try:
            start_time = datetime.now()
            
            # Get order book
            book = mt5.market_book_get(symbol)
            if not book:
                return {}
                
            # Process order book
            bids = []
            asks = []
            
            for item in book:
                if item.type == mt5.BOOK_TYPE_SELL:
                    asks.append([item.price, item.volume])
                else:
                    bids.append([item.price, item.volume])
                    
            # Calculate metrics
            metrics = await self._calculate_book_metrics(bids, asks)
            
            # Store in cache
            self.order_book_cache[symbol] = {
                'metrics': metrics,
                'timestamp': datetime.now()
            }
            
            # Track execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_times.append(execution_time)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Order book analysis error: {str(e)}")
            return {}
            
    async def _calculate_book_metrics(
        self,
        bids: List[List[float]],
        asks: List[List[float]]
    ) -> Dict:
        """Calculate order book metrics"""
        try:
            if not bids or not asks:
                return {}
                
            # Basic metrics
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid
            
            # Volume analysis
            bid_volume = sum(bid[1] for bid in bids[:self.params['book_depth']])
            ask_volume = sum(ask[1] for ask in asks[:self.params['book_depth']])
            volume_imbalance = bid_volume / (ask_volume + bid_volume)
            
            # Price levels analysis
            bid_levels = [bid[0] for bid in bids[:self.params['book_depth']]]
            ask_levels = [ask[0] for ask in asks[:self.params['book_depth']]]
            
            bid_density = np.std(bid_levels) if len(bid_levels) > 1 else 0
            ask_density = np.std(ask_levels) if len(ask_levels) > 1 else 0
            
            # Calculate order book pressure
            bid_pressure = sum(
                bid[1] * (1 / (i + 1))
                for i, bid in enumerate(bids[:self.params['book_depth']])
            )
            ask_pressure = sum(
                ask[1] * (1 / (i + 1))
                for i, ask in enumerate(asks[:self.params['book_depth']])
            )
            
            pressure_ratio = bid_pressure / ask_pressure if ask_pressure > 0 else 1
            
            return {
                'spread': spread,
                'mid_price': (best_bid + best_ask) / 2,
                'volume_imbalance': volume_imbalance,
                'bid_density': bid_density,
                'ask_density': ask_density,
                'pressure_ratio': pressure_ratio,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume
            }
            
        except Exception as e:
            self.logger.error(f"Book metrics calculation error: {str(e)}")
            return {}
            
    async def analyze_micro_trends(self, symbol: str) -> Dict:
        """Analyze micro-trends in price movement"""
        try:
            # Get recent ticks
            ticks = mt5.copy_ticks_from(
                symbol,
                datetime.now() - timedelta(seconds=5),
                1000,
                mt5.COPY_TICKS_ALL
            )
            
            if ticks is None:
                return {}
                
            # Convert to DataFrame
            df = pd.DataFrame(ticks)
            
            # Calculate micro-trend metrics
            metrics = await self._calculate_trend_metrics(df)
            
            # Store in cache
            self.price_cache[symbol] = {
                'metrics': metrics,
                'timestamp': datetime.now()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Micro-trend analysis error: {str(e)}")
            return {}
            
    async def _calculate_trend_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate micro-trend metrics"""
        try:
            # Price momentum
            price_changes = df['last'].diff()
            momentum = price_changes.mean()
            
            # Volume momentum
            volume_changes = df['volume'].diff()
            volume_momentum = volume_changes.mean()
            
            # Acceleration
            price_acceleration = price_changes.diff().mean()
            
            # Volatility
            volatility = price_changes.std()
            
            # Tick frequency
            tick_frequency = len(df) / 5  # ticks per second
            
            return {
                'momentum': momentum,
                'volume_momentum': volume_momentum,
                'acceleration': price_acceleration,
                'volatility': volatility,
                'tick_frequency': tick_frequency
            }
            
        except Exception as e:
            self.logger.error(f"Trend metrics calculation error: {str(e)}")
            return {}
            
    async def generate_signals(self, symbol: str) -> Dict:
        """Generate trading signals based on analysis"""
        try:
            # Get latest analyses
            book_analysis = self.order_book_cache.get(symbol, {}).get('metrics', {})
            trend_analysis = self.price_cache.get(symbol, {}).get('metrics', {})
            
            if not book_analysis or not trend_analysis:
                return {}
                
            signals = {}
            
            # Order book signals
            if book_analysis['spread'] <= self.params['max_spread']:
                # Volume imbalance signal
                if book_analysis['volume_imbalance'] > 0.6:
                    signals['book_pressure'] = {
                        'direction': 'buy',
                        'strength': book_analysis['volume_imbalance']
                    }
                elif book_analysis['volume_imbalance'] < 0.4:
                    signals['book_pressure'] = {
                        'direction': 'sell',
                        'strength': 1 - book_analysis['volume_imbalance']
                    }
                    
                # Order book pressure signal
                if book_analysis['pressure_ratio'] > 1.2:
                    signals['pressure_ratio'] = {
                        'direction': 'buy',
                        'strength': book_analysis['pressure_ratio'] - 1
                    }
                elif book_analysis['pressure_ratio'] < 0.8:
                    signals['pressure_ratio'] = {
                        'direction': 'sell',
                        'strength': 1 - book_analysis['pressure_ratio']
                    }
                    
            # Micro-trend signals
            if abs(trend_analysis['momentum']) > 0:
                signals['momentum'] = {
                    'direction': 'buy' if trend_analysis['momentum'] > 0 else 'sell',
                    'strength': abs(trend_analysis['momentum']) / trend_analysis['volatility']
                }
                
            # Store signals
            self.signals[symbol] = {
                'signals': signals,
                'timestamp': datetime.now()
            }
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {str(e)}")
            return {}
            
    async def execute_hft_trade(
        self,
        symbol: str,
        signals: Dict
    ) -> Dict:
        """Execute HFT trade based on signals"""
        try:
            if not signals:
                return {}
                
            # Aggregate signal strengths
            buy_strength = 0
            sell_strength = 0
            
            for signal in signals.values():
                if signal['direction'] == 'buy':
                    buy_strength += signal['strength']
                else:
                    sell_strength += signal['strength']
                    
            # Determine trade direction
            if buy_strength > sell_strength and buy_strength > 1:
                direction = 'buy'
                strength = buy_strength
            elif sell_strength > buy_strength and sell_strength > 1:
                direction = 'sell'
                strength = sell_strength
            else:
                return {}
                
            # Calculate position size
            position_size = await self._calculate_position_size(strength)
            
            # Execute trade
            result = await self._execute_trade(
                symbol=symbol,
                direction=direction,
                size=position_size
            )
            
            # Track trade
            if result:
                self.trade_cache[symbol] = {
                    'trade': result,
                    'signals': signals,
                    'timestamp': datetime.now()
                }
                
            return result
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {str(e)}")
            return {}
            
    async def _calculate_position_size(self, signal_strength: float) -> float:
        """Calculate position size based on signal strength"""
        try:
            base_size = self.params['max_position_size'] / 2
            size = base_size * signal_strength
            
            # Apply limits
            size = min(size, self.params['max_position_size'])
            size = max(size, self.params['min_volume'])
            
            return size
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {str(e)}")
            return self.params['min_volume']
            
    async def _execute_trade(
        self,
        symbol: str,
        direction: str,
        size: float
    ) -> Dict:
        """Execute trade with MT5"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {}
                
            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": size,
                "type": mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": tick.ask if direction == 'buy' else tick.bid,
                "deviation": 1,
                "magic": 234000,
                "comment": "hft_trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise Exception(f"Order failed: {result.comment}")
                
            return {
                'order_id': result.order,
                'price': result.price,
                'volume': result.volume,
                'direction': direction,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {str(e)}")
            return {}
            
    def get_performance_metrics(self) -> Dict:
        """Get HFT performance metrics"""
        try:
            metrics = {
                'execution_time': {
                    'mean': np.mean(self.execution_times),
                    'std': np.std(self.execution_times),
                    'max': max(self.execution_times)
                },
                'trades': len(self.trade_cache),
                'signals': {
                    symbol: len(data['signals'])
                    for symbol, data in self.signals.items()
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrics calculation error: {str(e)}")
            return {}
