import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from ..indicators.advanced_indicators import AdvancedIndicators
from ..market.regime_detector import MarketRegimeDetector
import ta

class EnhancedHFTStrategy:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = AdvancedIndicators()
        self.regime_detector = MarketRegimeDetector(config)
        
        # Strategy parameters
        self.params = {
            'tick_window': config.get('tick_window', 1000),
            'order_book_levels': config.get('order_book_levels', 10),
            'min_spread': config.get('min_spread', 0.00001),
            'max_spread': config.get('max_spread', 0.0001),
            'min_volume': config.get('min_volume', 1000),
            'latency_threshold': config.get('latency_threshold', 10),  # ms
            'position_timeout': config.get('position_timeout', 30),    # seconds
            'profit_target_ticks': config.get('profit_target_ticks', 2),
            'stop_loss_ticks': config.get('stop_loss_ticks', 1)
        }
        
        # Initialize order book state
        self.order_book_state = {
            'bids': [],
            'asks': [],
            'last_update': None
        }
        
        # Performance tracking
        self.latency_stats = []
        self.execution_stats = []
        
        self._init_parameters()

    def _init_parameters(self):
        """Initialize HFT strategy parameters"""
        try:
            # Price action parameters
            self.tick_window = 100
            self.volume_window = 20
            self.price_threshold = 0.0001  # 1 pip for major pairs
            
            # Order book parameters
            self.book_levels = 5
            self.imbalance_threshold = 1.5
            
            # Momentum parameters
            self.rsi_period = 14
            self.macd_fast = 12
            self.macd_slow = 26
            self.macd_signal = 9
            
            # Volatility parameters
            self.atr_period = 14
            self.volatility_window = 20
            
            # Initialize data containers
            self.tick_data = []
            self.volume_data = []
            self.order_book = {}
            
            self.logger.info("HFT parameters initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing HFT parameters: {str(e)}")
            raise

    def analyze_tick_data(self, tick_data: pd.DataFrame, 
                         order_book: Dict, 
                         market_depth: Dict) -> Dict:
        """
        Analyze tick data with order book and market depth information
        """
        try:
            # Update order book state
            self._update_order_book_state(order_book)
            
            # Check if market conditions are suitable
            if not self._check_market_conditions(tick_data, market_depth):
                return {'signal': 0, 'strength': 0}
            
            # Analyze order flow
            order_flow = self._analyze_order_flow(tick_data)
            
            # Analyze market microstructure
            microstructure = self._analyze_market_microstructure(
                tick_data, market_depth
            )
            
            # Generate trading signals
            signals = self._generate_signals(
                order_flow, microstructure
            )
            
            return {
                'signal': signals['signal'],
                'strength': signals['strength'],
                'latency': signals['latency'],
                'expected_edge': signals['edge']
            }
            
        except Exception as e:
            self.logger.error(f"Error in HFT analysis: {str(e)}")
            return None

    def _update_order_book_state(self, order_book: Dict):
        """
        Update internal order book state
        """
        self.order_book_state = {
            'bids': order_book['bids'][:self.params['order_book_levels']],
            'asks': order_book['asks'][:self.params['order_book_levels']],
            'last_update': datetime.now()
        }

    def _check_market_conditions(self, tick_data: pd.DataFrame, 
                               market_depth: Dict) -> bool:
        """
        Check if market conditions are suitable for HFT
        """
        current_spread = market_depth['asks'][0][0] - market_depth['bids'][0][0]
        current_volume = tick_data['volume'].sum()
        
        # Check spread conditions
        if not (self.params['min_spread'] <= current_spread <= 
                self.params['max_spread']):
            return False
            
        # Check volume conditions
        if current_volume < self.params['min_volume']:
            return False
            
        # Check latency
        if self._get_current_latency() > self.params['latency_threshold']:
            return False
            
        return True

    def _analyze_order_flow(self, tick_data: pd.DataFrame) -> Dict:
        """
        Analyze order flow patterns
        """
        # Calculate order flow imbalance
        buy_volume = tick_data[tick_data['side'] == 'buy']['volume'].sum()
        sell_volume = tick_data[tick_data['side'] == 'sell']['volume'].sum()
        
        imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        
        # Analyze trade sizes
        trade_sizes = tick_data['volume']
        large_trades = trade_sizes > trade_sizes.quantile(0.9)
        
        return {
            'imbalance': imbalance,
            'large_trades': large_trades.sum(),
            'avg_trade_size': trade_sizes.mean()
        }

    def _analyze_market_microstructure(self, tick_data: pd.DataFrame,
                                     market_depth: Dict) -> Dict:
        """
        Analyze market microstructure
        """
        # Calculate order book imbalance
        bid_volume = sum(level[1] for level in market_depth['bids'])
        ask_volume = sum(level[1] for level in market_depth['asks'])
        
        book_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        # Calculate price impact
        mid_price = (market_depth['asks'][0][0] + market_depth['bids'][0][0]) / 2
        price_impact = self._calculate_price_impact(market_depth, mid_price)
        
        return {
            'book_imbalance': book_imbalance,
            'price_impact': price_impact,
            'spread': market_depth['asks'][0][0] - market_depth['bids'][0][0]
        }

    def _calculate_price_impact(self, market_depth: Dict, 
                              mid_price: float) -> float:
        """
        Calculate potential price impact
        """
        total_bid_value = sum(
            level[0] * level[1] for level in market_depth['bids']
        )
        total_ask_value = sum(
            level[0] * level[1] for level in market_depth['asks']
        )
        
        return abs(total_bid_value - total_ask_value) / mid_price

    def _generate_signals(self, order_flow: Dict, 
                         microstructure: Dict) -> Dict:
        """
        Generate trading signals based on order flow and microstructure
        """
        # Combined signal calculation
        signal_strength = (
            0.4 * order_flow['imbalance'] +
            0.4 * microstructure['book_imbalance'] +
            0.2 * np.sign(-microstructure['price_impact'])
        )
        
        # Calculate expected edge
        edge = abs(signal_strength) * (
            1 - microstructure['spread'] / self.params['max_spread']
        )
        
        # Get current latency
        latency = self._get_current_latency()
        
        # Generate final signal
        if abs(signal_strength) > 0.3 and edge > 0.0002:  # 0.02% minimum edge
            signal = np.sign(signal_strength)
        else:
            signal = 0
            
        return {
            'signal': signal,
            'strength': abs(signal_strength),
            'edge': edge,
            'latency': latency
        }

    def _get_current_latency(self) -> float:
        """
        Get current system latency in milliseconds
        """
        if not self.latency_stats:
            return 0
        return np.mean(self.latency_stats[-100:])

    def get_position_size(self, signal: Dict, account_size: float) -> float:
        """
        Calculate position size based on signal strength and edge
        """
        base_size = account_size * 0.001  # 0.1% base risk for HFT
        
        # Adjust size based on signal strength and edge
        adjusted_size = base_size * signal['strength'] * (signal['edge'] * 1000)
        
        return min(adjusted_size, account_size * 0.01)  # Max 1% of account size

    def get_stop_loss_ticks(self, signal: Dict) -> int:
        """
        Get stop loss in ticks
        """
        return self.params['stop_loss_ticks']

    def get_take_profit_ticks(self, signal: Dict) -> int:
        """
        Get take profit in ticks
        """
        return self.params['profit_target_ticks']

    def update_execution_stats(self, execution_time: float, 
                             slippage: float):
        """
        Update execution statistics
        """
        self.execution_stats.append({
            'time': datetime.now(),
            'execution_time': execution_time,
            'slippage': slippage
        })
        
        # Keep only recent stats
        if len(self.execution_stats) > 1000:
            self.execution_stats = self.execution_stats[-1000:]

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signals from market data"""
        try:
            if data.empty:
                self.logger.error("Empty market data provided")
                return {'signal': 0, 'strength': 0, 'edge': 0}
                
            # Prepare market data
            market_data = data.copy()
            market_data['returns'] = market_data['close'].pct_change()
            market_data['volume_ma'] = market_data['volume'].rolling(self.volume_window).mean()
            
            # Calculate technical indicators
            market_data['rsi'] = ta.momentum.RSIIndicator(market_data['close'], window=self.rsi_period).rsi()
            
            macd = ta.trend.MACD(
                market_data['close'],
                window_slow=self.macd_slow,
                window_fast=self.macd_fast,
                window_sign=self.macd_signal
            )
            market_data['macd'] = macd.macd()
            market_data['macd_signal'] = macd.macd_signal()
            
            # Calculate volatility
            market_data['atr'] = ta.volatility.AverageTrueRange(
                high=market_data['high'],
                low=market_data['low'],
                close=market_data['close'],
                window=self.atr_period
            ).average_true_range()
            
            market_data['volatility'] = market_data['returns'].rolling(self.volatility_window).std()
            
            # Get latest data point
            latest = market_data.iloc[-1]
            
            # Check if volatility is within acceptable range
            if latest['volatility'] > self.config.get('max_volatility', 0.001):
                return {'signal': 0, 'strength': 0, 'edge': 0}
                
            # Calculate signal components
            momentum_signal = self._calculate_momentum_signal(latest)
            volume_signal = self._calculate_volume_signal(latest)
            volatility_signal = self._calculate_volatility_signal(latest)
            
            # Combine signals
            signal_strength = (
                0.4 * momentum_signal +
                0.3 * volume_signal +
                0.3 * volatility_signal
            )
            
            # Calculate edge based on market conditions
            edge = self._calculate_edge(latest)
            
            # Generate final signal
            if abs(signal_strength) > self.config.get('signal_threshold', 0.3) and edge > 0:
                signal = np.sign(signal_strength)
            else:
                signal = 0
                
            return {
                'signal': signal,
                'strength': abs(signal_strength),
                'edge': edge,
                'timestamp': market_data.index[-1],
                'price': latest['close'],
                'volume': latest['volume'],
                'indicators': {
                    'rsi': latest['rsi'],
                    'macd': latest['macd'],
                    'atr': latest['atr'],
                    'volatility': latest['volatility']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating HFT signals: {str(e)}")
            return {'signal': 0, 'strength': 0, 'edge': 0}
            
    def _calculate_momentum_signal(self, data: pd.Series) -> float:
        """Calculate momentum signal component"""
        try:
            momentum_signal = 0
            
            # RSI signal
            if data['rsi'] > 70:
                momentum_signal -= 1
            elif data['rsi'] < 30:
                momentum_signal += 1
                
            # MACD signal
            if data['macd'] > data['macd_signal']:
                momentum_signal += 0.5
            else:
                momentum_signal -= 0.5
                
            return np.clip(momentum_signal / 2, -1, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum signal: {str(e)}")
            return 0
            
    def _calculate_volume_signal(self, data: pd.Series) -> float:
        """Calculate volume signal component"""
        try:
            if data['volume'] > data['volume_ma'] * 1.5:
                return np.sign(data['returns'])
            elif data['volume'] < data['volume_ma'] * 0.5:
                return -np.sign(data['returns'])
            return 0
            
        except Exception as e:
            self.logger.error(f"Error calculating volume signal: {str(e)}")
            return 0
            
    def _calculate_volatility_signal(self, data: pd.Series) -> float:
        """Calculate volatility signal component"""
        try:
            # Higher volatility -> reduce signal strength
            vol_ratio = data['volatility'] / self.config.get('target_volatility', 0.0005)
            return 1 - np.clip(vol_ratio - 1, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility signal: {str(e)}")
            return 0
            
    def _calculate_edge(self, data: pd.Series) -> float:
        """Calculate trading edge based on market conditions"""
        try:
            # Base edge on volatility and momentum alignment
            volatility_edge = 1 - (data['volatility'] / self.config.get('max_volatility', 0.001))
            momentum_edge = abs(data['macd']) / (data['atr'] + 1e-6)
            
            # Combine edges
            edge = (volatility_edge + momentum_edge) / 2
            
            return max(0, edge)
            
        except Exception as e:
            self.logger.error(f"Error calculating edge: {str(e)}")
            return 0
