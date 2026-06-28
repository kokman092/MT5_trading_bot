from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import logging
import MetaTrader5 as mt5
from ..analytics.market_analyzer import MarketAnalyzer
from ..deployment.error_handler import ErrorHandler

class BehaviorManager:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('behavior_manager')
        self.market_analyzer = MarketAnalyzer(config)
        
        # Initialize behavior components
        self._init_behavior_components()
        
    def _init_behavior_components(self):
        """Initialize behavior components"""
        # Trading patterns
        self.patterns = {
            'aggressive': {
                'weight': 0.2,
                'order_size_multiplier': (1.2, 1.5),
                'timing_variance': (0.8, 1.2),
                'strategy_switch_prob': 0.3
            },
            'conservative': {
                'weight': 0.3,
                'order_size_multiplier': (0.7, 0.9),
                'timing_variance': (0.9, 1.1),
                'strategy_switch_prob': 0.1
            },
            'neutral': {
                'weight': 0.5,
                'order_size_multiplier': (0.9, 1.1),
                'timing_variance': (0.95, 1.05),
                'strategy_switch_prob': 0.2
            }
        }
        
        # Time-based behavior modifiers
        self.time_modifiers = {
            'market_open': {
                'hours': [0, 1],  # Hours after market open
                'size_modifier': 0.8,
                'timing_modifier': 1.2
            },
            'market_close': {
                'hours': [-1, 0],  # Hours before market close
                'size_modifier': 0.7,
                'timing_modifier': 1.3
            },
            'high_volatility': {
                'vol_threshold': 1.5,  # Volatility threshold
                'size_modifier': 0.6,
                'timing_modifier': 1.4
            }
        }
        
        # Strategy rotation parameters
        self.strategy_rotation = {
            'min_hold_time': 3600,  # Minimum time to hold a strategy (seconds)
            'max_hold_time': 14400,  # Maximum time to hold a strategy (seconds)
            'performance_threshold': 0.02  # 2% threshold for forced rotation
        }
        
        # Initialize state
        self.current_pattern = 'neutral'
        self.last_pattern_switch = datetime.now()
        self.pattern_history = []
        self.strategy_history = []
        
    async def modify_order_parameters(
        self,
        order_params: Dict,
        market_state: Dict
    ) -> Dict:
        """Modify order parameters to exhibit human-like behavior"""
        try:
            # Update current pattern if needed
            await self._update_current_pattern(market_state)
            
            # Get current pattern settings
            pattern = self.patterns[self.current_pattern]
            
            # Apply time-based modifiers
            time_modifiers = await self._get_time_modifiers(market_state)
            
            # Modify order size
            modified_size = await self._modify_order_size(
                order_params['volume'],
                pattern,
                time_modifiers
            )
            
            # Modify order timing
            modified_timing = await self._modify_order_timing(
                pattern,
                time_modifiers
            )
            
            # Modify order price
            modified_price = await self._modify_order_price(
                order_params['price'],
                pattern,
                market_state
            )
            
            return {
                **order_params,
                'volume': modified_size,
                'delay': modified_timing,
                'price': modified_price
            }
            
        except Exception as e:
            self.logger.error(f"Order parameter modification error: {str(e)}")
            return order_params
            
    async def should_switch_strategy(
        self,
        current_strategy: str,
        performance_metrics: Dict
    ) -> Tuple[bool, Optional[str]]:
        """Determine if strategy should be switched"""
        try:
            # Check minimum hold time
            time_since_switch = (
                datetime.now() - self._get_last_strategy_switch()
            ).total_seconds()
            
            if time_since_switch < self.strategy_rotation['min_hold_time']:
                return False, None
                
            # Check performance threshold
            if performance_metrics.get('recent_return', 0) < -self.strategy_rotation['performance_threshold']:
                return True, await self._select_new_strategy(current_strategy)
                
            # Random strategy switch based on pattern
            pattern = self.patterns[self.current_pattern]
            if random.random() < pattern['strategy_switch_prob']:
                return True, await self._select_new_strategy(current_strategy)
                
            return False, None
            
        except Exception as e:
            self.logger.error(f"Strategy switch decision error: {str(e)}")
            return False, None
            
    async def _update_current_pattern(self, market_state: Dict):
        """Update current trading pattern based on market conditions"""
        try:
            # Check if it's time to switch patterns
            time_since_switch = (
                datetime.now() - self.last_pattern_switch
            ).total_seconds()
            
            if time_since_switch < 1800:  # Minimum 30 minutes between switches
                return
                
            # Calculate pattern probabilities based on market state
            probabilities = await self._calculate_pattern_probabilities(
                market_state
            )
            
            # Randomly select new pattern
            patterns = list(self.patterns.keys())
            weights = [probabilities[p] for p in patterns]
            new_pattern = random.choices(patterns, weights=weights)[0]
            
            if new_pattern != self.current_pattern:
                self.current_pattern = new_pattern
                self.last_pattern_switch = datetime.now()
                self.pattern_history.append({
                    'pattern': new_pattern,
                    'timestamp': datetime.now(),
                    'market_state': market_state
                })
                
        except Exception as e:
            self.logger.error(f"Pattern update error: {str(e)}")
            
    async def _calculate_pattern_probabilities(
        self,
        market_state: Dict
    ) -> Dict[str, float]:
        """Calculate probabilities for each pattern based on market state"""
        try:
            probabilities = {
                pattern: self.patterns[pattern]['weight']
                for pattern in self.patterns
            }
            
            # Adjust based on volatility
            volatility = market_state.get('volatility', 1.0)
            if volatility > self.time_modifiers['high_volatility']['vol_threshold']:
                probabilities['aggressive'] *= 0.7
                probabilities['conservative'] *= 1.3
                
            # Adjust based on trend strength
            trend_strength = market_state.get('trend_strength', 0.5)
            if trend_strength > 0.7:
                probabilities['aggressive'] *= 1.2
                probabilities['conservative'] *= 0.8
                
            # Normalize probabilities
            total = sum(probabilities.values())
            return {k: v/total for k, v in probabilities.items()}
            
        except Exception as e:
            self.logger.error(f"Probability calculation error: {str(e)}")
            return {p: 1/len(self.patterns) for p in self.patterns}
            
    async def _modify_order_size(
        self,
        original_size: float,
        pattern: Dict,
        time_modifiers: Dict
    ) -> float:
        """Modify order size based on pattern and time modifiers"""
        try:
            # Get base multiplier range
            min_mult, max_mult = pattern['order_size_multiplier']
            
            # Apply random multiplier within range
            size_mult = random.uniform(min_mult, max_mult)
            
            # Apply time modifiers
            for modifier in time_modifiers:
                size_mult *= modifier.get('size_modifier', 1.0)
                
            # Calculate new size
            new_size = original_size * size_mult
            
            # Round to symbol's lot step
            symbol_info = mt5.symbol_info(self.config['symbol'])
            if symbol_info:
                lot_step = symbol_info.volume_step
                new_size = round(new_size / lot_step) * lot_step
                
            return new_size
            
        except Exception as e:
            self.logger.error(f"Order size modification error: {str(e)}")
            return original_size
            
    async def _modify_order_timing(
        self,
        pattern: Dict,
        time_modifiers: Dict
    ) -> float:
        """Calculate order timing delay"""
        try:
            # Get base timing variance range
            min_var, max_var = pattern['timing_variance']
            
            # Calculate base delay
            base_delay = random.uniform(min_var, max_var)
            
            # Apply time modifiers
            for modifier in time_modifiers:
                base_delay *= modifier.get('timing_modifier', 1.0)
                
            # Add random noise
            noise = random.uniform(-0.1, 0.1)
            delay = base_delay * (1 + noise)
            
            return max(0, delay)  # Ensure non-negative delay
            
        except Exception as e:
            self.logger.error(f"Order timing modification error: {str(e)}")
            return 0
            
    async def _modify_order_price(
        self,
        original_price: float,
        pattern: Dict,
        market_state: Dict
    ) -> float:
        """Modify order price based on pattern and market state"""
        try:
            # Calculate price variance based on pattern
            if self.current_pattern == 'aggressive':
                variance = random.uniform(-0.0005, 0.0005)
            elif self.current_pattern == 'conservative':
                variance = random.uniform(-0.0002, 0.0002)
            else:
                variance = random.uniform(-0.0003, 0.0003)
                
            # Apply market state adjustments
            volatility = market_state.get('volatility', 1.0)
            if volatility > self.time_modifiers['high_volatility']['vol_threshold']:
                variance *= 0.5  # Reduce variance in high volatility
                
            # Calculate new price
            new_price = original_price * (1 + variance)
            
            # Round to symbol's price precision
            symbol_info = mt5.symbol_info(self.config['symbol'])
            if symbol_info:
                digits = symbol_info.digits
                new_price = round(new_price, digits)
                
            return new_price
            
        except Exception as e:
            self.logger.error(f"Order price modification error: {str(e)}")
            return original_price
            
    async def _get_time_modifiers(self, market_state: Dict) -> List[Dict]:
        """Get applicable time-based modifiers"""
        try:
            modifiers = []
            current_time = datetime.now()
            
            # Check market open/close modifiers
            market_hours = await self._get_market_hours()
            if market_hours:
                time_since_open = (
                    current_time - market_hours['open']
                ).total_seconds() / 3600
                
                time_to_close = (
                    market_hours['close'] - current_time
                ).total_seconds() / 3600
                
                if time_since_open <= self.time_modifiers['market_open']['hours'][1]:
                    modifiers.append(self.time_modifiers['market_open'])
                    
                if time_to_close <= self.time_modifiers['market_close']['hours'][1]:
                    modifiers.append(self.time_modifiers['market_close'])
                    
            # Check volatility modifier
            if market_state.get('volatility', 0) > self.time_modifiers['high_volatility']['vol_threshold']:
                modifiers.append(self.time_modifiers['high_volatility'])
                
            return modifiers
            
        except Exception as e:
            self.logger.error(f"Time modifier calculation error: {str(e)}")
            return []
            
    async def _select_new_strategy(self, current_strategy: str) -> str:
        """Select new strategy based on pattern and history"""
        try:
            available_strategies = [
                s for s in self.config.get('strategies', [])
                if s != current_strategy
            ]
            
            if not available_strategies:
                return current_strategy
                
            # Weight strategies based on pattern
            weights = []
            for strategy in available_strategies:
                weight = 1.0
                
                # Adjust weight based on pattern
                if self.current_pattern == 'aggressive':
                    if 'aggressive' in strategy.lower():
                        weight *= 1.3
                elif self.current_pattern == 'conservative':
                    if 'conservative' in strategy.lower():
                        weight *= 1.3
                        
                # Adjust weight based on recent performance
                performance = await self._get_strategy_performance(strategy)
                if performance > 0:
                    weight *= (1 + performance)
                    
                weights.append(weight)
                
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w/total_weight for w in weights]
            else:
                weights = [1/len(weights)] * len(weights)
                
            # Select strategy
            new_strategy = random.choices(
                available_strategies,
                weights=weights
            )[0]
            
            # Record switch
            self.strategy_history.append({
                'timestamp': datetime.now(),
                'old_strategy': current_strategy,
                'new_strategy': new_strategy,
                'pattern': self.current_pattern
            })
            
            return new_strategy
            
        except Exception as e:
            self.logger.error(f"Strategy selection error: {str(e)}")
            return current_strategy
            
    def _get_last_strategy_switch(self) -> datetime:
        """Get timestamp of last strategy switch"""
        try:
            if self.strategy_history:
                return self.strategy_history[-1]['timestamp']
            return datetime.min
            
        except Exception as e:
            self.logger.error(f"Strategy history error: {str(e)}")
            return datetime.min
            
    async def get_behavior_metrics(self) -> Dict:
        """Get current behavior metrics"""
        try:
            return {
                'current_pattern': self.current_pattern,
                'pattern_duration': (
                    datetime.now() - self.last_pattern_switch
                ).total_seconds(),
                'strategy_switches': len(self.strategy_history),
                'pattern_switches': len(self.pattern_history)
            }
            
        except Exception as e:
            self.logger.error(f"Behavior metrics error: {str(e)}")
            return {}
