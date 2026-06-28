import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import MetaTrader5 as mt5

class TradeManager:
    """Advanced trade management and automated closure system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Trade management parameters
        self.params = {
            'profit_target': config.get('profit_target', 0.02),
            'stop_loss': config.get('stop_loss', 0.02),
            'trailing_stop': config.get('trailing_stop', 0.01),
            'time_stop': config.get('time_stop', 60),  # minutes
            'volatility_multiplier': config.get('volatility_multiplier', 2.0),
            'partial_tp_ratio': config.get('partial_tp_ratio', 0.5),
            'max_adverse_excursion': config.get('max_adverse_excursion', 0.03),
            'min_favorable_excursion': config.get('min_favorable_excursion', 0.01),
            'vol_window': config.get('vol_window', 20),  # Added volatility window
            'regime_fast_ma': config.get('regime_fast_ma', 10),  # Added regime parameters
            'regime_slow_ma': config.get('regime_slow_ma', 30)
        }
        
    def check_closures(self, positions: List[Dict], market_data: Dict) -> List[Dict]:
        """Check all positions for closure conditions"""
        try:
            closure_orders = []
            
            for position in positions:
                # Skip if position is not active
                if not self._is_position_active(position):
                    continue
                    
                # Check various closure conditions
                closure = self._check_position_closure(position, market_data)
                
                if closure['should_close']:
                    order = self._generate_closure_order(position, closure)
                    if order:  # Only append valid orders
                        closure_orders.append(order)
                        
            return closure_orders
            
        except Exception as e:
            self.logger.error(f"Error checking closures: {str(e)}")
            return []
            
    def _is_position_active(self, position: Dict) -> bool:
        """Check if position is still active"""
        try:
            return mt5.positions_get(ticket=position['ticket']) is not None
        except Exception as e:
            self.logger.error(f"Error checking position status: {str(e)}")
            return False
            
    def _check_position_closure(self, position: Dict, market_data: Dict) -> Dict:
        """Check individual position for closure conditions"""
        try:
            symbol = position['symbol']
            position_type = position['type']  # 'buy' or 'sell'
            entry_price = position['price']
            current_price = market_data[symbol]['close'].iloc[-1]
            
            # Initialize result
            result = {
                'should_close': False,
                'reason': None,
                'close_price': None,
                'close_type': None
            }
            
            # Check profit target
            if self._check_profit_target(position, current_price):
                result.update({
                    'should_close': True,
                    'reason': 'profit_target',
                    'close_price': current_price,
                    'close_type': 'market'
                })
                return result
                
            # Check stop loss
            if self._check_stop_loss(position, current_price):
                result.update({
                    'should_close': True,
                    'reason': 'stop_loss',
                    'close_price': current_price,
                    'close_type': 'market'
                })
                return result
                
            # Check trailing stop
            if self._check_trailing_stop(position, current_price):
                result.update({
                    'should_close': True,
                    'reason': 'trailing_stop',
                    'close_price': current_price,
                    'close_type': 'market'
                })
                return result
                
            # Check time stop
            if self._check_time_stop(position):
                result.update({
                    'should_close': True,
                    'reason': 'time_stop',
                    'close_price': current_price,
                    'close_type': 'market'
                })
                return result
                
            # Check volatility stop
            if self._check_volatility_stop(position, market_data):
                result.update({
                    'should_close': True,
                    'reason': 'volatility_stop',
                    'close_price': current_price,
                    'close_type': 'market'
                })
                return result
                
            # Check regime change
            if self._check_regime_change(position, market_data):
                result.update({
                    'should_close': True,
                    'reason': 'regime_change',
                    'close_price': current_price,
                    'close_type': 'market'
                })
                return result
                
            # Check adverse excursion
            if self._check_adverse_excursion(position, market_data):
                result.update({
                    'should_close': True,
                    'reason': 'adverse_excursion',
                    'close_price': current_price,
                    'close_type': 'market'
                })
                return result
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error checking position closure: {str(e)}")
            return {'should_close': False, 'reason': None, 'close_price': None}
            
    def _check_profit_target(self, position: Dict, current_price: float) -> bool:
        """Check if position has reached profit target"""
        try:
            entry_price = position['price']
            position_type = position['type']
            
            if position_type == 'buy':
                profit_ratio = (current_price - entry_price) / entry_price
            else:
                profit_ratio = (entry_price - current_price) / entry_price
                
            return profit_ratio >= self.params['profit_target']
            
        except Exception as e:
            self.logger.error(f"Error checking profit target: {str(e)}")
            return False
            
    def _check_stop_loss(self, position: Dict, current_price: float) -> bool:
        """Check if position has hit stop loss"""
        try:
            entry_price = position['price']
            position_type = position['type']
            
            if position_type == 'buy':
                loss_ratio = (entry_price - current_price) / entry_price
            else:
                loss_ratio = (current_price - entry_price) / entry_price
                
            return loss_ratio >= self.params['stop_loss']
            
        except Exception as e:
            self.logger.error(f"Error checking stop loss: {str(e)}")
            return False
            
    def _check_trailing_stop(self, position: Dict, current_price: float) -> bool:
        """Check if position has hit trailing stop"""
        try:
            highest_price = position.get('highest_price', position['price'])
            lowest_price = position.get('lowest_price', position['price'])
            position_type = position['type']
            
            if position_type == 'buy':
                trailing_stop_price = highest_price * (1 - self.params['trailing_stop'])
                return current_price <= trailing_stop_price
            else:
                trailing_stop_price = lowest_price * (1 + self.params['trailing_stop'])
                return current_price >= trailing_stop_price
                
        except Exception as e:
            self.logger.error(f"Error checking trailing stop: {str(e)}")
            return False
            
    def _check_time_stop(self, position: Dict) -> bool:
        """Check if position has exceeded time limit"""
        try:
            entry_time = position['entry_time']
            current_time = datetime.now()
            
            time_held = (current_time - entry_time).total_seconds() / 60
            return time_held >= self.params['time_stop']
            
        except Exception as e:
            self.logger.error(f"Error checking time stop: {str(e)}")
            return False
            
    def _check_volatility_stop(self, position: Dict, market_data: Dict) -> bool:
        """Check if volatility exceeds threshold"""
        try:
            symbol = position['symbol']
            if symbol not in market_data:
                return False
                
            df = market_data[symbol]
            if df.empty:
                return False
                
            # Calculate volatility
            returns = df['close'].pct_change()
            volatility = returns.rolling(
                window=self.params['vol_window']
            ).std() * np.sqrt(252)  # Annualized
            
            # Calculate average volatility
            avg_volatility = volatility.rolling(
                window=self.params['vol_window']
            ).mean()
            
            current_vol = volatility.iloc[-1]
            avg_vol = avg_volatility.iloc[-1]
            
            return current_vol >= (avg_vol * self.params['volatility_multiplier'])
            
        except Exception as e:
            self.logger.error(f"Error checking volatility stop: {str(e)}")
            return False
            
    def _check_regime_change(self, position: Dict, market_data: Dict) -> bool:
        """Check if market regime has changed"""
        try:
            symbol = position['symbol']
            if symbol not in market_data:
                return False
                
            df = market_data[symbol]
            if df.empty:
                return False
                
            # Calculate regime indicators
            fast_ma = df['close'].rolling(
                window=self.params['regime_fast_ma']
            ).mean()
            slow_ma = df['close'].rolling(
                window=self.params['regime_slow_ma']
            ).mean()
            
            if len(fast_ma) < 2 or len(slow_ma) < 2:
                return False
                
            # Check regime change
            current_regime = fast_ma.iloc[-1] > slow_ma.iloc[-1]
            previous_regime = fast_ma.iloc[-2] > slow_ma.iloc[-2]
            
            # Check if regime aligns with position
            position_type = position['type']
            regime_aligned = (
                (position_type == 'buy' and current_regime) or
                (position_type == 'sell' and not current_regime)
            )
            
            return not regime_aligned
            
        except Exception as e:
            self.logger.error(f"Error checking regime change: {str(e)}")
            return False
            
    def _check_adverse_excursion(self, position: Dict, market_data: Dict) -> bool:
        """Check maximum adverse excursion"""
        try:
            symbol = position['symbol']
            if symbol not in market_data:
                return False
                
            df = market_data[symbol]
            if df.empty:
                return False
                
            entry_price = position['price']
            position_type = position['type']
            
            if position_type == 'buy':
                mae = (df['low'].min() - entry_price) / entry_price
                return abs(mae) >= self.params['max_adverse_excursion']
            else:
                mae = (df['high'].max() - entry_price) / entry_price
                return abs(mae) >= self.params['max_adverse_excursion']
                
        except Exception as e:
            self.logger.error(f"Error checking adverse excursion: {str(e)}")
            return False
            
    def _generate_closure_order(self, position: Dict, closure: Dict) -> Dict:
        """Generate closure order for position"""
        try:
            return {
                'type': 'close',
                'position_id': position['ticket'],
                'symbol': position['symbol'],
                'volume': position['volume'],
                'price': closure['close_price'],
                'reason': closure['reason'],
                'close_type': closure['close_type']
            }
            
        except Exception as e:
            self.logger.error(f"Error generating closure order: {str(e)}")
            return {}
            
    def execute_closure(self, order: Dict) -> bool:
        """Execute closure order"""
        try:
            # Prepare closure request
            symbol = order['symbol']
            position_id = order['position_id']
            volume = order['volume']
            
            # Close position
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_CLOSE_BY,
                "position": position_id,
                "comment": f"Automated closure: {order['reason']}"
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.comment}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing closure: {str(e)}")
            return False
            
    def manage_partial_closure(self, position: Dict, market_data: Dict) -> Optional[Dict]:
        """Manage partial position closure"""
        try:
            entry_price = position['price']
            current_price = market_data[position['symbol']]['close'].iloc[-1]
            position_type = position['type']
            
            # Calculate profit ratio
            if position_type == 'buy':
                profit_ratio = (current_price - entry_price) / entry_price
            else:
                profit_ratio = (entry_price - current_price) / entry_price
                
            # Check if partial closure is needed
            if profit_ratio >= self.params['min_favorable_excursion']:
                closure_volume = position['volume'] * self.params['partial_tp_ratio']
                
                return {
                    'type': 'partial_close',
                    'position_id': position['ticket'],
                    'symbol': position['symbol'],
                    'volume': closure_volume,
                    'price': current_price,
                    'reason': 'partial_profit_target',
                    'close_type': 'market'
                }
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error managing partial closure: {str(e)}")
            return None
