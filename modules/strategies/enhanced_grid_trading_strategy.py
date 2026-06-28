import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from ..indicators.advanced_indicators import AdvancedIndicators
from ..market.regime_detector import MarketRegimeDetector

class EnhancedGridTradingStrategy:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = AdvancedIndicators()
        self.regime_detector = MarketRegimeDetector(config)
        
        # Strategy parameters
        self.params = {
            'grid_levels': config.get('grid_levels', 10),
            'base_grid_spacing': config.get('base_grid_spacing', 0.001),
            'atr_multiplier': config.get('atr_multiplier', 0.5),
            'max_active_positions': config.get('max_active_positions', 5),
            'max_drawdown': config.get('max_drawdown', 0.05),  # 5% max drawdown
            'adx_threshold': config.get('adx_threshold', 25),
            'min_volatility': config.get('min_volatility', 0.0005),
            'max_volatility': config.get('max_volatility', 0.003)
        }
        
        self.active_grids = {}
        self.total_pnl = 0
        self.max_equity = 0

    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """
        Enhanced grid trading analysis with dynamic spacing and market regime detection
        """
        try:
            # Get market regime
            regime = self.regime_detector.detect_regime(data)
            
            # Skip if market is strongly trending
            if regime['market_state'] in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
                return {'active': False, 'regime': regime}
            
            # Calculate ATR for dynamic grid spacing
            atr = self.indicators.calculate_atr(data['close'])
            current_volatility = atr.iloc[-1] / data['close'].iloc[-1]
            
            # Check if volatility is within acceptable range
            if (current_volatility < self.params['min_volatility'] or 
                current_volatility > self.params['max_volatility']):
                return {'active': False, 'regime': regime}
            
            # Calculate dynamic grid spacing
            grid_spacing = self._calculate_grid_spacing(
                current_price=data['close'].iloc[-1],
                atr=atr.iloc[-1]
            )
            
            # Generate grid levels
            grid_levels = self._generate_grid_levels(
                current_price=data['close'].iloc[-1],
                grid_spacing=grid_spacing
            )
            
            # Check drawdown limit
            if self._check_drawdown_limit():
                return {'active': False, 'regime': regime}
            
            return {
                'active': True,
                'regime': regime,
                'grid_levels': grid_levels,
                'current_volatility': current_volatility,
                'grid_spacing': grid_spacing
            }
            
        except Exception as e:
            self.logger.error(f"Error in grid trading analysis: {str(e)}")
            return None

    def _calculate_grid_spacing(self, current_price: float, atr: float) -> float:
        """
        Calculate dynamic grid spacing based on ATR
        """
        base_spacing = self.params['base_grid_spacing'] * current_price
        atr_spacing = atr * self.params['atr_multiplier']
        
        return max(base_spacing, atr_spacing)

    def _generate_grid_levels(self, current_price: float, 
                            grid_spacing: float) -> List[Dict]:
        """
        Generate grid levels with dynamic spacing
        """
        levels = []
        
        # Generate levels above current price
        for i in range(self.params['grid_levels'] // 2):
            level_price = current_price + (i + 1) * grid_spacing
            levels.append({
                'price': level_price,
                'type': 'SELL',
                'size': self._calculate_position_size(level_price)
            })
            
        # Generate levels below current price
        for i in range(self.params['grid_levels'] // 2):
            level_price = current_price - (i + 1) * grid_spacing
            levels.append({
                'price': level_price,
                'type': 'BUY',
                'size': self._calculate_position_size(level_price)
            })
            
        return sorted(levels, key=lambda x: x['price'])

    def _calculate_position_size(self, price: float) -> float:
        """
        Calculate position size for each grid level
        """
        return self.config['base_position_size']

    def _check_drawdown_limit(self) -> bool:
        """
        Check if current drawdown exceeds maximum allowed
        """
        if self.total_pnl < 0:
            current_drawdown = abs(self.total_pnl) / self.max_equity
            return current_drawdown > self.params['max_drawdown']
        return False

    def update_grid_positions(self, current_price: float, 
                            positions: List[Dict]) -> List[Dict]:
        """
        Update and manage grid positions
        """
        actions = []
        
        # Update max equity
        current_equity = self.total_pnl + sum(
            pos['unrealized_pnl'] for pos in positions
        )
        self.max_equity = max(self.max_equity, current_equity)
        
        # Check each grid level
        for level in self.active_grids.values():
            # Close positions at take profit levels
            if (level['type'] == 'BUY' and 
                current_price >= level['take_profit']):
                actions.append({
                    'action': 'CLOSE',
                    'position_id': level['position_id']
                })
            elif (level['type'] == 'SELL' and 
                  current_price <= level['take_profit']):
                actions.append({
                    'action': 'CLOSE',
                    'position_id': level['position_id']
                })
                
        return actions

    def get_stop_loss(self, entry_price: float, grid_type: str) -> float:
        """
        Calculate stop loss for grid positions
        """
        if grid_type == 'BUY':
            return entry_price * 0.98  # 2% stop loss
        else:
            return entry_price * 1.02  # 2% stop loss

    def get_take_profit(self, entry_price: float, grid_type: str, 
                       grid_spacing: float) -> float:
        """
        Calculate take profit for grid positions
        """
        if grid_type == 'BUY':
            return entry_price * (1 + grid_spacing * 2)
        else:
            return entry_price * (1 - grid_spacing * 2)

    async def generate_signals(self, market_data: Dict) -> Dict:
        """
        Generate trading signals based on market data
        
        Args:
            market_data (Dict): Dictionary containing:
                - symbol: Trading symbol
                - data: DataFrame with OHLCV data
                - timestamp: Current timestamp
                
        Returns:
            Dict: Signal information containing:
                - action: Trading action (BUY, SELL, NONE)
                - confidence: Signal confidence level
                - metadata: Additional signal information
        """
        try:
            # Analyze market conditions
            analysis = self.analyze_market(market_data['data'])
            if analysis is None or not analysis['active']:
                return {
                    'action': 'NONE',
                    'confidence': 0.0,
                    'metadata': {
                        'regime': analysis['regime'] if analysis else None,
                        'error': 'Analysis failed or market conditions not suitable'
                    }
                }
                
            current_price = market_data['data']['close'].iloc[-1]
            
            # Find closest grid level
            closest_level = None
            min_distance = float('inf')
            
            for level in analysis['grid_levels']:
                distance = abs(level['price'] - current_price)
                if distance < min_distance:
                    min_distance = distance
                    closest_level = level
            
            if closest_level is None:
                return {
                    'action': 'NONE',
                    'confidence': 0.0,
                    'metadata': {'error': 'No valid grid levels found'}
                }
                
            # Calculate confidence based on price distance to grid level
            confidence = max(0, 1 - (min_distance / analysis['grid_spacing']))
            
            # Generate signal based on closest grid level
            action = closest_level['type']
            
            # Calculate entry parameters
            metadata = {
                'entry_price': closest_level['price'],
                'position_size': closest_level['size'],
                'grid_spacing': analysis['grid_spacing'],
                'regime': analysis['regime'],
                'volatility': analysis['current_volatility']
            }
            
            if action != 'NONE':
                metadata.update({
                    'stop_loss': self.get_stop_loss(
                        closest_level['price'],
                        action
                    ),
                    'take_profit': self.get_take_profit(
                        closest_level['price'],
                        action,
                        analysis['grid_spacing']
                    )
                })
            
            return {
                'action': action,
                'confidence': confidence,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error generating grid trading signals: {str(e)}")
            return {
                'action': 'NONE',
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
