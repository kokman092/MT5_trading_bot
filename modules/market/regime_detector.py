import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from ..indicators.advanced_indicators import AdvancedIndicators

class MarketRegimeDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = AdvancedIndicators()
        
        # Configuration parameters
        self.volatility_window = config.get('volatility_window', 20)
        self.adx_threshold = config.get('adx_threshold', 25)
        self.volatility_threshold = config.get('volatility_threshold', {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5
        })

    def detect_regime(self, data: pd.DataFrame) -> Dict:
        """
        Detect current market regime using multiple indicators
        """
        try:
            # Calculate key indicators
            market_state = self.indicators.calculate_market_regime(
                data['close'],
                adx_period=14,
                ma_fast=10,
                ma_slow=30
            )
            
            # Calculate volatility state
            atr = self.indicators.calculate_atr(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=self.volatility_window
            )
            atr_pct = atr / data['close']
            current_volatility = atr_pct.iloc[-1]
            
            # Determine volatility regime
            if current_volatility < self.volatility_threshold['low']:
                volatility_state = "LOW"
            elif current_volatility < self.volatility_threshold['medium']:
                volatility_state = "MEDIUM"
            else:
                volatility_state = "HIGH"
            
            # Calculate additional market characteristics
            ichimoku = self.indicators.calculate_ichimoku(
                data['high'],
                data['low'],
                data['close']
            )
            
            # Determine trend strength and direction
            trend_strength = "STRONG" if market_state['regime'].startswith("STRONG") else "WEAK"
            trend_direction = "UP" if "UP" in market_state['regime'] else "DOWN" if "DOWN" in market_state['regime'] else "NEUTRAL"
            
            return {
                'market_state': market_state['regime'],
                'volatility_state': volatility_state,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'current_volatility': current_volatility,
                'confidence': market_state['confidence'],
                'ichimoku_cloud_state': self._analyze_ichimoku_state(ichimoku, data['close'].iloc[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error in regime detection: {str(e)}")
            return None

    def _analyze_ichimoku_state(self, ichimoku: Dict, current_price: float) -> str:
        """
        Analyze Ichimoku Cloud state for additional trend confirmation
        """
        if (current_price > ichimoku['senkou_span_a'].iloc[-1] and 
            current_price > ichimoku['senkou_span_b'].iloc[-1]):
            return "STRONG_BULLISH"
        elif (current_price < ichimoku['senkou_span_a'].iloc[-1] and 
              current_price < ichimoku['senkou_span_b'].iloc[-1]):
            return "STRONG_BEARISH"
        else:
            return "NEUTRAL"

    def get_suitable_strategies(self, regime: Dict) -> List[str]:
        """
        Determine which strategies are suitable for current market regime
        """
        suitable_strategies = []
        
        # Trending market strategies
        if regime['market_state'].startswith("STRONG"):
            suitable_strategies.extend(['MOMENTUM', 'TREND_FOLLOWING'])
            
        # Ranging market strategies
        if regime['market_state'] == "RANGING":
            suitable_strategies.extend(['MEAN_REVERSION', 'GRID_TRADING'])
            
        # Volatility-based strategy selection
        if regime['volatility_state'] == "LOW":
            suitable_strategies.extend(['GRID_TRADING', 'PAIRS_TRADING'])
        elif regime['volatility_state'] == "MEDIUM":
            suitable_strategies.extend(['SCALPING', 'MOMENTUM'])
        else:  # HIGH volatility
            suitable_strategies.extend(['TREND_FOLLOWING'])
            
        return list(set(suitable_strategies))  # Remove duplicates

    def should_adjust_parameters(self, regime: Dict) -> Dict:
        """
        Determine if strategy parameters should be adjusted based on market regime
        """
        adjustments = {
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'position_size_multiplier': 1.0,
            'entry_threshold_multiplier': 1.0
        }
        
        # Adjust based on volatility
        if regime['volatility_state'] == "HIGH":
            adjustments['stop_loss_multiplier'] = 1.5
            adjustments['take_profit_multiplier'] = 1.5
            adjustments['position_size_multiplier'] = 0.7
            adjustments['entry_threshold_multiplier'] = 1.3
        elif regime['volatility_state'] == "LOW":
            adjustments['stop_loss_multiplier'] = 0.8
            adjustments['take_profit_multiplier'] = 0.8
            adjustments['position_size_multiplier'] = 1.2
            adjustments['entry_threshold_multiplier'] = 0.8
            
        return adjustments
