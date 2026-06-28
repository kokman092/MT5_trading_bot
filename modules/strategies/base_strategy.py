from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.market_params = config['MARKET_PARAMS'][config['ACTIVE_MARKET']]
        
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Analyze market data and generate trading signals
        Returns: Dictionary with signal details or None if no signal
        """
        pass
        
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate if we have enough data for analysis"""
        return data is not None and len(data) >= 50
        
    def _calculate_signal_strength(self, value: float, lower_bound: float, upper_bound: float) -> float:
        """Calculate signal strength between 0 and 100"""
        strength = ((value - lower_bound) / (upper_bound - lower_bound)) * 100
        return max(0, min(100, strength))
        
    def _get_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate +DM and -DM
        high_diff = high - high.shift(1)
        low_diff = low.shift(1) - low
        
        plus_dm = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
        minus_dm = ((low_diff > high_diff) & (low_diff > 0)) * low_diff
        
        # Calculate TR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate smoothed averages
        period = 14
        tr_smooth = tr.rolling(window=period).mean()
        plus_di = (plus_dm.rolling(window=period).mean() / tr_smooth) * 100
        minus_di = (minus_dm.rolling(window=period).mean() / tr_smooth) * 100
        
        # Calculate ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = dx.rolling(window=period).mean()
        
        return adx.iloc[-1]
        
    def _is_volume_significant(self, data: pd.DataFrame) -> bool:
        """Check if current volume is significantly higher than average"""
        current_volume = data['tick_volume'].iloc[-1]
        avg_volume = data['tick_volume'].rolling(window=20).mean().iloc[-1]
        return current_volume > (avg_volume * 1.5)
        
    def _calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> tuple:
        """Calculate dynamic support and resistance levels"""
        rolling_high = data['high'].rolling(window=window).max()
        rolling_low = data['low'].rolling(window=window).min()
        
        resistance = rolling_high.iloc[-1]
        support = rolling_low.iloc[-1]
        
        return support, resistance
        
    def _is_breakout(self, data: pd.DataFrame, level: float, direction: str) -> bool:
        """Check if price has broken through support/resistance level"""
        current_price = data['close'].iloc[-1]
        previous_price = data['close'].iloc[-2]
        
        if direction == 'up':
            return previous_price <= level < current_price and self._is_volume_significant(data)
        else:
            return previous_price >= level > current_price and self._is_volume_significant(data)
            
    def _calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """Calculate price volatility using ATR"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=window).mean()
        return atr.iloc[-1]
        
    def _is_ranging_market(self, data: pd.DataFrame, threshold: float = 25) -> bool:
        """Determine if market is ranging (non-trending)"""
        adx = self._get_trend_strength(data)
        return adx < threshold
