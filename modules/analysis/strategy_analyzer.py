import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import MetaTrader5 as mt5
from .advanced_strategies import AdvancedStrategies

class StrategyAnalyzer:
    def __init__(self, config: Dict):
        """Initialize strategy analyzer with configuration"""
        self.config = config
        self.advanced = AdvancedStrategies(config)
        
    def analyze_market_regime(self, rates: pd.DataFrame) -> Dict:
        """
        Analyze the current market regime (trending, ranging, volatile)
        Args:
            rates: DataFrame with OHLCV data
        Returns:
            Dict with market regime analysis results
        """
        # Get basic regime analysis
        basic_regime = self._analyze_basic_regime(rates)
        
        # Get advanced analysis
        volatility_regime = self.advanced.volatility_regime_analysis(rates)
        market_structure = self.advanced.market_structure_analysis(rates)
        volume_profile = self.advanced.volume_profile_analysis(rates)
        
        # Combine analyses
        return {
            **basic_regime,
            'volatility_regime': volatility_regime['regime'],
            'support_levels': market_structure['support_levels'],
            'resistance_levels': market_structure['resistance_levels'],
            'high_volume_nodes': volume_profile['high_volume_nodes']
        }
    
    def _analyze_basic_regime(self, rates: pd.DataFrame) -> Dict:
        """Basic market regime analysis"""
        close = rates['close'].values
        high = rates['high'].values
        low = rates['low'].values
        
        # Calculate volatility
        atr = self._calculate_atr(rates)
        avg_atr = atr
        
        # Calculate trend strength
        adx = self._calculate_adx(high, low, close)
        
        # Determine market regime
        if adx > 25:  # Strong trend
            regime = 'trending'
            trend_strength = min(adx, 100)
        elif atr < avg_atr * 0.5:  # Low volatility
            regime = 'ranging'
            trend_strength = max(25 - adx, 0)
        else:
            regime = 'volatile'
            trend_strength = min(adx / 2, 50)
            
        return {
            'regime': regime,
            'trend_strength': float(trend_strength),
            'volatility': float(avg_atr),
            'atr': float(atr),
            'adx': float(adx)
        }
    
    def analyze_market_conditions(self, symbol: str) -> Dict:
        """Comprehensive market analysis using multiple approaches"""
        try:
            # Get data for multiple timeframes
            timeframes = self.config['analysis']['timeframes']
            mtf_analysis = self.advanced.analyze_multi_timeframe(symbol, timeframes)
            
            # Get order flow analysis
            order_flow = self.advanced.order_flow_analysis(symbol)
            
            # Get recent market data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
            if rates is None:
                return {}
                
            df = pd.DataFrame(rates)
            
            # Get regime analysis
            regime = self.analyze_market_regime(df)
            
            return {
                'multi_timeframe': mtf_analysis,
                'order_flow': order_flow,
                'regime': regime
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")
            return {}
    
    def calculate_position_metrics(self, symbol: str, confidence: float) -> Dict:
        """Calculate optimal position metrics based on comprehensive analysis"""
        try:
            # Get market conditions
            conditions = self.analyze_market_conditions(symbol)
            if not conditions:
                return {'size_multiplier': 0.5, 'risk_multiplier': 0.5}
            
            # Adjust multipliers based on market conditions
            regime = conditions['regime']
            mtf = conditions['multi_timeframe']
            
            # Base multipliers on trend alignment
            size_mult = min(1.0, confidence * (0.5 + mtf['trend_alignment']))
            
            # Adjust risk based on volatility regime
            vol_adjustments = {
                'high': 0.7,
                'normal': 1.0,
                'low': 1.2
            }
            risk_mult = vol_adjustments.get(regime['volatility_regime'], 1.0)
            
            # Further adjust based on order flow
            if conditions['order_flow'].get('imbalance', 0) > 0:
                size_mult *= 1.1  # Increase size on positive order flow
            
            return {
                'size_multiplier': min(size_mult, 1.5),  # Cap at 150%
                'risk_multiplier': max(0.5, min(risk_mult, 1.2)),  # Keep between 50-120%
                'market_conditions': conditions
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position metrics: {str(e)}")
            return {'size_multiplier': 0.5, 'risk_multiplier': 0.5}
    
    def detect_market_inefficiencies(self, rates: pd.DataFrame) -> List[Dict]:
        """
        Detect potential market inefficiencies like gaps, liquidity pools
        Args:
            rates: DataFrame with OHLCV data
        Returns:
            List of detected inefficiencies with their characteristics
        """
        inefficiencies = []
        
        # Detect price gaps
        gaps = self._find_price_gaps(rates)
        if gaps:
            inefficiencies.extend(gaps)
            
        # Detect liquidity pools
        pools = self._find_liquidity_pools(rates)
        if pools:
            inefficiencies.extend(pools)
            
        return inefficiencies
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # True Range calculations
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return 0.0
    
    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        # Simplified ADX calculation
        tr = self._calculate_atr(pd.DataFrame({'high': high, 'low': low, 'close': close}))
        dx = np.abs(np.diff(high)).mean() / np.mean(tr) * 100
        return float(dx)
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using price action"""
        try:
            # Calculate directional movement
            close = df['close'].values
            direction = np.diff(close)
            
            # Count consecutive moves in same direction
            consecutive = 0
            current_direction = None
            
            for move in direction[-20:]:  # Look at last 20 periods
                if move == 0:
                    continue
                    
                new_direction = 1 if move > 0 else -1
                
                if current_direction is None:
                    current_direction = new_direction
                    consecutive = 1
                elif new_direction == current_direction:
                    consecutive += 1
                else:
                    current_direction = new_direction
                    consecutive = 1
                    
            # Normalize trend strength
            trend_strength = min(1.0, consecutive / 10)  # Cap at 10 consecutive moves
            
            return trend_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.0
    
    def _find_price_gaps(self, rates: pd.DataFrame) -> List[Dict]:
        """Detect price gaps in the market"""
        gaps = []
        for i in range(1, len(rates)):
            if rates['low'].iloc[i] > rates['high'].iloc[i-1]:  # Up gap
                gaps.append({
                    'type': 'gap_up',
                    'size': float(rates['low'].iloc[i] - rates['high'].iloc[i-1]),
                    'price': float(rates['low'].iloc[i])
                })
            elif rates['high'].iloc[i] < rates['low'].iloc[i-1]:  # Down gap
                gaps.append({
                    'type': 'gap_down',
                    'size': float(rates['low'].iloc[i-1] - rates['high'].iloc[i]),
                    'price': float(rates['high'].iloc[i])
                })
        return gaps
    
    def _find_liquidity_pools(self, rates: pd.DataFrame) -> List[Dict]:
        """Detect potential liquidity pools"""
        pools = []
        volume = rates['tick_volume'].values
        high = rates['high'].values
        low = rates['low'].values
        
        # Look for areas with high volume accumulation
        for i in range(len(rates)-5):
            if np.mean(volume[i:i+5]) > np.mean(volume) * 1.5:
                pools.append({
                    'type': 'liquidity_pool',
                    'price_level': float(np.mean([high[i:i+5].max(), low[i:i+5].min()])),
                    'volume': float(np.sum(volume[i:i+5]))
                })
        return pools