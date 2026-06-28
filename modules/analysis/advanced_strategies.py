import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import MetaTrader5 as mt5
from scipy.signal import argrelextrema

class AdvancedStrategies:
    def __init__(self, config: Dict):
        self.config = config
        
    def analyze_multi_timeframe(self, symbol: str, timeframes: List[str]) -> Dict:
        """Analyze multiple timeframes for trend alignment"""
        results = {}
        for tf in timeframes:
            # Get data for each timeframe
            rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, f'TIMEFRAME_{tf}'), 0, 100)
            if rates is not None:
                df = pd.DataFrame(rates)
                results[tf] = self._analyze_single_timeframe(df)
        return self._combine_timeframe_analysis(results)
    
    def volume_profile_analysis(self, rates: pd.DataFrame, price_levels: int = 50) -> Dict:
        """Analyze volume distribution across price levels"""
        # Create price levels
        price_range = np.linspace(rates['low'].min(), rates['high'].max(), price_levels)
        volume_profile = np.zeros(price_levels - 1)
        
        # Calculate volume for each price level
        for i in range(len(price_range) - 1):
            mask = (rates['low'] >= price_range[i]) & (rates['high'] < price_range[i + 1])
            volume_profile[i] = rates.loc[mask, 'tick_volume'].sum()
            
        # Find high volume nodes
        high_volume_nodes = argrelextrema(volume_profile, np.greater)[0]
        
        return {
            'volume_profile': volume_profile.tolist(),
            'price_levels': price_range.tolist(),
            'high_volume_nodes': high_volume_nodes.tolist()
        }
    
    def market_structure_analysis(self, rates: pd.DataFrame) -> Dict:
        """Analyze market structure (swing highs/lows, support/resistance)"""
        # Find swing points
        highs = self._find_swing_points(rates['high'].values, 'high')
        lows = self._find_swing_points(rates['low'].values, 'low')
        
        # Identify support and resistance
        support_resistance = self._identify_support_resistance(rates, highs, lows)
        
        return {
            'swing_highs': highs,
            'swing_lows': lows,
            'support_levels': support_resistance['support'],
            'resistance_levels': support_resistance['resistance']
        }
    
    def order_flow_analysis(self, symbol: str) -> Dict:
        """Analyze order flow and market depth"""
        book = mt5.market_book_get(symbol)
        if book is None:
            return {}
            
        asks = [{'price': item.price, 'volume': item.volume} for item in book if item.type == 1]
        bids = [{'price': item.price, 'volume': item.volume} for item in book if item.type == 2]
        
        return {
            'asks': asks,
            'bids': bids,
            'imbalance': sum(b['volume'] for b in bids) - sum(a['volume'] for a in asks)
        }
    
    def volatility_regime_analysis(self, rates: pd.DataFrame) -> Dict:
        """Analyze volatility regimes using multiple methods"""
        # Calculate various volatility measures
        atr = self._calculate_atr(rates)
        std_dev = rates['close'].rolling(window=20).std()
        
        # Identify volatility regime
        current_vol = atr.iloc[-1]
        avg_vol = atr.mean()
        
        if current_vol > avg_vol * 1.5:
            regime = 'high'
        elif current_vol < avg_vol * 0.5:
            regime = 'low'
        else:
            regime = 'normal'
            
        return {
            'regime': regime,
            'current_volatility': float(current_vol),
            'average_volatility': float(avg_vol),
            'std_dev': float(std_dev.iloc[-1])
        }
    
    def _analyze_single_timeframe(self, df: pd.DataFrame) -> Dict:
        """Analyze a single timeframe for trend and momentum"""
        close = df['close'].values
        
        # Calculate EMAs
        ema20 = pd.Series(close).ewm(span=20).mean()
        ema50 = pd.Series(close).ewm(span=50).mean()
        
        # Determine trend
        trend = 'up' if ema20.iloc[-1] > ema50.iloc[-1] else 'down'
        
        # Calculate momentum
        momentum = (close[-1] / close[-20] - 1) * 100
        
        return {
            'trend': trend,
            'momentum': float(momentum),
            'ema20': float(ema20.iloc[-1]),
            'ema50': float(ema50.iloc[-1])
        }
    
    def _combine_timeframe_analysis(self, results: Dict) -> Dict:
        """Combine analysis from multiple timeframes"""
        trend_alignment = 0
        total_timeframes = len(results)
        
        # Check trend alignment across timeframes
        up_trends = sum(1 for tf in results.values() if tf['trend'] == 'up')
        trend_alignment = up_trends / total_timeframes
        
        # Determine overall bias
        if trend_alignment > 0.7:
            bias = 'strong_bullish'
        elif trend_alignment > 0.5:
            bias = 'bullish'
        elif trend_alignment < 0.3:
            bias = 'strong_bearish'
        elif trend_alignment < 0.5:
            bias = 'bearish'
        else:
            bias = 'neutral'
            
        return {
            'bias': bias,
            'trend_alignment': float(trend_alignment),
            'timeframe_results': results
        }
    
    def _find_swing_points(self, data: np.ndarray, point_type: str, window: int = 5) -> List[Dict]:
        """Find swing highs or lows in price data"""
        points = []
        for i in range(window, len(data) - window):
            if point_type == 'high':
                if data[i] == max(data[i-window:i+window+1]):
                    points.append({'index': i, 'value': float(data[i])})
            else:
                if data[i] == min(data[i-window:i+window+1]):
                    points.append({'index': i, 'value': float(data[i])})
        return points
    
    def _identify_support_resistance(self, rates: pd.DataFrame, highs: List[Dict], lows: List[Dict]) -> Dict:
        """Identify support and resistance levels"""
        def cluster_levels(levels, tolerance=0.0002):
            clustered = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - current_cluster[0]) / current_cluster[0] <= tolerance:
                    current_cluster.append(level)
                else:
                    clustered.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [level]
                    
            if current_cluster:
                clustered.append(sum(current_cluster) / len(current_cluster))
            return clustered
        
        # Extract price levels
        resistance_levels = [h['value'] for h in highs]
        support_levels = [l['value'] for l in lows]
        
        # Cluster nearby levels
        resistance = cluster_levels(resistance_levels)
        support = cluster_levels(support_levels)
        
        return {
            'resistance': resistance,
            'support': support
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift()
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
