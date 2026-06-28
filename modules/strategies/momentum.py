import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats

class MomentumStrategy:
    """Momentum strategies based on Ernie Chan's techniques"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.params = {
            'fast_ma': config.get('fast_ma', 10),
            'slow_ma': config.get('slow_ma', 30),
            'momentum_lookback': config.get('momentum_lookback', 20),
            'volatility_lookback': config.get('volatility_lookback', 20),
            'entry_threshold': config.get('entry_threshold', 0.01),
            'exit_threshold': config.get('exit_threshold', 0.005),
            'stop_loss': config.get('stop_loss', 0.02),
            'max_holding_periods': config.get('max_holding_periods', 10)
        }
        
    def analyze_market(self, df: pd.DataFrame) -> Dict:
        """Analyze market for momentum opportunities"""
        try:
            results = {}
            
            # Calculate momentum indicators
            indicators = self._calculate_indicators(df)
            results['indicators'] = indicators
            
            # Calculate trading signals
            signals = self._calculate_signals(df, indicators)
            results['signals'] = signals
            
            # Find trading opportunities
            opportunities = self._find_opportunities(df, signals)
            results['opportunities'] = opportunities
            
            # Calculate position sizing
            position_size = self._calculate_position_size(df, signals)
            results['position_size'] = position_size
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            return {}
            
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate momentum indicators"""
        try:
            # Moving averages
            fast_ma = df['close'].rolling(window=self.params['fast_ma']).mean()
            slow_ma = df['close'].rolling(window=self.params['slow_ma']).mean()
            
            # Price momentum
            momentum = df['close'].pct_change(self.params['momentum_lookback'])
            
            # Volatility
            volatility = df['close'].pct_change().rolling(
                window=self.params['volatility_lookback']
            ).std()
            
            # Rate of change
            roc = df['close'].pct_change(self.params['momentum_lookback'])
            
            # MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            
            return {
                'fast_ma': float(fast_ma.iloc[-1]),
                'slow_ma': float(slow_ma.iloc[-1]),
                'momentum': float(momentum.iloc[-1]),
                'volatility': float(volatility.iloc[-1]),
                'roc': float(roc.iloc[-1]),
                'macd': float(macd.iloc[-1]),
                'macd_signal': float(signal.iloc[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return {}
            
    def _calculate_signals(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Calculate momentum trading signals"""
        try:
            # Moving average crossover
            ma_cross = indicators['fast_ma'] > indicators['slow_ma']
            
            # Momentum signal
            momentum_signal = indicators['momentum'] > self.params['entry_threshold']
            
            # MACD signal
            macd_cross = indicators['macd'] > indicators['macd_signal']
            
            # Volume confirmation
            volume_ma = df['tick_volume'].rolling(window=20).mean()
            volume_signal = df['tick_volume'].iloc[-1] > volume_ma.iloc[-1]
            
            # Trend strength
            adx = self._calculate_adx(df)
            strong_trend = adx > 25
            
            return {
                'ma_cross': bool(ma_cross),
                'momentum': bool(momentum_signal),
                'macd_cross': bool(macd_cross),
                'volume_confirm': bool(volume_signal),
                'strong_trend': bool(strong_trend),
                'adx': float(adx)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating signals: {str(e)}")
            return {}
            
    def _find_opportunities(self, df: pd.DataFrame, signals: Dict) -> Dict:
        """Find momentum trading opportunities"""
        try:
            # Entry conditions
            long_entry = (
                signals['ma_cross'] and
                signals['momentum'] and
                signals['macd_cross'] and
                signals['volume_confirm'] and
                signals['strong_trend']
            )
            
            short_entry = (
                not signals['ma_cross'] and
                not signals['momentum'] and
                not signals['macd_cross'] and
                signals['volume_confirm'] and
                signals['strong_trend']
            )
            
            # Exit conditions
            long_exit = (
                not signals['ma_cross'] or
                not signals['momentum']
            )
            
            short_exit = (
                signals['ma_cross'] or
                signals['momentum']
            )
            
            return {
                'long_entry': bool(long_entry),
                'short_entry': bool(short_entry),
                'long_exit': bool(long_exit),
                'short_exit': bool(short_exit)
            }
            
        except Exception as e:
            self.logger.error(f"Error finding opportunities: {str(e)}")
            return {}
            
    def _calculate_position_size(self, df: pd.DataFrame, signals: Dict) -> float:
        """Calculate position size based on volatility and trend strength"""
        try:
            # Get volatility
            volatility = df['close'].pct_change().std() * np.sqrt(252)
            
            # Get trend strength
            trend_strength = signals['adx'] / 100  # Normalize ADX
            
            # Calculate base position size using volatility targeting
            target_volatility = 0.20  # 20% annual volatility target
            base_size = target_volatility / (volatility + 1e-6)
            
            # Adjust for trend strength
            adjusted_size = base_size * trend_strength
            
            # Apply maximum position limit
            max_position = 1.0
            position_size = min(adjusted_size, max_position)
            
            return float(max(0, position_size))
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
            
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index (ADX)"""
        try:
            # Calculate True Range
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            up_move = high - high.shift()
            down_move = low.shift() - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Calculate smoothed averages
            tr_smooth = tr.rolling(window=period).mean()
            plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / tr_smooth
            minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / tr_smooth
            
            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return float(adx.iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {str(e)}")
            return 0.0
