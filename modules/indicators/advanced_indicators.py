import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import ta
import logging

class AdvancedIndicators:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_kama(self, data: pd.Series, n: int = 10, fast_ema: int = 2, slow_ema: int = 30) -> pd.Series:
        """
        Calculate Kaufman's Adaptive Moving Average
        """
        er = abs(data - data.shift(1)).rolling(n).sum()
        dir_movement = abs(data - data.shift(n))
        efficiency_ratio = dir_movement / er
        
        fast_alpha = 2 / (fast_ema + 1)
        slow_alpha = 2 / (slow_ema + 1)
        sc = (efficiency_ratio * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        
        kama = pd.Series(index=data.index, dtype=float)
        kama.iloc[n-1] = data.iloc[n-1]
        for i in range(n, len(data)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (data.iloc[i] - kama.iloc[i-1])
        return kama

    def calculate_dynamic_bollinger(self, data: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Dynamic Bollinger Bands using ATR for width adjustment
        """
        mid = data.rolling(window=window).mean()
        atr = ta.volatility.AverageTrueRange(
            high=data,
            low=data,
            close=data,
            window=window
        ).average_true_range()
        
        # Adjust band width based on ATR
        width_factor = atr / data.rolling(window=window).std()
        upper = mid + (width_factor * num_std * data.rolling(window=window).std())
        lower = mid - (width_factor * num_std * data.rolling(window=window).std())
        
        return upper, mid, lower

    def calculate_ichimoku(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate Ichimoku Cloud components
        """
        tenkan_window = 9
        kijun_window = 26
        senkou_span_b_window = 52
        displacement = 26

        tenkan_sen = (high.rolling(window=tenkan_window).max() + 
                     low.rolling(window=tenkan_window).min()) / 2
        
        kijun_sen = (high.rolling(window=kijun_window).max() + 
                    low.rolling(window=kijun_window).min()) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        senkou_span_b = ((high.rolling(window=senkou_span_b_window).max() + 
                         low.rolling(window=senkou_span_b_window).min()) / 2).shift(displacement)
        
        chikou_span = close.shift(-displacement)

        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }

    def calculate_volume_profile(self, price: pd.Series, volume: pd.Series, num_bins: int = 50) -> pd.DataFrame:
        """
        Calculate Volume Profile
        """
        price_bins = pd.cut(price, bins=num_bins)
        volume_profile = volume.groupby(price_bins).sum()
        return volume_profile

    def calculate_market_regime(self, data: pd.Series, adx_period: int = 14, 
                              ma_fast: int = 10, ma_slow: int = 30) -> Dict:
        """
        Enhanced market regime detection with multiple confirmations
        """
        try:
            # Create DataFrame with high, low, close for ta package
            df = pd.DataFrame({'close': data})
            df['high'] = data
            df['low'] = data
            
            # Calculate ADX
            adx_indicator = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=adx_period
            )
            adx_value = adx_indicator.adx().iloc[-1]
            plus_di = adx_indicator.adx_pos().iloc[-1]
            minus_di = adx_indicator.adx_neg().iloc[-1]
            
            # Calculate Moving Averages
            ma_fast = data.rolling(window=ma_fast).mean()
            ma_slow = data.rolling(window=ma_slow).mean()
            
            # Calculate RSI for confirmation
            rsi = ta.momentum.RSIIndicator(close=df['close']).rsi().iloc[-1]
            
            # Calculate Bollinger Bands
            bb = ta.volatility.BollingerBands(close=df['close'])
            bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            
            # Market Regime Classification with Confidence
            regime = {}
            
            if adx_value > 25:  # Strong trend
                if plus_di > minus_di and ma_fast.iloc[-1] > ma_slow.iloc[-1]:
                    regime = {
                        'regime': "STRONG_UPTREND",
                        'confidence': min((adx_value/100 + plus_di/100 + (rsi/100))/3, 0.99),
                        'volatility': float(bb_width.iloc[-1])
                    }
                elif minus_di > plus_di and ma_fast.iloc[-1] < ma_slow.iloc[-1]:
                    regime = {
                        'regime': "STRONG_DOWNTREND",
                        'confidence': min((adx_value/100 + minus_di/100 + (100-rsi)/100)/3, 0.99),
                        'volatility': float(bb_width.iloc[-1])
                    }
            elif adx_value < 20:  # Ranging market
                regime = {
                    'regime': "RANGING",
                    'confidence': min((1 - adx_value/100) * (1 - abs(50-rsi)/50), 0.99),
                    'volatility': float(bb_width.iloc[-1])
                }
            else:  # Weak trend
                if ma_fast.iloc[-1] > ma_slow.iloc[-1]:
                    regime = {
                        'regime': "WEAK_UPTREND",
                        'confidence': min((plus_di/100 + (rsi/100))/2, 0.99),
                        'volatility': float(bb_width.iloc[-1])
                    }
                else:
                    regime = {
                        'regime': "WEAK_DOWNTREND",
                        'confidence': min((minus_di/100 + (100-rsi)/100)/2, 0.99),
                        'volatility': float(bb_width.iloc[-1])
                    }
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error in market regime calculation: {str(e)}")
            return {'regime': "UNKNOWN", 'confidence': 0, 'volatility': 0}

    def calculate_vwap(self, price: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate Volume Weighted Average Price
        """
        return (price * volume).cumsum() / volume.cumsum()

    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume
        """
        return ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            atr = ta.volatility.AverageTrueRange(
                high=high,
                low=low,
                close=close,
                window=window
            ).average_true_range()
            return atr
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(index=close.index, dtype=float)
