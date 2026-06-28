import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
import MetaTrader5 as mt5
from scipy import stats

class StatisticalMeanReversion:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.lookback_period = 20  # Default lookback period for z-score calculation
        self.zscore_threshold = 2.0  # Default z-score threshold for entry
        self.half_life = 10  # Default half-life for mean reversion
        
    def calculate_hurst_exponent(self, prices: np.array, max_lag: int = 20) -> float:
        """
        Calculate Hurst exponent to determine mean reversion tendency
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]  # Hurst exponent is the slope
        
    def calculate_half_life(self, prices: np.array) -> float:
        """Calculate half-life of mean reversion"""
        lag_1 = prices[1:]
        lag_0 = prices[:-1]
        ret = pd.Series(lag_1) / pd.Series(lag_0) - 1
        lag_ret = ret.shift(1)
        ret = ret[1:]
        lag_ret = lag_ret[1:]
        reg = np.polyfit(lag_ret, ret, 1)
        return -np.log(2) / reg[0]
        
    def calculate_zscore(self, price: float, prices: np.array) -> float:
        """Calculate z-score for mean reversion entry"""
        mean = np.mean(prices)
        std = np.std(prices)
        return (price - mean) / std if std > 0 else 0
        
    def calculate_kalman_filter(self, prices: np.array) -> Tuple[np.array, np.array]:
        """
        Implement Kalman filter for adaptive mean and variance estimation
        Returns: (filtered_mean, filtered_variance)
        """
        n = len(prices)
        filtered_mean = np.zeros(n)
        filtered_variance = np.zeros(n)
        
        # Initialize
        filtered_mean[0] = prices[0]
        filtered_variance[0] = 1
        
        # Kalman filter parameters
        transition_covariance = 0.01
        observation_covariance = 1
        
        for t in range(1, n):
            # Predict
            predicted_mean = filtered_mean[t-1]
            predicted_variance = filtered_variance[t-1] + transition_covariance
            
            # Update
            kalman_gain = predicted_variance / (predicted_variance + observation_covariance)
            filtered_mean[t] = predicted_mean + kalman_gain * (prices[t] - predicted_mean)
            filtered_variance[t] = (1 - kalman_gain) * predicted_variance
            
        return filtered_mean, filtered_variance
        
    def analyze_cointegration(self, prices1: np.array, prices2: np.array) -> Dict:
        """
        Analyze cointegration between two price series
        Returns cointegration statistics and trading parameters
        """
        # Perform ADF test on spread
        spread = np.log(prices1) - np.log(prices2)
        adf_result = stats.adfuller(spread)
        
        # Calculate optimal hedge ratio using OLS
        log_prices1 = np.log(prices1)
        log_prices2 = np.log(prices2)
        hedge_ratio = np.polyfit(log_prices2, log_prices1, 1)[0]
        
        return {
            'hedge_ratio': hedge_ratio,
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'is_cointegrated': adf_result[1] < 0.05
        }
        
    def generate_signal(self, symbol: str) -> Optional[Dict]:
        """Generate trading signals based on statistical analysis"""
        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
            if rates is None:
                return None
                
            prices = np.array([rate['close'] for rate in rates])
            
            # Calculate Hurst exponent
            hurst = self.calculate_hurst_exponent(prices)
            
            # Only proceed if market shows mean reversion tendency
            if hurst >= 0.5:
                return None
                
            # Calculate adaptive mean and variance using Kalman filter
            filtered_mean, filtered_variance = self.calculate_kalman_filter(prices)
            current_price = prices[-1]
            
            # Calculate z-score using adaptive statistics
            zscore = (current_price - filtered_mean[-1]) / np.sqrt(filtered_variance[-1])
            
            # Calculate optimal half-life
            half_life = self.calculate_half_life(prices)
            
            # Generate signals based on z-score
            signal = None
            if zscore < -self.zscore_threshold:
                signal = {
                    'type': 'buy',
                    'confidence': min(abs(zscore) / 4, 1.0),  # Normalize confidence
                    'entry_price': current_price,
                    'target_price': filtered_mean[-1],
                    'stop_loss': current_price * (1 - 0.01),  # 1% stop loss
                    'metrics': {
                        'zscore': zscore,
                        'hurst': hurst,
                        'half_life': half_life
                    }
                }
            elif zscore > self.zscore_threshold:
                signal = {
                    'type': 'sell',
                    'confidence': min(abs(zscore) / 4, 1.0),
                    'entry_price': current_price,
                    'target_price': filtered_mean[-1],
                    'stop_loss': current_price * (1 + 0.01),
                    'metrics': {
                        'zscore': zscore,
                        'hurst': hurst,
                        'half_life': half_life
                    }
                }
                
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {str(e)}")
            return None
