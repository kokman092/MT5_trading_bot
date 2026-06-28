import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
from scipy import stats

class MeanReversionStrategy:
    """Mean reversion strategies based on Ernie Chan's techniques"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.params = {
            'lookback': config.get('lookback', 20),
            'entry_zscore': config.get('entry_zscore', 2.0),
            'exit_zscore': config.get('exit_zscore', 0.5),
            'stop_loss': config.get('stop_loss', 0.02),
            'profit_target': config.get('profit_target', 0.02),
            'max_holding_periods': config.get('max_holding_periods', 10),
            'min_half_life': config.get('min_half_life', 1),
            'max_half_life': config.get('max_half_life', 20)
        }
        
    def analyze_market(self, df: pd.DataFrame) -> Dict:
        """Analyze market for mean reversion opportunities"""
        try:
            results = {}
            
            # Calculate mean reversion metrics
            results['stationarity'] = self._test_stationarity(df['close'])
            results['half_life'] = self._calculate_half_life(df['close'])
            
            # Calculate trading signals
            signals = self._calculate_signals(df)
            results['signals'] = signals
            
            # Calculate entry/exit points
            opportunities = self._find_opportunities(df, signals)
            results['opportunities'] = opportunities
            
            # Calculate position sizing
            position_size = self._calculate_position_size(df, signals)
            results['position_size'] = position_size
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            return {}
            
    def _test_stationarity(self, price: pd.Series) -> Dict:
        """Test for stationarity using Augmented Dickey-Fuller test"""
        try:
            # Calculate returns
            returns = price.pct_change()
            
            # Run ADF test
            adf_result = adfuller(returns.dropna())
            
            # Calculate Hurst exponent
            hurst = self._calculate_hurst_exponent(returns)
            
            return {
                'adf_statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'is_stationary': bool(adf_result[1] < 0.05),
                'hurst_exponent': float(hurst)
            }
            
        except Exception as e:
            self.logger.error(f"Error testing stationarity: {str(e)}")
            return {}
            
    def _calculate_half_life(self, price: pd.Series) -> float:
        """Calculate mean reversion half-life"""
        try:
            # Calculate price differences
            y = price.diff().dropna()
            x = price.shift(1).dropna()
            
            # Run OLS regression
            reg = np.polyfit(x, y, deg=1)
            beta = reg[0]
            
            # Calculate half-life
            half_life = -np.log(2) / beta if beta < 0 else np.inf
            
            return float(half_life)
            
        except Exception as e:
            self.logger.error(f"Error calculating half-life: {str(e)}")
            return np.inf
            
    def _calculate_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate mean reversion signals"""
        try:
            # Calculate z-score
            returns = df['close'].pct_change()
            ma = returns.rolling(window=self.params['lookback']).mean()
            std = returns.rolling(window=self.params['lookback']).std()
            zscore = (returns - ma) / std
            
            # Calculate Bollinger Bands
            price_ma = df['close'].rolling(window=self.params['lookback']).mean()
            price_std = df['close'].rolling(window=self.params['lookback']).std()
            upper_band = price_ma + 2 * price_std
            lower_band = price_ma - 2 * price_std
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return {
                'zscore': float(zscore.iloc[-1]),
                'ma': float(price_ma.iloc[-1]),
                'upper_band': float(upper_band.iloc[-1]),
                'lower_band': float(lower_band.iloc[-1]),
                'rsi': float(rsi.iloc[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating signals: {str(e)}")
            return {}
            
    def _find_opportunities(self, df: pd.DataFrame, signals: Dict) -> Dict:
        """Find mean reversion trading opportunities"""
        try:
            current_price = df['close'].iloc[-1]
            zscore = signals['zscore']
            rsi = signals['rsi']
            
            # Entry conditions
            long_entry = (
                zscore < -self.params['entry_zscore'] and
                rsi < 30 and
                current_price < signals['lower_band']
            )
            
            short_entry = (
                zscore > self.params['entry_zscore'] and
                rsi > 70 and
                current_price > signals['upper_band']
            )
            
            # Exit conditions
            long_exit = (
                zscore > -self.params['exit_zscore'] or
                current_price > signals['ma']
            )
            
            short_exit = (
                zscore < self.params['exit_zscore'] or
                current_price < signals['ma']
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
        """Calculate position size based on Kelly criterion"""
        try:
            # Calculate historical win rate and profit ratio
            returns = df['close'].pct_change()
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Estimate win probability
            win_prob = stats.norm.cdf(abs(signals['zscore']))
            
            # Calculate profit ratio (using historical data)
            profit_ratio = abs(mean_return) / std_return
            
            # Kelly fraction
            kelly = (win_prob * profit_ratio - (1 - win_prob)) / profit_ratio
            
            # Apply fractional Kelly (more conservative)
            fractional_kelly = kelly * 0.5
            
            # Adjust for z-score magnitude
            zscore_adjustment = 1 - min(abs(signals['zscore']) / 10, 0.5)
            
            return float(max(0, min(fractional_kelly * zscore_adjustment, 1.0)))
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
            
    def _calculate_hurst_exponent(self, returns: pd.Series,
                                lags: List[int] = None) -> float:
        """Calculate Hurst exponent"""
        try:
            if lags is None:
                lags = [2, 4, 8, 16, 32]
                
            # Calculate range/standard deviation for different lags
            rs_values = []
            for lag in lags:
                # Calculate price differences
                tau = [np.std(returns[lag:] - returns[:-lag])]
                rs_values.append(np.log(tau[0]))
                
            # Calculate Hurst exponent from slope
            reg = np.polyfit(np.log(lags), rs_values, deg=1)
            hurst = reg[0] / 2
            
            return float(hurst)
            
        except Exception as e:
            self.logger.error(f"Error calculating Hurst exponent: {str(e)}")
            return 0.5
