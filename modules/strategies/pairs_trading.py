import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from scipy import stats

class PairsTrading:
    """Pairs trading strategies based on Ernie Chan's techniques"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.params = {
            'lookback': config.get('lookback', 60),
            'entry_zscore': config.get('entry_zscore', 2.0),
            'exit_zscore': config.get('exit_zscore', 0.5),
            'stop_loss': config.get('stop_loss', 0.02),
            'max_holding_periods': config.get('max_holding_periods', 20),
            'min_correlation': config.get('min_correlation', 0.8),
            'min_cointegration': config.get('min_cointegration', 0.05),
            'kalman_delta': config.get('kalman_delta', 1e-4)
        }
        
    def analyze_pairs(self, pairs_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze pairs for trading opportunities"""
        try:
            results = {}
            
            # Analyze each pair
            for symbol, df in pairs_data.items():
                # Calculate pair metrics
                metrics = self._calculate_pair_metrics(df)
                
                # Calculate trading signals
                signals = self._calculate_signals(df)
                
                # Find trading opportunities
                opportunities = self._find_opportunities(df, signals)
                
                # Calculate position sizing
                position_size = self._calculate_position_size(df, signals)
                
                results[symbol] = {
                    'metrics': metrics,
                    'signals': signals,
                    'opportunities': opportunities,
                    'position_size': position_size
                }
                
            # Cross-pair analysis
            results['cross_pair'] = self._analyze_cross_pairs(pairs_data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing pairs: {str(e)}")
            return {}
            
    def _calculate_pair_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate pair trading metrics"""
        try:
            # Calculate price ratio
            price_ratio = df['close'] / df['close'].shift(1)
            
            # Calculate correlation
            correlation = df['close'].rolling(
                window=self.params['lookback']
            ).corr(df['close'].shift(1))
            
            # Test for cointegration
            coint_result = self._test_cointegration(df['close'], df['close'].shift(1))
            
            # Calculate hedge ratio using Kalman filter
            hedge_ratio = self._kalman_filter_hedge_ratio(
                df['close'],
                df['close'].shift(1)
            )
            
            return {
                'price_ratio': float(price_ratio.iloc[-1]),
                'correlation': float(correlation.iloc[-1]),
                'cointegration': coint_result,
                'hedge_ratio': float(hedge_ratio[-1]) if len(hedge_ratio) > 0 else 1.0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating pair metrics: {str(e)}")
            return {}
            
    def _calculate_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate pair trading signals"""
        try:
            # Calculate spread
            spread = self._calculate_spread(df)
            
            # Calculate z-score
            zscore = self._calculate_zscore(spread)
            
            # Calculate spread momentum
            momentum = spread.diff(self.params['lookback'])
            
            # Calculate spread volatility
            volatility = spread.rolling(window=self.params['lookback']).std()
            
            return {
                'spread': float(spread.iloc[-1]),
                'zscore': float(zscore.iloc[-1]),
                'momentum': float(momentum.iloc[-1]),
                'volatility': float(volatility.iloc[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating signals: {str(e)}")
            return {}
            
    def _find_opportunities(self, df: pd.DataFrame, signals: Dict) -> Dict:
        """Find pair trading opportunities"""
        try:
            zscore = signals['zscore']
            momentum = signals['momentum']
            
            # Entry conditions
            long_spread = (
                zscore < -self.params['entry_zscore'] and
                momentum > 0
            )
            
            short_spread = (
                zscore > self.params['entry_zscore'] and
                momentum < 0
            )
            
            # Exit conditions
            long_exit = zscore > -self.params['exit_zscore']
            short_exit = zscore < self.params['exit_zscore']
            
            return {
                'long_spread': bool(long_spread),
                'short_spread': bool(short_spread),
                'long_exit': bool(long_exit),
                'short_exit': bool(short_exit)
            }
            
        except Exception as e:
            self.logger.error(f"Error finding opportunities: {str(e)}")
            return {}
            
    def _calculate_position_size(self, df: pd.DataFrame, signals: Dict) -> Dict:
        """Calculate position sizes for both legs"""
        try:
            # Get current prices
            price1 = df['close'].iloc[-1]
            price2 = df['close'].shift(1).iloc[-1]
            
            # Get hedge ratio
            hedge_ratio = self._kalman_filter_hedge_ratio(
                df['close'],
                df['close'].shift(1)
            )[-1]
            
            # Calculate notional value ratio
            value_ratio = price1 / price2
            
            # Base position size on z-score
            zscore = abs(signals['zscore'])
            base_size = min(zscore / self.params['entry_zscore'], 1.0)
            
            # Calculate individual position sizes
            leg1_size = base_size
            leg2_size = base_size * hedge_ratio * value_ratio
            
            return {
                'leg1_size': float(leg1_size),
                'leg2_size': float(leg2_size),
                'hedge_ratio': float(hedge_ratio)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return {'leg1_size': 0.0, 'leg2_size': 0.0, 'hedge_ratio': 1.0}
            
    def _test_cointegration(self, series1: pd.Series,
                           series2: pd.Series) -> Dict:
        """Test for cointegration between two price series"""
        try:
            # Run cointegration test
            score, pvalue, _ = coint(series1, series2)
            
            # Run ADF test on spread
            spread = series1 - series2
            adf_result = adfuller(spread)
            
            return {
                'coint_score': float(score),
                'coint_pvalue': float(pvalue),
                'adf_statistic': float(adf_result[0]),
                'adf_pvalue': float(adf_result[1]),
                'is_cointegrated': bool(pvalue < self.params['min_cointegration'])
            }
            
        except Exception as e:
            self.logger.error(f"Error testing cointegration: {str(e)}")
            return {}
            
    def _kalman_filter_hedge_ratio(self, series1: pd.Series,
                                 series2: pd.Series) -> np.ndarray:
        """Calculate dynamic hedge ratio using Kalman filter"""
        try:
            # Initialize state
            state_mean = 0
            state_var = 1
            
            # Initialize measurement noise
            measurement_var = 1
            
            # Storage for hedge ratios
            hedge_ratios = []
            
            for t in range(len(series1)):
                # Prediction
                state_var = state_var + self.params['kalman_delta']
                
                # Update
                measurement = series2.iloc[t]
                kalman_gain = state_var / (state_var + measurement_var)
                state_mean = state_mean + kalman_gain * (series1.iloc[t] - measurement * state_mean)
                state_var = (1 - kalman_gain) * state_var
                
                hedge_ratios.append(state_mean)
                
            return np.array(hedge_ratios)
            
        except Exception as e:
            self.logger.error(f"Error calculating Kalman filter: {str(e)}")
            return np.array([1.0])
            
    def _calculate_spread(self, df: pd.DataFrame) -> pd.Series:
        """Calculate spread between pairs"""
        try:
            # Get hedge ratio
            hedge_ratio = self._kalman_filter_hedge_ratio(
                df['close'],
                df['close'].shift(1)
            )
            
            # Calculate spread
            spread = df['close'] - hedge_ratio[-1] * df['close'].shift(1)
            
            return spread
            
        except Exception as e:
            self.logger.error(f"Error calculating spread: {str(e)}")
            return pd.Series(0, index=df.index)
            
    def _calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """Calculate z-score of spread"""
        try:
            # Calculate rolling stats
            mean = spread.rolling(window=self.params['lookback']).mean()
            std = spread.rolling(window=self.params['lookback']).std()
            
            # Calculate z-score
            zscore = (spread - mean) / std
            
            return zscore
            
        except Exception as e:
            self.logger.error(f"Error calculating zscore: {str(e)}")
            return pd.Series(0, index=spread.index)
            
    def _analyze_cross_pairs(self, pairs_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze relationships between multiple pairs"""
        try:
            results = {}
            symbols = list(pairs_data.keys())
            
            # Calculate correlation matrix
            prices = pd.DataFrame({sym: data['close'] 
                                 for sym, data in pairs_data.items()})
            correlation_matrix = prices.corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > self.params['min_correlation']:
                        high_corr_pairs.append({
                            'pair': (symbols[i], symbols[j]),
                            'correlation': float(corr)
                        })
                        
            # Test cointegration for correlated pairs
            cointegrated_pairs = []
            for pair in high_corr_pairs:
                sym1, sym2 = pair['pair']
                coint_result = self._test_cointegration(
                    pairs_data[sym1]['close'],
                    pairs_data[sym2]['close']
                )
                if coint_result.get('is_cointegrated', False):
                    cointegrated_pairs.append({
                        'pair': (sym1, sym2),
                        'correlation': pair['correlation'],
                        'cointegration': coint_result
                    })
                    
            results['correlation_matrix'] = correlation_matrix.to_dict()
            results['high_correlation_pairs'] = high_corr_pairs
            results['cointegrated_pairs'] = cointegrated_pairs
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing cross pairs: {str(e)}")
            return {}
