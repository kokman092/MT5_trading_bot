import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm

class AdvancedStrategies:
    """Advanced trading strategies based on Ernest Chan's techniques"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.params = {
            'lookback': config.get('lookback', 60),
            'entry_threshold': config.get('entry_threshold', 2.0),
            'exit_threshold': config.get('exit_threshold', 0.5),
            'stop_loss': config.get('stop_loss', 0.02),
            'vol_lookback': config.get('vol_lookback', 20),
            'momentum_lookback': config.get('momentum_lookback', 10),
            'mean_reversion_lookback': config.get('mean_reversion_lookback', 20),
            'kalman_delta': config.get('kalman_delta', 1e-4),
            'min_half_life': config.get('min_half_life', 1),
            'max_half_life': config.get('max_half_life', 20),
            'hurst_lags': config.get('hurst_lags', [2, 4, 8, 16, 32])  # Added Hurst parameter
        }
        
    def analyze_market(self, df: pd.DataFrame) -> Dict:
        """Analyze market using multiple strategies"""
        try:
            results = {}
            
            # Mean reversion analysis
            results['mean_reversion'] = self._analyze_mean_reversion(df)
            
            # Momentum analysis
            results['momentum'] = self._analyze_momentum(df)
            
            # Volatility analysis
            results['volatility'] = self._analyze_volatility(df)
            
            # Regime detection
            results['regime'] = self._detect_regime(df)
            
            # Combined signals
            results['signals'] = self._combine_signals(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            return {}
            
    def _analyze_mean_reversion(self, df: pd.DataFrame) -> Dict:
        """Analyze mean reversion opportunities"""
        try:
            # Calculate returns
            returns = df['close'].pct_change()
            
            # Calculate z-score
            ma = returns.rolling(window=self.params['mean_reversion_lookback']).mean()
            std = returns.rolling(window=self.params['mean_reversion_lookback']).std()
            zscore = (returns - ma) / std
            
            # Calculate Hurst exponent
            hurst = self._calculate_hurst_exponent(returns)
            
            # Calculate half-life
            half_life = self._calculate_half_life(df['close'])
            
            # Calculate mean reversion strength
            mr_strength = self._calculate_mean_reversion_strength(returns)
            
            return {
                'zscore': float(zscore.iloc[-1]),
                'hurst': float(hurst),
                'half_life': float(half_life),
                'strength': float(mr_strength),
                'is_mean_reverting': bool(hurst < 0.5 and half_life > self.params['min_half_life'])
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing mean reversion: {str(e)}")
            return {}
            
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze momentum signals"""
        try:
            # Calculate returns
            returns = df['close'].pct_change()
            
            # Calculate momentum indicators
            mom = returns.rolling(window=self.params['momentum_lookback']).sum()
            
            # Calculate time-series momentum
            ts_mom = self._calculate_ts_momentum(df)
            
            # Calculate cross-sectional momentum
            cs_mom = self._calculate_cs_momentum(df)
            
            # Calculate momentum strength
            strength = self._calculate_momentum_strength(returns)
            
            return {
                'momentum': float(mom.iloc[-1]),
                'ts_momentum': float(ts_mom),
                'cs_momentum': float(cs_mom),
                'strength': float(strength),
                'is_trending': bool(abs(mom.iloc[-1]) > self.params['entry_threshold'])
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum: {str(e)}")
            return {}
            
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """Analyze volatility regimes"""
        try:
            # Calculate returns
            returns = df['close'].pct_change()
            
            # Calculate volatility estimators
            gk_vol = self._calculate_garman_klass_vol(df)
            rs_vol = self._calculate_rogers_satchell_vol(df)
            yang_zhang_vol = self._calculate_yang_zhang_vol(df)
            
            # Calculate volatility regime
            vol_regime = self._detect_volatility_regime(returns)
            
            return {
                'gk_volatility': float(gk_vol),
                'rs_volatility': float(rs_vol),
                'yz_volatility': float(yang_zhang_vol),
                'regime': vol_regime,
                'is_high_vol': bool(vol_regime['regime'] == 'high')
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {str(e)}")
            return {}
            
    def _detect_regime(self, df: pd.DataFrame) -> Dict:
        """Detect market regime using multiple indicators"""
        try:
            # Calculate regime indicators
            returns = df['close'].pct_change()
            
            # Trend regime
            trend = self._calculate_trend_regime(returns)
            
            # Volatility regime
            volatility = self._detect_volatility_regime(returns)
            
            # Mean reversion regime
            mean_reversion = self._detect_mean_reversion_regime(returns)
            
            # Combine regimes
            current_regime = self._combine_regimes(trend, volatility, mean_reversion)
            
            return {
                'trend': trend,
                'volatility': volatility,
                'mean_reversion': mean_reversion,
                'current_regime': current_regime
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting regime: {str(e)}")
            return {}
            
    def _calculate_garman_klass_vol(self, df: pd.DataFrame) -> float:
        """Calculate Garman-Klass volatility estimator"""
        try:
            high = np.log(df['high'])
            low = np.log(df['low'])
            open_price = np.log(df['open'])
            close = np.log(df['close'])
            
            vol = 0.5 * (high - low)**2 - (2*np.log(2)-1) * (close - open_price)**2
            vol = np.sqrt(vol.rolling(window=self.params['vol_lookback']).mean())
            
            return float(vol.iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error calculating GK volatility: {str(e)}")
            return 0.0
            
    def _calculate_rogers_satchell_vol(self, df: pd.DataFrame) -> float:
        """Calculate Rogers-Satchell volatility estimator"""
        try:
            high = np.log(df['high'])
            low = np.log(df['low'])
            open_price = np.log(df['open'])
            close = np.log(df['close'])
            
            vol = high*(high-close) + low*(low-close)
            vol = np.sqrt(vol.rolling(window=self.params['vol_lookback']).mean())
            
            return float(vol.iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error calculating RS volatility: {str(e)}")
            return 0.0
            
    def _calculate_yang_zhang_vol(self, df: pd.DataFrame) -> float:
        """Calculate Yang-Zhang volatility estimator"""
        try:
            high = np.log(df['high'])
            low = np.log(df['low'])
            open_price = np.log(df['open'])
            close = np.log(df['close'])
            
            k = 0.34 / (1.34 + (self.params['vol_lookback'] + 1) / (self.params['vol_lookback'] - 1))
            
            overnight_vol = (open_price - close.shift(1))**2
            open_vol = (high - open_price) * (high - open_price) - (low - open_price) * (low - open_price)
            close_vol = (high - close) * (high - close) - (low - close) * (low - close)
            
            vol = np.sqrt(overnight_vol.rolling(window=self.params['vol_lookback']).mean() +
                         k * open_vol.rolling(window=self.params['vol_lookback']).mean() +
                         (1-k) * close_vol.rolling(window=self.params['vol_lookback']).mean())
            
            return float(vol.iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error calculating YZ volatility: {str(e)}")
            return 0.0
            
    def _calculate_ts_momentum(self, df: pd.DataFrame) -> float:
        """Calculate time-series momentum"""
        try:
            # Calculate returns
            returns = df['close'].pct_change()
            
            # Calculate momentum score
            momentum = returns.rolling(window=self.params['momentum_lookback']).mean()
            vol = returns.rolling(window=self.params['momentum_lookback']).std()
            
            # Risk-adjusted momentum
            ts_mom = momentum / vol
            
            return float(ts_mom.iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error calculating TS momentum: {str(e)}")
            return 0.0
            
    def _calculate_cs_momentum(self, df: pd.DataFrame) -> float:
        """Calculate cross-sectional momentum"""
        try:
            # Calculate returns
            returns = df['close'].pct_change()
            
            # Calculate rolling returns
            roll_ret = returns.rolling(window=self.params['momentum_lookback']).sum()
            
            # Calculate cross-sectional z-score
            mean_ret = roll_ret.mean()
            std_ret = roll_ret.std()
            
            cs_mom = (roll_ret.iloc[-1] - mean_ret) / std_ret
            
            return float(cs_mom)
            
        except Exception as e:
            self.logger.error(f"Error calculating CS momentum: {str(e)}")
            return 0.0
            
    def _calculate_momentum_strength(self, returns: pd.Series) -> float:
        """Calculate momentum strategy strength"""
        try:
            # Calculate autocorrelation
            acf = pd.Series(returns).autocorr()
            
            # Calculate trend strength
            trend = returns.rolling(window=self.params['momentum_lookback']).mean()
            vol = returns.rolling(window=self.params['momentum_lookback']).std()
            
            strength = abs(trend.iloc[-1]) / vol.iloc[-1]
            
            # Combine metrics
            return float(strength * abs(acf))
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum strength: {str(e)}")
            return 0.0
            
    def _calculate_mean_reversion_strength(self, returns: pd.Series) -> float:
        """Calculate mean reversion strategy strength"""
        try:
            # Calculate autocorrelation
            acf = pd.Series(returns).autocorr()
            
            # Calculate mean reversion metrics
            ma = returns.rolling(window=self.params['mean_reversion_lookback']).mean()
            std = returns.rolling(window=self.params['mean_reversion_lookback']).std()
            
            # Calculate z-score
            zscore = (returns.iloc[-1] - ma.iloc[-1]) / std.iloc[-1]
            
            # Combine metrics
            strength = abs(zscore) * abs(acf)
            
            return float(strength)
            
        except Exception as e:
            self.logger.error(f"Error calculating mean reversion strength: {str(e)}")
            return 0.0
            
    def _calculate_trend_regime(self, returns: pd.Series) -> Dict:
        """Detect trend regime"""
        try:
            # Calculate trend indicators
            ma_fast = returns.rolling(window=self.params['momentum_lookback']).mean()
            ma_slow = returns.rolling(window=self.params['lookback']).mean()
            
            # Calculate trend strength
            trend_strength = abs(ma_fast.iloc[-1] / ma_slow.iloc[-1] - 1)
            
            # Determine regime
            if trend_strength > self.params['entry_threshold']:
                regime = 'strong_trend'
            elif trend_strength > self.params['exit_threshold']:
                regime = 'weak_trend'
            else:
                regime = 'no_trend'
                
            return {
                'regime': regime,
                'strength': float(trend_strength)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trend regime: {str(e)}")
            return {'regime': 'unknown', 'strength': 0.0}
            
    def _detect_volatility_regime(self, returns: pd.Series) -> Dict:
        """Detect volatility regime"""
        try:
            # Calculate volatility
            vol = returns.rolling(window=self.params['vol_lookback']).std()
            vol_ma = vol.rolling(window=self.params['lookback']).mean()
            vol_std = vol.rolling(window=self.params['lookback']).std()
            
            # Calculate z-score
            vol_zscore = (vol.iloc[-1] - vol_ma.iloc[-1]) / vol_std.iloc[-1]
            
            # Determine regime
            if vol_zscore > self.params['entry_threshold']:
                regime = 'high'
            elif vol_zscore < -self.params['entry_threshold']:
                regime = 'low'
            else:
                regime = 'normal'
                
            return {
                'regime': regime,
                'zscore': float(vol_zscore)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility regime: {str(e)}")
            return {'regime': 'unknown', 'zscore': 0.0}
            
    def _detect_mean_reversion_regime(self, returns: pd.Series) -> Dict:
        """Detect mean reversion regime"""
        try:
            # Calculate Hurst exponent
            hurst = self._calculate_hurst_exponent(returns)
            
            # Calculate half-life
            half_life = self._calculate_half_life(returns)
            
            # Determine regime
            if hurst < 0.4 and half_life > self.params['min_half_life']:
                regime = 'strong_mr'
            elif hurst < 0.5:
                regime = 'weak_mr'
            else:
                regime = 'no_mr'
                
            return {
                'regime': regime,
                'hurst': float(hurst),
                'half_life': float(half_life)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting mean reversion regime: {str(e)}")
            return {'regime': 'unknown', 'hurst': 0.5, 'half_life': np.inf}
            
    def _combine_regimes(self, trend: Dict, volatility: Dict,
                        mean_reversion: Dict) -> str:
        """Combine different regime indicators"""
        try:
            # Score each regime
            trend_score = 1 if trend['regime'] == 'strong_trend' else (
                0.5 if trend['regime'] == 'weak_trend' else 0)
                
            vol_score = 1 if volatility['regime'] == 'high' else (
                0.5 if volatility['regime'] == 'normal' else 0)
                
            mr_score = 1 if mean_reversion['regime'] == 'strong_mr' else (
                0.5 if mean_reversion['regime'] == 'weak_mr' else 0)
                
            # Determine dominant regime
            scores = {
                'trend': trend_score,
                'mean_reversion': mr_score,
                'volatility': vol_score
            }
            
            dominant_regime = max(scores.items(), key=lambda x: x[1])[0]
            
            return dominant_regime
            
        except Exception as e:
            self.logger.error(f"Error combining regimes: {str(e)}")
            return 'unknown'
            
    def _combine_signals(self, results: Dict) -> Dict:
        """Combine signals from different strategies"""
        try:
            # Get regime
            regime = results['regime']['current_regime']
            
            # Initialize signal strength
            signal_strength = 0.0
            direction = 0
            
            if regime == 'trend':
                # Use momentum signals
                mom = results['momentum']
                signal_strength = mom['strength']
                direction = np.sign(mom['momentum'])
                
            elif regime == 'mean_reversion':
                # Use mean reversion signals
                mr = results['mean_reversion']
                signal_strength = mr['strength']
                direction = -np.sign(mr['zscore'])
                
            elif regime == 'volatility':
                # Use volatility signals
                vol = results['volatility']
                signal_strength = abs(vol['gk_volatility'])
                direction = np.sign(vol['regime'] == 'high')
                
            # Calculate position size
            position_size = self._calculate_position_size(signal_strength)
            
            return {
                'regime': regime,
                'direction': float(direction),
                'strength': float(signal_strength),
                'position_size': float(position_size)
            }
            
        except Exception as e:
            self.logger.error(f"Error combining signals: {str(e)}")
            return {}
            
    def _calculate_position_size(self, signal_strength: float) -> float:
        """Calculate position size based on signal strength"""
        try:
            # Apply sigmoid function for smooth scaling
            scaled_strength = 1 / (1 + np.exp(-2 * signal_strength))
            
            # Apply maximum position limit
            max_position = 1.0
            position_size = scaled_strength * max_position
            
            return float(position_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
            
    def _calculate_half_life(self, price: pd.Series) -> float:
        """Calculate mean reversion half-life"""
        try:
            if len(price) < 2:
                return np.inf
                
            # Calculate price differences
            y = price.diff().dropna()
            x = price.shift(1).dropna()
            
            if len(y) < 2 or len(x) < 2:
                return np.inf
                
            # Run OLS regression
            reg = np.polyfit(x, y, deg=1)
            beta = reg[0]
            
            # Calculate half-life
            if beta >= 0:  # Not mean-reverting
                return np.inf
                
            half_life = -np.log(2) / beta
            
            # Validate half-life
            if not np.isfinite(half_life) or half_life <= 0:
                return np.inf
                
            return float(half_life)
            
        except Exception as e:
            self.logger.error(f"Error calculating half-life: {str(e)}")
            return np.inf
            
    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent"""
        try:
            if len(returns) < max(self.params['hurst_lags']):
                return 0.5
                
            # Calculate range/standard deviation for different lags
            rs_values = []
            
            for lag in self.params['hurst_lags']:
                # Calculate price differences
                tau = [np.std(returns[lag:] - returns[:-lag])]
                if np.isfinite(tau[0]) and tau[0] > 0:
                    rs_values.append(np.log(tau[0]))
                    
            if not rs_values:
                return 0.5
                
            # Calculate Hurst exponent from slope
            x = np.log(self.params['hurst_lags'][:len(rs_values)])
            reg = np.polyfit(x, rs_values, deg=1)
            hurst = reg[0] / 2
            
            # Validate Hurst exponent
            if not np.isfinite(hurst):
                return 0.5
                
            return max(0, min(1, float(hurst)))  # Bound between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating Hurst exponent: {str(e)}")
            return 0.5
