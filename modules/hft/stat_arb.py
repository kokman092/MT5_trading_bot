import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.stattools import coint
from ..ml.feature_engineering import FinancialFeatureEngineering

class StatisticalArbitrage:
    """Statistical arbitrage strategies based on Aldridge's techniques"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_engineering = FinancialFeatureEngineering(config)
        
        # Strategy parameters
        self.params = {
            'lookback_window': config.get('lookback_window', 100),
            'entry_threshold': config.get('entry_threshold', 2.0),
            'exit_threshold': config.get('exit_threshold', 0.5),
            'max_position': config.get('max_position', 1.0),
            'stop_loss': config.get('stop_loss', 3.0),
            'profit_target': config.get('profit_target', 2.0)
        }
        
    async def analyze_pairs(self, pairs_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze pairs for statistical arbitrage"""
        try:
            results = {}
            
            # Analyze each pair
            for symbol, df in pairs_data.items():
                # Engineer features
                features = self.feature_engineering.engineer_features(df)
                
                # Calculate pair-specific signals
                signals = self._calculate_pair_signals(features)
                
                # Get trading opportunities
                opportunities = self._find_opportunities(features, signals)
                
                results[symbol] = {
                    'signals': signals,
                    'opportunities': opportunities,
                    'features': features
                }
                
            # Cross-pair analysis
            cointegration = self._analyze_cointegration(pairs_data)
            
            return {
                'pair_results': results,
                'cointegration': cointegration
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing pairs: {str(e)}")
            return {}
            
    def _calculate_pair_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate signals for a single pair"""
        try:
            signals = {}
            
            # Mean reversion signals
            signals['zscore'] = self._calculate_zscore(df)
            
            # Correlation breakdown
            signals['correlation'] = self._calculate_correlation_signals(df)
            
            # Volatility regime
            signals['volatility'] = self._calculate_volatility_signals(df)
            
            # Liquidity signals
            signals['liquidity'] = self._calculate_liquidity_signals(df)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error calculating pair signals: {str(e)}")
            return {}
            
    def _find_opportunities(self, df: pd.DataFrame, signals: Dict) -> Dict:
        """Find trading opportunities"""
        try:
            current_zscore = signals['zscore']['current']
            vol_regime = signals['volatility']['regime']
            liquidity = signals['liquidity']['score']
            
            # Entry signals
            long_entry = (
                current_zscore < -self.params['entry_threshold'] and
                vol_regime < 1.5 and
                liquidity > 0.7
            )
            
            short_entry = (
                current_zscore > self.params['entry_threshold'] and
                vol_regime < 1.5 and
                liquidity > 0.7
            )
            
            # Exit signals
            long_exit = current_zscore > -self.params['exit_threshold']
            short_exit = current_zscore < self.params['exit_threshold']
            
            # Position sizing
            position_size = self._calculate_position_size(
                zscore=current_zscore,
                volatility=vol_regime,
                liquidity=liquidity
            )
            
            return {
                'long_entry': bool(long_entry),
                'short_entry': bool(short_entry),
                'long_exit': bool(long_exit),
                'short_exit': bool(short_exit),
                'position_size': float(position_size),
                'confidence': float(min(1.0, 1.0 / abs(current_zscore)))
            }
            
        except Exception as e:
            self.logger.error(f"Error finding opportunities: {str(e)}")
            return {}
            
    def _analyze_cointegration(self, pairs_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze cointegration between pairs"""
        try:
            results = {}
            symbols = list(pairs_data.keys())
            
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    sym1, sym2 = symbols[i], symbols[j]
                    
                    # Get price series
                    price1 = pairs_data[sym1]['close']
                    price2 = pairs_data[sym2]['close']
                    
                    # Calculate cointegration
                    score, pvalue, _ = coint(price1, price2)
                    
                    # Calculate hedge ratio
                    hedge_ratio = self._calculate_hedge_ratio(price1, price2)
                    
                    results[f"{sym1}_{sym2}"] = {
                        'score': float(score),
                        'pvalue': float(pvalue),
                        'hedge_ratio': float(hedge_ratio),
                        'is_cointegrated': bool(pvalue < 0.05)
                    }
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing cointegration: {str(e)}")
            return {}
            
    def _calculate_zscore(self, df: pd.DataFrame) -> Dict:
        """Calculate z-score for mean reversion"""
        try:
            # Calculate spread
            spread = df['close'] - df['close'].shift(1)
            
            # Calculate rolling stats
            mean = spread.rolling(window=self.params['lookback_window']).mean()
            std = spread.rolling(window=self.params['lookback_window']).std()
            
            # Calculate z-score
            zscore = (spread - mean) / std
            
            return {
                'current': float(zscore.iloc[-1]),
                'mean': float(mean.iloc[-1]),
                'std': float(std.iloc[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating zscore: {str(e)}")
            return {'current': 0.0, 'mean': 0.0, 'std': 1.0}
            
    def _calculate_correlation_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate correlation-based signals"""
        try:
            # Rolling correlation
            returns = df['close'].pct_change()
            corr = returns.rolling(window=self.params['lookback_window']).corr(
                returns.shift(1)
            )
            
            # Correlation change
            corr_change = corr.diff()
            
            return {
                'current': float(corr.iloc[-1]),
                'change': float(corr_change.iloc[-1]),
                'breakdown': bool(corr_change.iloc[-1] < -0.2)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation signals: {str(e)}")
            return {'current': 0.0, 'change': 0.0, 'breakdown': False}
            
    def _calculate_volatility_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility-based signals"""
        try:
            # Use Yang-Zhang volatility
            current_vol = df['yang_zhang_vol'].iloc[-1]
            avg_vol = df['yang_zhang_vol'].rolling(window=100).mean().iloc[-1]
            
            # Volatility regime
            vol_regime = current_vol / avg_vol
            
            return {
                'current': float(current_vol),
                'average': float(avg_vol),
                'regime': float(vol_regime)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility signals: {str(e)}")
            return {'current': 0.0, 'average': 0.0, 'regime': 1.0}
            
    def _calculate_liquidity_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate liquidity signals"""
        try:
            # Use multiple liquidity measures
            spread = df['roll_spread'].iloc[-1]
            impact = df['kyle_lambda'].iloc[-1]
            volume = df['tick_volume'].iloc[-1]
            
            # Normalize measures
            avg_spread = df['roll_spread'].mean()
            avg_impact = df['kyle_lambda'].mean()
            avg_volume = df['tick_volume'].mean()
            
            spread_score = 1 - min(spread / avg_spread, 2) / 2
            impact_score = 1 - min(impact / avg_impact, 2) / 2
            volume_score = min(volume / avg_volume, 2) / 2
            
            # Combined liquidity score
            liquidity_score = (spread_score + impact_score + volume_score) / 3
            
            return {
                'score': float(liquidity_score),
                'spread': float(spread_score),
                'impact': float(impact_score),
                'volume': float(volume_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity signals: {str(e)}")
            return {'score': 0.0, 'spread': 0.0, 'impact': 0.0, 'volume': 0.0}
            
    def _calculate_position_size(self, zscore: float,
                               volatility: float,
                               liquidity: float) -> float:
        """Calculate optimal position size"""
        try:
            # Base size on z-score distance
            base_size = (abs(zscore) - self.params['entry_threshold']) / (
                self.params['stop_loss'] - self.params['entry_threshold']
            )
            
            # Adjust for volatility
            vol_adjustment = 1 / volatility if volatility > 0 else 1
            
            # Adjust for liquidity
            liq_adjustment = liquidity
            
            # Calculate final size
            position_size = self.params['max_position'] * base_size * \
                          vol_adjustment * liq_adjustment
            
            return max(0.0, min(position_size, self.params['max_position']))
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
            
    def _calculate_hedge_ratio(self, price1: pd.Series,
                             price2: pd.Series) -> float:
        """Calculate hedge ratio between two price series"""
        try:
            # Use rolling OLS regression
            X = sm.add_constant(price1)
            model = sm.OLS(price2, X).fit()
            return model.params[1]
            
        except Exception as e:
            self.logger.error(f"Error calculating hedge ratio: {str(e)}")
            return 1.0
