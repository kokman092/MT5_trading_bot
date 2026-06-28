import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class OptimizationResult:
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    regime_alignment: float

class RegimePortfolioOptimizer:
    """Portfolio optimizer with regime awareness"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.lookback_period = config.get('lookback_period', 100)
        
    async def optimize(self, performance_metrics: Dict[str, Dict[str, float]]) -> Dict:
        """
        Optimize portfolio weights based on strategy performance and market regime
        
        Args:
            performance_metrics: Dict of strategy performance metrics
            
        Returns:
            Dict containing optimized weights and portfolio metrics
        """
        try:
            # Extract strategy metrics
            returns = np.array([metrics['total_profit'] for metrics in performance_metrics.values()])
            volatilities = np.array([metrics['max_drawdown'] for metrics in performance_metrics.values()])
            win_rates = np.array([metrics['win_rate'] for metrics in performance_metrics.values()])
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(returns)
            
            # Calculate regime alignment score
            regime_alignment = np.mean(win_rates)
            
            # Calculate optimal weights using risk-adjusted returns
            sharpe_ratios = (returns - self.risk_free_rate) / volatilities
            total_sharpe = np.sum(np.abs(sharpe_ratios))
            
            if total_sharpe == 0:
                # Equal weights if no clear signal
                weights = {strategy: 1.0/len(performance_metrics) 
                         for strategy in performance_metrics.keys()}
            else:
                # Weight proportional to Sharpe ratio
                raw_weights = sharpe_ratios / total_sharpe
                weights = {strategy: max(0.0, weight) 
                          for strategy, weight in zip(performance_metrics.keys(), raw_weights)}
                
                # Normalize weights to sum to 1
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v/total_weight for k, v in weights.items()}
                else:
                    weights = {strategy: 1.0/len(performance_metrics) 
                             for strategy in performance_metrics.keys()}
            
            # Calculate portfolio metrics
            portfolio_return = sum(w * r for w, r in zip(weights.values(), returns))
            portfolio_vol = np.sqrt(
                sum(w1 * w2 * correlation_matrix[i,j] * volatilities[i] * volatilities[j]
                    for i, (_, w1) in enumerate(weights.items())
                    for j, (_, w2) in enumerate(weights.items()))
            )
            
            portfolio_sharpe = ((portfolio_return - self.risk_free_rate) / portfolio_vol 
                              if portfolio_vol > 0 else 0)
            
            return {
                'weights': weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': portfolio_sharpe,
                'regime_alignment': regime_alignment
            }
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {str(e)}")
            # Return equal weights as fallback
            equal_weights = {strategy: 1.0/len(performance_metrics) 
                           for strategy in performance_metrics.keys()}
            return {
                'weights': equal_weights,
                'expected_return': 0.0,
                'volatility': 1.0,
                'sharpe_ratio': 0.0,
                'regime_alignment': 0.5
            }
            
    def _calculate_returns(self, market_data: Dict) -> pd.DataFrame:
        """Calculate returns from market data"""
        try:
            returns_dict = {}
            for symbol, data in market_data.items():
                if 'close' in data:
                    prices = pd.Series(data['close'])
                    returns_dict[symbol] = np.log(prices / prices.shift(1)).dropna()
            
            return pd.DataFrame(returns_dict).tail(self.lookback_period)
            
        except Exception as e:
            self.logger.error(f"Returns calculation error: {str(e)}")
            return pd.DataFrame()
            
    def _calculate_regime_returns(self, returns: pd.DataFrame, regime: str) -> pd.Series:
        """Calculate regime-adjusted expected returns"""
        try:
            base_returns = returns.mean() * 252  # Annualized returns
            
            # Regime adjustment factors
            regime_factors = {
                'STRONG_UPTREND': 1.2,
                'UPTREND': 1.1,
                'RANGING': 1.0,
                'DOWNTREND': 0.9,
                'STRONG_DOWNTREND': 0.8
            }
            
            adjustment = regime_factors.get(regime, 1.0)
            return base_returns * adjustment
            
        except Exception as e:
            self.logger.error(f"Regime returns calculation error: {str(e)}")
            return pd.Series()
            
    def _optimize_weights(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, regime: str) -> np.ndarray:
        """Optimize portfolio weights using regime-aware constraints"""
        try:
            n_assets = len(expected_returns)
            
            # Initial weights (equal weight)
            weights = np.array([1/n_assets] * n_assets)
            
            # Regime-based constraints
            max_weight = self._get_regime_max_weight(regime)
            min_weight = 0.05  # Minimum 5% allocation
            
            # Simple optimization (can be replaced with more sophisticated methods)
            for _ in range(100):  # Simple iteration-based optimization
                # Calculate gradient
                grad = self._calculate_gradient(weights, expected_returns, cov_matrix)
                
                # Update weights
                weights += 0.01 * grad
                
                # Apply constraints
                weights = np.clip(weights, min_weight, max_weight)
                weights = weights / weights.sum()  # Normalize
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Weight optimization error: {str(e)}")
            return np.array([1/n_assets] * n_assets)
            
    def _calculate_gradient(self, weights: np.ndarray, returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate gradient for optimization"""
        try:
            port_return = (returns * weights).sum()
            port_vol = np.sqrt(weights.dot(cov_matrix).dot(weights))
            
            # Gradient of Sharpe Ratio
            d_return = returns
            d_vol = (cov_matrix.dot(weights)) / port_vol
            
            gradient = (d_return * port_vol - port_return * d_vol) / (port_vol ** 2)
            return gradient
            
        except Exception as e:
            self.logger.error(f"Gradient calculation error: {str(e)}")
            return np.zeros_like(weights)
            
    def _get_regime_max_weight(self, regime: str) -> float:
        """Get maximum weight based on market regime"""
        regime_limits = {
            'STRONG_UPTREND': 0.4,    # More concentrated in strong trends
            'UPTREND': 0.35,
            'RANGING': 0.25,          # More diversified in ranging markets
            'DOWNTREND': 0.35,
            'STRONG_DOWNTREND': 0.4
        }
        return regime_limits.get(regime, 0.3)
        
    def _calculate_regime_alignment(self, weights: np.ndarray, regime: str) -> float:
        """Calculate how well the portfolio aligns with current regime"""
        try:
            regime_preferences = {
                'STRONG_UPTREND': {'concentration': 0.8, 'trend_following': 0.9},
                'UPTREND': {'concentration': 0.7, 'trend_following': 0.8},
                'RANGING': {'concentration': 0.5, 'trend_following': 0.5},
                'DOWNTREND': {'concentration': 0.7, 'trend_following': 0.8},
                'STRONG_DOWNTREND': {'concentration': 0.8, 'trend_following': 0.9}
            }
            
            prefs = regime_preferences.get(regime, {'concentration': 0.5, 'trend_following': 0.5})
            
            # Calculate concentration score (Herfindahl index)
            concentration = (weights ** 2).sum()
            concentration_score = 1 - abs(concentration - prefs['concentration'])
            
            # Simplified trend following score
            trend_score = prefs['trend_following']
            
            return (concentration_score + trend_score) / 2
            
        except Exception as e:
            self.logger.error(f"Regime alignment calculation error: {str(e)}")
            return 0.5 