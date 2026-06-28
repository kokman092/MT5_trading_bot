import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from sklearn.mixture import GaussianMixture
from datetime import datetime

class RegimePortfolioOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.n_regimes = config.get('portfolio', {}).get('n_regimes', 3)
        self.lookback_period = config.get('portfolio', {}).get('lookback_period', 100)
        self.risk_free_rate = config.get('portfolio', {}).get('risk_free_rate', 0.02)
        self.regime_model = None
        self.current_regime = None
        
    def detect_market_regime(self, returns: pd.DataFrame) -> int:
        """Detect current market regime using Gaussian Mixture Model"""
        try:
            if self.regime_model is None:
                self.regime_model = GaussianMixture(
                    n_components=self.n_regimes,
                    random_state=42
                )
                
            # Reshape returns for GMM
            X = returns.values.reshape(-1, 1)
            
            # Fit model if not fitted
            if not hasattr(self.regime_model, 'means_'):
                self.regime_model.fit(X)
                
            # Predict regime
            regime = self.regime_model.predict(X[-1].reshape(1, -1))[0]
            self.current_regime = regime
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return 0  # Default to low volatility regime
            
    def optimize_portfolio(self, returns: pd.DataFrame, market_data: Dict) -> Dict[str, float]:
        """Optimize portfolio weights based on current market regime"""
        try:
            # Detect current regime
            regime = self.detect_market_regime(returns)
            
            # Calculate basic statistics
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Adjust optimization based on regime
            if regime == 0:  # Low volatility regime
                return self._optimize_minimum_variance(mean_returns, cov_matrix)
            elif regime == 1:  # Normal regime
                return self._optimize_maximum_sharpe(mean_returns, cov_matrix)
            else:  # High volatility regime
                return self._optimize_risk_parity(returns)
                
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {str(e)}")
            return self._get_equal_weights(returns.columns)
            
    def _optimize_minimum_variance(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Optimize for minimum variance"""
        try:
            n_assets = len(mean_returns)
            inv_cov = np.linalg.inv(cov_matrix.values)
            ones = np.ones(n_assets)
            
            # Calculate weights
            weights = np.dot(inv_cov, ones)
            weights = weights / np.sum(weights)
            
            return dict(zip(mean_returns.index, weights))
            
        except Exception as e:
            self.logger.error(f"Error in minimum variance optimization: {str(e)}")
            return self._get_equal_weights(mean_returns.index)
            
    def _optimize_maximum_sharpe(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Optimize for maximum Sharpe ratio"""
        try:
            n_assets = len(mean_returns)
            inv_cov = np.linalg.inv(cov_matrix.values)
            
            # Calculate weights
            weights = np.dot(inv_cov, (mean_returns - self.risk_free_rate))
            weights = weights / np.sum(np.abs(weights))  # Normalize weights
            
            return dict(zip(mean_returns.index, weights))
            
        except Exception as e:
            self.logger.error(f"Error in maximum Sharpe optimization: {str(e)}")
            return self._get_equal_weights(mean_returns.index)
            
    def _optimize_risk_parity(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Implement risk parity portfolio"""
        try:
            # Calculate asset volatilities
            vols = returns.std()
            
            # Inverse volatility weighting
            weights = 1 / vols
            weights = weights / np.sum(weights)
            
            return dict(zip(returns.columns, weights))
            
        except Exception as e:
            self.logger.error(f"Error in risk parity optimization: {str(e)}")
            return self._get_equal_weights(returns.columns)
            
    def _get_equal_weights(self, assets) -> Dict[str, float]:
        """Return equal weights as fallback"""
        n_assets = len(assets)
        weight = 1.0 / n_assets
        return dict(zip(assets, [weight] * n_assets))
        
    def get_regime_metrics(self) -> Dict:
        """Get current regime metrics"""
        try:
            if self.regime_model is None or self.current_regime is None:
                return {}
                
            return {
                'current_regime': self.current_regime,
                'regime_probabilities': self.regime_model.predict_proba(
                    np.array([0]).reshape(-1, 1)
                ).tolist()[0],
                'regime_means': self.regime_model.means_.flatten().tolist(),
                'regime_variances': self.regime_model.covariances_.flatten().tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting regime metrics: {str(e)}")
            return {}

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