import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from scipy.optimize import minimize
from datetime import datetime, timedelta
import cvxopt as cv
from cvxopt import matrix, solvers

class PortfolioOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Optimization parameters
        self.params = {
            'risk_free_rate': config.get('risk_free_rate', 0.02),
            'target_return': config.get('target_return', 0.15),
            'max_position_size': config.get('max_position_size', 0.3),
            'min_position_size': config.get('min_position_size', 0.01),
            'rebalance_threshold': config.get('rebalance_threshold', 0.1),
            'optimization_method': config.get('optimization_method', 'SHARPE'),  # SHARPE, MIN_VAR, BLACK_LITTERMAN
            'correlation_threshold': config.get('correlation_threshold', 0.7),
            'lookback_period': config.get('lookback_period', 252)  # Trading days
        }
        
        self.current_portfolio = {}
        self.optimal_weights = {}
        self.last_optimization = None

    def optimize_portfolio(self, assets_data: Dict[str, pd.DataFrame], 
                         current_positions: Dict,
                         market_views: Dict = None) -> Dict:
        """
        Optimize portfolio based on selected method
        """
        try:
            # Update current portfolio
            self.current_portfolio = current_positions
            
            # Calculate returns and covariance
            returns, cov_matrix = self._calculate_risk_metrics(assets_data)
            
            # Choose optimization method
            if self.params['optimization_method'] == 'SHARPE':
                weights = self._maximize_sharpe_ratio(returns, cov_matrix)
            elif self.params['optimization_method'] == 'MIN_VAR':
                weights = self._minimize_variance(cov_matrix)
            elif self.params['optimization_method'] == 'BLACK_LITTERMAN':
                weights = self._black_litterman_optimization(
                    returns, cov_matrix, market_views
                )
            else:
                raise ValueError(f"Unknown optimization method: {self.params['optimization_method']}")
            
            # Apply position constraints
            weights = self._apply_position_constraints(weights)
            
            # Calculate rebalancing needs
            rebalancing = self._calculate_rebalancing_needs(weights)
            
            self.optimal_weights = weights
            self.last_optimization = datetime.now()
            
            return {
                'weights': weights,
                'rebalancing': rebalancing,
                'metrics': self._calculate_portfolio_metrics(returns, cov_matrix, weights)
            }
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {str(e)}")
            return None

    def _calculate_risk_metrics(self, assets_data: Dict[str, pd.DataFrame]) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate returns and covariance matrix
        """
        # Calculate returns
        returns = pd.DataFrame()
        for symbol, data in assets_data.items():
            returns[symbol] = data['close'].pct_change()
            
        # Remove outliers
        returns = returns.clip(
            lower=returns.quantile(0.01),
            upper=returns.quantile(0.99)
        )
        
        # Calculate mean returns and covariance
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        return mean_returns, cov_matrix

    def _maximize_sharpe_ratio(self, returns: pd.Series, 
                             cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Maximize Sharpe Ratio using quadratic programming
        """
        n = len(returns)
        
        # Setup optimization problem
        P = matrix(cov_matrix.values)
        q = matrix(0.0, (n, 1))
        
        # Constraints
        G = matrix(0.0, (n*2, n))
        G[:n] = -np.eye(n)  # Lower bound
        G[n:] = np.eye(n)   # Upper bound
        
        h = matrix(0.0, (n*2, 1))
        h[:n] = -matrix(self.params['min_position_size'], (n, 1))
        h[n:] = matrix(self.params['max_position_size'], (n, 1))
        
        A = matrix(1.0, (1, n))
        b = matrix(1.0)
        
        # Solve optimization problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        
        if solution['status'] != 'optimal':
            raise ValueError("Could not find optimal solution")
            
        weights = np.array(solution['x']).flatten()
        return dict(zip(returns.index, weights))

    def _minimize_variance(self, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Minimize portfolio variance
        """
        n = len(cov_matrix)
        
        # Setup optimization problem
        P = matrix(cov_matrix.values)
        q = matrix(0.0, (n, 1))
        
        # Constraints
        G = matrix(0.0, (n*2, n))
        G[:n] = -np.eye(n)
        G[n:] = np.eye(n)
        
        h = matrix(0.0, (n*2, 1))
        h[:n] = -matrix(self.params['min_position_size'], (n, 1))
        h[n:] = matrix(self.params['max_position_size'], (n, 1))
        
        A = matrix(1.0, (1, n))
        b = matrix(1.0)
        
        # Solve optimization problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        
        weights = np.array(solution['x']).flatten()
        return dict(zip(cov_matrix.index, weights))

    def _black_litterman_optimization(self, returns: pd.Series, 
                                    cov_matrix: pd.DataFrame,
                                    market_views: Dict) -> Dict[str, float]:
        """
        Black-Litterman portfolio optimization
        """
        n = len(returns)
        
        # Market equilibrium returns
        market_caps = np.ones(n) / n  # Equal weight as prior
        pi = self.params['risk_free_rate'] + market_caps @ returns
        
        # Process views
        if market_views:
            P = np.zeros((len(market_views), n))
            q = np.zeros(len(market_views))
            omega = np.eye(len(market_views)) * 0.01  # View uncertainty
            
            for i, (assets, view) in enumerate(market_views.items()):
                for asset in assets:
                    P[i, returns.index.get_loc(asset)] = 1 / len(assets)
                q[i] = view
                
            # Calculate posterior returns
            tau = 0.025  # Prior uncertainty
            pi = np.array(pi).reshape(-1, 1)
            returns = pi + tau * cov_matrix @ P.T @ \
                     np.linalg.inv(P @ tau @ cov_matrix @ P.T + omega) @ \
                     (q - P @ pi)
            
        return self._maximize_sharpe_ratio(
            pd.Series(returns.flatten(), index=cov_matrix.index),
            cov_matrix
        )

    def _apply_position_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply position size constraints and correlation limits
        """
        # Ensure minimum position size
        weights = {k: v for k, v in weights.items() 
                  if v >= self.params['min_position_size']}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights

    def _calculate_rebalancing_needs(self, target_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate required position changes
        """
        rebalancing = {}
        
        # Calculate current weights
        total_value = sum(pos['value'] for pos in self.current_portfolio.values())
        current_weights = {
            pos['symbol']: pos['value'] / total_value
            for pos in self.current_portfolio.values()
        }
        
        # Calculate required changes
        for symbol in set(target_weights.keys()) | set(current_weights.keys()):
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            
            if abs(target - current) > self.params['rebalance_threshold']:
                rebalancing[symbol] = target - current
                
        return rebalancing

    def _calculate_portfolio_metrics(self, returns: pd.Series, 
                                   cov_matrix: pd.DataFrame,
                                   weights: Dict[str, float]) -> Dict:
        """
        Calculate portfolio performance metrics
        """
        weights_array = np.array([weights.get(asset, 0) for asset in returns.index])
        
        portfolio_return = np.sum(returns * weights_array)
        portfolio_vol = np.sqrt(
            weights_array.T @ cov_matrix.values @ weights_array
        )
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': (portfolio_return - self.params['risk_free_rate']) / portfolio_vol,
            'diversification_ratio': 1 / np.sum(weights_array ** 2)
        }

    def get_optimization_status(self) -> Dict:
        """
        Get current optimization status
        """
        return {
            'last_optimization': self.last_optimization,
            'current_weights': self.optimal_weights,
            'needs_rebalancing': self._needs_rebalancing()
        }

    def _needs_rebalancing(self) -> bool:
        """
        Check if portfolio needs rebalancing
        """
        if not self.last_optimization:
            return True
            
        time_since_last = datetime.now() - self.last_optimization
        if time_since_last > timedelta(days=1):  # Daily check
            return True
            
        # Check for significant deviation from optimal weights
        total_value = sum(pos['value'] for pos in self.current_portfolio.values())
        current_weights = {
            pos['symbol']: pos['value'] / total_value
            for pos in self.current_portfolio.values()
        }
        
        for symbol, target_weight in self.optimal_weights.items():
            current = current_weights.get(symbol, 0)
            if abs(target_weight - current) > self.params['rebalance_threshold']:
                return True
                
        return False
