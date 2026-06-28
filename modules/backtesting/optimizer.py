import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import itertools
from concurrent.futures import ProcessPoolExecutor
from .backtest_engine import BacktestEngine
from ..strategies.base_strategy import BaseStrategy
from sklearn.model_selection import TimeSeriesSplit

class StrategyOptimizer:
    def __init__(self, config: Dict, strategy_class: Type[BaseStrategy]):
        self.config = config
        self.strategy_class = strategy_class
        self.param_ranges = {
            'trend': {
                'rsi_period': range(10, 21, 2),          # [10, 12, 14, 16, 18, 20]
                'macd_fast': range(8, 15, 2),            # [8, 10, 12, 14]
                'macd_slow': range(20, 31, 2),           # [20, 22, 24, 26, 28, 30]
                'macd_signal': range(7, 12, 2),          # [7, 9, 11]
                'adx_threshold': range(20, 36, 5),       # [20, 25, 30, 35]
                'atr_multiplier_sl': np.arange(1.5, 3.1, 0.5),  # [1.5, 2.0, 2.5, 3.0]
                'atr_multiplier_tp': np.arange(2.0, 4.1, 0.5)   # [2.0, 2.5, 3.0, 3.5, 4.0]
            },
            'scalping': {
                'rsi_period': range(7, 16, 2),           # [7, 9, 11, 13, 15]
                'rsi_overbought': range(70, 81, 5),      # [70, 75, 80]
                'rsi_oversold': range(20, 31, 5),        # [20, 25, 30]
                'macd_fast': range(6, 13, 2),            # [6, 8, 10, 12]
                'macd_slow': range(14, 25, 2),           # [14, 16, 18, 20, 22, 24]
                'macd_signal': range(5, 10, 2),          # [5, 7, 9]
                'min_risk_reward': np.arange(1.2, 2.1, 0.2)  # [1.2, 1.4, 1.6, 1.8, 2.0]
            }
        }
        
    async def optimize(self, start_date: datetime, end_date: datetime, strategy_type: str) -> Dict:
        """Optimize strategy parameters using walk-forward analysis"""
        try:
            # Get parameter combinations for the strategy type
            param_ranges = self.param_ranges[strategy_type]
            param_combinations = self._generate_param_combinations(param_ranges)
            
            # Prepare data for walk-forward optimization
            tscv = TimeSeriesSplit(n_splits=5)
            best_params = None
            best_score = float('-inf')
            
            # Walk-forward optimization
            for train_idx, test_idx in tscv.split(range(start_date, end_date)):
                train_start = start_date + pd.Timedelta(days=train_idx[0])
                train_end = start_date + pd.Timedelta(days=train_idx[-1])
                test_start = start_date + pd.Timedelta(days=test_idx[0])
                test_end = start_date + pd.Timedelta(days=test_idx[-1])
                
                # Find best parameters on training data
                train_results = await self._parallel_backtest(
                    param_combinations, train_start, train_end
                )
                
                if not train_results:
                    continue
                    
                # Sort by Sharpe ratio
                train_results.sort(key=lambda x: x[1]['metrics']['sharpe_ratio'], reverse=True)
                top_params = [r[0] for r in train_results[:5]]  # Top 5 parameter sets
                
                # Validate on test data
                test_results = await self._parallel_backtest(
                    top_params, test_start, test_end
                )
                
                if not test_results:
                    continue
                    
                # Find best performing parameters
                for params, metrics in test_results:
                    score = self._calculate_optimization_score(metrics['metrics'])
                    if score > best_score:
                        best_score = score
                        best_params = params
            
            if best_params is None:
                raise ValueError("Optimization failed to find suitable parameters")
                
            # Run final backtest with best parameters
            final_config = self.config.copy()
            final_config['trading']['strategy_params'][strategy_type].update(best_params)
            backtest = BacktestEngine(final_config, self.strategy_class)
            final_results = await backtest.run_backtest(start_date, end_date)
            
            return {
                'best_parameters': best_params,
                'optimization_score': best_score,
                'final_results': final_results
            }
            
        except Exception as e:
            logging.error(f"Optimization error: {str(e)}")
            return None
            
    def _generate_param_combinations(self, param_ranges: Dict) -> List[Dict]:
        """Generate all possible parameter combinations"""
        keys = param_ranges.keys()
        values = [list(param_ranges[key]) for key in keys]
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]
        
    async def _parallel_backtest(
        self, 
        param_combinations: List[Dict],
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[Dict, Dict]]:
        """Run backtests in parallel"""
        results = []
        with ProcessPoolExecutor() as executor:
            futures = []
            for params in param_combinations:
                config = self.config.copy()
                config['trading']['strategy_params'].update(params)
                backtest = BacktestEngine(config, self.strategy_class)
                future = executor.submit(
                    backtest.run_backtest, start_date, end_date
                )
                futures.append((params, future))
                
            for params, future in futures:
                result = future.result()
                if result:
                    results.append((params, result))
                    
        return results
        
    def _calculate_optimization_score(self, metrics: Dict) -> float:
        """Calculate overall optimization score"""
        if not metrics:
            return float('-inf')
            
        # Weighted scoring of different metrics
        weights = {
            'sharpe_ratio': 0.3,
            'sortino_ratio': 0.2,
            'win_rate': 0.2,
            'profit_factor': 0.2,
            'max_drawdown': 0.1
        }
        
        score = (
            weights['sharpe_ratio'] * metrics.get('sharpe_ratio', 0) +
            weights['sortino_ratio'] * metrics.get('sortino_ratio', 0) +
            weights['win_rate'] * metrics.get('win_rate', 0) +
            weights['profit_factor'] * metrics.get('profit_factor', 0) -
            weights['max_drawdown'] * abs(metrics.get('max_drawdown', 0))
        )
        
        return score
