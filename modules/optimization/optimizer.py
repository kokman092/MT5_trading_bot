import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import itertools
import concurrent.futures
import logging
from tqdm import tqdm
from ..backtesting.backtest_engine import BacktestEngine

class StrategyOptimizer:
    def __init__(self, config: Dict, strategy_class):
        self.config = config
        self.strategy_class = strategy_class
        self.parameter_ranges = self._get_parameter_ranges()
        self.logger = logging.getLogger('strategy_optimizer')
        
    def optimize(
        self,
        start_date: datetime,
        end_date: datetime,
        optimization_metric: str = 'sharpe_ratio',
        max_combinations: int = 1000
    ) -> Tuple[Dict, Dict]:
        """
        Optimize strategy parameters using grid search
        Returns: Best parameters and their performance metrics
        """
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations()
        total_combinations = len(param_combinations)
        
        self.logger.info(f"Generated {total_combinations} parameter combinations")
        
        # Limit number of combinations to prevent excessive computation
        if total_combinations > max_combinations:
            self.logger.warning(f"Limiting combinations from {total_combinations} to {max_combinations}")
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[:max_combinations]
            
        # Run parallel optimization
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for params in param_combinations:
                config_copy = self.config.copy()
                config_copy.update(params)
                futures.append(
                    executor.submit(
                        self._run_backtest,
                        config_copy,
                        start_date,
                        end_date,
                        optimization_metric
                    )
                )
                
            # Collect results with progress bar
            results = []
            with tqdm(total=len(futures), desc="Optimizing") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)
                    
        if not results:
            self.logger.warning("No valid results found during optimization")
            return None, None
            
        # Find best parameters
        best_result = max(results, key=lambda x: x['metrics'][optimization_metric])
        self.logger.info(f"Best parameters found: {best_result['parameters']}")
        self.logger.info(f"Best metrics: {best_result['metrics']}")
        
        return best_result['parameters'], best_result['metrics']
        
    def _get_parameter_ranges(self) -> Dict:
        """Define parameter ranges for optimization"""
        return {
            # Technical indicators
            'RSI_PERIOD': range(10, 30, 2),
            'RSI_OVERBOUGHT': range(65, 85, 5),
            'RSI_OVERSOLD': range(15, 35, 5),
            'MA_FAST_PERIOD': range(10, 50, 5),
            'MA_SLOW_PERIOD': range(20, 100, 10),
            'MACD_FAST': range(8, 20, 2),
            'MACD_SLOW': range(20, 40, 4),
            'MACD_SIGNAL': range(7, 15, 2),
            'BB_PERIOD': range(10, 30, 5),
            'BB_STD': range(2, 4),
            'ADX_PERIOD': range(10, 30, 5),
            'ADX_THRESHOLD': range(20, 40, 5),
            
            # Risk management
            'RISK_PERCENTAGE': np.arange(0.5, 3.1, 0.5),
            'TAKE_PROFIT_RATIO': np.arange(1.5, 4.1, 0.5),
            'STOP_LOSS_ATR_MULT': np.arange(1.0, 3.1, 0.5),
            'TRAILING_STOP_ATR_MULT': np.arange(1.5, 4.1, 0.5),
            'MAX_POSITIONS': range(1, 6),
            
            # Trade management
            'MIN_RR_RATIO': np.arange(1.5, 3.1, 0.5),
            'MAX_SPREAD_POINTS': range(10, 50, 10),
            'MIN_VOLATILITY': np.arange(0.001, 0.005, 0.001),
            'MAX_VOLATILITY': np.arange(0.005, 0.02, 0.002)
        }
        
    def _generate_parameter_combinations(self) -> List[Dict]:
        """Generate all possible parameter combinations"""
        param_names = list(self.parameter_ranges.keys())
        param_values = list(self.parameter_ranges.values())
        
        combinations = []
        for values in itertools.product(*param_values):
            combination = dict(zip(param_names, values))
            if self._is_valid_combination(combination):
                combinations.append(combination)
                
        return combinations
        
    def _is_valid_combination(self, params: Dict) -> bool:
        """Validate parameter combination"""
        try:
            # MA periods validation
            if params['MA_FAST_PERIOD'] >= params['MA_SLOW_PERIOD']:
                return False
                
            # MACD periods validation
            if params['MACD_FAST'] >= params['MACD_SLOW']:
                return False
                
            # RSI levels validation
            if params['RSI_OVERSOLD'] >= params['RSI_OVERBOUGHT']:
                return False
                
            # Volatility thresholds validation
            if params['MIN_VOLATILITY'] >= params['MAX_VOLATILITY']:
                return False
                
            # Risk management validation
            if params['RISK_PERCENTAGE'] * params['MAX_POSITIONS'] > 10:  # Max 10% total risk
                return False
                
            # Minimum profit potential validation
            if params['TAKE_PROFIT_RATIO'] / params['STOP_LOSS_ATR_MULT'] < params['MIN_RR_RATIO']:
                return False
                
            # Trailing stop validation
            if params['TRAILING_STOP_ATR_MULT'] <= params['STOP_LOSS_ATR_MULT']:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation error: {str(e)}")
            return False
            
    async def _run_backtest(
        self,
        config: Dict,
        start_date: datetime,
        end_date: datetime,
        optimization_metric: str
    ) -> Optional[Dict]:
        """Run backtest with specific parameters"""
        try:
            engine = BacktestEngine(config, self.strategy_class)
            results = await engine.run_backtest(start_date, end_date)
            
            if results and results['metrics'].get(optimization_metric):
                return {
                    'parameters': {k: config[k] for k in self.parameter_ranges.keys()},
                    'metrics': results['metrics']
                }
                
        except Exception as e:
            self.logger.error(f"Optimization error: {str(e)}")
            
        return None
        
    def walk_forward_analysis(
        self,
        start_date: datetime,
        end_date: datetime,
        window_size: timedelta = timedelta(days=30),
        optimization_metric: str = 'sharpe_ratio'
    ) -> List[Dict]:
        """
        Perform walk-forward analysis
        - Optimize parameters on in-sample data
        - Test on out-of-sample data
        """
        results = []
        current_date = start_date
        
        with tqdm(desc="Walk-forward Analysis") as pbar:
            while current_date < end_date:
                # Define in-sample and out-of-sample periods
                optimization_end = min(current_date + window_size, end_date)
                validation_end = min(optimization_end + window_size, end_date)
                
                self.logger.info(f"Optimizing period: {current_date} to {optimization_end}")
                
                # Optimize parameters on in-sample data
                best_params, _ = self.optimize(
                    current_date,
                    optimization_end,
                    optimization_metric
                )
                
                if best_params:
                    self.logger.info(f"Testing period: {optimization_end} to {validation_end}")
                    
                    # Test parameters on out-of-sample data
                    config_copy = self.config.copy()
                    config_copy.update(best_params)
                    engine = BacktestEngine(config_copy, self.strategy_class)
                    validation_results = await engine.run_backtest(
                        optimization_end,
                        validation_end
                    )
                    
                    if validation_results:
                        results.append({
                            'period_start': current_date,
                            'period_end': validation_end,
                            'parameters': best_params,
                            'metrics': validation_results['metrics']
                        })
                        
                current_date += window_size
                pbar.update(1)
                
        return results
        
    def monte_carlo_simulation(
        self,
        trades: List[Dict],
        num_simulations: int = 1000
    ) -> Dict:
        """
        Perform Monte Carlo simulation on trade results
        - Randomize trade sequence
        - Calculate confidence intervals for performance metrics
        """
        if not trades:
            self.logger.warning("No trades provided for Monte Carlo simulation")
            return {}
            
        results = []
        initial_balance = self.config['INITIAL_BALANCE']
        
        # Run simulations in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for _ in range(num_simulations):
                futures.append(
                    executor.submit(
                        self._run_single_simulation,
                        trades,
                        initial_balance
                    )
                )
                
            # Collect results with progress bar
            with tqdm(total=num_simulations, desc="Monte Carlo Simulation") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)
                    
        # Calculate confidence intervals
        confidence_intervals = {}
        metrics = ['final_balance', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                confidence_intervals[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    '5th_percentile': np.percentile(values, 5),
                    '25th_percentile': np.percentile(values, 25),
                    'median': np.percentile(values, 50),
                    '75th_percentile': np.percentile(values, 75),
                    '95th_percentile': np.percentile(values, 95)
                }
                
        return confidence_intervals
        
    def _run_single_simulation(self, trades: List[Dict], initial_balance: float) -> Dict:
        """Run a single Monte Carlo simulation"""
        try:
            # Shuffle trades
            shuffled_trades = trades.copy()
            np.random.shuffle(shuffled_trades)
            
            # Calculate equity curve
            balance = initial_balance
            equity_curve = [balance]
            
            for trade in shuffled_trades:
                balance += trade['pnl']
                equity_curve.append(balance)
                
            # Calculate metrics
            equity_curve = np.array(equity_curve)
            returns = np.diff(equity_curve) / equity_curve[:-1]
            
            winning_trades = len([t for t in shuffled_trades if t['pnl'] > 0])
            total_trades = len(shuffled_trades)
            
            return {
                'final_balance': balance,
                'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0,
                'max_drawdown': min(0, min(
                    (equity_curve[i] - max(equity_curve[:i+1])) / max(equity_curve[:i+1])
                    for i in range(len(equity_curve))
                )),
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'profit_factor': (
                    sum(t['pnl'] for t in shuffled_trades if t['pnl'] > 0) /
                    abs(sum(t['pnl'] for t in shuffled_trades if t['pnl'] < 0))
                    if sum(t['pnl'] for t in shuffled_trades if t['pnl'] < 0) != 0 else float('inf')
                )
            }
            
        except Exception as e:
            self.logger.error(f"Simulation error: {str(e)}")
            return None
