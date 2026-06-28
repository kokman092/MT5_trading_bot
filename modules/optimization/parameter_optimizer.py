import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import json
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from copy import deepcopy

class ParameterOptimizer:
    def __init__(self, config, strategy_class):
        self.config = config
        self.strategy_class = strategy_class
        self.logger = logging.getLogger(__name__)
        
        # Optimization settings
        self.population_size = config.get('POPULATION_SIZE', 20)
        self.max_generations = config.get('MAX_GENERATIONS', 50)
        self.mutation_factor = config.get('MUTATION_FACTOR', 0.8)
        self.crossover_prob = config.get('CROSSOVER_PROB', 0.7)
        self.parallel_jobs = min(multiprocessing.cpu_count() - 1, 8)
        
        # Parameter bounds
        self.parameter_bounds = {
            'RSI': {
                'OVERSOLD': (20, 40),
                'OVERBOUGHT': (60, 80)
            },
            'MACD': {
                'FAST': (8, 16),
                'SLOW': (20, 30),
                'SIGNAL': (7, 12)
            },
            'BOLLINGER': {
                'PERIOD': (15, 25),
                'STD_DEV': (1.5, 2.5)
            },
            'ATR': {
                'PERIOD': (10, 20),
                'TP_MULTIPLIER': (2.0, 4.0),
                'SL_MULTIPLIER': (1.5, 3.0)
            },
            'STOCHASTIC': {
                'K_PERIOD': (10, 20),
                'D_PERIOD': (2, 5),
                'OVERSOLD': (20, 40),
                'OVERBOUGHT': (60, 80)
            },
            'RISK_MANAGEMENT': {
                'RISK_PER_TRADE': (0.01, 0.03),
                'MAX_RISK_TOTAL': (0.05, 0.15)
            }
        }
    
    def create_parameter_bounds(self):
        """Convert parameter bounds dictionary to list format for optimization"""
        bounds = []
        self.param_names = []
        
        for category, params in self.parameter_bounds.items():
            for param_name, (low, high) in params.items():
                bounds.append((low, high))
                self.param_names.append(f"{category}.{param_name}")
        
        return bounds
    
    def update_config(self, parameters):
        """Update configuration with new parameters"""
        config = deepcopy(self.config)
        
        for param_name, value in zip(self.param_names, parameters):
            category, param = param_name.split('.')
            if category not in config:
                config[category] = {}
            config[category][param] = value
        
        return config
    
    def evaluate_parameters(self, parameters, data):
        """Evaluate a set of parameters using backtest results"""
        try:
            # Update configuration with new parameters
            test_config = self.update_config(parameters)
            
            # Create strategy instance
            strategy = self.strategy_class(test_config)
            
            # Run backtest
            results = strategy.backtest(data)
            
            if results is None:
                return float('-inf')
            
            # Calculate fitness score
            sharpe_ratio = results['metrics']['sharpe_ratio']
            profit_factor = results['metrics']['profit_factor']
            max_drawdown = abs(results['metrics']['max_drawdown'])
            win_rate = results['metrics']['win_rate']
            
            # Penalize strategies with few trades
            if results['metrics']['total_trades'] < 30:
                return float('-inf')
            
            # Custom fitness function
            fitness = (
                2 * sharpe_ratio +  # Emphasize risk-adjusted returns
                profit_factor +
                win_rate -
                2 * max_drawdown  # Penalize large drawdowns
            )
            
            return fitness
            
        except Exception as e:
            self.logger.error(f"Error evaluating parameters: {str(e)}")
            return float('-inf')
    
    def optimize(self, data, callback=None):
        """Run parameter optimization"""
        try:
            self.logger.info("Starting parameter optimization...")
            
            # Create bounds list
            bounds = self.create_parameter_bounds()
            
            # Optimization objective function
            def objective(parameters):
                return -self.evaluate_parameters(parameters, data)
            
            # Run differential evolution
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.max_generations,
                popsize=self.population_size,
                mutation=self.mutation_factor,
                recombination=self.crossover_prob,
                workers=self.parallel_jobs,
                updating='deferred',
                callback=callback
            )
            
            # Convert results to dictionary
            optimized_params = {}
            for param_name, value in zip(self.param_names, result.x):
                category, param = param_name.split('.')
                if category not in optimized_params:
                    optimized_params[category] = {}
                optimized_params[category][param] = value
            
            # Calculate final metrics
            final_config = self.update_config(result.x)
            strategy = self.strategy_class(final_config)
            final_results = strategy.backtest(data)
            
            optimization_results = {
                'parameters': optimized_params,
                'metrics': final_results['metrics'],
                'convergence': result.convergence,
                'nit': result.nit,
                'success': result.success,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save results
            with open('optimization_results.json', 'w') as f:
                json.dump(optimization_results, f, indent=4)
            
            self.logger.info("Parameter optimization completed successfully")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error in parameter optimization: {str(e)}")
            return None
    
    def optimize_regime_specific(self, data, regime_detector):
        """Optimize parameters for different market regimes"""
        try:
            # Detect regimes
            regimes = regime_detector.detect_regime(data)
            if regimes is None:
                return None
            
            # Split data by regime
            regime_data = {}
            current_regime = None
            current_chunk = []
            
            for i, row in data.iterrows():
                regime = regime_detector.detect_regime(data.loc[:i])
                if regime != current_regime:
                    if current_regime is not None:
                        regime_data[current_regime] = pd.DataFrame(current_chunk)
                    current_regime = regime
                    current_chunk = []
                current_chunk.append(row)
            
            # Optimize for each regime
            regime_parameters = {}
            for regime, regime_data in regime_data.items():
                self.logger.info(f"Optimizing parameters for regime {regime}")
                results = self.optimize(regime_data)
                if results:
                    regime_parameters[regime] = results
            
            return regime_parameters
            
        except Exception as e:
            self.logger.error(f"Error in regime-specific optimization: {str(e)}")
            return None
    
    def cross_validate_parameters(self, data, n_splits=5):
        """Cross-validate parameter optimization results"""
        try:
            # Split data into chunks
            chunk_size = len(data) // n_splits
            results = []
            
            for i in range(n_splits):
                # Create train/test split
                test_start = i * chunk_size
                test_end = (i + 1) * chunk_size
                train_data = pd.concat([
                    data[:test_start],
                    data[test_end:]
                ])
                test_data = data[test_start:test_end]
                
                # Optimize on train data
                train_results = self.optimize(train_data)
                if train_results:
                    # Test on validation set
                    test_config = self.update_config(train_results['parameters'])
                    strategy = self.strategy_class(test_config)
                    test_results = strategy.backtest(test_data)
                    
                    results.append({
                        'fold': i,
                        'train_metrics': train_results['metrics'],
                        'test_metrics': test_results['metrics']
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            return None
