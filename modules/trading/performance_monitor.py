import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import MetaTrader5 as mt5
import psutil
import time
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc

class PerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.trades_history = []
        self.daily_stats = {}
        self.risk_metrics = {}
        self.execution_metrics = {
            'slippage': 0.0,
            'execution_time': 0.0,
            'rejection_rate': 0.0,
            'fill_ratio': 1.0
        }
        self.consecutive_losses = 0
        self.daily_metrics = {
            'profit': 0.0,
            'trades': 0,
            'wins': 0,
            'losses': 0
        }
        self.peak_equity = 0.0
        self.current_equity = 0.0
        
        # Performance tracking
        self.system_metrics = {
            'mt5_latency': [],
            'memory_usage': [],
            'cpu_usage': [],
            'data_processing_time': [],
            'trades_per_minute': []
        }
        
        # Timing measurements
        self.operation_timings = {
            'market_data_fetch': [],
            'signal_generation': [],
            'order_execution': [],
            'position_management': []
        }
        
        # Initialize directories
        os.makedirs('logs/performance', exist_ok=True)
        
        # Add trade cache for optimized lookups
        self._trade_cache = {}
        self._last_clean = datetime.now()
        self._clean_interval = timedelta(hours=1)
        
        # Thread pool for report generation
        self._executor = ThreadPoolExecutor(max_workers=2)
        
    def log_trade(self, trade_data: Dict):
        """Log individual trade data"""
        # Add timestamp if not present
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now()
            
        # Store additional metrics
        trade_data['market_conditions'] = trade_data.get('market_conditions', {})
        trade_data['execution_quality'] = {
            'slippage': trade_data.get('slippage', 0.0),
            'execution_time': trade_data.get('execution_time', 0.0),
            'rejection': trade_data.get('rejection', False),
            'requotes': trade_data.get('requotes', 0)
        }
        
        # Add to history
        self.trades_history.append(trade_data)
        
        # Cache trade for faster lookups
        trade_id = trade_data.get('ticket', len(self.trades_history))
        self._trade_cache[trade_id] = trade_data
        
        # Update daily metrics
        self.daily_metrics['profit'] += trade_data['profit']
        self.daily_metrics['trades'] += 1
        if trade_data['profit'] > 0:
            self.daily_metrics['wins'] += 1
            self.consecutive_losses = 0
        else:
            self.daily_metrics['losses'] += 1
            self.consecutive_losses += 1
            
        # Update execution metrics
        self._update_execution_metrics(trade_data)
        
        # Update equity tracking
        self.current_equity += trade_data['profit']
        self.peak_equity = max(self.peak_equity, self.current_equity)
        
        # Update trades per minute
        minute = datetime.now().replace(second=0, microsecond=0)
        if 'trades_per_minute' not in self.system_metrics:
            self.system_metrics['trades_per_minute'] = {}
        self.system_metrics['trades_per_minute'][minute] = self.system_metrics['trades_per_minute'].get(minute, 0) + 1
        
        # Track trade types
        symbol = trade_data.get('symbol', 'unknown')
        if 'trade_counts_by_symbol' not in self.system_metrics:
            self.system_metrics['trade_counts_by_symbol'] = {}
        if symbol not in self.system_metrics['trade_counts_by_symbol']:
            self.system_metrics['trade_counts_by_symbol'][symbol] = 0
        self.system_metrics['trade_counts_by_symbol'][symbol] += 1
        
        self._update_metrics()
        
        # Periodically clean old cache entries
        current_time = datetime.now()
        if current_time - self._last_clean > self._clean_interval:
            self._clean_cache()
            self._last_clean = current_time
        
    async def log_operation_timing(self, operation_type: str, execution_time: float):
        """Log timing for different operations"""
        if operation_type in self.operation_timings:
            # Keep last 100 timing measurements per operation
            if len(self.operation_timings[operation_type]) >= 100:
                self.operation_timings[operation_type].pop(0)
            self.operation_timings[operation_type].append(execution_time)
            
            # Log warning if operation is taking too long
            avg_time = np.mean(self.operation_timings[operation_type])
            if execution_time > avg_time * 2 and execution_time > 1.0:  # More than double the average and > 1 second
                self.logger.warning(f"{operation_type} operation taking too long: {execution_time:.2f}s (avg: {avg_time:.2f}s)")
    
    def get_daily_metrics(self) -> Dict:
        """Get current day's trading metrics"""
        return self.daily_metrics
        
    def get_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity
        
    def get_consecutive_losses(self) -> int:
        """Get number of consecutive losses"""
        return self.consecutive_losses
        
    def get_execution_metrics(self) -> Dict:
        """Get execution quality metrics"""
        return self.execution_metrics
        
    def get_avg_operation_time(self, operation_type: str) -> float:
        """Get average time for a specific operation"""
        if operation_type in self.operation_timings and self.operation_timings[operation_type]:
            return np.mean(self.operation_timings[operation_type])
        return 0.0
        
    def _update_execution_metrics(self, trade_data: Dict):
        """Update execution quality metrics"""
        # Update slippage (use exponential moving average)
        new_slippage = trade_data.get('slippage', 0.0)
        self.execution_metrics['slippage'] = (
            self.execution_metrics['slippage'] * 0.95 + new_slippage * 0.05
        )
        
        # Update execution time
        new_execution_time = trade_data.get('execution_time', 0.0)
        self.execution_metrics['execution_time'] = (
            self.execution_metrics['execution_time'] * 0.95 + new_execution_time * 0.05
        )
        
        # Update rejection rate
        was_rejected = 1.0 if trade_data.get('rejection', False) else 0.0
        self.execution_metrics['rejection_rate'] = (
            self.execution_metrics['rejection_rate'] * 0.95 + was_rejected * 0.05
        )
        
        # Update fill ratio
        if 'requested_volume' in trade_data and trade_data['requested_volume'] > 0:
            fill_ratio = trade_data['volume'] / trade_data['requested_volume']
            self.execution_metrics['fill_ratio'] = (
                self.execution_metrics['fill_ratio'] * 0.95 + fill_ratio * 0.05
            )
            
    def _clean_cache(self):
        """Clean old entries from trade cache"""
        # Keep only the last 1000 trades in memory
        if len(self._trade_cache) > 1000:
            # Identify old trades to remove
            trade_ids = sorted(self._trade_cache.keys())
            trades_to_remove = trade_ids[:-1000]  # Keep the 1000 most recent
            
            # Remove old trades
            for trade_id in trades_to_remove:
                self._trade_cache.pop(trade_id, None)
                
            # Force garbage collection
            gc.collect()
            
            self.logger.debug(f"Cleaned {len(trades_to_remove)} old trades from cache")
            
    def _update_metrics(self):
        """Update trading metrics"""
        # Calculate win rate
        if self.daily_metrics['trades'] > 0:
            win_rate = self.daily_metrics['wins'] / self.daily_metrics['trades']
        else:
            win_rate = 0.0
            
        # Calculate profit factor
        if self.daily_metrics['losses'] == 0:
            profit_factor = float('inf') if self.daily_metrics['wins'] > 0 else 0.0
        else:
            total_profit = sum(trade['profit'] for trade in self.trades_history if trade['profit'] > 0)
            total_loss = abs(sum(trade['profit'] for trade in self.trades_history if trade['profit'] < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
        # Update risk metrics
        self.risk_metrics = {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'drawdown': self.get_current_drawdown(),
            'consecutive_losses': self.consecutive_losses
        }
        
    def reset_daily_metrics(self):
        """Reset daily metrics"""
        self.daily_metrics = {
            'profit': 0.0,
            'trades': 0,
            'wins': 0,
            'losses': 0
        }
        self.consecutive_losses = 0
        
    def _calculate_sharpe_ratio(self, returns: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
            
        # Annualize based on data frequency
        annual_factor = 252  # Assuming daily returns
        excess_returns = returns - (risk_free_rate / annual_factor)
        return np.sqrt(annual_factor) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() > 0 else 0
        
    def _calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if equity_curve.empty:
            return 0.0
            
        # Calculate running maximum
        running_max = equity_curve.cummax()
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        # Return the maximum drawdown
        return abs(drawdown.min()) if not np.isnan(drawdown.min()) else 0.0
    
    async def generate_report(self, report_path: str = "reports", detailed: bool = False):
        """Generate performance report asynchronously"""
        try:
            # Create reports directory
            os.makedirs(report_path, exist_ok=True)
            
            # Convert trades to DataFrame for analysis
            df_trades = pd.DataFrame(self.trades_history)
            
            if df_trades.empty:
                self.logger.warning("No trades to generate report")
                return
                
            # Save trades to CSV
            report_file = f"{report_path}/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, lambda: df_trades.to_csv(report_file, index=False))
            
            if detailed:
                # Generate detailed charts in separate threads
                equity_task = loop.run_in_executor(self._executor, self._generate_equity_curve, df_trades, report_path)
                drawdown_task = loop.run_in_executor(self._executor, self._generate_drawdown_chart, df_trades, report_path)
                wins_task = loop.run_in_executor(self._executor, self._generate_win_loss_distribution, df_trades, report_path)
                
                # Wait for all chart generation to complete
                await asyncio.gather(equity_task, drawdown_task, wins_task)
            
            # Generate summary metrics
            summary = {
                'total_trades': len(df_trades),
                'win_rate': self.risk_metrics.get('win_rate', 0.0),
                'profit_factor': self.risk_metrics.get('profit_factor', 0.0),
                'max_drawdown': self.risk_metrics.get('drawdown', 0.0),
                'sharpe_ratio': self._calculate_sharpe_ratio(df_trades['profit']) if 'profit' in df_trades.columns else 0.0,
                'execution_quality': self.execution_metrics
            }
            
            # Save summary to JSON
            summary_file = f"{report_path}/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            await loop.run_in_executor(self._executor, lambda: json.dump(summary, open(summary_file, 'w'), indent=4))
            
            self.logger.info(f"Performance report generated at {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            
    def _generate_equity_curve(self, df_trades: pd.DataFrame, report_path: str):
        """Generate equity curve chart"""
        try:
            if 'timestamp' not in df_trades.columns or 'profit' not in df_trades.columns:
                return
                
            # Sort by timestamp
            df_trades = df_trades.sort_values('timestamp')
            
            # Create equity curve
            df_trades['cumulative_profit'] = df_trades['profit'].cumsum()
            
            # Plot equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(df_trades['timestamp'], df_trades['cumulative_profit'])
            plt.title('Equity Curve')
            plt.xlabel('Time')
            plt.ylabel('Profit')
            plt.grid(True)
            
            # Save chart
            plt.savefig(f"{report_path}/equity_curve.png")
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error generating equity curve: {str(e)}")
    
    def _generate_drawdown_chart(self, df_trades: pd.DataFrame, report_path: str):
        """Generate drawdown chart"""
        try:
            if 'timestamp' not in df_trades.columns or 'profit' not in df_trades.columns:
                return
                
            # Sort by timestamp
            df_trades = df_trades.sort_values('timestamp')
            
            # Create equity curve
            df_trades['cumulative_profit'] = df_trades['profit'].cumsum()
            
            # Calculate drawdown
            df_trades['peak'] = df_trades['cumulative_profit'].cummax()
            df_trades['drawdown'] = (df_trades['cumulative_profit'] - df_trades['peak']) / df_trades['peak'].replace(0, 1) * 100
            
            # Plot drawdown
            plt.figure(figsize=(12, 6))
            plt.fill_between(df_trades['timestamp'], df_trades['drawdown'], 0, color='red', alpha=0.3)
            plt.plot(df_trades['timestamp'], df_trades['drawdown'], color='red')
            plt.title('Drawdown (%)')
            plt.xlabel('Time')
            plt.ylabel('Drawdown %')
            plt.grid(True)
            
            # Save chart
            plt.savefig(f"{report_path}/drawdown.png")
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error generating drawdown chart: {str(e)}")
    
    def _generate_win_loss_distribution(self, df_trades: pd.DataFrame, report_path: str):
        """Generate win/loss distribution chart"""
        try:
            if 'profit' not in df_trades.columns:
                return
                
            # Plot profit distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(df_trades['profit'], bins=50, kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Profit Distribution')
            plt.xlabel('Profit')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # Save chart
            plt.savefig(f"{report_path}/profit_distribution.png")
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error generating win/loss distribution: {str(e)}")
            
    async def track_system_performance(self):
        """Track system performance metrics"""
        try:
            # Memory usage
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            # CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # MT5 latency - measure time to get a simple data point
            start_time = time.time()
            if mt5.initialize():
                _ = mt5.symbol_info_tick('EURUSD')
                mt5_latency = time.time() - start_time
            else:
                mt5_latency = -1  # Error indicator
                
            # Update metrics
            timestamp = datetime.now()
            self.system_metrics['memory_usage'].append((timestamp, memory_mb))
            self.system_metrics['cpu_usage'].append((timestamp, cpu_percent))
            self.system_metrics['mt5_latency'].append((timestamp, mt5_latency))
            
            # Log warning if MT5 latency is high
            if mt5_latency > 0.5:  # More than 500ms
                self.logger.warning(f"High MT5 latency detected: {mt5_latency:.2f}s")
                
            # Keep only last 1000 measurements
            for metric in ['memory_usage', 'cpu_usage', 'mt5_latency']:
                if len(self.system_metrics[metric]) > 1000:
                    self.system_metrics[metric] = self.system_metrics[metric][-1000:]
                    
            # Return current performance
            return {
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'mt5_latency': mt5_latency
            }
            
        except Exception as e:
            self.logger.error(f"Error tracking system performance: {str(e)}")
            return None 