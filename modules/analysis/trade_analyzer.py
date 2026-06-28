from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import MetaTrader5 as mt5
from ..analytics.market_analyzer import MarketAnalyzer
from ..deployment.error_handler import ErrorHandler

class TradeAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('trade_analyzer')
        self.market_analyzer = MarketAnalyzer(config)
        
        # Initialize analysis components
        self._init_analysis_components()
        
    def _init_analysis_components(self):
        """Initialize analysis parameters and components"""
        # Performance metrics parameters
        self.metrics_params = {
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'rolling_window': 20,  # Rolling window for metrics
            'drawdown_window': 100,  # Window for drawdown calculation
            'volatility_window': 20  # Window for volatility calculation
        }
        
        # Trade journal parameters
        self.journal_params = {
            'log_file': 'trade_journal.json',
            'metrics_file': 'performance_metrics.json',
            'backup_frequency': 100  # Backup after every 100 trades
        }
        
        # Initialize storage
        self.trade_history = []
        self.metrics_history = []
        self.current_drawdown = 0
        self.peak_balance = self._get_account_balance()
        
        # Load existing data
        self._load_historical_data()
        
    async def analyze_trade(self, trade_data: Dict) -> Dict:
        """Analyze completed trade and update metrics"""
        try:
            # Enrich trade data
            enriched_trade = await self._enrich_trade_data(trade_data)
            
            # Update trade history
            self.trade_history.append(enriched_trade)
            
            # Calculate and update metrics
            metrics = await self._calculate_metrics()
            self.metrics_history.append(metrics)
            
            # Update journal
            await self._update_journal(enriched_trade, metrics)
            
            # Generate trade report
            report = await self._generate_trade_report(enriched_trade, metrics)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Trade analysis error: {str(e)}")
            return {}
            
    async def _enrich_trade_data(self, trade_data: Dict) -> Dict:
        """Enrich trade data with additional analysis"""
        try:
            # Basic trade info
            enriched_data = {
                'timestamp': datetime.now(),
                'symbol': trade_data.get('symbol', ''),
                'entry_price': trade_data.get('entry_price', 0),
                'exit_price': trade_data.get('exit_price', 0),
                'volume': trade_data.get('volume', 0),
                'profit': trade_data.get('profit', 0),
                'direction': trade_data.get('direction', ''),
                'strategy': trade_data.get('strategy', ''),
                'trade_id': trade_data.get('trade_id', '')
            }
            
            # Calculate additional metrics
            enriched_data.update({
                'holding_time': (
                    trade_data.get('exit_time', datetime.now()) -
                    trade_data.get('entry_time', datetime.now())
                ).total_seconds(),
                'return_pct': (
                    (trade_data.get('exit_price', 0) - trade_data.get('entry_price', 0)) /
                    trade_data.get('entry_price', 1) * 100
                ),
                'risk_reward_ratio': await self._calculate_risk_reward_ratio(trade_data),
                'market_conditions': await self._analyze_market_conditions(trade_data)
            })
            
            return enriched_data
            
        except Exception as e:
            self.logger.error(f"Trade enrichment error: {str(e)}")
            return trade_data
            
    async def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.trade_history:
                return {}
                
            # Convert trade history to DataFrame for analysis
            df = pd.DataFrame(self.trade_history)
            
            # Basic metrics
            metrics = {
                'total_trades': len(df),
                'winning_trades': len(df[df['profit'] > 0]),
                'losing_trades': len(df[df['profit'] < 0]),
                'win_rate': len(df[df['profit'] > 0]) / len(df) if len(df) > 0 else 0,
                'total_profit': df['profit'].sum(),
                'average_profit': df['profit'].mean(),
                'profit_factor': (
                    abs(df[df['profit'] > 0]['profit'].sum()) /
                    abs(df[df['profit'] < 0]['profit'].sum())
                    if len(df[df['profit'] < 0]) > 0 else float('inf')
                )
            }
            
            # Calculate Sharpe Ratio
            returns = df['return_pct'].values
            if len(returns) > 1:
                excess_returns = returns - (
                    self.metrics_params['risk_free_rate'] / 252
                )  # Daily risk-free rate
                sharpe = (
                    np.sqrt(252) * np.mean(excess_returns) /
                    np.std(excess_returns) if np.std(excess_returns) > 0 else 0
                )
                metrics['sharpe_ratio'] = sharpe
                
            # Calculate Maximum Drawdown
            cumulative_returns = (1 + returns / 100).cumprod()
            rolling_max = pd.Series(cumulative_returns).expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            metrics['max_drawdown'] = abs(drawdowns.min()) * 100
            
            # Calculate additional metrics
            metrics.update({
                'avg_holding_time': df['holding_time'].mean(),
                'avg_risk_reward': df['risk_reward_ratio'].mean(),
                'best_trade': df['profit'].max(),
                'worst_trade': df['profit'].min(),
                'profit_std': df['profit'].std(),
                'win_streak': self._calculate_longest_streak(df, 'win'),
                'loss_streak': self._calculate_longest_streak(df, 'loss')
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrics calculation error: {str(e)}")
            return {}
            
    async def _calculate_risk_reward_ratio(self, trade_data: Dict) -> float:
        """Calculate risk-reward ratio for a trade"""
        try:
            stop_loss = trade_data.get('stop_loss', 0)
            take_profit = trade_data.get('take_profit', 0)
            entry_price = trade_data.get('entry_price', 0)
            
            if stop_loss > 0 and entry_price > 0:
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price) if take_profit > 0 else abs(trade_data.get('exit_price', entry_price) - entry_price)
                return reward / risk if risk > 0 else 0
                
            return 0
            
        except Exception as e:
            self.logger.error(f"Risk-reward calculation error: {str(e)}")
            return 0
            
    async def _analyze_market_conditions(self, trade_data: Dict) -> Dict:
        """Analyze market conditions during trade"""
        try:
            symbol = trade_data.get('symbol', '')
            entry_time = trade_data.get('entry_time', datetime.now())
            
            # Get market data
            market_data = await self.market_analyzer.get_market_state(
                symbol,
                entry_time
            )
            
            return {
                'volatility': market_data.get('volatility', 0),
                'trend': market_data.get('trend', ''),
                'volume': market_data.get('volume', 0),
                'spread': market_data.get('spread', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Market analysis error: {str(e)}")
            return {}
            
    async def _update_journal(
        self,
        trade_data: Dict,
        metrics: Dict
    ) -> None:
        """Update trade journal with new trade and metrics"""
        try:
            # Create journal entry
            journal_entry = {
                'timestamp': datetime.now().isoformat(),
                'trade_data': trade_data,
                'metrics': metrics,
                'balance': self._get_account_balance(),
                'equity': self._get_account_equity()
            }
            
            # Save to file
            with open(self.journal_params['log_file'], 'a') as f:
                json.dump(journal_entry, f)
                f.write('\n')
                
            # Backup if needed
            if len(self.trade_history) % self.journal_params['backup_frequency'] == 0:
                await self._backup_data()
                
        except Exception as e:
            self.logger.error(f"Journal update error: {str(e)}")
            
    async def _generate_trade_report(
        self,
        trade_data: Dict,
        metrics: Dict
    ) -> Dict:
        """Generate comprehensive trade report"""
        try:
            report = {
                'trade_summary': {
                    'symbol': trade_data['symbol'],
                    'direction': trade_data['direction'],
                    'entry_price': trade_data['entry_price'],
                    'exit_price': trade_data['exit_price'],
                    'profit': trade_data['profit'],
                    'return_pct': trade_data['return_pct'],
                    'holding_time': str(timedelta(seconds=trade_data['holding_time'])),
                    'risk_reward_ratio': trade_data['risk_reward_ratio']
                },
                'market_conditions': trade_data['market_conditions'],
                'performance_metrics': {
                    'win_rate': f"{metrics['win_rate']*100:.2f}%",
                    'profit_factor': f"{metrics['profit_factor']:.2f}",
                    'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                    'max_drawdown': f"{metrics['max_drawdown']:.2f}%"
                },
                'strategy_analysis': {
                    'strategy': trade_data['strategy'],
                    'success_rate': await self._calculate_strategy_success_rate(
                        trade_data['strategy']
                    )
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation error: {str(e)}")
            return {}
            
    def _calculate_longest_streak(
        self,
        df: pd.DataFrame,
        streak_type: str
    ) -> int:
        """Calculate longest winning or losing streak"""
        try:
            # Create series of trade results (1 for win, 0 for loss)
            results = (df['profit'] > 0).astype(int)
            if streak_type == 'loss':
                results = 1 - results
                
            # Calculate streaks
            streaks = results.groupby(
                (results != results.shift()).cumsum()
            ).cumsum()
            
            return streaks.max()
            
        except Exception as e:
            self.logger.error(f"Streak calculation error: {str(e)}")
            return 0
            
    async def _calculate_strategy_success_rate(
        self,
        strategy: str
    ) -> float:
        """Calculate success rate for specific strategy"""
        try:
            strategy_trades = [
                t for t in self.trade_history
                if t['strategy'] == strategy
            ]
            
            if not strategy_trades:
                return 0
                
            winning_trades = len([
                t for t in strategy_trades
                if t['profit'] > 0
            ])
            
            return winning_trades / len(strategy_trades)
            
        except Exception as e:
            self.logger.error(f"Strategy success rate calculation error: {str(e)}")
            return 0
            
    def _get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return 0
            return account_info.balance
            
        except Exception as e:
            self.logger.error(f"Balance retrieval error: {str(e)}")
            return 0
            
    def _get_account_equity(self) -> float:
        """Get current account equity"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return 0
            return account_info.equity
            
        except Exception as e:
            self.logger.error(f"Equity retrieval error: {str(e)}")
            return 0
            
    def _load_historical_data(self) -> None:
        """Load historical trade and metrics data"""
        try:
            # Load trade journal
            if os.path.exists(self.journal_params['log_file']):
                with open(self.journal_params['log_file'], 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        self.trade_history.append(entry['trade_data'])
                        self.metrics_history.append(entry['metrics'])
                        
        except Exception as e:
            self.logger.error(f"Historical data loading error: {str(e)}")
            
    async def _backup_data(self) -> None:
        """Backup trade and metrics data"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Backup trade journal
            journal_backup = f"{self.journal_params['log_file']}.{timestamp}.bak"
            with open(journal_backup, 'w') as f:
                for trade in self.trade_history:
                    json.dump(trade, f)
                    f.write('\n')
                    
            # Backup metrics
            metrics_backup = f"{self.journal_params['metrics_file']}.{timestamp}.bak"
            with open(metrics_backup, 'w') as f:
                json.dump(self.metrics_history, f)
                
            self.logger.info(f"Data backup completed: {timestamp}")
            
        except Exception as e:
            self.logger.error(f"Data backup error: {str(e)}")
            
    async def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        try:
            if not self.metrics_history:
                return {}
                
            latest_metrics = self.metrics_history[-1]
            
            summary = {
                'overall_metrics': latest_metrics,
                'daily_metrics': await self._calculate_daily_metrics(),
                'strategy_metrics': await self._calculate_strategy_metrics(),
                'risk_metrics': await self._calculate_risk_metrics()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Performance summary error: {str(e)}")
            return {}
