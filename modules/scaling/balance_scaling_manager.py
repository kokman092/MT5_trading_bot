from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from ..analytics.market_analyzer import MarketAnalyzer
from ..risk.risk_manager import RiskManager
from ..deployment.error_handler import ErrorHandler

class BalanceScalingManager:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('balance_scaling_manager')
        self.market_analyzer = MarketAnalyzer(config)
        self.risk_manager = RiskManager(config)
        
        # Initialize scaling components
        self._init_scaling_components()
        
    def _init_scaling_components(self):
        """Initialize scaling parameters and components"""
        # Compounding parameters
        self.compounding_params = {
            'base_risk_percentage': 0.02,  # 2% base risk
            'max_risk_percentage': 0.05,  # 5% maximum risk
            'min_risk_percentage': 0.01,  # 1% minimum risk
            'risk_step_size': 0.001,  # 0.1% risk adjustment step
            'balance_threshold_multiplier': 2.0  # Double balance threshold
        }
        
        # Profit taking parameters
        self.profit_params = {
            'profit_taking_percentage': 0.05,  # 5% profit taking
            'reinvestment_rate': 0.95,  # 95% reinvestment
            'profit_threshold': 100.0,  # Minimum profit for taking
            'max_daily_profit_taking': 0.20  # 20% max daily profit taking
        }
        
        # Scaling thresholds
        self.scaling_thresholds = [
            {'balance': 10, 'risk_mult': 1.0},
            {'balance': 50, 'risk_mult': 1.2},
            {'balance': 100, 'risk_mult': 1.4},
            {'balance': 500, 'risk_mult': 1.6},
            {'balance': 1000, 'risk_mult': 1.8},
            {'balance': 5000, 'risk_mult': 2.0}
        ]
        
        # Performance tracking
        self.performance_history = []
        self.profit_taking_history = []
        self.last_balance_check = datetime.now()
        self.initial_balance = self._get_account_balance()
        
    async def calculate_position_size(
        self,
        symbol: str,
        risk_params: Dict
    ) -> float:
        """Calculate position size based on current balance and scaling factors"""
        try:
            # Get current balance
            current_balance = self._get_account_balance()
            
            # Calculate risk percentage
            risk_percentage = await self._calculate_risk_percentage(
                current_balance,
                risk_params
            )
            
            # Calculate risk amount
            risk_amount = current_balance * risk_percentage
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0
                
            # Calculate position size in lots
            tick_size = symbol_info.trade_tick_size
            tick_value = symbol_info.trade_tick_value
            stop_loss_points = abs(
                risk_params.get('stop_loss', 0) - risk_params.get('entry_price', 0)
            ) / tick_size
            
            if stop_loss_points > 0 and tick_value > 0:
                position_size = risk_amount / (stop_loss_points * tick_value)
                
                # Round to symbol's lot step
                lot_step = symbol_info.volume_step
                position_size = round(position_size / lot_step) * lot_step
                
                return position_size
                
            return 0
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {str(e)}")
            return 0
            
    async def update_after_trade(self, trade_result: Dict) -> None:
        """Update scaling parameters after trade completion"""
        try:
            # Record trade performance
            self.performance_history.append({
                'timestamp': datetime.now(),
                'profit': trade_result.get('profit', 0),
                'balance': self._get_account_balance(),
                'risk_percentage': trade_result.get('risk_percentage', 0)
            })
            
            # Check for profit taking opportunity
            await self._check_profit_taking()
            
            # Update risk parameters
            await self._update_risk_parameters()
            
            # Log scaling metrics
            await self._log_scaling_metrics()
            
        except Exception as e:
            self.logger.error(f"Trade update error: {str(e)}")
            
    async def _calculate_risk_percentage(
        self,
        balance: float,
        risk_params: Dict
    ) -> float:
        """Calculate appropriate risk percentage based on balance and performance"""
        try:
            # Get base risk percentage
            base_risk = self.compounding_params['base_risk_percentage']
            
            # Apply balance-based scaling
            risk_mult = await self._get_balance_risk_multiplier(balance)
            risk_percentage = base_risk * risk_mult
            
            # Adjust based on recent performance
            performance_mult = await self._get_performance_multiplier()
            risk_percentage *= performance_mult
            
            # Apply limits
            risk_percentage = min(
                risk_percentage,
                self.compounding_params['max_risk_percentage']
            )
            risk_percentage = max(
                risk_percentage,
                self.compounding_params['min_risk_percentage']
            )
            
            return risk_percentage
            
        except Exception as e:
            self.logger.error(f"Risk percentage calculation error: {str(e)}")
            return self.compounding_params['min_risk_percentage']
            
    async def _get_balance_risk_multiplier(self, balance: float) -> float:
        """Get risk multiplier based on account balance"""
        try:
            # Find appropriate threshold
            for threshold in reversed(self.scaling_thresholds):
                if balance >= threshold['balance']:
                    return threshold['risk_mult']
                    
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Risk multiplier calculation error: {str(e)}")
            return 1.0
            
    async def _get_performance_multiplier(self) -> float:
        """Calculate performance-based risk multiplier"""
        try:
            if not self.performance_history:
                return 1.0
                
            # Get recent trades
            recent_trades = self.performance_history[-20:]
            
            # Calculate win rate
            wins = sum(1 for trade in recent_trades if trade['profit'] > 0)
            win_rate = wins / len(recent_trades)
            
            # Calculate profit factor
            gross_profit = sum(t['profit'] for t in recent_trades if t['profit'] > 0)
            gross_loss = abs(sum(t['profit'] for t in recent_trades if t['profit'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate multiplier
            if win_rate >= 0.6 and profit_factor >= 2.0:
                return 1.2
            elif win_rate >= 0.5 and profit_factor >= 1.5:
                return 1.1
            elif win_rate < 0.4 or profit_factor < 1.0:
                return 0.8
                
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Performance multiplier calculation error: {str(e)}")
            return 1.0
            
    async def _check_profit_taking(self) -> None:
        """Check and execute profit taking if conditions are met"""
        try:
            current_balance = self._get_account_balance()
            total_profit = current_balance - self.initial_balance
            
            # Check if profit meets threshold
            if total_profit >= self.profit_params['profit_threshold']:
                # Calculate profit to take
                profit_to_take = total_profit * self.profit_params['profit_taking_percentage']
                
                # Check daily profit taking limit
                daily_profit_taking = sum(
                    pt['amount'] for pt in self.profit_taking_history
                    if (datetime.now() - pt['timestamp']).days == 0
                )
                
                max_daily_taking = current_balance * self.profit_params['max_daily_profit_taking']
                
                if daily_profit_taking + profit_to_take <= max_daily_taking:
                    # Execute profit taking
                    await self._execute_profit_taking(profit_to_take)
                    
        except Exception as e:
            self.logger.error(f"Profit taking check error: {str(e)}")
            
    async def _execute_profit_taking(self, amount: float) -> None:
        """Execute profit taking operation"""
        try:
            # Record profit taking
            self.profit_taking_history.append({
                'timestamp': datetime.now(),
                'amount': amount,
                'balance': self._get_account_balance()
            })
            
            # Log profit taking
            self.logger.info(
                f"Executed profit taking: ${amount:.2f}"
            )
            
            # Update initial balance for next profit calculation
            self.initial_balance = self._get_account_balance() - amount
            
        except Exception as e:
            self.logger.error(f"Profit taking execution error: {str(e)}")
            
    async def _update_risk_parameters(self) -> None:
        """Update risk parameters based on performance"""
        try:
            # Get recent performance metrics
            performance = await self._calculate_performance_metrics()
            
            # Adjust base risk percentage
            if performance['win_rate'] > 0.6 and performance['profit_factor'] > 2.0:
                self.compounding_params['base_risk_percentage'] = min(
                    self.compounding_params['base_risk_percentage'] + self.compounding_params['risk_step_size'],
                    self.compounding_params['max_risk_percentage']
                )
            elif performance['win_rate'] < 0.4 or performance['profit_factor'] < 1.0:
                self.compounding_params['base_risk_percentage'] = max(
                    self.compounding_params['base_risk_percentage'] - self.compounding_params['risk_step_size'],
                    self.compounding_params['min_risk_percentage']
                )
                
        except Exception as e:
            self.logger.error(f"Risk parameter update error: {str(e)}")
            
    async def _calculate_performance_metrics(self) -> Dict:
        """Calculate recent performance metrics"""
        try:
            if not self.performance_history:
                return {
                    'win_rate': 0,
                    'profit_factor': 0,
                    'average_profit': 0,
                    'max_drawdown': 0
                }
                
            recent_trades = self.performance_history[-50:]
            
            # Calculate metrics
            wins = sum(1 for t in recent_trades if t['profit'] > 0)
            win_rate = wins / len(recent_trades)
            
            gross_profit = sum(t['profit'] for t in recent_trades if t['profit'] > 0)
            gross_loss = abs(sum(t['profit'] for t in recent_trades if t['profit'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            average_profit = sum(t['profit'] for t in recent_trades) / len(recent_trades)
            
            # Calculate max drawdown
            balances = [t['balance'] for t in recent_trades]
            max_dd = 0
            peak = balances[0]
            for balance in balances:
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak
                max_dd = max(max_dd, dd)
                
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'average_profit': average_profit,
                'max_drawdown': max_dd
            }
            
        except Exception as e:
            self.logger.error(f"Performance metrics calculation error: {str(e)}")
            return {}
            
    async def _log_scaling_metrics(self) -> None:
        """Log current scaling metrics"""
        try:
            current_balance = self._get_account_balance()
            total_profit = current_balance - self.initial_balance
            profit_taken = sum(pt['amount'] for pt in self.profit_taking_history)
            
            metrics = {
                'current_balance': current_balance,
                'total_profit': total_profit,
                'profit_taken': profit_taken,
                'current_risk_percentage': self.compounding_params['base_risk_percentage'],
                'trades_taken': len(self.performance_history),
                'profit_taking_events': len(self.profit_taking_history)
            }
            
            self.logger.info(f"Scaling metrics: {metrics}")
            
        except Exception as e:
            self.logger.error(f"Metrics logging error: {str(e)}")
            
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
            
    async def get_scaling_metrics(self) -> Dict:
        """Get current scaling metrics"""
        try:
            current_balance = self._get_account_balance()
            performance = await self._calculate_performance_metrics()
            
            return {
                'initial_balance': self.initial_balance,
                'current_balance': current_balance,
                'total_profit': current_balance - self.initial_balance,
                'profit_taken': sum(pt['amount'] for pt in self.profit_taking_history),
                'current_risk_percentage': self.compounding_params['base_risk_percentage'],
                'performance_metrics': performance,
                'profit_taking_events': len(self.profit_taking_history),
                'trades_taken': len(self.performance_history)
            }
            
        except Exception as e:
            self.logger.error(f"Scaling metrics retrieval error: {str(e)}")
            return {}
