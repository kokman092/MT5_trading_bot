from typing import Dict, Optional
import logging
import numpy as np
from datetime import datetime, timedelta

class ScalingManager:
    def __init__(self, config: Dict):
        self.config = config
        self.initial_balance = config['INITIAL_BALANCE']
        self.current_balance = self.initial_balance
        self.risk_levels = self._initialize_risk_levels()
        self.performance_history = []
        self.last_adjustment = datetime.now()
        
    def _initialize_risk_levels(self) -> Dict:
        """Initialize risk levels based on account size"""
        return {
            'micro': {
                'max_balance': 100,
                'risk_percentage': 2.0,
                'max_positions': 2
            },
            'mini': {
                'max_balance': 1000,
                'risk_percentage': 1.5,
                'max_positions': 3
            },
            'standard': {
                'max_balance': 10000,
                'risk_percentage': 1.0,
                'max_positions': 5
            },
            'professional': {
                'max_balance': float('inf'),
                'risk_percentage': 0.5,
                'max_positions': 8
            }
        }
        
    def update_account_metrics(
        self,
        balance: float,
        equity: float,
        profit: float,
        trades_count: int
    ):
        """Update account performance metrics"""
        self.current_balance = balance
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'balance': balance,
            'equity': equity,
            'profit': profit,
            'trades_count': trades_count
        })
        
        # Keep only last 30 days of history
        cutoff_date = datetime.now() - timedelta(days=30)
        self.performance_history = [
            p for p in self.performance_history
            if p['timestamp'] > cutoff_date
        ]
        
    def get_current_risk_level(self) -> Dict:
        """Get appropriate risk level based on current balance"""
        for level_name, level_params in self.risk_levels.items():
            if self.current_balance <= level_params['max_balance']:
                return {
                    'level': level_name,
                    'params': level_params
                }
        return {
            'level': 'professional',
            'params': self.risk_levels['professional']
        }
        
    def calculate_position_size(self, price: float, stop_loss: float) -> float:
        """Calculate scaled position size based on current risk level"""
        risk_level = self.get_current_risk_level()
        risk_percentage = risk_level['params']['risk_percentage']
        
        # Calculate risk amount
        risk_amount = self.current_balance * (risk_percentage / 100)
        
        # Calculate position size based on stop loss distance
        stop_distance = abs(price - stop_loss)
        position_size = risk_amount / stop_distance
        
        return self._adjust_position_size(position_size)
        
    def _adjust_position_size(self, base_size: float) -> float:
        """Adjust position size based on recent performance"""
        if len(self.performance_history) < 10:
            return base_size
            
        # Calculate win rate and profit factor
        recent_trades = self.performance_history[-10:]
        profits = [t['profit'] for t in recent_trades]
        win_rate = len([p for p in profits if p > 0]) / len(profits)
        
        # Adjust size based on performance
        if win_rate > 0.6:
            return base_size * 1.2
        elif win_rate < 0.4:
            return base_size * 0.8
            
        return base_size
        
    def should_adjust_parameters(self) -> bool:
        """Check if trading parameters should be adjusted"""
        if datetime.now() - self.last_adjustment < timedelta(days=1):
            return False
            
        if len(self.performance_history) < 20:
            return False
            
        return True
        
    def get_adjusted_parameters(self) -> Optional[Dict]:
        """Get adjusted trading parameters based on performance"""
        if not self.should_adjust_parameters():
            return None
            
        try:
            # Calculate performance metrics
            recent_trades = self.performance_history[-20:]
            profits = [t['profit'] for t in recent_trades]
            win_rate = len([p for p in profits if p > 0]) / len(profits)
            avg_profit = np.mean(profits)
            profit_factor = (
                sum(p for p in profits if p > 0) /
                abs(sum(p for p in profits if p < 0))
                if any(p < 0 for p in profits) else float('inf')
            )
            
            # Adjust parameters based on performance
            current_risk = self.get_current_risk_level()
            adjusted_params = current_risk['params'].copy()
            
            if profit_factor > 2.0 and win_rate > 0.6:
                # Increase risk if performing well
                adjusted_params['risk_percentage'] *= 1.2
                adjusted_params['max_positions'] += 1
            elif profit_factor < 1.5 or win_rate < 0.4:
                # Decrease risk if performing poorly
                adjusted_params['risk_percentage'] *= 0.8
                adjusted_params['max_positions'] = max(1, adjusted_params['max_positions'] - 1)
                
            # Cap adjustments
            adjusted_params['risk_percentage'] = min(
                max(0.5, adjusted_params['risk_percentage']),
                3.0
            )
            adjusted_params['max_positions'] = min(
                max(1, adjusted_params['max_positions']),
                10
            )
            
            self.last_adjustment = datetime.now()
            return adjusted_params
            
        except Exception as e:
            logging.error(f"Error adjusting parameters: {str(e)}")
            return None
            
    def get_scaling_metrics(self) -> Dict:
        """Get current scaling metrics for monitoring"""
        try:
            current_risk = self.get_current_risk_level()
            
            return {
                'current_balance': self.current_balance,
                'risk_level': current_risk['level'],
                'risk_percentage': current_risk['params']['risk_percentage'],
                'max_positions': current_risk['params']['max_positions'],
                'growth_rate': self._calculate_growth_rate(),
                'last_adjustment': self.last_adjustment.isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error getting scaling metrics: {str(e)}")
            return {}
            
    def _calculate_growth_rate(self) -> float:
        """Calculate account growth rate"""
        if len(self.performance_history) < 2:
            return 0.0
            
        initial = self.performance_history[0]['balance']
        final = self.performance_history[-1]['balance']
        days = (self.performance_history[-1]['timestamp'] -
                self.performance_history[0]['timestamp']).days
                
        if days == 0 or initial == 0:
            return 0.0
            
        return ((final / initial) ** (365 / days) - 1) * 100  # Annualized growth rate
