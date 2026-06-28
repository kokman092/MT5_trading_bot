from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from ..risk.risk_manager import RiskManager
from ..portfolio.portfolio_manager import PortfolioManager
from ..deployment.error_handler import ErrorHandler

class ScalingManager:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('scaling_manager')
        self.risk_manager = RiskManager(config)
        self.portfolio_manager = PortfolioManager(config)
        
        # Initialize scaling parameters
        self._init_scaling_params()
        
    def _init_scaling_params(self):
        """Initialize scaling and growth parameters"""
        self.scaling_params = {
            # Account size brackets and corresponding risk %
            'risk_brackets': {
                10000: 1.0,     # Up to $10k: 1.0% risk
                50000: 0.8,     # Up to $50k: 0.8% risk
                100000: 0.6,    # Up to $100k: 0.6% risk
                500000: 0.4,    # Up to $500k: 0.4% risk
                1000000: 0.2    # Up to $1M: 0.2% risk
            },
            
            # Position size scaling factors
            'position_scaling': {
                10000: 1.0,     # Base position size
                50000: 1.2,     # 20% increase
                100000: 1.5,    # 50% increase
                500000: 2.0,    # 100% increase
                1000000: 3.0    # 200% increase
            },
            
            # Profit reinvestment ratios
            'reinvestment_ratios': {
                'trading': 0.7,      # 70% for trading
                'reserve': 0.2,      # 20% for reserve
                'development': 0.1   # 10% for strategy development
            },
            
            # Growth targets and milestones
            'milestones': [
                1000,      # First $1k
                5000,      # $5k
                10000,     # $10k
                50000,     # $50k
                100000,    # $100k
                500000,    # $500k
                1000000    # $1M
            ]
        }
        
    async def calculate_current_risk_percentage(self, account_balance: float) -> float:
        """Calculate appropriate risk percentage based on account size"""
        try:
            # Find appropriate risk bracket
            risk_percent = self.scaling_params['risk_brackets'][1000000]  # Default to lowest
            
            for balance_threshold, risk in sorted(
                self.scaling_params['risk_brackets'].items()
            ):
                if account_balance <= balance_threshold:
                    risk_percent = risk
                    break
                    
            # Adjust risk based on market conditions
            risk_percent = await self._adjust_risk_for_market_conditions(risk_percent)
            
            # Validate with risk manager
            if await self._validate_risk_percentage(risk_percent, account_balance):
                return risk_percent
            return self.config.get('min_risk_percentage', 0.5)
            
        except Exception as e:
            self.logger.error(f"Risk percentage calculation error: {str(e)}")
            return self.config.get('min_risk_percentage', 0.5)
            
    async def _adjust_risk_for_market_conditions(self, base_risk: float) -> float:
        """Adjust risk percentage based on market conditions"""
        try:
            # Get market volatility
            volatility = await self._get_market_volatility()
            
            # Adjust risk based on volatility
            if volatility > 0.8:  # High volatility
                return base_risk * 0.7  # Reduce risk by 30%
            elif volatility < 0.2:  # Low volatility
                return base_risk * 1.2  # Increase risk by 20%
                
            return base_risk
            
        except Exception as e:
            self.logger.error(f"Risk adjustment error: {str(e)}")
            return base_risk
            
    async def calculate_position_size(
        self,
        symbol: str,
        account_balance: float,
        risk_percentage: float
    ) -> float:
        """Calculate scaled position size"""
        try:
            # Get base position size
            base_size = account_balance * (risk_percentage / 100)
            
            # Get scaling factor based on account size
            scaling_factor = 1.0  # Default
            for balance_threshold, factor in sorted(
                self.scaling_params['position_scaling'].items()
            ):
                if account_balance <= balance_threshold:
                    scaling_factor = factor
                    break
                    
            # Apply scaling
            scaled_size = base_size * scaling_factor
            
            # Validate with risk manager
            max_position_size = await self.risk_manager.calculate_max_position_size(
                symbol,
                account_balance
            )
            
            return min(scaled_size, max_position_size)
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {str(e)}")
            return 0.0
            
    async def allocate_profits(self, profit_amount: float) -> Dict[str, float]:
        """Allocate profits according to reinvestment strategy"""
        try:
            allocations = {}
            ratios = self.scaling_params['reinvestment_ratios']
            
            # Calculate allocations
            for category, ratio in ratios.items():
                allocations[category] = profit_amount * ratio
                
            # Validate allocations
            if await self._validate_allocations(allocations):
                return allocations
                
            # Return default allocations if validation fails
            return {
                'trading': profit_amount * 0.7,
                'reserve': profit_amount * 0.2,
                'development': profit_amount * 0.1
            }
            
        except Exception as e:
            self.logger.error(f"Profit allocation error: {str(e)}")
            return {
                'trading': profit_amount,
                'reserve': 0.0,
                'development': 0.0
            }
            
    async def track_growth_progress(self, account_balance: float) -> Dict:
        """Track progress towards growth milestones"""
        try:
            progress = {
                'current_balance': account_balance,
                'next_milestone': None,
                'progress_percentage': 0.0,
                'milestones_reached': [],
                'remaining_milestones': []
            }
            
            # Find next milestone
            for milestone in sorted(self.scaling_params['milestones']):
                if account_balance < milestone:
                    progress['next_milestone'] = milestone
                    progress['progress_percentage'] = (account_balance / milestone) * 100
                    break
                else:
                    progress['milestones_reached'].append(milestone)
                    
            # Get remaining milestones
            if progress['next_milestone']:
                milestone_index = self.scaling_params['milestones'].index(
                    progress['next_milestone']
                )
                progress['remaining_milestones'] = self.scaling_params['milestones'][
                    milestone_index:
                ]
                
            # Calculate estimated time to next milestone
            if progress['next_milestone']:
                progress['estimated_time'] = await self._estimate_time_to_milestone(
                    account_balance,
                    progress['next_milestone']
                )
                
            return progress
            
        except Exception as e:
            self.logger.error(f"Growth progress tracking error: {str(e)}")
            return {
                'current_balance': account_balance,
                'error': str(e)
            }
            
    async def _estimate_time_to_milestone(
        self,
        current_balance: float,
        target_balance: float
    ) -> int:
        """Estimate days to reach next milestone based on current performance"""
        try:
            # Get historical returns
            returns = await self._get_historical_returns()
            
            if not returns:
                return None
                
            # Calculate average daily return
            avg_daily_return = np.mean(returns)
            
            if avg_daily_return <= 0:
                return None
                
            # Calculate time using compound interest formula
            time_days = np.log(target_balance / current_balance) / np.log(1 + avg_daily_return)
            
            return int(np.ceil(time_days))
            
        except Exception as e:
            self.logger.error(f"Time estimation error: {str(e)}")
            return None
            
    async def _get_historical_returns(self) -> List[float]:
        """Get historical daily returns"""
        try:
            # Get last 30 days of history
            history = await self.portfolio_manager.get_account_history(30)
            
            if not history:
                return []
                
            # Calculate daily returns
            balances = pd.Series(history['balance'])
            returns = balances.pct_change().dropna().values.tolist()
            
            return returns
            
        except Exception as e:
            self.logger.error(f"Historical returns error: {str(e)}")
            return []
            
    async def _get_market_volatility(self) -> float:
        """Calculate current market volatility"""
        try:
            # Get volatility for major symbols
            symbols = self.config.get('symbols', [])
            volatilities = []
            
            for symbol in symbols:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 20)
                if rates is not None:
                    df = pd.DataFrame(rates)
                    returns = np.log(df['close'] / df['close'].shift(1))
                    volatility = returns.std()
                    volatilities.append(volatility)
                    
            if volatilities:
                return np.mean(volatilities)
            return 0.5  # Default moderate volatility
            
        except Exception as e:
            self.logger.error(f"Market volatility calculation error: {str(e)}")
            return 0.5
            
    async def _validate_risk_percentage(
        self,
        risk_percent: float,
        account_balance: float
    ) -> bool:
        """Validate if risk percentage is appropriate"""
        try:
            # Check against maximum allowed risk
            if risk_percent > self.config.get('max_risk_percentage', 2.0):
                return False
                
            # Check against minimum balance requirements
            min_balance = self.config.get('min_balance_for_trading', 10.0)
            if account_balance < min_balance:
                return False
                
            # Check market conditions
            market_status = await self.risk_manager.validate_technical_health()
            if not market_status['is_healthy']:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Risk validation error: {str(e)}")
            return False
            
    async def _validate_allocations(self, allocations: Dict[str, float]) -> bool:
        """Validate profit allocations"""
        try:
            # Check if allocations sum to 100%
            total_allocation = sum(allocations.values())
            if not np.isclose(total_allocation, sum(allocations.values())):
                return False
                
            # Check minimum trading allocation
            if allocations.get('trading', 0) < self.config.get('min_trading_allocation', 0.5):
                return False
                
            # Check minimum reserve allocation
            if allocations.get('reserve', 0) < self.config.get('min_reserve_allocation', 0.1):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Allocation validation error: {str(e)}")
            return False
