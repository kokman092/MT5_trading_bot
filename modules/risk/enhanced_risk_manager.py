import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

class EnhancedRiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.params = {
            'max_account_risk': config.get('max_account_risk', 0.02),  # 2% per trade
            'max_daily_drawdown': config.get('max_daily_drawdown', 0.05),  # 5% daily
            'max_total_drawdown': config.get('max_total_drawdown', 0.15),  # 15% total
            'position_correlation_limit': config.get('position_correlation_limit', 0.7),
            'max_leverage': config.get('max_leverage', 20),
            'min_free_margin': config.get('min_free_margin', 0.3),  # 30% free margin
            'max_positions': config.get('max_positions', 10),
            'max_strategy_allocation': config.get('max_strategy_allocation', 0.3),
            'volatility_scaling': config.get('volatility_scaling', True)
        }
        
        # Initialize tracking
        self.daily_stats = {
            'pnl': 0,
            'max_equity': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0
        }
        self.positions = {}
        self.strategy_allocations = {}
        self.risk_metrics = {}

    def validate_trade(self, trade_request: Dict, 
                      account_info: Dict, 
                      market_data: Dict) -> Tuple[bool, str]:
        """
        Validate trade against all risk parameters
        """
        try:
            # Check account-level risks
            if not self._check_account_risk(trade_request, account_info):
                return False, "Account risk limit exceeded"
                
            # Check position-level risks
            if not self._check_position_risk(trade_request, market_data):
                return False, "Position risk too high"
                
            # Check portfolio-level risks
            if not self._check_portfolio_risk(trade_request):
                return False, "Portfolio risk limit exceeded"
                
            # Check strategy-level risks
            if not self._check_strategy_risk(trade_request):
                return False, "Strategy allocation limit exceeded"
                
            # All checks passed
            return True, "Trade validated"
            
        except Exception as e:
            self.logger.error(f"Error in trade validation: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def _check_account_risk(self, trade_request: Dict, 
                          account_info: Dict) -> bool:
        """
        Check account-level risk parameters
        """
        # Calculate required margin
        required_margin = self._calculate_required_margin(
            trade_request, account_info
        )
        
        # Check free margin
        if required_margin > account_info['free_margin']:
            return False
            
        # Check leverage
        current_leverage = self._calculate_total_leverage(
            account_info, required_margin
        )
        if current_leverage > self.params['max_leverage']:
            return False
            
        # Check daily drawdown
        if self.daily_stats['pnl'] < -self.params['max_daily_drawdown'] * \
           account_info['equity']:
            return False
            
        return True

    def _check_position_risk(self, trade_request: Dict, 
                           market_data: Dict) -> bool:
        """
        Check position-level risk parameters
        """
        # Calculate position size after volatility scaling
        if self.params['volatility_scaling']:
            position_size = self._scale_by_volatility(
                trade_request['volume'],
                market_data['volatility']
            )
        else:
            position_size = trade_request['volume']
            
        # Check correlation with existing positions
        if not self._check_position_correlation(
            trade_request['symbol'],
            market_data['correlations']
        ):
            return False
            
        # Check maximum positions per symbol
        if not self._check_symbol_positions(trade_request['symbol']):
            return False
            
        return True

    def _check_portfolio_risk(self, trade_request: Dict) -> bool:
        """
        Check portfolio-level risk parameters
        """
        # Check total number of positions
        if len(self.positions) >= self.params['max_positions']:
            return False
            
        # Check portfolio heat (total risk exposure)
        if self._calculate_portfolio_heat() > 1.0:
            return False
            
        return True

    def _check_strategy_risk(self, trade_request: Dict) -> bool:
        """
        Check strategy-level risk parameters
        """
        strategy = trade_request['strategy']
        
        # Check strategy allocation
        current_allocation = self.strategy_allocations.get(strategy, 0)
        if current_allocation >= self.params['max_strategy_allocation']:
            return False
            
        return True

    def _calculate_required_margin(self, trade_request: Dict, 
                                 account_info: Dict) -> float:
        """
        Calculate required margin for trade
        """
        # Implement margin calculation based on broker's requirements
        return trade_request['volume'] * trade_request['margin_rate']

    def _calculate_total_leverage(self, account_info: Dict, 
                                additional_margin: float) -> float:
        """
        Calculate total account leverage
        """
        total_margin = account_info['margin'] + additional_margin
        return total_margin / account_info['equity']

    def _scale_by_volatility(self, base_size: float, 
                           volatility: float) -> float:
        """
        Scale position size by market volatility
        """
        vol_factor = 1 / volatility if volatility > 0 else 1
        return base_size * min(vol_factor, 2.0)  # Cap at 2x scaling

    def _check_position_correlation(self, symbol: str, 
                                  correlations: Dict) -> bool:
        """
        Check correlation with existing positions
        """
        for pos_symbol in self.positions:
            if pos_symbol in correlations:
                if abs(correlations[pos_symbol][symbol]) > \
                   self.params['position_correlation_limit']:
                    return False
        return True

    def _check_symbol_positions(self, symbol: str) -> bool:
        """
        Check maximum positions per symbol
        """
        symbol_positions = sum(
            1 for pos in self.positions.values() 
            if pos['symbol'] == symbol
        )
        return symbol_positions < 3  # Max 3 positions per symbol

    def _calculate_portfolio_heat(self) -> float:
        """
        Calculate total portfolio risk exposure
        """
        total_heat = sum(
            pos['risk_contribution'] 
            for pos in self.positions.values()
        )
        return total_heat

    def update_position(self, position: Dict):
        """
        Update position tracking
        """
        self.positions[position['ticket']] = position
        
        # Update strategy allocations
        strategy = position['strategy']
        self.strategy_allocations[strategy] = sum(
            pos['margin'] for pos in self.positions.values()
            if pos['strategy'] == strategy
        )

    def update_daily_stats(self, trade_result: Dict):
        """
        Update daily trading statistics
        """
        self.daily_stats['pnl'] += trade_result['profit']
        self.daily_stats['trades'] += 1
        
        if trade_result['profit'] > 0:
            self.daily_stats['wins'] += 1
        else:
            self.daily_stats['losses'] += 1

    def calculate_risk_metrics(self) -> Dict:
        """
        Calculate current risk metrics
        """
        self.risk_metrics = {
            'daily_pnl': self.daily_stats['pnl'],
            'win_rate': (self.daily_stats['wins'] / 
                        max(self.daily_stats['trades'], 1)),
            'portfolio_heat': self._calculate_portfolio_heat(),
            'largest_position': max(
                (pos['margin'] for pos in self.positions.values()),
                default=0
            ),
            'strategy_diversification': len(self.strategy_allocations)
        }
        
        return self.risk_metrics

    def reset_daily_stats(self):
        """
        Reset daily statistics
        """
        self.daily_stats = {
            'pnl': 0,
            'max_equity': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0
        }
