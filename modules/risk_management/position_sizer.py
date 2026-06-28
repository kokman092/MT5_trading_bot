from typing import Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PositionSizer:
    def __init__(self, config: Dict):
        self.config = config
        self.risk_per_trade = config['RISK_MANAGEMENT']['max_risk_per_trade']
        self.max_daily_loss = config['RISK_MANAGEMENT']['max_daily_loss']
        self.max_positions = config['RISK_MANAGEMENT']['max_positions']
        self.min_margin_level = config['RISK_MANAGEMENT']['min_margin_level']
        self.position_history = []
        
    def calculate_position_size(self, 
                              account_info: Dict,
                              signal: Dict,
                              market_info: Dict) -> Optional[float]:
        """Calculate optimal position size based on risk management rules"""
        
        # Check if trading is allowed
        if not self._can_open_position(account_info):
            return None
            
        # Calculate position size based on risk
        risk_amount = account_info['balance'] * self.risk_per_trade
        stop_loss_pips = abs(signal['entry_price'] - signal['stop_loss']) / market_info['point']
        
        # Calculate base position size
        pip_value = market_info['pip_value']
        if pip_value <= 0 or stop_loss_pips <= 0:
            return None
            
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Apply position size constraints
        position_size = self._apply_position_constraints(
            position_size,
            account_info,
            market_info
        )
        
        return position_size
        
    def _can_open_position(self, account_info: Dict) -> bool:
        """Check if new position can be opened based on risk management rules"""
        
        # Check margin level
        if account_info['margin_level'] < self.min_margin_level:
            return False
            
        # Check max positions
        if len(account_info['positions']) >= self.max_positions:
            return False
            
        # Check daily loss limit
        daily_loss = self._calculate_daily_loss(account_info)
        if daily_loss >= self.max_daily_loss * account_info['balance']:
            return False
            
        return True
        
    def _apply_position_constraints(self,
                                  position_size: float,
                                  account_info: Dict,
                                  market_info: Dict) -> float:
        """Apply various constraints to position size"""
        
        # Minimum position size
        position_size = max(position_size, market_info['min_lot'])
        
        # Maximum position size based on margin
        max_size = self._calculate_max_size(account_info, market_info)
        position_size = min(position_size, max_size)
        
        # Round to valid lot size
        position_size = round(position_size / market_info['lot_step']) * market_info['lot_step']
        
        return position_size
        
    def _calculate_max_size(self,
                          account_info: Dict,
                          market_info: Dict) -> float:
        """Calculate maximum position size based on available margin"""
        
        # Get required margin per lot
        margin_per_lot = market_info['contract_size'] * market_info['margin_rate']
        
        # Calculate free margin with buffer
        free_margin = account_info['margin_free'] * 0.9  # 10% margin buffer
        
        # Calculate maximum position size
        max_size = free_margin / margin_per_lot
        
        return max_size
        
    def _calculate_daily_loss(self, account_info: Dict) -> float:
        """Calculate total loss for current trading day"""
        
        today = datetime.now().date()
        daily_loss = 0.0
        
        # Sum up losses from closed positions today
        for position in self.position_history:
            if position['close_time'].date() == today and position['profit'] < 0:
                daily_loss += abs(position['profit'])
                
        # Add unrealized losses from open positions
        for position in account_info['positions']:
            if position['profit'] < 0:
                daily_loss += abs(position['profit'])
                
        return daily_loss
        
    def update_position_history(self, position: Dict):
        """Update position history for tracking daily losses"""
        self.position_history.append(position)
        
        # Remove positions older than 7 days
        week_ago = datetime.now() - timedelta(days=7)
        self.position_history = [
            p for p in self.position_history
            if p['close_time'] > week_ago
        ]
        
    def calculate_correlation_risk(self, positions: list) -> float:
        """Calculate portfolio correlation risk"""
        if len(positions) < 2:
            return 0.0
            
        # Get position returns
        returns = []
        for position in positions:
            if 'price_history' in position:
                returns.append(pd.Series(position['price_history']))
                
        if len(returns) < 2:
            return 0.0
            
        # Calculate correlation matrix
        corr_matrix = pd.concat(returns, axis=1).corr()
        
        # Calculate average correlation
        n = len(corr_matrix)
        total_corr = 0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_corr += abs(corr_matrix.iloc[i, j])
                count += 1
                
        avg_corr = total_corr / count if count > 0 else 0
        return avg_corr
        
    def adjust_for_correlation(self, position_size: float, positions: list) -> float:
        """Adjust position size based on portfolio correlation"""
        correlation_risk = self.calculate_correlation_risk(positions)
        
        # Reduce position size as correlation increases
        correlation_factor = 1 - (correlation_risk * 0.5)  # Max 50% reduction
        
        return position_size * correlation_factor
        
    def calculate_kelly_criterion(self,
                                win_rate: float,
                                profit_factor: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        
        if win_rate <= 0 or win_rate >= 1 or profit_factor <= 0:
            return 0.0
            
        # Kelly formula: f = (bp - q) / b
        # where: b = profit_factor
        #        p = win_rate
        #        q = 1 - p
        
        q = 1 - win_rate
        kelly_pct = (profit_factor * win_rate - q) / profit_factor
        
        # Fractional Kelly to be more conservative
        fractional_kelly = kelly_pct * 0.5
        
        return max(0.0, min(fractional_kelly, self.risk_per_trade))
