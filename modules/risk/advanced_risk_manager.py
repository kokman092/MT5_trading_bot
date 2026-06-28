from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from ..analytics.market_analyzer import MarketAnalyzer
from ..deployment.error_handler import ErrorHandler

class AdvancedRiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('advanced_risk_manager')
        self.market_analyzer = MarketAnalyzer(config)
        
        # Initialize risk parameters
        self._init_risk_parameters()
        
        # Trading state
        self.daily_pnl = 0.0
        self.open_positions = {}
        self.last_reset = datetime.now()
        
    def _init_risk_parameters(self):
        """Initialize risk management parameters"""
        # Kelly Criterion parameters
        self.kelly_params = {
            'max_kelly_fraction': 0.2,  # Maximum allowed Kelly fraction
            'kelly_scaling': 0.5,  # Scale Kelly fraction for safety
            'min_win_rate': 0.55,  # Minimum required win rate
            'lookback_period': 100  # Number of trades for win rate calculation
        }
        
        # Daily loss limits
        self.daily_limits = {
            'max_daily_loss': 0.03,  # 3% max daily loss
            'warning_threshold': 0.02,  # 2% warning threshold
            'intraday_recovery': 0.01  # 1% recovery threshold
        }
        
        # Portfolio heat
        self.portfolio_limits = {
            'max_heat': 0.10,  # 10% maximum portfolio heat
            'position_limit': 0.02,  # 2% maximum per position
            'correlation_limit': 0.7,  # Maximum correlation between positions
            'sector_limit': 0.25  # Maximum exposure per sector
        }
        
        # Trade history
        self.trade_history = []
        
    async def calculate_position_size(
        self,
        symbol: str,
        strategy_metrics: Dict
    ) -> Optional[float]:
        """Calculate optimal position size using Kelly Criterion"""
        try:
            # Get account info
            account_info = mt5.account_info()
            if not account_info:
                return None
                
            equity = account_info.equity
            
            # Calculate Kelly fraction
            kelly_fraction = await self._calculate_kelly_fraction(strategy_metrics)
            
            # Apply portfolio heat constraints
            available_heat = await self._get_available_heat()
            max_position_size = min(
                equity * kelly_fraction,
                equity * self.portfolio_limits['position_limit'],
                equity * available_heat
            )
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return None
                
            # Calculate lots based on position size
            contract_size = symbol_info.trade_contract_size
            current_price = (symbol_info.ask + symbol_info.bid) / 2
            
            lots = max_position_size / (contract_size * current_price)
            
            # Round to symbol's lot step
            lots = round(lots / symbol_info.volume_step) * symbol_info.volume_step
            
            return lots
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {str(e)}")
            return None
            
    async def validate_trade(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float,
        stop_loss: float,
        take_profit: float
    ) -> Tuple[bool, str]:
        """Validate trade against risk management rules"""
        try:
            # Check daily loss limit
            if not await self._check_daily_limit():
                return False, "Daily loss limit reached"
                
            # Calculate trade risk
            risk_amount = await self._calculate_trade_risk(
                symbol, volume, price, stop_loss
            )
            
            # Check position limit
            if not await self._check_position_limit(risk_amount):
                return False, "Position size limit exceeded"
                
            # Check portfolio heat
            if not await self._check_portfolio_heat(symbol, risk_amount):
                return False, "Portfolio heat limit exceeded"
                
            # Check correlation limit
            if not await self._check_correlation_limit(symbol):
                return False, "Correlation limit exceeded"
                
            # Validate risk/reward ratio
            if not self._validate_risk_reward(price, stop_loss, take_profit):
                return False, "Invalid risk/reward ratio"
                
            return True, "Trade validated"
            
        except Exception as e:
            self.logger.error(f"Trade validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"
            
    async def update_trade_metrics(self, trade_result: Dict) -> None:
        """Update trade history and metrics"""
        try:
            # Update trade history
            self.trade_history.append(trade_result)
            
            # Update daily P&L
            self.daily_pnl += trade_result['profit']
            
            # Update open positions if needed
            symbol = trade_result['symbol']
            if trade_result['type'] == 'open':
                self.open_positions[symbol] = trade_result
            elif trade_result['type'] == 'close' and symbol in self.open_positions:
                del self.open_positions[symbol]
                
            # Check for daily reset
            await self._check_daily_reset()
            
        except Exception as e:
            self.logger.error(f"Trade metrics update error: {str(e)}")
            
    async def _calculate_kelly_fraction(self, strategy_metrics: Dict) -> float:
        """Calculate Kelly Criterion fraction"""
        try:
            # Extract metrics
            win_rate = strategy_metrics.get('win_rate', 0)
            avg_win = strategy_metrics.get('average_win', 0)
            avg_loss = strategy_metrics.get('average_loss', 0)
            
            # Check minimum win rate
            if win_rate < self.kelly_params['min_win_rate']:
                return 0
                
            # Calculate odds
            if avg_loss == 0:
                return 0
            odds = abs(avg_win / avg_loss)
            
            # Kelly formula
            kelly_fraction = (odds * win_rate - (1 - win_rate)) / odds
            
            # Apply safety scaling
            kelly_fraction *= self.kelly_params['kelly_scaling']
            
            # Apply maximum limit
            kelly_fraction = min(
                kelly_fraction,
                self.kelly_params['max_kelly_fraction']
            )
            
            return max(0, kelly_fraction)
            
        except Exception as e:
            self.logger.error(f"Kelly fraction calculation error: {str(e)}")
            return 0
            
    async def _check_daily_limit(self) -> bool:
        """Check if daily loss limit is reached"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return False
                
            equity = account_info.equity
            
            # Calculate daily loss percentage
            daily_loss_pct = -self.daily_pnl / equity
            
            # Check against limits
            if daily_loss_pct >= self.daily_limits['max_daily_loss']:
                self.logger.warning("Daily loss limit reached")
                return False
                
            if daily_loss_pct >= self.daily_limits['warning_threshold']:
                self.logger.warning("Approaching daily loss limit")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Daily limit check error: {str(e)}")
            return False
            
    async def _get_available_heat(self) -> float:
        """Calculate available portfolio heat"""
        try:
            total_heat = sum(
                position['risk_amount']
                for position in self.open_positions.values()
            )
            
            account_info = mt5.account_info()
            if not account_info:
                return 0
                
            equity = account_info.equity
            current_heat = total_heat / equity
            
            return max(0, self.portfolio_limits['max_heat'] - current_heat)
            
        except Exception as e:
            self.logger.error(f"Available heat calculation error: {str(e)}")
            return 0
            
    async def _calculate_trade_risk(
        self,
        symbol: str,
        volume: float,
        price: float,
        stop_loss: float
    ) -> float:
        """Calculate risk amount for a trade"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0
                
            # Calculate risk per pip
            contract_size = symbol_info.trade_contract_size
            risk_per_pip = contract_size * volume
            
            # Calculate risk amount
            risk_amount = abs(price - stop_loss) * risk_per_pip
            
            return risk_amount
            
        except Exception as e:
            self.logger.error(f"Trade risk calculation error: {str(e)}")
            return 0
            
    async def _check_position_limit(self, risk_amount: float) -> bool:
        """Check if position size is within limits"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return False
                
            equity = account_info.equity
            position_risk_pct = risk_amount / equity
            
            return position_risk_pct <= self.portfolio_limits['position_limit']
            
        except Exception as e:
            self.logger.error(f"Position limit check error: {str(e)}")
            return False
            
    async def _check_portfolio_heat(
        self,
        symbol: str,
        risk_amount: float
    ) -> bool:
        """Check if total portfolio heat is within limits"""
        try:
            # Calculate total heat including new position
            total_heat = risk_amount
            for position in self.open_positions.values():
                if position['symbol'] != symbol:
                    total_heat += position['risk_amount']
                    
            account_info = mt5.account_info()
            if not account_info:
                return False
                
            equity = account_info.equity
            heat_percentage = total_heat / equity
            
            return heat_percentage <= self.portfolio_limits['max_heat']
            
        except Exception as e:
            self.logger.error(f"Portfolio heat check error: {str(e)}")
            return False
            
    async def _check_correlation_limit(self, symbol: str) -> bool:
        """Check correlation limits with existing positions"""
        try:
            if not self.open_positions:
                return True
                
            # Get correlation data
            symbols = list(self.open_positions.keys()) + [symbol]
            correlation_data = await self.market_analyzer.perform_cluster_analysis(
                symbols
            )
            
            # Check correlations
            for existing_symbol in self.open_positions:
                correlation = correlation_data.get('metrics', {}).get(
                    f"{symbol}_{existing_symbol}", 0
                )
                if abs(correlation) > self.portfolio_limits['correlation_limit']:
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Correlation check error: {str(e)}")
            return True
            
    def _validate_risk_reward(
        self,
        price: float,
        stop_loss: float,
        take_profit: float
    ) -> bool:
        """Validate risk/reward ratio"""
        try:
            if not (price and stop_loss and take_profit):
                return False
                
            risk = abs(price - stop_loss)
            reward = abs(take_profit - price)
            
            if risk == 0:
                return False
                
            rr_ratio = reward / risk
            min_rr_ratio = self.config.get('min_risk_reward_ratio', 2)
            
            return rr_ratio >= min_rr_ratio
            
        except Exception as e:
            self.logger.error(f"Risk/reward validation error: {str(e)}")
            return False
            
    async def _check_daily_reset(self) -> None:
        """Check and perform daily reset if needed"""
        try:
            current_time = datetime.now()
            if current_time.date() > self.last_reset.date():
                # Reset daily P&L
                self.daily_pnl = 0.0
                self.last_reset = current_time
                self.logger.info("Daily metrics reset performed")
                
        except Exception as e:
            self.logger.error(f"Daily reset error: {str(e)}")
            
    async def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return {}
                
            equity = account_info.equity
            
            return {
                'daily_pnl_pct': self.daily_pnl / equity,
                'portfolio_heat': sum(
                    p['risk_amount'] for p in self.open_positions.values()
                ) / equity,
                'open_positions': len(self.open_positions),
                'available_heat': await self._get_available_heat()
            }
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation error: {str(e)}")
            return {}
