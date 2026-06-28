from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from dataclasses import dataclass
from scipy.stats import norm
from ..analytics.market_analyzer import MarketAnalyzer
from ..deployment.error_handler import ErrorHandler

@dataclass
class PositionSize:
    size: float  # Lot size
    risk_amount: float  # Risk amount in account currency
    risk_percentage: float  # Risk as percentage of equity
    kelly_fraction: float  # Kelly criterion fraction
    volatility_adjustment: float  # Volatility adjustment factor
    confidence: float  # Confidence score

@dataclass
class LeverageSignal:
    leverage: float  # Recommended leverage
    max_leverage: float  # Maximum allowed leverage
    sharpe_ratio: float  # Current Sharpe ratio
    win_rate: float  # Historical win rate
    risk_score: float  # Overall risk score
    recommendation: str  # Leverage recommendation

class PositionManager:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('position_manager')
        self.market_analyzer = MarketAnalyzer(config)
        
        # Initialize parameters
        self._init_position_parameters()
        
    def _init_position_parameters(self):
        """Initialize position management parameters"""
        # Position sizing parameters
        self.position_params = {
            'base_risk': 0.01,  # 1% base risk per trade
            'max_risk': 0.02,   # 2% maximum risk per trade
            'min_risk': 0.005,  # 0.5% minimum risk per trade
            'kelly_fraction': 0.5,  # Half-Kelly for conservative sizing
            'volatility_multiplier': 1.5,  # Volatility adjustment factor
            'confidence_threshold': 0.7  # Minimum confidence for full size
        }
        
        # Leverage parameters
        self.leverage_params = {
            'base_leverage': 2.0,  # Base leverage multiplier
            'max_leverage': 5.0,   # Maximum allowed leverage
            'min_leverage': 1.0,   # Minimum leverage (no leverage)
            'sharpe_threshold': 1.5,  # Minimum Sharpe ratio for increased leverage
            'win_rate_threshold': 0.6,  # Minimum win rate for increased leverage
            'drawdown_limit': 0.1,  # 10% drawdown limit for leverage reduction
            'volatility_limit': 2.0  # Volatility limit for leverage reduction
        }
        
        # Performance tracking
        self.performance_metrics = {
            'trades': [],
            'equity_curve': [],
            'win_rates': [],
            'sharpe_ratios': [],
            'drawdowns': []
        }
        
    async def get_position_size(
        self,
        symbol: str,
        strategy: str,
        account_info: Dict,
        market_state: Dict
    ) -> PositionSize:
        """Calculate optimal position size"""
        try:
            # Get base position size
            base_size = await self._calculate_base_size(
                account_info,
                market_state
            )
            
            # Apply Kelly Criterion
            kelly_size = await self._apply_kelly_criterion(
                strategy,
                account_info
            )
            
            # Adjust for volatility
            volatility_adj = await self._calculate_volatility_adjustment(
                symbol,
                market_state
            )
            
            # Calculate final size
            final_size = base_size * kelly_size * volatility_adj
            
            # Apply risk limits
            final_size = await self._apply_risk_limits(
                final_size,
                account_info
            )
            
            return PositionSize(
                size=final_size,
                risk_amount=final_size * account_info['equity'] * self.position_params['base_risk'],
                risk_percentage=self.position_params['base_risk'] * 100,
                kelly_fraction=kelly_size,
                volatility_adjustment=volatility_adj,
                confidence=await self._calculate_confidence_score(market_state)
            )
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {str(e)}")
            return None
            
    async def get_leverage_signal(
        self,
        strategy: str,
        account_info: Dict,
        market_state: Dict
    ) -> LeverageSignal:
        """Calculate optimal leverage"""
        try:
            # Calculate performance metrics
            sharpe = await self._calculate_sharpe_ratio(account_info)
            win_rate = await self._calculate_win_rate(strategy)
            
            # Calculate base leverage
            base_leverage = await self._calculate_base_leverage(
                sharpe,
                win_rate,
                market_state
            )
            
            # Apply risk adjustments
            adjusted_leverage = await self._adjust_leverage_for_risk(
                base_leverage,
                account_info,
                market_state
            )
            
            # Calculate risk score
            risk_score = await self._calculate_risk_score(
                account_info,
                market_state
            )
            
            return LeverageSignal(
                leverage=adjusted_leverage,
                max_leverage=self.leverage_params['max_leverage'],
                sharpe_ratio=sharpe,
                win_rate=win_rate,
                risk_score=risk_score,
                recommendation=await self._generate_leverage_recommendation(
                    adjusted_leverage,
                    risk_score
                )
            )
            
        except Exception as e:
            self.logger.error(f"Leverage signal calculation error: {str(e)}")
            return None
            
    async def _calculate_base_size(
        self,
        account_info: Dict,
        market_state: Dict
    ) -> float:
        """Calculate base position size"""
        try:
            equity = account_info['equity']
            
            # Calculate risk amount
            risk_amount = equity * self.position_params['base_risk']
            
            # Adjust for drawdown
            if 'drawdown' in account_info:
                drawdown = account_info['drawdown']
                risk_amount *= max(0.5, 1 - drawdown)
                
            # Adjust for market conditions
            if market_state.get('high_risk', False):
                risk_amount *= 0.5
                
            return risk_amount
            
        except Exception as e:
            self.logger.error(f"Base size calculation error: {str(e)}")
            return 0
            
    async def _apply_kelly_criterion(
        self,
        strategy: str,
        account_info: Dict
    ) -> float:
        """Apply Kelly Criterion for position sizing"""
        try:
            # Get strategy performance
            win_rate = await self._calculate_win_rate(strategy)
            avg_win = await self._calculate_average_win(strategy)
            avg_loss = await self._calculate_average_loss(strategy)
            
            # Calculate Kelly fraction
            if avg_loss != 0:
                kelly = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) / avg_win
            else:
                kelly = 0
                
            # Apply half-Kelly for conservation
            kelly *= self.position_params['kelly_fraction']
            
            # Ensure kelly is within bounds
            kelly = max(0, min(1, kelly))
            
            return kelly
            
        except Exception as e:
            self.logger.error(f"Kelly calculation error: {str(e)}")
            return 0.5
            
    async def _calculate_volatility_adjustment(
        self,
        symbol: str,
        market_state: Dict
    ) -> float:
        """Calculate volatility-based position adjustment"""
        try:
            # Get current volatility
            current_vol = market_state.get('volatility', 1.0)
            
            # Calculate adjustment factor
            if current_vol > self.position_params['volatility_multiplier']:
                return 1 / current_vol
            elif current_vol < 1 / self.position_params['volatility_multiplier']:
                return min(2.0, 1 / current_vol)
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Volatility adjustment error: {str(e)}")
            return 1.0
            
    async def _calculate_base_leverage(
        self,
        sharpe: float,
        win_rate: float,
        market_state: Dict
    ) -> float:
        """Calculate base leverage multiplier"""
        try:
            leverage = self.leverage_params['base_leverage']
            
            # Adjust for Sharpe ratio
            if sharpe > self.leverage_params['sharpe_threshold']:
                leverage *= min(2.0, sharpe / self.leverage_params['sharpe_threshold'])
                
            # Adjust for win rate
            if win_rate > self.leverage_params['win_rate_threshold']:
                leverage *= min(
                    1.5,
                    win_rate / self.leverage_params['win_rate_threshold']
                )
                
            # Limit leverage
            leverage = min(
                leverage,
                self.leverage_params['max_leverage']
            )
            
            return leverage
            
        except Exception as e:
            self.logger.error(f"Base leverage calculation error: {str(e)}")
            return self.leverage_params['base_leverage']
            
    async def _adjust_leverage_for_risk(
        self,
        leverage: float,
        account_info: Dict,
        market_state: Dict
    ) -> float:
        """Adjust leverage based on risk conditions"""
        try:
            # Check drawdown
            if account_info.get('drawdown', 0) > self.leverage_params['drawdown_limit']:
                leverage *= 0.5
                
            # Check volatility
            if market_state.get('volatility', 1.0) > self.leverage_params['volatility_limit']:
                leverage *= 0.7
                
            # Check market conditions
            if market_state.get('high_risk', False):
                leverage = min(leverage, self.leverage_params['base_leverage'])
                
            return max(
                self.leverage_params['min_leverage'],
                min(leverage, self.leverage_params['max_leverage'])
            )
            
        except Exception as e:
            self.logger.error(f"Leverage adjustment error: {str(e)}")
            return self.leverage_params['min_leverage']
            
    async def _calculate_sharpe_ratio(
        self,
        account_info: Dict,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio"""
        try:
            returns = account_info.get('returns', [])
            if not returns:
                return 0
                
            returns = np.array(returns)
            excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
            
            if len(excess_returns) > 1:
                sharpe = np.sqrt(252) * (
                    np.mean(excess_returns) / np.std(excess_returns)
                )
                return float(sharpe)
            return 0
            
        except Exception as e:
            self.logger.error(f"Sharpe ratio calculation error: {str(e)}")
            return 0
            
    async def _calculate_win_rate(self, strategy: str) -> float:
        """Calculate strategy win rate"""
        try:
            trades = self.performance_metrics['trades']
            if not trades:
                return 0.5
                
            strategy_trades = [
                trade for trade in trades
                if trade['strategy'] == strategy
            ]
            
            if not strategy_trades:
                return 0.5
                
            wins = sum(1 for trade in strategy_trades if trade['profit'] > 0)
            return wins / len(strategy_trades)
            
        except Exception as e:
            self.logger.error(f"Win rate calculation error: {str(e)}")
            return 0.5
            
    async def _calculate_risk_score(
        self,
        account_info: Dict,
        market_state: Dict
    ) -> float:
        """Calculate overall risk score"""
        try:
            risk_factors = [
                account_info.get('drawdown', 0) / self.leverage_params['drawdown_limit'],
                market_state.get('volatility', 1.0) / self.leverage_params['volatility_limit'],
                1 - await self._calculate_win_rate(market_state.get('strategy', '')),
                1 - (await self._calculate_sharpe_ratio(account_info) / self.leverage_params['sharpe_threshold'])
            ]
            
            return np.mean(risk_factors)
            
        except Exception as e:
            self.logger.error(f"Risk score calculation error: {str(e)}")
            return 0.5
            
    async def _generate_leverage_recommendation(
        self,
        leverage: float,
        risk_score: float
    ) -> str:
        """Generate leverage recommendation"""
        try:
            if risk_score > 0.8:
                return "High Risk - Reduce leverage to minimum"
            elif risk_score > 0.6:
                return f"Moderate Risk - Limit leverage to {leverage:.1f}x"
            elif risk_score > 0.4:
                return f"Normal Risk - Current leverage {leverage:.1f}x acceptable"
            else:
                return f"Low Risk - Can increase leverage up to {leverage:.1f}x"
                
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {str(e)}")
            return "Error generating recommendation"
