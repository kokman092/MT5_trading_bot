import logging
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
from .signal import Signal
from dataclasses import dataclass
import asyncio
import math
import json
from pathlib import Path
import os

@dataclass
class RiskParameters:
    max_daily_loss: float
    max_position_size: float
    max_correlation: float
    max_positions: int
    risk_per_trade: float
    max_drawdown: float

class RiskManager:
    def __init__(self, config: Dict):
        """Initialize risk manager with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.broker = None
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'profit': 0.0,
            'max_drawdown': 0.0,
            'win_streak': 0,
            'loss_streak': 0
        }
        self.risk_params = RiskParameters(
            config['risk_management']['max_daily_loss'],
            config['risk_management']['max_position_size'],
            config['risk_management']['position_limits']['max_correlation'],
            config['risk_management']['position_limits']['max_positions'],
            config['risk_management']['risk_per_trade'],
            config['risk_management']['loss_limits']['max_drawdown']
        )
        self._validate_risk_config()
        
        # Get risk settings from config
        self.risk_per_trade = self.config['risk_management'].get('risk_per_trade', 0.01)
        self.max_risk_per_symbol = self.config['risk_management'].get('max_risk_per_symbol', 0.03)
        self.max_daily_loss = self.config['risk_management'].get('emergency_stop', {}).get('max_daily_loss_percent', 0.03)
        self.max_drawdown = self.config['risk_management'].get('emergency_stop', {}).get('max_drawdown_percent', 0.05)
        
        # Performance tracking
        self.daily_loss = 0.0
        self.max_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown = 0.0
        self.drawdown_start_date = None
        self.consecutive_losses = 0
        self.max_consecutive_losses = self.config['risk_management'].get('emergency_stop', {}).get('max_consecutive_losses', 5)
        
        # Risk metrics history
        self.risk_metrics_history = []
        self.max_history_size = 1000
        
        # Dynamic position sizing
        self.position_sizing_method = self.config['risk_management'].get('adaptive_position_sizing', {}).get('base_sizing', 'fixed')
        self.position_size_scale_factor = self.config['risk_management'].get('adaptive_position_sizing', {}).get('scale_factor', 1.0)
        self.kelly_fraction = 0.3  # Conservative Kelly fraction
        
        # Dynamic risk adjustment
        self.in_recovery_mode = False
        self.recovery_risk_factor = self.config['risk_management'].get('drawdown_protection', {}).get('recovery_risk_factor', 0.5)
        self.recovery_threshold = self.config['risk_management'].get('drawdown_protection', {}).get('recovery_mode_threshold', 0.05)
        
        # Correlation data for portfolio risk management
        self.correlation_matrix = {}
        self.last_correlation_update = datetime.now() - timedelta(days=1)
        self.correlation_update_interval = timedelta(hours=4)
        
        # Trade history for performance analysis
        self.trade_history = []
        
        # Initialize history file
        self.history_file = 'data/risk_metrics_history.json'
        os.makedirs('data', exist_ok=True)
        self._load_risk_history()
        
    def _validate_risk_config(self) -> None:
        """Validate risk management parameters"""
        try:
            if self.risk_params.max_daily_loss <= 0:
                self.logger.warning("Max daily loss parameter must be positive.")
            if self.risk_params.risk_per_trade <= 0 or self.risk_params.risk_per_trade > 1:
                self.logger.warning("Risk per trade must be between 0 and 1.")
        except Exception as e:
            self.logger.error(f"Error validating risk config: {str(e)}")
        
    async def initialize(self) -> bool:
        """Initialize risk manager state"""
        try:
            # Load saved risk state if available
            state_file = 'data/risk_state.json'
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    
                self.daily_loss = state.get('daily_loss', 0.0)
                self.max_equity = state.get('max_equity', 0.0)
                self.current_equity = state.get('current_equity', 0.0)
                self.current_drawdown = state.get('current_drawdown', 0.0)
                self.consecutive_losses = state.get('consecutive_losses', 0)
                self.in_recovery_mode = state.get('in_recovery_mode', False)
                
                # Check if we need to reset daily metrics
                last_update = datetime.fromisoformat(state.get('last_update', datetime.now().isoformat()))
                if last_update.date() < datetime.now().date():
                    self.logger.info("New trading day - resetting daily metrics")
                    self.daily_loss = 0.0
                    
            # Initialize risk metrics based on account data
            await self._initialize_risk_metrics()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing risk manager: {str(e)}")
            return False
            
    async def _initialize_risk_metrics(self):
        """Initialize risk metrics based on account data"""
        try:
            # This would fetch current account equity and set initial values
            # For now, we'll use placeholder values
            self.current_equity = 10000.0  # Example value
            self.max_equity = self.current_equity
            
        except Exception as e:
            self.logger.error(f"Error initializing risk metrics: {str(e)}")
            
    def save_state(self):
        """Save current risk manager state"""
        try:
            state = {
                'daily_loss': self.daily_loss,
                'max_equity': self.max_equity,
                'current_equity': self.current_equity,
                'current_drawdown': self.current_drawdown,
                'consecutive_losses': self.consecutive_losses,
                'in_recovery_mode': self.in_recovery_mode,
                'last_update': datetime.now().isoformat()
            }
            
            with open('data/risk_state.json', 'w') as f:
                json.dump(state, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving risk state: {str(e)}")
    
    def _load_risk_history(self):
        """Load risk metrics history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.risk_metrics_history = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading risk history: {str(e)}")
            self.risk_metrics_history = []
            
    def _save_risk_history(self):
        """Save risk metrics history to file"""
        try:
            # Keep history size bounded
            if len(self.risk_metrics_history) > self.max_history_size:
                self.risk_metrics_history = self.risk_metrics_history[-self.max_history_size:]
                
            with open(self.history_file, 'w') as f:
                json.dump(self.risk_metrics_history, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving risk history: {str(e)}")
    
    async def calculate_position_size(self, 
                                     symbol: str, 
                                     entry_price: float, 
                                     stop_loss: float, 
                                     account_balance: float,
                                     market_data: Optional[Dict] = None) -> Tuple[float, Dict]:
        """Calculate position size based on risk parameters and current market conditions"""
        try:
            # Calculate basic position size
            if entry_price <= 0 or stop_loss <= 0:
                return 0.0, {"error": "Invalid price levels"}
                
            # Calculate risk amount
            risk_amount = self._get_current_risk_percentage() * account_balance
            
            # Calculate stop loss distance in pips/points
            is_long = entry_price > stop_loss
            pip_distance = abs(entry_price - stop_loss)
            
            # Validate minimum stop loss distance
            min_stop_distance = self.config['risk_management'].get('min_stop_distance', 10)
            if pip_distance < min_stop_distance:
                self.logger.warning(f"Stop loss too close to entry: {pip_distance} < {min_stop_distance}")
                pip_distance = min_stop_distance
                
            # Calculate position size in lots
            if pip_distance == 0:
                return 0.0, {"error": "Zero stop loss distance"}
                
            # Get tick value and lot size from symbol properties
            tick_value = self._get_tick_value(symbol)
            if tick_value <= 0:
                return 0.0, {"error": "Invalid tick value"}
                
            # Basic position size calculation
            position_size = risk_amount / (pip_distance * tick_value)
            
            # Get market regime if not provided
            if market_data and 'regime' not in market_data and 'ml_predictions' in market_data:
                market_data['regime'] = self._detect_market_regime(market_data['ml_predictions'], symbol)
            
            # Apply optimized position sizing based on market regime
            position_size = await self._apply_optimized_sizing(position_size, symbol, is_long, market_data)
                
            # Apply correlation-based portfolio adjustments
            portfolio_factor = await self._calculate_portfolio_risk_factor(symbol)
            position_size *= portfolio_factor
            
            # Apply volatility adjustments with enhanced sensitivity
            volatility_factor = self._calculate_enhanced_volatility_adjustment(symbol, market_data)
            position_size *= volatility_factor
            
            # Apply market regime adjustments with more granular categories
            regime_factor = self._calculate_regime_adjustment(symbol, market_data)
            position_size *= regime_factor
            
            # Apply account exposure limit
            exposure_factor = await self._calculate_exposure_factor(symbol, position_size, account_balance)
            position_size *= exposure_factor
            
            # Apply risk-adjusted multiplier based on trade confidence
            confidence_factor = self._calculate_confidence_adjustment(market_data)
            position_size *= confidence_factor
            
            # Get limits for position size
            min_size = self.config['risk_management'].get('position_sizing', {}).get('min_size', 0.01)
            max_size = self.config['risk_management'].get('position_sizing', {}).get('max_size', 1.0)
            
            # Ensure position size is within limits
            position_size = max(min_size, min(position_size, max_size))
            
            # Round position size to valid lot step
            lot_step = self._get_lot_step(symbol)
            position_size = math.floor(position_size / lot_step) * lot_step
            
            # Return position size and risk metrics
            return position_size, {
                "risk_amount": risk_amount,
                "risk_percent": self._get_current_risk_percentage(),
                "stop_distance": pip_distance,
                "portfolio_factor": portfolio_factor,
                "volatility_factor": volatility_factor,
                "regime_factor": regime_factor,
                "exposure_factor": exposure_factor,
                "confidence_factor": confidence_factor,
                "in_recovery_mode": self.in_recovery_mode
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0, {"error": str(e)}
            
    def _get_current_risk_percentage(self) -> float:
        """Get current risk percentage with adaptive adjustments"""
        # Start with base risk per trade
        current_risk = self.risk_per_trade
        
        # If in recovery mode, reduce risk
        if self.in_recovery_mode:
            current_risk *= self.recovery_risk_factor
            
        # If reached consecutive losses, reduce risk
        if self.consecutive_losses >= 3:
            reduction_factor = max(0.25, 1 - (self.consecutive_losses - 2) * 0.15)
            current_risk *= reduction_factor
            
        # If daily loss approaches limit, reduce risk
        daily_loss_factor = 1.0
        if self.daily_loss > 0:
            daily_loss_ratio = self.daily_loss / self.max_daily_loss
            if daily_loss_ratio > 0.5:
                daily_loss_factor = max(0.25, 1 - (daily_loss_ratio - 0.5) * 2)
            current_risk *= daily_loss_factor
            
        # Ensure minimum risk percentage
        min_risk = self.risk_per_trade * 0.1  # 10% of normal risk
        return max(min_risk, current_risk)
    
    def _apply_kelly_sizing(self, base_position_size: float, symbol: str, is_long: bool, market_data: Optional[Dict]) -> float:
        """Apply Kelly criterion sizing based on win rate and reward:risk"""
        try:
            # Get win rate and average win/loss ratio
            win_rate = self._get_historical_win_rate(symbol, is_long)
            avg_win_loss_ratio = self._get_historical_win_loss_ratio(symbol, is_long)
            
            # Calculate Kelly fraction
            kelly_fraction = win_rate - ((1 - win_rate) / avg_win_loss_ratio)
            
            # Limit Kelly fraction for safety
            kelly_fraction = max(0, min(kelly_fraction, 0.5))
            
            # Apply conservative Kelly factor
            kelly_fraction *= self.kelly_fraction
            
            # Adjust position size based on Kelly criterion
            position_size = base_position_size * kelly_fraction / self.risk_per_trade
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error applying Kelly sizing: {str(e)}")
            return base_position_size
            
    def _apply_adaptive_sizing(self, base_position_size: float, symbol: str, is_long: bool, market_data: Optional[Dict]) -> float:
        """Apply adaptive sizing based on recent performance and market conditions"""
        try:
            # Start with base position size
            position_size = base_position_size
            
            # Adjust based on win streak
            if self.consecutive_losses == 0:
                # Get number of consecutive wins
                consecutive_wins = self._get_consecutive_wins()
                if consecutive_wins > 2:
                    # Increase position size after consecutive wins (up to 50% increase after 5 wins)
                    win_scale_factor = min(1.5, 1 + (consecutive_wins - 2) * 0.1)
                    position_size *= win_scale_factor
                    
            # Adjust based on signal strength if market data available
            if market_data and 'signal_strength' in market_data:
                signal_strength = market_data['signal_strength']
                # Scale position size by signal strength (from 0.7 to 1.3)
                signal_factor = 0.7 + (signal_strength * 0.6)
                position_size *= signal_factor
                
            # Apply custom scale factor from configuration
            position_size *= self.position_size_scale_factor
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error applying adaptive sizing: {str(e)}")
            return base_position_size
            
    def _apply_martingale_sizing(self, base_position_size: float) -> float:
        """Apply modified martingale sizing based on consecutive losses"""
        try:
            # Only increase size after 2 consecutive losses
            if self.consecutive_losses <= 1:
                return base_position_size
                
            # Calculate martingale factor (conservative approach)
            # Instead of doubling, we increase by 30% per loss
            martingale_factor = 1 + (self.consecutive_losses - 1) * 0.3
            
            # Cap the maximum increase to 3x
            martingale_factor = min(3.0, martingale_factor)
            
            return base_position_size * martingale_factor
            
        except Exception as e:
            self.logger.error(f"Error applying martingale sizing: {str(e)}")
            return base_position_size
            
    async def _calculate_portfolio_risk_factor(self, symbol: str) -> float:
        """Calculate portfolio risk factor based on correlations and exposure"""
        try:
            # Update correlation matrix if needed
            if datetime.now() - self.last_correlation_update > self.correlation_update_interval:
                await self._update_correlation_matrix()
                
            # Get current portfolio exposure
            current_exposure = await self._get_current_exposure()
            
            # Get symbol correlations
            symbol_correlations = self.correlation_matrix.get(symbol, {})
            
            # If we have no correlation data, use default factor
            if not symbol_correlations:
                # Apply basic portfolio risk factor based on current exposure
                max_exposure = self.config['risk_management'].get('portfolio', {}).get('max_exposure', 0.8)
                return max(0.1, 1 - (sum(current_exposure.values()) / max_exposure))
                
            # Calculate average correlation with current portfolio
            correlated_exposure = 0
            for other_symbol, correlation in symbol_correlations.items():
                if other_symbol in current_exposure:
                    correlated_exposure += abs(correlation) * current_exposure[other_symbol]
                    
            # Calculate portfolio diversification factor
            # Higher correlation = lower position size
            diversification_factor = 1 - (correlated_exposure * 0.5)  # Range 0.5-1.0
            
            return max(0.25, diversification_factor)
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk factor: {str(e)}")
            return 1.0
            
    async def _get_current_exposure(self) -> Dict[str, float]:
        """Get current exposure by symbol"""
        if not self.broker:
            return {}
        try:
            positions = self.broker.get_positions()
            exposure = {}
            for pos in positions:
                symbol = pos.get('symbol')
                if symbol:
                    exposure[symbol] = exposure.get(symbol, 0.0) + pos.get('volume', 0.0)
            return exposure
        except Exception as e:
            self.logger.error(f"Error getting active exposure from broker: {str(e)}")
            return {}
            
    def _calculate_volatility_adjustment(self, symbol: str, market_data: Optional[Dict]) -> float:
        """Calculate position size adjustment based on market volatility"""
        try:
            if not market_data or 'volatility' not in market_data:
                return 1.0
                
            volatility = market_data['volatility']
            
            # Get volatility thresholds from config
            normal_volatility = self.config['market_analysis'].get('validation', {}).get('normal_volatility', 0.01)
            
            # If volatility is higher than normal, reduce position size
            if volatility > normal_volatility:
                # Volatility is X times normal, reduce size by square root of X
                volatility_ratio = volatility / normal_volatility
                return 1.0 / math.sqrt(volatility_ratio)
                
            # If volatility is lower than normal, consider increasing position size
            if volatility < normal_volatility * 0.5:
                # But limit the increase to 30%
                return min(1.3, normal_volatility / (volatility * 0.5))
                
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility adjustment: {str(e)}")
            return 1.0
            
    def _calculate_regime_adjustment(self, symbol: str, market_data: Optional[Dict]) -> float:
        """Calculate position size adjustment based on market regime"""
        try:
            if not market_data or 'regime' not in market_data:
                return 1.0
                
            regime = market_data['regime']
            regime_type = regime.get('regime', 'unknown')
            regime_confidence = regime.get('confidence', 0.5)
            
            # Define adjustment factors for different regimes
            regime_factors = {
                'bullish_trend': 1.2,
                'bearish_trend': 1.2,
                'ranging': 0.8,
                'volatile': 0.7,
                'undefined': 0.9,
                'unknown': 1.0
            }
            
            # Get adjustment factor for current regime
            adjustment = regime_factors.get(regime_type, 1.0)
            
            # Scale by confidence
            adjustment = 1.0 + ((adjustment - 1.0) * regime_confidence)
            
            return adjustment
            
        except Exception as e:
            self.logger.error(f"Error calculating regime adjustment: {str(e)}")
            return 1.0
    
    def _get_historical_win_rate(self, symbol: str, is_long: bool) -> float:
        """Get historical win rate for the symbol and direction"""
        try:
            # Filter trades for the symbol and direction
            direction = 'long' if is_long else 'short'
            relevant_trades = [t for t in self.trade_history 
                              if t['symbol'] == symbol and t['direction'] == direction]
            
            if not relevant_trades:
                return 0.5  # Default if no historical data
                
            # Calculate win rate
            wins = sum(1 for t in relevant_trades if t['profit'] > 0)
            return wins / len(relevant_trades)
            
        except Exception as e:
            self.logger.error(f"Error calculating historical win rate: {str(e)}")
            return 0.5
            
    def _get_historical_win_loss_ratio(self, symbol: str, is_long: bool, regime: str = 'unknown') -> float:
        """Get historical win/loss ratio for the symbol, direction and regime"""
        try:
            # Filter trades for the symbol and direction
            direction = 'long' if is_long else 'short'
            relevant_trades = [t for t in self.trade_history 
                              if t['symbol'] == symbol and t['direction'] == direction]
            
            # Further filter by regime if available
            if regime != 'unknown':
                relevant_trades = [t for t in relevant_trades 
                                  if t.get('market_regime', 'unknown') == regime]
            
            if not relevant_trades:
                # If no trades for this specific regime, try without regime filter
                if regime != 'unknown':
                    return self._get_historical_win_loss_ratio(symbol, is_long, 'unknown')
                # If still no data, use default
                return 2.0  # Default if no historical data
                
            # Calculate average win and loss
            wins = [t['profit'] for t in relevant_trades if t['profit'] > 0]
            losses = [abs(t['profit']) for t in relevant_trades if t['profit'] < 0]
            
            if not wins or not losses:
                return 2.0  # Default if no wins or no losses
                
            avg_win = sum(wins) / len(wins)
            avg_loss = sum(losses) / len(losses)
            
            if avg_loss == 0:
                return 5.0  # Cap the ratio
                
            ratio = avg_win / avg_loss
            
            # Cap the ratio for safety
            return min(5.0, ratio)
            
        except Exception as e:
            self.logger.error(f"Error calculating win/loss ratio: {str(e)}")
            return 2.0
            
    def _get_consecutive_wins(self) -> int:
        """Get number of consecutive winning trades"""
        count = 0
        for trade in reversed(self.trade_history):
            if trade['profit'] > 0:
                count += 1
            else:
                break
        return count
    
    async def _update_correlation_matrix(self):
        """Update correlation matrix for portfolio symbols"""
        try:
            # This would calculate correlations between trading symbols
            # For now, use a placeholder
            self.correlation_matrix = {}
            self.last_correlation_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating correlation matrix: {str(e)}")
    
    def _get_tick_value(self, symbol: str) -> float:
        """Get tick value for the symbol"""
        # This would get actual tick value from broker
        # For now, return a placeholder value for common forex pairs
        return 10.0  # Standard tick value for major forex pairs with 0.01 lot
        
    def _get_lot_step(self, symbol: str) -> float:
        """Get minimum lot step for the symbol"""
        # This would get actual lot step from broker
        # For now, return a placeholder value
        return 0.01  # Standard lot step for forex
            
    async def check_trade_allowed(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """Check if a trade is allowed based on risk parameters"""
        try:
            # Check if we're in emergency stop mode
            if self.daily_loss >= self.max_daily_loss:
                return False, "Daily loss limit reached"
                
            # Check if we're in max drawdown
            if self.current_drawdown >= self.max_drawdown:
                return False, "Maximum drawdown reached"
                
            # Check for too many consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                return False, "Too many consecutive losses"
                
            # Check symbol-specific risk limits
            symbols_exposure = await self._get_current_exposure()
            if symbol in symbols_exposure:
                symbol_exposure = symbols_exposure[symbol]
                max_symbol_exposure = self.config['risk_management'].get('portfolio', {}).get('max_exposure_per_instrument', 0.3)
                if symbol_exposure >= max_symbol_exposure:
                    return False, f"Maximum exposure reached for {symbol}"
                    
            # Check current market conditions
            market_conditions_ok, reason = await self._check_market_conditions(symbol, direction)
            if not market_conditions_ok:
                return False, reason
                
            return True, "Trade allowed"
            
        except Exception as e:
            self.logger.error(f"Error checking if trade allowed: {str(e)}")
            return False, f"Error checking trade permissions: {str(e)}"
            
    async def validate_trade(self, symbol: str, signal: Signal) -> bool:
        """Validate if a trade is allowed based on risk parameters"""
        try:
            allowed, reason = await self.check_trade_allowed(symbol, signal.direction)
            if not allowed:
                self.logger.warning(f"Trade validation failed for {symbol}: {reason}")
            return allowed
        except Exception as e:
            self.logger.error(f"Error validating trade: {str(e)}")
            return False
            
    async def _check_market_conditions(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """Check if current market conditions are suitable for trading"""
        # Placeholder - would check volatility, spread, etc.
        return True, "Market conditions OK"
            
    def update_risk_metrics(self, account_balance: float, equity: float):
        """Update risk metrics with current account information"""
        try:
            # Update equity tracking
            self.current_equity = equity
            
            # Update max equity if we have a new high
            if equity > self.max_equity:
                self.max_equity = equity
                
            # Calculate current drawdown
            if self.max_equity > 0:
                self.current_drawdown = (self.max_equity - self.current_equity) / self.max_equity
                
                # Check if we should enter recovery mode
                if not self.in_recovery_mode and self.current_drawdown >= self.recovery_threshold:
                    self.in_recovery_mode = True
                    self.logger.warning(f"Entering recovery mode - drawdown: {self.current_drawdown:.2%}")
                    
                # Check if we should exit recovery mode
                if self.in_recovery_mode and self.current_drawdown < self.recovery_threshold * 0.5:
                    self.in_recovery_mode = False
                    self.logger.info(f"Exiting recovery mode - drawdown reduced to {self.current_drawdown:.2%}")
                    
            # Record current metrics
            self.risk_metrics_history.append({
                'timestamp': datetime.now().isoformat(),
                'equity': self.current_equity,
                'max_equity': self.max_equity,
                'drawdown': self.current_drawdown,
                'in_recovery_mode': self.in_recovery_mode,
                'consecutive_losses': self.consecutive_losses,
                'daily_loss': self.daily_loss
            })
            
            # Save risk state
            self.save_state()
            
            # Periodically save full history
            if len(self.risk_metrics_history) % 10 == 0:
                self._save_risk_history()
                
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {str(e)}")
            
    def log_trade_result(self, trade_result: Dict):
        """Log a trade result and update risk metrics"""
        try:
            # Add timestamp if not present
            if 'timestamp' not in trade_result:
                trade_result['timestamp'] = datetime.now()
                
            # Ensure market regime is captured if available
            if 'market_data' in trade_result and 'regime' in trade_result['market_data']:
                trade_result['market_regime'] = trade_result['market_data']['regime'].get('regime', 'unknown')
                trade_result['regime_confidence'] = trade_result['market_data']['regime'].get('confidence', 0.5)
            
            # Record strategy type if available
            if 'strategy_type' not in trade_result and 'signal' in trade_result:
                signal = trade_result['signal']
                if hasattr(signal, 'strategy_type'):
                    trade_result['strategy_type'] = signal.strategy_type
                    
            # Calculate additional performance metrics
            trade_result['risk_reward_actual'] = self._calculate_actual_rr(trade_result)
            
            # Update trade history
            self.trade_history.append(trade_result)
            
            # If testing trade history limit
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
                
            # Get trade profit
            profit = trade_result.get('profit', 0.0)
            
            # Update daily loss tracking
            if profit < 0:
                self.daily_loss += abs(profit)
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
                
            # Update equity tracking
            self.current_equity += profit
            if self.current_equity > self.max_equity:
                self.max_equity = self.current_equity
                
            # Update current drawdown
            if self.max_equity > 0:
                self.current_drawdown = (self.max_equity - self.current_equity) / self.max_equity
                
            # Update strategy performance metrics for adaptive position sizing
            self._update_strategy_performance_metrics(trade_result)
                
            # Log warning if approaching risk limits
            if self.daily_loss > self.max_daily_loss * 0.7:
                self.logger.warning(f"Approaching daily loss limit: {self.daily_loss:.2f}/{self.max_daily_loss:.2f}")
                
            if self.current_drawdown > self.max_drawdown * 0.7:
                self.logger.warning(f"Approaching max drawdown: {self.current_drawdown:.2%}/{self.max_drawdown:.2%}")
                
            # Save state
            self.save_state()
            
        except Exception as e:
            self.logger.error(f"Error logging trade result: {str(e)}")
            
    def _calculate_actual_rr(self, trade_result: Dict) -> float:
        """Calculate actual risk-reward ratio achieved in the trade"""
        try:
            if 'entry_price' not in trade_result or 'exit_price' not in trade_result or 'stop_loss' not in trade_result:
                return 0.0
                
            entry_price = trade_result['entry_price']
            exit_price = trade_result['exit_price']
            stop_loss = trade_result['stop_loss']
            
            # Calculate risk and reward in price terms
            risk = abs(entry_price - stop_loss)
            actual_reward = abs(entry_price - exit_price)
            
            if risk == 0:
                return 0.0
                
            return actual_reward / risk
            
        except Exception as e:
            self.logger.error(f"Error calculating actual RR: {str(e)}")
            return 0.0
            
    def _update_strategy_performance_metrics(self, trade_result: Dict):
        """Update strategy-specific performance metrics for adaptive position sizing"""
        try:
            if 'strategy_type' not in trade_result:
                return
                
            strategy_type = trade_result['strategy_type']
            symbol = trade_result.get('symbol', 'unknown')
            direction = trade_result.get('direction', 'unknown')
            market_regime = trade_result.get('market_regime', 'unknown')
            profit = trade_result.get('profit', 0.0)
            
            # Create strategy performance tracking dict if not exists
            if not hasattr(self, 'strategy_performance'):
                self.strategy_performance = {}
                
            # Create nested dicts as needed
            if strategy_type not in self.strategy_performance:
                self.strategy_performance[strategy_type] = {}
                
            if symbol not in self.strategy_performance[strategy_type]:
                self.strategy_performance[strategy_type][symbol] = {}
                
            if direction not in self.strategy_performance[strategy_type][symbol]:
                self.strategy_performance[strategy_type][symbol][direction] = {}
                
            if market_regime not in self.strategy_performance[strategy_type][symbol][direction]:
                self.strategy_performance[strategy_type][symbol][direction][market_regime] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit_sum': 0.0,
                    'loss_sum': 0.0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'expected_value': 0.0
                }
                
            # Update metrics
            metrics = self.strategy_performance[strategy_type][symbol][direction][market_regime]
            metrics['trades'] += 1
            
            if profit > 0:
                metrics['wins'] += 1
                metrics['profit_sum'] += profit
            else:
                metrics['losses'] += 1
                metrics['loss_sum'] += abs(profit)
                
            # Recalculate derived metrics
            if metrics['trades'] > 0:
                metrics['win_rate'] = metrics['wins'] / metrics['trades']
                
            if metrics['wins'] > 0:
                metrics['avg_win'] = metrics['profit_sum'] / metrics['wins']
                
            if metrics['losses'] > 0:
                metrics['avg_loss'] = metrics['loss_sum'] / metrics['losses']
                
            if metrics['loss_sum'] > 0:
                metrics['profit_factor'] = metrics['profit_sum'] / metrics['loss_sum']
            else:
                metrics['profit_factor'] = metrics['profit_sum'] if metrics['profit_sum'] > 0 else 0.0
                
            # Calculate expected value
            metrics['expected_value'] = (metrics['win_rate'] * metrics['avg_win']) - ((1 - metrics['win_rate']) * metrics['avg_loss'])
            
            # Save updated metrics
            self.strategy_performance[strategy_type][symbol][direction][market_regime] = metrics
            
            # Save to disk periodically
            if hasattr(self, 'trade_counter'):
                self.trade_counter += 1
            else:
                self.trade_counter = 1
                
            if self.trade_counter % 10 == 0:
                self._save_strategy_performance()
                
        except Exception as e:
            self.logger.error(f"Error updating strategy performance metrics: {str(e)}")
            
    def _save_strategy_performance(self):
        """Save strategy performance metrics to disk"""
        try:
            if hasattr(self, 'strategy_performance'):
                with open('data/strategy_performance.json', 'w') as f:
                    json.dump(self.strategy_performance, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving strategy performance: {str(e)}")
            
    def get_strategy_performance(self, strategy_type: Optional[str] = None, 
                               symbol: Optional[str] = None,
                               direction: Optional[str] = None,
                               market_regime: Optional[str] = None) -> Dict:
        """Get strategy performance metrics for filtering position sizing"""
        try:
            if not hasattr(self, 'strategy_performance'):
                return {}
                
            if strategy_type is None:
                return self.strategy_performance
                
            if strategy_type not in self.strategy_performance:
                return {}
                
            if symbol is None:
                return self.strategy_performance[strategy_type]
                
            if symbol not in self.strategy_performance[strategy_type]:
                return {}
                
            if direction is None:
                return self.strategy_performance[strategy_type][symbol]
                
            if direction not in self.strategy_performance[strategy_type][symbol]:
                return {}
                
            if market_regime is None:
                return self.strategy_performance[strategy_type][symbol][direction]
                
            if market_regime not in self.strategy_performance[strategy_type][symbol][direction]:
                return {}
                
            return self.strategy_performance[strategy_type][symbol][direction][market_regime]
            
        except Exception as e:
            self.logger.error(f"Error getting strategy performance: {str(e)}")
            return {}

    async def _apply_optimized_sizing(self, base_position_size: float, symbol: str, is_long: bool, market_data: Optional[Dict]) -> float:
        """Apply optimized position sizing based on market regime and historical performance"""
        try:
            # Start with base position size
            position_size = base_position_size
            
            # If no market data, use base size
            if not market_data:
                return position_size
                
            # Get market regime
            regime = market_data.get('regime', {}).get('regime', 'unknown')
            
            # Determine optimal sizing method based on market regime
            if regime in ['bullish_trend', 'bearish_trend']:
                # For trending markets, use adaptive sizing based on trend strength
                trend_strength = market_data.get('trend_strength', 0.5)
                
                # For trend-following strategies, increase size with trend strength
                if 'strategy_type' in market_data and market_data['strategy_type'] == 'trend_following':
                    trend_factor = 1.0 + (trend_strength - 0.5) * 0.6  # 0.7-1.3 range
                    position_size *= trend_factor
                
                # Apply win rate adjustment
                win_rate = self._get_historical_win_rate(symbol, is_long, regime)
                if win_rate > 0.6:  # Only increase if win rate is good
                    position_size *= (1.0 + (win_rate - 0.6) * 0.5)  # Up to 20% increase for 100% win rate
                
            elif regime in ['ranging', 'choppy']:
                # For ranging markets, use more conservative sizing
                position_size *= 0.8
                
                # For mean-reversion strategies, can be more aggressive in ranges
                if 'strategy_type' in market_data and market_data['strategy_type'] == 'mean_reversion':
                    # Check oversold/overbought conditions
                    extreme_condition = market_data.get('overbought_oversold', 0)
                    if abs(extreme_condition) > 0.7:  # Strong mean-reversion signal
                        position_size *= 1.1
                
            elif regime == 'volatile':
                # For volatile markets, use more conservative sizing with volatility-based adjustment
                volatility = market_data.get('volatility', 1.0)
                position_size *= max(0.5, 1.0 / math.sqrt(volatility))
                
                # But for volatility breakout strategies, can be more aggressive
                if 'strategy_type' in market_data and market_data['strategy_type'] == 'volatility_breakout':
                    position_size *= 1.2
                    
            elif regime == 'high_opportunity':
                # For high opportunity regimes (e.g., post-news, clear setups)
                signal_strength = market_data.get('signal_strength', 0.5)
                position_size *= (1.0 + signal_strength * 0.5)  # Up to 50% increase
                
            # Apply Kelly criterion as a final modulator if we have win rate and payoff data
            kelly_factor = self._calculate_kelly_fraction(symbol, is_long, regime)
            
            # Use a conservative fraction of Kelly (typically 1/4 to 1/2)
            kelly_fraction = 0.3
            kelly_adjustment = 1.0 + ((kelly_factor - 0.5) * kelly_fraction)
            
            # Limit kelly adjustment to +/- 30%
            kelly_adjustment = max(0.7, min(1.3, kelly_adjustment))
            position_size *= kelly_adjustment
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error applying optimized sizing: {str(e)}")
            return base_position_size
            
    def _calculate_enhanced_volatility_adjustment(self, symbol: str, market_data: Optional[Dict]) -> float:
        """Calculate enhanced position size adjustment based on market volatility"""
        try:
            if not market_data:
                return 1.0
                
            # Get volatility metrics
            current_volatility = market_data.get('volatility', None)
            if current_volatility is None and 'indicators' in market_data:
                # Try to extract from indicators
                indicators = market_data.get('indicators', {})
                if 'atr' in indicators and 'atr_percent' in indicators:
                    current_volatility = indicators.get('atr_percent', 0.01)
                elif 'bbands_width' in indicators:
                    current_volatility = indicators.get('bbands_width', 1.0) / 20.0
                else:
                    # Default fallback
                    current_volatility = 0.01
                    
            # If still no volatility, use default
            if current_volatility is None:
                return 1.0
                
            # Get baseline volatility from config or historical data
            baseline_volatility = self.config['market_analysis'].get('volatility', {}).get('baseline', 0.01)
            
            # Calculate ratio of current to baseline volatility
            volatility_ratio = current_volatility / baseline_volatility
            
            # Use square root rule for position sizing (position size ∝ 1/√volatility)
            # But with dampening to avoid extreme adjustments
            if volatility_ratio > 1.0:
                # Higher volatility = smaller position
                volatility_factor = 1.0 / math.pow(volatility_ratio, 0.4)  # Less severe than square root
            else:
                # Lower volatility = larger position (but cap the increase)
                volatility_factor = min(1.3, math.pow(1.0 / volatility_ratio, 0.3))
                
            # Ensure factor stays within reasonable bounds
            volatility_factor = max(0.5, min(1.3, volatility_factor))
            
            # Apply additional adjustment based on volatility trend
            if 'volatility_trend' in market_data:
                vol_trend = market_data['volatility_trend']  # -1 to 1 (decreasing to increasing)
                
                # If volatility is trending up, be more conservative
                if vol_trend > 0.3:
                    volatility_factor *= max(0.8, 1.0 - vol_trend * 0.2)
                    
            return volatility_factor
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced volatility adjustment: {str(e)}")
            return 1.0
            
    def _calculate_confidence_adjustment(self, market_data: Optional[Dict]) -> float:
        """Calculate position size adjustment based on signal confidence"""
        try:
            if not market_data or 'confidence' not in market_data:
                return 1.0
                
            # Get signal confidence (expected to be between 0 and 1)
            confidence = market_data['confidence']
            
            # Define confidence thresholds
            min_confidence = 0.5  # Below this, reduce position size
            max_confidence = 0.8  # Above this, consider increasing position size
            
            if confidence < min_confidence:
                # Reduce position size for low confidence signals
                # Scale from 0.7 at 0.3 confidence to 1.0 at 0.5 confidence
                return max(0.7, 0.7 + (confidence - 0.3) * 1.5) if confidence >= 0.3 else 0.7
            elif confidence > max_confidence:
                # Increase position size for high confidence signals (up to 20% increase)
                return min(1.2, 1.0 + (confidence - max_confidence) * 0.5)
            else:
                # Default for normal confidence range
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Error calculating confidence adjustment: {str(e)}")
            return 1.0
            
    async def _calculate_exposure_factor(self, symbol: str, position_size: float, account_balance: float) -> float:
        """Calculate adjustment factor based on account exposure limits"""
        try:
            # Get current exposure
            current_exposure = await self._get_current_exposure()
            
            # Calculate total exposure as percentage of account balance
            total_exposure_pct = sum(current_exposure.values()) / account_balance if account_balance > 0 else 0
            
            # Get maximum allowed exposure
            max_exposure = self.config['risk_management'].get('portfolio', {}).get('max_exposure', 0.8)
            
            # If adding this position would exceed max exposure, scale it down
            estimated_new_position_value = position_size * self._get_position_notional_value(symbol)
            estimated_new_exposure_pct = total_exposure_pct + (estimated_new_position_value / account_balance)
            
            if estimated_new_exposure_pct > max_exposure:
                # Scale down position to fit within max exposure
                available_exposure_pct = max(0, max_exposure - total_exposure_pct)
                max_position_value = available_exposure_pct * account_balance
                max_position_size = max_position_value / self._get_position_notional_value(symbol)
                
                # Return factor to scale down position
                return max_position_size / position_size if position_size > 0 else 0.0
                
            # Check symbol-specific exposure
            symbol_exposure = current_exposure.get(symbol, 0)
            symbol_exposure_pct = symbol_exposure / account_balance if account_balance > 0 else 0
            
            max_symbol_exposure = self.config['risk_management'].get('portfolio', {}).get('max_exposure_per_instrument', 0.3)
            
            estimated_new_symbol_exposure_pct = symbol_exposure_pct + (estimated_new_position_value / account_balance)
            
            if estimated_new_symbol_exposure_pct > max_symbol_exposure:
                # Scale down position to fit within max symbol exposure
                available_symbol_exposure = max(0, max_symbol_exposure - symbol_exposure_pct)
                max_symbol_position_value = available_symbol_exposure * account_balance
                max_symbol_position_size = max_symbol_position_value / self._get_position_notional_value(symbol)
                
                return max_symbol_position_size / position_size if position_size > 0 else 0.0
                
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating exposure factor: {str(e)}")
            return 1.0
            
    def _get_position_notional_value(self, symbol: str) -> float:
        """Calculate notional value of a position with lot size 1.0"""
        try:
            # Get symbol properties from MT5
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 10000.0  # Default value for forex
                
            contract_size = symbol_info.trade_contract_size
            current_price = (symbol_info.ask + symbol_info.bid) / 2
            
            return contract_size * current_price
            
        except Exception as e:
            self.logger.error(f"Error getting position notional value: {str(e)}")
            return 10000.0  # Default fallback value
            
    def _detect_market_regime(self, predictions: Dict, symbol: str) -> Dict:
        """Detect market regime from ML predictions"""
        try:
            # Extract relevant predictions
            trend_prediction = predictions.get('trend', 0)  # -1 to 1
            volatility_prediction = predictions.get('volatility', 0.5)  # 0 to 1
            mean_reversion = predictions.get('mean_reversion', 0)  # -1 to 1
            
            # Determine market regime
            if volatility_prediction > 0.7:  # High volatility
                if trend_prediction > 0.5:
                    regime = 'volatile_bullish'
                elif trend_prediction < -0.5:
                    regime = 'volatile_bearish'
                else:
                    regime = 'volatile'
            elif abs(trend_prediction) > 0.5:  # Strong trend
                if trend_prediction > 0:
                    regime = 'bullish_trend'
                else:
                    regime = 'bearish_trend'
            elif abs(mean_reversion) > 0.5:  # Strong mean reversion
                regime = 'ranging'
            else:
                regime = 'undefined'
                
            # Calculate confidence in regime classification
            confidence = max(abs(trend_prediction), volatility_prediction, abs(mean_reversion))
            
            return {
                'regime': regime,
                'confidence': confidence,
                'trend': trend_prediction,
                'volatility': volatility_prediction,
                'mean_reversion': mean_reversion
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return {'regime': 'unknown', 'confidence': 0.5}
            
    def _calculate_kelly_fraction(self, symbol: str, is_long: bool, regime: str = 'unknown') -> float:
        """Calculate Kelly criterion fraction based on historical performance"""
        try:
            # Get win rate and win/loss ratio for the specific conditions
            win_rate = self._get_historical_win_rate(symbol, is_long, regime)
            avg_win_loss_ratio = self._get_historical_win_loss_ratio(symbol, is_long, regime)
            
            # Calculate Kelly fraction: f* = (p*b - q)/b where p=win rate, q=1-p, b=win/loss ratio
            kelly_fraction = win_rate - ((1 - win_rate) / avg_win_loss_ratio)
            
            # Limit kelly fraction for safety
            kelly_fraction = max(0, min(kelly_fraction, 0.5))
            
            return kelly_fraction
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly fraction: {str(e)}")
            return 0.5  # Conservative default
            
    def _get_historical_win_rate(self, symbol: str, is_long: bool, regime: str = 'unknown') -> float:
        """Get historical win rate for the symbol, direction and regime"""
        try:
            # Filter trades for the symbol, direction and regime
            direction = 'long' if is_long else 'short'
            relevant_trades = [t for t in self.trade_history 
                              if t['symbol'] == symbol and t['direction'] == direction]
            
            # Further filter by regime if available
            if regime != 'unknown':
                relevant_trades = [t for t in relevant_trades 
                                  if t.get('market_regime', 'unknown') == regime]
            
            if not relevant_trades:
                # If no trades for this specific regime, try without regime filter
                if regime != 'unknown':
                    return self._get_historical_win_rate(symbol, is_long, 'unknown')
                # If still no data, use default
                return 0.5
                
            # Calculate win rate
            wins = sum(1 for t in relevant_trades if t['profit'] > 0)
            return wins / len(relevant_trades)
            
        except Exception as e:
            self.logger.error(f"Error calculating historical win rate: {str(e)}")
            return 0.5

    def reset_daily_metrics(self):
        """Reset daily metrics (called at the start of a new trading day)"""
        try:
            self.daily_loss = 0.0
            self.logger.info("Daily risk metrics reset")
            self.save_state()
            
        except Exception as e:
            self.logger.error(f"Error resetting daily metrics: {str(e)}")
            
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics for monitoring"""
        return {
            'daily_loss': self.daily_loss,
            'max_daily_loss': self.max_daily_loss,
            'daily_loss_percentage': self.daily_loss / self.max_daily_loss if self.max_daily_loss > 0 else 0,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'in_recovery_mode': self.in_recovery_mode,
            'risk_per_trade': self._get_current_risk_percentage()
        }
        
    def analyze_risk_performance(self) -> Dict:
        """Analyze risk management performance"""
        try:
            if not self.trade_history:
                return {"error": "No trade history available"}
                
            # Calculate win rate and profit factor
            wins = [t for t in self.trade_history if t['profit'] > 0]
            losses = [t for t in self.trade_history if t['profit'] < 0]
            
            win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0
            
            total_profit = sum(t['profit'] for t in wins)
            total_loss = sum(abs(t['profit']) for t in losses)
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate maximum consecutive losses
            max_consecutive = 0
            current_consecutive = 0
            
            for trade in self.trade_history:
                if trade['profit'] < 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
                    
            # Calculate maximum historical drawdown
            if len(self.risk_metrics_history) > 0:
                max_historical_dd = max(item['drawdown'] for item in self.risk_metrics_history)
            else:
                max_historical_dd = self.current_drawdown
                
            # Add regime performance analysis
            regime_performance = {}
            strategy_performance = {}
            
            # Analyze performance by market regime
            for trade in self.trade_history:
                regime = trade.get('market_regime', 'unknown')
                if regime not in regime_performance:
                    regime_performance[regime] = {'trades': 0, 'wins': 0, 'profit': 0.0}
                
                regime_performance[regime]['trades'] += 1
                profit = trade.get('profit', 0.0)
                
                if profit > 0:
                    regime_performance[regime]['wins'] += 1
                regime_performance[regime]['profit'] += profit
                
                # Also track by strategy type
                strategy = trade.get('strategy_type', 'unknown')
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'trades': 0, 'wins': 0, 'profit': 0.0}
                    
                strategy_performance[strategy]['trades'] += 1
                if profit > 0:
                    strategy_performance[strategy]['wins'] += 1
                strategy_performance[strategy]['profit'] += profit
            
            # Calculate win rates and average profits
            for regime, data in regime_performance.items():
                if data['trades'] > 0:
                    data['win_rate'] = data['wins'] / data['trades']
                    data['avg_profit'] = data['profit'] / data['trades']
                else:
                    data['win_rate'] = 0
                    data['avg_profit'] = 0
                    
            for strategy, data in strategy_performance.items():
                if data['trades'] > 0:
                    data['win_rate'] = data['wins'] / data['trades']
                    data['avg_profit'] = data['profit'] / data['trades']
                else:
                    data['win_rate'] = 0
                    data['avg_profit'] = 0
                
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_consecutive_losses': max_consecutive,
                'max_historical_drawdown': max_historical_dd,
                'current_drawdown': self.current_drawdown,
                'recovery_periods': sum(1 for item in self.risk_metrics_history if item['in_recovery_mode']),
                'regime_performance': regime_performance,
                'strategy_performance': strategy_performance
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk performance: {str(e)}")
            return {"error": str(e)}
