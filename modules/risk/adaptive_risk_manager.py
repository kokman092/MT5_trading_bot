import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import MetaTrader5 as mt5

class AdaptiveRiskManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk limits
        self.max_risk_per_trade = config.get('MAX_RISK_PER_TRADE', 0.02)
        self.max_portfolio_risk = config.get('MAX_PORTFOLIO_RISK', 0.10)
        self.min_margin_level = config.get('MIN_MARGIN_LEVEL', 1.5)
        self.max_drawdown = config.get('MAX_DRAWDOWN', 0.10)
        
        # Position sizing parameters
        self.base_position_size = config.get('BASE_POSITION_SIZE', 0.01)
        self.volatility_factor = config.get('VOLATILITY_FACTOR', 1.0)
        self.correlation_threshold = config.get('CORRELATION_THRESHOLD', 0.7)
        
        # Risk adjustment factors
        self.regime_risk_factors = {
            'volatile_bearish': 0.5,    # Reduce risk in volatile bear markets
            'volatile_bullish': 0.7,    # Slightly reduce risk in volatile bull markets
            'trending_bearish': 0.8,    # Moderately reduce risk in bear trends
            'trending_bullish': 1.0,    # Normal risk in bull trends
            'ranging': 0.6              # Reduce risk in ranging markets
        }
        
        # Performance tracking
        self.trade_history = []
        self.daily_returns = {}
        self.drawdown_history = []
        
        # Load history if exists
        self.load_history()
    
    def calculate_position_size(self, symbol, signal_confidence, market_regime, current_positions):
        """Calculate adaptive position size based on multiple factors"""
        try:
            # Get account info
            account_info = mt5.account_info()
            if not account_info:
                self.logger.error("Failed to get account info")
                return 0
            
            # Check margin level
            if account_info.margin_level < self.min_margin_level * 100:
                self.logger.warning(f"Margin level too low: {account_info.margin_level}%")
                return 0
            
            # Base position size
            equity = account_info.equity
            position_size = equity * self.base_position_size
            
            # Adjust for signal confidence
            position_size *= signal_confidence
            
            # Adjust for market regime
            regime_factor = self.regime_risk_factors.get(market_regime, 0.5)
            position_size *= regime_factor
            
            # Adjust for volatility
            volatility = self.calculate_volatility(symbol)
            volatility_adjustment = 1.0 / (volatility * self.volatility_factor) if volatility > 0 else 1.0
            position_size *= volatility_adjustment
            
            # Adjust for correlation
            correlation_factor = self.calculate_correlation_factor(symbol, current_positions)
            position_size *= correlation_factor
            
            # Adjust for drawdown
            drawdown_factor = self.calculate_drawdown_factor()
            position_size *= drawdown_factor
            
            # Ensure position size doesn't exceed risk limits
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Failed to get symbol info for {symbol}")
                return 0
            
            max_position = self.calculate_max_position(equity, symbol_info)
            position_size = min(position_size, max_position)
            
            # Round to symbol lot step
            position_size = self.round_position_size(position_size, symbol_info)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    def calculate_volatility(self, symbol, period=20):
        """Calculate symbol volatility"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, period)
            if rates is None:
                return float('inf')
            
            df = pd.DataFrame(rates)
            returns = np.log(df['close'] / df['close'].shift(1))
            return returns.std() * np.sqrt(252)  # Annualized volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return float('inf')
    
    def calculate_correlation_factor(self, symbol, current_positions):
        """Calculate correlation-based position adjustment"""
        try:
            if not current_positions:
                return 1.0
            
            # Get historical data
            period = 100
            symbol_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, period)
            if symbol_rates is None:
                return 0.5
            
            symbol_df = pd.DataFrame(symbol_rates)
            symbol_returns = np.log(symbol_df['close'] / symbol_df['close'].shift(1))
            
            # Calculate correlations with existing positions
            correlations = []
            for pos in current_positions:
                pos_rates = mt5.copy_rates_from_pos(pos.symbol, mt5.TIMEFRAME_H1, 0, period)
                if pos_rates is not None:
                    pos_df = pd.DataFrame(pos_rates)
                    pos_returns = np.log(pos_df['close'] / pos_df['close'].shift(1))
                    corr = abs(symbol_returns.corr(pos_returns))
                    correlations.append(corr)
            
            if not correlations:
                return 1.0
            
            # Reduce position size based on highest correlation
            max_correlation = max(correlations)
            if max_correlation > self.correlation_threshold:
                return 1.0 - (max_correlation - self.correlation_threshold)
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation factor: {str(e)}")
            return 0.5
    
    def calculate_drawdown_factor(self):
        """Calculate drawdown-based risk adjustment"""
        try:
            if not self.trade_history:
                return 1.0
            
            # Calculate current drawdown
            equity_curve = [1.0]
            for trade in self.trade_history:
                equity_curve.append(equity_curve[-1] * (1 + trade['return']))
            
            equity_curve = np.array(equity_curve)
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak
            current_drawdown = abs(drawdown[-1])
            
            # Reduce position size as drawdown approaches limit
            if current_drawdown > self.max_drawdown:
                return 0.0  # Stop trading
            elif current_drawdown > self.max_drawdown * 0.7:
                # Gradually reduce position size
                return 1.0 - (current_drawdown / self.max_drawdown)
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown factor: {str(e)}")
            return 0.5
    
    def calculate_max_position(self, equity, symbol_info):
        """Calculate maximum allowed position size"""
        try:
            # Get symbol contract size and margin requirements
            contract_size = symbol_info.trade_contract_size
            margin_rate = symbol_info.margin_initial
            
            # Calculate maximum position based on risk limits
            max_loss_amount = equity * self.max_risk_per_trade
            max_position = max_loss_amount / (contract_size * margin_rate)
            
            return max_position
            
        except Exception as e:
            self.logger.error(f"Error calculating max position: {str(e)}")
            return 0
    
    def round_position_size(self, position_size, symbol_info):
        """Round position size to symbol's lot step"""
        try:
            lot_step = symbol_info.volume_step
            return round(position_size / lot_step) * lot_step
        except Exception as e:
            self.logger.error(f"Error rounding position size: {str(e)}")
            return 0
    
    def update_trade_history(self, trade):
        """Update trade history and risk metrics"""
        try:
            self.trade_history.append(trade)
            
            # Update daily returns
            trade_date = trade['time'].date()
            if trade_date not in self.daily_returns:
                self.daily_returns[trade_date] = []
            self.daily_returns[trade_date].append(trade['return'])
            
            # Calculate and store drawdown
            equity_curve = [1.0]
            for t in self.trade_history:
                equity_curve.append(equity_curve[-1] * (1 + t['return']))
            
            peak = max(equity_curve)
            current_equity = equity_curve[-1]
            drawdown = (peak - current_equity) / peak
            
            self.drawdown_history.append({
                'time': trade['time'].isoformat(),
                'drawdown': drawdown
            })
            
            # Save updated history
            self.save_history()
            
        except Exception as e:
            self.logger.error(f"Error updating trade history: {str(e)}")
    
    def get_risk_metrics(self):
        """Calculate current risk metrics"""
        try:
            if not self.trade_history:
                return None
            
            # Calculate daily metrics
            daily_returns = pd.Series([sum(returns) for returns in self.daily_returns.values()])
            
            metrics = {
                'sharpe_ratio': np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 1 else 0,
                'max_drawdown': max([d['drawdown'] for d in self.drawdown_history]) if self.drawdown_history else 0,
                'daily_var': np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0,
                'win_rate': sum(1 for t in self.trade_history if t['return'] > 0) / len(self.trade_history),
                'avg_return': np.mean([t['return'] for t in self.trade_history]),
                'volatility': np.std([t['return'] for t in self.trade_history]) * np.sqrt(252),
                'current_drawdown': self.drawdown_history[-1]['drawdown'] if self.drawdown_history else 0
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return None
    
    def save_history(self):
        """Save trade and risk history"""
        try:
            history_data = {
                'trade_history': [
                    {**trade, 'time': trade['time'].isoformat()}
                    for trade in self.trade_history
                ],
                'daily_returns': {
                    date.isoformat(): returns
                    for date, returns in self.daily_returns.items()
                },
                'drawdown_history': self.drawdown_history
            }
            
            with open('risk_history.json', 'w') as f:
                json.dump(history_data, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving history: {str(e)}")
    
    def load_history(self):
        """Load trade and risk history"""
        try:
            with open('risk_history.json', 'r') as f:
                history_data = json.load(f)
                
                self.trade_history = [
                    {**trade, 'time': datetime.fromisoformat(trade['time'])}
                    for trade in history_data['trade_history']
                ]
                
                self.daily_returns = {
                    datetime.fromisoformat(date).date(): returns
                    for date, returns in history_data['daily_returns'].items()
                }
                
                self.drawdown_history = history_data['drawdown_history']
                
        except FileNotFoundError:
            self.logger.info("No risk history file found. Starting fresh.")
        except Exception as e:
            self.logger.error(f"Error loading history: {str(e)}")
    
    def check_risk_limits(self, symbol, order_type, volume):
        """Check if order complies with risk limits"""
        try:
            # Get account info
            account_info = mt5.account_info()
            if not account_info:
                return False, "Failed to get account info"
            
            # Check margin level
            if account_info.margin_level < self.min_margin_level * 100:
                return False, f"Margin level too low: {account_info.margin_level}%"
            
            # Calculate order value
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return False, f"Failed to get symbol info for {symbol}"
            
            order_value = volume * symbol_info.trade_contract_size * symbol_info.ask
            
            # Check position value against equity
            if order_value / account_info.equity > self.max_risk_per_trade:
                return False, "Order exceeds maximum position size"
            
            # Check total risk exposure
            positions = mt5.positions_get()
            if positions:
                total_exposure = sum(pos.volume * mt5.symbol_info(pos.symbol).trade_contract_size * 
                                  mt5.symbol_info(pos.symbol).ask for pos in positions)
                total_exposure += order_value
                
                if total_exposure / account_info.equity > self.max_portfolio_risk:
                    return False, "Order would exceed maximum portfolio risk"
            
            return True, "Order within risk limits"
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
            return False, f"Error checking risk limits: {str(e)}"
