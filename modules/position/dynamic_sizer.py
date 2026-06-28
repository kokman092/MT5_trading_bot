from typing import Dict, Optional
import numpy as np
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from ..deployment.error_handler import ErrorHandler

class DynamicSizer:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('dynamic_sizer')
        self.performance_history = []
        self.last_adjustment = datetime.now()
        self.growth_factor = 1.0
        
    async def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        sentiment_score: float = 0.0
    ) -> float:
        """Calculate dynamic position size based on multiple factors"""
        try:
            # Get base position size
            base_size = await self._get_base_size(entry_price, stop_loss)
            
            # Apply adjustments
            size = base_size
            size *= await self._get_growth_adjustment()
            size *= await self._get_performance_adjustment()
            size *= await self._get_volatility_adjustment(symbol)
            size *= await self._get_sentiment_adjustment(sentiment_score)
            
            # Apply limits
            size = await self._apply_size_limits(size, symbol)
            
            return size
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {str(e)}")
            return 0.0
            
    async def _get_base_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate base position size using fixed risk"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return 0.0
                
            # Calculate risk amount
            risk_percentage = self.config.get('RISK_PERCENTAGE', 1.0)
            risk_amount = account_info.balance * (risk_percentage / 100)
            
            # Calculate position size based on stop loss
            stop_distance = abs(entry_price - stop_loss)
            if stop_distance == 0:
                return 0.0
                
            return risk_amount / stop_distance
            
        except Exception as e:
            self.logger.error(f"Base size calculation error: {str(e)}")
            return 0.0
            
    async def _get_growth_adjustment(self) -> float:
        """Calculate position size adjustment based on account growth"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return 1.0
                
            initial_balance = self.config.get('INITIAL_BALANCE', account_info.balance)
            if initial_balance <= 0:
                return 1.0
                
            # Calculate growth factor
            growth = account_info.balance / initial_balance
            
            # Apply progressive scaling
            if growth > 1:
                self.growth_factor = np.sqrt(growth)
            else:
                self.growth_factor = growth
                
            # Limit maximum growth factor
            max_growth = self.config.get('MAX_GROWTH_FACTOR', 3.0)
            return min(self.growth_factor, max_growth)
            
        except Exception as e:
            self.logger.error(f"Growth adjustment error: {str(e)}")
            return 1.0
            
    async def _get_performance_adjustment(self) -> float:
        """Adjust position size based on recent performance"""
        try:
            # Get recent trades
            from_date = datetime.now() - timedelta(days=30)
            trades = mt5.history_deals_get(from_date, datetime.now())
            
            if not trades:
                return 1.0
                
            # Calculate win rate and profit factor
            winning_trades = [t for t in trades if t.profit > 0]
            losing_trades = [t for t in trades if t.profit < 0]
            
            win_rate = len(winning_trades) / len(trades)
            
            total_profit = sum(t.profit for t in winning_trades)
            total_loss = abs(sum(t.profit for t in losing_trades))
            profit_factor = total_profit / total_loss if total_loss > 0 else 1.0
            
            # Calculate adjustment factor
            performance_score = (win_rate + profit_factor) / 2
            
            # Apply progressive scaling
            if performance_score > 0.6:
                return 1.2
            elif performance_score < 0.4:
                return 0.8
                
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Performance adjustment error: {str(e)}")
            return 1.0
            
    async def _get_volatility_adjustment(self, symbol: str) -> float:
        """Adjust position size based on market volatility"""
        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
            if rates is None:
                return 1.0
                
            # Calculate volatility
            prices = np.array([rate[4] for rate in rates])  # Close prices
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(24)  # Annualized
            
            # Compare to baseline volatility
            baseline_vol = self.config.get('BASELINE_VOLATILITY', 0.2)
            vol_ratio = volatility / baseline_vol
            
            # Adjust position size inversely to volatility
            if vol_ratio > 1.5:
                return 0.8
            elif vol_ratio < 0.5:
                return 1.2
                
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Volatility adjustment error: {str(e)}")
            return 1.0
            
    async def _get_sentiment_adjustment(self, sentiment_score: float) -> float:
        """Adjust position size based on market sentiment"""
        try:
            # Scale sentiment score to adjustment factor
            if abs(sentiment_score) < 0.2:
                return 1.0  # Neutral sentiment
            elif sentiment_score > 0.5:
                return 1.2  # Strong positive sentiment
            elif sentiment_score < -0.5:
                return 0.8  # Strong negative sentiment
                
            return 1.0 + (sentiment_score * 0.2)
            
        except Exception as e:
            self.logger.error(f"Sentiment adjustment error: {str(e)}")
            return 1.0
            
    async def _apply_size_limits(self, size: float, symbol: str) -> float:
        """Apply minimum and maximum position size limits"""
        try:
            # Get symbol information
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return size
                
            # Apply volume step size
            step_size = symbol_info.volume_step
            size = round(size / step_size) * step_size
            
            # Apply minimum size
            min_size = max(
                symbol_info.volume_min,
                self.config.get('MIN_POSITION_SIZE', 0.01)
            )
            size = max(size, min_size)
            
            # Apply maximum size
            max_size = min(
                symbol_info.volume_max,
                self.config.get('MAX_POSITION_SIZE', 100.0)
            )
            size = min(size, max_size)
            
            return size
            
        except Exception as e:
            self.logger.error(f"Size limit application error: {str(e)}")
            return size
            
    def get_sizing_metrics(self) -> Dict:
        """Get current position sizing metrics"""
        try:
            return {
                'growth_factor': self.growth_factor,
                'last_adjustment': self.last_adjustment.isoformat(),
                'performance_history': self.performance_history[-10:],
                'current_limits': {
                    'min_size': self.config.get('MIN_POSITION_SIZE', 0.01),
                    'max_size': self.config.get('MAX_POSITION_SIZE', 100.0)
                }
            }
        except Exception as e:
            self.logger.error(f"Metrics calculation error: {str(e)}")
            return {}
