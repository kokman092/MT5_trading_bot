from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import MetaTrader5 as mt5
from ..analytics.market_analyzer import MarketAnalyzer
from ..risk.dynamic_risk import DynamicRiskManager
from ..execution.smart_executor import SmartExecutor
from ..ml.model_trainer import ModelTrainer
from ..optimization.continuous_improver import ContinuousImprover

@dataclass
class MarketCondition:
    volatility: float  # Current volatility
    trend_strength: float  # Trend strength
    liquidity: float  # Market liquidity
    regime: str  # Market regime
    correlation: float  # Asset correlation

@dataclass
class TradingMetrics:
    profit_factor: float  # Gross profit / Gross loss
    win_rate: float  # Win percentage
    sharpe_ratio: float  # Risk-adjusted return
    max_drawdown: float  # Maximum drawdown
    recovery_time: int  # Days to recover

class AdvancedTrader:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('advanced_trader')
        
        # Initialize components
        self._init_trading_components()
        self._init_trading_parameters()
        
    def _init_trading_components(self):
        """Initialize trading components"""
        try:
            # Core components
            self.market_analyzer = MarketAnalyzer(self.config)
            self.risk_manager = DynamicRiskManager(self.config)
            self.executor = SmartExecutor(self.config)
            self.model_trainer = ModelTrainer(self.config)
            self.improver = ContinuousImprover(self.config)
            
            # Initialize MT5 connection
            if not mt5.initialize():
                raise Exception("MT5 initialization failed")
                
        except Exception as e:
            self.logger.error(f"Component initialization error: {str(e)}")
            
    def _init_trading_parameters(self):
        """Initialize trading parameters"""
        # Strategy parameters
        self.strategy_params = {
            'scalping': {
                'timeframes': ['M1', 'M5'],
                'max_spread': 3.0,
                'min_volatility': 0.1,
                'max_trades': 20,
                'profit_target': 0.05,
                'stop_loss': 0.03
            },
            'momentum': {
                'timeframes': ['M5', 'M15'],
                'trend_strength': 0.7,
                'entry_threshold': 0.8,
                'max_trades': 10,
                'profit_target': 0.1,
                'stop_loss': 0.05
            },
            'mean_reversion': {
                'timeframes': ['M15', 'H1'],
                'zscore_threshold': 2.0,
                'mean_period': 20,
                'max_trades': 5,
                'profit_target': 0.15,
                'stop_loss': 0.07
            }
        }
        
        # Risk parameters
        self.risk_params = {
            'max_risk_per_trade': 0.05,  # 5% max risk
            'max_daily_risk': 0.20,  # 20% max daily risk
            'max_drawdown': 0.30,  # 30% max drawdown
            'position_scaling': {
                'win_increase': 0.01,  # +1% after win
                'loss_decrease': 0.02  # -2% after loss
            },
            'volatility_adjustments': {
                'high_vol_factor': 0.5,  # Reduce size by 50%
                'low_vol_factor': 1.5  # Increase size by 50%
            }
        }
        
        # Execution parameters
        self.execution_params = {
            'max_slippage': 2,  # Maximum allowed slippage in points
            'retry_attempts': 3,  # Number of retry attempts
            'order_types': {
                'market': True,
                'limit': True,
                'stop': True
            },
            'smart_routing': True,
            'partial_fills': True
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'min_profit_factor': 2.0,
            'min_win_rate': 0.65,
            'min_sharpe': 2.0,
            'max_recovery_days': 14,
            'consistency_score': 0.7
        }
        
    async def execute_strategy(
        self,
        symbol: str,
        strategy_type: str
    ) -> Dict:
        """Execute trading strategy"""
        try:
            # Get market conditions
            conditions = await self._analyze_market_conditions(symbol)
            
            # Validate trading conditions
            if not await self._validate_trading_conditions(
                conditions,
                strategy_type
            ):
                return {
                    'executed': False,
                    'reason': 'Invalid market conditions'
                }
                
            # Generate trading signals
            signals = await self._generate_trading_signals(
                symbol,
                strategy_type,
                conditions
            )
            
            # Validate signals
            if not signals['valid']:
                return {
                    'executed': False,
                    'reason': 'No valid signals'
                }
                
            # Calculate position size
            position_size = await self.risk_manager.calculate_position_size(
                symbol,
                signals['entry'],
                signals['stop_loss']
            )
            
            # Execute trade
            trade_result = await self.executor.execute_trade(
                symbol,
                signals['direction'],
                position_size,
                signals['entry'],
                signals['stop_loss'],
                signals['take_profit']
            )
            
            # Update metrics
            await self._update_performance_metrics(trade_result)
            
            return {
                'executed': True,
                'trade_id': trade_result['trade_id'],
                'entry': trade_result['entry_price'],
                'size': trade_result['position_size'],
                'metrics': trade_result['metrics']
            }
            
        except Exception as e:
            self.logger.error(f"Strategy execution error: {str(e)}")
            return {'executed': False, 'reason': str(e)}
            
    async def _analyze_market_conditions(
        self,
        symbol: str
    ) -> MarketCondition:
        """Analyze current market conditions"""
        try:
            # Get market data
            volatility = await self.market_analyzer.calculate_volatility(symbol)
            trend = await self.market_analyzer.analyze_trend(symbol)
            liquidity = await self.market_analyzer.assess_liquidity(symbol)
            regime = await self.market_analyzer.detect_regime(symbol)
            correlation = await self.market_analyzer.calculate_correlation(symbol)
            
            return MarketCondition(
                volatility=volatility,
                trend_strength=trend['strength'],
                liquidity=liquidity,
                regime=regime,
                correlation=correlation
            )
            
        except Exception as e:
            self.logger.error(f"Market analysis error: {str(e)}")
            return None
            
    async def _generate_trading_signals(
        self,
        symbol: str,
        strategy_type: str,
        conditions: MarketCondition
    ) -> Dict:
        """Generate trading signals"""
        try:
            signals = {
                'valid': False,
                'direction': None,
                'entry': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0
            }
            
            # Get strategy parameters
            params = self.strategy_params[strategy_type]
            
            # Generate signals based on strategy
            if strategy_type == 'scalping':
                signals = await self._generate_scalping_signals(
                    symbol,
                    conditions,
                    params
                )
            elif strategy_type == 'momentum':
                signals = await self._generate_momentum_signals(
                    symbol,
                    conditions,
                    params
                )
            elif strategy_type == 'mean_reversion':
                signals = await self._generate_mean_reversion_signals(
                    symbol,
                    conditions,
                    params
                )
                
            # Validate signals
            if signals['valid']:
                # Adjust for market conditions
                signals = await self._adjust_signals_for_conditions(
                    signals,
                    conditions
                )
                
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {str(e)}")
            return {'valid': False}
            
    async def _update_performance_metrics(
        self,
        trade_result: Dict
    ) -> None:
        """Update performance metrics"""
        try:
            # Calculate metrics
            metrics = await self._calculate_metrics(trade_result)
            
            # Update continuous improver
            await self.improver.update_metrics(metrics)
            
            # Check performance thresholds
            if not await self._check_performance_thresholds(metrics):
                await self._adjust_trading_parameters()
                
        except Exception as e:
            self.logger.error(f"Metrics update error: {str(e)}")
            
    async def _check_performance_thresholds(
        self,
        metrics: TradingMetrics
    ) -> bool:
        """Check if performance meets thresholds"""
        try:
            # Check each threshold
            checks = [
                metrics.profit_factor >= self.performance_thresholds['min_profit_factor'],
                metrics.win_rate >= self.performance_thresholds['min_win_rate'],
                metrics.sharpe_ratio >= self.performance_thresholds['min_sharpe'],
                metrics.recovery_time <= self.performance_thresholds['max_recovery_days']
            ]
            
            return all(checks)
            
        except Exception as e:
            self.logger.error(f"Performance check error: {str(e)}")
            return False
