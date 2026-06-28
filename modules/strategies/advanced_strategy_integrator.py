import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime

from .event_driven_strategy import EventDrivenStrategy
from .adaptive_strategy_manager import AdaptiveStrategyManager
from .reinforcement_learning_strategy import ReinforcementLearningStrategy
from .enhanced_hft_strategy import EnhancedHFTStrategy
from .enhanced_scalping_strategy import EnhancedScalpingStrategy
from .enhanced_pairs_trading_strategy import EnhancedPairsTradingStrategy
from .enhanced_grid_trading_strategy import EnhancedGridTradingStrategy
from .enhanced_mean_reversion_strategy import EnhancedMeanReversionStrategy
from .enhanced_momentum_strategy import EnhancedMomentumStrategy
from ..risk.risk_manager import RiskManager
from ..models.model_selector import ModelSelector

@dataclass
class StrategySignal:
    strategy_name: str
    signal_type: str  # 'entry', 'exit', 'modify'
    direction: str    # 'buy', 'sell', 'hold'
    symbol: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    volume: Optional[float]
    confidence: float
    timestamp: datetime
    metadata: Dict

class AdvancedStrategyIntegrator:
    def __init__(self, config: Dict):
        """Initialize the advanced strategy integrator"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize model selector and risk manager
        self.model_selector = ModelSelector(config)
        self.risk_manager = RiskManager(config)
        
        # Initialize all advanced strategies
        self._init_strategies()
        
        # Strategy weights for dynamic allocation
        self._init_strategy_weights()
        
    def _init_strategies(self):
        """Initialize all trading strategies"""
        try:
            # Initialize strategies with proper error handling
            self.strategies = {}
            strategy_classes = {
                'event_driven': (EventDrivenStrategy, {'config': self.config}),
                'adaptive': (AdaptiveStrategyManager, {
                    'config': self.config,
                    'model_selector': self.model_selector,
                    'risk_manager': self.risk_manager
                }),
                'reinforcement': (ReinforcementLearningStrategy, {'config': self.config}),
                'hft': (EnhancedHFTStrategy, {'config': self.config}),
                'scalping': (EnhancedScalpingStrategy, {'config': self.config}),
                'pairs_trading': (EnhancedPairsTradingStrategy, {'config': self.config}),
                'grid_trading': (EnhancedGridTradingStrategy, {'config': self.config}),
                'mean_reversion': (EnhancedMeanReversionStrategy, {'config': self.config}),
                'momentum': (EnhancedMomentumStrategy, {'config': self.config})
            }
            
            # Get enabled strategies from config
            enabled_strategies = self.config.get('trading_parameters', {}).get('enabled_strategies', list(strategy_classes.keys()))
            
            for name, (strategy_class, kwargs) in strategy_classes.items():
                if name in enabled_strategies:
                    try:
                        self.strategies[name] = strategy_class(**kwargs)
                        self.logger.info(f"Successfully initialized {name} strategy")
                    except Exception as e:
                        self.logger.error(f"Error initializing {name} strategy: {str(e)}")
                        continue
            
            # Initialize performance tracking with safe defaults
            self.strategy_performance = {
                name: {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_profit': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'weight': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'average_trade': 0.0,
                    'max_consecutive_losses': 0,
                    'risk_adjusted_return': 0.0
                } for name in self.strategies.keys()
            }
            
            if not self.strategies:
                raise Exception("No strategies were successfully initialized")
            
            self.logger.info(f"Successfully initialized {len(self.strategies)} strategies")
            
        except Exception as e:
            self.logger.error(f"Strategy initialization error: {str(e)}")
            raise
            
    def _init_strategy_weights(self) -> None:
        """Initialize optimized strategy weights for different market conditions"""
        try:
            # Initialize weights only for successfully initialized strategies
            total_weight = 0.0
            self.strategy_weights = {}
            
            # Get weights from config or use defaults
            strategy_weights = self.config.get('trading_parameters', {}).get('strategy_weights', {})
            
            # Default weights for each strategy type
            default_weights = {
                'event_driven': 0.15,
                'adaptive': 0.15,
                'reinforcement': 0.15,
                'hft': 0.10,
                'scalping': 0.10,
                'pairs_trading': 0.10,
                'grid_trading': 0.10,
                'mean_reversion': 0.075,
                'momentum': 0.075
            }
            
            # Assign weights only to initialized strategies
            for name in self.strategies.keys():
                weight = strategy_weights.get(name, default_weights.get(name, 0.1))
                self.strategy_weights[name] = weight
                total_weight += weight
            
            # Normalize weights if necessary
            if total_weight != 1.0 and total_weight > 0:
                factor = 1.0 / total_weight
                self.strategy_weights = {
                    name: weight * factor 
                    for name, weight in self.strategy_weights.items()
                }
            
            # Update performance tracking with weights
            for name, weight in self.strategy_weights.items():
                if name in self.strategy_performance:
                    self.strategy_performance[name]['weight'] = weight
            
            # Initialize risk parameters from config
            risk_config = self.config.get('risk_management', {})
            self.risk_params = {
                'max_daily_risk': risk_config.get('max_daily_risk', 0.01),
                'max_position_risk': risk_config.get('risk_per_trade', 0.005),
                'max_correlated_positions': risk_config.get('max_correlated_positions', 2),
                'min_risk_reward': risk_config.get('min_risk_reward', 3.0),
                'max_drawdown': risk_config.get('max_drawdown', 0.03),
                'position_sizing': {
                    'conservative': risk_config.get('position_sizing', {}).get('conservative', 0.5),
                    'moderate': risk_config.get('position_sizing', {}).get('moderate', 0.75),
                    'aggressive': risk_config.get('position_sizing', {}).get('aggressive', 1.0)
                }
            }
            
            # Performance thresholds from config
            self.performance_thresholds = {
                'min_win_rate': self.config.get('performance_thresholds', {}).get('min_win_rate', 0.60),
                'min_profit_factor': self.config.get('performance_thresholds', {}).get('min_profit_factor', 1.5),
                'max_consecutive_losses': self.config.get('performance_thresholds', {}).get('max_consecutive_losses', 3),
                'min_trades_per_day': self.config.get('performance_thresholds', {}).get('min_trades_per_day', 2),
                'max_trades_per_day': self.config.get('performance_thresholds', {}).get('max_trades_per_day', 6)
            }
            
            self.logger.info("Strategy weights initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Strategy weight initialization error: {str(e)}")
            raise
            
    async def generate_signals(self, market_data: Dict) -> List[StrategySignal]:
        """Generate trading signals from all strategies"""
        try:
            all_signals = []
            
            # Get signals from each strategy
            for name, strategy in self.strategies.items():
                try:
                    weight = self.strategy_weights[name]
                    if weight > 0.01:  # Only use strategies with significant weight
                        signals = await strategy.generate_signals(market_data)
                        if signals:
                            # Process and filter signals
                            processed_signals = await self._process_strategy_signals(
                                signals, name, weight
                            )
                            all_signals.extend(processed_signals)
                        
                except Exception as e:
                    self.logger.error(f"Error in strategy {name}: {str(e)}")
                    continue
                    
            # Combine and filter signals
            combined_signals = await self._combine_signals(all_signals)
            
            return combined_signals
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {str(e)}")
            return []
            
    async def _process_strategy_signals(
        self,
        signals: List[Dict],
        strategy_name: str,
        weight: float
    ) -> List[StrategySignal]:
        """Process signals with enhanced filtering"""
        try:
            processed_signals = []
            
            for signal_dict in signals:
                if not isinstance(signal_dict, dict):
                    self.logger.error(f"Invalid signal format from {strategy_name}: {type(signal_dict)}")
                    continue

                # Enhanced confidence calculation
                strategy_factor = await self._get_strategy_performance_factor(strategy_name)
                market_condition_factor = await self._get_market_condition_factor(signal_dict)
                volatility_factor = await self._get_volatility_factor(signal_dict)
                
                # Get confirmations
                confirmations = await self._get_confirmation_indicators(signal_dict)
                confirmation_confidence = sum(c['confidence'] for c in confirmations) / max(len(confirmations), 1)
                
                # Calculate adjusted confidence with multiple factors
                base_confidence = signal_dict.get('confidence', 0.5)
                adjusted_confidence = (
                    base_confidence *
                    weight *
                    strategy_factor *
                    market_condition_factor *
                    volatility_factor *
                    confirmation_confidence
                )
                
                # Enhanced filtering criteria
                if (
                    adjusted_confidence >= 0.85 and  # Increased confidence threshold
                    len(confirmations) >= 3 and  # Require at least 3 confirmations
                    await self._validate_signal_conditions(signal_dict) and
                    await self._check_correlation_with_existing(signal_dict) and
                    await self._check_risk_reward_ratio(signal_dict) >= 2.0 and  # Minimum R:R ratio
                    await self._check_market_conditions(signal_dict)  # Additional market checks
                ):
                    # Create standardized signal with enhanced metadata
                    processed_signal = StrategySignal(
                        strategy_name=strategy_name,
                        signal_type=signal_dict.get('type', 'entry'),
                        direction=signal_dict.get('direction', 'none'),
                        symbol=signal_dict.get('symbol', ''),
                        entry_price=signal_dict.get('entry_price'),
                        stop_loss=await self._calculate_dynamic_stop_loss(signal_dict),
                        take_profit=await self._calculate_dynamic_take_profit(signal_dict),
                        volume=await self._calculate_position_size(signal_dict),
                        confidence=adjusted_confidence,
                        timestamp=datetime.now(),
                        metadata={
                            'strategy_factor': strategy_factor,
                            'market_condition_factor': market_condition_factor,
                            'volatility_factor': volatility_factor,
                            'confirmation_confidence': confirmation_confidence,
                            'confirmations': confirmations,
                            'risk_reward_ratio': await self._check_risk_reward_ratio(signal_dict),
                            'market_conditions': await self._get_market_conditions_summary(signal_dict)
                        }
                    )
                    
                    processed_signals.append(processed_signal)
                
            return processed_signals
            
        except Exception as e:
            self.logger.error(f"Signal processing error: {str(e)}")
            return []
            
    async def _combine_signals(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """Combine and filter signals from all strategies"""
        try:
            if not signals:
                return []
                
            # Group signals by symbol and direction
            grouped_signals = {}
            for signal in signals:
                key = (signal.symbol, signal.direction)
                if key not in grouped_signals:
                    grouped_signals[key] = []
                grouped_signals[key].append(signal)
                
            # Combine signals for each symbol/direction
            combined_signals = []
            for signals in grouped_signals.values():
                if len(signals) == 1:
                    combined_signals.append(signals[0])
                else:
                    # Combine multiple signals
                    combined_signal = await self._combine_multiple_signals(signals)
                    if combined_signal:
                        combined_signals.append(combined_signal)
                        
            # Sort by confidence
            combined_signals.sort(key=lambda x: x.confidence, reverse=True)
            
            return combined_signals
            
        except Exception as e:
            self.logger.error(f"Signal combination error: {str(e)}")
            return []
            
    def _combine_multiple_signals(self, signals: List[StrategySignal]) -> Optional[StrategySignal]:
        """Combine multiple signals for the same symbol/direction"""
        try:
            # Calculate weighted averages
            total_weight = sum(signal.confidence for signal in signals)
            if total_weight == 0:
                return None
                
            entry_prices = []
            stop_losses = []
            take_profits = []
            volumes = []
            
            for signal in signals:
                weight = signal.confidence / total_weight
                
                if signal.entry_price:
                    entry_prices.append(signal.entry_price * weight)
                if signal.stop_loss:
                    stop_losses.append(signal.stop_loss * weight)
                if signal.take_profit:
                    take_profits.append(signal.take_profit * weight)
                if signal.volume:
                    volumes.append(signal.volume * weight)
                    
            # Create combined signal
            return StrategySignal(
                strategy_name="combined",
                signal_type="entry",
                direction=signals[0].direction,
                symbol=signals[0].symbol,
                entry_price=sum(entry_prices) if entry_prices else None,
                stop_loss=sum(stop_losses) if stop_losses else None,
                take_profit=sum(take_profits) if take_profits else None,
                volume=sum(volumes) if volumes else None,
                confidence=total_weight / len(signals),  # Average confidence
                timestamp=datetime.now(),
                metadata={'source_strategies': [s.strategy_name for s in signals]}
            )
            
        except Exception as e:
            self.logger.error(f"Multiple signal combination error: {str(e)}")
            return None
            
    def _get_strategy_performance_factor(self, strategy_name: str) -> float:
        """Calculate strategy performance factor"""
        try:
            perf = self.strategy_performance[strategy_name]
            
            if perf['total_trades'] == 0:
                return 1.0
                
            # Calculate performance metrics
            win_rate = perf['winning_trades'] / perf['total_trades']
            profit_factor = perf['total_profit'] / abs(perf['total_profit']) if perf['total_profit'] != 0 else 0
            
            # Combine metrics
            performance_factor = (
                win_rate * 0.4 +
                min(profit_factor, 2.0) * 0.3 +
                max(0, 1 - perf['max_drawdown']) * 0.3
            )
            
            return max(0.5, min(performance_factor, 1.5))  # Limit factor range
            
        except Exception as e:
            self.logger.error(f"Performance factor calculation error: {str(e)}")
            return 1.0
            
    async def update_strategy_performance(self, trade_result: Dict):
        """Update strategy performance metrics"""
        try:
            strategy_name = trade_result['strategy']
            if strategy_name not in self.strategy_performance:
                return
                
            perf = self.strategy_performance[strategy_name]
            
            # Update metrics
            perf['total_trades'] += 1
            if trade_result['profit'] > 0:
                perf['winning_trades'] += 1
            perf['total_profit'] += trade_result['profit']
            
            # Update Sharpe ratio
            if 'returns' not in perf:
                perf['returns'] = []
            perf['returns'].append(trade_result['profit'])
            if len(perf['returns']) > 100:
                perf['returns'].pop(0)
            
            returns = np.array(perf['returns'])
            if len(returns) > 1:
                perf['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
                
            # Update max drawdown
            if 'equity_curve' not in perf:
                perf['equity_curve'] = []
            perf['equity_curve'].append(perf['total_profit'])
            perf['max_drawdown'] = self._calculate_max_drawdown(perf['equity_curve'])
            
            # Update strategy weights
            await self._update_strategy_weights()
            
        except Exception as e:
            self.logger.error(f"Performance update error: {str(e)}")
            
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve"""
        try:
            peak = equity_curve[0]
            max_dd = 0.0
            
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
                
            return max_dd
            
        except Exception as e:
            self.logger.error(f"Drawdown calculation error: {str(e)}")
            return 0.0
            
    async def _update_strategy_weights(self):
        """Update strategy weights based on performance"""
        try:
            total_score = 0
            scores = {}
            
            # Calculate scores for each strategy
            for name, perf in self.strategy_performance.items():
                if perf['total_trades'] == 0:
                    scores[name] = 1.0  # Default score
                else:
                    # Calculate weighted score
                    win_rate = perf['winning_trades'] / perf['total_trades']
                    profit_factor = min(2.0, perf['total_profit'] / abs(perf['total_profit'])) if perf['total_profit'] != 0 else 0
                    
                    score = (
                        win_rate * 0.4 +
                        profit_factor * 0.3 +
                        max(0, perf['sharpe_ratio']) * 0.2 +
                        max(0, 1 - perf['max_drawdown']) * 0.1
                    )
                    
                    scores[name] = max(0.1, score)  # Minimum weight of 10%
                    
                total_score += scores[name]
                
            # Update weights
            if total_score > 0:
                for name in self.strategy_weights:
                    self.strategy_weights[name] = scores[name] / total_score
                    
        except Exception as e:
            self.logger.error(f"Weight update error: {str(e)}")
            
    def get_active_strategies(self) -> List[str]:
        """Get list of currently active strategies"""
        return [name for name, weight in self.strategy_weights.items() if weight > 0.01]
        
    def get_strategy_metrics(self) -> Dict:
        """Get current strategy performance metrics"""
        return {
            name: {
                'weight': self.strategy_weights[name],
                'performance': self.strategy_performance[name]
            } for name in self.strategies.keys()
        }

    async def _get_market_condition_factor(self, signal: Dict) -> float:
        """Calculate market condition suitability factor"""
        try:
            if not isinstance(signal, dict):
                return 0.5

            market_regime = signal.get('market_regime', {})
            if not isinstance(market_regime, dict):
                return 0.5
                
            # Higher factor for strong trends in trend-following strategies
            if 'trend' in signal.get('strategy_type', '').lower():
                if 'STRONG' in market_regime.get('regime', ''):
                    return min(1.2, 0.8 + market_regime.get('confidence', 0))
                    
            # Higher factor for ranging markets in mean-reversion strategies
            if 'reversion' in signal.get('strategy_type', '').lower():
                if market_regime.get('regime') == 'RANGING':
                    return min(1.2, 0.8 + market_regime.get('confidence', 0))
                    
            return 0.8
            
        except Exception as e:
            self.logger.error(f"Market condition factor error: {str(e)}")
            return 0.5

    async def _get_volatility_factor(self, signal: Dict) -> float:
        """Calculate volatility suitability factor"""
        try:
            if not isinstance(signal, dict):
                return 0.5

            volatility = signal.get('volatility', 0.5)
            strategy_type = signal.get('strategy_type', '').lower()
            
            if 'scalping' in strategy_type:
                return 1.2 if volatility < 0.3 else 0.8
            elif 'trend' in strategy_type:
                return 1.2 if 0.2 <= volatility <= 0.6 else 0.8
            else:
                return 1.0 if 0.1 <= volatility <= 0.4 else 0.8
                
        except Exception as e:
            self.logger.error(f"Volatility factor error: {str(e)}")
            return 0.8

    async def _validate_signal_conditions(self, signal: Dict) -> bool:
        """Validate multiple conditions for signal quality"""
        try:
            # Check minimum required confirmations
            confirmations = signal.get('confirmations', [])
            if len(confirmations) < 3:  # Require at least 3 confirmations
                return False
                
            # Check signal strength
            if signal.get('strength', 0) < 0.7:  # Minimum strength threshold
                return False
                
            # Check market conditions
            regime = signal.get('market_regime', {})
            if regime.get('confidence', 0) < 0.6:  # Minimum regime confidence
                return False
                
            # Check timeframe alignment
            if not signal.get('timeframe_alignment', False):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {str(e)}")
            return False

    async def _calculate_dynamic_stop_loss(self, signal: Dict) -> Optional[float]:
        """Calculate dynamic stop loss based on market conditions"""
        try:
            if 'entry_price' not in signal:
                return None
                
            volatility = signal.get('volatility', 0.5)
            regime = signal.get('market_regime', {}).get('regime', 'UNKNOWN')
            
            # Base ATR multiple on market regime
            if 'STRONG' in regime:
                atr_multiple = 2.0
            elif 'WEAK' in regime:
                atr_multiple = 1.5
            else:
                atr_multiple = 1.2
                
            # Adjust for volatility
            atr_multiple *= (1 + volatility)
            
            # Calculate stop loss
            atr = signal.get('atr', signal.get('entry_price') * 0.001)  # Default 0.1%
            stop_distance = atr * atr_multiple
            
            return signal['entry_price'] * (0.99 if signal['direction'] == 'buy' else 1.01)
            
        except Exception as e:
            self.logger.error(f"Stop loss calculation error: {str(e)}")
            return None

    async def _calculate_dynamic_take_profit(self, signal: Dict) -> Optional[float]:
        """Calculate dynamic take profit based on market conditions"""
        try:
            if 'entry_price' not in signal:
                return None
                
            volatility = signal.get('volatility', 0.5)
            regime = signal.get('market_regime', {}).get('regime', 'UNKNOWN')
            
            # Base risk:reward on market regime
            if 'STRONG' in regime:
                rr_ratio = 3.0
            elif 'WEAK' in regime:
                rr_ratio = 2.5
            else:
                rr_ratio = 2.0
                
            # Adjust for volatility
            rr_ratio *= (1 + volatility)
            
            stop_loss = await self._calculate_dynamic_stop_loss(signal)
            if not stop_loss:
                return None
                
            # Calculate take profit based on risk:reward ratio
            stop_distance = abs(signal['entry_price'] - stop_loss)
            take_distance = stop_distance * rr_ratio
            
            return signal['entry_price'] * (1.01 if signal['direction'] == 'buy' else 0.99)
            
        except Exception as e:
            self.logger.error(f"Take profit calculation error: {str(e)}")
            return None

    async def _calculate_position_size(self, signal: Dict) -> Optional[float]:
        """Calculate optimal position size based on multiple factors"""
        try:
            if not all(key in signal for key in ['entry_price', 'stop_loss', 'confidence']):
                return None
                
            # Get account info
            account_balance = self._get_account_balance()
            if not account_balance:
                return None
                
            # Calculate base position size
            risk_amount = account_balance * self.risk_params['max_position_risk']
            price_distance = abs(signal['entry_price'] - signal['stop_loss'])
            base_size = risk_amount / price_distance
            
            # Apply confidence multiplier
            confidence_multiplier = min(signal['confidence'], 0.95)
            
            # Apply volatility adjustment
            volatility = signal.get('volatility', 0.5)
            volatility_multiplier = 1 - (volatility * 0.5)  # Reduce size in high volatility
            
            # Apply market regime adjustment
            regime = signal.get('market_regime', {}).get('regime', 'UNKNOWN')
            regime_multiplier = {
                'STRONG_TREND': 1.0,
                'WEAK_TREND': 0.8,
                'RANGING': 0.7,
                'UNKNOWN': 0.5
            }.get(regime, 0.5)
            
            # Calculate final position size
            position_size = (base_size * 
                           confidence_multiplier * 
                           volatility_multiplier * 
                           regime_multiplier)
            
            # Apply risk level adjustment
            risk_level = self._get_current_risk_level()
            position_size *= self.risk_params['position_sizing'][risk_level]
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {str(e)}")
            return None

    async def _get_current_risk_level(self) -> str:
        """Determine current risk level based on market conditions and performance"""
        try:
            # Get recent performance metrics
            win_rate = self._calculate_recent_win_rate()
            drawdown = self._calculate_current_drawdown()
            volatility = self._get_market_volatility()
            
            # Conservative conditions
            if (drawdown > self.risk_params['max_drawdown'] * 0.7 or
                win_rate < self.performance_thresholds['min_win_rate'] or
                volatility > 0.7):
                return 'conservative'
                
            # Aggressive conditions
            if (drawdown < self.risk_params['max_drawdown'] * 0.3 and
                win_rate > self.performance_thresholds['min_win_rate'] * 1.2 and
                volatility < 0.4):
                return 'aggressive'
                
            # Default to moderate
            return 'moderate'
            
        except Exception as e:
            self.logger.error(f"Risk level determination error: {str(e)}")
            return 'conservative'  # Default to conservative on error

    async def _get_account_balance(self):
        # Implementation of _get_account_balance method
        pass

    async def _calculate_recent_win_rate(self):
        # Implementation of _calculate_recent_win_rate method
        pass

    async def _calculate_current_drawdown(self):
        # Implementation of _calculate_current_drawdown method
        pass

    async def _get_market_volatility(self):
        # Implementation of _get_market_volatility method
        pass

    async def _get_confirmation_indicators(self, signal: Dict) -> List[Dict]:
        """Enhanced confirmation indicators with confidence levels"""
        try:
            confirmations = []
            
            # 1. Trend Confirmation
            if await self._check_trend_alignment(signal):
                confirmations.append({
                    'type': 'trend_alignment',
                    'confidence': await self._calculate_trend_confidence(signal),
                    'timeframes': ['H1', 'H4', 'D1']
                })
            
            # 2. Volume Analysis
            volume_conf = await self._analyze_volume_confirmation(signal)
            if volume_conf['valid']:
                confirmations.append({
                    'type': 'volume_analysis',
                    'confidence': volume_conf['confidence'],
                    'criteria': volume_conf['criteria']
                })
            
            # 3. Price Action
            price_conf = await self._analyze_price_action(signal)
            if price_conf['valid']:
                confirmations.append({
                    'type': 'price_action',
                    'confidence': price_conf['confidence'],
                    'pattern': price_conf['pattern']
                })
            
            # 4. Support/Resistance
            sr_conf = await self._check_support_resistance(signal)
            if sr_conf['valid']:
                confirmations.append({
                    'type': 'support_resistance',
                    'confidence': sr_conf['confidence'],
                    'level_type': sr_conf['level_type']
                })
            
            # 5. Momentum Analysis
            momentum_conf = await self._analyze_momentum(signal)
            if momentum_conf['valid']:
                confirmations.append({
                    'type': 'momentum',
                    'confidence': momentum_conf['confidence'],
                    'indicators': momentum_conf['indicators']
                })
            
            return confirmations
            
        except Exception as e:
            self.logger.error(f"Confirmation indicators error: {str(e)}")
            return []

    async def _check_trend_alignment(self, signal: Dict) -> bool:
        """Check trend alignment across multiple timeframes"""
        try:
            if not isinstance(signal, dict):
                return False

            trend_analysis = signal.get('trend_analysis', {})
            if not isinstance(trend_analysis, dict):
                return False

            trend_directions = trend_analysis.get('directions', {})
            if not trend_directions:
                return False
            
            # Check if trends align in at least 2 timeframes
            aligned_count = sum(1 for direction in trend_directions.values() 
                              if direction == signal.get('direction'))
            return aligned_count >= 2
            
        except Exception as e:
            self.logger.error(f"Trend alignment check error: {str(e)}")
            return False

    async def _calculate_trend_confidence(self, signal: Dict) -> float:
        """Calculate trend confidence based on multiple factors"""
        try:
            trend_data = signal.get('trend_analysis', {})
            if not trend_data:
                return 0.0
            
            factors = [
                trend_data.get('strength', 0),
                trend_data.get('consistency', 0),
                trend_data.get('duration_factor', 0),
                trend_data.get('momentum_alignment', 0)
            ]
            
            return min(sum(factors) / len(factors), 0.99)
            
        except Exception as e:
            self.logger.error(f"Trend confidence calculation error: {str(e)}")
            return 0.0

    async def _analyze_volume_confirmation(self, signal: Dict) -> Dict:
        """Analyze volume patterns for confirmation"""
        try:
            volume_data = signal.get('volume_analysis', {})
            if not volume_data:
                return {'valid': False}
            
            criteria = []
            confidence = 0.0
            
            # Check volume increase
            if volume_data.get('increasing', False):
                criteria.append('volume_increasing')
                confidence += 0.3
            
            # Check volume trend alignment
            if volume_data.get('trend_aligned', False):
                criteria.append('trend_aligned')
                confidence += 0.3
            
            # Check abnormal volume
            if volume_data.get('abnormal', False):
                criteria.append('abnormal_volume')
                confidence += 0.2
            
            # Check OBV confirmation
            if volume_data.get('obv_confirmed', False):
                criteria.append('obv_confirmed')
                confidence += 0.2
            
            return {
                'valid': len(criteria) >= 2,
                'confidence': min(confidence, 0.99),
                'criteria': criteria
            }
            
        except Exception as e:
            self.logger.error(f"Volume confirmation analysis error: {str(e)}")
            return {'valid': False}

    async def _analyze_price_action(self, signal: Dict) -> Dict:
        """Analyze price action patterns"""
        try:
            price_data = signal.get('price_action', {})
            if not price_data:
                return {'valid': False}
            
            pattern = price_data.get('pattern')
            if not pattern:
                return {'valid': False}
            
            # Calculate pattern confidence
            confidence = min(
                price_data.get('pattern_strength', 0) * 0.4 +
                price_data.get('confirmation_strength', 0) * 0.3 +
                price_data.get('context_alignment', 0) * 0.3,
                0.99
            )
            
            return {
                'valid': confidence > 0.7,
                'confidence': confidence,
                'pattern': pattern
            }
            
        except Exception as e:
            self.logger.error(f"Price action analysis error: {str(e)}")
            return {'valid': False}

    async def _check_support_resistance(self, signal: Dict) -> Dict:
        """Check support and resistance levels"""
        try:
            sr_data = signal.get('support_resistance', {})
            if not sr_data:
                return {'valid': False}
            
            # Get nearest level
            nearest_level = sr_data.get('nearest_level', {})
            if not nearest_level:
                return {'valid': False}
            
            # Calculate confidence based on level strength
            confidence = min(
                nearest_level.get('strength', 0) * 0.4 +
                nearest_level.get('historical_accuracy', 0) * 0.3 +
                nearest_level.get('confluence', 0) * 0.3,
                0.99
            )
            
            return {
                'valid': confidence > 0.7,
                'confidence': confidence,
                'level_type': nearest_level.get('type')
            }
            
        except Exception as e:
            self.logger.error(f"Support/Resistance check error: {str(e)}")
            return {'valid': False}

    async def _analyze_momentum(self, signal: Dict) -> Dict:
        """Analyze momentum indicators"""
        try:
            momentum_data = signal.get('momentum_analysis', {})
            if not momentum_data:
                return {'valid': False}
            
            # Check multiple momentum indicators
            indicators = []
            total_confidence = 0.0
            
            # RSI
            if momentum_data.get('rsi_confirmed', False):
                indicators.append('RSI')
                total_confidence += 0.25
            
            # MACD
            if momentum_data.get('macd_confirmed', False):
                indicators.append('MACD')
                total_confidence += 0.25
            
            # Stochastic
            if momentum_data.get('stoch_confirmed', False):
                indicators.append('Stochastic')
                total_confidence += 0.25
            
            # ADX
            if momentum_data.get('adx_confirmed', False):
                indicators.append('ADX')
                total_confidence += 0.25
            
            return {
                'valid': len(indicators) >= 2,
                'confidence': min(total_confidence, 0.99),
                'indicators': indicators
            }
            
        except Exception as e:
            self.logger.error(f"Momentum analysis error: {str(e)}")
            return {'valid': False}

    async def _check_correlation_with_existing(self, signal: Dict) -> bool:
        """Check correlation with existing signals"""
        try:
            # Implement logic to check correlation with existing signals
            # This is a placeholder and should be replaced with actual implementation
            return True
            
        except Exception as e:
            self.logger.error(f"Correlation check error: {str(e)}")
            return False

    async def _check_risk_reward_ratio(self, signal: Dict) -> float:
        """Calculate and validate risk-reward ratio"""
        try:
            entry = signal.get('entry_price')
            stop = signal.get('stop_loss')
            target = signal.get('take_profit')
            
            if not all([entry, stop, target]):
                return 0.0
            
            risk = abs(entry - stop)
            reward = abs(target - entry)
            
            if risk == 0:
                return 0.0
                
            return reward / risk
            
        except Exception as e:
            self.logger.error(f"Risk-reward calculation error: {str(e)}")
            return 0.0

    async def _check_market_conditions(self, signal: Dict) -> bool:
        """Enhanced market conditions validation"""
        try:
            market_data = signal.get('market_conditions', {})
            
            # Check trading session
            if not market_data.get('session_active', True):
                return False
            
            # Check volatility conditions
            volatility = market_data.get('volatility', 0.5)
            if volatility > 0.8:  # Too volatile
                return False
            
            # Check spread
            spread = market_data.get('spread', 0)
            max_spread = market_data.get('max_allowed_spread', 0)
            if spread > max_spread:
                return False
            
            # Check market regime confidence
            regime = market_data.get('regime', {})
            if regime.get('confidence', 0) < 0.7:
                return False
            
            # Check liquidity
            if market_data.get('liquidity_score', 0) < 0.6:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Market conditions check error: {str(e)}")
            return False

    async def _get_market_conditions_summary(self, signal: Dict) -> Dict:
        """Get summary of current market conditions"""
        try:
            market_data = signal.get('market_conditions', {})
            
            return {
                'session_status': market_data.get('session_active', True),
                'volatility_level': market_data.get('volatility', 0.5),
                'spread_condition': market_data.get('spread', 0),
                'liquidity_score': market_data.get('liquidity_score', 0),
                'regime': market_data.get('regime', {}).get('regime', 'UNKNOWN'),
                'regime_confidence': market_data.get('regime', {}).get('confidence', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Market conditions summary error: {str(e)}")
            return {} 