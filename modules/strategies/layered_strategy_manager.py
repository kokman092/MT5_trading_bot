from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from ..risk.risk_manager import RiskManager
from ..portfolio.portfolio_manager import PortfolioManager
from ..analytics.market_analyzer import MarketAnalyzer
from ..deployment.error_handler import ErrorHandler

class LayeredStrategyManager:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('layered_strategy_manager')
        self.risk_manager = RiskManager(config)
        self.portfolio_manager = PortfolioManager(config)
        self.market_analyzer = MarketAnalyzer(config)
        
        # Initialize strategy layers
        self._init_strategy_layers()
        
    def _init_strategy_layers(self):
        """Initialize strategy layers and allocations"""
        self.strategy_layers = {
            'primary': {
                'type': 'trend_following',
                'allocation': 0.5,  # 50% of capital
                'active': True,
                'performance': {'win_rate': 0, 'profit_factor': 0}
            },
            'hedge': {
                'type': 'correlation_hedge',
                'allocation': 0.3,  # 30% of capital
                'active': True,
                'performance': {'win_rate': 0, 'profit_factor': 0}
            },
            'rebalance': {
                'type': 'dynamic_rebalancing',
                'allocation': 0.2,  # 20% of capital
                'active': True,
                'performance': {'win_rate': 0, 'profit_factor': 0}
            }
        }
        
        # Initialize scaling parameters
        self.scaling_params = {
            'entry_levels': 5,  # Number of entry levels
            'exit_levels': 3,   # Number of exit levels
            'level_spacing': 0.2,  # Percentage spacing between levels
            'volume_distribution': 'pyramid'  # pyramid, equal, or custom
        }
        
    async def execute_layered_strategy(self, market_data: Dict) -> List[Dict]:
        """Execute all strategy layers"""
        try:
            signals = []
            
            # 1. Execute primary strategy
            if self.strategy_layers['primary']['active']:
                primary_signals = await self._execute_primary_strategy(market_data)
                signals.extend(primary_signals)
                
            # 2. Execute hedge strategy
            if self.strategy_layers['hedge']['active']:
                hedge_signals = await self._execute_hedge_strategy(market_data, primary_signals)
                signals.extend(hedge_signals)
                
            # 3. Execute rebalancing
            if self.strategy_layers['rebalance']['active']:
                rebalance_signals = await self._execute_rebalancing_strategy()
                signals.extend(rebalance_signals)
                
            # 4. Scale signals
            scaled_signals = await self._scale_signals(signals)
            
            return scaled_signals
            
        except Exception as e:
            self.logger.error(f"Layered strategy execution error: {str(e)}")
            return []
            
    async def _execute_primary_strategy(self, market_data: Dict) -> List[Dict]:
        """Execute primary trend-following strategy"""
        try:
            signals = []
            
            for symbol, data in market_data.items():
                # Analyze market regime
                regime = await self.market_analyzer.detect_market_regime(symbol)
                
                # Check for trend conditions
                if regime['regime'] in ['trending_up', 'trending_down']:
                    # Calculate position size
                    size = await self._calculate_position_size(
                        symbol,
                        self.strategy_layers['primary']['allocation']
                    )
                    
                    # Generate signal
                    signal = {
                        'symbol': symbol,
                        'type': 'primary',
                        'direction': 'buy' if regime['regime'] == 'trending_up' else 'sell',
                        'size': size,
                        'confidence': regime['characteristics']['trend']['strength']
                    }
                    
                    signals.append(signal)
                    
            return signals
            
        except Exception as e:
            self.logger.error(f"Primary strategy execution error: {str(e)}")
            return []
            
    async def _execute_hedge_strategy(
        self,
        market_data: Dict,
        primary_signals: List[Dict]
    ) -> List[Dict]:
        """Execute hedging strategy"""
        try:
            hedge_signals = []
            
            # Get primary positions
            primary_positions = [signal['symbol'] for signal in primary_signals]
            
            # Perform cluster analysis
            clusters = await self.market_analyzer.perform_cluster_analysis(
                list(market_data.keys())
            )
            
            # Find hedge opportunities
            for signal in primary_signals:
                # Find correlated symbols
                correlated_symbols = self._find_correlated_symbols(
                    signal['symbol'],
                    clusters
                )
                
                # Calculate hedge ratio
                hedge_size = signal['size'] * self.strategy_layers['hedge']['allocation']
                
                # Generate hedge signals
                for hedge_symbol in correlated_symbols:
                    hedge_signals.append({
                        'symbol': hedge_symbol,
                        'type': 'hedge',
                        'direction': 'sell' if signal['direction'] == 'buy' else 'buy',
                        'size': hedge_size / len(correlated_symbols),
                        'correlation': clusters['metrics'].get(hedge_symbol, {}).get('correlation', 0)
                    })
                    
            return hedge_signals
            
        except Exception as e:
            self.logger.error(f"Hedge strategy execution error: {str(e)}")
            return []
            
    async def _execute_rebalancing_strategy(self) -> List[Dict]:
        """Execute portfolio rebalancing strategy"""
        try:
            rebalance_signals = []
            
            # Get current portfolio state
            portfolio = await self.portfolio_manager.get_portfolio_state()
            
            # Calculate target allocations
            targets = await self._calculate_target_allocations()
            
            # Generate rebalancing signals
            for symbol, current in portfolio.items():
                target = targets.get(symbol, 0)
                if abs(current - target) > self.config.get('rebalance_threshold', 0.1):
                    size = abs(target - current)
                    direction = 'buy' if target > current else 'sell'
                    
                    rebalance_signals.append({
                        'symbol': symbol,
                        'type': 'rebalance',
                        'direction': direction,
                        'size': size,
                        'target_allocation': target
                    })
                    
            return rebalance_signals
            
        except Exception as e:
            self.logger.error(f"Rebalancing strategy execution error: {str(e)}")
            return []
            
    async def _scale_signals(self, signals: List[Dict]) -> List[Dict]:
        """Scale signals into multiple entry/exit levels"""
        try:
            scaled_signals = []
            
            for signal in signals:
                # Get scaling parameters
                entry_levels = self.scaling_params['entry_levels']
                level_spacing = self.scaling_params['level_spacing']
                
                # Calculate level prices and sizes
                base_price = await self._get_current_price(signal['symbol'])
                level_prices = await self._calculate_level_prices(
                    base_price,
                    signal['direction'],
                    entry_levels,
                    level_spacing
                )
                
                level_sizes = await self._calculate_level_sizes(
                    signal['size'],
                    entry_levels
                )
                
                # Generate scaled signals
                for i, (price, size) in enumerate(zip(level_prices, level_sizes)):
                    scaled_signals.append({
                        **signal,
                        'original_size': signal['size'],
                        'size': size,
                        'price': price,
                        'level': i + 1,
                        'total_levels': entry_levels
                    })
                    
            return scaled_signals
            
        except Exception as e:
            self.logger.error(f"Signal scaling error: {str(e)}")
            return signals  # Return original signals if scaling fails
            
    async def _calculate_level_prices(
        self,
        base_price: float,
        direction: str,
        levels: int,
        spacing: float
    ) -> List[float]:
        """Calculate prices for multiple entry levels"""
        try:
            prices = []
            for i in range(levels):
                if direction == 'buy':
                    level_price = base_price * (1 - spacing * i)
                else:
                    level_price = base_price * (1 + spacing * i)
                prices.append(level_price)
            return prices
            
        except Exception as e:
            self.logger.error(f"Level price calculation error: {str(e)}")
            return [base_price]
            
    async def _calculate_level_sizes(
        self,
        total_size: float,
        levels: int
    ) -> List[float]:
        """Calculate position sizes for multiple levels"""
        try:
            if self.scaling_params['volume_distribution'] == 'equal':
                return [total_size / levels] * levels
                
            elif self.scaling_params['volume_distribution'] == 'pyramid':
                # Pyramid distribution (larger sizes at better prices)
                weights = np.array(range(levels, 0, -1))
                weights = weights / weights.sum()
                return (total_size * weights).tolist()
                
            else:  # custom distribution
                weights = self.config.get('custom_level_weights', [1/levels] * levels)
                return (total_size * np.array(weights)).tolist()
                
        except Exception as e:
            self.logger.error(f"Level size calculation error: {str(e)}")
            return [total_size]
            
    async def _calculate_position_size(
        self,
        symbol: str,
        allocation: float
    ) -> float:
        """Calculate position size based on allocation"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return 0.0
                
            # Calculate size based on equity and allocation
            equity = account_info.equity
            position_value = equity * allocation
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.0
                
            # Calculate volume in lots
            volume = position_value / (symbol_info.trade_contract_size * symbol_info.ask)
            
            # Round to symbol's volume step
            volume_step = symbol_info.volume_step
            volume = round(volume / volume_step) * volume_step
            
            return volume
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {str(e)}")
            return 0.0
            
    async def _calculate_target_allocations(self) -> Dict[str, float]:
        """Calculate target allocations based on performance"""
        try:
            targets = {}
            
            # Get performance metrics
            performance = await self.portfolio_manager.get_performance_metrics()
            
            # Calculate allocation scores
            total_score = 0
            scores = {}
            
            for symbol, metrics in performance.items():
                # Calculate score based on Sharpe ratio and win rate
                score = metrics.get('sharpe_ratio', 0) * metrics.get('win_rate', 0)
                scores[symbol] = max(score, 0)  # Ensure non-negative
                total_score += score
                
            # Calculate target allocations
            if total_score > 0:
                for symbol, score in scores.items():
                    targets[symbol] = score / total_score
                    
            return targets
            
        except Exception as e:
            self.logger.error(f"Target allocation calculation error: {str(e)}")
            return {}
            
    def _find_correlated_symbols(
        self,
        symbol: str,
        clusters: Dict
    ) -> List[str]:
        """Find correlated symbols for hedging"""
        try:
            correlated_symbols = []
            
            # Find cluster containing the symbol
            for cluster in clusters.get('clusters', {}).values():
                if symbol in cluster:
                    # Get other symbols in same cluster
                    correlated_symbols = [s for s in cluster if s != symbol]
                    break
                    
            return correlated_symbols
            
        except Exception as e:
            self.logger.error(f"Correlated symbol search error: {str(e)}")
            return []
            
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            symbol_info = mt5.symbol_info_tick(symbol)
            if symbol_info is None:
                return 0.0
            return (symbol_info.bid + symbol_info.ask) / 2
            
        except Exception as e:
            self.logger.error(f"Price fetch error: {str(e)}")
            return 0.0
