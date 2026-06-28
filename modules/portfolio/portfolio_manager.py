from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from ..deployment.error_handler import ErrorHandler
from ..risk.risk_manager import RiskManager
from ..strategies.strategy_manager import StrategyManager

class PortfolioManager:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('portfolio_manager')
        
        # Initialize components
        self.risk_manager = RiskManager(config)
        self.strategy_manager = StrategyManager(config)
        
        # Initialize portfolio tracking
        self.portfolio = {
            'allocations': {},
            'positions': {},
            'performance': {},
            'reinvestment': {}
        }
        
        # Set parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize portfolio parameters"""
        self.params = {
            # Asset allocation
            'max_allocation_per_asset': self.config.get('MAX_ALLOCATION_PER_ASSET', 0.3),
            'min_allocation_per_asset': self.config.get('MIN_ALLOCATION_PER_ASSET', 0.05),
            
            # Risk parameters
            'base_risk_per_trade': self.config.get('BASE_RISK_PER_TRADE', 0.02),
            'max_portfolio_risk': self.config.get('MAX_PORTFOLIO_RISK', 0.05),
            
            # Scaling parameters
            'compound_rate': self.config.get('COMPOUND_RATE', 0.7),
            'scaling_threshold': self.config.get('SCALING_THRESHOLD', 1.2),
            
            # Reinvestment parameters
            'reinvestment_rate': self.config.get('REINVESTMENT_RATE', 0.3),
            'reserve_rate': self.config.get('RESERVE_RATE', 0.2)
        }
        
    async def update_portfolio_allocations(self) -> Dict:
        """Update portfolio allocations based on performance and risk"""
        try:
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                return {}
                
            current_equity = account_info.equity
            
            # Calculate optimal allocations
            allocations = await self._calculate_optimal_allocations(current_equity)
            
            # Update portfolio
            self.portfolio['allocations'] = allocations
            
            # Scale position sizes
            await self._scale_position_sizes(current_equity)
            
            # Process reinvestment
            await self._process_reinvestment(current_equity)
            
            return self.portfolio['allocations']
            
        except Exception as e:
            self.logger.error(f"Portfolio allocation update error: {str(e)}")
            return {}
            
    async def _calculate_optimal_allocations(self, equity: float) -> Dict:
        """Calculate optimal asset allocations"""
        try:
            # Get available symbols
            symbols = mt5.symbols_get()
            if symbols is None:
                return {}
                
            # Group symbols by asset class
            asset_classes = {
                'forex': [],
                'crypto': [],
                'stocks': [],
                'commodities': []
            }
            
            for symbol in symbols:
                if symbol.path.startswith('Forex'):
                    asset_classes['forex'].append(symbol.name)
                elif symbol.path.startswith('Crypto'):
                    asset_classes['crypto'].append(symbol.name)
                elif symbol.path.startswith('Stocks'):
                    asset_classes['stocks'].append(symbol.name)
                else:
                    asset_classes['commodities'].append(symbol.name)
                    
            # Calculate base allocations
            allocations = {
                'forex': 0.4,      # 40% to forex
                'crypto': 0.3,     # 30% to crypto
                'stocks': 0.2,     # 20% to stocks
                'commodities': 0.1  # 10% to commodities
            }
            
            # Adjust based on performance
            if self.portfolio['performance']:
                for asset_class in allocations:
                    performance = self.portfolio['performance'].get(asset_class, {})
                    if performance:
                        # Increase allocation for better performing assets
                        sharpe_ratio = performance.get('sharpe_ratio', 0)
                        allocations[asset_class] *= (1 + max(0, sharpe_ratio) * 0.1)
                        
            # Normalize allocations
            total = sum(allocations.values())
            allocations = {k: v/total for k, v in allocations.items()}
            
            # Calculate symbol-level allocations
            symbol_allocations = {}
            for asset_class, symbols_list in asset_classes.items():
                if not symbols_list:
                    continue
                    
                # Equal weight within asset class
                symbol_weight = allocations[asset_class] / len(symbols_list)
                for symbol in symbols_list:
                    symbol_allocations[symbol] = {
                        'allocation': min(
                            symbol_weight,
                            self.params['max_allocation_per_asset']
                        ),
                        'asset_class': asset_class,
                        'equity_allocation': symbol_weight * equity
                    }
                    
            return symbol_allocations
            
        except Exception as e:
            self.logger.error(f"Allocation calculation error: {str(e)}")
            return {}
            
    async def _scale_position_sizes(self, equity: float):
        """Scale position sizes based on account growth"""
        try:
            # Get baseline for scaling
            baseline = self.portfolio.get('baseline_equity', equity)
            
            # Calculate scaling factor
            growth_factor = equity / baseline
            if growth_factor >= self.params['scaling_threshold']:
                # Compound the position sizes
                compound_factor = growth_factor ** self.params['compound_rate']
                
                # Update risk per trade
                new_risk_per_trade = min(
                    self.params['base_risk_per_trade'] * compound_factor,
                    self.params['max_portfolio_risk']
                )
                
                # Update risk manager
                await self.risk_manager.update_risk_parameters({
                    'risk_per_trade': new_risk_per_trade
                })
                
                # Update baseline
                self.portfolio['baseline_equity'] = equity
                
        except Exception as e:
            self.logger.error(f"Position scaling error: {str(e)}")
            
    async def _process_reinvestment(self, equity: float):
        """Process profit reinvestment"""
        try:
            # Calculate profits since last reinvestment
            last_equity = self.portfolio.get('last_reinvestment_equity', equity)
            profit = max(0, equity - last_equity)
            
            if profit > 0:
                # Calculate reinvestment amounts
                reinvestment = profit * self.params['reinvestment_rate']
                reserve = profit * self.params['reserve_rate']
                
                # Update reinvestment tracking
                self.portfolio['reinvestment'] = {
                    'timestamp': datetime.now(),
                    'profit': profit,
                    'reinvested': reinvestment,
                    'reserve': reserve
                }
                
                # Update last reinvestment equity
                self.portfolio['last_reinvestment_equity'] = equity
                
        except Exception as e:
            self.logger.error(f"Reinvestment processing error: {str(e)}")
            
    async def get_position_size(
        self,
        symbol: str,
        signal: Dict
    ) -> float:
        """Get scaled position size for a symbol"""
        try:
            # Get symbol allocation
            allocation = self.portfolio['allocations'].get(symbol, {})
            if not allocation:
                return 0
                
            # Get account equity
            account_info = mt5.account_info()
            if account_info is None:
                return 0
                
            # Calculate base position size
            equity_allocation = allocation['equity_allocation']
            
            # Get risk-adjusted size
            size = await self.risk_manager.calculate_position_size(
                signal,
                equity_allocation,
                {'symbol': symbol}
            )
            
            return size
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {str(e)}")
            return 0
            
    async def update_performance_metrics(self):
        """Update portfolio performance metrics"""
        try:
            # Get positions by asset class
            positions = mt5.positions_get()
            if positions is None:
                return
                
            metrics = {}
            for position in positions:
                asset_class = self.portfolio['allocations'].get(
                    position.symbol, {}
                ).get('asset_class')
                
                if not asset_class:
                    continue
                    
                if asset_class not in metrics:
                    metrics[asset_class] = {
                        'profit': 0,
                        'volume': 0,
                        'margin': 0
                    }
                    
                metrics[asset_class]['profit'] += position.profit
                metrics[asset_class]['volume'] += position.volume
                metrics[asset_class]['margin'] += position.margin
                
            # Calculate performance metrics
            for asset_class in metrics:
                profit_series = pd.Series([
                    p.profit for p in positions
                    if self.portfolio['allocations'].get(p.symbol, {}).get('asset_class') == asset_class
                ])
                
                metrics[asset_class]['sharpe_ratio'] = (
                    profit_series.mean() / profit_series.std()
                    if len(profit_series) > 1 and profit_series.std() != 0
                    else 0
                )
                
            self.portfolio['performance'] = metrics
            
        except Exception as e:
            self.logger.error(f"Performance update error: {str(e)}")
            
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        try:
            return {
                'allocations': self.portfolio['allocations'],
                'performance': self.portfolio['performance'],
                'reinvestment': self.portfolio['reinvestment']
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio summary error: {str(e)}")
            return {}
