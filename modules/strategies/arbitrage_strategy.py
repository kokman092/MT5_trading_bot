from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from dataclasses import dataclass
from ..analytics.market_analyzer import MarketAnalyzer
from ..execution.smart_executor import SmartExecutor

@dataclass
class ArbitrageOpportunity:
    type: str  # Arbitrage type
    pairs: List[str]  # Currency pairs involved
    spread: float  # Price difference
    volume: float  # Available volume
    cost: float  # Execution cost
    profit: float  # Expected profit

class ArbitrageStrategy:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('arbitrage_strategy')
        self.market_analyzer = MarketAnalyzer(config)
        self.executor = SmartExecutor(config)
        
        # Initialize parameters
        self._init_strategy_parameters()
        
    def _init_strategy_parameters(self):
        """Initialize strategy parameters"""
        # Triangular arbitrage parameters
        self.triangular_params = {
            'min_profit': 0.001,  # 0.1% minimum profit
            'max_volume': 100000,  # Maximum position size
            'execution_timeout': 2,  # 2 seconds timeout
            'price_buffer': 0.0001  # Price buffer for slippage
        }
        
        # Statistical arbitrage parameters
        self.statistical_params = {
            'lookback_period': 100,  # Price history length
            'zscore_threshold': 2.0,  # Entry threshold
            'mean_reversion_time': 24,  # Expected hours to converge
            'stop_loss_std': 3.0  # Stop loss in standard deviations
        }
        
        # Cross-exchange parameters
        self.cross_exchange_params = {
            'min_spread': 0.002,  # 0.2% minimum spread
            'max_latency': 100,  # Maximum latency in ms
            'volume_ratio': 0.8,  # Minimum volume ratio
            'fee_threshold': 0.001  # Maximum acceptable fee
        }
        
    async def find_triangular_arbitrage(
        self,
        symbols: List[str]
    ) -> List[ArbitrageOpportunity]:
        """Find triangular arbitrage opportunities"""
        try:
            opportunities = []
            
            # Get currency pairs
            pairs = await self._get_currency_pairs(symbols)
            
            # Find triangular combinations
            triangles = await self._find_triangular_combinations(pairs)
            
            # Analyze each triangle
            for triangle in triangles:
                # Get real-time prices
                prices = await self._get_triangle_prices(triangle)
                
                # Calculate arbitrage spread
                spread = await self._calculate_triangular_spread(
                    triangle,
                    prices
                )
                
                # Check profitability
                if spread > self.triangular_params['min_profit']:
                    # Calculate available volume
                    volume = await self._calculate_triangle_volume(
                        triangle,
                        prices
                    )
                    
                    # Calculate execution cost
                    cost = await self._calculate_execution_cost(
                        triangle,
                        volume
                    )
                    
                    # Calculate expected profit
                    profit = spread * volume - cost
                    
                    if profit > 0:
                        opportunities.append(ArbitrageOpportunity(
                            type='triangular',
                            pairs=triangle,
                            spread=spread,
                            volume=volume,
                            cost=cost,
                            profit=profit
                        ))
                        
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Triangular arbitrage search error: {str(e)}")
            return []
            
    async def find_statistical_arbitrage(
        self,
        symbols: List[str]
    ) -> List[ArbitrageOpportunity]:
        """Find statistical arbitrage opportunities"""
        try:
            opportunities = []
            
            # Get correlated pairs
            pairs = await self._find_correlated_pairs(symbols)
            
            # Analyze each pair
            for pair in pairs:
                # Get price history
                history = await self._get_price_history(
                    pair,
                    self.statistical_params['lookback_period']
                )
                
                # Calculate spread
                spread = await self._calculate_pair_spread(history)
                
                # Check zscore
                zscore = await self._calculate_zscore(spread)
                
                if abs(zscore) > self.statistical_params['zscore_threshold']:
                    # Calculate position sizes
                    volume = await self._calculate_stat_arb_volume(pair)
                    
                    # Calculate execution cost
                    cost = await self._calculate_execution_cost(
                        pair,
                        volume
                    )
                    
                    # Calculate expected profit
                    profit = await self._calculate_stat_arb_profit(
                        spread,
                        zscore,
                        volume,
                        cost
                    )
                    
                    if profit > 0:
                        opportunities.append(ArbitrageOpportunity(
                            type='statistical',
                            pairs=[pair[0], pair[1]],
                            spread=spread,
                            volume=volume,
                            cost=cost,
                            profit=profit
                        ))
                        
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Statistical arbitrage search error: {str(e)}")
            return []
            
    async def find_cross_exchange_arbitrage(
        self,
        symbols: List[str]
    ) -> List[ArbitrageOpportunity]:
        """Find cross-exchange arbitrage opportunities"""
        try:
            opportunities = []
            
            # Get exchange prices
            prices = await self._get_exchange_prices(symbols)
            
            # Find price discrepancies
            discrepancies = await self._find_price_discrepancies(prices)
            
            # Analyze each opportunity
            for disc in discrepancies:
                # Check spread
                if disc['spread'] < self.cross_exchange_params['min_spread']:
                    continue
                    
                # Check latency
                if not await self._check_exchange_latency(
                    disc['exchanges']
                ):
                    continue
                    
                # Calculate volume
                volume = await self._calculate_cross_volume(disc)
                
                # Calculate fees
                fees = await self._calculate_exchange_fees(
                    disc['exchanges'],
                    volume
                )
                
                if fees > self.cross_exchange_params['fee_threshold']:
                    continue
                    
                # Calculate profit
                profit = disc['spread'] * volume - fees
                
                if profit > 0:
                    opportunities.append(ArbitrageOpportunity(
                        type='cross_exchange',
                        pairs=[disc['symbol']],
                        spread=disc['spread'],
                        volume=volume,
                        cost=fees,
                        profit=profit
                    ))
                    
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Cross-exchange arbitrage search error: {str(e)}")
            return []
            
    async def execute_arbitrage(
        self,
        opportunity: ArbitrageOpportunity
    ) -> Dict:
        """Execute arbitrage opportunity"""
        try:
            result = {}
            
            if opportunity.type == 'triangular':
                result = await self._execute_triangular_arbitrage(opportunity)
            elif opportunity.type == 'statistical':
                result = await self._execute_statistical_arbitrage(opportunity)
            elif opportunity.type == 'cross_exchange':
                result = await self._execute_cross_exchange_arbitrage(opportunity)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Arbitrage execution error: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    async def _calculate_triangular_spread(
        self,
        triangle: List[str],
        prices: Dict
    ) -> float:
        """Calculate triangular arbitrage spread"""
        try:
            # Extract prices
            ab_price = prices[triangle[0]]
            bc_price = prices[triangle[1]]
            ca_price = prices[triangle[2]]
            
            # Calculate cross-rate
            cross_rate = ab_price * bc_price / ca_price
            
            # Calculate spread
            spread = abs(cross_rate - 1.0)
            
            return spread
            
        except Exception as e:
            self.logger.error(f"Triangular spread calculation error: {str(e)}")
            return 0.0
            
    async def _calculate_stat_arb_profit(
        self,
        spread: float,
        zscore: float,
        volume: float,
        cost: float
    ) -> float:
        """Calculate statistical arbitrage profit"""
        try:
            # Calculate mean reversion probability
            prob = await self._calculate_reversion_probability(zscore)
            
            # Calculate expected convergence
            convergence = spread * prob
            
            # Calculate expected profit
            profit = convergence * volume - cost
            
            return profit
            
        except Exception as e:
            self.logger.error(f"Statistical arbitrage profit calculation error: {str(e)}")
            return 0.0
