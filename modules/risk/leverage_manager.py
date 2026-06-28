from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from dataclasses import dataclass
from ..analytics.market_analyzer import MarketAnalyzer
from ..deployment.error_handler import ErrorHandler
from ..risk.dynamic_risk import DynamicRiskManager

@dataclass
class LeverageProfile:
    base_leverage: float  # Base leverage ratio
    adjusted_leverage: float  # Market-adjusted leverage
    max_leverage: float  # Maximum allowed leverage
    risk_factor: float  # Current risk factor
    margin_level: float  # Current margin level

@dataclass
class MarketCondition:
    volatility_level: float  # Current volatility level
    trend_strength: float  # Trend strength indicator
    liquidity_score: float  # Market liquidity score
    risk_score: float  # Overall risk score
    recommendation: str  # Leverage recommendation

@dataclass
class HedgePosition:
    instrument: str  # Hedging instrument
    size: float  # Position size
    cost: float  # Hedging cost
    coverage: float  # Coverage ratio
    delta_neutral: bool  # Delta neutrality status

class LeverageManager:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('leverage_manager')
        self.market_analyzer = MarketAnalyzer(config)
        self.risk_manager = DynamicRiskManager(config)
        
        # Initialize parameters
        self._init_leverage_parameters()
        self._init_monitoring_system()
        
    def _init_leverage_parameters(self):
        """Initialize leverage management parameters"""
        # Base leverage parameters
        self.leverage_params = {
            'equity_tiers': {
                10000: 2.0,    # Up to $10k: 2x max
                50000: 5.0,    # Up to $50k: 5x max
                100000: 10.0,  # Up to $100k: 10x max
                500000: 15.0,  # Up to $500k: 15x max
                1000000: 20.0  # Up to $1M: 20x max
            },
            'min_leverage': 1.0,
            'max_leverage': 20.0,
            'leverage_step': 0.5,
            'margin_call_level': 0.8,  # 80% margin level warning
            'stop_out_level': 0.5  # 50% margin level stop
        }
        
        # Market condition parameters
        self.market_params = {
            'volatility_impact': {
                'low': 1.2,     # Increase leverage by 20%
                'medium': 1.0,  # Normal leverage
                'high': 0.7     # Reduce leverage by 30%
            },
            'trend_impact': {
                'strong': 1.1,  # Increase leverage by 10%
                'normal': 1.0,  # Normal leverage
                'weak': 0.9     # Reduce leverage by 10%
            },
            'liquidity_impact': {
                'high': 1.1,    # Increase leverage by 10%
                'medium': 1.0,  # Normal leverage
                'low': 0.8      # Reduce leverage by 20%
            }
        }
        
        # Hedging parameters
        self.hedge_params = {
            'option_coverage': 0.5,  # 50% option coverage
            'max_hedge_cost': 0.02,  # Max 2% hedging cost
            'delta_target': 0.0,     # Target delta-neutral
            'hedge_instruments': {
                'forex': ['options', 'futures'],
                'crypto': ['options', 'perpetuals'],
                'stocks': ['options', 'inverse_etfs']
            },
            'rebalance_threshold': 0.1  # 10% hedge rebalance trigger
        }
        
    def _init_monitoring_system(self):
        """Initialize monitoring system"""
        try:
            # Initialize tracking dictionaries
            self.leverage_tracking = {
                'current_leverage': 1.0,
                'peak_leverage': 1.0,
                'margin_usage': 0.0,
                'hedge_positions': {}
            }
            
            # Initialize market condition tracking
            self.market_tracking = {
                'volatility_history': [],
                'trend_history': [],
                'liquidity_history': []
            }
            
        except Exception as e:
            self.logger.error(f"Monitoring system initialization error: {str(e)}")
            
    async def get_leverage_profile(
        self,
        account_info: Dict,
        market_condition: Dict
    ) -> LeverageProfile:
        """Calculate optimal leverage profile"""
        try:
            # Get base leverage
            base_leverage = await self._calculate_base_leverage(
                account_info['equity']
            )
            
            # Adjust for market conditions
            adjusted_leverage = await self._adjust_leverage(
                base_leverage,
                market_condition
            )
            
            # Calculate risk factor
            risk_factor = await self._calculate_risk_factor(
                adjusted_leverage,
                account_info
            )
            
            # Get margin level
            margin_level = await self._calculate_margin_level(account_info)
            
            return LeverageProfile(
                base_leverage=base_leverage,
                adjusted_leverage=adjusted_leverage,
                max_leverage=self.leverage_params['max_leverage'],
                risk_factor=risk_factor,
                margin_level=margin_level
            )
            
        except Exception as e:
            self.logger.error(f"Leverage profile calculation error: {str(e)}")
            return None
            
    async def analyze_market_condition(
        self,
        symbol: str,
        timeframe: int
    ) -> MarketCondition:
        """Analyze current market conditions"""
        try:
            # Calculate volatility level
            volatility = await self._calculate_volatility_level(
                symbol,
                timeframe
            )
            
            # Calculate trend strength
            trend = await self._calculate_trend_strength(
                symbol,
                timeframe
            )
            
            # Calculate liquidity score
            liquidity = await self._calculate_liquidity_score(symbol)
            
            # Calculate overall risk score
            risk_score = await self._calculate_risk_score(
                volatility,
                trend,
                liquidity
            )
            
            return MarketCondition(
                volatility_level=volatility,
                trend_strength=trend,
                liquidity_score=liquidity,
                risk_score=risk_score,
                recommendation=await self._generate_leverage_recommendation(
                    risk_score
                )
            )
            
        except Exception as e:
            self.logger.error(f"Market condition analysis error: {str(e)}")
            return None
            
    async def get_hedge_position(
        self,
        symbol: str,
        position_size: float,
        leverage: float
    ) -> HedgePosition:
        """Calculate optimal hedge position"""
        try:
            # Determine hedge instrument
            instrument = await self._select_hedge_instrument(symbol)
            
            # Calculate hedge size
            size = await self._calculate_hedge_size(
                position_size,
                leverage
            )
            
            # Calculate hedging cost
            cost = await self._calculate_hedge_cost(
                instrument,
                size
            )
            
            # Calculate coverage ratio
            coverage = await self._calculate_coverage_ratio(
                size,
                position_size
            )
            
            return HedgePosition(
                instrument=instrument,
                size=size,
                cost=cost,
                coverage=coverage,
                delta_neutral=await self._check_delta_neutral(
                    position_size,
                    size
                )
            )
            
        except Exception as e:
            self.logger.error(f"Hedge position calculation error: {str(e)}")
            return None
            
    async def _calculate_base_leverage(self, equity: float) -> float:
        """Calculate base leverage based on equity"""
        try:
            # Find appropriate leverage tier
            base_leverage = self.leverage_params['min_leverage']
            
            for level, leverage in self.leverage_params['equity_tiers'].items():
                if equity <= level:
                    base_leverage = leverage
                    break
                    
            return min(
                base_leverage,
                self.leverage_params['max_leverage']
            )
            
        except Exception as e:
            self.logger.error(f"Base leverage calculation error: {str(e)}")
            return self.leverage_params['min_leverage']
            
    async def _adjust_leverage(
        self,
        base_leverage: float,
        market_condition: Dict
    ) -> float:
        """Adjust leverage based on market conditions"""
        try:
            # Get impact factors
            vol_impact = self.market_params['volatility_impact'][
                market_condition['volatility']
            ]
            trend_impact = self.market_params['trend_impact'][
                market_condition['trend']
            ]
            liq_impact = self.market_params['liquidity_impact'][
                market_condition['liquidity']
            ]
            
            # Calculate adjusted leverage
            adjusted = base_leverage * vol_impact * trend_impact * liq_impact
            
            # Round to nearest step
            step = self.leverage_params['leverage_step']
            adjusted = round(adjusted / step) * step
            
            return min(
                max(
                    adjusted,
                    self.leverage_params['min_leverage']
                ),
                self.leverage_params['max_leverage']
            )
            
        except Exception as e:
            self.logger.error(f"Leverage adjustment error: {str(e)}")
            return base_leverage
            
    async def _calculate_hedge_size(
        self,
        position_size: float,
        leverage: float
    ) -> float:
        """Calculate required hedge position size"""
        try:
            # Calculate notional exposure
            notional = position_size * leverage
            
            # Apply coverage ratio
            hedge_size = notional * self.hedge_params['option_coverage']
            
            return hedge_size
            
        except Exception as e:
            self.logger.error(f"Hedge size calculation error: {str(e)}")
            return 0.0
            
    async def _calculate_hedge_cost(
        self,
        instrument: str,
        size: float
    ) -> float:
        """Calculate hedging cost"""
        try:
            # Get instrument pricing
            pricing = await self._get_hedge_pricing(instrument)
            
            # Calculate total cost
            cost = size * pricing['premium']
            
            return min(
                cost,
                self.hedge_params['max_hedge_cost']
            )
            
        except Exception as e:
            self.logger.error(f"Hedge cost calculation error: {str(e)}")
            return 0.0
            
    async def _calculate_risk_score(
        self,
        volatility: float,
        trend: float,
        liquidity: float
    ) -> float:
        """Calculate overall market risk score"""
        try:
            # Weight factors
            weights = {
                'volatility': 0.4,
                'trend': 0.3,
                'liquidity': 0.3
            }
            
            # Calculate weighted score
            risk_score = (
                volatility * weights['volatility'] +
                trend * weights['trend'] +
                liquidity * weights['liquidity']
            )
            
            return risk_score
            
        except Exception as e:
            self.logger.error(f"Risk score calculation error: {str(e)}")
            return 0.5  # Neutral risk score
            
    async def _select_hedge_instrument(self, symbol: str) -> str:
        """Select appropriate hedging instrument"""
        try:
            # Determine asset class
            asset_class = self._get_asset_class(symbol)
            
            # Get available instruments
            instruments = self.hedge_params['hedge_instruments'][asset_class]
            
            # Select best instrument based on cost and liquidity
            best_instrument = await self._find_best_hedge(
                instruments,
                symbol
            )
            
            return best_instrument
            
        except Exception as e:
            self.logger.error(f"Hedge instrument selection error: {str(e)}")
            return None
