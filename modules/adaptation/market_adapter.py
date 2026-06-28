from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from dataclasses import dataclass
import aiohttp
import asyncio
from scipy.stats import pearsonr
from ..analytics.market_analyzer import MarketAnalyzer
from ..deployment.error_handler import ErrorHandler

@dataclass
class CorrelationData:
    primary: str  # Primary asset
    secondary: str  # Secondary asset
    correlation: float  # Correlation coefficient
    significance: float  # Statistical significance
    trend_alignment: float  # Trend alignment score
    volatility_ratio: float  # Relative volatility

@dataclass
class NewsEvent:
    timestamp: datetime  # Event time
    currency: str  # Affected currency
    event: str  # Event description
    impact: str  # Impact level (high/medium/low)
    forecast: float  # Expected value
    previous: float  # Previous value
    volatility_expected: float  # Expected volatility

@dataclass
class MarketState:
    volatility_level: str  # Current volatility state
    trend_strength: float  # Current trend strength
    market_regime: str  # Current market regime
    risk_level: float  # Current risk level
    stop_adjustment: float  # Stop-loss adjustment factor

class MarketAdapter:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('market_adapter')
        self.market_analyzer = MarketAnalyzer(config)
        
        # Initialize parameters and systems
        self._init_adaptation_parameters()
        self._init_monitoring_system()
        
    def _init_adaptation_parameters(self):
        """Initialize adaptation parameters"""
        # Correlation parameters
        self.correlation_params = {
            'window_size': 100,  # Correlation window
            'min_correlation': 0.7,  # Minimum significant correlation
            'update_frequency': 60,  # Update frequency in seconds
            'lookback_periods': {
                'short': 20,
                'medium': 50,
                'long': 100
            }
        }
        
        # News parameters
        self.news_params = {
            'impact_levels': {
                'high': 3,
                'medium': 2,
                'low': 1
            },
            'volatility_multipliers': {
                'high': 2.0,
                'medium': 1.5,
                'low': 1.2
            },
            'buffer_minutes': {
                'before': 30,
                'after': 60
            },
            'max_events': 100  # Maximum events to track
        }
        
        # Stop-loss parameters
        self.stop_params = {
            'volatility_zones': {
                'low': {'multiplier': 1.0, 'atr_factor': 1.5},
                'medium': {'multiplier': 1.5, 'atr_factor': 2.0},
                'high': {'multiplier': 2.0, 'atr_factor': 2.5}
            },
            'trend_factors': {
                'strong': 1.2,
                'normal': 1.0,
                'weak': 0.8
            },
            'regime_adjustments': {
                'trending': 1.2,
                'ranging': 0.8,
                'volatile': 1.5
            }
        }
        
    def _init_monitoring_system(self):
        """Initialize monitoring system"""
        try:
            # Initialize tracking dictionaries
            self.correlation_tracking = {
                'pairs': {},
                'history': {},
                'signals': {}
            }
            
            self.news_tracking = {
                'events': [],
                'impacts': {},
                'volatility': {}
            }
            
            self.market_tracking = {
                'states': {},
                'regimes': {},
                'stops': {}
            }
            
        except Exception as e:
            self.logger.error(f"Monitoring system initialization error: {str(e)}")
            
    async def track_correlations(
        self,
        primary: str,
        secondaries: List[str],
        timeframe: str
    ) -> List[CorrelationData]:
        """Track real-time correlations between assets"""
        try:
            correlations = []
            
            # Get primary asset data
            primary_data = await self._get_market_data(
                primary,
                timeframe,
                self.correlation_params['window_size']
            )
            
            # Calculate correlations with each secondary asset
            for secondary in secondaries:
                # Get secondary asset data
                secondary_data = await self._get_market_data(
                    secondary,
                    timeframe,
                    self.correlation_params['window_size']
                )
                
                # Calculate correlation
                corr, sig = await self._calculate_correlation(
                    primary_data,
                    secondary_data
                )
                
                # Calculate additional metrics
                trend_align = await self._calculate_trend_alignment(
                    primary_data,
                    secondary_data
                )
                
                vol_ratio = await self._calculate_volatility_ratio(
                    primary_data,
                    secondary_data
                )
                
                correlations.append(CorrelationData(
                    primary=primary,
                    secondary=secondary,
                    correlation=corr,
                    significance=sig,
                    trend_alignment=trend_align,
                    volatility_ratio=vol_ratio
                ))
                
            return correlations
            
        except Exception as e:
            self.logger.error(f"Correlation tracking error: {str(e)}")
            return []
            
    async def monitor_news_events(
        self,
        currencies: List[str]
    ) -> List[NewsEvent]:
        """Monitor economic calendar events"""
        try:
            events = []
            
            # Get current time
            current_time = datetime.now()
            
            # Fetch upcoming events
            raw_events = await self._fetch_economic_calendar(
                currencies,
                current_time,
                current_time + timedelta(days=1)
            )
            
            # Process each event
            for event in raw_events:
                # Calculate expected volatility
                volatility = await self._calculate_event_volatility(event)
                
                events.append(NewsEvent(
                    timestamp=event['time'],
                    currency=event['currency'],
                    event=event['description'],
                    impact=event['impact'],
                    forecast=event.get('forecast'),
                    previous=event.get('previous'),
                    volatility_expected=volatility
                ))
                
            return sorted(events, key=lambda x: x.timestamp)
            
        except Exception as e:
            self.logger.error(f"News monitoring error: {str(e)}")
            return []
            
    async def adjust_stops(
        self,
        symbol: str,
        base_stop: float,
        position_type: str
    ) -> float:
        """Calculate dynamic stop-loss adjustment"""
        try:
            # Get current market state
            state = await self._analyze_market_state(symbol)
            
            # Get adjustment factors
            vol_factor = self.stop_params['volatility_zones'][
                state.volatility_level
            ]['multiplier']
            
            trend_factor = self.stop_params['trend_factors'][
                'strong' if state.trend_strength > 0.7 else
                'weak' if state.trend_strength < 0.3 else
                'normal'
            ]
            
            regime_factor = self.stop_params['regime_adjustments'][
                state.market_regime
            ]
            
            # Calculate final adjustment
            adjustment = vol_factor * trend_factor * regime_factor
            
            # Apply adjustment to base stop
            adjusted_stop = base_stop * adjustment
            
            # Update tracking
            self.market_tracking['stops'][symbol] = {
                'base': base_stop,
                'adjusted': adjusted_stop,
                'factors': {
                    'volatility': vol_factor,
                    'trend': trend_factor,
                    'regime': regime_factor
                }
            }
            
            return adjusted_stop
            
        except Exception as e:
            self.logger.error(f"Stop-loss adjustment error: {str(e)}")
            return base_stop
            
    async def _calculate_correlation(
        self,
        primary_data: pd.DataFrame,
        secondary_data: pd.DataFrame
    ) -> Tuple[float, float]:
        """Calculate correlation coefficient and significance"""
        try:
            # Calculate returns
            primary_returns = primary_data['close'].pct_change().dropna()
            secondary_returns = secondary_data['close'].pct_change().dropna()
            
            # Calculate correlation
            correlation, p_value = pearsonr(
                primary_returns,
                secondary_returns
            )
            
            return correlation, 1 - p_value
            
        except Exception as e:
            self.logger.error(f"Correlation calculation error: {str(e)}")
            return 0.0, 0.0
            
    async def _calculate_trend_alignment(
        self,
        primary_data: pd.DataFrame,
        secondary_data: pd.DataFrame
    ) -> float:
        """Calculate trend alignment score"""
        try:
            # Calculate trends
            primary_trend = await self._calculate_trend(primary_data)
            secondary_trend = await self._calculate_trend(secondary_data)
            
            # Calculate alignment
            alignment = np.mean([
                1 if (p * s) > 0 else 0
                for p, s in zip(primary_trend, secondary_trend)
            ])
            
            return float(alignment)
            
        except Exception as e:
            self.logger.error(f"Trend alignment calculation error: {str(e)}")
            return 0.5
            
    async def _calculate_event_volatility(
        self,
        event: Dict
    ) -> float:
        """Calculate expected event volatility"""
        try:
            # Get base volatility multiplier
            base_volatility = self.news_params['volatility_multipliers'][
                event['impact'].lower()
            ]
            
            # Adjust for historical impact
            if event['currency'] in self.news_tracking['volatility']:
                historical = self.news_tracking['volatility'][
                    event['currency']
                ]
                base_volatility *= historical
                
            # Adjust for forecast deviation
            if event.get('forecast') and event.get('previous'):
                deviation = abs(
                    event['forecast'] - event['previous']
                ) / abs(event['previous'])
                base_volatility *= (1 + deviation)
                
            return base_volatility
            
        except Exception as e:
            self.logger.error(f"Event volatility calculation error: {str(e)}")
            return 1.0
            
    async def _analyze_market_state(
        self,
        symbol: str
    ) -> MarketState:
        """Analyze current market state"""
        try:
            # Get market data
            data = await self._get_market_data(symbol)
            
            # Calculate volatility level
            volatility = await self._calculate_volatility(data)
            vol_level = (
                'high' if volatility > 0.8 else
                'low' if volatility < 0.2 else
                'medium'
            )
            
            # Calculate trend strength
            trend_strength = await self._calculate_trend_strength(data)
            
            # Determine market regime
            regime = await self._determine_regime(
                data,
                volatility,
                trend_strength
            )
            
            # Calculate risk level
            risk = await self._calculate_risk_level(
                volatility,
                trend_strength,
                regime
            )
            
            # Calculate stop adjustment
            stop_adj = await self._calculate_stop_adjustment(
                vol_level,
                trend_strength,
                regime
            )
            
            return MarketState(
                volatility_level=vol_level,
                trend_strength=trend_strength,
                market_regime=regime,
                risk_level=risk,
                stop_adjustment=stop_adj
            )
            
        except Exception as e:
            self.logger.error(f"Market state analysis error: {str(e)}")
            return None
