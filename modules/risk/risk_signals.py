from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from dataclasses import dataclass
from ..analytics.market_analyzer import MarketAnalyzer
from ..deployment.error_handler import ErrorHandler

@dataclass
class VolatilitySignal:
    level: float  # Current volatility level
    state: str    # 'high', 'medium', 'low'
    atr: float    # Average True Range
    bb_width: float  # Bollinger Band Width
    recommendation: str  # Trading recommendation
    position_modifier: float  # Position size modifier
    stop_loss_modifier: float  # Stop loss distance modifier

@dataclass
class LiquiditySignal:
    level: float  # Current liquidity level
    state: str    # 'high', 'medium', 'low'
    spread: float  # Current spread
    depth: float  # Order book depth
    recommendation: str  # Trading recommendation
    max_position_size: float  # Maximum position size
    slippage_estimate: float  # Estimated slippage

@dataclass
class CorrelationSignal:
    matrix: pd.DataFrame  # Correlation matrix
    high_corr_pairs: List[Tuple[str, str, float]]  # Highly correlated pairs
    risk_exposure: float  # Overall portfolio risk exposure
    diversification_score: float  # Portfolio diversification score
    rebalance_needed: bool  # Whether portfolio rebalancing is needed

class RiskSignals:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('risk_signals')
        self.market_analyzer = MarketAnalyzer(config)
        
        # Initialize parameters
        self._init_risk_parameters()
        
    def _init_risk_parameters(self):
        """Initialize risk assessment parameters"""
        # Volatility parameters
        self.volatility_params = {
            'atr_period': 14,
            'bb_period': 20,
            'bb_std': 2,
            'high_vol_threshold': 1.5,
            'low_vol_threshold': 0.5,
            'position_modifiers': {
                'high': 0.5,    # Reduce position by 50%
                'medium': 1.0,  # Normal position size
                'low': 1.2      # Increase position by 20%
            },
            'stop_loss_modifiers': {
                'high': 1.5,    # Increase stop distance by 50%
                'medium': 1.0,  # Normal stop distance
                'low': 0.8      # Decrease stop distance by 20%
            }
        }
        
        # Liquidity parameters
        self.liquidity_params = {
            'min_depth': 100000,  # Minimum order book depth
            'max_spread': 0.0003,  # Maximum allowed spread
            'depth_levels': 5,    # Order book levels to analyze
            'spread_weight': 0.6,  # Weight for spread in liquidity score
            'depth_weight': 0.4,  # Weight for depth in liquidity score
            'slippage_factor': 0.00001  # Base slippage estimation factor
        }
        
        # Correlation parameters
        self.correlation_params = {
            'window': 100,  # Rolling window for correlation
            'high_corr_threshold': 0.7,  # High correlation threshold
            'max_portfolio_correlation': 0.5,  # Maximum portfolio correlation
            'min_pairs': 3,  # Minimum number of trading pairs
            'rebalance_threshold': 0.2  # Correlation change threshold for rebalancing
        }
        
    async def get_volatility_signal(
        self,
        symbol: str,
        timeframe: int,
        bars: int = 100
    ) -> VolatilitySignal:
        """Get volatility assessment signal"""
        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            df = pd.DataFrame(rates)
            
            # Calculate ATR
            atr = await self._calculate_atr(df, self.volatility_params['atr_period'])
            
            # Calculate Bollinger Bands Width
            bb_width = await self._calculate_bb_width(
                df,
                self.volatility_params['bb_period'],
                self.volatility_params['bb_std']
            )
            
            # Normalize volatility metrics
            vol_level = (atr / df['close'].mean() + bb_width) / 2
            
            # Determine volatility state
            if vol_level > self.volatility_params['high_vol_threshold']:
                state = 'high'
                recommendation = 'Reduce position size and increase stops'
            elif vol_level < self.volatility_params['low_vol_threshold']:
                state = 'low'
                recommendation = 'Increase position size cautiously'
            else:
                state = 'medium'
                recommendation = 'Normal trading conditions'
                
            return VolatilitySignal(
                level=vol_level,
                state=state,
                atr=atr,
                bb_width=bb_width,
                recommendation=recommendation,
                position_modifier=self.volatility_params['position_modifiers'][state],
                stop_loss_modifier=self.volatility_params['stop_loss_modifiers'][state]
            )
            
        except Exception as e:
            self.logger.error(f"Volatility signal error: {str(e)}")
            return None
            
    async def get_liquidity_signal(
        self,
        symbol: str,
        volume: float
    ) -> LiquiditySignal:
        """Get liquidity assessment signal"""
        try:
            # Get market depth
            depth = await self._get_market_depth(
                symbol,
                self.liquidity_params['depth_levels']
            )
            
            # Get current spread
            spread = mt5.symbol_info(symbol).spread * mt5.symbol_info(symbol).point
            
            # Calculate liquidity score
            liquidity_score = (
                self.liquidity_params['spread_weight'] * (1 - spread) +
                self.liquidity_params['depth_weight'] * (depth / self.liquidity_params['min_depth'])
            )
            
            # Determine liquidity state
            if liquidity_score > 0.8:
                state = 'high'
                recommendation = 'Suitable for aggressive trading'
                max_position = volume
                slippage = self.liquidity_params['slippage_factor']
            elif liquidity_score < 0.4:
                state = 'low'
                recommendation = 'Avoid trading - high slippage risk'
                max_position = 0
                slippage = self.liquidity_params['slippage_factor'] * 3
            else:
                state = 'medium'
                recommendation = 'Trade with caution'
                max_position = volume * 0.5
                slippage = self.liquidity_params['slippage_factor'] * 2
                
            return LiquiditySignal(
                level=liquidity_score,
                state=state,
                spread=spread,
                depth=depth,
                recommendation=recommendation,
                max_position_size=max_position,
                slippage_estimate=slippage
            )
            
        except Exception as e:
            self.logger.error(f"Liquidity signal error: {str(e)}")
            return None
            
    async def get_correlation_signal(
        self,
        symbols: List[str],
        timeframe: int
    ) -> CorrelationSignal:
        """Get correlation assessment signal"""
        try:
            # Get price data for all symbols
            price_data = {}
            for symbol in symbols:
                rates = mt5.copy_rates_from_pos(
                    symbol,
                    timeframe,
                    0,
                    self.correlation_params['window']
                )
                price_data[symbol] = pd.DataFrame(rates)['close']
                
            # Create price DataFrame
            df = pd.DataFrame(price_data)
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    correlation = corr_matrix.iloc[i, j]
                    if abs(correlation) > self.correlation_params['high_corr_threshold']:
                        high_corr_pairs.append(
                            (symbols[i], symbols[j], correlation)
                        )
                        
            # Calculate risk exposure
            risk_exposure = len(high_corr_pairs) / (len(symbols) * (len(symbols) - 1) / 2)
            
            # Calculate diversification score
            diversification_score = 1 - risk_exposure
            
            # Check if rebalancing is needed
            rebalance_needed = (
                risk_exposure > self.correlation_params['max_portfolio_correlation']
            )
            
            return CorrelationSignal(
                matrix=corr_matrix,
                high_corr_pairs=high_corr_pairs,
                risk_exposure=risk_exposure,
                diversification_score=diversification_score,
                rebalance_needed=rebalance_needed
            )
            
        except Exception as e:
            self.logger.error(f"Correlation signal error: {str(e)}")
            return None
            
    async def _calculate_atr(
        self,
        df: pd.DataFrame,
        period: int
    ) -> float:
        """Calculate Average True Range"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = tr.rolling(period).mean().iloc[-1]
            
            return atr
            
        except Exception as e:
            self.logger.error(f"ATR calculation error: {str(e)}")
            return 0
            
    async def _calculate_bb_width(
        self,
        df: pd.DataFrame,
        period: int,
        std: float
    ) -> float:
        """Calculate Bollinger Bands Width"""
        try:
            # Calculate middle band (SMA)
            middle = df['close'].rolling(period).mean()
            
            # Calculate standard deviation
            std_dev = df['close'].rolling(period).std()
            
            # Calculate upper and lower bands
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            
            # Calculate bandwidth
            bandwidth = (upper - lower) / middle
            
            return bandwidth.iloc[-1]
            
        except Exception as e:
            self.logger.error(f"BB width calculation error: {str(e)}")
            return 0
            
    async def _get_market_depth(
        self,
        symbol: str,
        levels: int
    ) -> float:
        """Get market depth from order book"""
        try:
            # Get order book
            book = mt5.market_book_get(symbol)
            
            if not book:
                return 0
                
            total_volume = 0
            
            # Calculate total volume for specified levels
            for i in range(min(levels, len(book))):
                if book[i].type == mt5.BOOK_TYPE_SELL:
                    total_volume += book[i].volume
                elif book[i].type == mt5.BOOK_TYPE_BUY:
                    total_volume += book[i].volume
                    
            return total_volume
            
        except Exception as e:
            self.logger.error(f"Market depth calculation error: {str(e)}")
            return 0
            
    async def get_combined_risk_signal(
        self,
        symbol: str,
        timeframe: int,
        volume: float,
        portfolio_symbols: List[str]
    ) -> Dict:
        """Get combined risk assessment signal"""
        try:
            # Get individual signals
            volatility = await self.get_volatility_signal(symbol, timeframe)
            liquidity = await self.get_liquidity_signal(symbol, volume)
            correlation = await self.get_correlation_signal(
                portfolio_symbols,
                timeframe
            )
            
            # Calculate overall risk score
            risk_score = (
                (1 if volatility.state == 'high' else 0) +
                (1 if liquidity.state == 'low' else 0) +
                (1 if correlation.rebalance_needed else 0)
            ) / 3
            
            return {
                'risk_score': risk_score,
                'volatility_signal': volatility,
                'liquidity_signal': liquidity,
                'correlation_signal': correlation,
                'trade_recommendation': await self._generate_recommendation(
                    volatility,
                    liquidity,
                    correlation
                )
            }
            
        except Exception as e:
            self.logger.error(f"Combined risk signal error: {str(e)}")
            return None
            
    async def _generate_recommendation(
        self,
        volatility: VolatilitySignal,
        liquidity: LiquiditySignal,
        correlation: CorrelationSignal
    ) -> str:
        """Generate trading recommendation based on all signals"""
        try:
            if liquidity.state == 'low':
                return "Avoid trading - Insufficient liquidity"
                
            if volatility.state == 'high' and correlation.rebalance_needed:
                return "High risk - Reduce exposure and rebalance portfolio"
                
            if volatility.state == 'low' and liquidity.state == 'high':
                return "Favorable conditions - Consider increasing position sizes"
                
            if correlation.risk_exposure > 0.7:
                return "High correlation - Consider portfolio diversification"
                
            return "Normal trading conditions - Maintain standard position sizes"
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {str(e)}")
            return "Error generating recommendation"
