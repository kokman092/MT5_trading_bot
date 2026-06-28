from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from dataclasses import dataclass
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from ..analytics.market_analyzer import MarketAnalyzer
from ..deployment.error_handler import ErrorHandler

@dataclass
class StopLevelSignal:
    price: float  # Stop level price
    type: str    # 'atr', 'support_resistance', 'moving_average', 'ai'
    confidence: float  # Confidence score
    distance_pips: float  # Distance in pips
    risk_reward: float  # Risk-reward ratio

@dataclass
class TimeExitSignal:
    should_exit: bool  # Whether to exit based on time
    hold_time: int    # Time position has been held
    momentum_score: float  # Current momentum score
    trend_strength: float  # Current trend strength
    recommendation: str   # Exit recommendation

@dataclass
class ProfitExitSignal:
    partial_exit_levels: List[Dict]  # List of partial exit levels
    trailing_stop: float  # Current trailing stop level
    locked_profit: float  # Amount of profit locked in
    remaining_position: float  # Remaining position size

class StopManager:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('stop_manager')
        self.market_analyzer = MarketAnalyzer(config)
        
        # Initialize parameters
        self._init_stop_parameters()
        self._init_models()
        
    def _init_stop_parameters(self):
        """Initialize stop management parameters"""
        # Dynamic stop-loss parameters
        self.stop_loss_params = {
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'ma_period': 50,
            'sr_lookback': 100,
            'min_stop_distance': 10,  # pips
            'max_stop_distance': 100,  # pips
            'confidence_threshold': 0.7
        }
        
        # Time-based exit parameters
        self.time_exit_params = {
            'max_hold_time': 48,  # hours
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
        
        # Profit lock-in parameters
        self.profit_lock_params = {
            'partial_exit_levels': [
                {'size': 0.5, 'rr_ratio': 1.0},
                {'size': 0.25, 'rr_ratio': 1.5},
                {'size': 0.25, 'rr_ratio': 2.0}
            ],
            'trailing_stop_multiplier': 1.5,
            'volatility_adjustment': True,
            'news_impact_threshold': 0.8
        }
        
    def _init_models(self):
        """Initialize AI models"""
        try:
            # AI stop-loss model
            self.stop_loss_model = self._create_stop_loss_model()
            
            # Time exit model
            self.time_exit_model = self._create_time_exit_model()
            
            # Load pre-trained models if available
            self._load_models()
            
        except Exception as e:
            self.logger.error(f"Model initialization error: {str(e)}")
            
    async def get_dynamic_stop_loss(
        self,
        symbol: str,
        timeframe: int,
        entry_price: float,
        direction: str
    ) -> StopLevelSignal:
        """Get dynamic stop-loss level"""
        try:
            stop_signals = []
            
            # ATR-based stop
            atr_stop = await self._calculate_atr_stop(
                symbol,
                timeframe,
                entry_price,
                direction
            )
            stop_signals.append(atr_stop)
            
            # Support/Resistance stop
            sr_stop = await self._calculate_sr_stop(
                symbol,
                timeframe,
                entry_price,
                direction
            )
            stop_signals.append(sr_stop)
            
            # Moving Average stop
            ma_stop = await self._calculate_ma_stop(
                symbol,
                timeframe,
                entry_price,
                direction
            )
            stop_signals.append(ma_stop)
            
            # AI-based stop
            ai_stop = await self._calculate_ai_stop(
                symbol,
                timeframe,
                entry_price,
                direction
            )
            stop_signals.append(ai_stop)
            
            # Select best stop level
            best_stop = await self._select_best_stop(stop_signals)
            
            return best_stop
            
        except Exception as e:
            self.logger.error(f"Dynamic stop-loss error: {str(e)}")
            return None
            
    async def get_time_exit_signal(
        self,
        symbol: str,
        timeframe: int,
        entry_time: datetime,
        position_data: Dict
    ) -> TimeExitSignal:
        """Get time-based exit signal"""
        try:
            # Calculate hold time
            current_time = datetime.now()
            hold_time = (current_time - entry_time).total_seconds() / 3600
            
            # Check maximum hold time
            if hold_time > self.time_exit_params['max_hold_time']:
                return TimeExitSignal(
                    should_exit=True,
                    hold_time=hold_time,
                    momentum_score=0,
                    trend_strength=0,
                    recommendation="Exit: Maximum hold time exceeded"
                )
                
            # Calculate momentum indicators
            momentum_data = await self._calculate_momentum_indicators(
                symbol,
                timeframe
            )
            
            # Get AI-based time exit signal
            ai_signal = await self._get_ai_time_signal(
                symbol,
                timeframe,
                momentum_data,
                position_data
            )
            
            return TimeExitSignal(
                should_exit=ai_signal['should_exit'],
                hold_time=hold_time,
                momentum_score=momentum_data['momentum_score'],
                trend_strength=momentum_data['trend_strength'],
                recommendation=ai_signal['recommendation']
            )
            
        except Exception as e:
            self.logger.error(f"Time exit signal error: {str(e)}")
            return None
            
    async def get_profit_exit_signal(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_size: float,
        direction: str
    ) -> ProfitExitSignal:
        """Get profit-taking exit signal"""
        try:
            partial_exits = []
            remaining_size = position_size
            locked_profit = 0
            
            # Calculate profit in pips
            profit_pips = abs(current_price - entry_price) / mt5.symbol_info(symbol).point
            
            # Calculate risk in pips
            risk_pips = await self._calculate_risk_pips(symbol, entry_price, direction)
            
            # Check each partial exit level
            for level in self.profit_lock_params['partial_exit_levels']:
                rr_ratio = profit_pips / risk_pips
                
                if rr_ratio >= level['rr_ratio']:
                    exit_size = position_size * level['size']
                    remaining_size -= exit_size
                    locked_profit += (exit_size / position_size) * profit_pips
                    
                    partial_exits.append({
                        'size': exit_size,
                        'price': current_price,
                        'rr_ratio': rr_ratio
                    })
                    
            # Calculate trailing stop
            trailing_stop = await self._calculate_trailing_stop(
                symbol,
                current_price,
                direction,
                profit_pips
            )
            
            return ProfitExitSignal(
                partial_exit_levels=partial_exits,
                trailing_stop=trailing_stop,
                locked_profit=locked_profit,
                remaining_position=remaining_size
            )
            
        except Exception as e:
            self.logger.error(f"Profit exit signal error: {str(e)}")
            return None
            
    async def _calculate_atr_stop(
        self,
        symbol: str,
        timeframe: int,
        entry_price: float,
        direction: str
    ) -> StopLevelSignal:
        """Calculate ATR-based stop-loss"""
        try:
            # Get ATR
            rates = mt5.copy_rates_from_pos(
                symbol,
                timeframe,
                0,
                self.stop_loss_params['atr_period']
            )
            df = pd.DataFrame(rates)
            
            atr = df['high'].rolling(self.stop_loss_params['atr_period']).max() - \
                  df['low'].rolling(self.stop_loss_params['atr_period']).min()
            atr = atr.mean()
            
            # Calculate stop distance
            stop_distance = atr * self.stop_loss_params['atr_multiplier']
            
            # Calculate stop price
            if direction == 'buy':
                stop_price = entry_price - stop_distance
            else:
                stop_price = entry_price + stop_distance
                
            return StopLevelSignal(
                price=stop_price,
                type='atr',
                confidence=0.8,
                distance_pips=stop_distance / mt5.symbol_info(symbol).point,
                risk_reward=1.0
            )
            
        except Exception as e:
            self.logger.error(f"ATR stop calculation error: {str(e)}")
            return None
            
    async def _calculate_sr_stop(
        self,
        symbol: str,
        timeframe: int,
        entry_price: float,
        direction: str
    ) -> StopLevelSignal:
        """Calculate Support/Resistance based stop-loss"""
        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos(
                symbol,
                timeframe,
                0,
                self.stop_loss_params['sr_lookback']
            )
            df = pd.DataFrame(rates)
            
            # Find support/resistance levels
            if direction == 'buy':
                levels = df['low'].rolling(20).min()
                stop_price = levels[levels < entry_price].max()
            else:
                levels = df['high'].rolling(20).max()
                stop_price = levels[levels > entry_price].min()
                
            stop_distance = abs(entry_price - stop_price)
            
            return StopLevelSignal(
                price=stop_price,
                type='support_resistance',
                confidence=0.75,
                distance_pips=stop_distance / mt5.symbol_info(symbol).point,
                risk_reward=1.2
            )
            
        except Exception as e:
            self.logger.error(f"S/R stop calculation error: {str(e)}")
            return None
            
    async def _calculate_ma_stop(
        self,
        symbol: str,
        timeframe: int,
        entry_price: float,
        direction: str
    ) -> StopLevelSignal:
        """Calculate Moving Average based stop-loss"""
        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos(
                symbol,
                timeframe,
                0,
                self.stop_loss_params['ma_period']
            )
            df = pd.DataFrame(rates)
            
            # Calculate MA
            ma = df['close'].rolling(self.stop_loss_params['ma_period']).mean()
            current_ma = ma.iloc[-1]
            
            # Calculate stop price
            if direction == 'buy':
                stop_price = min(current_ma, entry_price - self.stop_loss_params['min_stop_distance'])
            else:
                stop_price = max(current_ma, entry_price + self.stop_loss_params['min_stop_distance'])
                
            stop_distance = abs(entry_price - stop_price)
            
            return StopLevelSignal(
                price=stop_price,
                type='moving_average',
                confidence=0.7,
                distance_pips=stop_distance / mt5.symbol_info(symbol).point,
                risk_reward=1.1
            )
            
        except Exception as e:
            self.logger.error(f"MA stop calculation error: {str(e)}")
            return None
            
    async def _calculate_ai_stop(
        self,
        symbol: str,
        timeframe: int,
        entry_price: float,
        direction: str
    ) -> StopLevelSignal:
        """Calculate AI-based stop-loss"""
        try:
            # Prepare features for AI model
            features = await self._prepare_stop_features(
                symbol,
                timeframe,
                entry_price,
                direction
            )
            
            # Get model prediction
            prediction = self.stop_loss_model.predict(features)
            
            # Calculate stop price
            stop_distance = prediction[0] * self.stop_loss_params['atr_multiplier']
            
            if direction == 'buy':
                stop_price = entry_price - stop_distance
            else:
                stop_price = entry_price + stop_distance
                
            return StopLevelSignal(
                price=stop_price,
                type='ai',
                confidence=0.85,
                distance_pips=stop_distance / mt5.symbol_info(symbol).point,
                risk_reward=1.3
            )
            
        except Exception as e:
            self.logger.error(f"AI stop calculation error: {str(e)}")
            return None
            
    def _create_stop_loss_model(self):
        """Create AI model for stop-loss prediction"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='linear')
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Stop loss model creation error: {str(e)}")
            return None
            
    def _create_time_exit_model(self):
        """Create AI model for time-based exit decisions"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Time exit model creation error: {str(e)}")
            return None
            
    async def _calculate_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        direction: str,
        profit_pips: float
    ) -> float:
        """Calculate trailing stop level"""
        try:
            # Base trailing stop distance
            base_distance = profit_pips * self.profit_lock_params['trailing_stop_multiplier']
            
            # Adjust for volatility if enabled
            if self.profit_lock_params['volatility_adjustment']:
                volatility = await self._calculate_volatility(symbol)
                base_distance *= (1 + volatility)
                
            # Adjust for news impact if significant
            if await self._check_news_impact(symbol) > self.profit_lock_params['news_impact_threshold']:
                base_distance *= 1.5
                
            # Calculate final stop price
            if direction == 'buy':
                stop_price = current_price - base_distance
            else:
                stop_price = current_price + base_distance
                
            return stop_price
            
        except Exception as e:
            self.logger.error(f"Trailing stop calculation error: {str(e)}")
            return current_price
