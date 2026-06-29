from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any, Union
from enum import Enum
import json
import logging
import uuid
import numpy as np

class SignalType(Enum):
    TREND = "trend"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    
class SignalDirection(str, Enum):
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    NONE = "none"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class SignalStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"

class SignalTimeframe(str, Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"

class SignalSource(str, Enum):
    PRICE_ACTION = "price_action"
    INDICATOR = "indicator"
    PATTERN = "pattern"
    ML_MODEL = "ml_model"
    SENTIMENT = "sentiment"
    REGIME = "regime"
    ENSEMBLE = "ensemble"
    MANUAL = "manual"

class StrategyType(str, Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    PATTERN_RECOGNITION = "pattern_recognition"
    SENTIMENT_BASED = "sentiment_based"
    VOLATILITY_BASED = "volatility_based"

@dataclass
class Signal:
    """Enhanced trade signal class with detailed information for higher profitability trades"""
    
    # Basic signal information
    symbol: str
    direction: Union[SignalDirection, str]
    timeframe: Union[SignalTimeframe, str]
    
    # Entry/exit information
    entry_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    
    # Multiple take profit levels (optional)
    tp_levels: List[Dict[str, float]] = field(default_factory=list)  # [{'price': x, 'volume': y}, ...]
    
    # Signal metadata
    timestamp: datetime = field(default_factory=datetime.now)
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    expiration: Optional[datetime] = None
    
    # Order details
    order_type: Union[OrderType, str] = OrderType.MARKET
    position_size: Optional[float] = None
    risk_amount: Optional[float] = None
    risk_percent: Optional[float] = None
    
    # Signal quality metrics
    confidence: float = 0.5  # 0.0 to 1.0
    strength: Union[SignalStrength, str] = SignalStrength.MODERATE
    source: Union[SignalSource, str] = SignalSource.INDICATOR
    strategy_type: Union[StrategyType, str] = StrategyType.TREND_FOLLOWING
    
    # Signal reasoning
    trigger_indicators: Dict[str, Any] = field(default_factory=dict)
    confirming_indicators: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    # Extended signal data
    market_context: Dict[str, Any] = field(default_factory=dict)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    # Trade management
    use_trailing_stop: bool = False
    trailing_stop_trigger: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    
    # Execution preferences
    max_spread: Optional[float] = None
    min_volume: Optional[float] = None
    slippage_tolerance: Optional[float] = None
    use_advanced_order: bool = False
    advanced_order_type: Optional[str] = None
    
    def __init__(
        self,
        symbol: str,
        direction: Union[SignalDirection, str] = 'none',
        timeframe: Union[SignalTimeframe, str] = 'H1',
        entry_price: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: Optional[float] = None,
        tp_levels: List[Dict[str, float]] = None,
        timestamp: datetime = None,
        signal_id: str = None,
        expiration: Optional[datetime] = None,
        order_type: Union[OrderType, str] = OrderType.MARKET,
        position_size: Optional[float] = None,
        risk_amount: Optional[float] = None,
        risk_percent: Optional[float] = None,
        confidence: float = 0.5,
        strength: Union[SignalStrength, str] = SignalStrength.MODERATE,
        source: Union[SignalSource, str] = SignalSource.INDICATOR,
        strategy_type: Union[StrategyType, str] = StrategyType.TREND_FOLLOWING,
        trigger_indicators: Dict[str, Any] = None,
        confirming_indicators: Dict[str, Any] = None,
        description: str = "",
        market_context: Dict[str, Any] = None,
        custom_data: Dict[str, Any] = None,
        use_trailing_stop: bool = False,
        trailing_stop_trigger: Optional[float] = None,
        trailing_stop_distance: Optional[float] = None,
        max_spread: Optional[float] = None,
        min_volume: Optional[float] = None,
        slippage_tolerance: Optional[float] = None,
        use_advanced_order: bool = False,
        advanced_order_type: Optional[str] = None,
        **kwargs
    ):
        # Map alias 'type' to 'direction'
        if 'type' in kwargs:
            direction = kwargs.pop('type')
        # Map alias 'strategy' to 'strategy_type'
        if 'strategy' in kwargs:
            strategy_type = kwargs.pop('strategy')
            
        self.symbol = symbol
        self.direction = direction
        self.timeframe = timeframe
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.tp_levels = tp_levels if tp_levels is not None else []
        self.timestamp = timestamp if timestamp is not None else datetime.now()
        self.signal_id = signal_id if signal_id is not None else str(uuid.uuid4())
        self.expiration = expiration
        self.order_type = order_type
        self.position_size = position_size
        self.risk_amount = risk_amount
        self.risk_percent = risk_percent
        self.confidence = confidence
        self.strength = strength
        self.source = source
        self.strategy_type = strategy_type
        self.trigger_indicators = trigger_indicators if trigger_indicators is not None else {}
        self.confirming_indicators = confirming_indicators if confirming_indicators is not None else {}
        self.description = description
        self.market_context = market_context if market_context is not None else {}
        self.custom_data = custom_data if custom_data is not None else {}
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_trigger = trailing_stop_trigger
        self.trailing_stop_distance = trailing_stop_distance
        self.max_spread = max_spread
        self.min_volume = min_volume
        self.slippage_tolerance = slippage_tolerance
        self.use_advanced_order = use_advanced_order
        self.advanced_order_type = advanced_order_type
        
        # Call __post_init__ to finalize type casting and validation
        self.__post_init__()
        
    def __post_init__(self):
        """Convert string enums to proper enum types"""
        if isinstance(self.direction, str):
            try:
                self.direction = SignalDirection(self.direction.lower())
            except ValueError:
                pass
            
        if isinstance(self.timeframe, str):
            try:
                self.timeframe = SignalTimeframe(self.timeframe.upper())
            except ValueError:
                pass
            
        if isinstance(self.order_type, str):
            try:
                self.order_type = OrderType(self.order_type.lower())
            except ValueError:
                pass
            
        if isinstance(self.strength, str):
            try:
                self.strength = SignalStrength(self.strength.lower())
            except ValueError:
                pass
            
        if isinstance(self.source, str):
            try:
                self.source = SignalSource(self.source.lower())
            except ValueError:
                pass
            
        if isinstance(self.strategy_type, str):
            try:
                self.strategy_type = StrategyType(self.strategy_type.lower())
            except ValueError:
                pass
        
        # Validate required fields
        self._validate()
        
    def _validate(self):
        """Validate signal parameters"""
        # Direction must be valid
        if self.direction not in [d for d in SignalDirection]:
            raise ValueError(f"Invalid signal direction: {self.direction}")
            
        # Entry price must be positive
        if self.entry_price <= 0:
            raise ValueError(f"Invalid entry price: {self.entry_price}")
            
        # Stop loss must be valid based on direction
        if self.direction == SignalDirection.BUY and self.stop_loss >= self.entry_price:
            raise ValueError(f"Stop loss ({self.stop_loss}) must be below entry price ({self.entry_price}) for BUY signal")
            
        if self.direction == SignalDirection.SELL and self.stop_loss <= self.entry_price:
            raise ValueError(f"Stop loss ({self.stop_loss}) must be above entry price ({self.entry_price}) for SELL signal")
            
        # Validate take profit if provided
        if self.take_profit is not None:
            if self.direction == SignalDirection.BUY and self.take_profit <= self.entry_price:
                raise ValueError(f"Take profit ({self.take_profit}) must be above entry price ({self.entry_price}) for BUY signal")
                
            if self.direction == SignalDirection.SELL and self.take_profit >= self.entry_price:
                raise ValueError(f"Take profit ({self.take_profit}) must be below entry price ({self.entry_price}) for SELL signal")
                
        # Validate TP levels if provided
        for tp in self.tp_levels:
            if 'price' not in tp or 'volume' not in tp:
                raise ValueError(f"TP level must contain price and volume: {tp}")
                
            if tp['volume'] <= 0 or tp['volume'] > 1:
                raise ValueError(f"TP level volume must be between 0 and 1: {tp['volume']}")
                
            if self.direction == SignalDirection.BUY and tp['price'] <= self.entry_price:
                raise ValueError(f"TP level price ({tp['price']}) must be above entry price ({self.entry_price}) for BUY signal")
                
            if self.direction == SignalDirection.SELL and tp['price'] >= self.entry_price:
                raise ValueError(f"TP level price ({tp['price']}) must be below entry price ({self.entry_price}) for SELL signal")
        
    @property
    def volume(self) -> float:
        """Compatibility getter for position size/volume"""
        return self.position_size if self.position_size is not None else 0.01

    @volume.setter
    def volume(self, value: float):
        """Compatibility setter for position size/volume"""
        self.position_size = value

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk-to-reward ratio for the signal"""
        if self.take_profit is None:
            return None
            
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.entry_price - self.take_profit)
        
        if risk == 0:
            return None
            
        return reward / risk
        
    def get_risk_reward_ratio(self) -> Optional[float]:
        """Backward-compatible method to calculate risk-to-reward ratio"""
        return self.risk_reward_ratio
        
    @property
    def stop_distance(self) -> float:
        """Calculate stop loss distance in price units"""
        return abs(self.entry_price - self.stop_loss)
        
    @property
    def stop_percent(self) -> float:
        """Calculate stop loss distance as percentage of entry price"""
        return self.stop_distance / self.entry_price * 100
        
    @property
    def trade_id(self) -> str:
        """Generate unique trade ID based on signal properties"""
        return f"{self.symbol}_{self.timeframe}_{self.direction}_{self.signal_id[:8]}"
        
    def add_take_profit_level(self, price: float, volume: float) -> None:
        """Add a take profit level with partial position size"""
        if volume <= 0 or volume > 1:
            raise ValueError(f"TP level volume must be between 0 and 1: {volume}")
            
        # Validate price direction
        if self.direction == SignalDirection.BUY and price <= self.entry_price:
            raise ValueError(f"TP level price ({price}) must be above entry price ({self.entry_price}) for BUY signal")
            
        if self.direction == SignalDirection.SELL and price >= self.entry_price:
            raise ValueError(f"TP level price ({price}) must be below entry price ({self.entry_price}) for SELL signal")
            
        # Check if total TP volume exceeds 1.0
        total_volume = sum([tp['volume'] for tp in self.tp_levels]) + volume
        if total_volume > 1.0:
            raise ValueError(f"Total TP volume ({total_volume}) exceeds 1.0")
            
        self.tp_levels.append({'price': price, 'volume': volume})
        
        # Sort TP levels by price (ascending for SELL, descending for BUY)
        self.tp_levels.sort(key=lambda x: x['price'], reverse=(self.direction == SignalDirection.BUY))
        
    def update_confidence(self, new_confidence: float) -> None:
        """Update signal confidence score"""
        if new_confidence < 0 or new_confidence > 1:
            raise ValueError(f"Confidence must be between 0 and 1: {new_confidence}")
            
        self.confidence = new_confidence
        
        # Update strength based on confidence
        if new_confidence >= 0.8:
            self.strength = SignalStrength.STRONG
        elif new_confidence >= 0.5:
            self.strength = SignalStrength.MODERATE
        else:
            self.strength = SignalStrength.WEAK
    
    def add_indicator(self, name: str, value: Any, is_trigger: bool = False) -> None:
        """Add indicator value to the signal"""
        if is_trigger:
            self.trigger_indicators[name] = value
        else:
            self.confirming_indicators[name] = value
            
    def set_market_context(self, context: Dict[str, Any]) -> None:
        """Set market context data"""
        self.market_context = context
        
    def is_valid(self) -> bool:
        """Check if signal is valid and can be executed"""
        # Check expiration
        if self.expiration is not None and datetime.now() > self.expiration:
            return False
            
        # Basic validation
        try:
            self._validate()
            return True
        except ValueError:
            return False
            
    def is_expired(self) -> bool:
        """Check if signal has expired"""
        if self.expiration is None:
            return False
            
        return datetime.now() > self.expiration
        
    def to_dict(self) -> Dict:
        """Convert signal to dictionary for storage/transmission"""
        return {
            'symbol': self.symbol,
            'direction': self.direction.value if isinstance(self.direction, SignalDirection) else self.direction,
            'timeframe': self.timeframe.value if isinstance(self.timeframe, SignalTimeframe) else self.timeframe,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'tp_levels': self.tp_levels,
            'timestamp': self.timestamp.isoformat(),
            'signal_id': self.signal_id,
            'expiration': self.expiration.isoformat() if self.expiration else None,
            'order_type': self.order_type.value if isinstance(self.order_type, OrderType) else self.order_type,
            'position_size': self.position_size,
            'risk_amount': self.risk_amount,
            'risk_percent': self.risk_percent,
            'confidence': self.confidence,
            'strength': self.strength.value if isinstance(self.strength, SignalStrength) else self.strength,
            'source': self.source.value if isinstance(self.source, SignalSource) else self.source,
            'strategy_type': self.strategy_type.value if isinstance(self.strategy_type, StrategyType) else self.strategy_type,
            'trigger_indicators': self.trigger_indicators,
            'confirming_indicators': self.confirming_indicators,
            'description': self.description,
            'market_context': self.market_context,
            'custom_data': self.custom_data,
            'use_trailing_stop': self.use_trailing_stop,
            'trailing_stop_trigger': self.trailing_stop_trigger,
            'trailing_stop_distance': self.trailing_stop_distance,
            'max_spread': self.max_spread,
            'min_volume': self.min_volume,
            'slippage_tolerance': self.slippage_tolerance,
            'use_advanced_order': self.use_advanced_order,
            'advanced_order_type': self.advanced_order_type
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Signal':
        """Create signal from dictionary"""
        # Convert timestamp strings to datetime objects
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            
        if 'expiration' in data and data['expiration'] and isinstance(data['expiration'], str):
            data['expiration'] = datetime.fromisoformat(data['expiration'])
            
        return cls(**data)
        
    def to_json(self) -> str:
        """Convert signal to JSON string"""
        dict_data = self.to_dict()
        
        # Convert datetime objects to strings
        if 'timestamp' in dict_data and isinstance(dict_data['timestamp'], datetime):
            dict_data['timestamp'] = dict_data['timestamp'].isoformat()
            
        if 'expiration' in dict_data and dict_data['expiration'] and isinstance(dict_data['expiration'], datetime):
            dict_data['expiration'] = dict_data['expiration'].isoformat()
            
        return json.dumps(dict_data)
        
    @classmethod
    def from_json(cls, json_str: str) -> 'Signal':
        """Create signal from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
        
    def __str__(self) -> str:
        """String representation of signal"""
        return f"Signal({self.symbol}, {self.direction}, entry={self.entry_price}, SL={self.stop_loss}, TP={self.take_profit}, conf={self.confidence:.2f})"

class SignalManager:
    """Manages multiple signals, handles signal filtering, combination and prioritization"""
    
    def __init__(self, config: Dict):
        """Initialize signal manager with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Signal settings
        self.min_confidence = config.get('signals', {}).get('min_confidence', 0.5)
        self.min_risk_reward = config.get('signals', {}).get('min_risk_reward', 1.5)
        self.max_signals_per_symbol = config.get('signals', {}).get('max_signals_per_symbol', 1)
        
        # Active signals
        self.active_signals: Dict[str, List[Signal]] = {}
        
        # Historical signals for performance tracking
        self.signal_history: List[Dict] = []
        self.max_history_size = config.get('signals', {}).get('max_history_size', 1000)
        
        # Conflicting signal resolution
        self.conflicting_resolution = config.get('signals', {}).get('conflicting_resolution', 'highest_confidence')  # 'highest_confidence', 'newest', 'ensemble'
        
    def add_signal(self, signal: Signal) -> bool:
        """Add a new trading signal if it passes filters"""
        try:
            # Validate signal
            if not signal.is_valid():
                self.logger.warning(f"Invalid signal rejected: {signal}")
                return False
                
            # Check confidence threshold
            if signal.confidence < self.min_confidence:
                self.logger.info(f"Signal rejected due to low confidence ({signal.confidence}): {signal}")
                return False
                
            # Check risk/reward
            if signal.risk_reward_ratio is not None and signal.risk_reward_ratio < self.min_risk_reward:
                self.logger.info(f"Signal rejected due to poor risk/reward ({signal.risk_reward_ratio}): {signal}")
                return False
                
            # Check conflicting signals
            if self._has_conflicting_signals(signal):
                self.logger.info(f"Conflicting signal detected: {signal}")
                self._resolve_conflicting_signals(signal)
                
            # Add to active signals
            if signal.symbol not in self.active_signals:
                self.active_signals[signal.symbol] = []
                
            # Check max signals per symbol
            if len(self.active_signals[signal.symbol]) >= self.max_signals_per_symbol:
                # Remove lowest confidence signal if at capacity
                if self.conflicting_resolution == 'highest_confidence':
                    self.active_signals[signal.symbol].sort(key=lambda s: s.confidence)
                    removed = self.active_signals[signal.symbol].pop(0)
                    self.logger.info(f"Removed signal due to capacity: {removed}")
                else:
                    self.logger.info(f"Signal rejected due to max capacity for {signal.symbol}")
                    return False
                    
            # Add to active signals
            self.active_signals[signal.symbol].append(signal)
            self.logger.info(f"Added new signal: {signal}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding signal: {str(e)}")
            return False
    
    def get_active_signals(self, symbol: Optional[str] = None) -> List[Signal]:
        """Get active signals, optionally filtered by symbol"""
        try:
            # Return signals for specific symbol
            if symbol is not None:
                return self.active_signals.get(symbol, [])
                
            # Return all active signals
            all_signals = []
            for symbol_signals in self.active_signals.values():
                all_signals.extend(symbol_signals)
                
            return all_signals
            
        except Exception as e:
            self.logger.error(f"Error getting active signals: {str(e)}")
            return []
            
    def remove_signal(self, signal_id: str) -> bool:
        """Remove a signal by ID"""
        try:
            for symbol, signals in self.active_signals.items():
                for i, signal in enumerate(signals):
                    if signal.signal_id == signal_id:
                        removed = self.active_signals[symbol].pop(i)
                        self.logger.info(f"Removed signal: {removed}")
                        
                        # Add to history
                        self._add_to_history(removed.to_dict(), 'removed')
                        return True
                        
            return False
            
        except Exception as e:
            self.logger.error(f"Error removing signal: {str(e)}")
            return False
            
    def mark_signal_executed(self, signal_id: str, execution_result: Dict) -> None:
        """Mark a signal as executed and add to history"""
        try:
            for symbol, signals in self.active_signals.items():
                for i, signal in enumerate(signals):
                    if signal.signal_id == signal_id:
                        executed = self.active_signals[symbol].pop(i)
                        self.logger.info(f"Signal executed: {executed}")
                        
                        # Add to history
                        signal_dict = executed.to_dict()
                        signal_dict['execution_result'] = execution_result
                        self._add_to_history(signal_dict, 'executed')
                        return
                        
        except Exception as e:
            self.logger.error(f"Error marking signal executed: {str(e)}")
            
    def clear_expired_signals(self) -> int:
        """Remove expired signals"""
        try:
            expired_count = 0
            for symbol in list(self.active_signals.keys()):
                valid_signals = []
                for signal in self.active_signals[symbol]:
                    if signal.is_expired():
                        self.logger.info(f"Signal expired: {signal}")
                        self._add_to_history(signal.to_dict(), 'expired')
                        expired_count += 1
                    else:
                        valid_signals.append(signal)
                        
                if valid_signals:
                    self.active_signals[symbol] = valid_signals
                else:
                    del self.active_signals[symbol]
                    
            return expired_count
            
        except Exception as e:
            self.logger.error(f"Error clearing expired signals: {str(e)}")
            return 0
            
    def get_signal_by_id(self, signal_id: str) -> Optional[Signal]:
        """Get a signal by ID"""
        for signals in self.active_signals.values():
            for signal in signals:
                if signal.signal_id == signal_id:
                    return signal
                    
        return None
        
    def _has_conflicting_signals(self, new_signal: Signal) -> bool:
        """Check if there are conflicting signals for the same symbol"""
        if new_signal.symbol not in self.active_signals:
            return False
            
        existing_signals = self.active_signals[new_signal.symbol]
        
        # Check for direction conflicts
        for signal in existing_signals:
            if signal.timeframe == new_signal.timeframe and signal.direction != new_signal.direction:
                return True
                
        return False
        
    def _resolve_conflicting_signals(self, new_signal: Signal) -> None:
        """Resolve conflicting signals based on configuration"""
        if new_signal.symbol not in self.active_signals:
            return
            
        if self.conflicting_resolution == 'newest':
            # Keep only the new signal
            for i, signal in enumerate(self.active_signals[new_signal.symbol]):
                if signal.timeframe == new_signal.timeframe and signal.direction != new_signal.direction:
                    removed = self.active_signals[new_signal.symbol].pop(i)
                    self.logger.info(f"Removed conflicting signal (newer preferred): {removed}")
                    self._add_to_history(removed.to_dict(), 'conflicting')
                    
        elif self.conflicting_resolution == 'highest_confidence':
            # Keep signal with highest confidence
            for i, signal in enumerate(self.active_signals[new_signal.symbol]):
                if signal.timeframe == new_signal.timeframe and signal.direction != new_signal.direction:
                    if signal.confidence < new_signal.confidence:
                        removed = self.active_signals[new_signal.symbol].pop(i)
                        self.logger.info(f"Removed conflicting signal (lower confidence): {removed}")
                        self._add_to_history(removed.to_dict(), 'conflicting')
                    else:
                        # Existing signal has higher confidence, don't add new one
                        self.logger.info(f"New signal rejected due to lower confidence than existing conflicting signal: {new_signal}")
                        return
                        
        elif self.conflicting_resolution == 'ensemble':
            # Try to combine/aggregate signals
            # This is a simplified approach - a real implementation would be more sophisticated
            signals_by_direction = {'buy': [], 'sell': []}
            
            for signal in self.active_signals[new_signal.symbol]:
                signals_by_direction[signal.direction.value if isinstance(signal.direction, SignalDirection) else signal.direction].append(signal)
                
            signals_by_direction[new_signal.direction.value if isinstance(new_signal.direction, SignalDirection) else new_signal.direction].append(new_signal)
            
            # Calculate average confidence for each direction
            dir_confidence = {}
            for direction, signals in signals_by_direction.items():
                if signals:
                    dir_confidence[direction] = sum(s.confidence for s in signals) / len(signals)
                else:
                    dir_confidence[direction] = 0
                    
            # Keep signals only for the direction with highest confidence
            winning_direction = max(dir_confidence.items(), key=lambda x: x[1])[0]
            
            # Remove signals that don't match winning direction
            for i, signal in reversed(list(enumerate(self.active_signals[new_signal.symbol]))):
                sig_dir = signal.direction.value if isinstance(signal.direction, SignalDirection) else signal.direction
                if sig_dir != winning_direction:
                    removed = self.active_signals[new_signal.symbol].pop(i)
                    self.logger.info(f"Removed conflicting signal (ensemble resolution): {removed}")
                    self._add_to_history(removed.to_dict(), 'conflicting')
                    
            # Don't add the new signal if it's not the winning direction
            new_sig_dir = new_signal.direction.value if isinstance(new_signal.direction, SignalDirection) else new_signal.direction
            if new_sig_dir != winning_direction:
                return
                
    def _add_to_history(self, signal_dict: Dict, status: str) -> None:
        """Add signal to history with status"""
        signal_dict['status'] = status
        signal_dict['history_timestamp'] = datetime.now().isoformat()
        
        self.signal_history.append(signal_dict)
        
        # Limit history size
        if len(self.signal_history) > self.max_history_size:
            self.signal_history = self.signal_history[-self.max_history_size:]
            
    def get_signal_stats(self) -> Dict:
        """Get statistics about signals"""
        try:
            stats = {
                'active_count': sum(len(signals) for signals in self.active_signals.values()),
                'symbols_count': len(self.active_signals),
                'by_direction': {'buy': 0, 'sell': 0},
                'by_timeframe': {},
                'by_source': {},
                'by_strategy': {},
                'avg_confidence': 0,
                'history_count': len(self.signal_history)
            }
            
            # Count active signals
            active_signals = self.get_active_signals()
            
            if not active_signals:
                stats['avg_confidence'] = 0
                return stats
                
            # Calculate stats
            total_confidence = 0
            
            for signal in active_signals:
                # Direction counts
                dir_key = signal.direction.value if isinstance(signal.direction, SignalDirection) else signal.direction
                stats['by_direction'][dir_key] = stats['by_direction'].get(dir_key, 0) + 1
                
                # Timeframe counts
                tf_key = signal.timeframe.value if isinstance(signal.timeframe, SignalTimeframe) else signal.timeframe
                stats['by_timeframe'][tf_key] = stats['by_timeframe'].get(tf_key, 0) + 1
                
                # Source counts
                src_key = signal.source.value if isinstance(signal.source, SignalSource) else signal.source
                stats['by_source'][src_key] = stats['by_source'].get(src_key, 0) + 1
                
                # Strategy counts
                strat_key = signal.strategy_type.value if isinstance(signal.strategy_type, StrategyType) else signal.strategy_type
                stats['by_strategy'][strat_key] = stats['by_strategy'].get(strat_key, 0) + 1
                
                # Confidence sum
                total_confidence += signal.confidence
                
            # Calculate average confidence
            stats['avg_confidence'] = total_confidence / len(active_signals)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting signal stats: {str(e)}")
            return {'error': str(e)}
