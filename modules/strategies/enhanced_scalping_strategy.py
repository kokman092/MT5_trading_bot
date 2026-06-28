import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import tensorflow as tf
from datetime import datetime, time
from ..indicators.advanced_indicators import AdvancedIndicators
from ..market.regime_detector import MarketRegimeDetector

class EnhancedScalpingStrategy:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = AdvancedIndicators()
        self.regime_detector = MarketRegimeDetector(config)
        
        # Strategy parameters
        self.params = {
            'bb_window': config.get('bb_window', 20),
            'bb_std': config.get('bb_std', 2.0),
            'rsi_period': config.get('rsi_period', 14),
            'min_volatility': config.get('min_volatility', 0.0002),
            'max_volatility': config.get('max_volatility', 0.002),
            'profit_target': config.get('profit_target', 0.001),  # 0.1%
            'stop_loss': config.get('stop_loss', 0.0005),        # 0.05%
            'session_times': {
                'london': {'start': time(8, 0), 'end': time(16, 30)},
                'new_york': {'start': time(13, 30), 'end': time(20, 0)}
            }
        }
        
        # Initialize ML model
        self.ml_model = self._initialize_ml_model()

    def _initialize_ml_model(self) -> tf.keras.Model:
        """
        Initialize ML model for pattern recognition
        """
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, input_shape=(10, 5)),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def analyze_market(self, data: pd.DataFrame, current_time: datetime) -> Dict:
        """
        Enhanced scalping analysis with ML pattern recognition and session filtering
        """
        try:
            # Check if current time is within trading sessions
            if not self._is_valid_session(current_time):
                return {'signal': 0, 'strength': 0}
            
            # Get market regime
            regime = self.regime_detector.detect_regime(data)
            
            # Calculate indicators
            bb_upper, bb_middle, bb_lower = self.indicators.calculate_dynamic_bollinger(
                data['close'],
                window=self.params['bb_window'],
                num_std=self.params['bb_std']
            )
            
            # Calculate volatility using ATR
            atr = self.indicators.calculate_atr(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=self.params['bb_window']
            )
            current_volatility = atr.iloc[-1] / data['close'].iloc[-1]
            
            # Skip if volatility is outside acceptable range
            if not self._check_volatility(current_volatility):
                return {'signal': 0, 'strength': 0}
            
            # Prepare data for ML model
            ml_features = self._prepare_ml_features(data)
            pattern_probability = self.ml_model.predict(ml_features)[0][0]
            
            # Calculate combined signals
            signals = self._calculate_signals(
                data, bb_upper, bb_middle, bb_lower, pattern_probability
            )
            
            return {
                'signal': signals['signal'],
                'strength': signals['strength'],
                'pattern_probability': pattern_probability,
                'regime': regime,
                'current_volatility': current_volatility
            }
            
        except Exception as e:
            self.logger.error(f"Error in scalping analysis: {str(e)}")
            return None

    def _is_valid_session(self, current_time: datetime) -> bool:
        """
        Check if current time is within valid trading sessions
        """
        current_time = current_time.time()
        
        # Check London session
        in_london = (
            self.params['session_times']['london']['start'] <= current_time <=
            self.params['session_times']['london']['end']
        )
        
        # Check New York session
        in_newyork = (
            self.params['session_times']['new_york']['start'] <= current_time <=
            self.params['session_times']['new_york']['end']
        )
        
        return in_london or in_newyork

    def _check_volatility(self, volatility: float) -> bool:
        """
        Check if volatility is within acceptable range
        """
        return (
            self.params['min_volatility'] <= volatility <= 
            self.params['max_volatility']
        )

    def _prepare_ml_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for ML model
        """
        # Calculate technical features
        features = pd.DataFrame()
        features['returns'] = data['close'].pct_change()
        features['volume_ma_ratio'] = (
            data['volume'] / data['volume'].rolling(20).mean()
        )
        features['price_ma_ratio'] = (
            data['close'] / data['close'].rolling(20).mean()
        )
        features['high_low_ratio'] = (
            (data['high'] - data['low']) / data['close']
        )
        features['close_open_ratio'] = (
            (data['close'] - data['open']) / data['open']
        )
        
        # Prepare sequence data
        sequence_length = 10
        sequences = []
        for i in range(len(features) - sequence_length + 1):
            sequences.append(features.iloc[i:i+sequence_length].values)
            
        return np.array(sequences)

    def _calculate_signals(self, data: pd.DataFrame, bb_upper: pd.Series,
                         bb_middle: pd.Series, bb_lower: pd.Series,
                         pattern_probability: float) -> Dict:
        """
        Calculate trading signals combining technical indicators and ML predictions
        """
        current_price = data['close'].iloc[-1]
        
        # Bollinger Bands signals
        if current_price <= bb_lower.iloc[-1]:
            bb_signal = 1
        elif current_price >= bb_upper.iloc[-1]:
            bb_signal = -1
        else:
            bb_signal = 0
            
        # Candlestick pattern signals
        pattern_signal = 1 if pattern_probability > 0.7 else (
            -1 if pattern_probability < 0.3 else 0
        )
        
        # Combine signals
        if bb_signal == pattern_signal and bb_signal != 0:
            signal = bb_signal
            strength = pattern_probability if bb_signal == 1 else (1 - pattern_probability)
        else:
            signal = 0
            strength = 0
            
        return {
            'signal': signal,
            'strength': strength
        }

    def get_position_size(self, signal: Dict, account_size: float) -> float:
        """
        Calculate position size based on signal strength
        """
        base_size = account_size * 0.01  # 1% base risk for scalping
        return base_size * signal['strength']

    def get_stop_loss(self, entry_price: float, signal: Dict) -> float:
        """
        Calculate tight stop loss for scalping
        """
        return entry_price * (
            1 - self.params['stop_loss'] if signal['signal'] > 0
            else 1 + self.params['stop_loss']
        )

    def get_take_profit(self, entry_price: float, signal: Dict) -> float:
        """
        Calculate take profit level
        """
        return entry_price * (
            1 + self.params['profit_target'] if signal['signal'] > 0
            else 1 - self.params['profit_target']
        )

    async def generate_signals(self, market_data: Dict) -> Dict:
        """
        Generate trading signals based on market data
        
        Args:
            market_data (Dict): Dictionary containing:
                - symbol: Trading symbol
                - data: DataFrame with OHLCV data
                - timestamp: Current timestamp
                
        Returns:
            Dict: Signal information containing:
                - action: Trading action (BUY, SELL, NONE)
                - confidence: Signal confidence level
                - metadata: Additional signal information
        """
        try:
            # Analyze market conditions
            analysis = self.analyze_market(market_data['data'], market_data['timestamp'])
            if analysis is None:
                return {
                    'action': 'NONE',
                    'confidence': 0.0,
                    'metadata': {'error': 'Analysis failed'}
                }
                
            # Convert signal to action
            if analysis['signal'] > 0:
                action = 'BUY'
            elif analysis['signal'] < 0:
                action = 'SELL'
            else:
                action = 'NONE'
                
            # Calculate entry parameters if signal exists
            metadata = {
                'pattern_probability': analysis['pattern_probability'],
                'regime': analysis['regime'],
                'volatility': analysis['current_volatility']
            }
            
            if action != 'NONE':
                current_price = market_data['data']['close'].iloc[-1]
                metadata.update({
                    'entry_price': current_price,
                    'stop_loss': self.get_stop_loss(current_price, analysis),
                    'take_profit': self.get_take_profit(current_price, analysis)
                })
            
            return {
                'action': action,
                'confidence': analysis['strength'],
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error generating scalping signals: {str(e)}")
            return {
                'action': 'NONE',
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
