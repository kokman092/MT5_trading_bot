import numpy as np
import pandas as pd
import logging
from typing import Dict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import ta

class DeepLearningEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.input_dim = 32
        self.hidden_dim = 64
        self.num_layers = 3
        self.dropout = 0.2
        self.learning_rate = 0.001
        
        # Initialize models
        self._init_models()
        
    def _init_models(self):
        """Initialize deep learning models"""
        try:
            # Price prediction model
            self.price_model = Sequential([
                LSTM(self.hidden_dim, input_shape=(None, self.input_dim), return_sequences=True),
                Dropout(self.dropout),
                LSTM(self.hidden_dim, return_sequences=True),
                Dropout(self.dropout),
                LSTM(self.hidden_dim),
                Dense(1)
            ])
            
            # Volatility prediction model
            self.volatility_model = Sequential([
                LSTM(self.hidden_dim, input_shape=(None, self.input_dim), return_sequences=True),
                Dropout(self.dropout),
                LSTM(self.hidden_dim),
                Dense(1, activation='relu')
            ])
            
            # Regime classification model
            self.regime_model = Sequential([
                LSTM(self.hidden_dim, input_shape=(None, self.input_dim), return_sequences=True),
                Dropout(self.dropout),
                LSTM(self.hidden_dim),
                Dense(3, activation='softmax')  # 3 regime classes
            ])
            
            # Compile models
            optimizer = Adam(learning_rate=self.learning_rate)
            self.price_model.compile(optimizer=optimizer, loss='mse')
            self.volatility_model.compile(optimizer=optimizer, loss='mse')
            self.regime_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            
            self.logger.info("Deep learning models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing deep learning models: {str(e)}")
            raise
            
    def _prepare_deep_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare features for deep learning models"""
        try:
            features = {}
            
            # LSTM features
            lstm_features = []
            
            # Price-based features
            lstm_features.extend([
                data['close'].values,
                data['high'].values,
                data['low'].values,
                data['open'].values,
                data['volume'].values,
                
                # Price changes
                data['close'].pct_change().values,
                data['high'].pct_change().values,
                data['low'].pct_change().values,
                
                # Moving averages
                data['close'].rolling(5).mean().values,
                data['close'].rolling(10).mean().values,
                data['close'].rolling(20).mean().values,
                
                # Volatility
                data['close'].rolling(5).std().values,
                data['close'].rolling(10).std().values,
                data['close'].rolling(20).std().values,
                
                # Price momentum
                data['close'].pct_change(5).values,
                data['close'].pct_change(10).values,
                data['close'].pct_change(20).values,
                
                # Volume momentum
                data['volume'].pct_change(5).values,
                data['volume'].pct_change(10).values,
                data['volume'].pct_change(20).values,
            ])
            
            # Technical indicators
            rsi = ta.momentum.RSIIndicator(data['close']).rsi()
            macd = ta.trend.MACD(data['close'])
            bb = ta.volatility.BollingerBands(data['close'])
            
            lstm_features.extend([
                rsi.values,
                macd.macd().values,
                macd.macd_signal().values,
                bb.bollinger_hband().values,
                bb.bollinger_mavg().values,
                bb.bollinger_lband().values,
                
                # ADX
                ta.trend.ADXIndicator(data['high'], data['low'], data['close']).adx().values,
                
                # CCI
                ta.trend.CCIIndicator(data['high'], data['low'], data['close']).cci().values,
                
                # MFI
                ta.volume.MFIIndicator(data['high'], data['low'], data['close'], data['volume']).money_flow_index().values,
                
                # ROC
                ta.momentum.ROCIndicator(data['close']).roc().values,
                
                # ATR
                ta.volatility.AverageTrueRange(data['high'], data['low'], data['close']).average_true_range().values,
                
                # OBV
                ta.volume.OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume().values
            ])
            
            # Stack features and handle NaN values
            lstm_features = np.column_stack(lstm_features)
            lstm_features = np.nan_to_num(lstm_features, nan=0)
            
            # Prepare ensemble features
            ensemble_features = np.column_stack([
                data['close'].pct_change().values,
                data['volume'].pct_change().values,
                data['close'].rolling(5).std().values,
                rsi.values,
                macd.macd().values,
                bb.bollinger_mavg().values,
                ta.trend.ADXIndicator(data['high'], data['low'], data['close']).adx().values
            ])
            ensemble_features = np.nan_to_num(ensemble_features, nan=0)
            
            # Reshape features for LSTM input
            sequence_length = min(30, len(lstm_features))
            if len(lstm_features) < sequence_length:
                # Pad with zeros if not enough data
                pad_length = sequence_length - len(lstm_features)
                lstm_features = np.pad(lstm_features, ((pad_length, 0), (0, 0)), mode='constant')
                ensemble_features = np.pad(ensemble_features, ((pad_length, 0), (0, 0)), mode='constant')
            
            # Take last sequence_length samples and reshape
            lstm_features = lstm_features[-sequence_length:]
            lstm_features = lstm_features.reshape(1, sequence_length, -1)  # Shape: (1, sequence_length, features)
            
            ensemble_features = ensemble_features[-sequence_length:]
            ensemble_features = ensemble_features.reshape(1, sequence_length, -1)  # Shape: (1, sequence_length, features)
            
            features['lstm_features'] = lstm_features
            features['ensemble_features'] = ensemble_features
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing deep features: {str(e)}")
            return {
                'lstm_features': np.zeros((1, 30, self.input_dim)),
                'ensemble_features': np.zeros((1, 7))
            } 