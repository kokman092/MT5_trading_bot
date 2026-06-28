import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import logging
from datetime import datetime, timedelta

class SignalPredictor:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'adx', 'di_plus', 'di_minus',  # Trend strength
            'rsi',                         # Momentum
            'macd', 'macd_signal',         # Trend
            'atr',                         # Volatility
            'bb_width', 'bb_position',     # Mean reversion
            'hour', 'day_of_week'          # Time features
        ]
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature set for ML model"""
        try:
            features = pd.DataFrame()
            
            # Technical indicators (already calculated)
            for col in ['adx', 'di_plus', 'di_minus', 'rsi', 'macd', 'macd_signal', 'atr']:
                if col in df.columns:
                    features[col] = df[col]
                    
            # Calculate Bollinger Band features
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                features['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                features['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                
            # Time features
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            
            features = features.ffill().fillna(0)
            
            return features
            
        except Exception as e:
            logging.error(f"Feature preparation error: {str(e)}")
            return None
            
    def prepare_labels(self, df: pd.DataFrame, lookforward_period: int = 12) -> pd.Series:
        """Prepare labels for training"""
        try:
            # Calculate future returns
            future_returns = df['close'].shift(-lookforward_period) / df['close'] - 1
            
            # Create labels: 1 for profitable long, -1 for profitable short, 0 for no trade
            labels = pd.Series(0, index=df.index)
            min_return = self.config['ML_MIN_RETURN']  # Minimum return threshold
            
            labels[future_returns > min_return] = 1     # Long signals
            labels[future_returns < -min_return] = -1   # Short signals
            
            return labels
            
        except Exception as e:
            logging.error(f"Label preparation error: {str(e)}")
            return None
            
    def train(self, df: pd.DataFrame) -> bool:
        """Train the ML model"""
        try:
            # Prepare features and labels
            features = self.prepare_features(df)
            labels = self.prepare_labels(df)
            
            if features is None or labels is None:
                return False
                
            # Remove rows with NaN values
            valid_idx = ~(features.isna().any(axis=1) | labels.isna())
            features = features[valid_idx]
            labels = labels[valid_idx]
            
            if len(features) < 1000:  # Minimum required samples
                logging.warning("Insufficient training data")
                return False
                
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Initialize and train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            for train_idx, val_idx in tscv.split(scaled_features):
                X_train = scaled_features[train_idx]
                y_train = labels.iloc[train_idx]
                
                if len(np.unique(y_train)) < 2:
                    continue
                    
                self.model.fit(X_train, y_train)
                
            return True
            
        except Exception as e:
            logging.error(f"Model training error: {str(e)}")
            return False
            
    def predict(self, df: pd.DataFrame) -> Optional[int]:
        """Predict trading signal"""
        try:
            if self.model is None:
                return None
                
            # Prepare features
            features = self.prepare_features(df.iloc[-1:])
            if features is None or features.empty:
                return None
                
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Get prediction and probabilities
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
            
            # Only return prediction if probability is high enough
            max_prob = max(probabilities)
            if max_prob > self.config['ML_MIN_PROBABILITY']:
                return prediction
                
            return 0  # No trade if probability is too low
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return None
            
    def save_model(self, path: str) -> bool:
        """Save the trained model"""
        try:
            if self.model is None:
                return False
                
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            joblib.dump(model_data, path)
            return True
            
        except Exception as e:
            logging.error(f"Model save error: {str(e)}")
            return False
            
    def load_model(self, path: str) -> bool:
        """Load a trained model"""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            return True
            
        except Exception as e:
            logging.error(f"Model load error: {str(e)}")
            return False
