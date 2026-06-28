import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta
import joblib

class MarketRegimeDetector:
    def __init__(self, config):
        self.config = config
        self.model_path = config.get('REGIME_MODEL_PATH', 'models/regime_model.joblib')
        self.n_regimes = config.get('N_REGIMES', 3)
        self.window_size = config.get('REGIME_WINDOW', 50)
        self.update_interval = config.get('REGIME_UPDATE_INTERVAL', 24)  # hours
        
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.last_update = None
        
        # Load or initialize model
        self.load_model()
    
    def prepare_features(self, df):
        """Prepare features for regime detection"""
        try:
            # Returns and volatility
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Volume analysis
            df['volume_ma'] = df['tick_volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
            
            # Trend strength
            df['trend_strength'] = abs(df['ema_50'] - df['ema_200']) / df['close']
            
            # Momentum
            df['momentum'] = df['returns'].rolling(window=10).mean()
            
            # Create feature matrix
            features = pd.DataFrame({
                'returns': df['returns'],
                'volatility': df['volatility'],
                'volume_ratio': df['volume_ratio'],
                'trend_strength': df['trend_strength'],
                'momentum': df['momentum']
            })
            
            features = features.ffill().fillna(0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return None
    
    def train_model(self, df):
        """Train regime detection model"""
        try:
            # Prepare features
            features = self.prepare_features(df)
            if features is None:
                return False
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Train GMM model
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42,
                n_init=10
            )
            
            self.model.fit(scaled_features)
            
            # Save model
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler
            }, self.model_path)
            
            self.last_update = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"Error training regime model: {str(e)}")
            return False
    
    def detect_regime(self, df):
        """Detect current market regime"""
        try:
            # Check if model needs update
            if (self.last_update is None or 
                datetime.now() - self.last_update > timedelta(hours=self.update_interval)):
                self.train_model(df)
            
            # Prepare features
            features = self.prepare_features(df.tail(self.window_size))
            if features is None:
                return None
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Get regime probabilities
            regime_probs = self.model.predict_proba(scaled_features)
            current_regime = self.model.predict(scaled_features)
            
            # Analyze regime characteristics
            regime_stats = self.analyze_regime(df.tail(self.window_size), current_regime[-1])
            
            return {
                'regime': int(current_regime[-1]),
                'probabilities': regime_probs[-1].tolist(),
                'characteristics': regime_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting regime: {str(e)}")
            return None
    
    def analyze_regime(self, df, regime):
        """Analyze characteristics of current regime"""
        try:
            # Calculate regime statistics
            returns = df['returns'].iloc[-20:]  # Last 20 periods
            
            stats = {
                'mean_return': float(returns.mean()),
                'volatility': float(returns.std()),
                'skewness': float(returns.skew()),
                'kurtosis': float(returns.kurtosis()),
                'trend_direction': 'up' if df['ema_50'].iloc[-1] > df['ema_200'].iloc[-1] else 'down',
                'volume_profile': 'high' if df['volume_ratio'].iloc[-1] > 1.5 else 'low',
                'regime_type': self.classify_regime(returns)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime: {str(e)}")
            return None
    
    def classify_regime(self, returns):
        """Classify regime type based on return characteristics"""
        try:
            mean_return = returns.mean()
            volatility = returns.std()
            
            if volatility > 0.015:  # High volatility threshold
                if mean_return > 0:
                    return 'volatile_bullish'
                else:
                    return 'volatile_bearish'
            elif abs(mean_return) < 0.001:  # Low return threshold
                return 'ranging'
            elif mean_return > 0:
                return 'trending_bullish'
            else:
                return 'trending_bearish'
                
        except Exception as e:
            self.logger.error(f"Error classifying regime: {str(e)}")
            return 'unknown'
    
    def get_regime_transitions(self, df, lookback_days=30):
        """Analyze regime transitions over time"""
        try:
            features = self.prepare_features(df)
            if features is None:
                return None
            
            scaled_features = self.scaler.transform(features)
            regimes = self.model.predict(scaled_features)
            
            # Calculate transition matrix
            transitions = pd.DataFrame({
                'date': df.index,
                'regime': regimes
            })
            
            # Count regime changes
            regime_changes = (transitions['regime'] != transitions['regime'].shift(1)).sum()
            
            # Calculate regime durations
            regime_durations = []
            current_regime = regimes[0]
            current_duration = 1
            
            for regime in regimes[1:]:
                if regime == current_regime:
                    current_duration += 1
                else:
                    regime_durations.append(current_duration)
                    current_regime = regime
                    current_duration = 1
            
            regime_durations.append(current_duration)
            
            return {
                'regime_changes': int(regime_changes),
                'avg_duration': float(np.mean(regime_durations)),
                'regime_distribution': pd.Series(regimes).value_counts().to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime transitions: {str(e)}")
            return None
    
    def load_model(self):
        """Load saved regime detection model"""
        try:
            if os.path.exists(self.model_path):
                saved_model = joblib.load(self.model_path)
                self.model = saved_model['model']
                self.scaler = saved_model['scaler']
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error loading regime model: {str(e)}")
            return False
    
    def get_regime_summary(self):
        """Get summary of current regime detection model"""
        if self.model:
            return {
                'n_regimes': self.n_regimes,
                'n_features': self.model.means_.shape[1],
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'model_type': type(self.model).__name__
            }
        return None
