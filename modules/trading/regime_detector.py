import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class MarketRegime:
    regime_type: str  # 'trending', 'ranging', 'volatile'
    confidence: float
    direction: Optional[str]  # 'up', 'down', None
    volatility_level: str  # 'low', 'medium', 'high'
    start_time: datetime
    
class RegimeDetector:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.scaler = StandardScaler()
        self.hmm_model = None
        self.gmm_model = None
        self.kmeans_model = None
        self.history = []
        
    def initialize_models(self, historical_data: pd.DataFrame):
        """Initialize and train all models"""
        try:
            features = self._extract_features(historical_data)
            scaled_features = self.scaler.fit_transform(features)
            
            # Initialize HMM
            self.hmm_model = hmm.GaussianHMM(
                n_components=3,
                covariance_type="full",
                n_iter=100
            )
            self.hmm_model.fit(scaled_features)
            
            # Initialize GMM
            self.gmm_model = GaussianMixture(
                n_components=3,
                random_state=42
            )
            self.gmm_model.fit(scaled_features)
            
            # Initialize K-means
            self.kmeans_model = KMeans(
                n_clusters=3,
                random_state=42
            )
            self.kmeans_model.fit(scaled_features)
            
            self.logger.info("Successfully initialized all regime detection models")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing regime detection models: {str(e)}")
            return False
            
    def detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime using ensemble of models"""
        try:
            features = self._extract_features(market_data)
            scaled_features = self.scaler.transform(features)
            
            # Get predictions from all models
            hmm_regime = self.hmm_model.predict(scaled_features)[-1]
            gmm_regime = self.gmm_model.predict(scaled_features)[-1]
            kmeans_regime = self.kmeans_model.predict(scaled_features)[-1]
            
            # Ensemble voting
            regime_votes = [hmm_regime, gmm_regime, kmeans_regime]
            final_regime = max(set(regime_votes), key=regime_votes.count)
            
            # Calculate confidence
            confidence = regime_votes.count(final_regime) / len(regime_votes)
            
            # Determine regime characteristics
            regime_type = self._classify_regime_type(features.iloc[-1])
            direction = self._determine_direction(market_data)
            volatility_level = self._classify_volatility(market_data)
            
            regime = MarketRegime(
                regime_type=regime_type,
                confidence=confidence,
                direction=direction,
                volatility_level=volatility_level,
                start_time=market_data.index[-1]
            )
            
            self.history.append(regime)
            return regime
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return None
            
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract relevant features for regime detection"""
        features = pd.DataFrame()
        
        # Trend features
        features['sma_ratio'] = data['close'] / data['close'].rolling(20).mean()
        features['price_momentum'] = data['close'].pct_change(5)
        
        # Volatility features
        features['volatility'] = data['close'].rolling(20).std()
        features['atr'] = self._calculate_atr(data)
        
        # Volume features
        if 'volume' in data.columns:
            features['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Additional technical features
        features['rsi'] = self._calculate_rsi(data['close'])
        features['bb_width'] = self._calculate_bollinger_bandwidth(data['close'])
        
        return features.dropna()
        
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def _calculate_bollinger_bandwidth(self, prices: pd.Series, 
                                     period: int = 20, std_dev: int = 2) -> pd.Series:
        """Calculate Bollinger Bandwidth"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        bandwidth = (upper_band - lower_band) / sma
        return bandwidth
        
    def _classify_regime_type(self, features: pd.Series) -> str:
        """Classify the type of market regime"""
        if features['volatility'] > self.config['regime_detection']['thresholds']['volatile_threshold']:
            return 'volatile'
        elif abs(features['sma_ratio'] - 1) > self.config['regime_detection']['thresholds']['trending_threshold']:
            return 'trending'
        else:
            return 'ranging'
            
    def _determine_direction(self, data: pd.DataFrame) -> Optional[str]:
        """Determine market direction"""
        if len(data) < 2:
            return None
            
        short_ma = data['close'].rolling(5).mean()
        long_ma = data['close'].rolling(20).mean()
        
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            return 'up'
        elif short_ma.iloc[-1] < long_ma.iloc[-1]:
            return 'down'
        else:
            return None
            
    def _classify_volatility(self, data: pd.DataFrame) -> str:
        """Classify volatility level"""
        vol = data['close'].rolling(20).std()
        current_vol = vol.iloc[-1]
        
        if current_vol < self.config['regime_detection']['thresholds']['volatility']['low_threshold']:
            return 'low'
        elif current_vol > self.config['regime_detection']['thresholds']['volatility']['high_threshold']:
            return 'high'
        else:
            return 'medium'
            
    def get_regime_history(self, lookback_days: int = 30) -> List[MarketRegime]:
        """Get history of regime changes"""
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        return [regime for regime in self.history if regime.start_time >= cutoff_time] 