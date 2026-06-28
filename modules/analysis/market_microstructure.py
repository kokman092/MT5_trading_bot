import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from scipy.stats import norm
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn

class MarketMicrostructure:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize parameters
        self.params = {
            'tick_window': config.get('tick_window', 1000),
            'volume_threshold': config.get('volume_threshold', 0.95),
            'cluster_eps': config.get('cluster_eps', 0.001),
            'cluster_min_samples': config.get('cluster_min_samples', 5),
            'price_levels': config.get('price_levels', 10),
            'update_interval': config.get('update_interval', 100)  # ms
        }
        
        # Initialize models
        self._init_models()
        
        # Cache for analysis results
        self.analysis_cache = {}
        
    def _init_models(self):
        """Initialize ML models for microstructure analysis"""
        try:
            # Initialize OrderFlow Neural Network
            self.order_flow_nn = OrderFlowNN(
                input_size=self.params['tick_window'],
                hidden_size=64,
                output_size=3  # Buy, Sell, Neutral
            )
            
            # Load pre-trained models if available
            self._load_models()
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")

    def analyze_order_flow(self, tick_data: pd.DataFrame, 
                         order_book: Dict) -> Dict:
        """
        Analyze order flow patterns and market microstructure
        """
        try:
            # Extract features
            volume_profile = self._calculate_volume_profile(tick_data)
            price_impact = self._calculate_price_impact(tick_data)
            liquidity_imbalance = self._analyze_liquidity_imbalance(order_book)
            order_flow_toxicity = self._calculate_order_flow_toxicity(tick_data)
            
            # Combine analyses
            analysis = {
                'volume_profile': volume_profile,
                'price_impact': price_impact,
                'liquidity_imbalance': liquidity_imbalance,
                'order_flow_toxicity': order_flow_toxicity,
                'timestamp': datetime.now()
            }
            
            # Add predictions
            analysis['predictions'] = self._generate_predictions(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in order flow analysis: {str(e)}")
            return None

    def _calculate_volume_profile(self, tick_data: pd.DataFrame) -> Dict:
        """
        Calculate volume profile and identify key levels
        """
        try:
            # Calculate volume distribution
            volume_dist = pd.cut(tick_data['price'], 
                               bins=self.params['price_levels'])
            volume_profile = tick_data.groupby(volume_dist)['volume'].sum()
            
            # Identify significant levels using DBSCAN
            clustering = DBSCAN(
                eps=self.params['cluster_eps'],
                min_samples=self.params['cluster_min_samples']
            ).fit(volume_profile.values.reshape(-1, 1))
            
            # Extract key levels
            key_levels = []
            for cluster in set(clustering.labels_):
                if cluster != -1:  # Not noise
                    cluster_points = volume_profile.index[
                        clustering.labels_ == cluster
                    ]
                    key_levels.append({
                        'price': cluster_points.mid.mean(),
                        'volume': volume_profile[cluster_points].sum(),
                        'strength': len(cluster_points)
                    })
            
            return {
                'profile': volume_profile.to_dict(),
                'key_levels': key_levels,
                'concentration': self._calculate_volume_concentration(volume_profile)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {str(e)}")
            return None

    def _calculate_price_impact(self, tick_data: pd.DataFrame) -> Dict:
        """
        Calculate price impact of trades
        """
        try:
            # Calculate immediate price impact
            tick_data['price_change'] = tick_data['price'].diff()
            tick_data['signed_volume'] = tick_data['volume'] * \
                                       np.sign(tick_data['price_change'])
            
            # Calculate Kyle's lambda
            model = np.polyfit(tick_data['signed_volume'],
                             tick_data['price_change'], 1)
            kyle_lambda = model[0]
            
            # Calculate permanent vs temporary impact
            temp_impact = tick_data.groupby('trade_id').agg({
                'price_change': ['first', 'last']
            })
            
            return {
                'kyle_lambda': kyle_lambda,
                'temporary_impact': temp_impact['price_change']['first'].mean(),
                'permanent_impact': temp_impact['price_change']['last'].mean(),
                'impact_decay': self._calculate_impact_decay(tick_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating price impact: {str(e)}")
            return None

    def _analyze_liquidity_imbalance(self, order_book: Dict) -> Dict:
        """
        Analyze order book liquidity imbalance
        """
        try:
            # Calculate bid-ask imbalance
            bid_volume = sum(level['volume'] for level in order_book['bids'])
            ask_volume = sum(level['volume'] for level in order_book['asks'])
            
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            # Calculate depth imbalance
            bid_depth = self._calculate_depth(order_book['bids'])
            ask_depth = self._calculate_depth(order_book['asks'])
            
            return {
                'volume_imbalance': imbalance,
                'depth_imbalance': (bid_depth - ask_depth) / (bid_depth + ask_depth),
                'bid_concentration': self._calculate_concentration(order_book['bids']),
                'ask_concentration': self._calculate_concentration(order_book['asks'])
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing liquidity imbalance: {str(e)}")
            return None

    def _calculate_order_flow_toxicity(self, tick_data: pd.DataFrame) -> Dict:
        """
        Calculate order flow toxicity metrics
        """
        try:
            # Calculate VPIN (Volume-synchronized Probability of Informed Trading)
            vpin = self._calculate_vpin(tick_data)
            
            # Calculate order flow run statistics
            runs = self._calculate_order_flow_runs(tick_data)
            
            # Calculate trade initiation statistics
            initiation = self._calculate_trade_initiation(tick_data)
            
            return {
                'vpin': vpin,
                'order_flow_runs': runs,
                'trade_initiation': initiation,
                'toxicity_score': self._calculate_toxicity_score(
                    vpin, runs, initiation
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating order flow toxicity: {str(e)}")
            return None

    def _generate_predictions(self, analysis: Dict) -> Dict:
        """
        Generate predictions based on microstructure analysis
        """
        try:
            # Prepare features
            features = torch.tensor([
                analysis['volume_profile']['concentration'],
                analysis['price_impact']['kyle_lambda'],
                analysis['liquidity_imbalance']['volume_imbalance'],
                analysis['order_flow_toxicity']['toxicity_score']
            ], dtype=torch.float32)
            
            # Generate prediction
            with torch.no_grad():
                prediction = self.order_flow_nn(features)
                
            # Calculate confidence
            confidence = torch.softmax(prediction, dim=0)
            
            return {
                'direction': ['sell', 'neutral', 'buy'][prediction.argmax()],
                'confidence': confidence.max().item(),
                'probability_distribution': confidence.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            return None

class OrderFlowNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(OrderFlowNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        return self.network(x)
