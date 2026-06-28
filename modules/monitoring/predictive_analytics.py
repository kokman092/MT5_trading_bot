import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import joblib
import os

class PredictiveAnalytics:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize parameters
        self.params = {
            'anomaly_threshold': config.get('anomaly_threshold', -0.5),
            'prediction_horizon': config.get('prediction_horizon', 24),  # hours
            'model_update_interval': config.get('model_update_interval', 168),  # hours
            'min_training_samples': config.get('min_training_samples', 1000),
            'model_path': config.get('model_path', 'models/')
        }
        
        # Create models directory if it doesn't exist
        os.makedirs(self.params['model_path'], exist_ok=True)
        
        # Initialize models
        self.anomaly_detector = None
        self.forecaster = None
        self.scaler = StandardScaler()
        
        # Store last model update time
        self.last_model_update = None
        
        # Initialize prediction storage
        self.latest_predictions = {}
        self.detected_anomalies = []

    def update_models(self, historical_data: pd.DataFrame) -> None:
        """
        Update machine learning models with new data
        """
        try:
            current_time = datetime.now()
            
            # Check if models need updating
            if (self.last_model_update is None or 
                (current_time - self.last_model_update).total_seconds() / 3600 
                >= self.params['model_update_interval']):
                
                if len(historical_data) >= self.params['min_training_samples']:
                    # Update anomaly detection model
                    self._update_anomaly_detector(historical_data)
                    
                    # Update forecasting model
                    self._update_forecaster(historical_data)
                    
                    self.last_model_update = current_time
                    self.logger.info("Models updated successfully")
                else:
                    self.logger.warning("Insufficient data for model update")
                    
        except Exception as e:
            self.logger.error(f"Error updating models: {str(e)}")

    def _update_anomaly_detector(self, data: pd.DataFrame) -> None:
        """
        Update the anomaly detection model
        """
        # Prepare features for anomaly detection
        features = self._prepare_features(data)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Train Isolation Forest
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.anomaly_detector.fit(scaled_features)
        
        # Save model
        joblib.dump(self.anomaly_detector, 
                   os.path.join(self.params['model_path'], 'anomaly_detector.pkl'))
        joblib.dump(self.scaler,
                   os.path.join(self.params['model_path'], 'scaler.pkl'))

    def _update_forecaster(self, data: pd.DataFrame) -> None:
        """
        Update the forecasting model
        """
        # Prepare data for Prophet
        forecast_data = pd.DataFrame({
            'ds': data.index,
            'y': data['equity']
        })
        
        # Train Prophet model
        self.forecaster = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        self.forecaster.fit(forecast_data)
        
        # Save model
        self.forecaster.save(
            os.path.join(self.params['model_path'], 'prophet_model.json')
        )

    def detect_anomalies(self, current_metrics: Dict) -> List[Dict]:
        """
        Detect anomalies in current trading metrics
        """
        if self.anomaly_detector is None:
            return []
            
        try:
            # Prepare features from current metrics
            features = self._prepare_features_from_metrics(current_metrics)
            
            # Scale features
            scaled_features = self.scaler.transform(features.reshape(1, -1))
            
            # Detect anomalies
            anomaly_score = self.anomaly_detector.score_samples(scaled_features)[0]
            
            if anomaly_score < self.params['anomaly_threshold']:
                anomaly = {
                    'timestamp': datetime.now(),
                    'score': anomaly_score,
                    'metrics': current_metrics,
                    'type': self._classify_anomaly(current_metrics)
                }
                
                self.detected_anomalies.append(anomaly)
                return [anomaly]
                
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            
        return []

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for anomaly detection
        """
        features = []
        
        # Calculate technical indicators
        data['returns'] = data['equity'].pct_change()
        data['volatility'] = data['returns'].rolling(window=20).std()
        data['drawdown'] = (data['equity'].cummax() - data['equity']) / data['equity'].cummax()
        
        # Create feature matrix
        features = data[['returns', 'volatility', 'drawdown']].fillna(0)
        
        return features

    def _prepare_features_from_metrics(self, metrics: Dict) -> np.ndarray:
        """
        Prepare features from current metrics
        """
        features = np.array([
            metrics['trading'].get('win_rate', 0),
            metrics['risk'].get('risk_exposure', 0),
            metrics.get('drawdown', 0)
        ])
        
        return features

    def _classify_anomaly(self, metrics: Dict) -> str:
        """
        Classify the type of anomaly
        """
        if metrics.get('drawdown', 0) > self.config['alert_thresholds']['drawdown']:
            return 'EXCESSIVE_DRAWDOWN'
        elif metrics['risk'].get('risk_exposure', 0) > self.config['alert_thresholds']['risk_exposure']:
            return 'HIGH_RISK_EXPOSURE'
        elif metrics['trading'].get('consecutive_losses', 0) >= self.config['alert_thresholds']['consecutive_losses']:
            return 'CONSECUTIVE_LOSSES'
        else:
            return 'UNKNOWN_ANOMALY'

    def generate_predictions(self) -> Dict:
        """
        Generate predictions for future performance
        """
        if self.forecaster is None:
            return {}
            
        try:
            # Create future dataframe
            future = self.forecaster.make_future_dataframe(
                periods=self.params['prediction_horizon'],
                freq='H'
            )
            
            # Generate forecast
            forecast = self.forecaster.predict(future)
            
            # Extract relevant predictions
            predictions = {
                'equity_forecast': {
                    'timestamp': forecast['ds'].iloc[-1],
                    'predicted_value': forecast['yhat'].iloc[-1],
                    'lower_bound': forecast['yhat_lower'].iloc[-1],
                    'upper_bound': forecast['yhat_upper'].iloc[-1]
                },
                'trend': forecast['trend'].iloc[-1],
                'weekly_seasonality': self.forecaster.weekly_seasonality,
                'yearly_seasonality': self.forecaster.yearly_seasonality
            }
            
            self.latest_predictions = predictions
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            return {}

    def get_anomaly_history(self, 
                          start_time: datetime = None,
                          end_time: datetime = None) -> List[Dict]:
        """
        Get historical anomalies within the specified time range
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=7)
        if end_time is None:
            end_time = datetime.now()
            
        return [
            anomaly for anomaly in self.detected_anomalies
            if start_time <= anomaly['timestamp'] <= end_time
        ]

    def get_latest_predictions(self) -> Dict:
        """
        Get the most recent predictions
        """
        return self.latest_predictions

    def get_model_status(self) -> Dict:
        """
        Get the current status of the predictive models
        """
        return {
            'last_update': self.last_model_update,
            'anomaly_detector_ready': self.anomaly_detector is not None,
            'forecaster_ready': self.forecaster is not None,
            'total_anomalies_detected': len(self.detected_anomalies)
        }
