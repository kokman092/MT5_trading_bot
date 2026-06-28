import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from datetime import datetime, timedelta
import json

class ModelSelector:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_window = config.get('PERFORMANCE_WINDOW', 100)
        self.model_performance = {}
        self.model_weights = {}
        self.model_history = []
        
        # Model selection parameters
        self.min_confidence = config.get('MIN_CONFIDENCE', 0.6)
        self.max_drawdown = config.get('MAX_DRAWDOWN', 0.1)
        self.min_trades = config.get('MIN_TRADES', 20)
        
        # Load history if exists
        self.load_history()
        
    def update_performance(self, model_name, prediction, actual_return):
        """Update model performance metrics"""
        try:
            if model_name not in self.model_performance:
                self.model_performance[model_name] = {
                    'predictions': [],
                    'returns': [],
                    'accuracy': 0,
                    'sharpe': 0,
                    'drawdown': 0
                }
            
            # Update predictions and returns
            model_data = self.model_performance[model_name]
            model_data['predictions'].append(prediction)
            model_data['returns'].append(actual_return)
            
            # Keep only recent data
            if len(model_data['predictions']) > self.performance_window:
                model_data['predictions'] = model_data['predictions'][-self.performance_window:]
                model_data['returns'] = model_data['returns'][-self.performance_window:]
            
            # Calculate metrics
            if len(model_data['predictions']) >= self.min_trades:
                # Accuracy
                predicted_direction = np.array(model_data['predictions']) > 0
                actual_direction = np.array(model_data['returns']) > 0
                model_data['accuracy'] = accuracy_score(actual_direction, predicted_direction)
                
                # Sharpe ratio
                returns = np.array(model_data['returns'])
                model_data['sharpe'] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                
                # Maximum drawdown
                cumulative_returns = np.cumprod(1 + returns)
                rolling_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - rolling_max) / rolling_max
                model_data['drawdown'] = abs(np.min(drawdowns))
            
            # Update model weights
            self._update_weights()
            
            # Save history
            self.model_history.append({
                'timestamp': datetime.now().isoformat(),
                'model': model_name,
                'accuracy': model_data['accuracy'],
                'sharpe': model_data['sharpe'],
                'drawdown': model_data['drawdown']
            })
            
            self.save_history()
            
        except Exception as e:
            self.logger.error(f"Error updating performance: {str(e)}")
            
    def _update_weights(self):
        """Update model weights based on performance metrics"""
        try:
            total_score = 0
            for model_name, data in self.model_performance.items():
                if len(data['predictions']) >= self.min_trades:
                    # Calculate composite score
                    score = (
                        2 * data['accuracy'] +  # Emphasize accuracy
                        data['sharpe'] -        # Consider risk-adjusted returns
                        2 * data['drawdown']    # Penalize large drawdowns
                    )
                    
                    # Apply minimum performance thresholds
                    if (data['accuracy'] < 0.5 or
                        data['drawdown'] > self.max_drawdown):
                        score = 0
                    
                    self.model_weights[model_name] = max(0, score)
                    total_score += max(0, score)
            
            # Normalize weights
            if total_score > 0:
                for model_name in self.model_weights:
                    self.model_weights[model_name] /= total_score
            else:
                # Equal weights if no model meets criteria
                n_models = len(self.model_performance)
                for model_name in self.model_performance:
                    self.model_weights[model_name] = 1.0 / n_models
                    
        except Exception as e:
            self.logger.error(f"Error updating weights: {str(e)}")
            
    def select_models(self, predictions):
        """Select best performing models"""
        try:
            selected_predictions = []
            for model_name, prediction in predictions.items():
                if prediction['confidence'] >= self.min_confidence:
                    weight = self.model_weights.get(model_name, 0)
                    if weight > 0:
                        selected_predictions.append({
                            'model': model_name,
                            'prediction': prediction,
                            'weight': weight
                        })
            
            return selected_predictions
            
        except Exception as e:
            self.logger.error(f"Error selecting models: {str(e)}")
            return []
            
    def combine_predictions(self, selected_predictions):
        """Combine predictions using weighted ensemble"""
        try:
            if not selected_predictions:
                return None
            
            total_weight = sum(p['weight'] for p in selected_predictions)
            if total_weight == 0:
                return None
            
            # Normalize weights
            for pred in selected_predictions:
                pred['weight'] /= total_weight
            
            # Weighted average of predictions
            weighted_signal = sum(
                p['prediction']['signal'] * p['prediction']['confidence'] * p['weight']
                for p in selected_predictions
            )
            
            # Calculate overall confidence
            weighted_confidence = sum(
                p['prediction']['confidence'] * p['weight']
                for p in selected_predictions
            )
            
            return {
                'signal': 1 if weighted_signal > 0 else -1,
                'confidence': weighted_confidence,
                'models_used': [p['model'] for p in selected_predictions],
                'weights_used': [p['weight'] for p in selected_predictions]
            }
            
        except Exception as e:
            self.logger.error(f"Error combining predictions: {str(e)}")
            return None
            
    def get_performance_metrics(self):
        """Get performance metrics for all models"""
        try:
            metrics = {
                'model_performance': self.model_performance,
                'model_weights': self.model_weights,
                'history': self.model_history[-100:] if self.model_history else []
            }
            return metrics
        except Exception as e:
            self.logger.error(f"Error getting metrics: {str(e)}")
            return None
            
    def save_history(self):
        """Save model selection history"""
        try:
            with open('model_selection_history.json', 'w') as f:
                json.dump({
                    'model_performance': self.model_performance,
                    'model_weights': self.model_weights,
                    'history': self.model_history
                }, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving history: {str(e)}")
            
    def load_history(self):
        """Load model selection history"""
        try:
            with open('model_selection_history.json', 'r') as f:
                data = json.load(f)
                self.model_performance = data['model_performance']
                self.model_weights = data['model_weights']
                self.model_history = data['history']
        except FileNotFoundError:
            self.logger.info("No history file found. Starting fresh.")
        except Exception as e:
            self.logger.error(f"Error loading history: {str(e)}") 