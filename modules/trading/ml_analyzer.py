import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import joblib
import os
import json
import asyncio
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor

class MLAnalyzer:
    def __init__(self, config: Dict):
        """Initialize ML analyzer with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validate ML configuration
        self._validate_ml_config()
        
        # Initialize ML models
        self.models = {}
        self.model_performances = {}
        self.feature_importances = {}
        self.last_retrain_time = {}
        self.prediction_history = {}
        self.ensemble_weights = {}
        
        # Initialize data preprocessing
        self.scalers = {}
        self.feature_selectors = {}
        
        # Model file path
        self.models_dir = Path('data/models')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model hyperparameters
        self.hyperparameters = self.config['machine_learning'].get('hyperparameters', {})
        
        # Training settings
        self.min_training_samples = self.config['machine_learning'].get('training', {}).get('min_samples', 1000)
        self.validation_size = self.config['machine_learning'].get('training', {}).get('validation_size', 0.2)
        self.retrain_interval_hours = self.config['machine_learning'].get('training', {}).get('retrain_interval_hours', 24)
        
        # Feature engineering settings
        self.feature_engineering = self.config['machine_learning'].get('feature_engineering', {})
        self.use_feature_selection = self.feature_engineering.get('feature_selection', {}).get('enabled', True)
        self.n_features_to_select = self.feature_engineering.get('feature_selection', {}).get('n_features', 20)
        
        # Prediction thresholds (conservative by default)
        self.prediction_threshold = self.config['machine_learning'].get('prediction', {}).get('confidence_threshold', 0.65)
        self.ensemble_threshold = self.config['machine_learning'].get('prediction', {}).get('ensemble_threshold', 0.7)
        
        # Initialize model types
        self.model_types = {
            'random_forest': {
                'class': RandomForestClassifier,
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'class': GradientBoostingClassifier,
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'xgboost': {
                'class': xgb.XGBClassifier,
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'lightgbm': {
                'class': lgb.LGBMClassifier,
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            }
        }
        
        # Metrics tracking
        self.metrics_history = []
        self.tracking_enabled = self.config['machine_learning'].get('tracking', {}).get('enabled', True)
        self.metrics_file = 'data/ml_metrics_history.json'
        
        # Performance monitoring
        self.execution_times = {}
        self.batch_size = self.config['machine_learning'].get('performance', {}).get('batch_size', 1000)
        self.parallel_training = self.config['machine_learning'].get('performance', {}).get('parallel_training', True)
        self.max_workers = self.config['machine_learning'].get('performance', {}).get('max_workers', 4)
        
        # Adaptive model selection
        self.use_adaptive_selection = self.config['machine_learning'].get('adaptive', {}).get('enabled', True)
        self.adaptation_window = self.config['machine_learning'].get('adaptive', {}).get('window_size', 100)
        self.adaptation_metrics = self.config['machine_learning'].get('adaptive', {}).get('metrics', 'f1')
        
        # Model warmup mode
        self.warmup_enabled = True
        self.min_predictions_before_live = self.config['machine_learning'].get('warmup', {}).get('min_predictions', 50)
        
        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Symbol-specific parameters
        self.symbol_params = {}
        
        # Load existing models if available
        self._load_existing_models()
        
    async def initialize(self) -> bool:
        """Initialize ML analyzer state"""
        try:
            # Load metrics history if available
            self._load_metrics_history()
            
            # Initialize symbol-specific parameters
            for symbol in self.config['trading']['symbols']:
                timeframes = self.config['trading']['timeframes']
                for timeframe in timeframes:
                    symbol_tf = f"{symbol}_{timeframe}"
                    
                    # Set up symbol parameters if not already present
                    if symbol_tf not in self.symbol_params:
                        self.symbol_params[symbol_tf] = {
                            'prediction_history': [],
                            'accuracy': 0.0,
                            'f1_score': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'last_training_time': None,
                            'samples_since_training': 0,
                            'correct_predictions': 0,
                            'total_predictions': 0
                        }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing ML analyzer: {str(e)}")
            return False
            
    def _load_metrics_history(self):
        """Load metrics history from file"""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    self.metrics_history = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading metrics history: {str(e)}")
            self.metrics_history = []
            
    def _save_metrics_history(self):
        """Save metrics history to file"""
        try:
            # Keep history size bounded
            max_history = 1000
            if len(self.metrics_history) > max_history:
                self.metrics_history = self.metrics_history[-max_history:]
                
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving metrics history: {str(e)}")
    
    def _validate_ml_config(self):
        """Validate machine learning configuration"""
        required_sections = ['machine_learning', 'market_analysis']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
                
        ml_config = self.config['machine_learning']
        required_ml_sections = ['enabled', 'models', 'training', 'prediction']
        for section in required_ml_sections:
            if section not in ml_config:
                raise ValueError(f"Missing required ML config section: {section}")
                
        if not ml_config['enabled']:
            self.logger.warning("Machine learning is disabled in config")
            
    def _load_existing_models(self):
        """Load existing models if available"""
        try:
            if not self.models_dir.exists():
                return
                
            # Load models for each symbol and timeframe
            for symbol in self.config['trading']['symbols']:
                timeframes = self.config['trading']['timeframes']
                for timeframe in timeframes:
                    symbol_tf = f"{symbol}_{timeframe}"
                    model_file = self.models_dir / f"{symbol_tf}_model.joblib"
                    
                    if model_file.exists():
                        try:
                            model_data = joblib.load(model_file)
                            self.models[symbol_tf] = model_data['models']
                            self.scalers[symbol_tf] = model_data.get('scaler')
                            self.feature_selectors[symbol_tf] = model_data.get('feature_selector')
                            self.feature_importances[symbol_tf] = model_data.get('feature_importances', {})
                            self.model_performances[symbol_tf] = model_data.get('performances', {})
                            self.ensemble_weights[symbol_tf] = model_data.get('ensemble_weights', {})
                            
                            # Initialize symbol parameters if needed
                            if symbol_tf not in self.symbol_params:
                                self.symbol_params[symbol_tf] = {}
                                
                            self.symbol_params[symbol_tf]['last_training_time'] = model_data.get('last_training_time')
                            
                            self.logger.info(f"Loaded existing model for {symbol_tf}")
                        except Exception as e:
                            self.logger.error(f"Error loading model for {symbol_tf}: {str(e)}")
                            
        except Exception as e:
            self.logger.error(f"Error loading existing models: {str(e)}")
            
    async def preprocess_data(self, data: pd.DataFrame, symbol_tf: str, is_training: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """Preprocess data for machine learning"""
        try:
            start_time = time.time()
            
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Remove any infinite or NaN values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            if df.empty:
                self.logger.warning(f"Empty dataframe after removing NaN values for {symbol_tf}")
                return pd.DataFrame(), []
                
            # Sanitize column names to avoid special JSON characters that crash LightGBM
            import re
            df.columns = [re.sub(r'[ \t\n\r\f\v,:{}""\[\]]', '_', col) for col in df.columns]
            
            # Extract features and target
            feature_cols = [col for col in df.columns if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'target']]
            
            if not feature_cols:
                self.logger.error(f"No feature columns found in data for {symbol_tf}")
                return pd.DataFrame(), []
                
            # Feature scaling
            if is_training:
                # Create new scaler during training
                scaler = StandardScaler()
                df[feature_cols] = scaler.fit_transform(df[feature_cols])
                self.scalers[symbol_tf] = scaler
            elif symbol_tf in self.scalers:
                # Use existing scaler for prediction
                df[feature_cols] = self.scalers[symbol_tf].transform(df[feature_cols])
            else:
                self.logger.warning(f"No scaler found for {symbol_tf}, cannot preprocess data")
                return pd.DataFrame(), []
                
            # Feature selection (only during training)
            if is_training and self.use_feature_selection and 'target' in df.columns:
                X = df[feature_cols]
                y = df['target']
                
                # Select best features
                selector = SelectKBest(f_classif, k=min(self.n_features_to_select, len(feature_cols)))
                selector.fit(X, y)
                
                # Get selected feature indices and feature names
                selected_indices = selector.get_support(indices=True)
                selected_features = [feature_cols[i] for i in selected_indices]
                
                # Store feature selector and selected features
                self.feature_selectors[symbol_tf] = selector
                
                # Calculate and store feature importances
                importances = selector.scores_
                feature_importances = dict(zip(feature_cols, importances))
                sorted_importances = {k: v for k, v in sorted(
                    feature_importances.items(), key=lambda item: item[1], reverse=True
                )}
                self.feature_importances[symbol_tf] = sorted_importances
                
                # Use only selected features
                feature_cols = selected_features
                
            # If we have a feature selector from previous training
            elif not is_training and symbol_tf in self.feature_selectors and 'target' not in df.columns:
                # Get selected feature indices and feature names from previous training
                selector = self.feature_selectors[symbol_tf]
                selected_indices = selector.get_support(indices=True)
                original_features = [col for col in df.columns if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'target']]
                selected_features = [original_features[i] for i in selected_indices if i < len(original_features)]
                feature_cols = selected_features
            
            # Record execution time
            end_time = time.time()
            self.execution_times['preprocess'] = end_time - start_time
            
            return df, feature_cols
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            return pd.DataFrame(), []
            
    async def train_model(self, data: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """Train machine learning model"""
        try:
            symbol_tf = f"{symbol}_{timeframe}"
            start_time = time.time()
            
            # Check if we have enough data
            if len(data) < self.min_training_samples:
                self.logger.warning(f"Not enough data to train model for {symbol_tf}. Need {self.min_training_samples}, got {len(data)}.")
                return False
                
            # Ensure we have target column
            if 'target' not in data.columns:
                self.logger.error(f"No target column in training data for {symbol_tf}")
                return False
                
            # Preprocess data
            processed_data, feature_cols = await self.preprocess_data(data, symbol_tf, is_training=True)
            
            if processed_data.empty or not feature_cols:
                self.logger.error(f"Failed to preprocess data for {symbol_tf}")
                return False
                
            # Extract features and target
            X = processed_data[feature_cols]
            y = processed_data['target']
            
            # Split data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_size, shuffle=False
            )
            
            # Get models to train based on configuration
            model_types_to_train = self.config['machine_learning']['models']
            
            # Create and train models
            models = {}
            performances = {}
            
            # Prepare training tasks
            training_tasks = []
            
            for model_type in model_types_to_train:
                if model_type not in self.model_types:
                    self.logger.warning(f"Unknown model type {model_type}, skipping")
                    continue
                    
                model_info = self.model_types[model_type]
                model_class = model_info['class']
                
                # Get default params and update with custom params from config
                params = model_info['default_params'].copy()
                custom_params = self.hyperparameters.get(model_type, {})
                params.update(custom_params)
                
                # Create model instance
                model = model_class(**params)
                
                # Add training task
                if self.parallel_training:
                    task = (model, model_type, X_train, y_train, X_val, y_val)
                    training_tasks.append(task)
                else:
                    # Train and evaluate model
                    model.fit(X_train, y_train)
                    
                    # Evaluate model
                    y_pred = model.predict(X_val)
                    performance = self._evaluate_model(y_val, y_pred)
                    
                    # Store model and performance
                    models[model_type] = model
                    performances[model_type] = performance
                    
                    self.logger.info(f"Trained {model_type} model for {symbol_tf} - accuracy: {performance['accuracy']:.4f}")
            
            # Execute parallel training if enabled
            if self.parallel_training and training_tasks:
                results = list(self.thread_pool.map(self._train_and_evaluate_model, training_tasks))
                
                for model_type, model, performance in results:
                    models[model_type] = model
                    performances[model_type] = performance
                    self.logger.info(f"Trained {model_type} model for {symbol_tf} - accuracy: {performance['accuracy']:.4f}")
            
            # Calculate ensemble weights based on model performance
            ensemble_weights = self._calculate_ensemble_weights(performances)
            
            # Store models and related info
            self.models[symbol_tf] = models
            self.model_performances[symbol_tf] = performances
            self.ensemble_weights[symbol_tf] = ensemble_weights
            
            # Get best model based on F1 score
            best_model_type = max(performances.keys(), key=lambda k: performances[k]['f1_score'])
            best_performance = performances[best_model_type]
            
            # Save model to disk
            model_data = {
                'models': models,
                'performances': performances,
                'feature_importances': self.feature_importances.get(symbol_tf, {}),
                'scaler': self.scalers.get(symbol_tf),
                'feature_selector': self.feature_selectors.get(symbol_tf),
                'ensemble_weights': ensemble_weights,
                'last_training_time': datetime.now().isoformat(),
                'feature_cols': feature_cols
            }
            
            model_file = self.models_dir / f"{symbol_tf}_model.joblib"
            joblib.dump(model_data, model_file)
            
            # Update symbol parameters
            self.symbol_params[symbol_tf]['last_training_time'] = datetime.now().isoformat()
            self.symbol_params[symbol_tf]['samples_since_training'] = 0
            self.symbol_params[symbol_tf]['accuracy'] = best_performance['accuracy']
            self.symbol_params[symbol_tf]['f1_score'] = best_performance['f1_score']
            self.symbol_params[symbol_tf]['precision'] = best_performance['precision']
            self.symbol_params[symbol_tf]['recall'] = best_performance['recall']
            
            # Record and save metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'symbol_tf': symbol_tf,
                'accuracy': best_performance['accuracy'],
                'f1_score': best_performance['f1_score'],
                'precision': best_performance['precision'],
                'recall': best_performance['recall'],
                'data_samples': len(data),
                'training_time_seconds': time.time() - start_time
            }
            
            self.metrics_history.append(metrics)
            self._save_metrics_history()
            
            # Generate training report
            if self.config['machine_learning'].get('tracking', {}).get('generate_reports', False):
                await self._generate_training_report(symbol_tf, performances, self.feature_importances.get(symbol_tf, {}))
            
            # Record execution time
            end_time = time.time()
            self.execution_times['train'] = end_time - start_time
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return False
            
    def _train_and_evaluate_model(self, task_data):
        """Helper function for parallel model training"""
        model, model_type, X_train, y_train, X_val, y_val = task_data
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_val)
            performance = self._evaluate_model(y_val, y_pred)
            
            return model_type, model, performance
            
        except Exception as e:
            self.logger.error(f"Error in parallel training for {model_type}: {str(e)}")
            return model_type, None, {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    
    def _evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        try:
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
            }
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'confusion_matrix': [[0, 0], [0, 0]]
            }
            
    def _calculate_ensemble_weights(self, performances):
        """Calculate weights for ensemble model based on performance"""
        weights = {}
        
        # Use F1 score as the basis for weights
        f1_scores = {model_type: perf['f1_score'] for model_type, perf in performances.items()}
        
        # Ensure all scores are positive
        min_score = min(f1_scores.values())
        if min_score < 0:
            adjusted_scores = {model_type: score - min_score + 0.01 for model_type, score in f1_scores.items()}
        else:
            adjusted_scores = f1_scores
            
        # Calculate total score
        total_score = sum(adjusted_scores.values())
        
        if total_score > 0:
            # Calculate normalized weights
            weights = {model_type: score / total_score for model_type, score in adjusted_scores.items()}
        else:
            # Equal weights if all scores are 0
            equal_weight = 1.0 / len(performances)
            weights = {model_type: equal_weight for model_type in performances}
            
        return weights
            
    async def predict(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Make prediction using trained model"""
        try:
            symbol_tf = f"{symbol}_{timeframe}"
            start_time = time.time()
            
            # Check if model exists
            if symbol_tf not in self.models:
                self.logger.warning(f"No trained model available for {symbol_tf}")
                return {'prediction': None, 'confidence': 0, 'signal': 'neutral'}
                
            # Get the most recent row for prediction
            latest_data = data.iloc[-1:].copy()
            
            # Preprocess data
            processed_data, feature_cols = await self.preprocess_data(latest_data, symbol_tf, is_training=False)
            
            if processed_data.empty or not feature_cols:
                self.logger.error(f"Failed to preprocess prediction data for {symbol_tf}")
                return {'prediction': None, 'confidence': 0, 'signal': 'neutral'}
                
            # Extract features
            X = processed_data[feature_cols]
            
            # Get individual model predictions and confidence scores
            predictions = {}
            confidence_scores = {}
            
            for model_type, model in self.models[symbol_tf].items():
                try:
                    # Get prediction
                    pred = model.predict(X)[0]
                    
                    # Get confidence score
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)[0]
                        confidence = proba[1] if pred == 1 else proba[0]
                    else:
                        # Fallback if predict_proba not available
                        confidence = 0.6  # Default confidence
                        
                    predictions[model_type] = pred
                    confidence_scores[model_type] = confidence
                    
                except Exception as e:
                    self.logger.error(f"Error making prediction with {model_type} model: {str(e)}")
                    predictions[model_type] = 0
                    confidence_scores[model_type] = 0
            
            # Calculate ensemble prediction and confidence
            ensemble_prediction, ensemble_confidence = self._calculate_ensemble_prediction(
                predictions, confidence_scores, self.ensemble_weights.get(symbol_tf, {})
            )
            
            # Determine signal based on prediction and confidence threshold
            signal = 'neutral'
            if ensemble_confidence >= self.ensemble_threshold:
                signal = 'buy' if ensemble_prediction == 1 else 'sell'
                
            # Update prediction history
            prediction_entry = {
                'timestamp': datetime.now().isoformat(),
                'prediction': int(ensemble_prediction),
                'confidence': float(ensemble_confidence),
                'signal': signal,
                'individual_predictions': predictions,
                'individual_confidences': confidence_scores
            }
            
            # Add to prediction history
            if symbol_tf not in self.prediction_history:
                self.prediction_history[symbol_tf] = []
                
            self.prediction_history[symbol_tf].append(prediction_entry)
            
            # Limit history size
            max_history = 1000
            if len(self.prediction_history[symbol_tf]) > max_history:
                self.prediction_history[symbol_tf] = self.prediction_history[symbol_tf][-max_history:]
                
            # Increment samples since training
            if symbol_tf in self.symbol_params:
                self.symbol_params[symbol_tf]['samples_since_training'] += 1
                
            # Check if model retraining is needed
            await self._check_if_retraining_needed(symbol_tf)
            
            # Create result dict
            result = {
                'prediction': int(ensemble_prediction),
                'confidence': float(ensemble_confidence),
                'signal': signal,
                'individual_predictions': predictions,
                'model_performances': self.model_performances.get(symbol_tf, {}),
                'warmup_mode': self._is_in_warmup_mode(symbol_tf)
            }
            
            # Record execution time
            end_time = time.time()
            self.execution_times['predict'] = end_time - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return {'prediction': None, 'confidence': 0, 'signal': 'neutral'}
            
    def _calculate_ensemble_prediction(self, predictions, confidence_scores, weights):
        """Calculate ensemble prediction by weighted voting"""
        if not predictions or not weights:
            return 0, 0
            
        weighted_votes = 0
        total_weight = 0
        
        for model_type, prediction in predictions.items():
            if model_type in weights:
                weight = weights[model_type]
                confidence = confidence_scores.get(model_type, 0.5)
                
                # Weight the vote by both model weight and prediction confidence
                vote_value = (prediction * 2 - 1)  # Convert 0/1 to -1/+1
                weighted_vote = vote_value * weight * confidence
                
                weighted_votes += weighted_vote
                total_weight += weight
                
        # Normalize the result
        if total_weight > 0:
            normalized_vote = weighted_votes / total_weight
        else:
            normalized_vote = 0
            
        # Convert to binary prediction and confidence
        ensemble_prediction = 1 if normalized_vote > 0 else 0
        ensemble_confidence = abs(normalized_vote)
        
        return ensemble_prediction, ensemble_confidence
            
    async def _check_if_retraining_needed(self, symbol_tf):
        """Check if model retraining is needed"""
        try:
            if symbol_tf not in self.symbol_params:
                return
                
            params = self.symbol_params[symbol_tf]
            last_training_time = params.get('last_training_time')
            
            if not last_training_time:
                return
                
            # Convert to datetime
            last_training = datetime.fromisoformat(last_training_time)
            hours_since_training = (datetime.now() - last_training).total_seconds() / 3600
            
            # Check if retraining interval has passed
            if hours_since_training >= self.retrain_interval_hours:
                self.logger.info(f"Retraining interval reached for {symbol_tf} ({hours_since_training:.1f} hours)")
                # Note: actual retraining should be triggered externally
                
            # Check samples since last training
            samples_threshold = self.config['machine_learning'].get('training', {}).get('retrain_samples', 5000)
            if params.get('samples_since_training', 0) >= samples_threshold:
                self.logger.info(f"Sample threshold reached for {symbol_tf}, retraining recommended")
                
        except Exception as e:
            self.logger.error(f"Error checking retraining need: {str(e)}")
            
    def feedback_actual_outcome(self, symbol: str, timeframe: str, prediction_time: datetime, actual_outcome: int):
        """Provide feedback on prediction accuracy"""
        try:
            symbol_tf = f"{symbol}_{timeframe}"
            
            if symbol_tf not in self.prediction_history:
                return
                
            # Find matching prediction
            matching_pred = None
            for pred in reversed(self.prediction_history[symbol_tf]):
                pred_time = datetime.fromisoformat(pred['timestamp'])
                if abs((pred_time - prediction_time).total_seconds()) < 60:  # Within 60 seconds
                    matching_pred = pred
                    break
                    
            if not matching_pred:
                return
                
            # Update symbol parameters with prediction outcome
            if symbol_tf in self.symbol_params:
                params = self.symbol_params[symbol_tf]
                params['total_predictions'] += 1
                
                # Check if prediction was correct
                if matching_pred['prediction'] == actual_outcome:
                    params['correct_predictions'] += 1
                    
                # Update accuracy
                if params['total_predictions'] > 0:
                    params['accuracy'] = params['correct_predictions'] / params['total_predictions']
                    
            # Update adaptive weights if enabled
            if self.use_adaptive_selection:
                self._update_adaptive_weights(symbol_tf, matching_pred, actual_outcome)
                
        except Exception as e:
            self.logger.error(f"Error processing prediction feedback: {str(e)}")
            
    def _update_adaptive_weights(self, symbol_tf, prediction, actual_outcome):
        """Update model weights adaptively based on prediction accuracy"""
        try:
            if symbol_tf not in self.ensemble_weights:
                return
                
            weights = self.ensemble_weights[symbol_tf]
            
            # Get individual model predictions
            individual_preds = prediction.get('individual_predictions', {})
            
            # Calculate adjustment factor
            adjustment_factor = 0.05  # Small adjustment per prediction
            
            for model_type, pred in individual_preds.items():
                if model_type in weights:
                    # Increase weight if prediction was correct, decrease if wrong
                    if pred == actual_outcome:
                        weights[model_type] += adjustment_factor
                    else:
                        weights[model_type] = max(0.1, weights[model_type] - adjustment_factor)
                        
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                self.ensemble_weights[symbol_tf] = {
                    model_type: weight / total_weight for model_type, weight in weights.items()
                }
                
            # Save updated weights
            model_file = self.models_dir / f"{symbol_tf}_model.joblib"
            if model_file.exists():
                model_data = joblib.load(model_file)
                model_data['ensemble_weights'] = self.ensemble_weights[symbol_tf]
                joblib.dump(model_data, model_file)
                
        except Exception as e:
            self.logger.error(f"Error updating adaptive weights: {str(e)}")
            
    def _is_in_warmup_mode(self, symbol_tf):
        """Check if the model is still in warmup mode"""
        if symbol_tf not in self.symbol_params:
            return True
            
        params = self.symbol_params[symbol_tf]
        
        # Not enough predictions yet
        if params.get('total_predictions', 0) < self.min_predictions_before_live:
            return True
            
        # Low accuracy during warmup period
        min_accuracy = self.config['machine_learning'].get('warmup', {}).get('min_accuracy', 0.55)
        if params.get('accuracy', 0) < min_accuracy:
            return True
            
        return False
        
    async def _generate_training_report(self, symbol_tf, performances, feature_importances):
        """Generate training report with visualizations"""
        try:
            reports_dir = Path('reports/ml')
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = reports_dir / f"{symbol_tf}_training_report.json"
            
            # Create report data
            report = {
                'symbol_tf': symbol_tf,
                'timestamp': datetime.now().isoformat(),
                'model_performances': performances,
                'feature_importances': {k: float(v) for k, v in list(feature_importances.items())[:20]},
                'training_params': {
                    'min_samples': self.min_training_samples,
                    'validation_size': self.validation_size,
                    'feature_selection': self.use_feature_selection,
                    'n_features': self.n_features_to_select
                },
                'ensemble_weights': self.ensemble_weights.get(symbol_tf, {})
            }
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4)
                
            # Generate visualizations if matplotlib is available
            try:
                # Plot feature importances
                if feature_importances:
                    plt.figure(figsize=(12, 8))
                    
                    # Get top 15 features
                    top_features = dict(list(feature_importances.items())[:15])
                    
                    plt.barh(list(top_features.keys()), list(top_features.values()))
                    plt.xlabel('Importance Score')
                    plt.title(f'Top 15 Feature Importances - {symbol_tf}')
                    plt.tight_layout()
                    
                    # Save plot
                    plt.savefig(reports_dir / f"{symbol_tf}_feature_importances.png")
                    plt.close()
                    
                # Plot model performances
                if performances:
                    plt.figure(figsize=(10, 6))
                    
                    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                    model_types = list(performances.keys())
                    
                    x = np.arange(len(metrics))
                    width = 0.8 / len(model_types)
                    
                    for i, model_type in enumerate(model_types):
                        perf = performances[model_type]
                        values = [perf[metric] for metric in metrics]
                        
                        plt.bar(x + i * width, values, width, label=model_type)
                        
                    plt.xlabel('Metrics')
                    plt.ylabel('Score')
                    plt.title(f'Model Performance Comparison - {symbol_tf}')
                    plt.xticks(x + width * (len(model_types) - 1) / 2, metrics)
                    plt.legend()
                    plt.tight_layout()
                    
                    # Save plot
                    plt.savefig(reports_dir / f"{symbol_tf}_model_performance.png")
                    plt.close()
                    
            except Exception as e:
                self.logger.error(f"Error generating visualizations: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error generating training report: {str(e)}")
            
    async def get_feature_importances(self, symbol: str, timeframe: str, top_n: int = 20) -> Dict:
        """Get top feature importances for a symbol/timeframe"""
        symbol_tf = f"{symbol}_{timeframe}"
        
        if symbol_tf not in self.feature_importances:
            return {}
            
        importances = self.feature_importances[symbol_tf]
        return dict(list(importances.items())[:top_n])
        
    async def get_model_performance(self, symbol: str, timeframe: str) -> Dict:
        """Get model performance metrics"""
        symbol_tf = f"{symbol}_{timeframe}"
        
        if symbol_tf not in self.model_performances:
            return {}
            
        return self.model_performances[symbol_tf]
        
    def get_execution_times(self) -> Dict:
        """Get model execution times"""
        return self.execution_times
        
    def check_model_health(self) -> Dict:
        """Check overall model health"""
        try:
            results = {
                'total_models': 0,
                'healthy_models': 0,
                'models_needing_retraining': 0,
                'models_in_warmup': 0,
                'average_accuracy': 0,
                'average_f1': 0,
                'model_health': {}
            }
            
            accuracies = []
            f1_scores = []
            
            for symbol_tf, params in self.symbol_params.items():
                # Check if model exists
                if symbol_tf not in self.models:
                    continue
                    
                results['total_models'] += 1
                
                # Get model health metrics
                accuracy = params.get('accuracy', 0)
                f1_score = params.get('f1_score', 0)
                last_training_time = params.get('last_training_time')
                
                if last_training_time:
                    last_training = datetime.fromisoformat(last_training_time)
                    hours_since_training = (datetime.now() - last_training).total_seconds() / 3600
                else:
                    hours_since_training = float('inf')
                    
                # Check warmup mode
                in_warmup = self._is_in_warmup_mode(symbol_tf)
                
                # Check if retraining needed
                needs_retraining = (
                    hours_since_training >= self.retrain_interval_hours or
                    params.get('samples_since_training', 0) >= self.config['machine_learning'].get('training', {}).get('retrain_samples', 5000)
                )
                
                # Define health status
                health_status = 'critical'
                if accuracy >= 0.65 and f1_score >= 0.65:
                    health_status = 'healthy'
                elif accuracy >= 0.6 and f1_score >= 0.6:
                    health_status = 'moderate'
                elif accuracy >= 0.55 and f1_score >= 0.55:
                    health_status = 'poor'
                    
                # Update counters
                if health_status == 'healthy':
                    results['healthy_models'] += 1
                if needs_retraining:
                    results['models_needing_retraining'] += 1
                if in_warmup:
                    results['models_in_warmup'] += 1
                    
                # Collect metrics for averages
                accuracies.append(accuracy)
                f1_scores.append(f1_score)
                
                # Add model-specific health info
                results['model_health'][symbol_tf] = {
                    'accuracy': accuracy,
                    'f1_score': f1_score,
                    'hours_since_training': hours_since_training,
                    'samples_since_training': params.get('samples_since_training', 0),
                    'health_status': health_status,
                    'needs_retraining': needs_retraining,
                    'in_warmup': in_warmup
                }
                
            # Calculate averages
            if accuracies:
                results['average_accuracy'] = sum(accuracies) / len(accuracies)
            if f1_scores:
                results['average_f1'] = sum(f1_scores) / len(f1_scores)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error checking model health: {str(e)}")
            return {'error': str(e)}
