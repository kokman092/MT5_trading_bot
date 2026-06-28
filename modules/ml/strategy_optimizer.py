from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import optuna
from ..deployment.error_handler import ErrorHandler

class StrategyOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('strategy_optimizer')
        self.model = None
        self.scaler = StandardScaler()
        self.optimization_history = []
        self.best_params = {}
        
    async def optimize_strategy(self, trade_history: pd.DataFrame) -> Dict:
        """Optimize strategy parameters using machine learning"""
        try:
            # Prepare data
            X, y = await self._prepare_training_data(trade_history)
            if len(X) == 0:
                return {}
                
            # Train model
            await self._train_model(X, y)
            
            # Optimize parameters
            best_params = await self._optimize_parameters(X, y)
            
            # Store results
            self.best_params = best_params
            self.optimization_history.append({
                'params': best_params,
                'timestamp': datetime.now()
            })
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Strategy optimization error: {str(e)}")
            return {}
            
    async def _prepare_training_data(self, trade_history: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        try:
            if trade_history.empty:
                return np.array([]), np.array([])
                
            # Extract features
            features = []
            for _, trade in trade_history.iterrows():
                trade_features = [
                    trade['volume'],
                    trade['price'],
                    trade['profit'],
                    self._extract_time_features(trade['time']),
                    self._calculate_market_volatility(trade),
                    self._get_market_condition(trade)
                ]
                features.append(trade_features)
                
            X = np.array(features)
            
            # Create labels (1 for profitable trades, 0 for losing trades)
            y = (trade_history['profit'] > 0).astype(int).values
            
            # Scale features
            X = self.scaler.fit_transform(X)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Data preparation error: {str(e)}")
            return np.array([]), np.array([])
            
    async def _train_model(self, X: np.ndarray, y: np.ndarray):
        """Train the machine learning model"""
        try:
            # Initialize model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            
            # Train model
            self.model.fit(X, y)
            
            # Calculate feature importance
            importance = self.model.feature_importances_
            self.logger.info(f"Feature importance: {importance}")
            
        except Exception as e:
            self.logger.error(f"Model training error: {str(e)}")
            
    async def _optimize_parameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize strategy parameters using Optuna"""
        try:
            def objective(trial):
                # Define parameter space
                params = {
                    'stop_loss': trial.suggest_float('stop_loss', 0.001, 0.05),
                    'take_profit': trial.suggest_float('take_profit', 0.002, 0.1),
                    'rsi_period': trial.suggest_int('rsi_period', 5, 30),
                    'rsi_overbought': trial.suggest_int('rsi_overbought', 65, 85),
                    'rsi_oversold': trial.suggest_int('rsi_oversold', 15, 35),
                    'ma_fast_period': trial.suggest_int('ma_fast_period', 5, 50),
                    'ma_slow_period': trial.suggest_int('ma_slow_period', 10, 200)
                }
                
                # Evaluate parameters
                score = self._evaluate_parameters(params, X, y)
                return score
                
            # Create study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            
            return study.best_params
            
        except Exception as e:
            self.logger.error(f"Parameter optimization error: {str(e)}")
            return {}
            
    def _evaluate_parameters(self, params: Dict, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate strategy parameters using cross-validation"""
        try:
            # Initialize cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            # Perform cross-validation
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate score
                score = self._calculate_strategy_score(y_test, y_pred, params)
                scores.append(score)
                
            return np.mean(scores)
            
        except Exception as e:
            self.logger.error(f"Parameter evaluation error: {str(e)}")
            return 0.0
            
    def _calculate_strategy_score(self, y_true: np.ndarray, y_pred: np.ndarray, params: Dict) -> float:
        """Calculate custom strategy score"""
        try:
            # Calculate basic metrics
            accuracy = np.mean(y_true == y_pred)
            precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
            
            # Calculate risk-adjusted score
            risk_ratio = params['take_profit'] / params['stop_loss']
            
            # Combine metrics
            score = (accuracy * 0.4 + precision * 0.4 + (risk_ratio / 10) * 0.2)
            
            return score
            
        except Exception as e:
            self.logger.error(f"Score calculation error: {str(e)}")
            return 0.0
            
    def _extract_time_features(self, timestamp) -> List[float]:
        """Extract time-based features"""
        try:
            dt = pd.to_datetime(timestamp)
            return [
                np.sin(2 * np.pi * dt.hour / 24),
                np.cos(2 * np.pi * dt.hour / 24),
                dt.dayofweek / 7
            ]
        except:
            return [0.0, 0.0, 0.0]
            
    def _calculate_market_volatility(self, trade: pd.Series) -> float:
        """Calculate market volatility feature"""
        try:
            return abs(trade['profit'] / trade['volume'])
        except:
            return 0.0
            
    def _get_market_condition(self, trade: pd.Series) -> float:
        """Determine market condition feature"""
        try:
            if trade['profit'] > 0:
                return 1.0
            elif trade['profit'] < 0:
                return -1.0
            return 0.0
        except:
            return 0.0
            
    async def predict_trade_outcome(self, trade_features: np.ndarray) -> Tuple[float, Dict]:
        """Predict outcome of a potential trade"""
        try:
            if self.model is None:
                return 0.5, {}
                
            # Scale features
            scaled_features = self.scaler.transform(trade_features.reshape(1, -1))
            
            # Make prediction
            prob = self.model.predict_proba(scaled_features)[0][1]
            
            # Get feature importance for this prediction
            importance = dict(zip(
                ['volume', 'price', 'profit', 'time', 'volatility', 'market'],
                self.model.feature_importances_
            ))
            
            return prob, importance
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return 0.5, {}
            
    def get_optimization_metrics(self) -> Dict:
        """Get optimization metrics summary"""
        try:
            if not self.optimization_history:
                return {}
                
            latest_optimization = self.optimization_history[-1]
            return {
                'best_params': latest_optimization['params'],
                'optimization_count': len(self.optimization_history),
                'last_optimization': latest_optimization['timestamp'].isoformat()
            }
        except Exception as e:
            self.logger.error(f"Metrics calculation error: {str(e)}")
            return {}
