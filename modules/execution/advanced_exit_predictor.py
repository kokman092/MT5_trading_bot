import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import MetaTrader5 as mt5
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
import tensorflow as tf
from collections import deque
import joblib

class AdvancedExitPredictor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # ML model parameters
        self.lookback_periods = config.get('LOOKBACK_PERIODS', {
            'M1': 100,   # 1 minute
            'M5': 100,   # 5 minutes
            'M15': 50,   # 15 minutes
            'H1': 24,    # 1 hour
            'H4': 24,    # 4 hours
            'D1': 20     # 1 day
        })
        
        # Correlation parameters
        self.correlation_threshold = config.get('CORRELATION_THRESHOLD', 0.7)
        self.correlation_lookback = config.get('CORRELATION_LOOKBACK', 100)
        
        # Exit opportunity parameters
        self.min_profit_threshold = config.get('MIN_PROFIT_THRESHOLD', 0.001)
        self.max_loss_threshold = config.get('MAX_LOSS_THRESHOLD', 0.002)
        
        # Initialize models
        self.exit_models = {}
        self.correlation_cache = {}
        self.market_state_cache = {}
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.prediction_history = []
        self.model_performance = {}
        
        # Load models and history
        self.load_models()
        self.load_history()
    
    def prepare_features(self, symbol, timeframes=None):
        """Prepare multi-timeframe features for prediction"""
        try:
            if timeframes is None:
                timeframes = ['M1', 'M5', 'M15', 'H1']
            
            features = {}
            for tf in timeframes:
                # Get MT5 timeframe constant
                mt5_tf = getattr(mt5, f'TIMEFRAME_{tf}')
                lookback = self.lookback_periods[tf]
                
                # Get historical data
                rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, lookback)
                if rates is None:
                    continue
                
                df = pd.DataFrame(rates)
                
                # Calculate technical features
                df['returns'] = df['close'].pct_change()
                df['volatility'] = df['returns'].rolling(20).std()
                df['rsi'] = self.calculate_rsi(df['close'])
                df['macd'] = self.calculate_macd(df['close'])
                df['bb_position'] = self.calculate_bollinger_position(df['close'])
                df['volume_ma_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
                
                # Store features
                features[tf] = df.iloc[-1].to_dict()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return None
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        try:
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal).mean()
            return macd - signal_line
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            return None
    
    def calculate_bollinger_position(self, prices, period=20, std_dev=2):
        """Calculate position within Bollinger Bands"""
        try:
            ma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = ma + (std * std_dev)
            lower = ma - (std * std_dev)
            return (prices - lower) / (upper - lower)
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger position: {str(e)}")
            return None
    
    def analyze_correlations(self, symbol, position_type):
        """Analyze correlations with other instruments"""
        try:
            # Get correlation symbols
            symbols = mt5.symbols_get()
            if not symbols:
                return None
            
            correlations = {}
            base_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, self.correlation_lookback)
            if base_rates is None:
                return None
            
            base_returns = pd.DataFrame(base_rates)['close'].pct_change()
            
            for sym in symbols[:20]:  # Limit to top 20 symbols
                if sym.name == symbol:
                    continue
                
                rates = mt5.copy_rates_from_pos(sym.name, mt5.TIMEFRAME_M5, 0, self.correlation_lookback)
                if rates is not None:
                    returns = pd.DataFrame(rates)['close'].pct_change()
                    corr = returns.corr(base_returns)
                    if abs(corr) > self.correlation_threshold:
                        correlations[sym.name] = corr
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {str(e)}")
            return None
    
    def train_exit_model(self, symbol):
        """Train ML model for exit prediction"""
        try:
            # Get historical data
            data = []
            labels = []
            
            # Get past trades
            trades = mt5.history_deals_get(
                datetime.now() - timedelta(days=30),
                datetime.now()
            )
            
            if not trades:
                return False
            
            # Prepare training data
            for trade in trades:
                if trade.symbol != symbol:
                    continue
                
                features = self.prepare_features(symbol)
                if not features:
                    continue
                
                # Flatten features
                flat_features = {}
                for tf, tf_features in features.items():
                    for k, v in tf_features.items():
                        flat_features[f"{tf}_{k}"] = v
                
                data.append(flat_features)
                
                # Label is 1 if profit > threshold
                labels.append(1 if trade.profit > self.min_profit_threshold else 0)
            
            if not data:
                return False
            
            # Convert to numpy arrays
            X = pd.DataFrame(data)
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            accuracy = model.score(X_test_scaled, y_test)
            
            # Save model
            self.exit_models[symbol] = {
                'model': model,
                'scaler': self.scaler,
                'accuracy': accuracy,
                'last_trained': datetime.now().isoformat()
            }
            
            self.save_models()
            return True
            
        except Exception as e:
            self.logger.error(f"Error training exit model: {str(e)}")
            return False
    
    def predict_exit_opportunity(self, symbol, position):
        """Predict if current market conditions are good for exit"""
        try:
            # Get current features
            features = self.prepare_features(symbol)
            if not features:
                return None
            
            # Flatten features
            flat_features = {}
            for tf, tf_features in features.items():
                for k, v in tf_features.items():
                    flat_features[f"{tf}_{k}"] = v
            
            # Check if model exists
            if symbol not in self.exit_models:
                if not self.train_exit_model(symbol):
                    return None
            
            # Prepare features
            X = pd.DataFrame([flat_features])
            X_scaled = self.exit_models[symbol]['scaler'].transform(X)
            
            # Get prediction and probability
            model = self.exit_models[symbol]['model']
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1]
            
            # Get correlations
            correlations = self.analyze_correlations(symbol, position.type)
            
            # Calculate exit score
            exit_score = self.calculate_exit_score(
                prediction,
                probability,
                features,
                correlations,
                position
            )
            
            result = {
                'should_exit': exit_score > 0.7,
                'exit_score': exit_score,
                'prediction': bool(prediction),
                'probability': float(probability),
                'correlations': correlations,
                'features': features
            }
            
            # Record prediction
            self.record_prediction(symbol, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting exit opportunity: {str(e)}")
            return None
    
    def calculate_exit_score(self, prediction, probability, features, correlations, position):
        """Calculate comprehensive exit score"""
        try:
            score = 0.0
            weights = {
                'ml_prediction': 0.4,
                'technical': 0.3,
                'correlation': 0.2,
                'market_condition': 0.1
            }
            
            # ML prediction score
            score += weights['ml_prediction'] * (prediction * probability)
            
            # Technical score
            tech_score = 0.0
            m1_features = features.get('M1', {})
            
            if position.type == 0:  # Buy position
                if m1_features.get('rsi', 50) > 70:
                    tech_score += 0.3
                if m1_features.get('macd', 0) < 0:
                    tech_score += 0.3
                if m1_features.get('bb_position', 0.5) > 0.8:
                    tech_score += 0.4
            else:  # Sell position
                if m1_features.get('rsi', 50) < 30:
                    tech_score += 0.3
                if m1_features.get('macd', 0) > 0:
                    tech_score += 0.3
                if m1_features.get('bb_position', 0.5) < 0.2:
                    tech_score += 0.4
            
            score += weights['technical'] * tech_score
            
            # Correlation score
            if correlations:
                corr_score = np.mean([abs(c) for c in correlations.values()])
                score += weights['correlation'] * corr_score
            
            # Market condition score
            market_score = 0.0
            if m1_features.get('volatility', 0) > 0.002:
                market_score += 0.3
            if m1_features.get('volume_ma_ratio', 1) > 1.2:
                market_score += 0.3
            
            score += weights['market_condition'] * market_score
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating exit score: {str(e)}")
            return 0.0
    
    def record_prediction(self, symbol, prediction_result):
        """Record prediction for performance tracking"""
        try:
            record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'prediction': prediction_result
            }
            
            self.prediction_history.append(record)
            
            # Maintain history size
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            self.save_history()
            
        except Exception as e:
            self.logger.error(f"Error recording prediction: {str(e)}")
    
    def get_prediction_analytics(self):
        """Get analytics about prediction performance"""
        try:
            if not self.prediction_history:
                return None
            
            analytics = {}
            for symbol in set(p['symbol'] for p in self.prediction_history):
                symbol_predictions = [p for p in self.prediction_history if p['symbol'] == symbol]
                
                analytics[symbol] = {
                    'total_predictions': len(symbol_predictions),
                    'exit_signals': len([p for p in symbol_predictions if p['prediction']['should_exit']]),
                    'avg_probability': np.mean([p['prediction']['probability'] for p in symbol_predictions]),
                    'avg_exit_score': np.mean([p['prediction']['exit_score'] for p in symbol_predictions])
                }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error getting prediction analytics: {str(e)}")
            return None
    
    def save_models(self):
        """Save trained models"""
        try:
            model_data = {
                symbol: {
                    'scaler': joblib.dumps(model_info['scaler']).decode('latin1'),
                    'model': joblib.dumps(model_info['model']).decode('latin1'),
                    'accuracy': model_info['accuracy'],
                    'last_trained': model_info['last_trained']
                }
                for symbol, model_info in self.exit_models.items()
            }
            
            with open('exit_models.json', 'w') as f:
                json.dump(model_data, f)
                
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self):
        """Load trained models"""
        try:
            with open('exit_models.json', 'r') as f:
                model_data = json.load(f)
                
            self.exit_models = {
                symbol: {
                    'scaler': joblib.loads(model_info['scaler'].encode('latin1')),
                    'model': joblib.loads(model_info['model'].encode('latin1')),
                    'accuracy': model_info['accuracy'],
                    'last_trained': model_info['last_trained']
                }
                for symbol, model_info in model_data.items()
            }
                
        except FileNotFoundError:
            self.logger.info("No saved models found. Starting fresh.")
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
    
    def save_history(self):
        """Save prediction history"""
        try:
            with open('prediction_history.json', 'w') as f:
                json.dump(self.prediction_history, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving history: {str(e)}")
    
    def load_history(self):
        """Load prediction history"""
        try:
            with open('prediction_history.json', 'r') as f:
                self.prediction_history = json.load(f)
                
        except FileNotFoundError:
            self.logger.info("No prediction history found. Starting fresh.")
        except Exception as e:
            self.logger.error(f"Error loading history: {str(e)}")
            
    def scan_exit_opportunities(self, positions):
        """Scan for exit opportunities across all open positions"""
        try:
            opportunities = []
            
            for position in positions:
                # Get prediction
                prediction = self.predict_exit_opportunity(position.symbol, position)
                if not prediction:
                    continue
                
                if prediction['should_exit']:
                    opportunities.append({
                        'position': position,
                        'prediction': prediction,
                        'priority': prediction['exit_score']
                    })
            
            # Sort by priority
            opportunities.sort(key=lambda x: x['priority'], reverse=True)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error scanning exit opportunities: {str(e)}")
            return []
