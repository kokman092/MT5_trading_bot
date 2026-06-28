import logging
import numpy as np
import pandas as pd
import ta
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from .signal import Signal
import scipy.stats as stats
from collections import Counter
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from scipy.signal import find_peaks
import talib

class Strategy:
    def __init__(self, config: Dict):
        """Initialize trading strategy"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = {}
        self._initialize_indicators()
        
        # Add ensemble model support
        self.ensemble_models = {}
        self.ensemble_weights = {}
        self.signal_history = []
        self.max_signal_history = 100
        self.min_confidence_threshold = 0.65
        self.last_regime = None
        self.strategy_performance = {}
        
        # Initialize strategy types with weights
        self.strategy_weights = {
            'trend_following': 0.35,
            'mean_reversion': 0.25,
            'breakout': 0.20,
            'pattern_recognition': 0.20
        }
        
        # Advanced settings
        self.volatility_adjustment = True
        self.market_regime_adaptation = True
        self.pattern_recognition_enabled = True
        self.harmonic_patterns_enabled = True
        self.advanced_filtering = True
        
        # Performance metrics
        self.wins_per_strategy = {strategy: 0 for strategy in self.strategy_weights}
        self.losses_per_strategy = {strategy: 0 for strategy in self.strategy_weights}
        self.total_trades_per_strategy = {strategy: 0 for strategy in self.strategy_weights}
        
        # Cache for technical indicators
        self.indicator_cache = {}
        
        # Initialize strategy-specific models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize machine learning models for each strategy type"""
        try:
            os.makedirs('models/strategy', exist_ok=True)
            
            # Initialize or load models for each strategy type
            for strategy_type in self.strategy_weights.keys():
                model_path = f'models/strategy/{strategy_type}_model.pkl'
                
                if os.path.exists(model_path):
                    self.ensemble_models[strategy_type] = joblib.load(model_path)
                    self.logger.info(f"Loaded {strategy_type} model from {model_path}")
                else:
                    # Use different model types for different strategies
                    if strategy_type == 'trend_following':
                        # XGBoost for trend following - good at capturing directional momentum
                        self.ensemble_models[strategy_type] = xgb.XGBClassifier(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42
                        )
                    elif strategy_type == 'mean_reversion':
                        # LightGBM for mean reversion - efficient with categorical features
                        self.ensemble_models[strategy_type] = lgb.LGBMClassifier(
                            n_estimators=100,
                            num_leaves=31,
                            learning_rate=0.1,
                            random_state=42
                        )
                    elif strategy_type == 'breakout':
                        # Gradient Boosting for breakout - good at handling imbalanced data
                        self.ensemble_models[strategy_type] = GradientBoostingClassifier(
                            n_estimators=100,
                            max_depth=5,
                            learning_rate=0.1,
                            subsample=0.8,
                            random_state=42
                        )
                    else:  # pattern_recognition
                        # Random Forest for patterns - good at capturing complex interactions
                        self.ensemble_models[strategy_type] = RandomForestClassifier(
                            n_estimators=200,
                            max_depth=10,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=42
                        )
                    self.logger.info(f"Created new {strategy_type} model")
                
                # Initialize performance metrics for each strategy
                self.strategy_performance[strategy_type] = {
                    'accuracy': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'trades': 0,
                    'avg_profit': 0.0,
                    'avg_loss': 0.0,
                    'expectancy': 0.0,
                    'sharpe_ratio': 0.0
                }
                
            # Initialize ensemble weights based on backtest performance
            total_weight = sum(self.strategy_weights.values())
            for strategy, weight in self.strategy_weights.items():
                self.ensemble_weights[strategy] = weight / total_weight
                
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            # Fallback to equal weights
            for strategy in self.strategy_weights.keys():
                self.ensemble_weights[strategy] = 0.25

    def _initialize_indicators(self):
        """Initialize technical indicators"""
        try:
            # Get indicator settings from config
            indicator_config = self.config['market_analysis']['technical_indicators']
            
            # Moving averages
            self.indicators['moving_averages'] = {
                'periods': indicator_config['moving_averages'],
                'types': ['sma', 'ema', 'wma', 'hma']  # Added weighted and Hull MAs
            }
            
            # Momentum indicators
            self.indicators['momentum'] = {
                'rsi': indicator_config['rsi'],
                'macd': indicator_config['macd'],
                'stochastic': {
                    'k_period': 14,
                    'd_period': 3
                },
                'awesome_oscillator': True,
                'cci': {
                    'period': 14
                },
                'mfi': {
                    'period': 14
                },
                'tsi': {
                    'long_period': 25,
                    'short_period': 13
                }
            }
            
            # Volatility indicators
            self.indicators['volatility'] = {
                'atr': indicator_config['atr'],
                'bollinger': indicator_config['bollinger'],
                'keltner': {
                    'period': 20,
                    'atr_multiplier': 2
                },
                'donchian': {
                    'period': 20
                },
                'historical_volatility': {
                    'period': 20
                }
            }
            
            # Volume indicators
            self.indicators['volume'] = {
                'obv': True,
                'mfi': {
                    'period': 14
                },
                'cmf': {
                    'period': 20
                },
                'vwap': True,
                'pvt': True
            }
            
            # Add advanced pattern recognition
            self.indicators['patterns'] = {
                'enabled': True,
                'candlestick': ['engulfing', 'doji', 'hammer', 'shooting_star', 'marubozu', 'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows'],
                'harmonic': ['gartley', 'butterfly', 'bat', 'crab'],
                'chart': ['head_and_shoulders', 'double_top', 'double_bottom', 'triangle', 'wedge', 'channel']
            }
            
            # Add market regime detection
            self.indicators['regime'] = {
                'lookback': 50,
                'threshold': 0.6,
                'hurst_exponent': True,
                'fractal_dimension': True,
                'spectral_analysis': True
            }
            
            # Add order flow indicators
            self.indicators['order_flow'] = {
                'vwap_bands': {
                    'std_dev': 1.5,
                    'period': 14
                },
                'delta_volume': True,
                'cumulative_delta': True,
                'imbalance_ratio': True
            }
            
            # Add correlation indicators
            self.indicators['correlation'] = {
                'benchmark_symbols': ['SPY', 'QQQ', 'DXY', 'VIX'],
                'lookback': 30
            }
            
        except Exception as e:
            self.logger.error(f"Error initializing indicators: {str(e)}")
            
    async def analyze_market(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze market data across multiple timeframes"""
        try:
            analysis = {}
            
            # Clear indicator cache for new analysis
            self.indicator_cache = {}
            
            # Analyze each timeframe
            for timeframe, df in data.items():
                if df.empty:
                    continue
                    
                analysis[timeframe] = {}
                
                # Multi-timeframe analysis
                analysis[timeframe]['mtf'] = self._analyze_multiple_timeframes(data)
                
                # Trend analysis
                analysis[timeframe]['trend'] = self._analyze_trend(df)
                
                # Momentum analysis
                analysis[timeframe]['momentum'] = self._analyze_momentum(df)
                
                # Volatility analysis
                analysis[timeframe]['volatility'] = self._analyze_volatility(df)
                
                # Price action analysis
                analysis[timeframe]['price_action'] = self._analyze_price_action(df)
                
                # Market regime analysis
                analysis[timeframe]['regime'] = self._analyze_market_regime(data)
                
                # Add adaptive regime identification
                analysis[timeframe]['adaptive_regime'] = self._identify_adaptive_regime(df)
                
                # Add order flow analysis
                analysis[timeframe]['order_flow'] = self._analyze_order_flow(df)
                
                # Add correlation analysis with other symbols
                analysis[timeframe]['correlations'] = self._analyze_correlations(df, timeframe)
                
                # Add market sentiment analysis if available
                analysis[timeframe]['sentiment'] = self._analyze_sentiment(timeframe)
                
                # Add harmonic pattern detection
                if self.harmonic_patterns_enabled:
                    analysis[timeframe]['harmonic_patterns'] = self._detect_harmonic_patterns(df)
                
                # Add support/resistance levels
                analysis[timeframe]['support_resistance'] = self._identify_support_resistance(df)
                
                # Add pivot points
                analysis[timeframe]['pivot_points'] = self._calculate_pivot_points(df)
                
                # Calculate market strength index
                analysis[timeframe]['market_strength'] = self._calculate_market_strength(df, analysis[timeframe])
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            return {}
            
    def _identify_adaptive_regime(self, df: pd.DataFrame) -> Dict:
        """Identify the current market regime adaptively"""
        try:
            if len(df) < 50:
                return {'regime': 'unknown', 'confidence': 0.0}
                
            # Calculate daily returns
            returns = df['close'].pct_change().dropna()
            
            # Calculate momentum and volatility
            momentum = returns.rolling(20).mean().iloc[-1]
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Calculate trend strength using Fisher transform
            price_series = (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min())
            # Handle potential division by zero
            price_series = price_series.clip(0.01, 0.99)
            fisher_transform = 0.5 * np.log((1 + price_series) / (1 - price_series))
            trend_strength = fisher_transform.rolling(20).mean().iloc[-1]
            
            # Calculate Hurst exponent for fractal analysis
            hurst = self._calculate_hurst_exponent(df['close'].values)
            
            # Calculate market efficiency ratio
            mer = self._calculate_market_efficiency_ratio(df)
            
            # Determine regime with enhanced methods
            if hurst < 0.4:  # Strong mean reversion
                regime = 'mean_reverting'
                confidence = 0.7 + (0.4 - hurst) * 0.75
            elif hurst > 0.6:  # Strong trending
                if momentum > 0:
                    regime = 'bullish_trend'
                    confidence = 0.6 + (hurst - 0.6) * 1.25
                else:
                    regime = 'bearish_trend'
                    confidence = 0.6 + (hurst - 0.6) * 1.25
            elif abs(momentum) < 0.0005 and volatility < 0.01:  # Low volatility, low momentum
                regime = 'ranging'
                confidence = 0.6 + (0.01 - volatility) * 10
            elif volatility > 0.015:  # High volatility
                if volatility > 0.025:  # Extremely high volatility
                    regime = 'chaotic'
                    confidence = 0.7 + min((volatility - 0.025) * 10, 0.3)
                else:
                    regime = 'volatile'
                    confidence = 0.6 + (volatility - 0.015) * 20
            elif mer < 0.3:  # Inefficient market
                regime = 'choppy'
                confidence = 0.6 + (0.3 - mer) * 1.5
            else:
                regime = 'undefined'
                confidence = 0.5
                
            # Clip confidence to valid range
            confidence = min(0.95, max(0.5, confidence))
                
            self.last_regime = regime
            
            return {
                'regime': regime,
                'confidence': confidence,
                'momentum': momentum,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'hurst_exponent': hurst,
                'market_efficiency': mer
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying adaptive regime: {str(e)}")
            return {'regime': 'unknown', 'confidence': 0.0}
    
    def _calculate_hurst_exponent(self, price_series, max_lag=20):
        """Calculate Hurst exponent to determine long-term memory in price series"""
        try:
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(price_series[lag:], price_series[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] / 2.0  # Hurst exponent is slope/2
        except Exception as e:
            self.logger.error(f"Error calculating Hurst exponent: {str(e)}")
            return 0.5  # Default to random walk
    
    def _calculate_market_efficiency_ratio(self, df):
        """Calculate Market Efficiency Ratio (MER)"""
        try:
            if len(df) < 20:
                return 0.5
                
            # Directional movement
            directional_movement = abs(df['close'].iloc[-1] - df['close'].iloc[-20])
            
            # Volatility path
            path = sum([abs(df['close'].iloc[i] - df['close'].iloc[i-1]) for i in range(-19, 0)])
            
            # MER is directional movement divided by path
            if path == 0:
                return 0.5
                
            return directional_movement / path
        except Exception as e:
            self.logger.error(f"Error calculating market efficiency ratio: {str(e)}")
            return 0.5
    
    def _analyze_order_flow(self, df: pd.DataFrame) -> Dict:
        """Analyze order flow and buying/selling pressure"""
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return {'buying_pressure': 0.0, 'selling_pressure': 0.0}
                
            # Calculate buying and selling volume
            df['up_candle'] = df['close'] > df['open']
            buying_volume = df.loc[df['up_candle'], 'volume'].sum()
            selling_volume = df.loc[~df['up_candle'], 'volume'].sum()
            total_volume = buying_volume + selling_volume
            
            # Calculate pressure indicators
            buying_pressure = buying_volume / total_volume if total_volume > 0 else 0.5
            selling_pressure = selling_volume / total_volume if total_volume > 0 else 0.5
            
            # Calculate recent buying/selling pressure (last 5 candles)
            recent_df = df.iloc[-5:]
            recent_buying = recent_df.loc[recent_df['up_candle'], 'volume'].sum()
            recent_selling = recent_df.loc[~recent_df['up_candle'], 'volume'].sum()
            recent_total = recent_buying + recent_selling
            
            recent_buying_pressure = recent_buying / recent_total if recent_total > 0 else 0.5
            recent_selling_pressure = recent_selling / recent_total if recent_total > 0 else 0.5
            
            # Calculate volume delta (difference between buying and selling volume)
            volume_delta = buying_volume - selling_volume
            volume_delta_ratio = volume_delta / total_volume if total_volume > 0 else 0
            
            # Calculate volume imbalance
            volume_imbalance = abs(volume_delta_ratio)
            
            # Determine if there's buying or selling dominance
            buying_dominance = buying_pressure > 0.6 and recent_buying_pressure > 0.65
            selling_dominance = selling_pressure > 0.6 and recent_selling_pressure > 0.65
            
            return {
                'buying_pressure': buying_pressure,
                'selling_pressure': selling_pressure,
                'recent_buying_pressure': recent_buying_pressure,
                'recent_selling_pressure': recent_selling_pressure,
                'volume_delta_ratio': volume_delta_ratio,
                'volume_imbalance': volume_imbalance,
                'buying_dominance': buying_dominance,
                'selling_dominance': selling_dominance
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing order flow: {str(e)}")
            return {'buying_pressure': 0.0, 'selling_pressure': 0.0}
            
    def _analyze_correlations(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Analyze correlations with other symbols"""
        # In a real implementation, this would compare the current symbol with other symbols
        # For demonstration, we'll return a more detailed placeholder
        try:
            # Example correlation data
            correlations = {
                'SPY': 0.75,  # Positive correlation with S&P 500
                'QQQ': 0.68,  # Positive correlation with Nasdaq
                'DXY': -0.42, # Negative correlation with US Dollar Index
                'VIX': -0.35  # Negative correlation with volatility index
            }
            
            # Calculate overall correlation metrics
            avg_correlation = sum(abs(v) for v in correlations.values()) / len(correlations)
            max_correlation = max(abs(v) for v in correlations.values())
            diversification_score = 1 - avg_correlation  # Higher is better diversification
            
            return {
                'correlations': correlations,
                'avg_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'diversification_score': diversification_score
            }
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {str(e)}")
            return {'avg_correlation': 0.0}
        
    def _analyze_sentiment(self, timeframe: str) -> Dict:
        """Analyze market sentiment"""
        # In a real implementation, this would incorporate external sentiment data
        # For demonstration, we'll return a more detailed placeholder
        try:
            # Example sentiment metrics
            sentiment = {
                'bullish_sentiment': 0.55,
                'bearish_sentiment': 0.45,
                'bullish_consensus': 0.58,  # Analyst consensus
                'bearish_consensus': 0.42,
                'retail_sentiment': 0.52,   # Retail trader sentiment
                'institutional_sentiment': 0.57  # Institutional sentiment
            }
            
            # Calculate sentiment divergence (difference between retail and institutional)
            sentiment['sentiment_divergence'] = sentiment['institutional_sentiment'] - sentiment['retail_sentiment']
            
            # Calculate sentiment strength
            sentiment['sentiment_strength'] = abs(sentiment['bullish_sentiment'] - 0.5) * 2  # 0 to 1 scale
            
            return sentiment
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {'bullish_sentiment': 0.5, 'bearish_sentiment': 0.5}
    
    def _detect_harmonic_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect harmonic price patterns"""
        try:
            if len(df) < 50:
                return {'patterns': []}
                
            # Find swing highs and lows
            highs = df['high'].values
            lows = df['low'].values
            
            # Use scipy to find peaks
            peak_indices = find_peaks(highs, distance=5)[0]
            trough_indices = find_peaks(-lows, distance=5)[0]
            
            # Need at least 5 points to identify a harmonic pattern
            if len(peak_indices) < 3 or len(trough_indices) < 3:
                return {'patterns': []}
                
            # Identify potential patterns
            patterns = []
            
            # Get most recent 5 significant points
            points = []
            all_indices = sorted(list(peak_indices) + list(trough_indices))[-7:]
            if len(all_indices) < 5:
                return {'patterns': []}
                
            # Analyze ratios for harmonic patterns
            for pattern_name, ratio_ranges in self._get_harmonic_ratio_ranges().items():
                is_match, confidence = self._check_pattern_match(df, all_indices, ratio_ranges)
                if is_match:
                    completion_price = self._calculate_pattern_completion(df, all_indices, pattern_name)
                    patterns.append({
                        'name': pattern_name,
                        'confidence': confidence,
                        'completion_price': completion_price,
                        'points': [int(i) for i in all_indices[-5:]]
                    })
                    
            return {'patterns': patterns}
            
        except Exception as e:
            self.logger.error(f"Error detecting harmonic patterns: {str(e)}")
            return {'patterns': []}
    
    def _get_harmonic_ratio_ranges(self):
        """Get the Fibonacci ratio ranges for different harmonic patterns"""
        return {
            'gartley': {
                'xab': (0.618, 0.618),
                'abc': (0.382, 0.886),
                'bcd': (1.13, 1.618),
                'xad': (0.786, 0.786)
            },
            'butterfly': {
                'xab': (0.786, 0.786),
                'abc': (0.382, 0.886),
                'bcd': (1.618, 2.618),
                'xad': (1.27, 1.27)
            },
            'bat': {
                'xab': (0.382, 0.5),
                'abc': (0.382, 0.886),
                'bcd': (1.618, 2.618),
                'xad': (0.886, 0.886)
            },
            'crab': {
                'xab': (0.382, 0.618),
                'abc': (0.382, 0.886),
                'bcd': (2.618, 3.618),
                'xad': (1.618, 1.618)
            }
        }
    
    def _check_pattern_match(self, df, points, ratio_ranges):
        """Check if the points match a harmonic pattern's ratio requirements"""
        # Implementation would analyze the ratio relationships between price points
        # Simplified for demonstration
        return False, 0
    
    def _calculate_pattern_completion(self, df, points, pattern_name):
        """Calculate the pattern completion price level"""
        # This would calculate the D point of the harmonic pattern
        # Simplified for demonstration
        return df['close'].iloc[-1]
    
    def _identify_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Identify support and resistance levels"""
        try:
            if len(df) < 30:
                return {'support': [], 'resistance': []}
                
            # Find swing highs and lows using peak detection
            high_peaks = find_peaks(df['high'].values, distance=5, prominence=1)[0]
            low_peaks = find_peaks(-df['low'].values, distance=5, prominence=1)[0]
            
            # Get the price values at peak locations
            highs = [df['high'].iloc[i] for i in high_peaks]
            lows = [df['low'].iloc[i] for i in low_peaks]
            
            # Cluster nearby levels
            resistance_clusters = self._cluster_price_levels(highs)
            support_clusters = self._cluster_price_levels(lows)
            
            # Sort levels by strength (frequency)
            resistance_levels = sorted([(level, count) for level, count in resistance_clusters.items()], 
                                      key=lambda x: x[1], reverse=True)
            support_levels = sorted([(level, count) for level, count in support_clusters.items()], 
                                   key=lambda x: x[1], reverse=True)
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Filter for levels near current price and sort by distance
            nearby_resistance = sorted([(level, count) for level, count in resistance_levels if level > current_price], 
                                      key=lambda x: x[0])
            nearby_support = sorted([(level, count) for level, count in support_levels if level < current_price], 
                                   key=lambda x: x[0], reverse=True)
            
            # Limit to top 3 levels
            resistance = [{'price': float(level), 'strength': count} for level, count in nearby_resistance[:3]]
            support = [{'price': float(level), 'strength': count} for level, count in nearby_support[:3]]
            
            return {
                'support': support,
                'resistance': resistance
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying support/resistance: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def _cluster_price_levels(self, price_levels, tolerance=0.005):
        """Cluster nearby price levels"""
        clusters = {}
        for price in price_levels:
            # Check if price is close to an existing cluster
            for cluster_price in list(clusters.keys()):
                if abs(price - cluster_price) / cluster_price < tolerance:
                    # Add to existing cluster
                    clusters[cluster_price] += 1
                    break
            else:
                # Create new cluster
                clusters[price] = 1
                
        return clusters
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """Calculate pivot points for potential support/resistance"""
        try:
            if len(df) < 2:
                return {}
                
            # Get the last complete candle
            high = df['high'].iloc[-2]
            low = df['low'].iloc[-2]
            close = df['close'].iloc[-2]
            
            # Calculate the pivot point
            pivot = (high + low + close) / 3
            
            # Calculate support and resistance levels
            s1 = (2 * pivot) - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            r1 = (2 * pivot) - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            return {
                'pivot': float(pivot),
                'support': [float(s1), float(s2), float(s3)],
                'resistance': [float(r1), float(r2), float(r3)]
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating pivot points: {str(e)}")
            return {}
    
    def _calculate_market_strength(self, df: pd.DataFrame, analysis: Dict) -> float:
        """Calculate overall market strength index"""
        try:
            if len(df) < 20:
                return 0.5
                
            # Combine multiple factors to calculate market strength
            factors = []
            
            # Trend strength (using ADX if available)
            trend_strength = analysis.get('trend', {}).get('adx', 0) / 100
            factors.append((trend_strength, 2.0))  # Higher weight for trend
            
            # Momentum (using RSI)
            rsi = ta.momentum.RSIIndicator(df['close']).rsi().iloc[-1] / 100
            normalized_rsi = (rsi - 0.3) / 0.4  # Normalize to -0.75 to 1.75 range
            factors.append((normalized_rsi, 1.5))
            
            # Volume strength
            if 'volume' in df.columns:
                volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                volume_factor = min(2, volume_ratio) / 2  # Normalize to 0-1
                factors.append((volume_factor, 1.0))
            
            # Order flow
            buying_pressure = analysis.get('order_flow', {}).get('buying_pressure', 0.5)
            factors.append((buying_pressure, 1.2))
            
            # Market regime
            regime_data = analysis.get('adaptive_regime', {})
            regime = regime_data.get('regime', 'unknown')
            regime_factor = 0.5
            if regime == 'bullish_trend':
                regime_factor = 0.8
            elif regime == 'bearish_trend':
                regime_factor = 0.2
            factors.append((regime_factor, 2.0))
            
            # Calculate weighted average
            weighted_sum = sum(value * weight for value, weight in factors)
            total_weight = sum(weight for _, weight in factors)
            
            market_strength = weighted_sum / total_weight if total_weight > 0 else 0.5
            
            # Clip to valid range
            return min(1.0, max(0.0, market_strength))
            
        except Exception as e:
            self.logger.error(f"Error calculating market strength: {str(e)}")
            return 0.5
            
    def generate_signals(self, analysis: Dict, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate trading signals based on market analysis using ensemble approach"""
        try:
            signals = []
            
            # Generate signals for each strategy type
            trend_signals = self._generate_trend_signals(analysis, market_data)
            breakout_signals = self._generate_breakout_signals(analysis, market_data)
            reversal_signals = self._generate_reversal_signals(analysis, market_data)
            pattern_signals = self._generate_pattern_signals(analysis, market_data)
            
            # Add signals to the combined list
            signals.extend(trend_signals)
            signals.extend(breakout_signals)
            signals.extend(reversal_signals)
            signals.extend(pattern_signals)
            
            # Enhance signals with ML confidence scores if available
            signals = self._enhance_signals_with_ml(signals, market_data)
            
            # Apply ensemble weighting based on strategy performance
            signals = self._apply_ensemble_weights(signals)
            
            # Filter signals based on quality and regime compatibility
            signals = self._filter_signals(signals)
            
            # Rank signals by confidence
            signals = self._rank_signals(signals)
            
            # Add signal to history
            if signals:
                self.signal_history.append({
                    'timestamp': datetime.now(),
                    'signals': signals,
                    'count': len(signals)
                })
                
                # Keep history bounded
                if len(self.signal_history) > self.max_signal_history:
                    self.signal_history = self.signal_history[-self.max_signal_history:]
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return []
            
    def _enhance_signals_with_ml(self, signals: List[Signal], market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Enhance signals with ML model predictions"""
        try:
            if not signals:
                return []
                
            primary_timeframe = list(market_data.keys())[0] if market_data else None
            if not primary_timeframe or primary_timeframe not in market_data:
                return signals
                
            df = market_data[primary_timeframe]
            if len(df) < 50:  # Need enough data for feature calculation
                return signals
                
            # Prepare features for ML prediction
            features = self._prepare_features(df)
            if features is None:
                return signals
                
            # Get predictions from each model
            for signal in signals:
                strategy_type = signal.strategy_type if hasattr(signal, 'strategy_type') else 'trend_following'
                
                if strategy_type in self.ensemble_models:
                    model = self.ensemble_models[strategy_type]
                    
                    try:
                        # Make prediction
                        prediction = model.predict([features])[0]
                        
                        # Get probability if available
                        confidence = 0.65  # Default confidence
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba([features])[0]
                            confidence = proba[1] if prediction == 1 else proba[0]
                            
                        # If signal direction matches prediction, boost confidence
                        if (signal.direction == 'buy' and prediction == 1) or \
                           (signal.direction == 'sell' and prediction == 0):
                            signal.confidence = max(signal.confidence, confidence)
                        else:
                            # If they don't match, reduce confidence
                            signal.confidence = min(signal.confidence, 0.5)
                            
                    except Exception as e:
                        self.logger.error(f"Error making prediction with {strategy_type} model: {str(e)}")
                        
            return signals
            
        except Exception as e:
            self.logger.error(f"Error enhancing signals with ML: {str(e)}")
            return signals
            
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model input"""
        try:
            # Create a single feature vector
            features = {}
            
            # Moving average features
            features['sma20_dist'] = (df['close'].iloc[-1] / df['close'].rolling(20).mean().iloc[-1]) - 1
            features['sma50_dist'] = (df['close'].iloc[-1] / df['close'].rolling(50).mean().iloc[-1]) - 1
            
            # Momentum features
            rsi = ta.momentum.RSIIndicator(df['close']).rsi()
            macd = ta.trend.MACD(df['close'])
            
            features['rsi'] = rsi.iloc[-1]
            features['macd'] = macd.macd().iloc[-1]
            features['macd_signal'] = macd.macd_signal().iloc[-1]
            features['macd_diff'] = macd.macd_diff().iloc[-1]
            
            # Volatility features
            bb = ta.volatility.BollingerBands(df['close'])
            features['bb_width'] = (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]) / df['close'].iloc[-1]
            
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            features['atr_ratio'] = atr.iloc[-1] / df['close'].iloc[-1]
            
            # Volume features if available
            if 'volume' in df.columns:
                features['volume_change'] = df['volume'].pct_change().iloc[-1]
                features['volume_ma_ratio'] = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                
            # Trend features
            features['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx().iloc[-1]
            
            # Add candlestick pattern features
            features['doji'] = 1 if abs(df['open'].iloc[-1] - df['close'].iloc[-1]) / (df['high'].iloc[-1] - df['low'].iloc[-1]) < 0.1 else 0
            features['engulfing'] = 1 if (df['close'].iloc[-1] > df['open'].iloc[-1] and 
                                    df['open'].iloc[-1] < df['close'].iloc[-2] and 
                                    df['close'].iloc[-1] > df['open'].iloc[-2]) or 
                                   (df['close'].iloc[-1] < df['open'].iloc[-1] and 
                                    df['open'].iloc[-1] > df['close'].iloc[-2] and 
                                    df['close'].iloc[-1] < df['open'].iloc[-2]) else 0
            
            # Add price pattern features
            features['higher_high'] = 1 if df['high'].iloc[-1] > df['high'].iloc[-2] and df['high'].iloc[-2] > df['high'].iloc[-3] else 0
            features['lower_low'] = 1 if df['low'].iloc[-1] < df['low'].iloc[-2] and df['low'].iloc[-2] < df['low'].iloc[-3] else 0
            
            # Add market regime features
            features['hurst'] = self._calculate_hurst_exponent(df['close'].values)
            features['market_efficiency'] = self._calculate_market_efficiency_ratio(df)
            
            # Fill any NaN values
            features = pd.Series(features).fillna(0)
            
            # Return as numpy array
            return features.values
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return None
            
    def _apply_ensemble_weights(self, signals: List[Signal]) -> List[Signal]:
        """Apply ensemble weights to signals based on strategy type"""
        try:
            for signal in signals:
                strategy_type = signal.strategy_type if hasattr(signal, 'strategy_type') else 'trend_following'
                weight = self.ensemble_weights.get(strategy_type, 0.25)
                
                # Apply weight to confidence
                signal.confidence = signal.confidence * weight
                
                # Adjust confidence based on market regime compatibility
                if self.last_regime:
                    regime_compatibility = self._get_regime_compatibility(strategy_type, self.last_regime)
                    signal.confidence *= regime_compatibility
                    
                # Adjust confidence based on strategy historical performance
                performance_factor = self._get_strategy_performance_factor(strategy_type)
                signal.confidence *= performance_factor
                    
            return signals
            
        except Exception as e:
            self.logger.error(f"Error applying ensemble weights: {str(e)}")
            return signals
    
    def _get_strategy_performance_factor(self, strategy_type: str) -> float:
        """Get performance factor based on historical strategy performance"""
        try:
            wins = self.wins_per_strategy.get(strategy_type, 0)
            losses = self.losses_per_strategy.get(strategy_type, 0)
            
            if wins + losses < 10:  # Not enough data
                return 1.0
                
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5
            
            # Scale factor based on win rate (0.7 to 1.3)
            factor = 0.7 + win_rate * 0.6
            
            return factor
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy performance factor: {str(e)}")
            return 1.0
            
    def _get_regime_compatibility(self, strategy_type: str, regime: str) -> float:
        """Get compatibility score between strategy type and market regime"""
        compatibility_matrix = {
            'trend_following': {
                'bullish_trend': 1.2,
                'bearish_trend': 1.2,
                'ranging': 0.7,
                'volatile': 0.8,
                'undefined': 1.0,
                'mean_reverting': 0.6,
                'choppy': 0.7,
                'chaotic': 0.6
            },
            'mean_reversion': {
                'bullish_trend': 0.7,
                'bearish_trend': 0.7,
                'ranging': 1.2,
                'volatile': 0.9,
                'undefined': 0.8,
                'mean_reverting': 1.3,
                'choppy': 0.8,
                'chaotic': 0.7
            },
            'breakout': {
                'bullish_trend': 1.1,
                'bearish_trend': 1.1,
                'ranging': 0.9,
                'volatile': 1.2,
                'undefined': 0.9,
                'mean_reverting': 0.7,
                'choppy': 0.6,
                'chaotic': 1.0
            },
            'pattern_recognition': {
                'bullish_trend': 1.0,
                'bearish_trend': 1.0,
                'ranging': 1.0,
                'volatile': 1.0,
                'undefined': 1.0,
                'mean_reverting': 1.0,
                'choppy': 1.0,
                'chaotic': 0.8
            }
        }
        
        if strategy_type in compatibility_matrix and regime in compatibility_matrix[strategy_type]:
            return compatibility_matrix[strategy_type][regime]
        return 1.0
        
    def _filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals based on quality criteria"""
        try:
            filtered_signals = []
            
            for signal in signals:
                # Filter by minimum confidence
                if signal.confidence < self.min_confidence_threshold:
                    continue
                    
                # Filter by risk/reward ratio
                if not hasattr(signal, 'risk_reward_ratio') or signal.risk_reward_ratio < 1.5:
                    continue
                    
                # Additional filters for specific strategy types
                strategy_type = signal.strategy_type if hasattr(signal, 'strategy_type') else 'trend_following'
                
                if strategy_type == 'trend_following' and self.last_regime in ['ranging', 'choppy', 'mean_reverting']:
                    # Lower threshold for trend signals in non-trending regimes
                    if signal.confidence < self.min_confidence_threshold * 1.2:
                        continue
                        
                if strategy_type == 'mean_reversion' and self.last_regime in ['bullish_trend', 'bearish_trend']:
                    # Lower threshold for reversal signals in trending regimes
                    if signal.confidence < self.min_confidence_threshold * 1.2:
                        continue
                
                # Add signal if it passes all filters
                filtered_signals.append(signal)
                
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error filtering signals: {str(e)}")
            return signals
            
    def _rank_signals(self, signals: List[Signal]) -> List[Signal]:
        """Rank signals by confidence and other factors"""
        try:
            # Sort by confidence
            ranked_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)
            
            # Return top signals (limited to max 3 per symbol)
            symbol_count = {}
            filtered_signals = []
            
            for signal in ranked_signals:
                symbol = signal.symbol
                symbol_count[symbol] = symbol_count.get(symbol, 0) + 1
                
                if symbol_count[symbol] <= 3:
                    filtered_signals.append(signal)
                    
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error ranking signals: {str(e)}")
            return signals
            
    def update_strategy_performance(self, trade_results: List[Dict]):
        """Update strategy performance metrics based on trade results"""
        try:
            for result in trade_results:
                strategy_type = result.get('strategy_type', 'trend_following')
                profit = result.get('profit', 0)
                
                # Update win/loss counts
                if profit > 0:
                    self.wins_per_strategy[strategy_type] = self.wins_per_strategy.get(strategy_type, 0) + 1
                else:
                    self.losses_per_strategy[strategy_type] = self.losses_per_strategy.get(strategy_type, 0) + 1
                    
                # Update total trades
                self.total_trades_per_strategy[strategy_type] = self.total_trades_per_strategy.get(strategy_type, 0) + 1
                
                # Calculate performance metrics
                if strategy_type in self.strategy_performance:
                    perf = self.strategy_performance[strategy_type]
                    wins = self.wins_per_strategy.get(strategy_type, 0)
                    losses = self.losses_per_strategy.get(strategy_type, 0)
                    total = wins + losses
                    
                    if total > 0:
                        perf['win_rate'] = wins / total
                        perf['trades'] = total
                        
                    # Update weights based on performance
                    self._update_strategy_weights()
                    
        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {str(e)}")
            
    def _update_strategy_weights(self):
        """Update strategy weights based on performance"""
        try:
            # Calculate performance scores
            scores = {}
            for strategy_type, perf in self.strategy_performance.items():
                win_rate = perf.get('win_rate', 0.5)
                trades = perf.get('trades', 0)
                
                if trades < 10:  # Not enough data
                    scores[strategy_type] = 1.0
                else:
                    # Score based on win rate, with minimum of 0.5
                    scores[strategy_type] = max(0.5, win_rate * 2)
                    
            # Normalize scores
            total_score = sum(scores.values())
            if total_score > 0:
                for strategy_type in scores:
                    self.strategy_weights[strategy_type] = scores[strategy_type] / total_score
                    
                # Update ensemble weights
                total_weight = sum(self.strategy_weights.values())
                for strategy_type, weight in self.strategy_weights.items():
                    self.ensemble_weights[strategy_type] = weight / total_weight
                    
        except Exception as e:
            self.logger.error(f"Error updating strategy weights: {str(e)}")
    
    def get_strategy_performance(self) -> Dict:
        """Get current strategy performance metrics"""
        return {
            'strategy_weights': self.strategy_weights,
            'ensemble_weights': self.ensemble_weights,
            'performance': self.strategy_performance,
            'wins_per_strategy': self.wins_per_strategy,
            'losses_per_strategy': self.losses_per_strategy,
            'total_trades_per_strategy': self.total_trades_per_strategy
        }
        
    def save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs('models/strategy', exist_ok=True)
            
            for strategy_type, model in self.ensemble_models.items():
                model_path = f'models/strategy/{strategy_type}_model.pkl'
                joblib.dump(model, model_path)
                self.logger.info(f"Saved {strategy_type} model to {model_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}") 