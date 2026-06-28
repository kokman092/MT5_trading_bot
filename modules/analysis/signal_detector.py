import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from ..core.error_handler import ErrorHandler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SignalDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler(config)
        self.signal_config = config.get('signal_detection', {})
        
        # Initialize ML models
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=42
        )
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=42
        )
        
        # Model weights for ensemble
        self.model_weights = {
            'random_forest': 0.3,
            'xgboost': 0.4,
            'lightgbm': 0.3
        }

    def analyze_market(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Analyze market data and generate trading signals"""
        try:
            # Detect anomalies and patterns using ML
            ml_patterns = self._detect_ml_patterns(df)
            
            signals = {
                'price_action': self._analyze_price_action(df),
                'volume': self._analyze_volume(df),
                'momentum': self._analyze_momentum(df),
                'support_resistance': self._analyze_support_resistance(df),
                'volatility': self._analyze_volatility(df),
                'trend': self._analyze_trend(df),
                'harmonic': self._analyze_harmonic_patterns(df),
                'order_flow': self._analyze_order_flow(df),
                'market_structure': self._analyze_market_structure(df),
                'ml_patterns': ml_patterns
            }
            
            # Apply advanced filters
            if not self._check_filters(df, symbol):
                return {'signal': 'NONE', 'confidence': 0, 'details': {}}
                
            # Calculate overall signal with market context
            signal = self._calculate_composite_signal(signals)
            
            # Add market context and signal details
            signal['details'] = signals
            signal['market_context'] = self._analyze_market_context(df)
            signal['timestamp'] = datetime.now()
            signal['symbol'] = symbol
            
            # Generate visualizations
            if self.signal_config.get('generate_visualizations', False):
                signal['visualizations'] = self._generate_visualizations(df, signals)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            return {'signal': 'NONE', 'confidence': 0, 'details': {}}
            
    def _analyze_price_action(self, df: pd.DataFrame) -> Dict:
        """Analyze price action patterns"""
        try:
            patterns = {}
            config = self.signal_config['methods']['price_action']
            
            if not config['enabled']:
                return {'score': 0, 'patterns': {}}
                
            # Check for candlestick patterns
            if 'DOJI' in config['patterns']:
                patterns['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
                
            if 'HAMMER' in config['patterns']:
                patterns['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
                
            if 'ENGULFING' in config['patterns']:
                patterns['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
                
            # Calculate pattern strength
            pattern_signals = pd.DataFrame(patterns)
            current_signals = pattern_signals.iloc[-config['confirmation_candles']:]
            
            score = (current_signals != 0).any(axis=1).mean()
            
            return {
                'score': score,
                'patterns': {k: v.iloc[-1] for k, v in patterns.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Error in price action analysis: {str(e)}")
            return {'score': 0, 'patterns': {}}
            
    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        try:
            config = self.signal_config['methods']['volume_analysis']
            if not config['enabled']:
                return {'score': 0, 'signals': {}}
                
            # Calculate volume metrics
            volume = df['tick_volume']
            volume_sma = volume.rolling(config['volume_trend_periods']).mean()
            volume_ratio = volume / volume_sma
            
            # Detect volume surges
            volume_surge = volume_ratio > config['volume_surge_threshold']
            
            # Calculate volume trend
            volume_trend = (volume > volume_sma).rolling(config['volume_trend_periods']).mean()
            
            score = (volume_surge.iloc[-1] * 0.6 + volume_trend.iloc[-1] * 0.4)
            
            return {
                'score': score,
                'signals': {
                    'volume_surge': volume_surge.iloc[-1],
                    'volume_trend': volume_trend.iloc[-1],
                    'volume_ratio': volume_ratio.iloc[-1]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in volume analysis: {str(e)}")
            return {'score': 0, 'signals': {}}
            
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze momentum indicators"""
        try:
            config = self.signal_config['methods']['momentum']
            if not config['enabled']:
                return {'score': 0, 'signals': {}}
                
            signals = {}
            
            # RSI Divergence
            if config['rsi_divergence']['enabled']:
                rsi = talib.RSI(df['close'], timeperiod=config['rsi_divergence']['lookback_periods'])
                rsi_div = self._check_divergence(df['close'], rsi, config['rsi_divergence']['threshold'])
                signals['rsi_divergence'] = rsi_div
                
            # MACD Crossover
            if config['macd_crossover']['enabled']:
                macd, signal, hist = talib.MACD(df['close'])
                macd_cross = (hist > config['macd_crossover']['signal_threshold']).astype(int)
                signals['macd_cross'] = macd_cross.iloc[-1]
                
            # Stochastic
            if config['stochastic']['enabled']:
                slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                                         fastk_period=config['stochastic']['k_period'],
                                         slowk_period=config['stochastic']['d_period'],
                                         slowd_period=config['stochastic']['d_period'])
                                         
                stoch_signal = 0
                if slowk.iloc[-1] < config['stochastic']['oversold']:
                    stoch_signal = 1
                elif slowk.iloc[-1] > config['stochastic']['overbought']:
                    stoch_signal = -1
                    
                signals['stochastic'] = stoch_signal
                
            # Calculate composite momentum score
            score = (
                signals.get('rsi_divergence', 0) * 0.4 +
                signals.get('macd_cross', 0) * 0.3 +
                abs(signals.get('stochastic', 0)) * 0.3
            )
            
            return {
                'score': score,
                'signals': signals
            }
            
        except Exception as e:
            self.logger.error(f"Error in momentum analysis: {str(e)}")
            return {'score': 0, 'signals': {}}
            
    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Analyze support and resistance levels"""
        try:
            config = self.signal_config['methods']['support_resistance']
            if not config['enabled']:
                return {'score': 0, 'levels': {}}
                
            # Find potential levels
            highs = df['high'].rolling(20).max()
            lows = df['low'].rolling(20).min()
            
            # Count touches
            high_touches = (abs(df['high'] - highs) < config['level_threshold']).rolling(
                config['levels_lookback']).sum()
            low_touches = (abs(df['low'] - lows) < config['level_threshold']).rolling(
                config['levels_lookback']).sum()
                
            # Find valid levels
            resistance_levels = highs[high_touches >= config['min_touches']]
            support_levels = lows[low_touches >= config['min_touches']]
            
            # Check for breakouts
            current_price = df['close'].iloc[-1]
            resistance_break = current_price > resistance_levels.iloc[-1] if not resistance_levels.empty else False
            support_break = current_price < support_levels.iloc[-1] if not support_levels.empty else False
            
            # Calculate score based on proximity to levels and breakouts
            score = 0
            if resistance_break or support_break:
                score = 1 if len(df) >= config['breakout_confirmation'] else 0.5
                
            return {
                'score': score,
                'levels': {
                    'resistance': resistance_levels.iloc[-1] if not resistance_levels.empty else None,
                    'support': support_levels.iloc[-1] if not support_levels.empty else None,
                    'resistance_break': resistance_break,
                    'support_break': support_break
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in support/resistance analysis: {str(e)}")
            return {'score': 0, 'levels': {}}
            
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """Analyze volatility conditions"""
        try:
            config = self.signal_config['methods']['volatility']
            if not config['enabled']:
                return {'score': 0, 'signals': {}}
                
            # Calculate ATR
            atr = talib.ATR(df['high'], df['low'], df['close'])
            
            # Calculate Bollinger Bands
            if config['bollinger_bands']['enabled']:
                upper, middle, lower = talib.BBANDS(
                    df['close'],
                    timeperiod=config['bollinger_bands']['period'],
                    nbdevup=config['bollinger_bands']['std_dev'],
                    nbdevdn=config['bollinger_bands']['std_dev']
                )
                
                # Check for Bollinger Band squeeze
                band_width = (upper - lower) / middle
                squeeze = band_width < config['bollinger_bands']['squeeze_threshold']
                
            # Calculate volatility score
            vol_score = min(atr.iloc[-1] * config['atr_multiplier'], 1.0)
            
            signals = {
                'atr': atr.iloc[-1],
                'atr_signal': vol_score,
                'bb_squeeze': squeeze.iloc[-1] if config['bollinger_bands']['enabled'] else False
            }
            
            return {
                'score': vol_score,
                'signals': signals
            }
            
        except Exception as e:
            self.logger.error(f"Error in volatility analysis: {str(e)}")
            return {'score': 0, 'signals': {}}
            
    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Analyze trend strength and direction"""
        try:
            config = self.signal_config['methods']['trend']
            if not config['enabled']:
                return {'score': 0, 'signals': {}}
                
            # Calculate EMAs
            emas = {}
            for period in config['ema_periods']:
                emas[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
                
            # Check trend alignment
            current_price = df['close'].iloc[-1]
            trend_direction = 0
            
            # Check if EMAs are aligned (ascending or descending)
            ema_values = [ema.iloc[-1] for ema in emas.values()]
            ascending = all(ema_values[i] <= ema_values[i+1] for i in range(len(ema_values)-1))
            descending = all(ema_values[i] >= ema_values[i+1] for i in range(len(ema_values)-1))
            
            if ascending and current_price > ema_values[-1]:
                trend_direction = 1
            elif descending and current_price < ema_values[-1]:
                trend_direction = -1
                
            # Calculate trend strength
            trend_strength = abs(
                (current_price - ema_values[0]) / 
                (max(ema_values) - min(ema_values)) if max(ema_values) != min(ema_values) else 0
            )
            
            # Check for trend confirmation
            confirmed = (trend_direction != 0 and 
                       trend_strength > config['min_trend_strength'] and
                       abs(trend_direction) == 1)
                       
            return {
                'score': trend_strength if confirmed else 0,
                'signals': {
                    'direction': trend_direction,
                    'strength': trend_strength,
                    'confirmed': confirmed,
                    'ema_values': emas
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
            return {'score': 0, 'signals': {}}
            
    def _check_filters(self, df: pd.DataFrame, symbol: str) -> bool:
        """Apply trading filters"""
        try:
            filters = self.signal_config['filters']
            
            # Time filter
            if filters['time_filter']['enabled']:
                if not self._check_time_filter():
                    return False
                    
            # Spread filter
            if filters['spread_filter']['enabled']:
                if not self._check_spread_filter(symbol):
                    return False
                    
            # Volatility filter
            if filters['volatility_filter']['enabled']:
                atr = talib.ATR(df['high'], df['low'], df['close'])
                if not filters['volatility_filter']['min_atr'] <= atr.iloc[-1] <= filters['volatility_filter']['max_atr']:
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error in filter check: {str(e)}")
            return False
            
    def _calculate_composite_signal(self, signals: Dict) -> Dict:
        """Calculate composite trading signal"""
        try:
            weights = self.signal_config['confirmation']['signal_weights']
            required_signals = self.signal_config['confirmation']['required_signals']
            min_confidence = self.signal_config['confirmation']['min_confidence']
            
            # Calculate weighted score
            total_score = 0
            for signal_type, weight in weights.items():
                total_score += signals[signal_type]['score'] * weight
                
            # Count significant signals
            significant_signals = sum(1 for s in signals.values() if s['score'] > 0.5)
            
            # Determine signal direction
            if significant_signals >= required_signals and total_score >= min_confidence:
                # Determine direction based on trend and momentum
                if signals['trend']['signals']['direction'] > 0:
                    return {'signal': 'BUY', 'confidence': total_score}
                elif signals['trend']['signals']['direction'] < 0:
                    return {'signal': 'SELL', 'confidence': total_score}
                    
            return {'signal': 'NONE', 'confidence': total_score}
            
        except Exception as e:
            self.logger.error(f"Error calculating composite signal: {str(e)}")
            return {'signal': 'NONE', 'confidence': 0}
            
    def _check_divergence(self, price: pd.Series, indicator: pd.Series, threshold: float) -> float:
        """Check for divergence between price and indicator"""
        try:
            # Get local extrema
            price_diff = price.diff()
            indicator_diff = indicator.diff()
            
            # Check for divergence
            price_trend = price_diff.iloc[-threshold:].mean()
            indicator_trend = indicator_diff.iloc[-threshold:].mean()
            
            if abs(price_trend) > 0 and abs(indicator_trend) > 0:
                if price_trend * indicator_trend < 0:  # Opposite directions
                    return abs(indicator_trend)
                    
            return 0
            
        except Exception as e:
            self.logger.error(f"Error checking divergence: {str(e)}")
            return 0
            
    def _check_time_filter(self) -> bool:
        """Check if current time is valid for trading"""
        try:
            config = self.signal_config['filters']['time_filter']
            current_time = datetime.now()
            
            # Check for news events if enabled
            if config['avoid_news']:
                if self._is_news_time(current_time, config['news_buffer_minutes']):
                    return False
                    
            # Check trading session if enabled
            if config['session_trading_only']:
                if not self._is_valid_session(current_time):
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error in time filter: {str(e)}")
            return False
            
    def _check_spread_filter(self, symbol: str) -> bool:
        """Check if spread is within acceptable range"""
        try:
            config = self.signal_config['filters']['spread_filter']
            
            # Get current spread
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
                
            current_spread = symbol_info.spread * symbol_info.point
            
            # Check against maximum allowed spread
            if current_spread > config['max_spread']:
                return False
                
            # Check for sudden spread increase
            if hasattr(self, 'last_spread'):
                if current_spread > self.last_spread * config['spread_increase_threshold']:
                    return False
                    
            self.last_spread = current_spread
            return True
            
        except Exception as e:
            self.logger.error(f"Error in spread filter: {str(e)}")
            return False

    def _analyze_harmonic_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze harmonic price patterns (Gartley, Butterfly, etc.)"""
        try:
            patterns = {}
            swings = self._find_swing_points(df)
            
            # Check for Gartley pattern
            patterns['gartley'] = self._check_gartley_pattern(swings)
            
            # Check for Butterfly pattern
            patterns['butterfly'] = self._check_butterfly_pattern(swings)
            
            # Check for Bat pattern
            patterns['bat'] = self._check_bat_pattern(swings)
            
            # Check for Crab pattern
            patterns['crab'] = self._check_crab_pattern(swings)
            
            # Calculate pattern completion percentage
            completion_scores = [score for score in patterns.values() if score is not None]
            pattern_score = max(completion_scores) if completion_scores else 0
            
            return {
                'score': pattern_score,
                'patterns': patterns
            }
            
        except Exception as e:
            self.logger.error(f"Error in harmonic pattern analysis: {str(e)}")
            return {'score': 0, 'patterns': {}}

    def _analyze_order_flow(self, df: pd.DataFrame) -> Dict:
        """Analyze order flow dynamics"""
        try:
            signals = {}
            
            # Calculate basic order flow metrics
            delta = (df['close'] - df['open']) * df['tick_volume']
            cumulative_delta = delta.cumsum()
            
            # Calculate advanced order flow metrics
            footprint = self._calculate_footprint(df)
            imbalance = self._calculate_order_imbalance(df)
            liquidity = self._analyze_liquidity(df)
            
            # Detect absorption levels
            absorption = self._detect_absorption(df, delta)
            
            # Calculate volume profile
            volume_profile = self._calculate_volume_profile(df)
            
            # Identify institutional levels
            inst_levels = self._find_institutional_levels(df, volume_profile)
            
            # Calculate order flow score
            score = (
                absorption['score'] * 0.3 +
                (1 if inst_levels['valid_level'] else 0) * 0.2 +
                footprint['score'] * 0.2 +
                imbalance['score'] * 0.15 +
                liquidity['score'] * 0.15
            )
            
            return {
                'score': score,
                'signals': {
                    'delta': delta.iloc[-1],
                    'cumulative_delta': cumulative_delta.iloc[-1],
                    'absorption': absorption,
                    'institutional_levels': inst_levels,
                    'footprint': footprint,
                    'imbalance': imbalance,
                    'liquidity': liquidity
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in order flow analysis: {str(e)}")
            return {'score': 0, 'signals': {}}

    def _calculate_footprint(self, df: pd.DataFrame) -> Dict:
        """Calculate footprint chart metrics"""
        try:
            # Calculate price levels
            price_levels = pd.interval_range(
                start=df['low'].min(),
                end=df['high'].max(),
                periods=50
            )
            
            # Initialize footprint data
            footprint = {level: {'buy_volume': 0, 'sell_volume': 0} for level in price_levels}
            
            # Populate footprint data
            for idx, row in df.iterrows():
                price_level = next(
                    level for level in price_levels
                    if row['close'] >= level.left and row['close'] <= level.right
                )
                
                if row['close'] >= row['open']:
                    footprint[price_level]['buy_volume'] += row['tick_volume']
                else:
                    footprint[price_level]['sell_volume'] += row['tick_volume']
            
            # Calculate imbalance at each level
            imbalances = {}
            for level, volumes in footprint.items():
                total = volumes['buy_volume'] + volumes['sell_volume']
                if total > 0:
                    imbalances[level] = (volumes['buy_volume'] - volumes['sell_volume']) / total
                    
            # Calculate footprint score
            recent_level = next(
                level for level in price_levels
                if df['close'].iloc[-1] >= level.left and df['close'].iloc[-1] <= level.right
            )
            score = abs(imbalances.get(recent_level, 0))
            
            return {
                'score': score,
                'imbalances': imbalances,
                'levels': footprint
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating footprint: {str(e)}")
            return {'score': 0, 'imbalances': {}, 'levels': {}}

    def _calculate_order_imbalance(self, df: pd.DataFrame) -> Dict:
        """Calculate order flow imbalance metrics"""
        try:
            # Calculate trade direction
            trade_direction = np.sign(df['close'] - df['open'])
            
            # Calculate volume-weighted imbalance
            volume_imbalance = (trade_direction * df['tick_volume']).rolling(20).sum()
            normalized_imbalance = volume_imbalance / df['tick_volume'].rolling(20).sum()
            
            # Calculate trade intensity
            trade_intensity = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
            
            # Calculate aggressive order ratio
            aggressive_buys = ((df['close'] > df['open']) & (df['close'] >= df['high'].shift(1)))
            aggressive_sells = ((df['close'] < df['open']) & (df['close'] <= df['low'].shift(1)))
            aggressive_ratio = (
                (aggressive_buys | aggressive_sells).rolling(20).mean()
            )
            
            # Calculate imbalance score
            score = (
                abs(normalized_imbalance.iloc[-1]) * 0.4 +
                min(trade_intensity.iloc[-1], 2) / 2 * 0.3 +
                aggressive_ratio.iloc[-1] * 0.3
            )
            
            return {
                'score': score,
                'signals': {
                    'normalized_imbalance': normalized_imbalance.iloc[-1],
                    'trade_intensity': trade_intensity.iloc[-1],
                    'aggressive_ratio': aggressive_ratio.iloc[-1]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating order imbalance: {str(e)}")
            return {'score': 0, 'signals': {}}

    def _analyze_liquidity(self, df: pd.DataFrame) -> Dict:
        """Analyze market liquidity"""
        try:
            # Calculate basic liquidity metrics
            spread = df['high'] - df['low']
            relative_spread = spread / df['close']
            
            # Calculate volume-based liquidity
            volume_ma = df['tick_volume'].rolling(20).mean()
            volume_stdev = df['tick_volume'].rolling(20).std()
            normalized_volume = (df['tick_volume'] - volume_ma) / volume_stdev
            
            # Calculate price impact
            returns = df['close'].pct_change()
            volume_returns = returns / df['tick_volume']
            price_impact = volume_returns.abs().rolling(20).mean()
            
            # Calculate market depth
            depth = 1 / (relative_spread * price_impact)
            normalized_depth = (depth - depth.rolling(20).mean()) / depth.rolling(20).std()
            
            # Calculate resiliency
            autocorr = returns.rolling(20).apply(
                lambda x: x.autocorr(1) if len(x.dropna()) > 1 else 0
            )
            
            # Calculate liquidity score
            score = (
                (1 - min(relative_spread.iloc[-1] * 100, 1)) * 0.3 +
                (1 - min(price_impact.iloc[-1] * 100, 1)) * 0.3 +
                normalized_depth.iloc[-1] * 0.2 +
                (1 - abs(autocorr.iloc[-1])) * 0.2
            )
            
            return {
                'score': max(min(score, 1), 0),
                'signals': {
                    'relative_spread': relative_spread.iloc[-1],
                    'normalized_volume': normalized_volume.iloc[-1],
                    'price_impact': price_impact.iloc[-1],
                    'market_depth': normalized_depth.iloc[-1],
                    'resiliency': autocorr.iloc[-1]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing liquidity: {str(e)}")
            return {'score': 0, 'signals': {}}

    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure (higher highs/lows, swing points)"""
        try:
            # Find swing highs and lows
            swing_highs = self._find_swing_highs(df)
            swing_lows = self._find_swing_lows(df)
            
            # Analyze structure
            structure = self._analyze_swing_structure(swing_highs, swing_lows)
            
            # Detect structure breaks
            breaks = self._detect_structure_breaks(df, structure)
            
            score = (
                structure['trend_strength'] * 0.5 +
                breaks['break_strength'] * 0.5
            )
            
            return {
                'score': score,
                'signals': {
                    'structure': structure,
                    'breaks': breaks
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in market structure analysis: {str(e)}")
            return {'score': 0, 'signals': {}}

    def _analyze_market_context(self, df: pd.DataFrame) -> Dict:
        """Analyze overall market context"""
        try:
            context = {}
            
            # Analyze market regime
            context['regime'] = self._detect_market_regime(df)
            
            # Analyze volatility state
            context['volatility_state'] = self._analyze_volatility_state(df)
            
            # Analyze trend strength and maturity
            context['trend_state'] = self._analyze_trend_state(df)
            
            # Analyze market efficiency
            context['efficiency'] = self._calculate_market_efficiency(df)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error analyzing market context: {str(e)}")
            return {}

    def _find_swing_points(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Find swing points in price data"""
        try:
            highs = df['high'].rolling(window=window, center=True).apply(
                lambda x: 1 if x.iloc[window//2] == max(x) else 0
            )
            lows = df['low'].rolling(window=window, center=True).apply(
                lambda x: 1 if x.iloc[window//2] == min(x) else 0
            )
            
            swings = pd.DataFrame({
                'high': df['high'][highs == 1],
                'low': df['low'][lows == 1]
            })
            
            return swings
            
        except Exception as e:
            self.logger.error(f"Error finding swing points: {str(e)}")
            return pd.DataFrame()

    def _detect_market_regime(self, df: pd.DataFrame) -> Dict:
        """Detect current market regime (trending, ranging, volatile)"""
        try:
            # Calculate ADX for trend strength
            adx = talib.ADX(df['high'], df['low'], df['close'])
            
            # Calculate volatility
            atr = talib.ATR(df['high'], df['low'], df['close'])
            
            # Determine regime
            if adx.iloc[-1] > 25:
                regime = 'TRENDING'
            elif atr.iloc[-1] > atr.mean() * 1.5:
                regime = 'VOLATILE'
            else:
                regime = 'RANGING'
                
            return {
                'regime': regime,
                'strength': adx.iloc[-1] / 100,
                'volatility': atr.iloc[-1] / atr.mean()
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return {'regime': 'UNKNOWN', 'strength': 0, 'volatility': 0}

    def _calculate_market_efficiency(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate market efficiency ratio"""
        try:
            price_change = abs(df['close'].diff(period).iloc[-1])
            path_length = abs(df['close'].diff()).iloc[-period:].sum()
            
            efficiency = price_change / path_length if path_length > 0 else 0
            return min(efficiency, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating market efficiency: {str(e)}")
            return 0.0

    def _detect_absorption(self, df: pd.DataFrame, delta: pd.Series) -> Dict:
        """Detect absorption levels in order flow"""
        try:
            # Calculate volume-weighted average price
            vwap = (df['high'] + df['low'] + df['close']) / 3 * df['tick_volume']
            vwap = vwap.cumsum() / df['tick_volume'].cumsum()
            
            # Detect absorption zones
            delta_ma = delta.rolling(20).mean()
            volume_ma = df['tick_volume'].rolling(20).mean()
            
            absorption_zone = (
                (abs(delta) < delta_ma.std()) &
                (df['tick_volume'] > volume_ma * 1.5)
            )
            
            # Calculate absorption strength
            strength = (
                absorption_zone.rolling(5).sum() / 5
            ).iloc[-1]
            
            return {
                'score': strength,
                'level': vwap.iloc[-1] if strength > 0.6 else None
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting absorption: {str(e)}")
            return {'score': 0, 'level': None}

    def _calculate_volume_profile(self, df: pd.DataFrame, num_bins: int = 20) -> Dict:
        """Calculate volume profile"""
        try:
            price_range = df['high'].max() - df['low'].min()
            bin_size = price_range / num_bins
            
            # Create price bins
            bins = np.linspace(df['low'].min(), df['high'].max(), num_bins + 1)
            
            # Calculate volume per bin
            volume_profile = pd.cut(df['close'], bins=bins, labels=bins[:-1])
            volume_per_bin = df.groupby(volume_profile)['tick_volume'].sum()
            
            # Find point of control
            poc_price = volume_per_bin.idxmax()
            
            return {
                'profile': volume_per_bin,
                'poc': poc_price,
                'value_area': self._calculate_value_area(volume_per_bin)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {str(e)}")
            return {'profile': None, 'poc': None, 'value_area': None}

    def _find_institutional_levels(self, df: pd.DataFrame, volume_profile: Dict) -> Dict:
        """Identify potential institutional trading levels"""
        try:
            if volume_profile['poc'] is None:
                return {'valid_level': False, 'levels': []}
                
            # Find high volume nodes
            profile = volume_profile['profile']
            mean_volume = profile.mean()
            high_volume_nodes = profile[profile > mean_volume * 1.5]
            
            # Check for price rejection from levels
            levels = []
            for price_level in high_volume_nodes.index:
                rejection = self._check_price_rejection(df, price_level)
                if rejection['valid']:
                    levels.append({
                        'price': price_level,
                        'strength': rejection['strength']
                    })
                    
            return {
                'valid_level': len(levels) > 0,
                'levels': levels
            }
            
        except Exception as e:
            self.logger.error(f"Error finding institutional levels: {str(e)}")
            return {'valid_level': False, 'levels': []}

    def _check_price_rejection(self, df: pd.DataFrame, level: float, threshold: float = 0.001) -> Dict:
        """Check for price rejection from a level"""
        try:
            # Find touches of the level
            touches = (
                (abs(df['high'] - level) < threshold) |
                (abs(df['low'] - level) < threshold)
            )
            
            if not touches.any():
                return {'valid': False, 'strength': 0}
                
            # Calculate rejection strength
            rejection_candles = df[touches].copy()
            
            rejection_strength = (
                rejection_candles['high'].max() - rejection_candles['close']
            ).mean() / level
            
            return {
                'valid': rejection_strength > threshold,
                'strength': min(rejection_strength * 100, 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error checking price rejection: {str(e)}")
            return {'valid': False, 'strength': 0}

    def _find_swing_highs(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """Find swing high points"""
        try:
            highs = df['high'].rolling(window=window, center=True).apply(
                lambda x: x.iloc[window//2] if x.iloc[window//2] == max(x) else np.nan
            )
            return highs.dropna()
            
        except Exception as e:
            self.logger.error(f"Error finding swing highs: {str(e)}")
            return pd.Series()

    def _find_swing_lows(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """Find swing low points"""
        try:
            lows = df['low'].rolling(window=window, center=True).apply(
                lambda x: x.iloc[window//2] if x.iloc[window//2] == min(x) else np.nan
            )
            return lows.dropna()
            
        except Exception as e:
            self.logger.error(f"Error finding swing lows: {str(e)}")
            return pd.Series()

    def _analyze_swing_structure(self, highs: pd.Series, lows: pd.Series) -> Dict:
        """Analyze market structure based on swing points"""
        try:
            if len(highs) < 2 or len(lows) < 2:
                return {'trend': 'UNDEFINED', 'trend_strength': 0}
                
            # Check higher highs and higher lows
            higher_highs = highs.is_monotonic_increasing
            higher_lows = lows.is_monotonic_increasing
            
            # Check lower highs and lower lows
            lower_highs = highs.is_monotonic_decreasing
            lower_lows = lows.is_monotonic_decreasing
            
            # Determine trend structure
            if higher_highs and higher_lows:
                trend = 'UPTREND'
                strength = 1.0
            elif lower_highs and lower_lows:
                trend = 'DOWNTREND'
                strength = 1.0
            elif higher_highs and lower_lows:
                trend = 'EXPANDING'
                strength = 0.5
            elif lower_highs and higher_lows:
                trend = 'CONTRACTING'
                strength = 0.5
            else:
                trend = 'CHOPPY'
                strength = 0.0
                
            return {
                'trend': trend,
                'trend_strength': strength
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing swing structure: {str(e)}")
            return {'trend': 'UNDEFINED', 'trend_strength': 0}

    def _detect_structure_breaks(self, df: pd.DataFrame, structure: Dict) -> Dict:
        """Detect breaks in market structure"""
        try:
            if structure['trend'] == 'UNDEFINED':
                return {'break_detected': False, 'break_strength': 0}
                
            # Calculate recent price action
            recent_high = df['high'].iloc[-5:].max()
            recent_low = df['low'].iloc[-5:].min()
            
            # Check for structure breaks
            break_detected = False
            break_strength = 0
            
            if structure['trend'] == 'UPTREND':
                if recent_low < df['low'].iloc[-10:-5].min():
                    break_detected = True
                    break_strength = abs(
                        (recent_low - df['low'].iloc[-10:-5].min()) /
                        df['low'].iloc[-10:-5].min()
                    )
            elif structure['trend'] == 'DOWNTREND':
                if recent_high > df['high'].iloc[-10:-5].max():
                    break_detected = True
                    break_strength = abs(
                        (recent_high - df['high'].iloc[-10:-5].max()) /
                        df['high'].iloc[-10:-5].max()
                    )
                    
            return {
                'break_detected': break_detected,
                'break_strength': min(break_strength * 10, 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting structure breaks: {str(e)}")
            return {'break_detected': False, 'break_strength': 0}

    def _analyze_volatility_state(self, df: pd.DataFrame) -> Dict:
        """Analyze current volatility state"""
        try:
            # Calculate ATR
            atr = talib.ATR(df['high'], df['low'], df['close'])
            
            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['close'])
            
            # Calculate volatility state
            current_atr = atr.iloc[-1]
            atr_percentile = (atr < current_atr).mean()
            
            bb_width = (upper - lower) / middle
            current_bb_width = bb_width.iloc[-1]
            bb_percentile = (bb_width < current_bb_width).mean()
            
            # Determine volatility state
            if atr_percentile > 0.8 or bb_percentile > 0.8:
                state = 'HIGH'
            elif atr_percentile < 0.2 or bb_percentile < 0.2:
                state = 'LOW'
            else:
                state = 'NORMAL'
                
            return {
                'state': state,
                'atr_percentile': atr_percentile,
                'bb_percentile': bb_percentile
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility state: {str(e)}")
            return {'state': 'UNKNOWN', 'atr_percentile': 0, 'bb_percentile': 0}

    def _analyze_trend_state(self, df: pd.DataFrame) -> Dict:
        """Analyze trend strength and maturity"""
        try:
            # Calculate ADX for trend strength
            adx = talib.ADX(df['high'], df['low'], df['close'])
            
            # Calculate trend duration
            trend_direction = np.sign(df['close'].diff())
            current_trend = trend_direction.iloc[-1]
            trend_duration = len(
                trend_direction[trend_direction == current_trend].iloc[-20:]
            )
            
            # Determine trend maturity
            if trend_duration > 15 and adx.iloc[-1] > 40:
                maturity = 'MATURE'
            elif trend_duration > 10 and adx.iloc[-1] > 25:
                maturity = 'DEVELOPING'
            else:
                maturity = 'EARLY'
                
            return {
                'strength': adx.iloc[-1] / 100,
                'duration': trend_duration,
                'maturity': maturity
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend state: {str(e)}")
            return {'strength': 0, 'duration': 0, 'maturity': 'UNKNOWN'}

    def _calculate_value_area(self, volume_profile: pd.Series, value_area_volume: float = 0.7) -> Tuple[float, float]:
        """Calculate the value area of the volume profile"""
        try:
            total_volume = volume_profile.sum()
            target_volume = total_volume * value_area_volume
            
            # Start from POC and expand
            poc_idx = volume_profile.argmax()
            current_volume = volume_profile.iloc[poc_idx]
            
            lower_idx = upper_idx = poc_idx
            
            while current_volume < target_volume and (lower_idx > 0 or upper_idx < len(volume_profile) - 1):
                lower_vol = volume_profile.iloc[lower_idx - 1] if lower_idx > 0 else 0
                upper_vol = volume_profile.iloc[upper_idx + 1] if upper_idx < len(volume_profile) - 1 else 0
                
                if lower_vol > upper_vol:
                    lower_idx -= 1
                    current_volume += lower_vol
                else:
                    upper_idx += 1
                    current_volume += upper_vol
                    
            return (
                volume_profile.index[lower_idx],
                volume_profile.index[upper_idx]
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating value area: {str(e)}")
            return (None, None)

    def _check_gartley_pattern(self, swings: pd.DataFrame) -> float:
        """Check for Gartley pattern completion"""
        try:
            if len(swings) < 5:
                return 0
                
            # Get last 5 swing points
            points = swings.tail(5)
            
            # Calculate retracement ratios
            xab = abs(points.iloc[2]['high'] - points.iloc[1]['low']) / abs(points.iloc[1]['low'] - points.iloc[0]['high'])
            abc = abs(points.iloc[3]['low'] - points.iloc[2]['high']) / abs(points.iloc[2]['high'] - points.iloc[1]['low'])
            bcd = abs(points.iloc[4]['high'] - points.iloc[3]['low']) / abs(points.iloc[3]['low'] - points.iloc[2]['high'])
            xad = abs(points.iloc[4]['high'] - points.iloc[0]['high'])
            
            # Check Gartley ratios
            is_gartley = (
                0.618 <= xab <= 0.618 and
                0.382 <= abc <= 0.886 and
                1.27 <= bcd <= 1.618 and
                0.786 <= xad <= 0.786
            )
            
            if is_gartley:
                return self._calculate_pattern_strength(points)
            return 0
            
        except Exception as e:
            self.logger.error(f"Error checking Gartley pattern: {str(e)}")
            return 0

    def _check_butterfly_pattern(self, swings: pd.DataFrame) -> float:
        """Check for Butterfly pattern completion"""
        try:
            if len(swings) < 5:
                return 0
                
            # Get last 5 swing points
            points = swings.tail(5)
            
            # Calculate retracement ratios
            xab = abs(points.iloc[2]['high'] - points.iloc[1]['low']) / abs(points.iloc[1]['low'] - points.iloc[0]['high'])
            abc = abs(points.iloc[3]['low'] - points.iloc[2]['high']) / abs(points.iloc[2]['high'] - points.iloc[1]['low'])
            bcd = abs(points.iloc[4]['high'] - points.iloc[3]['low']) / abs(points.iloc[3]['low'] - points.iloc[2]['high'])
            xad = abs(points.iloc[4]['high'] - points.iloc[0]['high'])
            
            # Check Butterfly ratios
            is_butterfly = (
                0.786 <= xab <= 0.786 and
                0.382 <= abc <= 0.886 and
                1.618 <= bcd <= 2.618 and
                1.27 <= xad <= 1.618
            )
            
            if is_butterfly:
                return self._calculate_pattern_strength(points)
            return 0
            
        except Exception as e:
            self.logger.error(f"Error checking Butterfly pattern: {str(e)}")
            return 0

    def _check_bat_pattern(self, swings: pd.DataFrame) -> float:
        """Check for Bat pattern completion"""
        try:
            if len(swings) < 5:
                return 0
                
            # Get last 5 swing points
            points = swings.tail(5)
            
            # Calculate retracement ratios
            xab = abs(points.iloc[2]['high'] - points.iloc[1]['low']) / abs(points.iloc[1]['low'] - points.iloc[0]['high'])
            abc = abs(points.iloc[3]['low'] - points.iloc[2]['high']) / abs(points.iloc[2]['high'] - points.iloc[1]['low'])
            bcd = abs(points.iloc[4]['high'] - points.iloc[3]['low']) / abs(points.iloc[3]['low'] - points.iloc[2]['high'])
            xad = abs(points.iloc[4]['high'] - points.iloc[0]['high'])
            
            # Check Bat ratios
            is_bat = (
                0.382 <= xab <= 0.5 and
                0.382 <= abc <= 0.886 and
                1.618 <= bcd <= 2.618 and
                0.886 <= xad <= 0.886
            )
            
            if is_bat:
                return self._calculate_pattern_strength(points)
            return 0
            
        except Exception as e:
            self.logger.error(f"Error checking Bat pattern: {str(e)}")
            return 0

    def _check_crab_pattern(self, swings: pd.DataFrame) -> float:
        """Check for Crab pattern completion"""
        try:
            if len(swings) < 5:
                return 0
                
            # Get last 5 swing points
            points = swings.tail(5)
            
            # Calculate retracement ratios
            xab = abs(points.iloc[2]['high'] - points.iloc[1]['low']) / abs(points.iloc[1]['low'] - points.iloc[0]['high'])
            abc = abs(points.iloc[3]['low'] - points.iloc[2]['high']) / abs(points.iloc[2]['high'] - points.iloc[1]['low'])
            bcd = abs(points.iloc[4]['high'] - points.iloc[3]['low']) / abs(points.iloc[3]['low'] - points.iloc[2]['high'])
            xad = abs(points.iloc[4]['high'] - points.iloc[0]['high'])
            
            # Check Crab ratios
            is_crab = (
                0.382 <= xab <= 0.618 and
                0.382 <= abc <= 0.886 and
                2.618 <= bcd <= 3.618 and
                1.618 <= xad <= 1.618
            )
            
            if is_crab:
                return self._calculate_pattern_strength(points)
            return 0
            
        except Exception as e:
            self.logger.error(f"Error checking Crab pattern: {str(e)}")
            return 0

    def _calculate_pattern_strength(self, points: pd.DataFrame) -> float:
        """Calculate the strength of a harmonic pattern"""
        try:
            # Calculate pattern size relative to recent price movement
            pattern_size = abs(points.iloc[-1]['high'] - points.iloc[0]['high'])
            avg_candle_size = abs(points['high'] - points['low']).mean()
            
            # Calculate pattern symmetry
            leg_sizes = []
            for i in range(len(points) - 1):
                leg_sizes.append(abs(points.iloc[i+1]['high'] - points.iloc[i]['high']))
            leg_symmetry = min(leg_sizes) / max(leg_sizes) if max(leg_sizes) > 0 else 0
            
            # Calculate time symmetry
            time_diffs = []
            for i in range(len(points) - 1):
                time_diffs.append(abs((points.index[i+1] - points.index[i]).total_seconds()))
            time_symmetry = min(time_diffs) / max(time_diffs) if max(time_diffs) > 0 else 0
            
            # Combine factors
            strength = (
                (pattern_size / (avg_candle_size * 10)) * 0.4 +
                leg_symmetry * 0.3 +
                time_symmetry * 0.3
            )
            
            return min(strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern strength: {str(e)}")
            return 0

    def _detect_ml_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect patterns using machine learning"""
        try:
            # Prepare features
            features = self._prepare_ml_features(df)
            
            # Generate labels for training
            labels = self._generate_pattern_labels(df)
            
            # Split data for training
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, shuffle=False
            )
            
            # Train models
            self.random_forest.fit(X_train, y_train)
            self.xgb_model.fit(X_train, y_train)
            self.lgb_model.fit(X_train, y_train)
            
            # Get predictions
            rf_pred = self.random_forest.predict_proba(features.iloc[-1:])
            xgb_pred = self.xgb_model.predict_proba(features.iloc[-1:])
            lgb_pred = self.lgb_model.predict_proba(features.iloc[-1:])
            
            # Weighted ensemble prediction
            ensemble_pred = (
                rf_pred * self.model_weights['random_forest'] +
                xgb_pred * self.model_weights['xgboost'] +
                lgb_pred * self.model_weights['lightgbm']
            )
            
            # Get feature importance
            feature_importance = self._get_feature_importance()
            
            return {
                'ensemble_prediction': ensemble_pred.tolist(),
                'feature_importance': feature_importance,
                'confidence': self._calculate_ml_confidence(ensemble_pred)
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML pattern detection: {str(e)}")
            return {'confidence': 0, 'signals': {}}

    def _prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML analysis"""
        try:
            features = pd.DataFrame()
            
            # Price features
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            features['volatility'] = features['returns'].rolling(20).std()
            
            # Volume features
            features['volume_ma'] = df['tick_volume'].rolling(20).mean()
            features['volume_std'] = df['tick_volume'].rolling(20).std()
            features['volume_ratio'] = df['tick_volume'] / features['volume_ma']
            
            # Technical indicators
            features['rsi'] = talib.RSI(df['close'])
            features['macd'], _, _ = talib.MACD(df['close'])
            features['adx'] = talib.ADX(df['high'], df['low'], df['close'])
            features['atr'] = talib.ATR(df['high'], df['low'], df['close'])
            features['cci'] = talib.CCI(df['high'], df['low'], df['close'])
            
            # Clean and scale features
            features = features.dropna()
            features = pd.DataFrame(
                self.scaler.fit_transform(features),
                columns=features.columns,
                index=features.index
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing ML features: {str(e)}")
            return pd.DataFrame()

    def _generate_pattern_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate labels for pattern detection"""
        try:
            # Calculate future returns
            future_returns = df['close'].shift(-5).pct_change(5)
            
            # Generate labels based on return thresholds
            labels = np.zeros(len(df))
            labels[future_returns > 0.01] = 1  # Bullish pattern
            labels[future_returns < -0.01] = 2  # Bearish pattern
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Error generating pattern labels: {str(e)}")
            return np.zeros(len(df))

    def _get_feature_importance(self) -> Dict:
        """Get feature importance from all models"""
        try:
            rf_importance = dict(zip(
                self.random_forest.feature_names_in_,
                self.random_forest.feature_importances_
            ))
            xgb_importance = dict(zip(
                self.xgb_model.feature_names_in_,
                self.xgb_model.feature_importances_
            ))
            lgb_importance = dict(zip(
                self.lgb_model.feature_name_,
                self.lgb_model.feature_importances_
            ))
            
            # Combine feature importance with weights
            combined_importance = {}
            for feature in rf_importance.keys():
                combined_importance[feature] = (
                    rf_importance[feature] * self.model_weights['random_forest'] +
                    xgb_importance[feature] * self.model_weights['xgboost'] +
                    lgb_importance[feature] * self.model_weights['lightgbm']
                )
            
            return combined_importance
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return {}

    def _calculate_ml_confidence(self, ensemble_pred: np.ndarray) -> float:
        """Calculate ML prediction confidence"""
        try:
            # Get maximum probability from ensemble prediction
            max_prob = np.max(ensemble_pred)
            
            # Scale confidence to be between 0 and 1
            confidence = min(max(max_prob, 0), 1)
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating ML confidence: {str(e)}")
            return 0.0

    def _generate_visualizations(self, df: pd.DataFrame, signals: Dict) -> Dict:
        """Generate interactive visualizations"""
        try:
            visualizations = {}
            
            # Create main chart
            main_chart = self._create_main_chart(df, signals)
            visualizations['main_chart'] = main_chart
            
            # Create order flow chart
            order_flow = self._create_order_flow_chart(df, signals)
            visualizations['order_flow'] = order_flow
            
            # Create pattern chart
            pattern_chart = self._create_pattern_chart(df, signals)
            visualizations['pattern_chart'] = pattern_chart
            
            # Create market quality dashboard
            quality_dashboard = self._create_quality_dashboard(df, signals)
            visualizations['quality_dashboard'] = quality_dashboard
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            return {}

    def _create_main_chart(self, df: pd.DataFrame, signals: Dict) -> go.Figure:
        """Create main trading chart"""
        try:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add volume
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['tick_volume'],
                    name='Volume'
                ),
                row=2, col=1
            )
            
            # Add indicators
            self._add_indicators_to_chart(fig, df, signals)
            
            # Add patterns
            self._add_patterns_to_chart(fig, df, signals)
            
            # Update layout
            fig.update_layout(
                title='Market Analysis',
                xaxis_title='Time',
                yaxis_title='Price',
                height=800
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating main chart: {str(e)}")
            return None

    def _create_order_flow_chart(self, df: pd.DataFrame, signals: Dict) -> go.Figure:
        """Create order flow visualization"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Footprint Chart',
                    'Volume Profile',
                    'Order Flow Imbalance',
                    'Liquidity Heatmap'
                )
            )
            
            # Add footprint chart
            self._add_footprint_to_chart(fig, signals['order_flow']['footprint'], row=1, col=1)
            
            # Add volume profile
            self._add_volume_profile_to_chart(fig, signals['order_flow']['volume_profile'], row=1, col=2)
            
            # Add order flow imbalance
            self._add_imbalance_to_chart(fig, signals['order_flow']['imbalance'], row=2, col=1)
            
            # Add liquidity heatmap
            self._add_liquidity_heatmap_to_chart(fig, signals['order_flow']['liquidity'], row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title='Order Flow Analysis',
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating order flow chart: {str(e)}")
            return None

    def _create_pattern_chart(self, df: pd.DataFrame, signals: Dict) -> go.Figure:
        """Create pattern visualization"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Harmonic Patterns',
                    'Market Structure',
                    'ML Pattern Detection',
                    'Regime Analysis'
                )
            )
            
            # Add harmonic patterns
            self._add_harmonic_patterns_to_chart(fig, signals['harmonic'], row=1, col=1)
            
            # Add market structure
            self._add_structure_to_chart(fig, signals['market_structure'], row=1, col=2)
            
            # Add ML patterns
            self._add_ml_patterns_to_chart(fig, signals['ml_patterns'], row=2, col=1)
            
            # Add regime analysis
            self._add_regime_analysis_to_chart(fig, signals['market_context']['regime'], row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title='Pattern Analysis',
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating pattern chart: {str(e)}")
            return None

    def _create_quality_dashboard(self, df: pd.DataFrame, signals: Dict) -> go.Figure:
        """Create market quality dashboard"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Market Efficiency',
                    'Liquidity Analysis',
                    'Volatility Profile',
                    'Market Quality Metrics'
                )
            )
            
            # Add market efficiency
            self._add_efficiency_to_chart(fig, signals['microstructure']['efficiency'], row=1, col=1)
            
            # Add liquidity analysis
            self._add_liquidity_analysis_to_chart(fig, signals['microstructure']['liquidity'], row=1, col=2)
            
            # Add volatility profile
            self._add_volatility_profile_to_chart(fig, signals['volatility'], row=2, col=1)
            
            # Add market quality metrics
            self._add_quality_metrics_to_chart(fig, signals['microstructure']['quality'], row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title='Market Quality Analysis',
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating quality dashboard: {str(e)}")
            return None

    def _add_footprint_to_chart(self, fig: go.Figure, footprint: Dict, row: int, col: int) -> None:
        """Add footprint chart to the figure"""
        try:
            # Implementation of _add_footprint_to_chart method
            pass
        except Exception as e:
            self.logger.error(f"Error adding footprint to chart: {str(e)}")

    def _add_volume_profile_to_chart(self, fig: go.Figure, volume_profile: Dict, row: int, col: int) -> None:
        """Add volume profile chart to the figure"""
        try:
            # Implementation of _add_volume_profile_to_chart method
            pass
        except Exception as e:
            self.logger.error(f"Error adding volume profile to chart: {str(e)}")

    def _add_imbalance_to_chart(self, fig: go.Figure, imbalance: Dict, row: int, col: int) -> None:
        """Add order flow imbalance chart to the figure"""
        try:
            # Implementation of _add_imbalance_to_chart method
            pass
        except Exception as e:
            self.logger.error(f"Error adding imbalance to chart: {str(e)}")

    def _add_liquidity_heatmap_to_chart(self, fig: go.Figure, liquidity: Dict, row: int, col: int) -> None:
        """Add liquidity heatmap to the figure"""
        try:
            # Implementation of _add_liquidity_heatmap_to_chart method
            pass
        except Exception as e:
            self.logger.error(f"Error adding liquidity heatmap to chart: {str(e)}")

    def _add_harmonic_patterns_to_chart(self, fig: go.Figure, harmonic_patterns: Dict, row: int, col: int) -> None:
        """Add harmonic patterns chart to the figure"""
        try:
            # Implementation of _add_harmonic_patterns_to_chart method
            pass
        except Exception as e:
            self.logger.error(f"Error adding harmonic patterns to chart: {str(e)}")

    def _add_structure_to_chart(self, fig: go.Figure, structure: Dict, row: int, col: int) -> None:
        """Add market structure chart to the figure"""
        try:
            # Implementation of _add_structure_to_chart method
            pass
        except Exception as e:
            self.logger.error(f"Error adding structure to chart: {str(e)}")

    def _add_ml_patterns_to_chart(self, fig: go.Figure, ml_patterns: Dict, row: int, col: int) -> None:
        """Add ML patterns chart to the figure"""
        try:
            # Implementation of _add_ml_patterns_to_chart method
            pass
        except Exception as e:
            self.logger.error(f"Error adding ML patterns to chart: {str(e)}")

    def _add_regime_analysis_to_chart(self, fig: go.Figure, regime: str, row: int, col: int) -> None:
        """Add regime analysis chart to the figure"""
        try:
            # Implementation of _add_regime_analysis_to_chart method
            pass
        except Exception as e:
            self.logger.error(f"Error adding regime analysis to chart: {str(e)}")

    def _add_efficiency_to_chart(self, fig: go.Figure, efficiency: Dict, row: int, col: int) -> None:
        """Add market efficiency chart to the figure"""
        try:
            # Implementation of _add_efficiency_to_chart method
            pass
        except Exception as e:
            self.logger.error(f"Error adding efficiency to chart: {str(e)}")

    def _add_liquidity_analysis_to_chart(self, fig: go.Figure, liquidity: Dict, row: int, col: int) -> None:
        """Add liquidity analysis chart to the figure"""
        try:
            # Implementation of _add_liquidity_analysis_to_chart method
            pass
        except Exception as e:
            self.logger.error(f"Error adding liquidity analysis to chart: {str(e)}")

    def _add_volatility_profile_to_chart(self, fig: go.Figure, volatility: Dict, row: int, col: int) -> None:
        """Add volatility profile chart to the figure"""
        try:
            # Implementation of _add_volatility_profile_to_chart method
            pass
        except Exception as e:
            self.logger.error(f"Error adding volatility profile to chart: {str(e)}")

    def _add_quality_metrics_to_chart(self, fig: go.Figure, quality: Dict, row: int, col: int) -> None:
        """Add market quality metrics chart to the figure"""
        try:
            # Implementation of _add_quality_metrics_to_chart method
            pass
        except Exception as e:
            self.logger.error(f"Error adding quality metrics to chart: {str(e)}")

    def _detect_anomaly_clusters(self, features: pd.DataFrame, anomalies: Dict) -> Dict:
        """Detect anomaly clusters"""
        try:
            # Implementation of _detect_anomaly_clusters method
            pass
        except Exception as e:
            self.logger.error(f"Error detecting anomaly clusters: {str(e)}")
            return {}

    def _detect_trend_patterns(self, features: pd.DataFrame) -> Dict:
        """Detect trend patterns"""
        try:
            # Implementation of _detect_trend_patterns method
            pass
        except Exception as e:
            self.logger.error(f"Error detecting trend patterns: {str(e)}")
            return {}

    def _detect_reversal_patterns(self, features: pd.DataFrame) -> Dict:
        """Detect reversal patterns"""
        try:
            # Implementation of _detect_reversal_patterns method
            pass
        except Exception as e:
            self.logger.error(f"Error detecting reversal patterns: {str(e)}")
            return {}

    def _detect_continuation_patterns(self, features: pd.DataFrame) -> Dict:
        """Detect continuation patterns"""
        try:
            # Implementation of _detect_continuation_patterns method
            pass
        except Exception as e:
            self.logger.error(f"Error detecting continuation patterns: {str(e)}")
            return {}

    def _detect_regime_transitions(self, features: pd.DataFrame) -> Dict:
        """Detect regime transitions"""
        try:
            # Implementation of _detect_regime_transitions method
            pass
        except Exception as e:
            self.logger.error(f"Error detecting regime transitions: {str(e)}")
            return {}

    def _calculate_trend_probability(self, features: pd.DataFrame) -> float:
        """Calculate trend probability based on multiple indicators
        
        Args:
            features (pd.DataFrame): DataFrame containing price and indicator data
            
        Returns:
            float: Probability of current market being in a trend (0-1)
        """
        try:
            # Calculate ADX for trend strength
            adx = features.get('adx', pd.Series(dtype=float))
            
            # Calculate moving average convergence/divergence
            macd = features.get('macd', pd.Series(dtype=float))
            macd_signal = features.get('macd_signal', pd.Series(dtype=float))
            
            # Calculate directional movement
            plus_di = features.get('plus_di', pd.Series(dtype=float))
            minus_di = features.get('minus_di', pd.Series(dtype=float))
            
            # Combine signals
            trend_signals = []
            
            # ADX above 25 indicates strong trend
            if not adx.empty:
                trend_signals.append(min(adx.iloc[-1] / 50.0, 1.0))
                
            # MACD signal line crossover
            if not (macd.empty or macd_signal.empty):
                macd_cross = (macd.iloc[-1] - macd_signal.iloc[-1]) / macd.std()
                trend_signals.append(abs(min(macd_cross, 1.0)))
                
            # DMI signals
            if not (plus_di.empty or minus_di.empty):
                di_diff = abs(plus_di.iloc[-1] - minus_di.iloc[-1]) / 40.0
                trend_signals.append(min(di_diff, 1.0))
                
            # Calculate final probability
            if trend_signals:
                return sum(trend_signals) / len(trend_signals)
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating trend probability: {str(e)}")
            return 0.5

    def _calculate_range_probability(self, features: pd.DataFrame) -> float:
        """Calculate ranging market probability based on multiple indicators
        
        Args:
            features (pd.DataFrame): DataFrame containing price and indicator data
            
        Returns:
            float: Probability of current market being in a range (0-1)
        """
        try:
            # Get Bollinger Bands
            bb_upper = features.get('bb_upper', pd.Series(dtype=float))
            bb_lower = features.get('bb_lower', pd.Series(dtype=float))
            bb_middle = features.get('bb_middle', pd.Series(dtype=float))
            
            # Get RSI
            rsi = features.get('rsi', pd.Series(dtype=float))
            
            # Calculate signals
            range_signals = []
            
            # Check price contained within Bollinger Bands
            if not (bb_upper.empty or bb_lower.empty or bb_middle.empty):
                price = features['close'].iloc[-1]
                bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
                containment = 1 - min(abs(price - bb_middle.iloc[-1]) / (bb_upper.iloc[-1] - bb_middle.iloc[-1]), 1)
                range_signals.append(containment)
                range_signals.append(1 - min(bb_width / 0.05, 1.0))  # Narrow bands indicate range
                
            # RSI between 40-60 indicates range
            if not rsi.empty:
                rsi_val = rsi.iloc[-1]
                rsi_range = 1 - min(abs(rsi_val - 50) / 30.0, 1.0)
                range_signals.append(rsi_range)
                
            # Calculate final probability
            if range_signals:
                return sum(range_signals) / len(range_signals)
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating range probability: {str(e)}")
            return 0.5

    def _calculate_volatility_probability(self, features: pd.DataFrame) -> float:
        """Calculate market volatility probability based on multiple indicators
        
        Args:
            features (pd.DataFrame): DataFrame containing price and indicator data
            
        Returns:
            float: Probability of current market being volatile (0-1)
        """
        try:
            # Get ATR and historical volatility
            atr = features.get('atr', pd.Series(dtype=float))
            hist_vol = features.get('historical_volatility', pd.Series(dtype=float))
            
            # Get Bollinger Bands
            bb_upper = features.get('bb_upper', pd.Series(dtype=float))
            bb_lower = features.get('bb_lower', pd.Series(dtype=float))
            bb_middle = features.get('bb_middle', pd.Series(dtype=float))
            
            # Calculate volatility signals
            vol_signals = []
            
            # ATR relative to price
            if not atr.empty:
                rel_atr = atr.iloc[-1] / features['close'].iloc[-1]
                vol_signals.append(min(rel_atr / 0.02, 1.0))  # Scale ATR
                
            # Historical volatility 
            if not hist_vol.empty:
                vol_signals.append(min(hist_vol.iloc[-1] / 30.0, 1.0))
                
            # Bollinger Band width
            if not (bb_upper.empty or bb_lower.empty or bb_middle.empty):
                bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
                vol_signals.append(min(bb_width / 0.05, 1.0))
                
            # Calculate final probability
            if vol_signals:
                return sum(vol_signals) / len(vol_signals)
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility probability: {str(e)}")
            return 0.5

   


