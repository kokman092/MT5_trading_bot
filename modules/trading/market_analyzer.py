import logging
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import ta
import pandas_ta as pta
from .signal import Signal
from dataclasses import dataclass

@dataclass
class MarketAnalysis:
    trend: str  # 'up', 'down', 'sideways'
    strength: float  # 0.0 to 1.0
    support_levels: List[float]
    resistance_levels: List[float]
    volatility: float
    momentum: float
    volume_profile: Dict
    key_levels: Dict
    signals: Dict

class MarketAnalyzer:
    def __init__(self, config: Dict):
        """Initialize market analyzer with professional trading configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._data_cache = {}
        self._last_update = {}
        self._initialize_indicators()
        
        # Initialize ML analyzer
        from .ml_analyzer import MLAnalyzer
        self.ml_analyzer = MLAnalyzer(config)
        
    def _initialize_indicators(self):
        """Initialize technical indicators"""
        self.indicators = {
            'trend': {
                'ema': self.config['market_analysis']['technical_indicators']['moving_averages'],
                'supertrend': {'period': 10, 'multiplier': 3.0}
            },
            'momentum': {
                'rsi': self.config['market_analysis']['technical_indicators']['rsi'],
                'macd': {'fast': 12, 'slow': 26, 'signal': 9}
            },
            'volatility': {
                'atr': self.config['market_analysis']['technical_indicators']['atr'],
                'bollinger': self.config['market_analysis']['technical_indicators']['bollinger']
            },
            'volume': self.config['market_analysis'].get('volume_analysis', {
                'sma_period': 20,
                'threshold': 1.5
            })
        }
        
    async def analyze_market(self, symbol: str, timeframe: str = 'H1') -> Optional[Dict]:
        """Analyze market for a given symbol and timeframe"""
        try:
            # Update current symbol and timeframe in config
            self.config['trading']['current_symbol'] = symbol
            self.config['trading']['current_timeframe'] = timeframe
            
            # Get market data
            market_data = await self._get_market_data(symbol, timeframe, 1000)
            if market_data is None:
                self.logger.error(f"Failed to get market data for {symbol}")
                return None
                
            # Analyze market structure
            market_structure = await self._analyze_market_structure(market_data, timeframe)
            if market_structure is None:
                self.logger.error(f"Failed to analyze market structure for {symbol}")
                return None
                
            # Return analysis results
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now(),
                'market_structure': market_structure
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market for {symbol}: {str(e)}")
            return None
            
    async def _analyze_market_structure(self, data: pd.DataFrame, timeframe: str = 'H1') -> Optional[Dict]:
        """Analyze market structure and return key metrics."""
        try:
            if data is None or len(data) == 0:
                self.logger.error("Invalid market data provided")
                return None
                
            # Calculate volatility
            volatility = self._calculate_volatility(data, timeframe)
            
            # Determine market phase
            market_phase = self._determine_market_phase(data)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(data)
            
            # Find support and resistance levels
            support_levels = self._find_support_levels(data)
            resistance_levels = self._find_resistance_levels(data)
            
            # Return analysis results
            return {
                'volatility': volatility,
                'market_phase': market_phase,
                'trend_strength': trend_strength,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Market structure analysis error: {str(e)}")
            return None
            
    async def _get_market_data(self, symbol: str, timeframe: str, bars: int = 1000) -> Optional[pd.DataFrame]:
        """Get market data with validation"""
        try:
            # Ensure symbol is selected
            if not mt5.symbol_select(symbol, True):
                self.logger.error(f"Failed to select symbol {symbol}")
                return None
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Failed to get symbol info for {symbol}")
                return None
            
            # Get raw data
            rates = mt5.copy_rates_from_pos(symbol, self._get_timeframe_value(timeframe), 0, bars)
            if rates is None:
                self.logger.error(f"Failed to get rates for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            try:
                # Trend indicators
                df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
                df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
                df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
                
                # RSI
                df['rsi'] = ta.momentum.rsi(df['close'], window=14)
                
                # MACD
                macd_indicator = ta.trend.MACD(
                    close=df['close'],
                    window_slow=26,
                    window_fast=12,
                    window_sign=9
                )
                df['macd'] = macd_indicator.macd()
                df['macd_signal'] = macd_indicator.macd_signal()
                
                # Bollinger Bands
                bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
                df['bollinger_upper'] = bollinger.bollinger_hband()
                df['bollinger_lower'] = bollinger.bollinger_lband()
                
                # ATR
                df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
                
                # Forward fill NaN values that might appear at the beginning
                df.ffill(inplace=True)
                # Fill any remaining NaN values with 0
                df.fillna(0, inplace=True)
                
            except Exception as e:
                self.logger.error(f"Error calculating technical indicators: {str(e)}")
                return None
            
            # Validate data
            if not self._validate_market_data(df):
                self.logger.error(f"Market data validation failed for {symbol}")
                return None
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None
            
    def _validate_market_data(self, df: pd.DataFrame) -> bool:
        """Validate market data quality with robust error handling"""
        try:
            config = self.config['market_analysis']['validation']
            validation_errors = []
            
            # Check if DataFrame is empty
            if df is None or df.empty:
                self.logger.error("Empty market data")
                return False
                
            # Check minimum data points
            min_points = config.get('min_data_points', 100)
            if len(df) < min_points:
                validation_errors.append(f"Insufficient data points: {len(df)} < {min_points}")
                
            # Check for required price columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_errors.append(f"Missing required price columns: {missing_columns}")
                
            # Check for gaps in time series
            if 'time' in df.columns:
                time_diff = df['time'].diff()
                max_gap_hours = config.get('max_gap', 24)
                if time_diff.max().total_seconds() > max_gap_hours * 3600:
                    validation_errors.append(
                        f"Data gap detected: {time_diff.max().total_seconds() / 3600:.2f} hours"
                    )
                    
            # Validate price data if available
            if all(col in df.columns for col in required_columns):
                # Check for non-positive prices
                if (df[required_columns] <= 0).any().any():
                    validation_errors.append("Invalid price values detected (zero or negative)")
                    
                # Check high/low relationship
                if (df['high'] < df['low']).any():
                    validation_errors.append("Invalid high/low price relationship detected")
                    
                # Check for extreme price changes
                max_price_change = config.get('max_price_change_percent', 10)
                # Adjust max price change for gold
                if 'symbol' in df.columns and 'XAU' in df['symbol'].iloc[0]:
                    max_price_change *= 2  # Gold can be more volatile
                price_changes = df['close'].pct_change().abs() * 100
                if (price_changes > max_price_change).any():
                    validation_errors.append(
                        f"Extreme price changes detected: {price_changes.max():.2f}% > {max_price_change}%"
                    )
                    
            # Handle volume data
            if 'volume' in df.columns:
                # If volume is all zeros, try to use tick volume
                if (df['volume'] == 0).all() and 'tick_volume' in df.columns:
                    df['volume'] = df['tick_volume']
                    self.logger.warning("Using tick volume instead of real volume")
                elif (df['volume'] == 0).all():
                    df['volume'] = 1
                    self.logger.warning("No valid volume data, using placeholder values")
                    
                # Check for minimum volume if configured
                min_volume = config.get('min_volume', 0)
                # Adjust minimum volume for gold
                if 'symbol' in df.columns and 'XAU' in df['symbol'].iloc[0]:
                    min_volume = min_volume / 100  # Gold typically has lower volume
                if min_volume > 0 and df['volume'].mean() < min_volume:
                    self.logger.warning(f"Average volume low: {df['volume'].mean():.2f} < {min_volume}")
            else:
                # If no volume data, try to use tick volume
                if 'tick_volume' in df.columns:
                    df['volume'] = df['tick_volume']
                    self.logger.warning("Using tick volume as volume data")
                else:
                    # If no volume data available, use placeholder
                    df['volume'] = 1
                    self.logger.warning("No volume data available, using placeholder values")
                    
            # Log validation errors if any
            if validation_errors:
                for error in validation_errors:
                    self.logger.error(error)
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating market data: {str(e)}")
            return False
            
    def _analyze_trend(self, data: pd.DataFrame) -> str:
        """Analyze market trend"""
        try:
            # Calculate EMAs
            data['ema20'] = pta.ema(data['close'], length=20)
            data['ema50'] = pta.ema(data['close'], length=50)
            data['ema200'] = pta.ema(data['close'], length=200)
            
            # Get latest values
            current_price = data['close'].iloc[-1]
            current_ema20 = data['ema20'].iloc[-1]
            current_ema50 = data['ema50'].iloc[-1]
            current_ema200 = data['ema200'].iloc[-1]
            
            # Determine trend
            if current_ema20 > current_ema50 > current_ema200:
                return 'up'
            elif current_ema20 < current_ema50 < current_ema200:
                return 'down'
            else:
                return 'sideways'
                
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {str(e)}")
            return 'sideways'
            
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength"""
        try:
            # Calculate ADX
            adx = pta.adx(data['high'], data['low'], data['close'], length=14)
            current_adx = adx['ADX_14'].iloc[-1]
            
            # Normalize ADX to 0-1 range
            strength = min(current_adx / 50.0, 1.0)
            
            return strength
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.0
            
    def _find_support_levels(self, data: pd.DataFrame) -> List[float]:
        """Find support levels"""
        try:
            # Calculate swing lows
            window = 20
            lows = data['low']
            support_levels = []
            
            for i in range(window, len(data) - window):
                if self._is_swing_low(lows, i, window):
                    support_levels.append(lows[i])
                    
            # Remove duplicates and sort
            support_levels = sorted(list(set(support_levels)))
            
            return support_levels
            
        except Exception as e:
            self.logger.error(f"Error finding support levels: {str(e)}")
            return []
            
    def _find_resistance_levels(self, data: pd.DataFrame) -> List[float]:
        """Find resistance levels"""
        try:
            # Calculate swing highs
            window = 20
            highs = data['high']
            resistance_levels = []
            
            for i in range(window, len(data) - window):
                if self._is_swing_high(highs, i, window):
                    resistance_levels.append(highs[i])
                    
            # Remove duplicates and sort
            resistance_levels = sorted(list(set(resistance_levels)))
            
            return resistance_levels
            
        except Exception as e:
            self.logger.error(f"Error finding resistance levels: {str(e)}")
            return []
            
    def _is_swing_low(self, data: pd.Series, index: int, window: int) -> bool:
        """Check if point is swing low"""
        for i in range(index - window, index):
            if data[i] < data[index]:
                return False
        for i in range(index + 1, index + window + 1):
            if data[i] < data[index]:
                return False
        return True
        
    def _is_swing_high(self, data: pd.Series, index: int, window: int) -> bool:
        """Check if point is swing high"""
        for i in range(index - window, index):
            if data[i] > data[index]:
                return False
        for i in range(index + 1, index + window + 1):
            if data[i] > data[index]:
                return False
        return True
        
    def _calculate_volatility(self, data: pd.DataFrame, timeframe: str = 'H1') -> float:
        """Calculate market volatility using ATR"""
        try:
            # Calculate ATR
            atr = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)
            current_atr = atr.iloc[-1]
            avg_price = data['close'].mean()
            
            # Normalize ATR
            normalized_atr = current_atr / avg_price
            
            # Apply timeframe multiplier
            timeframe_multipliers = {
                'M1': 1.0,
                'M5': 0.8,
                'M15': 0.7,
                'M30': 0.6,
                'H1': 0.5,
                'H4': 0.4,
                'D1': 0.3
            }
            multiplier = timeframe_multipliers.get(timeframe, 0.5)
            
            # Adjust volatility calculation for gold
            if 'symbol' in data.columns and 'XAU' in data['symbol'].iloc[0]:
                # Gold typically has higher nominal volatility due to its price
                normalized_atr = normalized_atr / 100
            
            return normalized_atr * multiplier
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0
            
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum using RSI"""
        try:
            rsi = pta.rsi(data['close'], length=14)
            current_rsi = rsi.iloc[-1]
            
            # Normalize RSI to -1 to 1 range
            momentum = (current_rsi - 50) / 50
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum: {str(e)}")
            return 0.0
            
    def _analyze_volume_profile(self, data: pd.DataFrame) -> Dict:
        """Analyze volume profile"""
        try:
            if 'volume' not in data.columns:
                return {}
                
            # Calculate volume profile
            price_bins = pd.qcut(data['close'], q=10)
            volume_profile = data.groupby(price_bins)['volume'].sum()
            
            # Convert to dictionary
            profile = {
                'price_levels': list(volume_profile.index.map(str)),
                'volumes': list(volume_profile.values)
            }
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume profile: {str(e)}")
            return {}
            
    def _identify_key_levels(self, data: pd.DataFrame) -> Dict:
        """Identify key price levels"""
        try:
            levels = {
                'daily_pivot': self._calculate_pivot_points(data),
                'fibonacci_levels': self._calculate_fibonacci_levels(data),
                'volume_nodes': self._find_volume_nodes(data),
                'psychological_levels': self._find_psychological_levels(data)
            }
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error identifying key levels: {str(e)}")
            return {}
            
    def _calculate_pivot_points(self, data: pd.DataFrame) -> Dict:
        """Calculate pivot points"""
        try:
            high = data['high'].iloc[-1]
            low = data['low'].iloc[-1]
            close = data['close'].iloc[-1]
            
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            
            return {
                'pivot': pivot,
                'r1': r1,
                'r2': r2,
                's1': s1,
                's2': s2
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating pivot points: {str(e)}")
            return {}
            
    def _calculate_fibonacci_levels(self, data: pd.DataFrame) -> Dict:
        """Calculate Fibonacci levels"""
        try:
            high = data['high'].max()
            low = data['low'].min()
            diff = high - low
            
            levels = {
                '0.0': low,
                '0.236': low + 0.236 * diff,
                '0.382': low + 0.382 * diff,
                '0.5': low + 0.5 * diff,
                '0.618': low + 0.618 * diff,
                '0.786': low + 0.786 * diff,
                '1.0': high
            }
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci levels: {str(e)}")
            return {}
            
    def _find_volume_nodes(self, data: pd.DataFrame) -> List[float]:
        """Find significant volume nodes"""
        try:
            if 'volume' not in data.columns:
                return []
                
            # Calculate volume-weighted price levels
            price_volume = data['close'] * data['volume']
            total_volume = data['volume'].sum()
            vwap = price_volume.sum() / total_volume
            
            # Find price levels with high volume
            volume_threshold = data['volume'].mean() + data['volume'].std()
            high_volume_points = data[data['volume'] > volume_threshold]['close']
            
            return list(high_volume_points)
            
        except Exception as e:
            self.logger.error(f"Error finding volume nodes: {str(e)}")
            return []
            
    def _find_psychological_levels(self, data: pd.DataFrame) -> List[float]:
        """Find psychological price levels"""
        try:
            # Get price range
            price_min = np.floor(data['low'].min())
            price_max = np.ceil(data['high'].max())
            
            # Generate psychological levels
            levels = []
            for price in range(int(price_min), int(price_max) + 1):
                if price % 100 == 0:  # Major levels
                    levels.append(float(price))
                elif price % 50 == 0:  # Semi-major levels
                    levels.append(float(price))
                elif price % 10 == 0:  # Minor levels
                    levels.append(float(price))
                    
            return levels
            
        except Exception as e:
            self.logger.error(f"Error finding psychological levels: {str(e)}")
            return []
            
    def _generate_signals(self, data: pd.DataFrame) -> Dict:
        """Generate trading signals"""
        try:
            signals = {
                'trend_signals': self._generate_trend_signals(data),
                'momentum_signals': self._generate_momentum_signals(data),
                'volatility_signals': self._generate_volatility_signals(data),
                'support_resistance_signals': self._generate_sr_signals(data)
            }
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {}
            
    def _generate_trend_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """Generate trend-based signals"""
        try:
            signals = []
            
            # Calculate EMAs
            data['ema20'] = ta.trend.ema_indicator(data['close'], window=20)
            data['ema50'] = ta.trend.ema_indicator(data['close'], window=50)
            data['ema200'] = ta.trend.ema_indicator(data['close'], window=200)
            
            # Get latest values
            current_price = data['close'].iloc[-1]
            current_ema20 = data['ema20'].iloc[-1]
            current_ema50 = data['ema50'].iloc[-1]
            current_ema200 = data['ema200'].iloc[-1]
            
            # Calculate ATR for stop loss
            atr = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)
            current_atr = atr.iloc[-1]
            
            # EMA crossover signals
            if data['ema20'].iloc[-2] <= data['ema50'].iloc[-2] and current_ema20 > current_ema50:
                stop_loss = current_price - (current_atr * 2)
                take_profit = current_price + (current_atr * 3)
                signals.append(Signal(
                    symbol=symbol,
                    type='BUY',
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy='trend_ema_cross',
                    confidence=0.7,
                    entry_conditions={'ema_cross': 'ema20_above_ema50'},
                    market_context={'trend_strength': self._calculate_trend_strength(data)}
                ))
            elif data['ema20'].iloc[-2] >= data['ema50'].iloc[-2] and current_ema20 < current_ema50:
                stop_loss = current_price + (current_atr * 2)
                take_profit = current_price - (current_atr * 3)
                signals.append(Signal(
                    symbol=symbol,
                    type='SELL',
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy='trend_ema_cross',
                    confidence=0.7,
                    entry_conditions={'ema_cross': 'ema20_below_ema50'},
                    market_context={'trend_strength': self._calculate_trend_strength(data)}
                ))
                    
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trend signals: {str(e)}")
            return []
            
    def _generate_momentum_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """Generate momentum-based signals"""
        try:
            signals = []
            
            # Calculate RSI
            rsi = ta.momentum.rsi(data['close'], window=14)
            current_rsi = rsi.iloc[-1]
            
            # Calculate ATR for stop loss
            atr = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)
            current_atr = atr.iloc[-1]
            
            current_price = data['close'].iloc[-1]
            
            # RSI signals
            if current_rsi < 30:
                stop_loss = current_price - (current_atr * 1.5)
                take_profit = current_price + (current_atr * 2)
                signals.append(Signal(
                    symbol=symbol,
                    type='BUY',
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy='momentum_rsi',
                    confidence=0.8,
                    entry_conditions={'rsi': current_rsi},
                    market_context={'oversold': True}
                ))
            elif current_rsi > 70:
                stop_loss = current_price + (current_atr * 1.5)
                take_profit = current_price - (current_atr * 2)
                signals.append(Signal(
                    symbol=symbol,
                    type='SELL',
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy='momentum_rsi',
                    confidence=0.8,
                    entry_conditions={'rsi': current_rsi},
                    market_context={'overbought': True}
                ))
                    
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating momentum signals: {str(e)}")
            return []
            
    def _generate_volatility_signals(self, data: pd.DataFrame, current_volatility: float, symbol: str) -> List[Signal]:
        """Generate volatility-based signals"""
        try:
            signals = []
            
            # Calculate Bollinger Bands
            bb = ta.volatility.BollingerBands(data['close'])
            upper = bb.bollinger_hband()
            lower = bb.bollinger_lband()
            
            current_price = data['close'].iloc[-1]
            
            # Calculate ATR for stop loss
            atr = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)
            current_atr = atr.iloc[-1]
            
            # Bollinger Band signals
            if current_price <= lower.iloc[-1]:
                stop_loss = current_price - (current_atr * 2)
                take_profit = current_price + (current_atr * 3)
                signals.append(Signal(
                    symbol=symbol,
                    type='BUY',
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy='volatility_bb',
                    confidence=0.6,
                    entry_conditions={'bb_position': 'lower_band'},
                    market_context={'volatility': current_volatility}
                ))
            elif current_price >= upper.iloc[-1]:
                stop_loss = current_price + (current_atr * 2)
                take_profit = current_price - (current_atr * 3)
                signals.append(Signal(
                    symbol=symbol,
                    type='SELL',
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy='volatility_bb',
                    confidence=0.6,
                    entry_conditions={'bb_position': 'upper_band'},
                    market_context={'volatility': current_volatility}
                ))
                    
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating volatility signals: {str(e)}")
            return []
            
    def _generate_sr_signals(self, data: pd.DataFrame, market_structure: Dict, symbol: str) -> List[Signal]:
        """Generate support/resistance-based signals"""
        try:
            signals = []
            current_price = data['close'].iloc[-1]
            
            support_levels = market_structure.get('support_levels', [])
            resistance_levels = market_structure.get('resistance_levels', [])
            
            # Calculate ATR for stop loss
            atr = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)
            current_atr = atr.iloc[-1]
            
            # Find closest levels
            closest_support = next((level for level in support_levels if level < current_price), None)
            closest_resistance = next((level for level in resistance_levels if level > current_price), None)
            
            if closest_support is not None:
                price_to_support = (current_price - closest_support) / current_price
                if price_to_support < 0.001:  # Price near support
                    stop_loss = closest_support - (current_atr * 1.5)
                    take_profit = current_price + (current_price - closest_support) * 2
                    signals.append(Signal(
                        symbol=symbol,
                        type='BUY',
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        strategy='support_bounce',
                        confidence=0.7,
                        entry_conditions={'support_level': closest_support},
                        market_context={'price_to_support': price_to_support}
                    ))
                    
            if closest_resistance is not None:
                price_to_resistance = (closest_resistance - current_price) / current_price
                if price_to_resistance < 0.001:  # Price near resistance
                    stop_loss = closest_resistance + (current_atr * 1.5)
                    take_profit = current_price - (closest_resistance - current_price) * 2
                    signals.append(Signal(
                        symbol=symbol,
                        type='SELL',
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        strategy='resistance_bounce',
                        confidence=0.7,
                        entry_conditions={'resistance_level': closest_resistance},
                        market_context={'price_to_resistance': price_to_resistance}
                    ))
                    
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating S/R signals: {str(e)}")
            return []
            
    def _validate_signal(self, signal: Signal, analysis: Dict) -> bool:
        """Validate trading signal against current market conditions"""
        try:
            market_structure = analysis.get('market_structure', {})
            market_phase = market_structure.get('market_phase', 'unknown')
            volatility = market_structure.get('volatility', 0.0)
            
            # Validate based on market phase
            if market_phase == 'consolidation' and signal.confidence < 0.8:
                return False  # Only strong signals in consolidation
                
            if market_phase == 'choppy' and signal.strategy not in ['support_bounce', 'resistance_bounce']:
                return False  # Only S/R signals in choppy markets
                
            # Validate based on volatility
            if volatility > 0.03 and signal.confidence < 0.7:
                return False  # Only strong signals in high volatility
                
            # Validate risk-reward ratio
            rr_ratio = signal.get_risk_reward_ratio()
            if rr_ratio is not None and rr_ratio < 1.5:
                return False  # Minimum risk-reward ratio
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {str(e)}")
            return False

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached analysis is still valid"""
        if cache_key not in self._last_update:
            return False
            
        cache_duration = self.config['market_analysis'].get('cache_duration', 300)  # 5 minutes default
        return (datetime.now() - self._last_update[cache_key]).total_seconds() < cache_duration

    def get_market_data(self, symbol: str, timeframe: str = None) -> Optional[pd.DataFrame]:
        """Public method to get market data for a symbol and timeframe"""
        if timeframe is None:
            timeframe = self.config['trading']['timeframes'][0]  # Use first configured timeframe as default
        return self._get_market_data(symbol, timeframe)

    def _get_timeframe_value(self, timeframe: str) -> int:
        """Convert timeframe string to MT5 timeframe value"""
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        return timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)

    def _determine_market_phase(self, data: pd.DataFrame) -> str:
        """Determine the current market phase"""
        try:
            # Calculate EMAs
            data['ema20'] = ta.trend.ema_indicator(data['close'], window=20)
            data['ema50'] = ta.trend.ema_indicator(data['close'], window=50)
            
            # Get latest values
            current_price = data['close'].iloc[-1]
            current_ema20 = data['ema20'].iloc[-1]
            current_ema50 = data['ema50'].iloc[-1]
            
            # Calculate momentum
            momentum = self._calculate_momentum(data)
            
            # Determine market phase
            if current_price > current_ema20 > current_ema50:
                if momentum > 0.5:
                    return 'accumulation'
                else:
                    return 'uptrend'
            elif current_price < current_ema20 < current_ema50:
                if momentum < -0.5:
                    return 'distribution'
                else:
                    return 'downtrend'
            else:
                return 'consolidation'
                
        except Exception as e:
            self.logger.error(f"Error determining market phase: {str(e)}")
            return 'unknown'

    async def generate_signals(self, symbol: str, timeframe: str = 'H1') -> List[Signal]:
        """Generate trading signals based on market analysis"""
        try:
            # Get market data
            market_data = await self._get_market_data(symbol, timeframe, 1000)
            if market_data is None:
                self.logger.error(f"Failed to get market data for {symbol}")
                return []
                
            # Get market analysis
            analysis = await self.analyze_market(symbol, timeframe)
            if analysis is None:
                self.logger.error(f"Failed to analyze market for {symbol}")
                return []
                
            signals = []
            
            # Generate trend signals
            trend_signals = self._generate_trend_signals(market_data, symbol)
            if trend_signals:
                signals.extend(trend_signals)
                
            # Generate momentum signals
            momentum_signals = self._generate_momentum_signals(market_data, symbol)
            if momentum_signals:
                signals.extend(momentum_signals)
                
            # Generate volatility signals
            volatility_signals = self._generate_volatility_signals(market_data, analysis['market_structure']['volatility'], symbol)
            if volatility_signals:
                signals.extend(volatility_signals)
                
            # Generate support/resistance signals
            sr_signals = self._generate_sr_signals(market_data, analysis['market_structure'], symbol)
            if sr_signals:
                signals.extend(sr_signals)
                
            # Filter and validate signals
            valid_signals = []
            for signal in signals:
                if self._validate_signal(signal, analysis):
                    valid_signals.append(signal)
                    
            return valid_signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return []
