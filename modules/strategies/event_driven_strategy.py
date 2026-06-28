import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union
import ta
import traceback

class EventDrivenStrategy:
    def __init__(self, config: Dict):
        """Initialize the event-driven strategy"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.timeframes = config.get('trading_parameters', {}).get('timeframes', ['M1', 'M5', 'M15', 'H1'])
        self.risk_params = config.get('risk_management', {})
        self._init_indicators()
    
    def _init_indicators(self):
        """Initialize technical indicators"""
        try:
            # Initialize indicator parameters
            self.rsi_period = 14
            self.macd_fast = 12
            self.macd_slow = 26
            self.macd_signal = 9
            self.bb_period = 20
            self.bb_std = 2
            self.atr_period = 14
            
            self.logger.info("Technical indicators initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing indicators: {str(e)}")
            raise
            
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for analysis"""
        try:
            if df.empty:
                return df

            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
            
            # Moving averages
            df['ma_fast'] = ta.trend.sma_indicator(df['close'], window=self.macd_fast)
            df['ma_slow'] = ta.trend.sma_indicator(df['close'], window=self.macd_slow)
            
            # MACD
            macd, macd_signal, macd_hist = ta.trend.MACD(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_diff'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # ATR for volatility
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=self.atr_period)
            
            # Volume indicators
            df['volume_ma'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['tick_volume'])
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # ADX for trend strength
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            traceback.print_exc()
            return df
            
    def get_market_data(self, symbol: str, timeframe: str, lookback: int = 100) -> Optional[pd.DataFrame]:
        """Get market data with technical indicators"""
        try:
            # Get MT5 timeframe constant
            mt5_tf = getattr(mt5, f'TIMEFRAME_{timeframe}')
            
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, lookback)
            if rates is None:
                self.logger.error(f"Failed to get market data for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            traceback.print_exc()
            return None
    
    def analyze_market_conditions(self, df: pd.DataFrame) -> Dict:
        """Analyze current market conditions"""
        try:
            current = df.iloc[-1]
            
            conditions = {
                'trend': self.detect_trend(df),
                'volatility': self.detect_volatility(df),
                'momentum': self.detect_momentum(df),
                'support_resistance': self.detect_support_resistance(df),
                'volume': self.analyze_volume(df)
            }
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")
            traceback.print_exc()
            return {}
    
    def detect_trend(self, df: pd.DataFrame) -> Dict:
        """Detect market trend"""
        try:
            current = df.iloc[-1]
            
            trend = {
                'direction': 'none',
                'strength': 0,
                'duration': 0
            }
            
            # Trend direction from moving averages
            if current['ma_fast'] > current['ma_slow']:
                trend['direction'] = 'up'
            elif current['ma_fast'] < current['ma_slow']:
                trend['direction'] = 'down'
            
            # Trend strength from ADX
            trend['strength'] = current['adx']
            
            # Trend duration
            if trend['direction'] == 'up':
                trend['duration'] = len(df[df['ma_fast'] > df['ma_slow']])
            elif trend['direction'] == 'down':
                trend['duration'] = len(df[df['ma_fast'] < df['ma_slow']])
            
            return trend
            
        except Exception as e:
            self.logger.error(f"Error detecting trend: {str(e)}")
            traceback.print_exc()
            return {'direction': 'none', 'strength': 0, 'duration': 0}
    
    def detect_volatility(self, df: pd.DataFrame) -> Dict:
        """Detect market volatility"""
        try:
            current_atr = df['atr'].iloc[-1]
            avg_atr = df['atr'].mean()
            
            volatility = {
                'current': current_atr,
                'average': avg_atr,
                'is_high': current_atr > avg_atr * 1.5,
                'is_low': current_atr < avg_atr * 0.5,
                'trend': 'increasing' if df['atr'].iloc[-1] > df['atr'].iloc[-2] else 'decreasing'
            }
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility: {str(e)}")
            traceback.print_exc()
            return {}
    
    def detect_momentum(self, df: pd.DataFrame) -> Dict:
        """Detect market momentum"""
        try:
            current = df.iloc[-1]
            
            momentum = {
                'rsi': current['rsi'],
                'macd': current['macd'],
                'is_overbought': current['rsi'] > 70,
                'is_oversold': current['rsi'] < 30,
                'momentum_strength': abs(current['macd'])
            }
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error detecting momentum: {str(e)}")
            traceback.print_exc()
            return {}
    
    def detect_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Detect support and resistance levels"""
        try:
            current_price = df['close'].iloc[-1]
            
            # Use Bollinger Bands as dynamic support/resistance
            support_resistance = {
                'upper_resistance': df['bb_upper'].iloc[-1],
                'lower_support': df['bb_lower'].iloc[-1],
                'middle_line': df['bb_middle'].iloc[-1],
                'price_position': (current_price - df['bb_lower'].iloc[-1]) / (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
            }
            
            return support_resistance
            
        except Exception as e:
            self.logger.error(f"Error detecting support/resistance: {str(e)}")
            traceback.print_exc()
            return {}
    
    def analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze volume profile"""
        try:
            current_volume = df['tick_volume'].iloc[-1]
            avg_volume = df['tick_volume'].mean()
            
            volume = {
                'current': current_volume,
                'average': avg_volume,
                'is_high': current_volume > avg_volume * 1.2,
                'is_low': current_volume < avg_volume * 0.8,
                'trend': 'increasing' if df['tick_volume'].iloc[-1] > df['tick_volume'].iloc[-2] else 'decreasing'
            }
            
            return volume
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume: {str(e)}")
            traceback.print_exc()
            return {}
    
    def get_signal(self, symbol: str) -> Dict:
        """Get trading signal based on strategy rules"""
        try:
            signals = {}
            
            # Get data for multiple timeframes
            for tf in self.timeframes:
                df = self.get_market_data(symbol, tf)
                if df is None:
                    continue
                
                # Analyze market conditions
                conditions = self.analyze_market_conditions(df)
                
                # Generate signal for this timeframe
                signal = self.generate_signal(df, conditions)
                signals[tf] = signal
            
            # Combine signals from multiple timeframes
            final_signal = self.combine_signals(signals)
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Error getting signal: {str(e)}")
            traceback.print_exc()
            return {'signal': 'NONE', 'type': 'error'}
    
    def generate_signal(self, df: pd.DataFrame, conditions: Dict) -> Dict:
        """Generate trading signal for a single timeframe"""
        try:
            current = df.iloc[-1]
            
            signal = {
                'signal': 'NONE',
                'strength': 0,
                'reason': []
            }
            
            # Log current conditions
            self.logger.debug(f"Current conditions: {conditions}")
            
            # Check for buy conditions
            trend_up = conditions['trend']['direction'] == 'up'
            rsi_above_oversold = conditions['momentum']['rsi'] > 30
            volume_high = conditions['volume']['is_high']
            
            self.logger.debug(f"Buy conditions - Trend up: {trend_up}, RSI > oversold: {rsi_above_oversold}, Volume high: {volume_high}")
            
            if trend_up and rsi_above_oversold and volume_high:
                signal['signal'] = 'BUY'
                signal['strength'] = conditions['trend']['strength'] * 0.5 + conditions['momentum']['momentum_strength'] * 0.3
                signal['reason'].append('Uptrend with strong momentum and volume')
            
            # Check for sell conditions
            trend_down = conditions['trend']['direction'] == 'down'
            rsi_below_overbought = conditions['momentum']['rsi'] < 70
            
            self.logger.debug(f"Sell conditions - Trend down: {trend_down}, RSI < overbought: {rsi_below_overbought}, Volume high: {volume_high}")
            
            if trend_down and rsi_below_overbought and volume_high:
                signal['signal'] = 'SELL'
                signal['strength'] = conditions['trend']['strength'] * 0.5 + conditions['momentum']['momentum_strength'] * 0.3
                signal['reason'].append('Downtrend with strong momentum and volume')
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            traceback.print_exc()
            return {'signal': 'NONE', 'strength': 0, 'reason': ['Error generating signal']}
    
    def combine_signals(self, signals: Dict) -> Dict:
        """Combine signals from multiple timeframes"""
        try:
            if not signals:
                return {'signal': 'NONE', 'strength': 0, 'reason': ['No signals available']}
            
            # Weight for each timeframe
            weights = {
                'M1': 0.1,
                'M5': 0.2,
                'M15': 0.3,
                'H1': 0.4
            }
            
            buy_strength = 0
            sell_strength = 0
            reasons = []
            
            for tf, signal in signals.items():
                if signal['signal'] == 'BUY':
                    buy_strength += signal['strength'] * weights.get(tf, 0.1)
                    reasons.extend([f"{tf}: {r}" for r in signal['reason']])
                elif signal['signal'] == 'SELL':
                    sell_strength += signal['strength'] * weights.get(tf, 0.1)
                    reasons.extend([f"{tf}: {r}" for r in signal['reason']])
            
            # Determine final signal
            if buy_strength > sell_strength and buy_strength > 0.6:
                return {'signal': 'BUY', 'strength': buy_strength, 'reason': reasons}
            elif sell_strength > buy_strength and sell_strength > 0.6:
                return {'signal': 'SELL', 'strength': sell_strength, 'reason': reasons}
            
            return {'signal': 'NONE', 'strength': 0, 'reason': ['No strong signals']}
            
        except Exception as e:
            self.logger.error(f"Error combining signals: {str(e)}")
            traceback.print_exc()
            return {'signal': 'NONE', 'strength': 0, 'reason': ['Error combining signals']}
    
    def get_company_news(self, symbol: str) -> List[Dict]:
        """Get recent news for the symbol's company"""
        try:
            # This is a placeholder. Implement actual news retrieval logic
            return []
        except Exception as e:
            self.logger.error(f"Error getting company news: {str(e)}")
            return []

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signals based on event-driven analysis"""
        try:
            signals = {}
            
            # Calculate technical indicators
            rsi = ta.momentum.RSIIndicator(data['close'], window=self.rsi_period).rsi()
            macd = ta.trend.MACD(
                data['close'],
                window_slow=self.macd_slow,
                window_fast=self.macd_fast,
                window_sign=self.macd_signal
            )
            bb = ta.volatility.BollingerBands(
                data['close'],
                window=self.bb_period,
                window_dev=self.bb_std
            )
            atr = ta.volatility.AverageTrueRange(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=self.atr_period
            ).average_true_range()
            
            # Get latest values
            current_close = data['close'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_macd = macd.macd().iloc[-1]
            current_macd_signal = macd.macd_signal().iloc[-1]
            current_bb_upper = bb.bollinger_hband().iloc[-1]
            current_bb_lower = bb.bollinger_lband().iloc[-1]
            current_atr = atr.iloc[-1]
            
            # Generate signals based on events
            signals['rsi_signal'] = 1 if current_rsi < 30 else (-1 if current_rsi > 70 else 0)
            signals['macd_signal'] = 1 if current_macd > current_macd_signal else (-1 if current_macd < current_macd_signal else 0)
            signals['bb_signal'] = 1 if current_close < current_bb_lower else (-1 if current_close > current_bb_upper else 0)
            
            # Calculate signal strength
            signal_strength = (
                signals['rsi_signal'] +
                signals['macd_signal'] +
                signals['bb_signal']
            ) / 3.0
            
            # Add volatility adjustment
            volatility_factor = current_atr / current_close
            signal_strength *= (1 - volatility_factor)  # Reduce signal strength in high volatility
            
            # Final signal
            signals['final_signal'] = np.clip(signal_strength, -1, 1)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {'final_signal': 0.0}  # Neutral signal as fallback
