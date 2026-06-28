import logging
from typing import Dict, Optional
import pandas as pd
import MetaTrader5 as mt5
import numpy as np
from datetime import datetime, timedelta

class MultiTimeframeAnalyzer:
    def __init__(self, config: Dict):
        """Initialize multi-timeframe analyzer"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Define timeframes to analyze
        self.timeframes = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze market data across multiple timeframes"""
        try:
            if df is None or len(df) < 200:
                self.logger.error("Not enough data for analysis")
                return {}
                
            # Calculate technical indicators
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['sma50'] = df['close'].rolling(window=50).mean()
            df['sma200'] = df['close'].rolling(window=200).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # ROC (Rate of Change)
            df['roc'] = df['close'].pct_change(periods=10) * 100
            df['roc_ma'] = df['roc'].rolling(window=10).mean()
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Calculate trend
            trend = 0.0
            if latest['sma20'] > latest['sma50'] and latest['sma50'] > latest['sma200']:
                trend = 1.0
            elif latest['sma20'] < latest['sma50'] and latest['sma50'] < latest['sma200']:
                trend = -1.0
            else:
                trend = 0.0
                
            # Adjust trend by ROC
            if latest['roc'] > 0:
                trend *= 1.2
            elif latest['roc'] < 0:
                trend *= 0.8
                
            # Calculate momentum
            momentum = 0.0
            
            # RSI
            if latest['rsi'] > 70:
                momentum -= 0.3
            elif latest['rsi'] < 30:
                momentum += 0.3
                
            # MACD
            if latest['macd'] > latest['signal']:
                momentum += 0.3
            elif latest['macd'] < latest['signal']:
                momentum -= 0.3
                
            # ROC
            if latest['roc'] > latest['roc_ma']:
                momentum += 0.4
            elif latest['roc'] < latest['roc_ma']:
                momentum -= 0.4
                
            # Calculate volatility
            volatility = latest['atr'] / latest['close']
            
            # Log analysis
            self.logger.debug(
                f"Moving Averages - "
                f"Close: {latest['close']:.5f}, "
                f"SMA20: {latest['sma20']:.5f}, "
                f"SMA50: {latest['sma50']:.5f}, "
                f"SMA200: {latest['sma200']:.5f}"
            )
            
            self.logger.debug(
                f"Momentum Analysis - "
                f"RSI: {latest['rsi']:.2f}, "
                f"ROC: {latest['roc']:.2f}, "
                f"ROC MA: {latest['roc_ma']:.2f}"
            )
            
            self.logger.debug(
                f"MACD Analysis - "
                f"MACD: {latest['macd']:.5f}, "
                f"Signal: {latest['signal']:.5f}"
            )
            
            self.logger.debug(
                f"Volatility Analysis - "
                f"ATR: {latest['atr']:.5f}, "
                f"ATR MA: {df['atr'].rolling(window=20).mean().iloc[-1]:.5f}"
            )
            
            return {
                'trend': trend,
                'momentum': momentum,
                'volatility': volatility,
                'atr': latest['atr']
            }
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            return {}
            
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Analyze a symbol across multiple timeframes"""
        try:
            analysis = {}
            
            # Analyze each timeframe
            valid_analyses = {}
            for tf_name, tf_value in self.timeframes.items():
                tf_analysis = self._analyze_timeframe(symbol, tf_value)
                if isinstance(tf_analysis, dict) and tf_analysis.get('valid', True):
                    valid_analyses[tf_name] = tf_analysis
                
            # Only proceed if we have at least one valid timeframe
            if not valid_analyses:
                self.logger.warning(f"No valid timeframe analysis for {symbol}")
                return {
                    'valid': False,
                    'error': 'No valid timeframe analysis available'
                }
                
            # Combine analyses for overall view
            overall = self._combine_timeframe_analyses(valid_analyses)
            
            return {
                'valid': True,
                'timeframes': valid_analyses,
                'overall': overall
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            return {
                'valid': False,
                'error': str(e)
            }
            
    def _analyze_timeframe(self, symbol: str, timeframe: int) -> Dict:
        """Analyze a single timeframe"""
        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
            if rates is None:
                self.logger.error(f"No data available for {symbol} on timeframe {timeframe}")
                return {'valid': False, 'error': 'No data available'}
                
            df = pd.DataFrame(rates)
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('datetime', inplace=True)
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Calculate moving averages
            sma20 = np.mean(close[-20:])
            sma50 = np.mean(close[-50:])
            sma200 = np.mean(close[-200:])
            
            self.logger.debug(f"Moving Averages - Close: {close[-1]}, SMA20: {sma20}, SMA50: {sma50}, SMA200: {sma200}")
            
            # Calculate trend based on moving averages
            short_trend = (close[-1] - sma20) / sma20
            medium_trend = (close[-1] - sma50) / sma50
            long_trend = (close[-1] - sma200) / sma200
            
            # Weight the trends (increased short-term weight)
            trend = (short_trend * 0.5) + (medium_trend * 0.3) + (long_trend * 0.2)
            
            # Calculate RSI
            delta = np.diff(close)
            gain = (delta > 0) * delta
            loss = (delta < 0) * -delta
            avg_gain = np.mean(gain[-14:])
            avg_loss = np.mean(loss[-14:])
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
            # Calculate ROC (Rate of Change)
            roc = ((close[-1] - close[-13]) / close[-13]) * 100
            roc_ma = ((close[-1] - close[-21]) / close[-21]) * 100
            
            self.logger.debug(f"Momentum Analysis - RSI: {rsi:.2f}, ROC: {roc:.2f}, ROC MA: {roc_ma:.2f}")
            
            # Calculate momentum (more weight to RSI)
            momentum = ((rsi - 50) / 50) * 0.5 + (roc / 100) * 0.3 + (roc_ma / 100) * 0.2
            
            # Calculate MACD
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd = ema12 - ema26
            signal = self._ema(macd, 9)
            
            self.logger.debug(f"MACD Analysis - MACD: {macd[-1]:.6f}, Signal: {signal[-1]:.6f}")
            
            # Calculate ATR
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1])
                )
            )
            atr = np.mean(tr[-14:])
            atr_ma = np.mean(tr[-28:])
            
            self.logger.debug(f"Volatility Analysis - ATR: {atr:.5f}, ATR MA: {atr_ma:.5f}")
            
            # Calculate volatility
            volatility = atr / close[-1]
            
            return {
                'trend': trend,
                'momentum': momentum,
                'volatility': volatility,
                'atr': atr
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe: {str(e)}")
            return {'valid': False, 'error': str(e)}
            
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        return atr
        
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        try:
            window = 20
            levels = {}
            
            # Find potential support levels
            low_points = df['low'].rolling(window=window, center=True).min()
            support_levels = low_points.dropna().unique()
            support_levels.sort()
            
            # Find potential resistance levels
            high_points = df['high'].rolling(window=window, center=True).max()
            resistance_levels = high_points.dropna().unique()
            resistance_levels.sort()
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Find nearest levels
            levels['support'] = support_levels[support_levels < current_price][-3:]
            levels['resistance'] = resistance_levels[resistance_levels > current_price][:3]
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {str(e)}")
            return {'support': [], 'resistance': []}
            
    def _generate_signals(self, df: pd.DataFrame, analysis: Dict) -> Dict:
        """Generate trading signals based on analysis"""
        signals = {
            'trend': 0,  # -1: downtrend, 0: neutral, 1: uptrend
            'strength': 0,  # 0 to 1
            'support_resistance': 0,  # -1: at resistance, 0: neutral, 1: at support
            'volatility': 0  # 0: low, 1: high
        }
        
        # Trend Analysis
        trend_scores = list(analysis['trend'].values())
        signals['trend'] = np.sign(sum(trend_scores))
        signals['strength'] = abs(sum(trend_scores)) / len(trend_scores)
        
        # Support/Resistance Analysis
        current_price = float(df['close'].iloc[-1])
        sr_levels = analysis['support_resistance']
        
        if len(sr_levels['support']) > 0 and len(sr_levels['resistance']) > 0:
            nearest_support = float(sr_levels['support'][-1])
            nearest_resistance = float(sr_levels['resistance'][0])
            
            # Calculate price position between support and resistance
            range_size = nearest_resistance - nearest_support
            if range_size > 0:
                position = (current_price - nearest_support) / range_size
                if position < 0.2:
                    signals['support_resistance'] = 1  # Near support
                elif position > 0.8:
                    signals['support_resistance'] = -1  # Near resistance
                    
        # Volatility Analysis
        avg_atr = analysis['volatility']['atr_ma']
        current_atr = analysis['volatility']['atr']
        signals['volatility'] = 1 if current_atr > avg_atr * 1.2 else 0
        
        return signals
        
    def _combine_timeframe_analyses(self, analyses: Dict) -> Dict:
        """Combine analyses from multiple timeframes into overall view"""
        try:
            # Initialize overall metrics
            trend_scores = []
            momentum_scores = []
            volatility_scores = []
            
            # Weight factors for different timeframes
            timeframe_weights = {
                'M5': 0.1,
                'M15': 0.15,
                'H1': 0.25,
                'H4': 0.25,
                'D1': 0.25
            }
            
            for tf_name, analysis in analyses.items():
                weight = timeframe_weights.get(tf_name, 0.1)
                
                # Trend score (-1 to 1)
                trend = analysis.get('trend', {})
                trend_score = sum([
                    trend.get('sma20', 0) * 0.4,
                    trend.get('sma50', 0) * 0.35,
                    trend.get('sma200', 0) * 0.25
                ]) * weight
                trend_scores.append(trend_score)
                
                # Momentum score (0 to 1)
                momentum = analysis.get('momentum', {})
                rsi = momentum.get('rsi', 50)
                rsi_score = (rsi - 50) / 50  # Convert RSI to -1 to 1 range
                
                roc = momentum.get('roc', 0)
                roc_score = max(min(roc / 10, 1), -1)  # Normalize ROC to -1 to 1
                
                macd = momentum.get('macd', 0)
                signal = momentum.get('signal', 0)
                macd_score = 1 if macd > signal else -1 if macd < signal else 0
                
                momentum_score = (rsi_score * 0.4 + roc_score * 0.3 + macd_score * 0.3) * weight
                momentum_scores.append(momentum_score)
                
                # Volatility score (0 to 1)
                volatility = analysis.get('volatility', {})
                atr = volatility.get('atr', 0)
                atr_ma = volatility.get('atr_ma', 1)
                vol_score = (atr / atr_ma if atr_ma > 0 else 1) * weight
                volatility_scores.append(vol_score)
                
            # Calculate overall scores
            overall_trend = sum(trend_scores)
            overall_momentum = sum(momentum_scores)
            overall_volatility = min(sum(volatility_scores), 1)  # Cap at 1
            
            self.logger.debug(
                f"Overall Analysis - "
                f"Trend: {overall_trend:.3f}, "
                f"Momentum: {overall_momentum:.3f}, "
                f"Volatility: {overall_volatility:.3f}"
            )
            
            return {
                'trend': overall_trend,
                'momentum': overall_momentum,
                'strength': abs(overall_trend),  # How strong the trend is
                'volatility': overall_volatility
            }
            
        except Exception as e:
            self.logger.error(f"Error combining timeframe analyses: {str(e)}")
            return {
                'trend': 0,
                'momentum': 0,
                'strength': 0,
                'volatility': 0
            }

    def _ema(self, data, window):
        """Calculate Exponential Moving Average"""
        alpha = 2 / (window + 1)
        ema = []
        for i in range(len(data)):
            if i == 0:
                ema.append(data[i])
            else:
                ema.append(alpha * data[i] + (1 - alpha) * ema[i - 1])
        return np.array(ema)
