from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import talib
import pandas_ta as ta
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from ..deployment.error_handler import ErrorHandler

class TechnicalAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('technical_analyzer')
        
        # Initialize indicators
        self._init_indicators()
        
    def _init_indicators(self):
        """Initialize technical indicators"""
        self.indicators = {
            # Trend Indicators
            'sma': lambda df, **kwargs: talib.SMA(df['close'], **kwargs),
            'ema': lambda df, **kwargs: talib.EMA(df['close'], **kwargs),
            'macd': lambda df, **kwargs: pd.DataFrame(
                talib.MACD(df['close'], **kwargs)
            ).T.values,
            'adx': lambda df, **kwargs: talib.ADX(
                df['high'],
                df['low'],
                df['close'],
                **kwargs
            ),
            
            # Momentum Indicators
            'rsi': lambda df, **kwargs: talib.RSI(df['close'], **kwargs),
            'stoch': lambda df, **kwargs: pd.DataFrame(
                talib.STOCH(
                    df['high'],
                    df['low'],
                    df['close'],
                    **kwargs
                )
            ).T.values,
            'cci': lambda df, **kwargs: talib.CCI(
                df['high'],
                df['low'],
                df['close'],
                **kwargs
            ),
            
            # Volatility Indicators
            'bbands': lambda df, **kwargs: pd.DataFrame(
                talib.BBANDS(df['close'], **kwargs)
            ).T.values,
            'atr': lambda df, **kwargs: talib.ATR(
                df['high'],
                df['low'],
                df['close'],
                **kwargs
            ),
            
            # Volume Indicators
            'obv': lambda df, **kwargs: talib.OBV(
                df['close'],
                df['volume'].astype(float),
                **kwargs
            ),
            'ad': lambda df, **kwargs: talib.AD(
                df['high'],
                df['low'],
                df['close'],
                df['volume'].astype(float),
                **kwargs
            ),
            
            # Custom Indicators (using pandas_ta)
            'vwap': lambda df, **kwargs: df.ta.vwap(**kwargs),
            'supertrend': lambda df, **kwargs: df.ta.supertrend(**kwargs),
            'squeeze': lambda df, **kwargs: df.ta.squeeze(**kwargs)
        }
        
    async def analyze_symbol(
        self,
        symbol: str,
        timeframe: str = 'H1',
        lookback: int = 100,
        indicators: Optional[List[str]] = None
    ) -> Dict:
        """Perform technical analysis on a symbol"""
        try:
            # Get historical data
            df = await self._get_historical_data(symbol, timeframe, lookback)
            if df.empty:
                return {}
                
            # Calculate indicators
            analysis = {}
            
            # Use specified indicators or all
            indicator_list = indicators if indicators else self.indicators.keys()
            
            for indicator in indicator_list:
                if indicator in self.indicators:
                    try:
                        # Calculate indicator
                        result = self.indicators[indicator](
                            df,
                            **self._get_indicator_params(indicator)
                        )
                        
                        # Store latest values
                        if isinstance(result, np.ndarray):
                            analysis[indicator] = result[-1]
                        elif isinstance(result, pd.DataFrame):
                            analysis[indicator] = result.iloc[-1].to_dict()
                        else:
                            analysis[indicator] = result
                            
                    except Exception as e:
                        self.logger.error(f"Error calculating {indicator}: {str(e)}")
                        
            # Add market context
            analysis['context'] = await self._analyze_market_context(df)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Technical analysis error: {str(e)}")
            return {}
            
    async def _get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        lookback: int
    ) -> pd.DataFrame:
        """Get historical data from MT5"""
        try:
            # Convert timeframe string to MT5 timeframe
            timeframes = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = timeframes.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Get rates
            rates = mt5.copy_rates_from_pos(
                symbol,
                mt5_timeframe,
                0,
                lookback
            )
            
            if rates is None:
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Historical data fetch error: {str(e)}")
            return pd.DataFrame()
            
    def _get_indicator_params(self, indicator: str) -> Dict:
        """Get parameters for technical indicators"""
        params = {
            'sma': {'timeperiod': 20},
            'ema': {'timeperiod': 20},
            'macd': {
                'fastperiod': 12,
                'slowperiod': 26,
                'signalperiod': 9
            },
            'adx': {'timeperiod': 14},
            'rsi': {'timeperiod': 14},
            'stoch': {
                'fastk_period': 14,
                'slowk_period': 3,
                'slowd_period': 3
            },
            'cci': {'timeperiod': 20},
            'bbands': {
                'timeperiod': 20,
                'nbdevup': 2,
                'nbdevdn': 2
            },
            'atr': {'timeperiod': 14},
            'vwap': {'length': 14},
            'supertrend': {
                'length': 10,
                'multiplier': 3.0
            },
            'squeeze': {'length': 20}
        }
        
        return params.get(indicator, {})
        
    async def _analyze_market_context(self, df: pd.DataFrame) -> Dict:
        """Analyze market context"""
        try:
            context = {}
            
            # Trend Analysis
            sma20 = talib.SMA(df['close'], timeperiod=20)
            sma50 = talib.SMA(df['close'], timeperiod=50)
            sma200 = talib.SMA(df['close'], timeperiod=200)
            
            current_price = df['close'].iloc[-1]
            
            context['trend'] = {
                'short_term': 'bullish' if current_price > sma20.iloc[-1] else 'bearish',
                'medium_term': 'bullish' if current_price > sma50.iloc[-1] else 'bearish',
                'long_term': 'bullish' if current_price > sma200.iloc[-1] else 'bearish'
            }
            
            # Volatility Analysis
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'])
            
            context['volatility'] = {
                'atr': atr.iloc[-1],
                'bb_width': (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
            }
            
            # Volume Analysis
            context['volume'] = {
                'current': df['volume'].iloc[-1],
                'avg_20': df['volume'].rolling(20).mean().iloc[-1],
                'trend': 'increasing' if df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1]
                else 'decreasing'
            }
            
            # Support/Resistance
            context['levels'] = await self._calculate_support_resistance(df)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Market context analysis error: {str(e)}")
            return {}
            
    async def _calculate_support_resistance(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> Dict:
        """Calculate support and resistance levels"""
        try:
            levels = {
                'support': [],
                'resistance': []
            }
            
            # Calculate pivot points
            pivots = df.ta.pivots(high='high', low='low', close='close', window=window)
            
            if not pivots.empty:
                support_levels = pivots[pivots['pivot'] == 'S'].sort_values('low')['low'].values
                resistance_levels = pivots[pivots['pivot'] == 'R'].sort_values('high')['high'].values
                
                levels['support'] = support_levels[-3:] if len(support_levels) > 0 else []
                levels['resistance'] = resistance_levels[:3] if len(resistance_levels) > 0 else []
                
            return levels
            
        except Exception as e:
            self.logger.error(f"Support/Resistance calculation error: {str(e)}")
            return {'support': [], 'resistance': []}
            
    def get_signal_strength(self, analysis: Dict) -> float:
        """Calculate signal strength from technical analysis"""
        try:
            signals = []
            
            # Trend signals
            if 'trend' in analysis.get('context', {}):
                trend = analysis['context']['trend']
                signals.extend([
                    1 if trend['short_term'] == 'bullish' else -1,
                    1 if trend['medium_term'] == 'bullish' else -1,
                    1 if trend['long_term'] == 'bullish' else -1
                ])
                
            # RSI signals
            if 'rsi' in analysis:
                rsi = analysis['rsi']
                if rsi < 30:
                    signals.append(1)  # Oversold
                elif rsi > 70:
                    signals.append(-1)  # Overbought
                    
            # MACD signals
            if 'macd' in analysis:
                macd = analysis['macd']
                if macd[0] > macd[1]:  # MACD line > Signal line
                    signals.append(1)
                else:
                    signals.append(-1)
                    
            # Bollinger Bands signals
            if 'bbands' in analysis:
                bb = analysis['bbands']
                close = analysis.get('close', 0)
                if close < bb[0]:  # Price below lower band
                    signals.append(1)
                elif close > bb[2]:  # Price above upper band
                    signals.append(-1)
                    
            # Calculate average signal strength
            if signals:
                return sum(signals) / len(signals)
            return 0
            
        except Exception as e:
            self.logger.error(f"Signal strength calculation error: {str(e)}")
            return 0
