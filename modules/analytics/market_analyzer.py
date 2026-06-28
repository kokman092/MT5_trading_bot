from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from arch import arch_model
import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta
from ..deployment.error_handler import ErrorHandler

class MarketAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('market_analyzer')
        self.order_book_cache = {}
        self.regime_states = {}
        self.volatility_forecasts = {}
        self.cluster_maps = {}
        
    async def analyze_order_book(self, symbol: str, depth: int = 20) -> Dict:
        """Analyze order book for trading insights"""
        try:
            # Get order book data
            order_book = mt5.market_book_get(symbol)
            if not order_book:
                return {}
                
            analysis = {
                'imbalance': 0.0,
                'pressure': 'neutral',
                'hidden_orders': [],
                'depth_changes': {},
                'timestamp': datetime.now()
            }
            
            # Separate bids and asks
            bids = [order for order in order_book if order.type == mt5.BOOK_TYPE_SELL]
            asks = [order for order in order_book if order.type == mt5.BOOK_TYPE_BUY]
            
            # Calculate order book imbalance
            bid_volume = sum(order.volume_real for order in bids[:depth])
            ask_volume = sum(order.volume_real for order in asks[:depth])
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                analysis['imbalance'] = (bid_volume - ask_volume) / total_volume
                analysis['pressure'] = 'buy' if analysis['imbalance'] > 0.2 else 'sell' if analysis['imbalance'] < -0.2 else 'neutral'
                
            # Detect hidden orders
            analysis['hidden_orders'] = await self._detect_hidden_orders(bids, asks)
            
            # Analyze depth changes
            if symbol in self.order_book_cache:
                analysis['depth_changes'] = await self._analyze_depth_changes(
                    symbol,
                    self.order_book_cache[symbol],
                    order_book
                )
                
            # Update cache
            self.order_book_cache[symbol] = order_book
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Order book analysis error: {str(e)}")
            return {}
            
    async def _detect_hidden_orders(
        self,
        bids: List,
        asks: List,
        threshold: float = 2.0
    ) -> List[Dict]:
        """Detect potential hidden (iceberg) orders"""
        try:
            hidden_orders = []
            
            # Analyze bid side
            for i in range(1, len(bids)):
                volume_ratio = bids[i].volume_real / bids[i-1].volume_real
                if volume_ratio > threshold:
                    hidden_orders.append({
                        'side': 'bid',
                        'price': bids[i].price,
                        'volume': bids[i].volume_real,
                        'ratio': volume_ratio
                    })
                    
            # Analyze ask side
            for i in range(1, len(asks)):
                volume_ratio = asks[i].volume_real / asks[i-1].volume_real
                if volume_ratio > threshold:
                    hidden_orders.append({
                        'side': 'ask',
                        'price': asks[i].price,
                        'volume': asks[i].volume_real,
                        'ratio': volume_ratio
                    })
                    
            return hidden_orders
            
        except Exception as e:
            self.logger.error(f"Hidden order detection error: {str(e)}")
            return []
            
    async def _analyze_depth_changes(
        self,
        symbol: str,
        previous_book: List,
        current_book: List
    ) -> Dict:
        """Analyze changes in order book depth"""
        try:
            changes = {
                'bid_depth_change': 0.0,
                'ask_depth_change': 0.0,
                'significant_changes': []
            }
            
            # Convert to DataFrames for easier analysis
            prev_df = pd.DataFrame([
                {
                    'type': order.type,
                    'price': order.price,
                    'volume': order.volume_real
                }
                for order in previous_book
            ])
            
            curr_df = pd.DataFrame([
                {
                    'type': order.type,
                    'price': order.price,
                    'volume': order.volume_real
                }
                for order in current_book
            ])
            
            # Calculate depth changes
            if not prev_df.empty and not curr_df.empty:
                # Bid depth change
                prev_bid_depth = prev_df[prev_df['type'] == mt5.BOOK_TYPE_SELL]['volume'].sum()
                curr_bid_depth = curr_df[curr_df['type'] == mt5.BOOK_TYPE_SELL]['volume'].sum()
                changes['bid_depth_change'] = ((curr_bid_depth - prev_bid_depth) / prev_bid_depth) * 100
                
                # Ask depth change
                prev_ask_depth = prev_df[prev_df['type'] == mt5.BOOK_TYPE_BUY]['volume'].sum()
                curr_ask_depth = curr_df[curr_df['type'] == mt5.BOOK_TYPE_BUY]['volume'].sum()
                changes['ask_depth_change'] = ((curr_ask_depth - prev_ask_depth) / prev_ask_depth) * 100
                
                # Detect significant changes
                threshold = self.config.get('depth_change_threshold', 10.0)
                if abs(changes['bid_depth_change']) > threshold:
                    changes['significant_changes'].append({
                        'side': 'bid',
                        'change': changes['bid_depth_change']
                    })
                    
                if abs(changes['ask_depth_change']) > threshold:
                    changes['significant_changes'].append({
                        'side': 'ask',
                        'change': changes['ask_depth_change']
                    })
                    
            return changes
            
        except Exception as e:
            self.logger.error(f"Depth change analysis error: {str(e)}")
            return {}
            
    async def perform_cluster_analysis(self, symbols: List[str]) -> Dict:
        """Perform cluster analysis on symbols"""
        try:
            # Get price data for all symbols
            price_data = {}
            for symbol in symbols:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 100)
                if rates is not None:
                    df = pd.DataFrame(rates)
                    price_data[symbol] = df['close'].pct_change().dropna()
                    
            if not price_data:
                return {}
                
            # Create correlation matrix
            returns_df = pd.DataFrame(price_data)
            corr_matrix = returns_df.corr()
            
            # Perform clustering
            X = StandardScaler().fit_transform(corr_matrix)
            clustering = DBSCAN(eps=0.3, min_samples=2).fit(X)
            
            # Organize results
            clusters = {}
            for i, label in enumerate(clustering.labels_):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(symbols[i])
                
            # Calculate cluster metrics
            cluster_metrics = {}
            for label, cluster_symbols in clusters.items():
                if label != -1:  # Ignore noise points
                    cluster_returns = returns_df[cluster_symbols]
                    cluster_metrics[label] = {
                        'symbols': cluster_symbols,
                        'avg_correlation': cluster_returns.corr().mean().mean(),
                        'volatility': cluster_returns.std().mean(),
                        'size': len(cluster_symbols)
                    }
                    
            # Update cluster maps
            self.cluster_maps = cluster_metrics
            
            return {
                'clusters': clusters,
                'metrics': cluster_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Cluster analysis error: {str(e)}")
            return {}
            
    async def forecast_volatility(self, symbol: str, lookback: int = 100) -> Dict:
        """Forecast volatility using GARCH model"""
        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, lookback)
            if rates is None:
                return {}
                
            # Prepare data
            df = pd.DataFrame(rates)
            returns = 100 * df['close'].pct_change().dropna()
            
            # Fit GARCH model
            model = arch_model(
                returns,
                vol='Garch',
                p=1,
                q=1,
                dist='normal'
            )
            model_fit = model.fit(disp='off')
            
            # Generate forecast
            forecast = model_fit.forecast(horizon=24)
            variance = forecast.variance.values[-1]
            
            # Calculate forecast metrics
            forecast_vol = np.sqrt(variance) * np.sqrt(24)  # Scale to daily
            current_vol = returns.std() * np.sqrt(24)
            
            forecast_data = {
                'current_volatility': current_vol,
                'forecast_volatility': forecast_vol,
                'change_percent': ((forecast_vol - current_vol) / current_vol) * 100,
                'forecast_horizon': 24,
                'confidence_intervals': {
                    '95': (forecast_vol - 1.96 * forecast_vol/np.sqrt(lookback),
                          forecast_vol + 1.96 * forecast_vol/np.sqrt(lookback))
                }
            }
            
            # Update volatility forecasts
            self.volatility_forecasts[symbol] = forecast_data
            
            return forecast_data
            
        except Exception as e:
            self.logger.error(f"Volatility forecasting error: {str(e)}")
            return {}
            
    async def detect_market_regime(
        self,
        symbol: str,
        lookback: int = 100
    ) -> Dict:
        """Detect current market regime"""
        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, lookback)
            if rates is None:
                return {}
                
            df = pd.DataFrame(rates)
            
            # Calculate technical indicators
            df['sma20'] = df['close'].rolling(20).mean()
            df['sma50'] = df['close'].rolling(50).mean()
            df['std20'] = df['close'].rolling(20).std()
            df['rsi'] = self._calculate_rsi(df['close'])
            
            # Define regime characteristics
            regime_data = {
                'trend': await self._detect_trend(df),
                'volatility': await self._classify_volatility(df),
                'momentum': await self._analyze_momentum(df),
                'support_resistance': await self._find_support_resistance(df)
            }
            
            # Determine overall regime
            regime = await self._classify_regime(regime_data)
            
            # Update regime states
            self.regime_states[symbol] = {
                'regime': regime,
                'characteristics': regime_data,
                'timestamp': datetime.now()
            }
            
            return self.regime_states[symbol]
            
        except Exception as e:
            self.logger.error(f"Regime detection error: {str(e)}")
            return {}
            
    async def _detect_trend(self, df: pd.DataFrame) -> Dict:
        """Detect market trend"""
        try:
            current_price = df['close'].iloc[-1]
            sma20 = df['sma20'].iloc[-1]
            sma50 = df['sma50'].iloc[-1]
            
            # Calculate trend strength
            trend_strength = abs(sma20 - sma50) / df['std20'].iloc[-1]
            
            # Determine trend direction
            if current_price > sma20 > sma50:
                direction = 'uptrend'
            elif current_price < sma20 < sma50:
                direction = 'downtrend'
            else:
                direction = 'sideways'
                
            return {
                'direction': direction,
                'strength': trend_strength,
                'sma20_slope': (df['sma20'].iloc[-1] - df['sma20'].iloc[-5]) / 5
            }
            
        except Exception as e:
            self.logger.error(f"Trend detection error: {str(e)}")
            return {}
            
    async def _classify_volatility(self, df: pd.DataFrame) -> Dict:
        """Classify volatility regime"""
        try:
            # Calculate volatility metrics
            current_vol = df['std20'].iloc[-1]
            avg_vol = df['std20'].mean()
            vol_ratio = current_vol / avg_vol
            
            # Classify volatility
            if vol_ratio > 1.5:
                regime = 'high'
            elif vol_ratio < 0.75:
                regime = 'low'
            else:
                regime = 'normal'
                
            return {
                'regime': regime,
                'current': current_vol,
                'average': avg_vol,
                'ratio': vol_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Volatility classification error: {str(e)}")
            return {}
            
    async def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze price momentum"""
        try:
            rsi = df['rsi'].iloc[-1]
            rsi_slope = df['rsi'].iloc[-1] - df['rsi'].iloc[-5]
            
            # Classify momentum
            if rsi > 70:
                strength = 'overbought'
            elif rsi < 30:
                strength = 'oversold'
            else:
                strength = 'neutral'
                
            return {
                'rsi': rsi,
                'strength': strength,
                'slope': rsi_slope
            }
            
        except Exception as e:
            self.logger.error(f"Momentum analysis error: {str(e)}")
            return {}
            
    async def _find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        try:
            # Find local extrema
            highs = df[df['high'] == df['high'].rolling(10).max()]
            lows = df[df['low'] == df['low'].rolling(10).min()]
            
            current_price = df['close'].iloc[-1]
            
            # Find nearest levels
            support = lows[lows['low'] < current_price]['low'].max()
            resistance = highs[highs['high'] > current_price]['high'].min()
            
            return {
                'support': support,
                'resistance': resistance,
                'range_size': resistance - support if support and resistance else None
            }
            
        except Exception as e:
            self.logger.error(f"Support/Resistance detection error: {str(e)}")
            return {}
            
    async def _classify_regime(self, regime_data: Dict) -> str:
        """Classify overall market regime"""
        try:
            trend = regime_data['trend']['direction']
            volatility = regime_data['volatility']['regime']
            momentum = regime_data['momentum']['strength']
            
            # Define regime classification rules
            if trend == 'uptrend' and momentum != 'overbought':
                return 'trending_up'
            elif trend == 'downtrend' and momentum != 'oversold':
                return 'trending_down'
            elif trend == 'sideways' and volatility == 'low':
                return 'ranging'
            elif volatility == 'high':
                return 'volatile'
            else:
                return 'transitioning'
                
        except Exception as e:
            self.logger.error(f"Regime classification error: {str(e)}")
            return 'unknown'
            
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            return 100 - (100 / (1 + rs))
            
        except Exception as e:
            self.logger.error(f"RSI calculation error: {str(e)}")
            return pd.Series()
