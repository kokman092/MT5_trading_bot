import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta

class FinancialFeatureEngineering:
    """Advanced financial feature engineering based on Lopez de Prado's techniques"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced financial features"""
        try:
            features = df.copy()
            
            # Volume-based features
            features = self._add_volume_features(features)
            
            # Price-based features
            features = self._add_price_features(features)
            
            # Microstructure features
            features = self._add_microstructure_features(features)
            
            # Orderbook features
            features = self._add_orderbook_features(features)
            
            # Volatility features
            features = self._add_volatility_features(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {str(e)}")
            return df
            
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        try:
            # VWAP
            df['vwap'] = (df['high'] + df['low'] + df['close']) / 3 * df['tick_volume']
            df['vwap'] = df['vwap'].cumsum() / df['tick_volume'].cumsum()
            
            # Volume Imbalance
            df['volume_imbalance'] = self._calculate_volume_imbalance(df)
            
            # Volume Variance
            df['volume_variance'] = df['tick_volume'].rolling(window=20).std()
            
            # Relative Volume
            df['relative_volume'] = df['tick_volume'] / df['tick_volume'].rolling(window=20).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding volume features: {str(e)}")
            return df
            
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        try:
            # Price differences
            df['price_diff'] = df['close'].diff()
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # Triple barrier labels
            df['triple_barrier'] = self._triple_barrier_labels(
                df['close'],
                upper_barrier=0.01,  # 1% profit target
                lower_barrier=-0.005,  # 0.5% stop loss
                max_holding=24  # 24 periods max holding
            )
            
            # Fractional differentiation
            df['frac_diff'] = self._fractional_differentiation(
                df['close'],
                d=0.4  # Differentiation order
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding price features: {str(e)}")
            return df
            
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        try:
            # Bid-ask spread estimator (Roll's model)
            df['roll_spread'] = self._rolls_spread(df['close'])
            
            # Kyle's lambda (price impact)
            df['kyle_lambda'] = self._kyle_lambda(df)
            
            # Amihud's illiquidity
            df['illiquidity'] = abs(df['log_return']) / df['tick_volume']
            
            # Tick Rule
            df['tick_rule'] = np.where(df['price_diff'] > 0, 1,
                                     np.where(df['price_diff'] < 0, -1, 0))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding microstructure features: {str(e)}")
            return df
            
    def _add_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add orderbook-based features"""
        try:
            # Order flow imbalance
            df['order_flow_imbalance'] = self._calculate_order_flow_imbalance(df)
            
            # Price impact
            df['price_impact'] = self._calculate_price_impact(df)
            
            # Order book pressure
            df['book_pressure'] = self._calculate_book_pressure(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding orderbook features: {str(e)}")
            return df
            
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        try:
            # Garman-Klass volatility
            df['garman_klass_vol'] = self._garman_klass_volatility(df)
            
            # Rogers-Satchell volatility
            df['rogers_satchell_vol'] = self._rogers_satchell_volatility(df)
            
            # Yang-Zhang volatility
            df['yang_zhang_vol'] = self._yang_zhang_volatility(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding volatility features: {str(e)}")
            return df
            
    def _calculate_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume imbalance"""
        try:
            # Buyer-initiated volume vs seller-initiated volume
            return np.where(df['close'] > df['close'].shift(1),
                          df['tick_volume'],
                          -df['tick_volume']).cumsum()
        except Exception as e:
            self.logger.error(f"Error calculating volume imbalance: {str(e)}")
            return pd.Series(0, index=df.index)
            
    def _triple_barrier_labels(self, price: pd.Series,
                             upper_barrier: float,
                             lower_barrier: float,
                             max_holding: int) -> pd.Series:
        """Implement triple-barrier labeling method"""
        try:
            labels = pd.Series(0, index=price.index)
            for i in range(len(price) - max_holding):
                returns = price[i+1:i+max_holding+1] / price[i] - 1
                if (returns >= upper_barrier).any():
                    labels[i] = 1  # Profit target hit
                elif (returns <= lower_barrier).any():
                    labels[i] = -1  # Stop loss hit
                elif returns[-1] > 0:
                    labels[i] = 1  # End of period, positive return
                elif returns[-1] < 0:
                    labels[i] = -1  # End of period, negative return
            return labels
            
        except Exception as e:
            self.logger.error(f"Error calculating triple barrier labels: {str(e)}")
            return pd.Series(0, index=price.index)
            
    def _fractional_differentiation(self, series: pd.Series,
                                  d: float,
                                  window: int = 100) -> pd.Series:
        """Implement fractional differentiation"""
        try:
            weights = [1.]
            for k in range(1, window):
                weights.append(-weights[-1] * (d - k + 1) / k)
            weights = np.array(weights)
            return pd.Series(
                np.convolve(series, weights, mode='valid'),
                index=series.index[window-1:]
            )
        except Exception as e:
            self.logger.error(f"Error calculating fractional diff: {str(e)}")
            return series
            
    def _rolls_spread(self, price: pd.Series, window: int = 20) -> pd.Series:
        """Implement Roll's spread estimator"""
        try:
            price_diff = price.diff()
            spread = pd.Series(index=price.index)
            for i in range(window, len(price)):
                cov = np.cov(price_diff[i-window:i], price_diff[i-window+1:i+1])[0,1]
                spread[i] = 2 * np.sqrt(-cov) if cov < 0 else 0
            return spread
            
        except Exception as e:
            self.logger.error(f"Error calculating Roll's spread: {str(e)}")
            return pd.Series(0, index=price.index)
            
    def _kyle_lambda(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Implement Kyle's lambda (price impact)"""
        try:
            price_diff = df['close'].diff()
            volume = df['tick_volume']
            lambda_series = pd.Series(index=df.index)
            
            for i in range(window, len(df)):
                reg = np.polyfit(volume[i-window:i], price_diff[i-window:i], 1)
                lambda_series[i] = reg[0]
                
            return lambda_series
            
        except Exception as e:
            self.logger.error(f"Error calculating Kyle's lambda: {str(e)}")
            return pd.Series(0, index=df.index)
            
    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate order flow imbalance"""
        try:
            # Proxy using tick rule and volume
            return df['tick_rule'] * df['tick_volume']
        except Exception as e:
            self.logger.error(f"Error calculating order flow imbalance: {str(e)}")
            return pd.Series(0, index=df.index)
            
    def _calculate_price_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price impact"""
        try:
            return abs(df['log_return']) / df['tick_volume']
        except Exception as e:
            self.logger.error(f"Error calculating price impact: {str(e)}")
            return pd.Series(0, index=df.index)
            
    def _calculate_book_pressure(self, df: pd.DataFrame) -> pd.Series:
        """Calculate order book pressure"""
        try:
            # Proxy using price differences and volume
            return (df['close'] - df['close'].shift(1)) * df['tick_volume']
        except Exception as e:
            self.logger.error(f"Error calculating book pressure: {str(e)}")
            return pd.Series(0, index=df.index)
            
    def _garman_klass_volatility(self, df: pd.DataFrame,
                                window: int = 20) -> pd.Series:
        """Calculate Garman-Klass volatility"""
        try:
            log_hl = np.log(df['high'] / df['low'])
            log_co = np.log(df['close'] / df['open'])
            vol = 0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
            return np.sqrt(vol.rolling(window=window).mean())
            
        except Exception as e:
            self.logger.error(f"Error calculating GK volatility: {str(e)}")
            return pd.Series(0, index=df.index)
            
    def _rogers_satchell_volatility(self, df: pd.DataFrame,
                                   window: int = 20) -> pd.Series:
        """Calculate Rogers-Satchell volatility"""
        try:
            log_ho = np.log(df['high'] / df['open'])
            log_lo = np.log(df['low'] / df['open'])
            log_hc = np.log(df['high'] / df['close'])
            log_lc = np.log(df['low'] / df['close'])
            vol = log_ho * log_hc + log_lo * log_lc
            return np.sqrt(vol.rolling(window=window).mean())
            
        except Exception as e:
            self.logger.error(f"Error calculating RS volatility: {str(e)}")
            return pd.Series(0, index=df.index)
            
    def _yang_zhang_volatility(self, df: pd.DataFrame,
                              window: int = 20) -> pd.Series:
        """Calculate Yang-Zhang volatility"""
        try:
            log_ho = np.log(df['high'] / df['open'])
            log_lo = np.log(df['low'] / df['open'])
            log_co = np.log(df['close'] / df['open'])
            rs_vol = self._rogers_satchell_volatility(df, window)
            close_vol = df['close'].pct_change().rolling(window=window).std()
            open_vol = (df['open'] / df['close'].shift(1)).apply(np.log).rolling(window=window).std()
            
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            vol = (open_vol**2 + k * close_vol**2 + (1-k) * rs_vol**2)
            return np.sqrt(vol)
            
        except Exception as e:
            self.logger.error(f"Error calculating YZ volatility: {str(e)}")
            return pd.Series(0, index=df.index)
