import numpy as np
import pandas as pd
import ta
from typing import Dict, List, Optional, Tuple
from scipy import stats

def calculate_support_resistance(df: pd.DataFrame, window: int = 20, num_points: int = 5) -> Dict[str, List[float]]:
    """Calculate support and resistance levels using price swings"""
    highs = df['high'].rolling(window=window, center=True).apply(lambda x: is_peak(x, num_points))
    lows = df['low'].rolling(window=window, center=True).apply(lambda x: is_trough(x, num_points))
    
    resistance_levels = cluster_levels(df[highs]['high'].dropna().values)
    support_levels = cluster_levels(df[lows]['low'].dropna().values)
    
    return {
        'support': support_levels,
        'resistance': resistance_levels
    }

def is_peak(x: np.array, num_points: int) -> bool:
    """Check if the center point is a peak"""
    if len(x) < 2 * num_points + 1:
        return False
    center_idx = len(x) // 2
    center_val = x[center_idx]
    
    left_points = x[center_idx - num_points:center_idx]
    right_points = x[center_idx + 1:center_idx + num_points + 1]
    
    return all(center_val >= left_points) and all(center_val >= right_points)

def is_trough(x: np.array, num_points: int) -> bool:
    """Check if the center point is a trough"""
    if len(x) < 2 * num_points + 1:
        return False
    center_idx = len(x) // 2
    center_val = x[center_idx]
    
    left_points = x[center_idx - num_points:center_idx]
    right_points = x[center_idx + 1:center_idx + num_points + 1]
    
    return all(center_val <= left_points) and all(center_val <= right_points)

def cluster_levels(prices: np.array, tolerance: float = 0.001) -> List[float]:
    """Cluster similar price levels together"""
    if len(prices) == 0:
        return []
        
    clusters = []
    current_cluster = [prices[0]]
    
    for price in prices[1:]:
        if abs(price - np.mean(current_cluster)) / np.mean(current_cluster) <= tolerance:
            current_cluster.append(price)
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [price]
            
    clusters.append(np.mean(current_cluster))
    return sorted(clusters)

def calculate_volatility_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate various volatility metrics"""
    # ATR
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'])
    current_atr = atr.average_true_range().iloc[-1]
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    bb_width = bb.bollinger_wband().iloc[-1]
    
    # Historical Volatility
    returns = np.log(df['close'] / df['close'].shift(1))
    hist_vol = returns.std() * np.sqrt(252)  # Annualized
    
    # Volatility Regime
    vol_percentile = stats.percentileofscore(
        returns.rolling(window=20).std().dropna(),
        returns.rolling(window=20).std().iloc[-1]
    )
    
    return {
        'atr': current_atr,
        'bb_width': bb_width,
        'historical_volatility': hist_vol,
        'volatility_percentile': vol_percentile,
        'is_high_volatility': vol_percentile > 80,
        'is_low_volatility': vol_percentile < 20
    }

def calculate_trend_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate various trend metrics"""
    # ADX for trend strength
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    current_adx = adx.adx().iloc[-1]
    
    # Moving averages
    ema_20 = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator().iloc[-1]
    ema_50 = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator().iloc[-1]
    ema_200 = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator().iloc[-1]
    
    # Linear regression
    x = np.arange(len(df['close']))
    slope, _, r_value, _, _ = stats.linregress(x[-20:], df['close'].iloc[-20:])
    
    return {
        'adx': current_adx,
        'trend_strength': 'strong' if current_adx > 25 else 'weak',
        'ema_20': ema_20,
        'ema_50': ema_50,
        'ema_200': ema_200,
        'slope': slope,
        'r_squared': r_value ** 2,
        'is_trending': current_adx > 25 and r_value ** 2 > 0.7
    }

def calculate_momentum_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate various momentum metrics"""
    # RSI
    rsi = ta.momentum.RSIIndicator(df['close'])
    current_rsi = rsi.rsi().iloc[-1]
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    current_macd = macd.macd().iloc[-1]
    current_signal = macd.macd_signal().iloc[-1]
    
    # Rate of Change
    roc = ta.momentum.ROCIndicator(df['close'])
    current_roc = roc.roc().iloc[-1]
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    current_k = stoch.stoch().iloc[-1]
    current_d = stoch.stoch_signal().iloc[-1]
    
    return {
        'rsi': current_rsi,
        'macd': current_macd,
        'macd_signal': current_signal,
        'macd_hist': current_macd - current_signal,
        'roc': current_roc,
        'stoch_k': current_k,
        'stoch_d': current_d,
        'is_overbought': current_rsi > 70 or (current_k > 80 and current_d > 80),
        'is_oversold': current_rsi < 30 or (current_k < 20 and current_d < 20)
    }

def detect_divergence(price: pd.Series, indicator: pd.Series, window: int = 20) -> Optional[str]:
    """Detect regular and hidden divergences"""
    # Get local extrema
    price_highs = price.rolling(window=window, center=True).apply(lambda x: is_peak(x, window//4))
    price_lows = price.rolling(window=window, center=True).apply(lambda x: is_trough(x, window//4))
    ind_highs = indicator.rolling(window=window, center=True).apply(lambda x: is_peak(x, window//4))
    ind_lows = indicator.rolling(window=window, center=True).apply(lambda x: is_trough(x, window//4))
    
    # Get last two extrema
    last_price_high = price[price_highs].dropna().iloc[-2:]
    last_price_low = price[price_lows].dropna().iloc[-2:]
    last_ind_high = indicator[ind_highs].dropna().iloc[-2:]
    last_ind_low = indicator[ind_lows].dropna().iloc[-2:]
    
    if len(last_price_high) == 2 and len(last_ind_high) == 2:
        price_trend = last_price_high.iloc[1] > last_price_high.iloc[0]
        ind_trend = last_ind_high.iloc[1] > last_ind_high.iloc[0]
        
        if price_trend and not ind_trend:
            return 'bearish'
        elif not price_trend and ind_trend:
            return 'bullish'
            
    if len(last_price_low) == 2 and len(last_ind_low) == 2:
        price_trend = last_price_low.iloc[1] > last_price_low.iloc[0]
        ind_trend = last_ind_low.iloc[1] > last_ind_low.iloc[0]
        
        if not price_trend and ind_trend:
            return 'bearish'
        elif price_trend and not ind_trend:
            return 'bullish'
            
    return None

def identify_chart_patterns(df: pd.DataFrame, window: int = 20) -> Dict[str, bool]:
    """Identify common chart patterns"""
    patterns = {
        'double_top': False,
        'double_bottom': False,
        'head_and_shoulders': False,
        'inverse_head_and_shoulders': False,
        'ascending_triangle': False,
        'descending_triangle': False
    }
    
    # Implement pattern recognition logic here
    # This is a placeholder for more sophisticated pattern recognition
    
    return patterns

def calculate_pivot_points(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate pivot points and their support/resistance levels"""
    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    close = df['close'].iloc[-1]
    
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

def analyze_volume_profile(df: pd.DataFrame, num_bins: int = 20) -> Dict[str, float]:
    """Analyze volume profile and identify key levels"""
    price_bins = pd.cut(df['close'], bins=num_bins)
    volume_profile = df.groupby(price_bins)['volume'].sum()
    
    poc_price = volume_profile.idxmax().mid  # Point of Control
    
    value_area = volume_profile.nlargest(int(num_bins * 0.7))
    va_high = value_area.index[-1].right
    va_low = value_area.index[0].left
    
    return {
        'poc': poc_price,
        'va_high': va_high,
        'va_low': va_low,
        'volume_profile': volume_profile.to_dict()
    } 