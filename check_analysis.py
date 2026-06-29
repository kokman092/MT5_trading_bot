import asyncio
import os
import sys
import pandas as pd
from dotenv import load_dotenv
import MetaTrader5 as mt5
import ta
import pandas_ta as pta
from modules.trading.market_analyzer import MarketAnalyzer

def debug_validate_market_data(analyzer, df: pd.DataFrame) -> bool:
    print("    [DEBUG-VAL] Starting debug_validate_market_data...")
    
    print("    [DEBUG-VAL] 1. Reading config['market_analysis']['validation']...")
    config = analyzer.config.get('market_analysis', {}).get('validation', {})
    print("    [DEBUG-VAL] Config read successfully.")
    
    validation_errors = []
    
    print("    [DEBUG-VAL] 2. Checking if empty...")
    if df is None or df.empty:
        print("    [DEBUG-VAL] DataFrame is empty!")
        return False
        
    print("    [DEBUG-VAL] 3. Checking minimum data points...")
    min_points = config.get('min_data_points', 100)
    if len(df) < min_points:
        validation_errors.append(f"Insufficient data points: {len(df)} < {min_points}")
    print(f"    [DEBUG-VAL] Data points check complete. Total rows: {len(df)}")
        
    print("    [DEBUG-VAL] 4. Checking required columns list...")
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_errors.append(f"Missing required price columns: {missing_columns}")
    print("    [DEBUG-VAL] Required columns check complete.")
        
    print("    [DEBUG-VAL] 5. Checking for gaps in time series...")
    if 'time' in df.columns:
        print("    [DEBUG-VAL] 5.1. Calculating time diff...")
        time_diff = df['time'].diff()
        print("    [DEBUG-VAL] 5.2. Reading max gap...")
        max_gap_hours = config.get('max_gap', 24)
        
        print("    [DEBUG-VAL] 5.3. Filtering large gaps...")
        # Check large gaps to filter out standard weekend closures
        time_diff_hours = time_diff.dt.total_seconds() / 3600.0
        large_gaps = time_diff[time_diff_hours > max_gap_hours]
        print(f"    [DEBUG-VAL] Found {len(large_gaps)} large gaps.")
        
        print("    [DEBUG-VAL] 5.4. Looping through large gaps indices...")
        for idx in large_gaps.index:
            print(f"      [DEBUG-VAL] Checking gap index: {idx}")
            t1 = df['time'].iloc[idx-1]
            t2 = df['time'].iloc[idx]
            gap_seconds = (t2 - t1).total_seconds()
            
            is_weekend = False
            if gap_seconds <= 75 * 3600:
                if t1.weekday() in [4, 5, 6] or t2.weekday() in [5, 6, 0]:
                    is_weekend = True
                    
            if not is_weekend:
                validation_errors.append(
                    f"Data gap detected: {gap_seconds / 3600:.2f} hours between {t1} and {t2}"
                )
                break
        print("    [DEBUG-VAL] Gaps check complete.")
            
    print("    [DEBUG-VAL] 6. Validating price data...")
    if all(col in df.columns for col in required_columns):
        print("    [DEBUG-VAL] 6.1. Checking for non-positive prices...")
        # Check for non-positive prices
        has_negative = (df[required_columns] <= 0).any().any()
        print(f"    [DEBUG-VAL] Negative prices check complete. Result: {has_negative}")
        if has_negative:
            validation_errors.append("Invalid price values detected (zero or negative)")
            
        print("    [DEBUG-VAL] 6.2. Checking high/low relationship...")
        # Check high/low relationship
        has_invalid_hl = (df['high'] < df['low']).any()
        print(f"    [DEBUG-VAL] High/low check complete. Result: {has_invalid_hl}")
        if has_invalid_hl:
            validation_errors.append("Invalid high/low price relationship detected")
            
        print("    [DEBUG-VAL] 6.3. Checking extreme price changes...")
        max_price_change = config.get('max_price_change_percent', 10)
        if 'symbol' in df.columns and 'XAU' in df['symbol'].iloc[0]:
            max_price_change *= 2  # Gold can be more volatile
            
        print("    [DEBUG-VAL] 6.4. Calculating percent change...")
        price_changes = df['close'].pct_change().abs() * 100
        print("    [DEBUG-VAL] 6.5. Evaluating extreme change criteria...")
        has_extreme_change = (price_changes > max_price_change).any()
        print(f"    [DEBUG-VAL] Extreme change check complete. Result: {has_extreme_change}")
        if has_extreme_change:
            validation_errors.append(
                f"Extreme price changes detected: {price_changes.max():.2f}% > {max_price_change}%"
            )
    print("    [DEBUG-VAL] Price validation complete.")
            
    print("    [DEBUG-VAL] 7. Handling volume data...")
    if 'volume' in df.columns:
        print("    [DEBUG-VAL] 7.1. Checking if volume is all zeros...")
        if (df['volume'] == 0).all() and 'tick_volume' in df.columns:
            df['volume'] = df['tick_volume']
        elif (df['volume'] == 0).all():
            df['volume'] = 1
            
        print("    [DEBUG-VAL] 7.2. Checking minimum volume...")
        min_volume = config.get('min_volume', 0)
        if 'symbol' in df.columns and 'XAU' in df['symbol'].iloc[0]:
            min_volume = 0
            
        if min_volume > 0:
            avg_volume = df['volume'].mean()
            if avg_volume < min_volume:
                validation_errors.append(f"Insufficient volume: {avg_volume:.2f} < {min_volume}")
    print("    [DEBUG-VAL] Volume validation complete.")
                
    if validation_errors:
        print(f"    [DEBUG-VAL] Validation errors found: {validation_errors}")
        return False
        
    print("    [DEBUG-VAL] Validation PASSED!")
    return True

async def debug_get_market_data(analyzer, symbol: str, timeframe: str, bars: int = 1000):
    print("  [DEBUG-DATA] Starting debug_get_market_data...")
    if not mt5.symbol_select(symbol, True):
        return None
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return None
    tf_val = analyzer._get_timeframe_value(timeframe)
    rates = mt5.copy_rates_from_pos(symbol, tf_val, 0, bars)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['symbol'] = symbol
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd_indicator = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bollinger_upper'] = bollinger.bollinger_hband()
    df['bollinger_lower'] = bollinger.bollinger_lband()
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    
    print("  [DEBUG-DATA] 14. Validating market data...")
    val_res = debug_validate_market_data(analyzer, df)
    print(f"  [DEBUG-DATA] Validation result: {val_res}")
    
    return df

async def run_diagnostics():
    load_dotenv()
    login = int(os.getenv("MT5_LOGIN"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    
    print("1. Initializing MT5 connection...")
    if not mt5.initialize(login=login, password=password, server=server):
        print("MT5 Init failed!")
        return
        
    symbol = "XAUUSD.std"
    timeframe = "M15"
    
    import json
    with open("config/config.json", "r") as f:
        config = json.load(f)
        
    print("2. Initializing MarketAnalyzer components...")
    analyzer = MarketAnalyzer(config)
    
    print("3. Fetching raw market data via debug_get_market_data...")
    data = await debug_get_market_data(analyzer, symbol, timeframe, 1000)
    if data is None or data.empty:
        print("Failed to get market data!")
        mt5.shutdown()
        return
    print(f"Data retrieved successfully. Rows: {len(data)}")
    
    print("DIAGNOSTICS COMPLETED SUCCESSFULLY!")
    mt5.shutdown()

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
