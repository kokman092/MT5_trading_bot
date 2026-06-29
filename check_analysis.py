import asyncio
import os
import sys
import pandas as pd
from dotenv import load_dotenv
import MetaTrader5 as mt5
import ta
import pandas_ta as pta
from modules.trading.market_analyzer import MarketAnalyzer

async def debug_get_market_data(analyzer, symbol: str, timeframe: str, bars: int = 1000):
    print("  [DEBUG-DATA] Starting debug_get_market_data...")
    
    print("  [DEBUG-DATA] 1. Calling mt5.symbol_select...")
    if not mt5.symbol_select(symbol, True):
        print("  [DEBUG-DATA] Symbol select failed!")
        return None
    print("  [DEBUG-DATA] Symbol selected successfully.")
        
    print("  [DEBUG-DATA] 2. Calling mt5.symbol_info...")
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print("  [DEBUG-DATA] Symbol info returned None!")
        return None
    print(f"  [DEBUG-DATA] Symbol info retrieved. Point={symbol_info.point}")
    
    print("  [DEBUG-DATA] 3. Translating timeframe value...")
    tf_val = analyzer._get_timeframe_value(timeframe)
    print(f"  [DEBUG-DATA] Timeframe value: {tf_val}")
    
    print("  [DEBUG-DATA] 4. Calling mt5.copy_rates_from_pos...")
    rates = mt5.copy_rates_from_pos(symbol, tf_val, 0, bars)
    if rates is None:
        print("  [DEBUG-DATA] copy_rates_from_pos returned None!")
        return None
    print(f"  [DEBUG-DATA] copy_rates_from_pos retrieved {len(rates)} bars.")
    
    print("  [DEBUG-DATA] 5. Creating pd.DataFrame...")
    df = pd.DataFrame(rates)
    print("  [DEBUG-DATA] DataFrame created.")
    
    print("  [DEBUG-DATA] 6. Converting time column to datetime...")
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print("  [DEBUG-DATA] Datetime conversion complete.")
    
    print("  [DEBUG-DATA] 7. Adding symbol column...")
    df['symbol'] = symbol
    print("  [DEBUG-DATA] Symbol column added.")
    
    print("  [DEBUG-DATA] 8. Calculating Trend indicators (SMA/EMA/ADX)...")
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    print("  [DEBUG-DATA] SMA 20 calculated.")
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    print("  [DEBUG-DATA] EMA 20 calculated.")
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    print("  [DEBUG-DATA] ADX calculated.")
    
    print("  [DEBUG-DATA] 9. Calculating RSI...")
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    print("  [DEBUG-DATA] RSI calculated.")
    
    print("  [DEBUG-DATA] 10. Calculating MACD...")
    macd_indicator = ta.trend.MACD(
        close=df['close'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    print("  [DEBUG-DATA] MACD calculated.")
    
    print("  [DEBUG-DATA] 11. Calculating Bollinger Bands...")
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bollinger_upper'] = bollinger.bollinger_hband()
    df['bollinger_lower'] = bollinger.bollinger_lband()
    print("  [DEBUG-DATA] Bollinger Bands calculated.")
    
    print("  [DEBUG-DATA] 12. Calculating ATR...")
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    print("  [DEBUG-DATA] ATR calculated.")
    
    print("  [DEBUG-DATA] 13. Forward filling NaN values...")
    df.ffill(inplace=True)
    print("  [DEBUG-DATA] Forward fill complete.")
    df.fillna(0, inplace=True)
    print("  [DEBUG-DATA] Fillna 0 complete.")
    
    print("  [DEBUG-DATA] 14. Validating market data...")
    val_res = analyzer._validate_market_data(df)
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
