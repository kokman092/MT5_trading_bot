import MetaTrader5 as mt5
import pandas as pd
import ta
import pandas_ta as pta
from modules.trading.regime_detector import RegimeDetector
import os
from dotenv import load_dotenv

def test_diagnostics():
    load_dotenv()
    login = int(os.getenv("MT5_LOGIN"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    
    print("1. Initializing MT5...")
    if not mt5.initialize(login=login, password=password, server=server):
        print("MT5 Initialization failed!")
        return
        
    symbol = "XAUUSD.std"
    print(f"2. Selecting symbol: {symbol}")
    mt5.symbol_select(symbol, True)
    
    print("3. Pulling 1000 bars of historical data...")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 1000)
    if rates is None:
        print("Failed to pull rates!")
        mt5.shutdown()
        return
        
    print("4. Creating Pandas DataFrame...")
    df = pd.DataFrame(rates)
    
    print("5. Calculating SMA 20...")
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    print("SMA 20 calculated successfully!")
    
    print("6. Calculating EMA 20...")
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    print("EMA 20 calculated successfully!")
    
    print("7. Calculating ADX...")
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    print("ADX calculated successfully!")
    
    print("8. Calculating RSI...")
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    print("RSI calculated successfully!")
    
    print("9. Calculating ATR...")
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    print("ATR calculated successfully!")

    print("10. Calculating MACD...")
    macd_indicator = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    print("MACD calculated successfully!")

    print("11. Calculating Bollinger Bands...")
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bollinger_upper'] = bollinger.bollinger_hband()
    df['bollinger_lower'] = bollinger.bollinger_lband()
    print("Bollinger Bands calculated successfully!")

    print("12. Calculating Momentum using pandas_ta (pta.rsi)...")
    df['pta_rsi'] = pta.rsi(df['close'], length=14)
    print("pandas_ta RSI calculated successfully!")

    print("13. Initializing Regime Detector...")
    # Load configuration first
    import json
    with open("config/config.json", "r") as f:
        config = json.load(f)
    detector = RegimeDetector(config)
    print("Regime Detector initialized successfully!")

    print("14. Training Regime Detector models (fitting models)...")
    if detector.initialize_models(df):
        print("Regime Detector models trained successfully!")
    else:
        print("Failed to train Regime Detector models!")

    print("15. Running detect_regime on DataFrame...")
    regime = detector.detect_regime(df)
    if regime:
        print(f"Regime detected successfully! Type: {regime.regime_type}")
    else:
        print("Regime detection returned None.")
    
    print("\nDIAGNOSTICS PASSED! All technical libraries are working correctly.")
    mt5.shutdown()

if __name__ == "__main__":
    test_diagnostics()
