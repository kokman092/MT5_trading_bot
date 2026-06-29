import asyncio
import os
import sys
import pandas as pd
from dotenv import load_dotenv
import MetaTrader5 as mt5
from modules.trading.market_analyzer import MarketAnalyzer

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
    
    print("3. Fetching raw market data via MarketAnalyzer...")
    data = await analyzer._get_market_data(symbol, timeframe, 1000)
    if data is None or data.empty:
        print("Failed to get market data!")
        mt5.shutdown()
        return
    print(f"Data retrieved successfully. Rows: {len(data)}")
    
    print("4. Testing _calculate_volatility...")
    vol = analyzer._calculate_volatility(data, timeframe)
    print(f"Volatility calculated: {vol}")
    
    print("5. Testing _determine_market_phase...")
    phase = analyzer._determine_market_phase(data)
    print(f"Market phase calculated: {phase}")
    
    print("6. Testing _calculate_trend_strength...")
    trend_str = analyzer._calculate_trend_strength(data)
    print(f"Trend strength calculated: {trend_str}")
    
    print("7. Testing support and resistance levels...")
    supports = analyzer._find_support_levels(data)
    resistances = analyzer._find_resistance_levels(data)
    print(f"Supports found: {len(supports)}, Resistances found: {len(resistances)}")
    
    print("8. Testing _analyze_market_structure...")
    structure = await analyzer._analyze_market_structure(data, timeframe)
    print("Market structure completed!")
    
    print("9. Testing _generate_trend_signals...")
    trend_sigs = analyzer._generate_trend_signals(data, symbol)
    print(f"Trend signals generated: {len(trend_sigs)}")
    
    print("10. Testing _generate_momentum_signals...")
    mom_sigs = analyzer._generate_momentum_signals(data, symbol)
    print(f"Momentum signals generated: {len(mom_sigs)}")
    
    print("11. Testing _generate_volatility_signals...")
    vol_sigs = analyzer._generate_volatility_signals(data, vol, symbol)
    print(f"Volatility signals generated: {len(vol_sigs)}")
    
    print("12. Testing _generate_sr_signals...")
    sr_sigs = analyzer._generate_sr_signals(data, structure, symbol)
    print(f"S/R signals generated: {len(sr_sigs)}")
    
    print("\nDIAGNOSTICS PASSED! All analysis functions are stable.")
    mt5.shutdown()

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
