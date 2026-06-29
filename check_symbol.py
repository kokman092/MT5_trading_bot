import MetaTrader5 as mt5

import os
from dotenv import load_dotenv

def check_gold():
    load_dotenv()
    login = int(os.getenv("MT5_LOGIN"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    
    print(f"Attempting login to {server} for {login}...")
    if not mt5.initialize(login=login, password=password, server=server):
        print(f"initialize() failed. Error: {mt5.last_error()}")
        return
        
    print(f"MT5 Connected! Build: {mt5.version()}")
    
    # Try selecting XAUUSD
    if mt5.symbol_select("XAUUSD", True):
        print("XAUUSD selected successfully!")
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 1000)
        if rates is None:
            print(f"Failed to get rates for XAUUSD. Error: {mt5.last_error()}")
        else:
            print(f"Successfully got {len(rates)} bars for XAUUSD!")
    else:
        print(f"Could not select XAUUSD. Error: {mt5.last_error()}")
        
    # Search for all symbols that might be Gold
    symbols = mt5.symbols_get()
    if symbols:
        gold_symbols = [s.name for s in symbols if "XAU" in s.name or "GOLD" in s.name.upper()]
        print(f"\nFound potential Gold symbols on your broker: {gold_symbols}")
    
    mt5.shutdown()

if __name__ == "__main__":
    check_gold()
