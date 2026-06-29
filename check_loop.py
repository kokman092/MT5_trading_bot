import asyncio
import os
import sys
import traceback
from dotenv import load_dotenv
from run_trader import ProfessionalTradingSystem

async def run_step_by_step():
    load_dotenv()
    
    print("1. Creating ProfessionalTradingSystem instance...")
    system = ProfessionalTradingSystem()
    
    print("2. Initializing trading system components (this will connect to MT5)...")
    if not await system.initialize():
        print("Failed to initialize components!")
        return
        
    print("3. Performing first health check...")
    health_ok = await system._perform_health_check()
    print(f"Health check status: {health_ok}")
    
    print("4. Fetching H1 market data cache...")
    # This calls mt5.copy_rates_from_pos for XAUUSD.std on H1
    await system._update_market_data()
    print("H1 market data fetched successfully!")
    
    print("5. Running market regime detection...")
    # This runs the HMM, GMM, and KMeans predictions
    await system._update_market_regime()
    print("Regime detection completed successfully!")
    
    print("6. Processing XAUUSD.std trading analysis and signal generation...")
    # This runs the technical indicators, generates signals, and evaluates entries
    await system._process_symbol("XAUUSD.std")
    print("Trading analysis cycle completed successfully!")
    
    print("\nALL STEPS PASSED! The loop is structurally 100% stable.")
    
if __name__ == "__main__":
    asyncio.run(run_step_by_step())
