import asyncio
import logging
from modules.trading.risk_manager import RiskManager
from modules.config.config_manager import ConfigManager

logging.basicConfig(level=logging.INFO)

async def test_risk_math():
    config_mgr = ConfigManager()
    config = config_mgr.load_config("config/config.json")
    
    risk_manager = RiskManager(config)
    
    market_data = {
        'regime': {'regime': 'trend'},
        'confidence': 0.8,
        'volatility': 1.2,
        'signal_strength': 0.9,
        'strategy_type': 'trend_following'
    }
    
    # Simulate the gold trade from the screenshot
    entry = 4055.43
    sl = 4062.45
    
    try:
        size, metrics = await risk_manager.calculate_position_size(
            symbol="XAUUSD",
            entry_price=entry,
            stop_loss=sl,
            account_balance=100000.0,
            market_data=market_data
        )
        print(f"\nSUCCESS! Calculated Size: {size}")
        print(f"Metrics: {metrics}")
    except Exception as e:
        print(f"\nCRASHED! Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_risk_math())
