import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import MetaTrader5 as mt5
from modules.backtesting.backtester import Backtester

def execute_backtest():
    load_dotenv()
    login = int(os.getenv("MT5_LOGIN"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    
    print("1. Initializing MT5 connection for historical data...")
    if not mt5.initialize(login=login, password=password, server=server):
        print("MT5 Initialization failed!")
        return
        
    print("MT5 Connected successfully.")
    
    # Define configuration specifically mapped to backtester keys
    backtest_config = {
        "risk_management": {
            "risk_per_trade": 0.02,            # 2% risk per trade
            "stop_loss_atr_multiplier": 1.5,   # 1.5x ATR Stop Loss
            "risk_reward_ratio": 2.0           # 2.0x Take Profit multiplier (1:2 R/R)
        },
        "market_analysis": {
            "volatility": {
                "min_threshold": 0.0001,       # Minimum required ATR percentage
                "max_threshold": 2.0           # Allow up to 2% ATR range (normal for Gold)
            }
        }
    }
    
    # 30 days history range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Detect valid Gold symbol on this broker
    gold_symbol = 'XAUUSD'
    for candidate in ['XAUUSD', 'XAUUSD.std', 'GOLD', 'GOLD.std']:
        if mt5.symbol_select(candidate, True):
            gold_symbol = candidate
            break
            
    params = {
        'symbol': gold_symbol,
        'timeframe': 'M15',
        'start_date': start_date.strftime('%Y-%m-%d %H:%M'),
        'end_date': end_date.strftime('%Y-%m-%d %H:%M'),
        'initial_balance': 100000.0            # $100k demo balance
    }
    
    print(f"\n2. Launching Backtest for {params['symbol']} on {params['timeframe']}...")
    print(f"   Period: {params['start_date']} to {params['end_date']}")
    
    backtester = Backtester(backtest_config)
    result = backtester.run_backtest(params)
    
    if not result.get('success'):
        print(f"Backtest failed: {result.get('error')}")
        mt5.shutdown()
        return
        
    res = result['results']
    print("\n================ BACKTEST RESULTS ================")
    print(f"Initial Balance  : ${res['initial_balance']:,.2f}")
    print(f"Final Balance    : ${res['final_balance']:,.2f}")
    print(f"Total Return %   : {res['total_return']:.2f}%")
    print(f"Total Trades     : {res['total_trades']}")
    print(f"Win Rate         : {res['win_rate']:.2f}%")
    print(f"Max Drawdown     : {res['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio     : {res['sharpe_ratio']:.2f}")
    print(f"Max Consecutive Loss: {res['max_consecutive_losses']}")
    print("==================================================")
    
    mt5.shutdown()

if __name__ == "__main__":
    execute_backtest()
