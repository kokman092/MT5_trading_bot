import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from run_trader import ProfessionalTradingSystem
from modules.trading.ml_analyzer import MLAnalyzer

async def main():
    print("Initializing Professional Trading System...")
    system = ProfessionalTradingSystem()
    config = system._load_config()
    
    # Instantiate MLAnalyzer
    print("Initializing MLAnalyzer...")
    ml_analyzer = MLAnalyzer(config)
    await ml_analyzer.initialize()
    
    # Train for each symbol and timeframe
    symbols = config['trading']['symbols']
    timeframes = config['trading']['timeframes']
    
    print(f"Symbols to train: {symbols}")
    print(f"Timeframes to train: {timeframes}")
    
    for symbol in symbols:
        for timeframe in timeframes:
            csv_path = Path(f"data/ml_training_data_{symbol}_{timeframe}.csv")
            if not csv_path.exists():
                print(f"Skipping {symbol} {timeframe} - no data file found.")
                continue
                
            print(f"Loading data for {symbol} {timeframe}...")
            df = pd.read_csv(csv_path)
            
            # Ensure time index is set
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                
            # Create target column: 1 if next bar's close is higher than current bar's close, else 0
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df.dropna(subset=['target'], inplace=True)
            
            print(f"Training MLAnalyzer ensemble for {symbol}_{timeframe} with {len(df)} samples...")
            success = await ml_analyzer.train_model(df, symbol, timeframe)
            if success:
                print(f"Successfully trained and saved model for {symbol}_{timeframe}!")
            else:
                print(f"Failed to train model for {symbol}_{timeframe}.")

if __name__ == "__main__":
    asyncio.run(main())
