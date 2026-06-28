import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def initialize_mt5():
    """Initialize MT5 connection"""
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return False
        
    # Login to MT5
    login = int(os.getenv('MT5_LOGIN'))
    password = os.getenv('MT5_PASSWORD')
    server = os.getenv('MT5_SERVER')
    
    if not mt5.login(login, password, server):
        logger.error("Failed to login to MT5")
        return False
        
    return True

def download_market_data(symbol: str, timeframe: str, bars: int = 10000):
    """Download market data from MT5"""
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }
    
    tf = timeframe_map.get(timeframe)
    if tf is None:
        logger.error(f"Invalid timeframe: {timeframe}")
        return None
        
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None:
        logger.error(f"Failed to get rates for {symbol}")
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    import ta
    
    # Trend indicators
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    
    # Momentum indicators
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # Volatility indicators
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    
    # Fill NaN values
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def main():
    """Main function to download and save training data"""
    if not initialize_mt5():
        return
        
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Define symbols and timeframes to download
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD']
    timeframes = ['M5', 'M15', 'H1']
    
    # Download data for each symbol and timeframe
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"Downloading {symbol} {timeframe} data...")
            
            # Download market data
            df = download_market_data(symbol, timeframe)
            if df is None:
                continue
                
            # Add technical indicators
            df = add_technical_indicators(df)
            
            # Save to CSV
            filename = f'data/ml_training_data_{symbol}_{timeframe}.csv'
            df.to_csv(filename, index=True)
            logger.info(f"Saved {filename}")
            
    mt5.shutdown()
    logger.info("Data download complete")

if __name__ == '__main__':
    main() 