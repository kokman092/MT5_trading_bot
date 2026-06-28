import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union
import json
import os

def initialize_mt5() -> bool:
    """Initialize MT5 connection"""
    try:
        if not mt5.initialize():
            logging.error("Failed to initialize MT5")
            return False
        return True
    except Exception as e:
        logging.error(f"Error initializing MT5: {str(e)}")
        return False

def get_account_info() -> Optional[Dict]:
    """Get current account information"""
    try:
        if not initialize_mt5():
            return None
            
        account_info = mt5.account_info()
        if account_info is None:
            return None
            
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'margin_level': account_info.margin_level,
            'leverage': account_info.leverage,
            'currency': account_info.currency,
            'profit': account_info.profit,
            'name': account_info.name,
            'server': account_info.server,
            'trade_mode': account_info.trade_mode
        }
    except Exception as e:
        logging.error(f"Error getting account info: {str(e)}")
        return None

def get_open_positions() -> List[Dict]:
    """Get current open positions"""
    try:
        if not initialize_mt5():
            return []
            
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        positions_list = []
        for position in positions:
            positions_list.append({
                'ticket': position.ticket,
                'symbol': position.symbol,
                'type': 'BUY' if position.type == 0 else 'SELL',
                'volume': position.volume,
                'entry_price': position.price_open,
                'current_price': position.price_current,
                'sl': position.sl,
                'tp': position.tp,
                'profit': position.profit,
                'swap': position.swap,
                'commission': position.commission,
                'magic': position.magic,
                'comment': position.comment,
                'time': datetime.fromtimestamp(position.time).isoformat()
            })
        return positions_list
    except Exception as e:
        logging.error(f"Error getting open positions: {str(e)}")
        return []

def calculate_daily_pl() -> float:
    """Calculate daily profit/loss"""
    try:
        if not initialize_mt5():
            return 0.0
            
        today = datetime.now().date()
        deals = mt5.history_deals_get(
            datetime.combine(today, datetime.min.time()),
            datetime.now()
        )
        
        if deals is None:
            return 0.0
        
        total_profit = sum(deal.profit + deal.swap + deal.commission for deal in deals)
        return total_profit
    except Exception as e:
        logging.error(f"Error calculating daily P/L: {str(e)}")
        return 0.0

def get_market_data(symbol: str, timeframe: str, bars: int = 1000) -> Optional[pd.DataFrame]:
    """Get market data for analysis"""
    try:
        if not initialize_mt5():
            return None
            
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        mt5_timeframe = timeframe_map.get(timeframe)
        if mt5_timeframe is None:
            logging.error(f"Invalid timeframe: {timeframe}")
            return None
            
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
        if rates is None:
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        logging.error(f"Error getting market data: {str(e)}")
        return None

def analyze_market_regime(market_data: pd.DataFrame) -> Dict:
    """Analyze current market regime"""
    try:
        if market_data is None or len(market_data) < 100:
            return {'error': 'Insufficient data for analysis'}
            
        # Calculate volatility
        returns = np.log(market_data['close'] / market_data['close'].shift(1))
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate trend strength
        sma_20 = market_data['close'].rolling(window=20).mean()
        sma_50 = market_data['close'].rolling(window=50).mean()
        trend_strength = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] * 100
        
        # Determine market regime
        if volatility > 0.25:  # High volatility threshold
            if trend_strength > 1:
                regime = 'TRENDING_VOLATILE_BULLISH'
            elif trend_strength < -1:
                regime = 'TRENDING_VOLATILE_BEARISH'
            else:
                regime = 'VOLATILE_RANGING'
        else:
            if trend_strength > 1:
                regime = 'TRENDING_STABLE_BULLISH'
            elif trend_strength < -1:
                regime = 'TRENDING_STABLE_BEARISH'
            else:
                regime = 'STABLE_RANGING'
                
        return {
            'regime': regime,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Error analyzing market regime: {str(e)}")
        return {'error': str(e)}

def get_model_metrics(symbol: str, timeframe: str) -> Dict:
    """Get ML model performance metrics"""
    try:
        # Load model metrics from file
        metrics_file = f'models/metrics/{symbol}_{timeframe}_metrics.json'
        if not os.path.exists(metrics_file):
            return {'error': 'No metrics available'}
            
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        return {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Error getting model metrics: {str(e)}")
        return {'error': str(e)}

def get_market_predictions(symbol: str, timeframe: str) -> Dict:
    """Get current market predictions"""
    try:
        # Get market data
        market_data = get_market_data(symbol, timeframe)
        if market_data is None:
            return {'error': 'Failed to get market data'}
            
        # Load prediction model
        model_file = f'models/{symbol}_{timeframe}_model.pkl'
        if not os.path.exists(model_file):
            return {'error': 'Model not available'}
            
        # Make predictions
        # Note: This is a placeholder. Implement actual model loading and prediction logic
        predictions = {
            'direction': 'UP',
            'probability': 0.65,
            'target_price': market_data['close'].iloc[-1] * 1.01,
            'stop_loss': market_data['close'].iloc[-1] * 0.99
        }
        
        return {
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Error getting predictions: {str(e)}")
        return {'error': str(e)}

def validate_trading_conditions() -> bool:
    """Validate current trading conditions"""
    try:
        if not initialize_mt5():
            return False
            
        # Get account info
        account_info = get_account_info()
        if account_info is None:
            return False
            
        # Check margin level (minimum 150%)
        if account_info['margin_level'] < 150:
            logging.warning(f"Insufficient margin level: {account_info['margin_level']}%")
            return False
            
        # Check maximum positions (limit to 10)
        positions = get_open_positions()
        if len(positions) >= 10:
            logging.warning("Maximum positions limit reached")
            return False
            
        # Check market hours for major pairs
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        for symbol in symbols:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info or not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                logging.warning(f"Market closed for {symbol}")
                return False
                
        return True
    except Exception as e:
        logging.error(f"Error validating trading conditions: {str(e)}")
        return False 