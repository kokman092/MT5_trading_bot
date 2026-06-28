from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
import logging

class StrategyEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.market_params = config['MARKET_PARAMS'][config['ACTIVE_MARKET']]
        
    def analyze_market(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Analyze market data and generate trading signals
        Returns: Dictionary with signal details or None if no signal
        """
        if data is None or len(data) < 50:
            return None
            
        try:
            signal = self._generate_signal(data)
            if signal:
                signal = self._validate_signal(signal, data)
                if signal:
                    signal = self._add_risk_levels(signal, data)
            return signal
        except Exception as e:
            logging.error(f"Error in market analysis: {str(e)}")
            return None
            
    def _generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Generate trading signal based on technical indicators"""
        current_price = data['close'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        sma20 = data['SMA20'].iloc[-1]
        ema20 = data['EMA20'].iloc[-1]
        bb_upper = data['BB_upper'].iloc[-1]
        bb_lower = data['BB_lower'].iloc[-1]
        
        # Trend determination
        short_term_trend = 1 if ema20 > sma20 else -1
        
        signal = None
        
        # Oversold condition with bullish trend
        if rsi < self.config['RSI_OVERSOLD'] and short_term_trend > 0:
            signal = {
                'action': 'BUY',
                'price': current_price,
                'strength': min(100, (self.config['RSI_OVERSOLD'] - rsi) * 2)
            }
            
        # Overbought condition with bearish trend
        elif rsi > self.config['RSI_OVERBOUGHT'] and short_term_trend < 0:
            signal = {
                'action': 'SELL',
                'price': current_price,
                'strength': min(100, (rsi - self.config['RSI_OVERBOUGHT']) * 2)
            }
            
        return signal
        
    def _validate_signal(self, signal: Dict, data: pd.DataFrame) -> Optional[Dict]:
        """Validate trading signal against additional criteria"""
        if signal is None:
            return None
            
        # Check spread
        current_spread = data['ask'].iloc[-1] - data['bid'].iloc[-1]
        if current_spread > self.market_params['min_spread']:
            logging.info(f"Signal rejected: spread too high ({current_spread})")
            return None
            
        # Check volume
        recent_volume = data['tick_volume'].iloc[-5:].mean()
        if recent_volume < self.market_params['min_volume']:
            logging.info(f"Signal rejected: volume too low ({recent_volume})")
            return None
            
        # Validate price movement
        price_change = abs(data['close'].iloc[-1] - data['close'].iloc[-2])
        if price_change > (data['close'].iloc[-2] * 0.02):  # 2% price jump
            logging.info(f"Signal rejected: excessive price movement ({price_change})")
            return None
            
        return signal
        
    def _add_risk_levels(self, signal: Dict, data: pd.DataFrame) -> Dict:
        """Calculate stop loss and take profit levels"""
        atr = self._calculate_atr(data)
        
        if signal['action'] == 'BUY':
            stop_loss = signal['price'] - (atr * 2)
            take_profit = signal['price'] + (atr * self.config['TAKE_PROFIT_RATIO'] * 2)
        else:
            stop_loss = signal['price'] + (atr * 2)
            take_profit = signal['price'] - (atr * self.config['TAKE_PROFIT_RATIO'] * 2)
            
        signal.update({
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr
        })
        
        return signal
        
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
