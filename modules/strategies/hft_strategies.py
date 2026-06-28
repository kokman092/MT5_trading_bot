import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from collections import deque

class HFTStrategies:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def generate_signal(self, df):
        """Generate HFT signals from market data"""
        try:
            # Ensure we have the required columns
            required_columns = ['close', 'high', 'low', 'tick_volume']
            if not all(col in df.columns for col in required_columns):
                return None
                
            # Calculate tick patterns
            tick_patterns = self._analyze_tick_patterns(df)
            if not tick_patterns:
                return None
                
            # Analyze trade sizes
            trade_sizes = self._analyze_trade_sizes(df)
            if not trade_sizes:
                return None
                
            # Combine signals
            signal = self._combine_signals(tick_patterns, trade_sizes)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating HFT signals: {str(e)}")
            return None
            
    def _analyze_tick_patterns(self, df):
        """Analyze tick patterns for HFT signals"""
        try:
            # Calculate price changes
            df['price_change'] = df['close'].diff()
            df['direction'] = df['price_change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            
            # Look for patterns in last few ticks
            last_directions = df['direction'].tail(5).values
            
            # Count consecutive moves
            consecutive_up = 0
            consecutive_down = 0
            for d in reversed(last_directions):
                if d > 0:
                    consecutive_up += 1
                    consecutive_down = 0
                elif d < 0:
                    consecutive_down += 1
                    consecutive_up = 0
                else:
                    break
                    
            # Generate signal based on patterns
            if consecutive_up >= 3:
                return {'direction': -1, 'confidence': min(consecutive_up / 5, 1.0)}
            elif consecutive_down >= 3:
                return {'direction': 1, 'confidence': min(consecutive_down / 5, 1.0)}
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing tick patterns: {str(e)}")
            return None
            
    def _analyze_trade_sizes(self, df):
        """Analyze trade sizes for HFT signals"""
        try:
            # Calculate relative trade sizes
            df['rel_volume'] = df['tick_volume'] / df['tick_volume'].rolling(window=20).mean()
            
            # Look for unusual volume
            last_volumes = df['rel_volume'].tail(5).values
            avg_volume = last_volumes.mean()
            
            if avg_volume > 2.0:  # Volume spike
                price_direction = 1 if df['close'].iloc[-1] > df['close'].iloc[-2] else -1
                return {
                    'direction': price_direction,
                    'confidence': min(avg_volume / 4, 1.0)
                }
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing trade sizes: {str(e)}")
            return None
            
    def _combine_signals(self, tick_patterns, trade_sizes):
        """Combine different HFT signals"""
        try:
            if not tick_patterns and not trade_sizes:
                return None
                
            # Initialize combined signal
            direction = 0
            confidence = 0.0
            
            # Combine tick pattern signals
            if tick_patterns:
                direction += tick_patterns['direction']
                confidence = max(confidence, tick_patterns['confidence'])
                
            # Combine trade size signals
            if trade_sizes:
                direction += trade_sizes['direction']
                confidence = max(confidence, trade_sizes['confidence'])
                
            # Average direction and boost confidence if signals agree
            final_direction = 1 if direction > 0 else (-1 if direction < 0 else 0)
            if tick_patterns and trade_sizes and tick_patterns['direction'] == trade_sizes['direction']:
                confidence *= 1.2  # 20% confidence boost for agreeing signals
                
            return {
                'direction': final_direction,
                'confidence': min(confidence, 1.0),
                'type': 'hft'
            }
            
        except Exception as e:
            self.logger.error(f"Error combining HFT signals: {str(e)}")
            return None
