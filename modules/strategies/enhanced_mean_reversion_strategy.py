import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from ..indicators.advanced_indicators import AdvancedIndicators
from ..market.regime_detector import MarketRegimeDetector

class EnhancedMeanReversionStrategy:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = AdvancedIndicators()
        self.regime_detector = MarketRegimeDetector(config)
        
        # Strategy parameters
        self.params = {
            'bb_window': config.get('bb_window', 20),
            'bb_std': config.get('bb_std', 2.0),
            'rsi_period': config.get('rsi_period', 14),
            'volume_window': config.get('volume_window', 20),
            'partial_tp_levels': config.get('partial_tp_levels', [0.3, 0.5, 0.7]),
            'position_scale': config.get('position_scale', [0.4, 0.3, 0.3])
        }

    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """
        Enhanced mean reversion analysis with dynamic bands and volume confirmation
        """
        try:
            # Get market regime
            regime = self.regime_detector.detect_regime(data)
            
            # Skip if market is strongly trending
            if regime['market_state'] in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
                return {'signal': 0, 'strength': 0, 'regime': regime}
            
            # Calculate dynamic Bollinger Bands
            upper, middle, lower = self.indicators.calculate_dynamic_bollinger(
                data['close'],
                window=self.params['bb_window'],
                num_std=self.params['bb_std']
            )
            
            # Calculate volume indicators
            vwap = self.indicators.calculate_vwap(data['close'], data['volume'])
            obv = self.indicators.calculate_obv(data['close'], data['volume'])
            
            # Calculate mean reversion signals
            signal = self._calculate_reversion_signal(
                data, upper, middle, lower, vwap, obv
            )
            
            # Calculate entry levels for scaling in
            entry_levels = self._calculate_entry_levels(
                data['close'].iloc[-1], upper.iloc[-1], lower.iloc[-1]
            )
            
            # Calculate exit levels for partial profit taking
            exit_levels = self._calculate_exit_levels(
                data['close'].iloc[-1], middle.iloc[-1], signal['signal']
            )
            
            return {
                'signal': signal['signal'],
                'strength': signal['strength'],
                'regime': regime,
                'entry_levels': entry_levels,
                'exit_levels': exit_levels,
                'volume_confirmation': signal['volume_confirmation']
            }
            
        except Exception as e:
            self.logger.error(f"Error in mean reversion analysis: {str(e)}")
            return None

    def _calculate_reversion_signal(self, data: pd.DataFrame, 
                                  upper: pd.Series, middle: pd.Series, 
                                  lower: pd.Series, vwap: pd.Series,
                                  obv: pd.Series) -> Dict:
        """
        Calculate mean reversion signal with volume confirmation
        """
        current_price = data['close'].iloc[-1]
        
        # Basic mean reversion signal
        if current_price > upper.iloc[-1]:
            signal = -1  # Sell signal
        elif current_price < lower.iloc[-1]:
            signal = 1   # Buy signal
        else:
            signal = 0
            
        # Volume confirmation
        volume_trend = obv.diff(self.params['volume_window']).iloc[-1]
        price_to_vwap = current_price / vwap.iloc[-1] - 1
        
        volume_confirmation = (
            (signal == 1 and volume_trend > 0 and price_to_vwap < -0.001) or
            (signal == -1 and volume_trend < 0 and price_to_vwap > 0.001)
        )
        
        # Calculate signal strength
        if signal != 0:
            deviation = abs(current_price - middle.iloc[-1]) / middle.iloc[-1]
            strength = min(deviation * 100, 1.0)  # Cap at 1.0
            if not volume_confirmation:
                strength *= 0.7  # Reduce strength without volume confirmation
        else:
            strength = 0
            
        return {
            'signal': signal,
            'strength': strength,
            'volume_confirmation': volume_confirmation
        }

    def _calculate_entry_levels(self, current_price: float, 
                              upper: float, lower: float) -> List[Dict]:
        """
        Calculate scaled entry levels for position building
        """
        if current_price > upper:  # Short entries
            range_size = current_price - upper
            levels = [
                {
                    'price': current_price - (range_size * scale),
                    'size': pos_scale
                }
                for scale, pos_scale in zip(
                    [0.2, 0.5, 0.8],
                    self.params['position_scale']
                )
            ]
        elif current_price < lower:  # Long entries
            range_size = lower - current_price
            levels = [
                {
                    'price': current_price + (range_size * scale),
                    'size': pos_scale
                }
                for scale, pos_scale in zip(
                    [0.2, 0.5, 0.8],
                    self.params['position_scale']
                )
            ]
        else:
            levels = []
            
        return levels

    def _calculate_exit_levels(self, current_price: float, 
                             middle: float, signal: int) -> List[Dict]:
        """
        Calculate partial profit taking levels
        """
        if signal == 0:
            return []
            
        price_to_mean = middle - current_price
        levels = [
            {
                'price': current_price + (price_to_mean * tp_level * signal),
                'size': pos_scale
            }
            for tp_level, pos_scale in zip(
                self.params['partial_tp_levels'],
                self.params['position_scale']
            )
        ]
        
        return levels

    def get_position_size(self, signal: Dict, account_size: float) -> float:
        """
        Calculate position size based on signal strength and market regime
        """
        base_size = account_size * 0.02  # 2% base risk
        
        # Adjust size based on signal strength and volume confirmation
        size = base_size * signal['strength']
        if not signal['volume_confirmation']:
            size *= 0.7
            
        return min(size, account_size * 0.05)  # Cap at 5% of account

    def get_stop_loss(self, signal: Dict, current_price: float) -> float:
        """
        Calculate adaptive stop loss based on volatility
        """
        if signal['signal'] > 0:  # Long position
            return current_price * 0.98  # 2% stop loss
        else:  # Short position
            return current_price * 1.02  # 2% stop loss

    async def generate_signals(self, market_data: Dict) -> Dict:
        """
        Generate trading signals based on market data
        
        Args:
            market_data (Dict): Dictionary containing:
                - symbol: Trading symbol
                - data: DataFrame with OHLCV data
                - timestamp: Current timestamp
                
        Returns:
            Dict: Signal information containing:
                - action: Trading action (BUY, SELL, NONE)
                - confidence: Signal confidence level
                - metadata: Additional signal information
        """
        try:
            # Analyze market conditions
            analysis = self.analyze_market(market_data['data'])
            if analysis is None:
                return {
                    'action': 'NONE',
                    'confidence': 0.0,
                    'metadata': {'error': 'Analysis failed'}
                }
                
            # Convert signal to action
            if analysis['signal'] > 0:
                action = 'BUY'
            elif analysis['signal'] < 0:
                action = 'SELL'
            else:
                action = 'NONE'
                
            # Calculate entry parameters if signal exists
            metadata = {
                'regime': analysis['regime'],
                'volume_confirmation': analysis['volume_confirmation'],
                'entry_levels': analysis['entry_levels'],
                'exit_levels': analysis['exit_levels']
            }
            
            if action != 'NONE':
                current_price = market_data['data']['close'].iloc[-1]
                position_size = self.get_position_size(
                    analysis,
                    float(market_data.get('account_balance', 10000))  # Default if not provided
                )
                
                metadata.update({
                    'entry_price': current_price,
                    'position_size': position_size,
                    'stop_loss': self.get_stop_loss(analysis, current_price)
                })
                
                # Add scaled entry and exit information
                if analysis['entry_levels']:
                    metadata['scaled_entries'] = [
                        {
                            'price': level['price'],
                            'size': level['size'] * position_size
                        }
                        for level in analysis['entry_levels']
                    ]
                    
                if analysis['exit_levels']:
                    metadata['scaled_exits'] = [
                        {
                            'price': level['price'],
                            'size': level['size'] * position_size
                        }
                        for level in analysis['exit_levels']
                    ]
            
            return {
                'action': action,
                'confidence': analysis['strength'],
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error generating mean reversion signals: {str(e)}")
            return {
                'action': 'NONE',
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
