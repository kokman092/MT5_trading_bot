import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from ..indicators.advanced_indicators import AdvancedIndicators
from ..market.regime_detector import MarketRegimeDetector

class EnhancedMomentumStrategy:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = AdvancedIndicators()
        self.regime_detector = MarketRegimeDetector(config)
        
        # Strategy parameters
        self.params = {
            'kama_fast': config.get('kama_fast', 2),
            'kama_slow': config.get('kama_slow', 30),
            'kama_period': config.get('kama_period', 10),
            'volatility_window': config.get('volatility_window', 20),
            'entry_threshold': config.get('entry_threshold', 0.01),
            'exit_threshold': config.get('exit_threshold', 0.005),
            'stop_loss': config.get('stop_loss', 0.02),
            'timeframes': config.get('timeframes', ['H1', 'H4', 'D1'])
        }

    def analyze_market(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Enhanced market analysis using adaptive indicators and multi-timeframe analysis
        """
        try:
            signals = {}
            
            # Get market regime
            regime = self.regime_detector.detect_regime(data['H1'])
            
            # Skip if market conditions aren't suitable
            if regime['market_state'] not in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
                return {'signal': 0, 'strength': 0, 'regime': regime}
            
            # Calculate adaptive indicators for each timeframe
            timeframe_signals = []
            for timeframe in self.params['timeframes']:
                tf_data = data[timeframe]
                
                # Calculate KAMA for this timeframe
                kama = self.indicators.calculate_kama(
                    tf_data['close'],
                    n=self.params['kama_period'],
                    fast_ema=self.params['kama_fast'],
                    slow_ema=self.params['kama_slow']
                )
                
                # Calculate dynamic lookback period based on volatility
                atr = self.indicators.calculate_atr(tf_data['close'])
                volatility_factor = atr.iloc[-1] / atr.rolling(window=20).mean().iloc[-1]
                dynamic_lookback = int(self.params['kama_period'] * volatility_factor)
                
                # Calculate momentum signals
                momentum = self._calculate_momentum_signal(tf_data, kama, dynamic_lookback)
                timeframe_signals.append(momentum)
            
            # Combine signals from different timeframes with higher weight to higher timeframes
            weights = [0.3, 0.3, 0.4]  # H1, H4, D1 weights
            combined_signal = sum(s * w for s, w in zip(timeframe_signals, weights))
            
            # Calculate signal strength
            strength = abs(combined_signal)
            
            # Apply regime-based adjustments
            adjustments = self.regime_detector.should_adjust_parameters(regime)
            
            return {
                'signal': np.sign(combined_signal),
                'strength': strength,
                'regime': regime,
                'adjustments': adjustments,
                'timeframe_signals': dict(zip(self.params['timeframes'], timeframe_signals))
            }
            
        except Exception as e:
            self.logger.error(f"Error in momentum analysis: {str(e)}")
            return None

    def _calculate_momentum_signal(self, data: pd.DataFrame, kama: pd.Series, 
                                 lookback: int) -> float:
        """
        Calculate momentum signal with dynamic parameters
        """
        # Price momentum
        price_momentum = (data['close'].iloc[-1] - data['close'].iloc[-lookback]) / data['close'].iloc[-lookback]
        
        # KAMA momentum
        kama_momentum = (kama.iloc[-1] - kama.iloc[-lookback]) / kama.iloc[-lookback]
        
        # Volume-weighted momentum
        volume_momentum = self.indicators.calculate_obv(data['close'], data['volume']).diff(lookback).iloc[-1]
        volume_momentum = np.sign(volume_momentum) * min(abs(volume_momentum), 1.0)
        
        # Combine signals
        combined_momentum = (price_momentum + kama_momentum + volume_momentum) / 3
        
        return combined_momentum

    def get_position_size(self, signal: Dict, account_size: float) -> float:
        """
        Calculate position size based on signal strength and market regime
        """
        base_size = account_size * 0.02  # 2% base risk
        
        # Adjust size based on signal strength
        size = base_size * signal['strength']
        
        # Apply regime-based adjustments
        size *= signal['adjustments']['position_size_multiplier']
        
        return min(size, account_size * 0.05)  # Cap at 5% of account

    def get_stop_loss(self, signal: Dict, current_price: float) -> float:
        """
        Calculate adaptive stop loss based on volatility and regime
        """
        base_stop = self.params['stop_loss']
        
        # Adjust stop loss based on regime
        adjusted_stop = base_stop * signal['adjustments']['stop_loss_multiplier']
        
        return current_price * (1 - adjusted_stop)

    async def generate_signals(self, market_data: Dict) -> Dict:
        """
        Generate trading signals based on momentum analysis across multiple timeframes.
        
        Args:
            market_data: Dict containing:
                - data: Dict of DataFrames with OHLCV data for different timeframes
                - account_balance: Current account balance for position sizing
                
        Returns:
            Dict containing:
                - action: str, one of 'BUY', 'SELL', or 'NONE'
                - confidence: float between 0 and 1
                - metadata: Dict with entry parameters, stop loss, etc.
        """
        try:
            # Extract data and account balance
            data = market_data['data']
            account_size = float(market_data.get('account_balance', 10000))  # Default if not provided
            
            # Perform market analysis
            analysis = self.analyze_market(data)
            if analysis is None:
                return {'action': 'NONE', 'confidence': 0.0, 'metadata': {}}
            
            # Default response
            response = {
                'action': 'NONE',
                'confidence': 0.0,
                'metadata': {
                    'regime': analysis['regime'],
                    'timeframe_signals': analysis['timeframe_signals']
                }
            }
            
            # Get current price
            current_price = data['H1']['close'].iloc[-1]
            
            # Generate signal based on analysis
            if analysis['signal'] > 0 and analysis['strength'] > self.params['entry_threshold']:
                response['action'] = 'BUY'
                response['confidence'] = min(analysis['strength'], 1.0)
                
                # Calculate position parameters
                position_size = self.get_position_size(analysis, account_size)
                stop_loss = self.get_stop_loss(analysis, current_price)
                
                response['metadata'].update({
                    'entry_price': current_price,
                    'position_size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': current_price * (1 + self.params['entry_threshold'] * 2)
                })
                
            elif analysis['signal'] < 0 and analysis['strength'] > self.params['entry_threshold']:
                response['action'] = 'SELL'
                response['confidence'] = min(analysis['strength'], 1.0)
                
                # Calculate position parameters
                position_size = self.get_position_size(analysis, account_size)
                stop_loss = current_price * (1 + self.params['stop_loss'])
                
                response['metadata'].update({
                    'entry_price': current_price,
                    'position_size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': current_price * (1 - self.params['entry_threshold'] * 2)
                })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating momentum signals: {str(e)}")
            return {'action': 'NONE', 'confidence': 0.0, 'metadata': {'error': str(e)}}
