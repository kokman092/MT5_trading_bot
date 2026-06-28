import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import statsmodels.api as sm
from scipy import stats
from ..indicators.advanced_indicators import AdvancedIndicators
from ..market.regime_detector import MarketRegimeDetector

class EnhancedPairsTradingStrategy:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = AdvancedIndicators()
        self.regime_detector = MarketRegimeDetector(config)
        
        # Strategy parameters
        self.params = {
            'correlation_window': config.get('correlation_window', 60),
            'zscore_entry': config.get('zscore_entry', 2.0),
            'zscore_exit': config.get('zscore_exit', 0.5),
            'min_correlation': config.get('min_correlation', 0.8),
            'min_cointegration_pvalue': config.get('min_cointegration_pvalue', 0.05),
            'volatility_lookback': config.get('volatility_lookback', 20),
            'news_impact_threshold': config.get('news_impact_threshold', 0.5),
            'hedge_ratio_update_freq': config.get('hedge_ratio_update_freq', 5)
        }
        
        self.pair_metrics = {}
        self.news_events = {}

    def analyze_pairs(self, pairs_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Enhanced pairs analysis with dynamic correlation and cointegration tests
        """
        try:
            results = {}
            
            for pair_name, data in pairs_data.items():
                # Skip if high-impact news is expected
                if self._check_news_impact(pair_name):
                    continue
                
                # Calculate pair metrics
                metrics = self._calculate_pair_metrics(data)
                
                # Skip if pair doesn't meet criteria
                if not self._validate_pair_metrics(metrics):
                    continue
                
                # Calculate trading signals
                signals = self._calculate_trading_signals(data, metrics)
                
                if signals['signal'] != 0:
                    results[pair_name] = {
                        'signal': signals['signal'],
                        'strength': signals['strength'],
                        'hedge_ratio': metrics['hedge_ratio'],
                        'zscore': signals['zscore'],
                        'correlation': metrics['correlation'],
                        'volatility_ratio': metrics['volatility_ratio']
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in pairs analysis: {str(e)}")
            return None

    def _calculate_pair_metrics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate advanced pair metrics including dynamic correlation and hedge ratios
        """
        # Rolling correlation
        correlation = data['asset1'].rolling(
            window=self.params['correlation_window']
        ).corr(data['asset2']).iloc[-1]
        
        # Cointegration test
        coint_result = sm.tsa.stattools.coint(data['asset1'], data['asset2'])
        
        # Calculate dynamic hedge ratio using Kalman Filter
        hedge_ratio = self._calculate_dynamic_hedge_ratio(
            data['asset1'], 
            data['asset2']
        )
        
        # Calculate volatility ratio
        vol1 = data['asset1'].pct_change().std()
        vol2 = data['asset2'].pct_change().std()
        volatility_ratio = vol1 / vol2
        
        return {
            'correlation': correlation,
            'cointegration_pvalue': coint_result[1],
            'hedge_ratio': hedge_ratio,
            'volatility_ratio': volatility_ratio
        }

    def _calculate_dynamic_hedge_ratio(self, asset1: pd.Series, 
                                     asset2: pd.Series) -> float:
        """
        Calculate hedge ratio using Kalman Filter
        """
        # State space model for Kalman Filter
        delta = 1e-4
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_mat = np.vstack(
            [asset2, np.ones_like(asset2)]
        ).T[:, np.newaxis]
        
        kf = sm.regression.linear_model.KalmanFilter(
            k_states=2,
            k_endog=1,
            initialization='stationary'
        )
        
        # Run Kalman Filter
        state_means, _ = kf.filter(asset1)
        return state_means[-1, 0]

    def _validate_pair_metrics(self, metrics: Dict) -> bool:
        """
        Validate if pair meets trading criteria
        """
        return (
            abs(metrics['correlation']) >= self.params['min_correlation'] and
            metrics['cointegration_pvalue'] <= self.params['min_cointegration_pvalue'] and
            0.5 <= metrics['volatility_ratio'] <= 2.0
        )

    def _calculate_trading_signals(self, data: pd.DataFrame, 
                                 metrics: Dict) -> Dict:
        """
        Calculate trading signals with dynamic thresholds
        """
        # Calculate spread
        spread = data['asset1'] - metrics['hedge_ratio'] * data['asset2']
        
        # Calculate dynamic z-score thresholds based on volatility
        vol_adj = self.indicators.calculate_atr(spread).iloc[-1] / \
                 spread.std()
        entry_threshold = self.params['zscore_entry'] * vol_adj
        exit_threshold = self.params['zscore_exit'] * vol_adj
        
        # Calculate z-score
        zscore = (spread - spread.mean()) / spread.std()
        current_zscore = zscore.iloc[-1]
        
        # Generate signals
        if current_zscore > entry_threshold:
            signal = -1  # Short spread
        elif current_zscore < -entry_threshold:
            signal = 1   # Long spread
        elif abs(current_zscore) < exit_threshold:
            signal = 0   # Exit position
        else:
            signal = 0   # No action
            
        return {
            'signal': signal,
            'strength': min(abs(current_zscore) / entry_threshold, 1.0),
            'zscore': current_zscore
        }

    def _check_news_impact(self, pair_name: str) -> bool:
        """
        Check for high-impact news events
        """
        if pair_name in self.news_events:
            impact = self.news_events[pair_name].get('impact', 0)
            return impact > self.params['news_impact_threshold']
        return False

    def calculate_position_sizes(self, signal: Dict, 
                               account_size: float) -> Tuple[float, float]:
        """
        Calculate position sizes for both assets
        """
        base_size = account_size * 0.02  # 2% base risk
        
        # Adjust size based on signal strength
        adjusted_size = base_size * signal['strength']
        
        # Calculate individual position sizes
        asset1_size = adjusted_size
        asset2_size = adjusted_size * signal['hedge_ratio']
        
        return asset1_size, asset2_size

    def get_stop_loss(self, signal: Dict, spread: float) -> float:
        """
        Calculate stop loss based on spread volatility
        """
        spread_std = spread.std()
        return spread + (signal['signal'] * spread_std * 3)  # 3 standard deviations

    def update_pair_metrics(self, pair_name: str, metrics: Dict):
        """
        Update stored pair metrics
        """
        self.pair_metrics[pair_name] = {
            'last_update': pd.Timestamp.now(),
            'metrics': metrics
        }

    async def generate_signals(self, market_data: Dict) -> Dict:
        """
        Generate trading signals based on market data
        
        Args:
            market_data (Dict): Dictionary containing:
                - symbol: Trading symbol
                - data: DataFrame with OHLCV data for both assets in the pair
                - timestamp: Current timestamp
                
        Returns:
            Dict: Signal information containing:
                - action: Trading action (BUY, SELL, NONE)
                - confidence: Signal confidence level
                - metadata: Additional signal information
        """
        try:
            # Prepare pairs data
            pairs_data = {
                market_data['symbol']: market_data['data']
            }
            
            # Analyze pairs
            analysis = self.analyze_pairs(pairs_data)
            if analysis is None or not analysis:
                return {
                    'action': 'NONE',
                    'confidence': 0.0,
                    'metadata': {'error': 'Analysis failed or no valid pairs found'}
                }
                
            # Get analysis for current pair
            pair_analysis = analysis.get(market_data['symbol'])
            if pair_analysis is None:
                return {
                    'action': 'NONE',
                    'confidence': 0.0,
                    'metadata': {'error': 'No signals for current pair'}
                }
                
            # Convert signal to action
            if pair_analysis['signal'] > 0:
                action = 'BUY'
            elif pair_analysis['signal'] < 0:
                action = 'SELL'
            else:
                action = 'NONE'
                
            # Calculate entry parameters if signal exists
            metadata = {
                'hedge_ratio': pair_analysis['hedge_ratio'],
                'zscore': pair_analysis['zscore'],
                'correlation': pair_analysis['correlation'],
                'volatility_ratio': pair_analysis['volatility_ratio']
            }
            
            if action != 'NONE':
                # Calculate position sizes and stop loss
                asset1_size, asset2_size = self.calculate_position_sizes(
                    pair_analysis,
                    float(market_data.get('account_balance', 10000))  # Default if not provided
                )
                
                spread = market_data['data']['asset1'] - \
                        pair_analysis['hedge_ratio'] * market_data['data']['asset2']
                stop_loss = self.get_stop_loss(pair_analysis, spread)
                
                metadata.update({
                    'asset1_size': asset1_size,
                    'asset2_size': asset2_size,
                    'stop_loss': stop_loss
                })
            
            # Update pair metrics
            self.update_pair_metrics(market_data['symbol'], metadata)
            
            return {
                'action': action,
                'confidence': pair_analysis['strength'],
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error generating pairs trading signals: {str(e)}")
            return {
                'action': 'NONE',
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
