import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import MetaTrader5 as mt5
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from typing import Dict, List
import ta

class AdaptiveStrategyManager:
    def __init__(self, config, model_selector, risk_manager):
        """Initialize the adaptive strategy manager with configuration"""
        self.config = config
        self.model_selector = model_selector
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
        
        # Load strategy parameters from config
        strategy_params = self.config.get('trading_parameters', {}).get('STRATEGIES', {})
        
        # Strategy parameters with defaults
        self.param_ranges = {
            'volatile_bearish': {
                'stop_loss_mult': strategy_params.get('volatile_bearish', {}).get('stop_loss_mult', (2.0, 3.0)),
                'take_profit_mult': strategy_params.get('volatile_bearish', {}).get('take_profit_mult', (1.5, 2.0)),
                'entry_threshold': strategy_params.get('volatile_bearish', {}).get('entry_threshold', (0.7, 0.8)),
                'position_hold_time': strategy_params.get('volatile_bearish', {}).get('position_hold_time', (2, 4))
            },
            'volatile_bullish': {
                'stop_loss_mult': strategy_params.get('volatile_bullish', {}).get('stop_loss_mult', (1.5, 2.0)),
                'take_profit_mult': strategy_params.get('volatile_bullish', {}).get('take_profit_mult', (2.0, 3.0)),
                'entry_threshold': strategy_params.get('volatile_bullish', {}).get('entry_threshold', (0.6, 0.7)),
                'position_hold_time': strategy_params.get('volatile_bullish', {}).get('position_hold_time', (4, 8))
            },
            'trending_bearish': {
                'stop_loss_mult': strategy_params.get('trending_bearish', {}).get('stop_loss_mult', (1.8, 2.5)),
                'take_profit_mult': strategy_params.get('trending_bearish', {}).get('take_profit_mult', (1.8, 2.5)),
                'entry_threshold': strategy_params.get('trending_bearish', {}).get('entry_threshold', (0.65, 0.75)),
                'position_hold_time': strategy_params.get('trending_bearish', {}).get('position_hold_time', (6, 12))
            },
            'trending_bullish': {
                'stop_loss_mult': strategy_params.get('trending_bullish', {}).get('stop_loss_mult', (1.5, 2.0)),
                'take_profit_mult': strategy_params.get('trending_bullish', {}).get('take_profit_mult', (2.5, 3.5)),
                'entry_threshold': strategy_params.get('trending_bullish', {}).get('entry_threshold', (0.6, 0.7)),
                'position_hold_time': strategy_params.get('trending_bullish', {}).get('position_hold_time', (8, 16))
            },
            'ranging': {
                'stop_loss_mult': strategy_params.get('ranging', {}).get('stop_loss_mult', (1.2, 1.5)),
                'take_profit_mult': strategy_params.get('ranging', {}).get('take_profit_mult', (1.2, 1.5)),
                'entry_threshold': strategy_params.get('ranging', {}).get('entry_threshold', (0.75, 0.85)),
                'position_hold_time': strategy_params.get('ranging', {}).get('position_hold_time', (1, 2))
            }
        }
        
        # Performance tracking
        self.strategy_performance = {}
        self.regime_history = []
        self.anomaly_detector = IsolationForest(
            contamination=self.config.get('ml_parameters', {}).get('anomaly_detection', {}).get('contamination', 0.1)
        )
        
        # Load history and initialize strategies
        self.load_history()
        self._init_strategies()
    
    def _init_strategies(self):
        """Initialize adaptive strategies with configuration parameters"""
        try:
            # Get strategy parameters from config
            strategy_params = self.config.get('trading_parameters', {}).get('STRATEGIES', {})
            
            # Initialize trend following strategy
            self.trend_following = {
                'ma_fast': strategy_params.get('sma_crossover', {}).get('fast_period', 10),
                'ma_slow': strategy_params.get('sma_crossover', {}).get('slow_period', 20),
                'atr_period': strategy_params.get('atr', {}).get('period', 14),
                'weight': strategy_params.get('sma_crossover', {}).get('weight', 0.4)
            }
            
            # Initialize mean reversion strategy
            self.mean_reversion = {
                'lookback': strategy_params.get('mean_reversion', {}).get('lookback', 20),
                'std_dev': strategy_params.get('mean_reversion', {}).get('std_dev', 2.0),
                'weight': strategy_params.get('mean_reversion', {}).get('weight', 0.3)
            }
            
            # Initialize momentum strategy
            self.momentum = {
                'rsi_period': strategy_params.get('rsi', {}).get('period', 14),
                'macd_fast': strategy_params.get('macd', {}).get('fast_period', 12),
                'macd_slow': strategy_params.get('macd', {}).get('slow_period', 26),
                'macd_signal': strategy_params.get('macd', {}).get('signal_period', 9),
                'weight': strategy_params.get('momentum', {}).get('weight', 0.3)
            }
            
            # Initialize HFT strategy
            hft_params = strategy_params.get('hft', {})
            self.hft_strategy = {
                'tick_window': hft_params.get('tick_window', 100),
                'volume_window': hft_params.get('volume_window', 20),
                'price_threshold': hft_params.get('price_threshold', 0.0001),
                'min_volume': hft_params.get('min_volume', 1000),
                'max_spread': hft_params.get('max_spread', 0.0001),
                'latency_threshold': hft_params.get('latency_threshold', 10),
                'position_timeout': hft_params.get('position_timeout', 30),
                'profit_target_ticks': hft_params.get('profit_target_ticks', 2),
                'stop_loss_ticks': hft_params.get('stop_loss_ticks', 1),
                'weight': hft_params.get('weight', 0.2)
            }
            
            self.logger.info("Adaptive strategies initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing adaptive strategies: {str(e)}")
            raise
    
    def detect_market_conditions(self, data: pd.DataFrame) -> Dict:
        """Detect current market conditions"""
        try:
            # Calculate technical indicators
            rsi = ta.momentum.RSIIndicator(data['close']).rsi()
            macd = ta.trend.MACD(data['close'])
            bb = ta.volatility.BollingerBands(data['close'])
            adx = ta.trend.ADXIndicator(data['high'], data['low'], data['close'])
            
            # Get latest values
            current_rsi = rsi.iloc[-1]
            current_macd = macd.macd().iloc[-1]
            current_macd_signal = macd.macd_signal().iloc[-1]
            current_adx = adx.adx().iloc[-1]
            current_bb_width = (bb.bollinger_hband() - bb.bollinger_lband()).iloc[-1] / bb.bollinger_mavg().iloc[-1]
            
            # Determine market regime
            regime = "UNKNOWN"
            confidence = 0.5
            
            if current_adx > 25:  # Strong trend
                if current_macd > current_macd_signal:
                    regime = "STRONG_TREND"
                    confidence = min(current_adx / 100 + (current_rsi / 100), 0.99)
                else:
                    regime = "WEAK_TREND"
                    confidence = min(current_adx / 100 + ((100 - current_rsi) / 100), 0.99)
            elif current_bb_width < 0.1:  # Low volatility
                regime = "LOW_VOLATILITY"
                confidence = min(1 - current_bb_width, 0.99)
            elif current_bb_width > 0.3:  # High volatility
                regime = "HIGH_VOLATILITY"
                confidence = min(current_bb_width, 0.99)
            else:  # Ranging market
                regime = "RANGING"
                confidence = min(1 - (current_adx / 50), 0.99)
            
            return {
                'regime': regime,
                'confidence': confidence,
                'indicators': {
                    'rsi': current_rsi,
                    'adx': current_adx,
                    'bb_width': current_bb_width,
                    'macd': current_macd
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting market conditions: {str(e)}")
            return {'regime': "UNKNOWN", 'confidence': 0.5}
    
    def adjust_strategy_parameters(self, symbol, regime, state):
        """Adjust strategy parameters based on market conditions"""
        try:
            # Get base parameters for regime
            params = {
                param: np.random.uniform(low, high)
                for param, (low, high) in self.param_ranges[regime].items()
            }
            
            # Adjust based on current state
            if state['volatility'] > 0.3:  # High volatility
                params['stop_loss_mult'] *= 1.2
                params['take_profit_mult'] *= 0.8
                params['position_hold_time'] *= 0.7
            
            if abs(state['momentum']) > 0.02:  # Strong momentum
                params['take_profit_mult'] *= 1.2
                params['entry_threshold'] *= 0.9
            
            if state['volume_ratio'] < 0.8:  # Low volume
                params['entry_threshold'] *= 1.2
                params['position_hold_time'] *= 0.8
            
            return params
            
        except Exception as e:
            self.logger.error(f"Error adjusting strategy parameters: {str(e)}")
            return None
    
    def evaluate_strategy_performance(self, symbol, timeframe=mt5.TIMEFRAME_H1):
        """Evaluate strategy performance and adapt if necessary"""
        try:
            if symbol not in self.strategy_performance:
                self.strategy_performance[symbol] = []
            
            # Get recent trades
            trades = mt5.history_deals_get(
                datetime.now() - timedelta(days=7),
                datetime.now()
            )
            
            if trades is None:
                return
            
            # Calculate performance metrics
            symbol_trades = [t for t in trades if t.symbol == symbol]
            if not symbol_trades:
                return
            
            performance = {
                'win_rate': sum(1 for t in symbol_trades if t.profit > 0) / len(symbol_trades),
                'avg_profit': np.mean([t.profit for t in symbol_trades]),
                'max_drawdown': self.calculate_max_drawdown(symbol_trades),
                'sharpe_ratio': self.calculate_sharpe_ratio(symbol_trades)
            }
            
            # Store performance
            self.strategy_performance[symbol].append({
                'timestamp': datetime.now().isoformat(),
                'metrics': performance
            })
            
            # Check if adaptation is needed
            if len(self.strategy_performance[symbol]) > 1:
                prev_performance = self.strategy_performance[symbol][-2]['metrics']
                if (performance['win_rate'] < prev_performance['win_rate'] * 0.8 or
                    performance['sharpe_ratio'] < prev_performance['sharpe_ratio'] * 0.8):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating strategy performance: {str(e)}")
            return False
    
    def calculate_max_drawdown(self, trades):
        """Calculate maximum drawdown from trade history"""
        try:
            equity_curve = [0]
            for trade in trades:
                equity_curve.append(equity_curve[-1] + trade.profit)
            
            equity_curve = np.array(equity_curve)
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak
            return abs(min(drawdown))
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0
    
    def calculate_sharpe_ratio(self, trades):
        """Calculate Sharpe ratio from trade history"""
        try:
            returns = [t.profit for t in trades]
            if not returns:
                return 0
            
            return np.mean(returns) / (np.std(returns) if np.std(returns) > 0 else 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0
    
    def get_optimal_entry_time(self, symbol, timeframe=mt5.TIMEFRAME_H1):
        """Determine optimal entry time based on historical patterns"""
        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1000)
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            df['hour'] = pd.to_datetime(df['time'], unit='s').dt.hour
            
            # Calculate hour performance
            hour_performance = {}
            for hour in range(24):
                hour_data = df[df['hour'] == hour]
                if len(hour_data) > 0:
                    hour_performance[hour] = {
                        'returns': hour_data['close'].pct_change().mean(),
                        'volatility': hour_data['close'].pct_change().std(),
                        'volume': hour_data['tick_volume'].mean()
                    }
            
            # Score each hour
            hour_scores = {}
            for hour, perf in hour_performance.items():
                score = (perf['returns'] / perf['volatility'] if perf['volatility'] > 0 else 0) * perf['volume']
                hour_scores[hour] = score
            
            # Return best hours
            best_hours = sorted(hour_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            return [hour for hour, _ in best_hours]
            
        except Exception as e:
            self.logger.error(f"Error finding optimal entry time: {str(e)}")
            return None
    
    def save_history(self):
        """Save strategy history"""
        try:
            history_data = {
                'regime_history': self.regime_history,
                'strategy_performance': self.strategy_performance
            }
            
            with open('strategy_history.json', 'w') as f:
                json.dump(history_data, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving history: {str(e)}")
    
    def load_history(self):
        """Load strategy history"""
        try:
            with open('strategy_history.json', 'r') as f:
                history_data = json.load(f)
                self.regime_history = history_data['regime_history']
                self.strategy_performance = history_data['strategy_performance']
                
        except FileNotFoundError:
            self.logger.info("No strategy history file found. Starting fresh.")
        except Exception as e:
            self.logger.error(f"Error loading history: {str(e)}")
    
    def get_strategy_insights(self):
        """Get insights about strategy performance"""
        try:
            if not self.strategy_performance:
                return None
            
            insights = {}
            for symbol, performance in self.strategy_performance.items():
                if not performance:
                    continue
                
                # Calculate performance trends
                metrics = pd.DataFrame([p['metrics'] for p in performance])
                
                insights[symbol] = {
                    'win_rate_trend': metrics['win_rate'].diff().mean(),
                    'profit_trend': metrics['avg_profit'].diff().mean(),
                    'risk_trend': metrics['max_drawdown'].diff().mean(),
                    'overall_trend': 'improving' if metrics['sharpe_ratio'].diff().mean() > 0 else 'degrading',
                    'best_regime': self.find_best_regime(symbol)
                }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting strategy insights: {str(e)}")
            return None
    
    def find_best_regime(self, symbol):
        """Find best performing regime for a symbol"""
        try:
            regime_performance = {}
            
            for entry in self.regime_history:
                if entry['symbol'] != symbol:
                    continue
                
                regime = entry['regime']
                if regime not in regime_performance:
                    regime_performance[regime] = []
                
                # Find trades during this regime
                timestamp = datetime.fromisoformat(entry['timestamp'])
                trades = mt5.history_deals_get(
                    timestamp,
                    timestamp + timedelta(hours=1),
                    symbol=symbol
                )
                
                if trades:
                    performance = sum(t.profit for t in trades)
                    regime_performance[regime].append(performance)
            
            # Calculate average performance per regime
            avg_performance = {
                regime: np.mean(perfs) if perfs else 0
                for regime, perfs in regime_performance.items()
            }
            
            return max(avg_performance.items(), key=lambda x: x[1])[0] if avg_performance else None
            
        except Exception as e:
            self.logger.error(f"Error finding best regime: {str(e)}")
            return None
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signals using adaptive strategy combination"""
        try:
            signals = {}
            
            # Get current market regime
            regime = self.detect_market_conditions(data)
            
            # Initialize strategy weights based on regime
            strategy_weights = {
                'hft': 0.3 if regime['regime'] in ['RANGING', 'LOW_VOLATILITY'] else 0.1,
                'event_driven': 0.3 if regime['regime'] in ['HIGH_VOLATILITY', 'NEWS_DRIVEN'] else 0.2,
                'trend': 0.4 if regime['regime'] in ['STRONG_TREND', 'WEAK_TREND'] else 0.2
            }
            
            # Normalize weights
            total_weight = sum(strategy_weights.values())
            strategy_weights = {k: v/total_weight for k, v in strategy_weights.items()}
            
            # Get signals from each strategy
            hft_signals = self.hft_strategy.generate_signals(data)
            event_signals = self.event_strategy.generate_signals(data)
            trend_signals = self.trend_strategy.generate_signals(data)
            
            # Combine signals using weighted average
            final_signal = (
                hft_signals.get('final_signal', 0) * strategy_weights['hft'] +
                event_signals.get('final_signal', 0) * strategy_weights['event_driven'] +
                trend_signals.get('final_signal', 0) * strategy_weights['trend']
            )
            
            # Calculate signal confidence based on agreement between strategies
            signals_list = [
                hft_signals.get('final_signal', 0),
                event_signals.get('final_signal', 0),
                trend_signals.get('final_signal', 0)
            ]
            
            # Calculate agreement score (how many strategies agree on direction)
            agreement = sum(1 for s in signals_list if np.sign(s) == np.sign(final_signal))
            confidence = agreement / len(signals_list)
            
            # Adjust signal strength based on regime confidence
            regime_confidence = regime['confidence']
            final_signal *= regime_confidence
            
            # Apply risk adjustments
            risk_score = self._get_risk_score()
            if risk_score > 0.8:  # High risk environment
                final_signal *= 0.5  # Reduce signal strength
            
            # Compile final signals dictionary
            signals['final_signal'] = np.clip(final_signal, -1, 1)
            signals['confidence'] = confidence
            signals['regime'] = regime['regime']
            signals['risk_score'] = risk_score
            signals['strategy_weights'] = strategy_weights
            
            # Log signal generation
            self.logger.info(f"Generated adaptive signals: {signals}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating adaptive signals: {str(e)}")
            return {'final_signal': 0.0, 'confidence': 0.0}  # Neutral signal as fallback
            
    def _get_risk_score(self) -> float:
        """Calculate current risk score based on market conditions and performance"""
        try:
            # Get recent performance metrics
            recent_trades = self.get_recent_trades()
            
            # Calculate components of risk score
            drawdown = self.calculate_max_drawdown(recent_trades)
            volatility = self.calculate_market_volatility()
            win_rate = self.calculate_win_rate(recent_trades)
            
            # Combine into final risk score
            risk_score = (
                0.4 * drawdown +      # 40% weight on drawdown
                0.3 * volatility +    # 30% weight on volatility
                0.3 * (1 - win_rate)  # 30% weight on inverse win rate
            )
            
            return np.clip(risk_score, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {str(e)}")
            return 0.5  # Moderate risk as fallback
            
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from recent trades"""
        if not trades:
            return 0.5
            
        winning_trades = sum(1 for trade in trades if trade['profit'] > 0)
        return winning_trades / len(trades)
        
    def calculate_market_volatility(self) -> float:
        """Calculate current market volatility"""
        try:
            # Get recent price data
            recent_prices = self.get_recent_prices()
            if recent_prices.empty:
                return 0.5
                
            # Calculate returns
            returns = recent_prices['close'].pct_change().dropna()
            
            # Calculate annualized volatility
            volatility = returns.std() * np.sqrt(252)
            
            # Normalize to 0-1 range
            normalized_vol = min(volatility / 0.02, 1)  # Cap at 2% daily volatility
            
            return normalized_vol
            
        except Exception as e:
            self.logger.error(f"Error calculating market volatility: {str(e)}")
            return 0.5
