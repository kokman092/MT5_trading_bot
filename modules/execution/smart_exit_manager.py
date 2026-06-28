import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import MetaTrader5 as mt5
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from collections import deque

class SmartExitManager:
    def __init__(self, config, execution_engine):
        self.config = config
        self.execution_engine = execution_engine
        self.logger = logging.getLogger(__name__)
        
        # Exit parameters
        self.profit_target_multiplier = config.get('PROFIT_TARGET_MULTIPLIER', 1.5)
        self.trailing_stop_activation = config.get('TRAILING_STOP_ACTIVATION', 0.8)  # % of target
        self.partial_exit_threshold = config.get('PARTIAL_EXIT_THRESHOLD', 0.5)  # % of target
        self.max_adverse_excursion = config.get('MAX_ADVERSE_EXCURSION', 0.02)
        
        # Market condition thresholds
        self.volatility_threshold = config.get('VOLATILITY_THRESHOLD', 0.002)
        self.volume_threshold = config.get('VOLUME_THRESHOLD', 1.5)
        self.spread_threshold = config.get('SPREAD_THRESHOLD', 0.0003)
        
        # Exit analytics
        self.exit_history = []
        self.market_conditions = {}
        self.position_analytics = {}
        
        # Anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.1)
        
        # Load history
        self.load_history()
    
    def analyze_position(self, position):
        """Analyze position and market conditions for exit decision"""
        try:
            symbol = position.symbol
            current_price = mt5.symbol_info_tick(symbol).bid if position.type == 0 else mt5.symbol_info_tick(symbol).ask
            
            # Calculate position metrics
            entry_price = position.price_open
            profit_pips = (current_price - entry_price) if position.type == 0 else (entry_price - current_price)
            profit_pct = profit_pips / entry_price
            time_held = (datetime.now() - pd.to_datetime(position.time, unit='s'))
            
            # Get market conditions
            market_state = self.get_market_state(symbol)
            if not market_state:
                return None
            
            # Calculate optimal exit parameters
            optimal_exit = self.calculate_optimal_exit(position, market_state, profit_pct)
            
            return {
                'position_id': position.ticket,
                'symbol': symbol,
                'type': 'buy' if position.type == 0 else 'sell',
                'profit_pips': profit_pips,
                'profit_pct': profit_pct,
                'time_held': time_held,
                'market_state': market_state,
                'optimal_exit': optimal_exit
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing position: {str(e)}")
            return None
    
    def get_market_state(self, symbol, timeframe=mt5.TIMEFRAME_M1, lookback=100):
        """Analyze current market state for exit decisions"""
        try:
            # Get recent price data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback)
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            
            # Calculate market metrics
            current_volatility = df['close'].pct_change().std() * np.sqrt(252)
            volume_ratio = df['tick_volume'].iloc[-1] / df['tick_volume'].mean()
            price_momentum = df['close'].pct_change().mean()
            current_spread = mt5.symbol_info(symbol).ask - mt5.symbol_info(symbol).bid
            
            # Get order book
            book = mt5.market_book_get(symbol)
            if book:
                book_df = pd.DataFrame([{'type': b.type, 'volume': b.volume, 'price': b.price} for b in book])
                buy_volume = book_df[book_df['type'] == 1]['volume'].sum()
                sell_volume = book_df[book_df['type'] == 2]['volume'].sum()
                order_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
            else:
                order_imbalance = 0
            
            market_state = {
                'volatility': current_volatility,
                'volume_ratio': volume_ratio,
                'momentum': price_momentum,
                'spread': current_spread,
                'order_imbalance': order_imbalance,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update market conditions history
            if symbol not in self.market_conditions:
                self.market_conditions[symbol] = deque(maxlen=1000)
            self.market_conditions[symbol].append(market_state)
            
            return market_state
            
        except Exception as e:
            self.logger.error(f"Error getting market state: {str(e)}")
            return None
    
    def calculate_optimal_exit(self, position, market_state, current_profit):
        """Calculate optimal exit parameters based on position and market conditions"""
        try:
            # Base exit levels
            base_profit_target = position.tp if position.tp else position.price_open * (1 + self.profit_target_multiplier)
            base_stop_loss = position.sl if position.sl else position.price_open * (1 - self.max_adverse_excursion)
            
            # Adjust based on market conditions
            if market_state['volatility'] > self.volatility_threshold:
                # Tighten stops in high volatility
                base_profit_target *= 0.9
                base_stop_loss = position.price_open + (base_stop_loss - position.price_open) * 0.8
            
            if abs(market_state['momentum']) > 0.001:
                # Adjust for strong momentum
                momentum_factor = 1.2 if market_state['momentum'] > 0 else 0.8
                base_profit_target *= momentum_factor
            
            if market_state['volume_ratio'] < self.volume_threshold:
                # More conservative in low volume
                base_profit_target *= 0.95
            
            # Calculate trailing stop
            trailing_stop = None
            if current_profit > self.trailing_stop_activation * (base_profit_target - position.price_open):
                trailing_stop = position.price_open + current_profit * 0.7
            
            # Determine if partial exit is warranted
            partial_exit = current_profit > self.partial_exit_threshold * (base_profit_target - position.price_open)
            
            return {
                'profit_target': base_profit_target,
                'stop_loss': base_stop_loss,
                'trailing_stop': trailing_stop,
                'partial_exit': partial_exit,
                'exit_urgency': self.calculate_exit_urgency(position, market_state)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal exit: {str(e)}")
            return None
    
    def calculate_exit_urgency(self, position, market_state):
        """Calculate the urgency of exiting a position"""
        try:
            urgency_factors = []
            
            # Volatility factor
            if market_state['volatility'] > self.volatility_threshold:
                urgency_factors.append(0.8)
            
            # Spread factor
            if market_state['spread'] > self.spread_threshold:
                urgency_factors.append(0.7)
            
            # Volume factor
            if market_state['volume_ratio'] < 1.0:
                urgency_factors.append(0.6)
            
            # Order imbalance factor
            if abs(market_state['order_imbalance']) > 0.3:
                if (position.type == 0 and market_state['order_imbalance'] < 0) or \
                   (position.type == 1 and market_state['order_imbalance'] > 0):
                    urgency_factors.append(0.9)
            
            # Momentum factor
            if (position.type == 0 and market_state['momentum'] < 0) or \
               (position.type == 1 and market_state['momentum'] > 0):
                urgency_factors.append(0.8)
            
            return max(urgency_factors) if urgency_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating exit urgency: {str(e)}")
            return 0.0
    
    def execute_smart_exit(self, position, analysis=None):
        """Execute smart exit strategy for position"""
        try:
            if not analysis:
                analysis = self.analyze_position(position)
                if not analysis:
                    return False
            
            optimal_exit = analysis['optimal_exit']
            
            # Check for immediate exit conditions
            if analysis['market_state']['spread'] > self.spread_threshold * 2:
                self.logger.warning(f"Spread too high for {position.symbol}, delaying exit")
                return False
            
            # Execute partial exit if warranted
            if optimal_exit['partial_exit'] and position.volume >= self.execution_engine.min_volume * 2:
                partial_volume = position.volume * 0.5
                result = self.execution_engine.execute_order(
                    position.symbol,
                    'sell' if position.type == 0 else 'buy',
                    partial_volume,
                    execution_algo='smart'
                )
                if result:
                    self.record_exit('partial', position, analysis, result)
            
            # Update stops if needed
            if optimal_exit['trailing_stop']:
                self.update_position_stops(position, optimal_exit['trailing_stop'], optimal_exit['profit_target'])
            
            # Check for full exit
            if analysis['optimal_exit']['exit_urgency'] > 0.8:
                result = self.execution_engine.execute_order(
                    position.symbol,
                    'sell' if position.type == 0 else 'buy',
                    position.volume,
                    execution_algo='smart'
                )
                if result:
                    self.record_exit('full', position, analysis, result)
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing smart exit: {str(e)}")
            return False
    
    def update_position_stops(self, position, stop_loss, take_profit):
        """Update position stop loss and take profit levels"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "sl": stop_loss,
                "tp": take_profit,
                "position": position.ticket
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.warning(f"Failed to update stops: {result.comment}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating position stops: {str(e)}")
            return False
    
    def record_exit(self, exit_type, position, analysis, result):
        """Record exit execution details"""
        try:
            exit_record = {
                'timestamp': datetime.now().isoformat(),
                'position_id': position.ticket,
                'symbol': position.symbol,
                'exit_type': exit_type,
                'volume': position.volume,
                'profit': position.profit,
                'market_state': analysis['market_state'],
                'exit_urgency': analysis['optimal_exit']['exit_urgency'],
                'execution_result': {
                    'order_id': result.order,
                    'price': result.price,
                    'volume': result.volume
                }
            }
            
            self.exit_history.append(exit_record)
            self.save_history()
            
        except Exception as e:
            self.logger.error(f"Error recording exit: {str(e)}")
    
    def get_exit_analytics(self):
        """Get analytics about exit performance"""
        try:
            if not self.exit_history:
                return None
            
            analytics = {}
            for symbol in set(e['symbol'] for e in self.exit_history):
                symbol_exits = [e for e in self.exit_history if e['symbol'] == symbol]
                
                # Calculate metrics
                profits = [e['profit'] for e in symbol_exits]
                urgencies = [e['exit_urgency'] for e in symbol_exits]
                
                analytics[symbol] = {
                    'total_exits': len(symbol_exits),
                    'avg_profit': np.mean(profits),
                    'profit_std': np.std(profits),
                    'avg_urgency': np.mean(urgencies),
                    'exit_types': {
                        'partial': len([e for e in symbol_exits if e['exit_type'] == 'partial']),
                        'full': len([e for e in symbol_exits if e['exit_type'] == 'full'])
                    }
                }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error getting exit analytics: {str(e)}")
            return None
    
    def save_history(self):
        """Save exit history"""
        try:
            history_data = {
                'exit_history': self.exit_history,
                'market_conditions': {
                    symbol: list(conditions) for symbol, conditions in self.market_conditions.items()
                }
            }
            
            with open('exit_history.json', 'w') as f:
                json.dump(history_data, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving history: {str(e)}")
    
    def load_history(self):
        """Load exit history"""
        try:
            with open('exit_history.json', 'r') as f:
                history_data = json.load(f)
                self.exit_history = history_data['exit_history']
                self.market_conditions = {
                    symbol: deque(conditions, maxlen=1000)
                    for symbol, conditions in history_data['market_conditions'].items()
                }
                
        except FileNotFoundError:
            self.logger.info("No exit history file found. Starting fresh.")
        except Exception as e:
            self.logger.error(f"Error loading history: {str(e)}")
