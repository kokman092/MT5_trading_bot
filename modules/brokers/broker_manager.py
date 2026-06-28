import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import MetaTrader5 as mt5
import ccxt
import ibpy2
from oandapyV20 import API
from fxcmpy import fxcmpy
import numpy as np
import pandas as pd

class BrokerManager:
    """Professional multi-broker management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_brokers = {}
        self.broker_weights = {}
        self._init_brokers()
        
    def _init_brokers(self):
        """Initialize connections to multiple brokers"""
        try:
            # Initialize MT5 (Primary)
            if self.config.get('mt5_enabled', True):
                mt5_config = self.config.get('mt5', {})
                if mt5.initialize(
                    login=mt5_config.get('login'),
                    server=mt5_config.get('server'),
                    password=mt5_config.get('password')
                ):
                    self.active_brokers['mt5'] = {
                        'instance': mt5,
                        'weight': 0.3,
                        'type': 'forex_cfd'
                    }
                    self.logger.info("MT5 connection established")
            
            # Initialize Interactive Brokers
            if self.config.get('ib_enabled', False):
                ib_config = self.config.get('interactive_brokers', {})
                ib = ibpy2.IB()
                ib.connect(
                    host=ib_config.get('host', 'localhost'),
                    port=ib_config.get('port', 7497),
                    clientId=ib_config.get('client_id', 1)
                )
                self.active_brokers['interactive_brokers'] = {
                    'instance': ib,
                    'weight': 0.2,
                    'type': 'stocks_options'
                }
                self.logger.info("Interactive Brokers connection established")
            
            # Initialize OANDA
            if self.config.get('oanda_enabled', False):
                oanda_config = self.config.get('oanda', {})
                oanda = API(
                    access_token=oanda_config.get('access_token'),
                    environment=oanda_config.get('environment', 'practice')
                )
                self.active_brokers['oanda'] = {
                    'instance': oanda,
                    'weight': 0.15,
                    'type': 'forex'
                }
                self.logger.info("OANDA connection established")
            
            # Initialize FXCM
            if self.config.get('fxcm_enabled', False):
                fxcm_config = self.config.get('fxcm', {})
                fxcm = fxcmpy(
                    access_token=fxcm_config.get('access_token'),
                    log_level='error',
                    server=fxcm_config.get('server', 'demo')
                )
                self.active_brokers['fxcm'] = {
                    'instance': fxcm,
                    'weight': 0.15,
                    'type': 'forex'
                }
                self.logger.info("FXCM connection established")
            
            # Initialize Binance (Crypto)
            if self.config.get('binance_enabled', False):
                binance_config = self.config.get('binance', {})
                binance = ccxt.binance({
                    'apiKey': binance_config.get('api_key'),
                    'secret': binance_config.get('api_secret'),
                    'enableRateLimit': True
                })
                self.active_brokers['binance'] = {
                    'instance': binance,
                    'weight': 0.2,
                    'type': 'crypto'
                }
                self.logger.info("Binance connection established")
            
            # Update broker weights
            self._update_broker_weights()
            
        except Exception as e:
            self.logger.error(f"Broker initialization error: {str(e)}")
            raise
            
    def _update_broker_weights(self):
        """Update broker allocation weights based on performance"""
        try:
            total_weight = sum(broker['weight'] for broker in self.active_brokers.values())
            if total_weight > 0:
                self.broker_weights = {
                    name: broker['weight'] / total_weight 
                    for name, broker in self.active_brokers.items()
                }
            
        except Exception as e:
            self.logger.error(f"Broker weight update error: {str(e)}")
            
    def execute_trade(self, trade_signal: Dict) -> Dict:
        """Execute trade with fixed risk management"""
        try:
            # Validate trade against risk rules
            if not self._validate_trade(trade_signal):
                return {
                    'status': 'error',
                    'message': 'Trade rejected by risk management rules'
                }
            
            # Calculate fixed position size
            volume = self._calculate_position_size(trade_signal)
            if not volume:
                return {
                    'status': 'error',
                    'message': 'Invalid position size'
                }
            
            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": trade_signal['symbol'],
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if trade_signal['direction'] == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": mt5.symbol_info_tick(trade_signal['symbol']).ask if trade_signal['direction'] == 'buy' else mt5.symbol_info_tick(trade_signal['symbol']).bid,
                "deviation": 10,
                "magic": 234000,
                "comment": "fixed risk trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add stop loss and take profit
            if trade_signal.get('stop_loss'):
                request["sl"] = trade_signal['stop_loss']
            if trade_signal.get('take_profit'):
                request["tp"] = trade_signal['take_profit']
            
            # Execute trade
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'status': 'error',
                    'message': f'Order failed: {result.comment}',
                    'retcode': result.retcode
                }
            
            return {
                'status': 'success',
                'order': result.order,
                'volume': result.volume,
                'price': result.price,
                'comment': result.comment
            }
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size using adaptive risk parameters based on volatility"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return 0.0
            
            # Get market volatility and adjust risk
            volatility = self._calculate_market_volatility()
            signal_strength = self._calculate_signal_strength(signal)
            
            # Base risk (0.5% to 1.5% based on conditions)
            base_risk = 0.005  # Start with 0.5% base risk
            
            # Adjust risk based on volatility
            if volatility < 0.3:  # Low volatility
                volatility_factor = 1.2  # Increase position size
            elif volatility > 0.7:  # High volatility
                volatility_factor = 0.6  # Reduce position size
            else:
                volatility_factor = 1.0
                
            # Adjust risk based on signal strength (0.8 to 1.5)
            signal_factor = 0.8 + (signal_strength * 0.7)
            
            # Calculate adaptive risk percentage
            adaptive_risk = min(base_risk * volatility_factor * signal_factor, 0.015)  # Cap at 1.5%
            
            # Get current balance
            balance = account_info.balance
            
            # Calculate risk amount in account currency
            risk_amount = balance * adaptive_risk
            
            # Get symbol info
            symbol_info = mt5.symbol_info(signal['symbol'])
            if not symbol_info:
                return 0.0
            
            # Calculate position size
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            
            if not entry_price or not stop_loss:
                return 0.0
            
            # Calculate pip value and position size
            pip_value = symbol_info.trade_tick_value
            points_at_risk = abs(entry_price - stop_loss) / symbol_info.point
            
            # Calculate lot size with adaptive position sizing
            lot_size = risk_amount / (points_at_risk * pip_value)
            
            # Apply correlation-based position reduction if needed
            correlation_factor = self._get_correlation_adjustment(signal['symbol'])
            lot_size *= correlation_factor
            
            # Round to valid lot step
            lot_step = symbol_info.volume_step
            lot_size = round(lot_size / lot_step) * lot_step
            
            # Apply limits
            lot_size = max(min(lot_size, symbol_info.volume_max), symbol_info.volume_min)
            
            return lot_size
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {str(e)}")
            return 0.0

    def _calculate_signal_strength(self, signal: Dict) -> float:
        """Calculate signal strength based on multiple factors"""
        try:
            strength = 0.0
            weight = 0.0
            
            # Factor 1: Risk-Reward Ratio (0.3 weight)
            if 'take_profit' in signal and 'stop_loss' in signal:
                rr_ratio = abs(signal['take_profit'] - signal['entry_price']) / abs(signal['entry_price'] - signal['stop_loss'])
                strength += min(rr_ratio / 3.0, 1.0) * 0.3
                weight += 0.3
            
            # Factor 2: Technical Confirmation (0.4 weight)
            if 'technical_score' in signal:
                strength += signal['technical_score'] * 0.4
                weight += 0.4
            
            # Factor 3: Market Regime Alignment (0.3 weight)
            if 'market_regime' in signal:
                regime_alignment = self._check_regime_alignment(signal['market_regime'], signal['direction'])
                strength += regime_alignment * 0.3
                weight += 0.3
            
            return strength / weight if weight > 0 else 0.5
            
        except Exception as e:
            self.logger.error(f"Signal strength calculation error: {str(e)}")
            return 0.5

    def _get_correlation_adjustment(self, symbol: str) -> float:
        """Calculate position size adjustment based on portfolio correlation"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return 1.0  # No correlation adjustment needed
            
            # Get correlation matrix for current positions
            symbols = [pos.symbol for pos in positions]
            if symbol not in symbols:
                symbols.append(symbol)
            
            if len(symbols) < 2:
                return 1.0
            
            # Calculate correlation matrix using recent price data
            correlation_matrix = self._calculate_correlation_matrix(symbols)
            
            # Calculate average correlation with existing positions
            correlations = []
            for existing_symbol in symbols:
                if existing_symbol != symbol:
                    if (symbol, existing_symbol) in correlation_matrix:
                        correlations.append(abs(correlation_matrix[(symbol, existing_symbol)]))
            
            if not correlations:
                return 1.0
            
            avg_correlation = sum(correlations) / len(correlations)
            
            # Apply correlation-based reduction
            if avg_correlation > 0.8:
                return 0.4  # Significant reduction for high correlation
            elif avg_correlation > 0.6:
                return 0.7  # Moderate reduction for medium correlation
            else:
                return 1.0  # No reduction for low correlation
            
        except Exception as e:
            self.logger.error(f"Correlation adjustment calculation error: {str(e)}")
            return 1.0

    def _calculate_correlation_matrix(self, symbols: List[str]) -> Dict[tuple, float]:
        """Calculate correlation matrix for given symbols"""
        try:
            correlation_matrix = {}
            
            # Get recent price data for all symbols
            price_data = {}
            for symbol in symbols:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
                if rates is not None:
                    price_data[symbol] = pd.DataFrame(rates)['close']
            
            # Calculate correlations
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    sym1, sym2 = symbols[i], symbols[j]
                    if sym1 in price_data and sym2 in price_data:
                        correlation = price_data[sym1].corr(price_data[sym2])
                        correlation_matrix[(sym1, sym2)] = correlation
                        correlation_matrix[(sym2, sym1)] = correlation
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Correlation matrix calculation error: {str(e)}")
            return {}

    def _check_regime_alignment(self, market_regime: str, trade_direction: str) -> float:
        """Check if trade direction aligns with market regime"""
        try:
            regime_alignments = {
                'STRONG_UPTREND': {'buy': 1.0, 'sell': 0.3},
                'UPTREND': {'buy': 0.8, 'sell': 0.5},
                'RANGING': {'buy': 0.6, 'sell': 0.6},
                'DOWNTREND': {'buy': 0.5, 'sell': 0.8},
                'STRONG_DOWNTREND': {'buy': 0.3, 'sell': 1.0}
            }
            
            return regime_alignments.get(market_regime, {}).get(trade_direction, 0.5)
            
        except Exception as e:
            self.logger.error(f"Regime alignment check error: {str(e)}")
            return 0.5

    def _validate_trade(self, signal: Dict) -> bool:
        """Validate trade using institutional risk management rules"""
        try:
            # 1. Check daily risk limit (1% institutional standard)
            daily_loss = self._get_daily_loss()
            if daily_loss >= (mt5.account_info().balance * 0.01):  # Reduced from 2% to 1%
                self.logger.warning("Daily risk limit reached")
                return False
            
            # 2. Check maximum drawdown (3% institutional standard)
            current_drawdown = self._get_current_drawdown()
            if current_drawdown >= 0.03:  # Reduced from 5% to 3%
                self.logger.warning("Maximum drawdown limit reached")
                return False
            
            # 3. Check position correlation
            if not self._check_position_correlation(signal):
                self.logger.warning("Position correlation check failed")
                return False
            
            # 4. Check risk-reward ratio (minimum 3:1 institutional standard)
            if not self._check_risk_reward_ratio(signal, min_ratio=3.0):
                self.logger.warning("Risk-reward ratio check failed")
                return False
            
            # 5. Check maximum positions (institutional standard)
            open_positions = len(mt5.positions_get())
            if open_positions >= 3:  # Reduced from 5 to 3 for better management
                self.logger.warning("Maximum position limit reached")
                return False
            
            # 6. Check daily trade limit (institutional standard)
            daily_trades = len(self._get_daily_trades())
            if daily_trades >= 6:  # Reduced from 10 to 6 for quality over quantity
                self.logger.warning("Daily trade limit reached")
                return False
            
            # 7. Check session volatility
            if not self._check_session_volatility():
                self.logger.warning("Session volatility check failed")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Trade validation error: {str(e)}")
            return False
    
    def _get_daily_loss(self) -> float:
        """Get total daily loss"""
        try:
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            trades = mt5.history_deals_get(today_start, datetime.now())
            
            if not trades:
                return 0.0
            
            total_loss = sum(trade.profit for trade in trades if trade.profit < 0)
            return abs(total_loss)
            
        except Exception as e:
            self.logger.error(f"Daily loss calculation error: {str(e)}")
            return 0.0
    
    def _get_current_drawdown(self) -> float:
        """Get current drawdown percentage"""
        try:
            account = mt5.account_info()
            if not account:
                return 0.0
            
            return (account.balance - account.equity) / account.balance
            
        except Exception as e:
            self.logger.error(f"Drawdown calculation error: {str(e)}")
            return 0.0
    
    def _get_daily_trades(self) -> List:
        """Get list of today's trades"""
        try:
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            return mt5.history_deals_get(today_start, datetime.now()) or []
            
        except Exception as e:
            self.logger.error(f"Daily trades retrieval error: {str(e)}")
            return []

    def get_broker_status(self) -> Dict:
        """Get status of all connected brokers"""
        return {
            name: {
                'connected': broker['instance'].connected if hasattr(broker['instance'], 'connected') else True,
                'weight': self.broker_weights.get(name, 0),
                'type': broker['type']
            }
            for name, broker in self.active_brokers.items()
        }
        
    def close_all_positions(self) -> Dict:
        """Close all positions across all brokers"""
        results = {}
        for name, broker in self.active_brokers.items():
            try:
                if broker['type'] == 'forex_cfd':
                    results[name] = self._close_mt5_positions(broker['instance'])
                elif broker['type'] == 'stocks_options':
                    results[name] = self._close_ib_positions(broker['instance'])
                elif broker['type'] == 'forex':
                    results[name] = self._close_forex_positions(broker['instance'])
                elif broker['type'] == 'crypto':
                    results[name] = self._close_crypto_positions(broker['instance'])
            except Exception as e:
                self.logger.error(f"Error closing positions for {name}: {str(e)}")
                results[name] = {'status': 'error', 'message': str(e)}
        
        return results 

    def _calculate_adaptive_take_profit(self, signal: Dict) -> float:
        """Calculate adaptive take profit based on multiple factors"""
        try:
            symbol = signal['symbol']
            direction = signal['direction']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            
            # 1. Calculate base take profit using risk-reward ratio
            risk_distance = abs(entry_price - stop_loss)
            base_tp_distance = risk_distance * 3.0  # Base RR ratio of 1:3
            
            # 2. Get market volatility adjustment
            volatility = self._calculate_market_volatility()
            volatility_factor = self._get_volatility_tp_factor(volatility)
            
            # 3. Get market regime adjustment
            regime_factor = self._get_regime_tp_factor(signal.get('market_regime', 'RANGING'))
            
            # 4. Calculate ATR-based adjustment
            atr_factor = self._get_atr_tp_factor(symbol)
            
            # 5. Find key price levels
            price_levels = self._find_key_price_levels(symbol, direction)
            
            # Calculate adaptive take profit distance
            adaptive_tp_distance = base_tp_distance * volatility_factor * regime_factor * atr_factor
            
            # Calculate initial take profit level
            take_profit = entry_price + (adaptive_tp_distance if direction == 'buy' else -adaptive_tp_distance)
            
            # Adjust take profit to nearest significant price level
            take_profit = self._adjust_to_price_level(take_profit, price_levels, direction)
            
            # Apply minimum and maximum TP constraints
            min_tp_distance = risk_distance * 2.0  # Minimum 1:2 RR ratio
            max_tp_distance = risk_distance * 5.0  # Maximum 1:5 RR ratio
            
            if direction == 'buy':
                take_profit = min(entry_price + max_tp_distance, max(entry_price + min_tp_distance, take_profit))
            else:
                take_profit = max(entry_price - max_tp_distance, min(entry_price - min_tp_distance, take_profit))
            
            # Round to symbol digits
            digits = mt5.symbol_info(symbol).digits
            take_profit = round(take_profit, digits)
            
            return take_profit
            
        except Exception as e:
            self.logger.error(f"Adaptive take profit calculation error: {str(e)}")
            return 0.0
            
    def _get_volatility_tp_factor(self, volatility: float) -> float:
        """Get take profit adjustment factor based on volatility"""
        try:
            if volatility < 0.2:  # Low volatility
                return 0.8  # Tighter TP
            elif volatility > 0.7:  # High volatility
                return 1.4  # Wider TP
            else:
                # Linear scaling between 0.8 and 1.4 for medium volatility
                return 0.8 + (volatility - 0.2) * (0.6 / 0.5)
                
        except Exception as e:
            self.logger.error(f"Volatility factor calculation error: {str(e)}")
            return 1.0
            
    def _get_regime_tp_factor(self, market_regime: str) -> float:
        """Get take profit adjustment factor based on market regime"""
        try:
            regime_factors = {
                'STRONG_UPTREND': 1.5,    # Wider TP in strong trends
                'UPTREND': 1.3,
                'RANGING': 0.8,           # Tighter TP in ranging markets
                'DOWNTREND': 1.3,
                'STRONG_DOWNTREND': 1.5
            }
            return regime_factors.get(market_regime, 1.0)
            
        except Exception as e:
            self.logger.error(f"Regime factor calculation error: {str(e)}")
            return 1.0
            
    def _get_atr_tp_factor(self, symbol: str) -> float:
        """Calculate ATR-based take profit factor"""
        try:
            # Get recent price data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
            if rates is None:
                return 1.0
                
            df = pd.DataFrame(rates)
            
            # Calculate ATR
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift(1))
            df['low_close'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            atr = df['tr'].mean()
            
            # Calculate ATR factor
            symbol_info = mt5.symbol_info(symbol)
            avg_spread = (symbol_info.ask - symbol_info.bid) / symbol_info.point
            
            # Normalize ATR relative to spread
            atr_spread_ratio = atr / avg_spread
            
            if atr_spread_ratio < 3:
                return 0.8  # Tighter TP for low ATR
            elif atr_spread_ratio > 10:
                return 1.4  # Wider TP for high ATR
            else:
                # Linear scaling between 0.8 and 1.4
                return 0.8 + (atr_spread_ratio - 3) * (0.6 / 7)
                
        except Exception as e:
            self.logger.error(f"ATR factor calculation error: {str(e)}")
            return 1.0
            
    def _find_key_price_levels(self, symbol: str, direction: str) -> List[float]:
        """Find key price levels for take profit adjustment"""
        try:
            levels = []
            
            # Get recent price data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 100)
            if rates is None:
                return levels
                
            df = pd.DataFrame(rates)
            
            # 1. Recent swing highs and lows
            window = 5
            df['high_roll_max'] = df['high'].rolling(window=window, center=True).max()
            df['low_roll_min'] = df['low'].rolling(window=window, center=True).min()
            
            swing_highs = df[df['high'] == df['high_roll_max']]['high'].tolist()
            swing_lows = df[df['low'] == df['low_roll_min']]['low'].tolist()
            
            # 2. Round numbers
            current_price = mt5.symbol_info_tick(symbol).ask if direction == 'buy' else mt5.symbol_info_tick(symbol).bid
            digits = mt5.symbol_info(symbol).digits
            multiplier = 10 ** digits
            
            round_numbers = [
                round(current_price + (i * 0.1), digits)
                for i in range(-5, 6)
            ]
            
            # 3. Previous day high/low
            daily_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 2)
            if daily_rates is not None:
                prev_day = pd.DataFrame(daily_rates).iloc[0]
                levels.extend([prev_day['high'], prev_day['low']])
            
            # Combine all levels
            levels.extend(swing_highs)
            levels.extend(swing_lows)
            levels.extend(round_numbers)
            
            # Remove duplicates and sort
            levels = sorted(list(set(levels)))
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Price levels calculation error: {str(e)}")
            return []
            
    def _adjust_to_price_level(self, take_profit: float, price_levels: List[float], direction: str) -> float:
        """Adjust take profit to nearest significant price level"""
        try:
            if not price_levels:
                return take_profit
                
            # Filter levels based on direction
            valid_levels = [level for level in price_levels if 
                          (direction == 'buy' and level > take_profit) or
                          (direction == 'sell' and level < take_profit)]
            
            if not valid_levels:
                return take_profit
                
            # Find nearest level
            nearest_level = min(valid_levels, key=lambda x: abs(x - take_profit))
            
            # Calculate adjustment threshold (0.2% of current price)
            threshold = take_profit * 0.002
            
            # Only adjust if nearest level is within threshold
            if abs(nearest_level - take_profit) <= threshold:
                return nearest_level
                
            return take_profit
            
        except Exception as e:
            self.logger.error(f"Price level adjustment error: {str(e)}")
            return take_profit

class MT5Manager:
    """Optimized MT5 Trading Manager"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.mt5_config = config['mt5']
        self.risk_config = config['risk_management']
        self.performance_config = config['performance_thresholds']
        self._init_mt5()
        
    def _init_mt5(self):
        """Initialize MT5 connection with enhanced error handling"""
        try:
            # Initialize MT5
            if not mt5.initialize(
                login=self.mt5_config.get('login'),
                server=self.mt5_config.get('server'),
                password=self.mt5_config.get('password'),
                timeout=self.mt5_config.get('timeout', 60000)
            ):
                raise Exception("MT5 initialization failed")
                
            # Verify account
            if not mt5.account_info():
                raise Exception("Failed to connect to trading account")
                
            self.logger.info("MT5 connection established successfully")
            
            # Initialize symbol settings
            self._init_symbols()
            
        except Exception as e:
            self.logger.error(f"MT5 initialization error: {str(e)}")
            raise
            
    def _init_symbols(self):
        """Initialize trading symbols with optimal settings"""
        try:
            self.symbols = {}
            
            # Process all symbol categories
            for category in ['major_pairs', 'cross_pairs', 'commodities', 'indices']:
                if category in self.mt5_config.get('symbols', {}):
                    for symbol in self.mt5_config['symbols'][category]:
                        # Enable symbol for trading
                        mt5.symbol_select(symbol, True)
                        
                        # Get symbol info
                        symbol_info = mt5.symbol_info(symbol)
                        if symbol_info:
                            self.symbols[symbol] = {
                                'tick_size': symbol_info.trade_tick_size,
                                'contract_size': symbol_info.trade_contract_size,
                                'max_lot': symbol_info.volume_max,
                                'min_lot': symbol_info.volume_min,
                                'lot_step': symbol_info.volume_step,
                                'points': symbol_info.point,
                                'digits': symbol_info.digits
                            }
                            
            self.logger.info(f"Initialized {len(self.symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Symbol initialization error: {str(e)}")
            raise
            
    def execute_trade(self, trade_signal: Dict) -> Dict:
        """Execute trade with enhanced validation and risk management"""
        try:
            # Validate trading session
            if not self._validate_trading_session():
                return {'status': 'error', 'message': 'Market closed or invalid session'}
                
            # Calculate adaptive take profit
            adaptive_tp = self._calculate_adaptive_take_profit(trade_signal)
            if adaptive_tp > 0:
                trade_signal['take_profit'] = adaptive_tp
            
            # Validate signal
            if not self._validate_signal(trade_signal):
                return {'status': 'error', 'message': 'Invalid trade signal'}
                
            # Calculate position size
            volume = self._calculate_position_size(trade_signal)
            if not volume:
                return {'status': 'error', 'message': 'Invalid position size'}
                
            # Prepare trade request
            request = self._prepare_trade_request(trade_signal, volume)
            
            # Execute trade
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'status': 'error',
                    'message': f'Order failed: {result.comment}',
                    'retcode': result.retcode
                }
                
            # Log trade
            self._log_trade(request, result)
            
            return {
                'status': 'success',
                'order': result.order,
                'volume': result.volume,
                'price': result.price,
                'bid': result.bid,
                'ask': result.ask,
                'comment': result.comment,
                'request_id': result.request_id,
                'retcode': result.retcode
            }
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    def _validate_trading_session(self) -> bool:
        """Validate if current time is within allowed trading sessions"""
        try:
            current_time = datetime.now().time()
            sessions = self.mt5_config.get('trading_sessions', {})
            
            for session in sessions.values():
                if not session.get('active', True):
                    continue
                    
                start_time = datetime.strptime(session['start'], "%H:%M").time()
                end_time = datetime.strptime(session['end'], "%H:%M").time()
                
                if start_time <= current_time <= end_time:
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Session validation error: {str(e)}")
            return False
            
    def _validate_signal(self, signal: Dict) -> bool:
        """Validate trade signal with enhanced checks"""
        try:
            required_fields = ['symbol', 'direction', 'entry_price', 'stop_loss', 'take_profit']
            if not all(field in signal for field in required_fields):
                return False
                
            symbol = signal['symbol']
            if symbol not in self.symbols:
                return False
                
            # Validate price levels
            if not self._validate_prices(signal):
                return False
                
            # Check risk-reward ratio
            if not self._check_risk_reward(signal):
                return False
                
            # Check maximum positions
            if not self._check_max_positions():
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {str(e)}")
            return False
            
    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size using adaptive risk parameters based on volatility"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return 0.0
            
            # Get market volatility and adjust risk
            volatility = self._calculate_market_volatility()
            signal_strength = self._calculate_signal_strength(signal)
            
            # Base risk (0.5% to 1.5% based on conditions)
            base_risk = 0.005  # Start with 0.5% base risk
            
            # Adjust risk based on volatility
            if volatility < 0.3:  # Low volatility
                volatility_factor = 1.2  # Increase position size
            elif volatility > 0.7:  # High volatility
                volatility_factor = 0.6  # Reduce position size
            else:
                volatility_factor = 1.0
                
            # Adjust risk based on signal strength (0.8 to 1.5)
            signal_factor = 0.8 + (signal_strength * 0.7)
            
            # Calculate adaptive risk percentage
            adaptive_risk = min(base_risk * volatility_factor * signal_factor, 0.015)  # Cap at 1.5%
            
            # Get current balance
            balance = account_info.balance
            
            # Calculate risk amount in account currency
            risk_amount = balance * adaptive_risk
            
            # Get symbol info
            symbol_info = mt5.symbol_info(signal['symbol'])
            if not symbol_info:
                return 0.0
            
            # Calculate position size
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            
            if not entry_price or not stop_loss:
                return 0.0
            
            # Calculate pip value and position size
            pip_value = symbol_info.trade_tick_value
            points_at_risk = abs(entry_price - stop_loss) / symbol_info.point
            
            # Calculate lot size with adaptive position sizing
            lot_size = risk_amount / (points_at_risk * pip_value)
            
            # Apply correlation-based position reduction if needed
            correlation_factor = self._get_correlation_adjustment(signal['symbol'])
            lot_size *= correlation_factor
            
            # Round to valid lot step
            lot_step = symbol_info.volume_step
            lot_size = round(lot_size / lot_step) * lot_step
            
            # Apply limits
            lot_size = max(min(lot_size, symbol_info.volume_max), symbol_info.volume_min)
            
            return lot_size
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {str(e)}")
            return 0.0

    def _calculate_signal_strength(self, signal: Dict) -> float:
        """Calculate signal strength based on multiple factors"""
        try:
            strength = 0.0
            weight = 0.0
            
            # Factor 1: Risk-Reward Ratio (0.3 weight)
            if 'take_profit' in signal and 'stop_loss' in signal:
                rr_ratio = abs(signal['take_profit'] - signal['entry_price']) / abs(signal['entry_price'] - signal['stop_loss'])
                strength += min(rr_ratio / 3.0, 1.0) * 0.3
                weight += 0.3
            
            # Factor 2: Technical Confirmation (0.4 weight)
            if 'technical_score' in signal:
                strength += signal['technical_score'] * 0.4
                weight += 0.4
            
            # Factor 3: Market Regime Alignment (0.3 weight)
            if 'market_regime' in signal:
                regime_alignment = self._check_regime_alignment(signal['market_regime'], signal['direction'])
                strength += regime_alignment * 0.3
                weight += 0.3
            
            return strength / weight if weight > 0 else 0.5
            
        except Exception as e:
            self.logger.error(f"Signal strength calculation error: {str(e)}")
            return 0.5

    def _get_correlation_adjustment(self, symbol: str) -> float:
        """Calculate position size adjustment based on portfolio correlation"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return 1.0  # No correlation adjustment needed
            
            # Get correlation matrix for current positions
            symbols = [pos.symbol for pos in positions]
            if symbol not in symbols:
                symbols.append(symbol)
            
            if len(symbols) < 2:
                return 1.0
            
            # Calculate correlation matrix using recent price data
            correlation_matrix = self._calculate_correlation_matrix(symbols)
            
            # Calculate average correlation with existing positions
            correlations = []
            for existing_symbol in symbols:
                if existing_symbol != symbol:
                    if (symbol, existing_symbol) in correlation_matrix:
                        correlations.append(abs(correlation_matrix[(symbol, existing_symbol)]))
            
            if not correlations:
                return 1.0
            
            avg_correlation = sum(correlations) / len(correlations)
            
            # Apply correlation-based reduction
            if avg_correlation > 0.8:
                return 0.4  # Significant reduction for high correlation
            elif avg_correlation > 0.6:
                return 0.7  # Moderate reduction for medium correlation
            else:
                return 1.0  # No reduction for low correlation
            
        except Exception as e:
            self.logger.error(f"Correlation adjustment calculation error: {str(e)}")
            return 1.0

    def _calculate_correlation_matrix(self, symbols: List[str]) -> Dict[tuple, float]:
        """Calculate correlation matrix for given symbols"""
        try:
            correlation_matrix = {}
            
            # Get recent price data for all symbols
            price_data = {}
            for symbol in symbols:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
                if rates is not None:
                    price_data[symbol] = pd.DataFrame(rates)['close']
            
            # Calculate correlations
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    sym1, sym2 = symbols[i], symbols[j]
                    if sym1 in price_data and sym2 in price_data:
                        correlation = price_data[sym1].corr(price_data[sym2])
                        correlation_matrix[(sym1, sym2)] = correlation
                        correlation_matrix[(sym2, sym1)] = correlation
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Correlation matrix calculation error: {str(e)}")
            return {}

    def _check_regime_alignment(self, market_regime: str, trade_direction: str) -> float:
        """Check if trade direction aligns with market regime"""
        try:
            regime_alignments = {
                'STRONG_UPTREND': {'buy': 1.0, 'sell': 0.3},
                'UPTREND': {'buy': 0.8, 'sell': 0.5},
                'RANGING': {'buy': 0.6, 'sell': 0.6},
                'DOWNTREND': {'buy': 0.5, 'sell': 0.8},
                'STRONG_DOWNTREND': {'buy': 0.3, 'sell': 1.0}
            }
            
            return regime_alignments.get(market_regime, {}).get(trade_direction, 0.5)
            
        except Exception as e:
            self.logger.error(f"Regime alignment check error: {str(e)}")
            return 0.5

    def _validate_trade(self, signal: Dict) -> bool:
        """Validate trade against fixed risk management rules"""
        try:
            # 1. Check daily risk limit (2%)
            daily_loss = self._get_daily_loss()
            if daily_loss >= (mt5.account_info().balance * 0.02):
                return False
            
            # 2. Check maximum drawdown (5%)
            current_drawdown = self._get_current_drawdown()
            if current_drawdown >= 0.05:
                return False
            
            # 3. Check position limits (max 5 positions)
            open_positions = len(mt5.positions_get())
            if open_positions >= 5:
                return False
            
            # 4. Check daily trade limit (max 10 trades)
            daily_trades = len(self._get_daily_trades())
            if daily_trades >= 10:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Trade validation error: {str(e)}")
            return False
    
    def _get_daily_loss(self) -> float:
        """Get total daily loss"""
        try:
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            trades = mt5.history_deals_get(today_start, datetime.now())
            
            if not trades:
                return 0.0
            
            total_loss = sum(trade.profit for trade in trades if trade.profit < 0)
            return abs(total_loss)
            
        except Exception as e:
            self.logger.error(f"Daily loss calculation error: {str(e)}")
            return 0.0
    
    def _get_current_drawdown(self) -> float:
        """Get current drawdown percentage"""
        try:
            account = mt5.account_info()
            if not account:
                return 0.0
            
            return (account.balance - account.equity) / account.balance
            
        except Exception as e:
            self.logger.error(f"Drawdown calculation error: {str(e)}")
            return 0.0
    
    def _get_daily_trades(self) -> List:
        """Get list of today's trades"""
        try:
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            return mt5.history_deals_get(today_start, datetime.now()) or []
            
        except Exception as e:
            self.logger.error(f"Daily trades retrieval error: {str(e)}")
            return []

    def _prepare_trade_request(self, signal: Dict, volume: float) -> Dict:
        """Prepare trade request with enhanced parameters"""
        try:
            symbol = signal['symbol']
            direction = signal['direction']
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": mt5.symbol_info_tick(symbol).ask if direction == 'buy' else mt5.symbol_info_tick(symbol).bid,
                "deviation": self.mt5_config['expert_settings']['deviation'],
                "magic": self.mt5_config['expert_settings']['magic_number'],
                "comment": self.mt5_config['expert_settings']['comment'],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            if signal.get('stop_loss'):
                request["sl"] = signal['stop_loss']
            if signal.get('take_profit'):
                request["tp"] = signal['take_profit']
                
            return request
            
        except Exception as e:
            self.logger.error(f"Trade request preparation error: {str(e)}")
            raise
            
    def _get_current_risk_level(self) -> str:
        """Determine current risk level based on performance and market conditions"""
        try:
            # Get recent performance metrics
            win_rate = self._calculate_win_rate()
            drawdown = self._calculate_drawdown()
            volatility = self._calculate_market_volatility()
            
            # Conservative conditions
            if (drawdown > self.risk_config['max_drawdown'] * 0.7 or
                win_rate < self.performance_config['min_win_rate'] or
                volatility > 0.7):
                return 'conservative'
                
            # Aggressive conditions
            if (drawdown < self.risk_config['max_drawdown'] * 0.3 and
                win_rate > self.performance_config['min_win_rate'] * 1.2 and
                volatility < 0.4):
                return 'aggressive'
                
            # Default to moderate
            return 'moderate'
            
        except Exception as e:
            self.logger.error(f"Risk level determination error: {str(e)}")
            return 'conservative'
            
    def _calculate_win_rate(self) -> float:
        """Calculate recent win rate"""
        try:
            # Get recent trades
            trades = pd.DataFrame(mt5.history_deals_get(
                datetime.now() - pd.Timedelta(days=7),
                datetime.now()
            ))
            
            if trades.empty:
                return 0.0
                
            winning_trades = len(trades[trades['profit'] > 0])
            total_trades = len(trades)
            
            return winning_trades / total_trades if total_trades > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Win rate calculation error: {str(e)}")
            return 0.0
            
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return 0.0
                
            equity = account_info.equity
            balance = account_info.balance
            
            return (balance - equity) / balance if balance > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Drawdown calculation error: {str(e)}")
            return 0.0
            
    def _calculate_market_volatility(self) -> float:
        """Calculate current market volatility"""
        try:
            # Get recent price data
            prices = pd.DataFrame(mt5.copy_rates_from(
                "EURUSD", mt5.TIMEFRAME_H1,
                datetime.now(),
                100
            ))
            
            if prices.empty:
                return 0.0
                
            # Calculate volatility using standard deviation of returns
            returns = np.log(prices['close'] / prices['close'].shift(1))
            volatility = returns.std()
            
            return min(volatility * 100, 1.0)  # Normalize to 0-1 range
            
        except Exception as e:
            self.logger.error(f"Volatility calculation error: {str(e)}")
            return 0.0
            
    def _log_trade(self, request: Dict, result: Any):
        """Log trade details for analysis"""
        try:
            log_entry = {
                'timestamp': datetime.now(),
                'symbol': request['symbol'],
                'type': request['type'],
                'volume': request['volume'],
                'price': result.price,
                'order_id': result.order,
                'profit': result.profit if hasattr(result, 'profit') else 0,
                'comment': result.comment
            }
            
            # Log to file or database
            self.logger.info(f"Trade executed: {log_entry}")
            
        except Exception as e:
            self.logger.error(f"Trade logging error: {str(e)}")
            
    def close_all_positions(self) -> Dict:
        """Close all open positions"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return {'status': 'success', 'message': 'No open positions'}
                
            results = []
            for position in positions:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                    "deviation": self.mt5_config['expert_settings']['deviation'],
                    "magic": self.mt5_config['expert_settings']['magic_number'],
                    "comment": "position close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(close_request)
                results.append({
                    'ticket': position.ticket,
                    'symbol': position.symbol,
                    'volume': position.volume,
                    'status': 'success' if result.retcode == mt5.TRADE_RETCODE_DONE else 'error',
                    'message': result.comment
                })
                
            return {
                'status': 'success',
                'closed_positions': results
            }
            
        except Exception as e:
            self.logger.error(f"Position closing error: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    def get_account_info(self) -> Dict:
        """Get detailed account information"""
        try:
            account = mt5.account_info()
            if not account:
                return {'status': 'error', 'message': 'Failed to get account info'}
                
            return {
                'balance': account.balance,
                'equity': account.equity,
                'profit': account.profit,
                'margin': account.margin,
                'margin_free': account.margin_free,
                'margin_level': account.margin_level,
                'leverage': account.leverage,
                'currency': account.currency
            }
            
        except Exception as e:
            self.logger.error(f"Account info error: {str(e)}")
            return {'status': 'error', 'message': str(e)} 