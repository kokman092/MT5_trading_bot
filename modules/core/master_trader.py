from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
import MetaTrader5 as mt5
import ta
import pytz
import asyncio
import math
import re

@dataclass
class MarketState:
    """Market state data structure"""
    symbol: str = ''
    strategy: str = ''
    volatility: float = 0.0
    trend_strength: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0
    support: float = 0.0
    resistance: float = 0.0
    atr: float = 0.0
    in_session: bool = False
    market_data: Dict = field(default_factory=dict)

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    valid: bool = False
    executed: bool = False
    direction: Optional[str] = None
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    order_id: Optional[int] = None
    reason: str = ""
    strategy: str = ""
    
    @classmethod
    def create_invalid(cls, reason: str) -> 'TradingSignal':
        """Create an invalid signal with a reason"""
        return cls(valid=False, executed=False, reason=reason)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'valid': self.valid,
            'executed': self.executed,
            'direction': self.direction,
            'entry': self.entry,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'order_id': self.order_id,
            'reason': self.reason
        }

class MasterTrader:
    """Master trading class that manages all trading operations"""
    
    # Define timeframe mappings
    TIMEFRAME_MAP = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }
    
    # Define pip value mapping
    PIP_MULTIPLIER = {
        'JPY': 0.01,    # 1 pip = 0.01 for JPY pairs
        'DEFAULT': 0.0001  # 1 pip = 0.0001 for other pairs
    }
    
    def __init__(self, config: Dict):
        """Initialize MasterTrader with configuration"""
        self.config = config
        self.logger = logging.getLogger('master_trader')
        
        # Initialize MT5 connection if not already connected
        if not mt5.initialize():
            self.logger.error("MT5 initialization failed")
            raise Exception("MT5 initialization failed")
            
        # Set timezone to UTC
        timezone = pytz.timezone("Etc/UTC")
        now = datetime.now(timezone)
        
        # Login to MT5
        if not mt5.login(
            login=int(self.config['account']['login']),
            password=self.config['account']['password'],
            server=self.config['account']['server']
        ):
            self.logger.error("MT5 login failed")
            raise Exception("MT5 login failed")
            
        # Initialize trading parameters
        self._init_trading_parameters()
        
        # Enable market data for symbols
        for symbol in self.config['trading']['symbols']:
            if not mt5.symbol_select(symbol, True):
                self.logger.error(f"Failed to enable market data for {symbol}")
                raise Exception(f"Failed to enable market data for {symbol}")
                
            # Wait for data to be available
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Failed to get info for {symbol}")
                raise Exception(f"Failed to get info for {symbol}")
                
            # Log symbol info in a single statement
            self.logger.info(
                f"Symbol {symbol} initialized:\n"
                f"  Spread: {symbol_info.spread} points\n"
                f"  Tick Size: {symbol_info.trade_tick_size}\n"
                f"  Contract Size: {symbol_info.trade_contract_size}\n"
                f"  Volume Step: {symbol_info.volume_step}"
            )

    def _init_trading_parameters(self):
        """Initialize trading parameters"""
        # Load strategy parameters from config
        self.strategy_params = self.config['trading']['strategy_params']
        
        # Risk management parameters
        self.risk_params = {
            'max_risk_per_trade': self.config['trading']['risk_management']['max_risk_per_trade'],
            'max_daily_risk': self.config['trading']['risk_management']['max_daily_risk'],
            'max_positions': self.config['trading']['risk_management']['max_positions'],
            'use_trailing_stop': self.config['trading']['risk_management']['use_trailing_stop'],
            'trailing_stop_activation': self.config['trading']['risk_management']['trailing_stop_activation'],
            'max_volatility': 0.02  # 2% max volatility
        }
        
        # Execution parameters
        self.execution_params = {
            'max_slippage': self.config['trading']['execution']['max_slippage'],
            'magic_number': self.config['trading']['execution']['magic_number'],
            'max_retries': self.config['trading']['execution']['max_retries'],
            'retry_delay': self.config['trading']['execution']['retry_delay']
        }
        
    async def execute_strategy(
        self,
        symbol: str,
        strategy_type: str
    ) -> List[TradingSignal]:
        """Execute trading strategy"""
        try:
            # Analyze market state
            market_state = await self._analyze_market_state(symbol, strategy_type)
            self.logger.info(f"Market analysis completed for {symbol} using {strategy_type} strategy")
            
            # Generate signals
            signals = await self._generate_trading_signals(symbol, strategy_type, market_state)
            if not signals:
                return []
                
            # Execute valid signals
            for signal in signals:
                if not signal.valid:
                    continue
                    
                # Execute trade
                success = await self._execute_trade(symbol, signal)
                if not success:
                    signal.valid = False
                    signal.executed = False
                    signal.reason = "Trade execution failed"
                    continue
                    
            return signals
            
        except Exception as e:
            self.logger.error(f"Strategy execution error for {symbol} ({strategy_type}): {str(e)}")
            return []
            
    async def execute_trade(self, request: Dict) -> Dict:
        """Execute a trade with retry logic"""
        try:
            # Validate request
            if not self._validate_trade_request(request):
                return {'success': False, 'error': 'Invalid trade request'}
                
            # Calculate position size
            lot_size = await self._calculate_position_size(
                request['symbol'],
                request['signal']
            )
            
            if lot_size == 0:
                return {'success': False, 'error': 'Invalid lot size'}
                
            # Prepare trade request
            trade_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": request['symbol'],
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY if request['direction'] == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": request['entry'],
                "sl": request['stop_loss'],
                "tp": request['take_profit'],
                "deviation": self.config['execution']['max_slippage'],
                "magic": self.config['execution']['magic_number'],
                "comment": self._sanitize_comment("trade"),  # Use a simple comment that will be sanitized
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK
            }
            
            # Execute trade with retries
            max_retries = self.config['execution']['max_retries']
            retry_delay = self.config['execution']['retry_delay']
            
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"Executing trade (attempt {attempt + 1}/{max_retries}):")
                    self.logger.info(f"Symbol: {trade_request['symbol']}")
                    self.logger.info(f"Direction: {request['direction']}")
                    self.logger.info(f"Entry: {trade_request['price']}")
                    self.logger.info(f"Stop Loss: {trade_request['sl']}")
                    self.logger.info(f"Take Profit: {trade_request['tp']}")
                    self.logger.info(f"Volume: {trade_request['volume']}")
                    
                    # Send trade request
                    result = mt5.order_send(trade_request)
                    
                    if result is None:
                        error = mt5.last_error()
                        self.logger.error(f"Trade failed: {error}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        return {'success': False, 'error': f'Trade failed: {error}'}
                        
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        self.logger.error(f"Trade failed with retcode {result.retcode}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        return {'success': False, 'error': f'Trade failed with retcode {result.retcode}'}
                        
                    # Trade successful
                    self.logger.info(f"Trade executed successfully. Order ID: {result.order}")
                    return {
                        'success': True,
                        'order_id': result.order,
                        'entry': result.price,
                        'volume': result.volume
                    }
                    
                except Exception as e:
                    self.logger.error(f"Trade execution error: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    return {'success': False, 'error': f'Trade execution error: {str(e)}'}
            
            return {'success': False, 'error': 'Max retries exceeded'}
            
        except Exception as e:
            self.logger.error(f"Fatal trade execution error: {str(e)}")
            return {'success': False, 'error': f'Fatal trade execution error: {str(e)}'}
            
    async def _analyze_market_state(
        self,
        symbol: str,
        strategy_type: str
    ) -> MarketState:
        """Analyze market state for a given symbol and strategy"""
        try:
            # Get strategy parameters
            strategy_params = self.config['trading']['strategy_params'][strategy_type]
            timeframes = strategy_params['timeframes']
            
            market_state = MarketState()
            market_state.symbol = symbol
            market_state.strategy = strategy_type
            market_state.market_data = {}
            
            # Get data for each timeframe
            for timeframe in timeframes:
                if timeframe not in self.TIMEFRAME_MAP:
                    raise ValueError(f"Invalid timeframe: {timeframe}")
                    
                mt5_timeframe = self.TIMEFRAME_MAP[timeframe]
                
                # Fetch market data
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, 100)
                if rates is None:
                    error = mt5.last_error()
                    raise Exception(f"Failed to get market data: {error}")
                    
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Calculate indicators
                df = self._calculate_indicators(df, strategy_params)
                
                # Store in market state
                market_state.market_data[timeframe] = df
                
            # Calculate volatility
            market_state.volatility = self._calculate_volatility(df)
            
            # Calculate trend strength
            market_state.trend_strength = self._calculate_trend_strength(df)
            
            # Calculate RSI
            market_state.rsi = df['rsi'].iloc[-1]
            
            # Calculate MACD
            market_state.macd = df['macd'].iloc[-1]
            
            # Calculate support and resistance
            market_state.support = self._find_support(df)
            market_state.resistance = self._find_resistance(df)
            
            # Calculate ATR
            market_state.atr = df['atr'].iloc[-1]
            
            # Check if we're in a valid trading session
            market_state.in_session = self._is_in_trading_session()
            
            self.logger.info(f"Market analysis completed for {symbol} using {strategy_type} strategy")
            return market_state
            
        except Exception as e:
            self.logger.error(f"Market analysis error: {str(e)}")
            raise
            
    async def _generate_trading_signals(
        self,
        symbol: str,
        strategy_type: str,
        market_state: MarketState
    ) -> List[TradingSignal]:
        """Generate trading signals based on strategy type and market state"""
        try:
            # Get strategy parameters
            strategy_params = self.config['trading']['strategy_params'][strategy_type]
            
            # Get market data
            df = market_state.market_data[strategy_params['timeframes'][0]]
            
            # Generate signals based on strategy type
            if strategy_type == 'scalping':
                signals = await self._generate_scalping_signals(df, market_state)
            elif strategy_type == 'trend':
                signals = await self._generate_trend_signals(df, market_state)
            else:
                self.logger.error(f"Unknown strategy type: {strategy_type}")
                return []
                
            # Log signal generation
            for signal in signals:
                if signal.valid:
                    self.logger.info(f"Signal validated with reasons: {signal.reason}")
                    
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            return []

    async def _generate_scalping_signals(self, df: pd.DataFrame, market_state: MarketState) -> List[TradingSignal]:
        """Generate scalping signals with enhanced indicator confirmation"""
        try:
            signals = []
            
            # Calculate indicators
            df = self._calculate_indicators(df, self.config['trading']['strategy_params']['scalping'])
            
            # Get latest values
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            
            # Check for potential long setup
            if (rsi < 30 and  # Oversold
                macd > macd_signal and  # MACD bullish crossover
                df['trend_direction'].iloc[-1] == 1):  # Uptrend
                
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
                
                signal = TradingSignal(
                    valid=True,
                    executed=False,
                    direction=1,
                    entry=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    order_id=None,
                    reason="Scalping long: RSI oversold + MACD bullish + Uptrend",
                    strategy='scalping'
                )
                signals.append(signal)
            
            # Check for potential short setup
            elif (rsi > 70 and  # Overbought
                macd < macd_signal and  # MACD bearish crossover
                df['trend_direction'].iloc[-1] == -1):  # Downtrend
                
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
                
                signal = TradingSignal(
                    valid=True,
                    executed=False,
                    direction=-1,
                    entry=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    order_id=None,
                    reason="Scalping short: RSI overbought + MACD bearish + Downtrend",
                    strategy='scalping'
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating scalping signals: {str(e)}")
            return []

    async def _generate_trend_signals(self, df: pd.DataFrame, market_state: MarketState) -> List[TradingSignal]:
        """Generate trend signals with enhanced indicator confirmation"""
        try:
            signals = []
            
            # Calculate indicators
            df = self._calculate_indicators(df, self.config['trading']['strategy_params']['trend'])
            
            # Get latest values
            current_adx = df['adx'].iloc[-1]
            current_dmi_plus = df['di_plus'].iloc[-1]
            current_dmi_minus = df['di_minus'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            current_macd = df['macd'].iloc[-1]
            current_macd_signal = df['macd_signal'].iloc[-1]
            
            # Log indicator values
            self.logger.info(f"Trend indicators for {market_state.symbol}:")
            self.logger.info(f"  ADX: {current_adx:.1f}")
            self.logger.info(f"  DMI+: {current_dmi_plus:.1f}")
            self.logger.info(f"  DMI-: {current_dmi_minus:.1f}")
            self.logger.info(f"  RSI: {current_rsi:.1f}")
            self.logger.info(f"  MACD: {current_macd:.5f}")
            self.logger.info(f"  MACD Signal: {current_macd_signal:.5f}")
            
            # Create signal
            signal = TradingSignal()
            signal.strategy = "trend"
            
            # Strong trend conditions
            if current_adx > 30:  # Strong trend
                if current_dmi_plus > current_dmi_minus:  # Uptrend
                    if current_rsi > 50:  # Momentum confirmation
                        signal.valid = True
                        signal.direction = "buy"
                        signal.reason = f"Trend long: Strong uptrend (ADX={current_adx:.1f}) + DMI+ dominant + RSI momentum"
                        
                        # Get current price info
                        tick = mt5.symbol_info_tick(market_state.symbol)
                        if tick is not None:
                            signal.entry = tick.ask
                            
                            # Calculate stop loss and take profit
                            atr = df['atr'].iloc[-1]
                            signal.stop_loss = signal.entry - (atr * 2)  # 2 ATR for stop loss
                            signal.take_profit = signal.entry + (atr * 3)  # 3 ATR for take profit
                            
                            self.logger.info(f"Generated BUY signal for {market_state.symbol}:")
                            self.logger.info(f"  Entry: {signal.entry:.5f}")
                            self.logger.info(f"  Stop Loss: {signal.stop_loss:.5f}")
                            self.logger.info(f"  Take Profit: {signal.take_profit:.5f}")
                            
                elif current_dmi_minus > current_dmi_plus:  # Downtrend
                    if current_rsi < 50:  # Momentum confirmation
                        signal.valid = True
                        signal.direction = "sell"
                        signal.reason = f"Trend short: Strong downtrend (ADX={current_adx:.1f}) + DMI- dominant + RSI momentum"
                        
                        # Get current price info
                        tick = mt5.symbol_info_tick(market_state.symbol)
                        if tick is not None:
                            signal.entry = tick.bid
                            
                            # Calculate stop loss and take profit
                            atr = df['atr'].iloc[-1]
                            signal.stop_loss = signal.entry + (atr * 2)  # 2 ATR for stop loss
                            signal.take_profit = signal.entry - (atr * 3)  # 3 ATR for take profit
                            
                            self.logger.info(f"Generated SELL signal for {market_state.symbol}:")
                            self.logger.info(f"  Entry: {signal.entry:.5f}")
                            self.logger.info(f"  Stop Loss: {signal.stop_loss:.5f}")
                            self.logger.info(f"  Take Profit: {signal.take_profit:.5f}")
                
            signals.append(signal)
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trend signals: {str(e)}")
            return []
            
    async def process_symbol(self, symbol: str, strategy_type: str) -> None:
        """Process a single symbol for trading opportunities"""
        try:
            # Get market state
            market_state = await self._get_market_state(symbol)
            if market_state is None:
                return
                
            # Skip if market is closed or in high spread
            if not market_state.is_tradeable:
                return
                
            # Log market analysis start
            self.logger.info(f"Starting market analysis for {symbol} using {strategy_type} strategy")
            
            # Generate trading signals
            signals = await self._generate_trading_signals(symbol, strategy_type, market_state)
            
            # Execute valid signals
            for signal in signals:
                if signal.valid:
                    self.logger.info(f"Checking risk parameters for {symbol}...")
                    if self._check_risk_parameters(symbol, signal):
                        self.logger.info(f"Risk parameters passed for {symbol}")
                        if self._validate_spread(symbol):
                            self.logger.info(f"Spread validation passed for {symbol}")
                            if self._check_correlation(symbol):
                                self.logger.info(f"Correlation check passed for {symbol}")
                                if self._is_in_trading_session():
                                    self.logger.info(f"Trading session check passed for {symbol}")
                                    result = await self._execute_trade(symbol, signal)
                                    if result['success']:
                                        self.logger.info(f"Trade executed for {symbol}: {result['order_id']}")
                                    else:
                                        self.logger.warning(f"Trade execution failed for {symbol}: {result.get('error', 'Unknown error')}")
                                else:
                                    self.logger.warning(f"Not in trading session for {symbol}")
                            else:
                                self.logger.warning(f"Failed correlation check for {symbol}")
                        else:
                            self.logger.warning(f"Failed spread validation for {symbol}")
                    else:
                        self.logger.warning(f"Failed risk parameters check for {symbol}")
            
            # Manage existing positions
            await self._manage_open_positions()
            
            # Log completion
            self.logger.info(f"Market analysis completed for {symbol} using {strategy_type} strategy")
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {str(e)}")

    async def run(self):
        """Main trading loop"""
        try:
            while True:
                for symbol in self.config['trading']['symbols']:
                    try:
                        # Get latest prices
                        symbol_info = mt5.symbol_info_tick(symbol)
                        if symbol_info is None:
                            self.logger.error(f"Failed to get tick data for {symbol}")
                            continue
                            
                        self.logger.info(f"Processing {symbol}")
                        self.logger.info(f"Bid: {symbol_info.bid}, Ask: {symbol_info.ask}")
                        
                        # Check for active strategies
                        for strategy_type in self.config['trading']['active_strategies']:
                            try:
                                self.logger.info(f"Executing {strategy_type} strategy for {symbol}")
                                
                                # Process symbol
                                await self.process_symbol(symbol, strategy_type)
                                
                            except Exception as e:
                                self.logger.error(f"Error executing {strategy_type} strategy for {symbol}: {str(e)}")
                                continue
                                
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {str(e)}")
                        continue
                        
                # Wait before next iteration
                await asyncio.sleep(self.config['trading']['scan_interval'])
                
        except Exception as e:
            self.logger.error(f"Fatal error in trading loop: {str(e)}")
            raise
            
        finally:
            # Clean up resources
            mt5.shutdown()

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        try:
            # Use ADX for trend strength
            adx = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            ).adx()
            
            # Use last value
            return adx.iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {str(e)}")
            return 0

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate market volatility"""
        try:
            # Calculate daily returns
            returns = df['close'].pct_change()
            
            # Calculate volatility (standard deviation of returns)
            volatility = returns.std()
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return 0

    def _check_market_conditions(self, df: pd.DataFrame, strategy_params: Dict, symbol: str) -> bool:
        """Check if market conditions are suitable for trading"""
        try:
            # Check spread
            if not self._validate_spread(symbol):
                return False
            
            # Check volatility
            volatility = self._calculate_volatility(df)
            if volatility > self.config['trading']['risk_management'].get('max_volatility', 0.02):
                self.logger.info(f"Volatility too high: {volatility:.4f}")
                return False
            
            # Check trend strength for trend strategy
            if 'min_trend_strength' in strategy_params:
                trend_strength = self._calculate_trend_strength(df)
                if trend_strength < strategy_params['min_trend_strength']:
                    self.logger.info(f"Trend strength too low: {trend_strength:.2f}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking market conditions: {str(e)}")
            return False

    def _calculate_trend_direction(self, df: pd.DataFrame) -> int:
        """Calculate trend direction using multiple timeframes
        Returns:
            int: 1 for uptrend, -1 for downtrend, 0 for no clear trend
        """
        try:
            if df.empty:
                return 0

            # Calculate moving averages
            df['ma_fast'] = ta.trend.SMAIndicator(df['close'], window=10).sma_indicator()
            df['ma_slow'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            
            # Get latest values
            current_price = df['close'].iloc[-1]
            ma_fast = df['ma_fast'].iloc[-1]
            ma_slow = df['ma_slow'].iloc[-1]
            
            # Calculate ADX for trend strength
            adx = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            ).adx()
            
            trend_strength = adx.iloc[-1]
            
            # Determine trend direction
            if trend_strength >= 25:  # Strong trend
                if ma_fast > ma_slow and current_price > ma_fast:
                    return 1  # Uptrend
                elif ma_fast < ma_slow and current_price < ma_fast:
                    return -1  # Downtrend
            
            return 0  # No clear trend or weak trend
            
        except Exception as e:
            self.logger.error(f"Trend direction calculation error: {str(e)}")
            return 0

    def _calculate_spread_in_pips(self, symbol: str) -> float:
        """
        Calculate spread in pips for a given symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'USDJPY')
            
        Returns:
            float: Spread in pips, or inf if calculation fails
        """
        try:
            # Get symbol info and tick data
            symbol_info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            
            if symbol_info is None or tick is None:
                self.logger.error(f"Failed to get market data for spread calculation: {symbol}")
                return float('inf')
                
            # Calculate spread in price terms
            spread_price = tick.ask - tick.bid
            
            # Determine pip value based on symbol
            pip_multiplier = self.PIP_MULTIPLIER.get(symbol[-3:], self.PIP_MULTIPLIER['DEFAULT'])
            
            # Calculate spread in pips
            spread_pips = spread_price / pip_multiplier
            
            self.logger.debug(
                f"Spread calculation for {symbol}:\n"
                f"  Ask: {tick.ask:.5f}\n"
                f"  Bid: {tick.bid:.5f}\n"
                f"  Spread (price): {spread_price:.5f}\n"
                f"  Pip Value: {pip_multiplier}\n"
                f"  Spread (pips): {spread_pips:.2f}"
            )
            
            return spread_pips
            
        except Exception as e:
            self.logger.error(f"Spread calculation error for {symbol}: {str(e)}")
            return float('inf')
            
    def _validate_spread(self, symbol: str) -> bool:
        """
        Validate if current spread is within acceptable limits.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            bool: True if spread is acceptable, False otherwise
        """
        try:
            # Get max allowed spread from config (in pips)
            max_spread = self.config['trading']['risk_management']['max_spread'].get(
                symbol,
                self.config['trading']['risk_management']['max_spread'].get('default', 3.0)  # Default 3 pips
            )
            
            # Calculate current spread
            current_spread = self._calculate_spread_in_pips(symbol)
            
            # Check if spread is acceptable
            if current_spread > max_spread:
                self.logger.warning(
                    f"Spread validation failed for {symbol}:\n"
                    f"  Current Spread: {current_spread:.2f} pips\n"
                    f"  Maximum Allowed: {max_spread:.2f} pips"
                )
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Spread validation error for {symbol}: {str(e)}")
            return False

    def _check_risk_parameters(self, symbol: str, signal: TradingSignal) -> bool:
        """Check if signal meets risk parameters"""
        try:
            # Log risk check start
            self.logger.info(f"Starting risk parameter checks for {symbol}")
            
            # Check max positions
            total_positions = len(mt5.positions_get())
            if total_positions >= self.config['trading']['risk_management']['max_positions']:
                self.logger.warning(f"Max positions limit reached: {total_positions}")
                return False
                
            # Calculate position size
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                return False
                
            equity = account_info.equity
            risk_per_trade = self.config['trading']['risk_management']['max_risk_per_trade']
            risk_amount = equity * risk_per_trade
            
            self.logger.info(f"Risk calculation for {symbol}:")
            self.logger.info(f"  Account equity: {equity}")
            self.logger.info(f"  Risk per trade: {risk_per_trade}")
            self.logger.info(f"  Risk amount: {risk_amount}")
            
            # Check margin requirements
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Failed to get symbol info for {symbol}")
                return False
                
            margin_init = symbol_info.margin_initial
            margin_required = margin_init * 1.5  # Add 50% margin buffer
            
            if margin_required > account_info.margin_free:
                self.logger.warning(
                    f"Insufficient margin for {symbol}:\n"
                    f"  Required: {margin_required}\n"
                    f"  Available: {account_info.margin_free}"
                )
                return False
                
            self.logger.info(f"Margin check passed for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Risk parameter check error: {str(e)}")
            return False

    def _check_correlation(self, symbol: str) -> bool:
        """Check correlation with existing positions"""
        try:
            # Get all open positions
            positions = mt5.positions_get()
            if positions is None:
                return True  # No positions, correlation check passes
                
            # Get position symbols
            position_symbols = [pos.symbol for pos in positions]
            if not position_symbols:
                return True
                
            # Define correlated pairs
            correlations = {
                'EURUSD': ['GBPUSD', 'EURJPY', 'EURGBP'],
                'GBPUSD': ['EURUSD', 'GBPJPY', 'EURGBP'],
                'USDJPY': ['EURJPY', 'GBPJPY'],
                'EURJPY': ['EURUSD', 'USDJPY', 'GBPJPY'],
                'GBPJPY': ['GBPUSD', 'USDJPY', 'EURJPY'],
                'EURGBP': ['EURUSD', 'GBPUSD']
            }
            
            # Check if we already have correlated pairs
            if symbol in correlations:
                for corr_symbol in correlations[symbol]:
                    if corr_symbol in position_symbols:
                        self.logger.warning(f"Correlation conflict: {symbol} with {corr_symbol}")
                        return False
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking correlation: {str(e)}")
            return True  # Allow trade if correlation check fails

    def _prepare_trade_request(self, symbol: str, signal: TradingSignal) -> Dict:
        """Prepare trade request for MT5"""
        try:
            # Calculate position size
            position_size = self._calculate_position_size(symbol, signal)
            
            # Create trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position_size,
                "type": mt5.ORDER_TYPE_BUY if signal.direction == 1 else mt5.ORDER_TYPE_SELL,
                "price": signal.entry,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "deviation": 10,
                "magic": 234000,
                "comment": self._sanitize_comment("trade"),  # Use a simple comment that will be sanitized
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            return request
            
        except Exception as e:
            self.logger.error(f"Error preparing trade request: {str(e)}")
            raise

    def _find_support(self, df: pd.DataFrame) -> float:
        """Find support level"""
        try:
            if df.empty:
                return 0.0
            
            # Use recent lows to find support
            window = 20
            recent_lows = df['low'].rolling(window=window, center=True).min()
            support = recent_lows.iloc[-1]
            
            return float(support)
            
        except Exception as e:
            self.logger.error(f"Support calculation error: {str(e)}")
            return 0

    def _find_resistance(self, df: pd.DataFrame) -> float:
        """Find resistance level"""
        try:
            if df.empty:
                return 0.0
            
            # Use recent highs to find resistance
            window = 20
            recent_highs = df['high'].rolling(window=window, center=True).max()
            resistance = recent_highs.iloc[-1]
            
            return float(resistance)
            
        except Exception as e:
            self.logger.error(f"Resistance calculation error: {str(e)}")
            return 0

    def _calculate_indicators(self, df: pd.DataFrame, strategy_params: Dict) -> pd.DataFrame:
        """Calculate all technical indicators with enhanced combinations"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate ATR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=14).mean()
            
            # Calculate RSI
            delta = close.diff()
            period = strategy_params['indicators']['rsi']['period']
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            macd_params = strategy_params['indicators']['macd']
            exp1 = close.ewm(span=macd_params['fast'], adjust=False).mean()
            exp2 = close.ewm(span=macd_params['slow'], adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=macd_params['signal'], adjust=False).mean()
            
            # Calculate ADX and DMI
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm = plus_dm.where(plus_dm > 0, 0)
            minus_dm = minus_dm.where(minus_dm < 0, 0).abs()
            
            tr = high - low
            tr = pd.concat([tr, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
            
            plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14).mean() / tr.ewm(alpha=1/14, min_periods=14).mean())
            minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14).mean() / tr.ewm(alpha=1/14, min_periods=14).mean())
            
            df['di_plus'] = plus_di
            df['di_minus'] = minus_di
            
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            df['adx'] = dx.ewm(alpha=1/14, min_periods=14).mean()
            
            # Calculate Moving Averages if present
            if 'ma' in strategy_params['indicators']:
                ma_params = strategy_params['indicators']['ma']
                df['ma_fast'] = close.rolling(window=ma_params['fast']).mean()
                df['ma_slow'] = close.rolling(window=ma_params['slow']).mean()
            
            # Calculate Bollinger Bands if present
            if 'bollinger' in strategy_params['indicators']:
                bb_params = strategy_params['indicators']['bollinger']
                df['bb_middle'] = close.rolling(window=bb_params['period']).mean()
                std = close.rolling(window=bb_params['period']).std()
                df['bb_upper'] = df['bb_middle'] + (std * bb_params['std_dev'])
                df['bb_lower'] = df['bb_middle'] - (std * bb_params['std_dev'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return df

    def _monitor_performance(self) -> Dict:
        """Monitor trading performance metrics"""
        try:
            # Get recent trades
            from_date = datetime.now() - timedelta(days=30)
            trades = mt5.history_deals_get(from_date, datetime.now())
            
            if trades is None or len(trades) < self.config['trading']['performance_monitoring']['min_trades_for_stats']:
                return {'status': 'insufficient_data'}
            
            # Calculate metrics
            wins = sum(1 for trade in trades if trade.profit > 0)
            losses = sum(1 for trade in trades if trade.profit < 0)
            total_profit = sum(trade.profit for trade in trades if trade.profit > 0)
            total_loss = abs(sum(trade.profit for trade in trades if trade.profit < 0))
            
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate drawdown
            balance_curve = []
            current_balance = 0
            max_balance = 0
            max_drawdown = 0
            
            for trade in trades:
                current_balance += trade.profit
                max_balance = max(max_balance, current_balance)
                drawdown = (max_balance - current_balance) / max_balance if max_balance > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
                balance_curve.append(current_balance)
            
            # Check against thresholds
            monitoring = self.config['trading']['performance_monitoring']
            status = 'normal'
            
            if win_rate < monitoring['win_rate_threshold']:
                status = 'warning'
                self.logger.warning(f"Win rate below threshold: {win_rate:.2%}")
            
            if profit_factor < monitoring['profit_factor_threshold']:
                status = 'warning'
                self.logger.warning(f"Profit factor below threshold: {profit_factor:.2f}")
            
            if max_drawdown > monitoring['max_drawdown_threshold']:
                status = 'warning'
                self.logger.warning(f"Max drawdown above threshold: {max_drawdown:.2%}")
            
            return {
                'status': status,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'total_trades': len(trades),
                'net_profit': total_profit - total_loss
            }
            
        except Exception as e:
            self.logger.error(f"Error monitoring performance: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _is_in_trading_session(self) -> bool:
        """Check if current time is within configured trading sessions"""
        try:
            # Get current UTC time
            current_time = datetime.now(pytz.UTC)
            
            # Convert to configured timezone if specified
            if 'timezone' in self.config['trading']:
                tz = pytz.timezone(self.config['trading']['timezone'])
                current_time = current_time.astimezone(tz)
            
            # Get current time as hours and minutes
            current_hours = current_time.hour + current_time.minute / 60
            
            # Check each trading session
            for session, times in self.config['trading']['trading_sessions'].items():
                # Convert session times to hours
                start_h, start_m = map(int, times['start'].split(':'))
                end_h, end_m = map(int, times['end'].split(':'))
                
                session_start = start_h + start_m / 60
                session_end = end_h + end_m / 60
                
                # Handle sessions that cross midnight
                if session_end < session_start:
                    if current_hours >= session_start or current_hours <= session_end:
                        self.logger.debug(f"In trading session: {session}")
                        return True
                else:
                    if session_start <= current_hours <= session_end:
                        self.logger.debug(f"In trading session: {session}")
                        return True
            
            self.logger.debug("Not in any trading session")
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking trading session: {str(e)}")
            return False

    def _sanitize_comment(self, comment: str) -> str:
        """Sanitize comment to only include alphanumeric characters and underscores"""
        # Create a basic comment with just timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_comment = f"trade_{timestamp}"
        
        # Ensure it's alphanumeric and underscore only
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', base_comment)
        
        # Truncate to MT5's limit (31 characters)
        return sanitized[:31]

    async def _manage_open_positions(self):
        """Manage open positions with trailing stops"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return
                
            for position in positions:
                symbol_info = mt5.symbol_info(position.symbol)
                if symbol_info is None:
                    continue
                    
                # Get current price
                tick = mt5.symbol_info_tick(position.symbol)
                if tick is None:
                    continue
                    
                # Calculate profit in pips
                pip_size = 0.01 if 'JPY' in position.symbol else 0.0001
                if position.type == mt5.POSITION_TYPE_BUY:
                    profit_pips = (tick.bid - position.price_open) / pip_size
                else:
                    profit_pips = (position.price_open - tick.ask) / pip_size
                    
                # Update trailing stop if in profit
                if profit_pips >= 20:  # Start trailing after 20 pips profit
                    # Calculate new stop loss
                    atr = self._calculate_atr(position.symbol)
                    trail_points = atr * 2  # Trail by 2x ATR
                    
                    if position.type == mt5.POSITION_TYPE_BUY:
                        new_sl = tick.bid - trail_points
                        if new_sl > position.sl and new_sl - position.sl > symbol_info.point:
                            self._modify_position_sl(position, new_sl)
                    else:
                        new_sl = tick.ask + trail_points
                        if new_sl < position.sl and position.sl - new_sl > symbol_info.point:
                            self._modify_position_sl(position, new_sl)
                            
                # Check for position closure based on indicators
                if self._should_close_position(position):
                    self._close_position(position)
                    
        except Exception as e:
            self.logger.error(f"Error managing positions: {str(e)}")
            
    def _calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate ATR for a symbol"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, period + 1)
            if rates is None:
                return 0.0
                
            df = pd.DataFrame(rates)
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            return df['tr'].mean()
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return 0.0
            
    def _modify_position_sl(self, position, new_sl: float):
        """Modify position stop loss"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position.ticket,
                "sl": new_sl,
                "tp": position.tp
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Failed to modify SL: {result.comment}")
                
        except Exception as e:
            self.logger.error(f"Error modifying SL: {str(e)}")
            
    def _should_close_position(self, position) -> bool:
        """Check if position should be closed based on indicators"""
        try:
            # Get latest indicator values
            rates = mt5.copy_rates_from_pos(position.symbol, mt5.TIMEFRAME_M5, 0, 50)
            if rates is None:
                return False
                
            df = pd.DataFrame(rates)
            df = self._calculate_indicators(df, self.config['trading']['strategy_params']['trend'])
            
            current = df.iloc[-1]
            
            # Close long positions
            if position.type == mt5.POSITION_TYPE_BUY:
                if (current['trend_direction'] == -1 and  # Trend reversed
                    current['rsi'] > 70 and  # Overbought
                    current['macd'] < current['macd_signal']):  # MACD crossed down
                    return True
                    
            # Close short positions
            else:
                if (current['trend_direction'] == 1 and  # Trend reversed
                    current['rsi'] < 30 and  # Oversold
                    current['macd'] > current['macd_signal']):  # MACD crossed up
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking position closure: {str(e)}")
            return False
            
    def _close_position(self, position):
        """Close an open position"""
        try:
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return
                
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "type": mt5.ORDER_TYPE_BUY if position.type == mt5.POSITION_TYPE_SELL else mt5.ORDER_TYPE_SELL,
                "position": position.ticket,
                "volume": position.volume,
                "price": tick.ask if position.type == mt5.POSITION_TYPE_SELL else tick.bid,
                "deviation": 20,
                "magic": 234000,
                "comment": self._sanitize_comment("close_position"),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Failed to close position: {result.comment}")
            else:
                self.logger.info(f"Position closed: {position.ticket}")
                
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
