import logging
from typing import Dict, List, Optional
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json


class PositionManager:
    def __init__(self, config: Dict):
        """Initialize position manager with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Load risk management configuration
        if 'risk_management' not in config:
            raise ValueError("Missing risk_management section in config")

        self.risk_config = config['risk_management']

        # Initialize position tracking
        self.positions = {}
        self.last_update = datetime.now()

        # Load risk limits
        self.risk_limits = self.risk_config.get('risk_limits', {
            'max_open_positions': 5,
            'max_daily_trades': 20,
            'max_position_size': 1.0,
            'max_total_risk': 0.05,
            'max_correlation': 0.7,
            'drawdown_limit': 0.1,
            'max_leverage': 30,
            'min_margin_level': 200,
            'max_spread_multiplier': 1.5
        })

        # Initialize risk metrics
        self.daily_trades = 0
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.recovery_mode = False
        self.last_reset = datetime.now()

        self.logger.info(
            "Position manager initialized with risk configuration")

        self.trading_config = config.get('trading', {})
        self.position_config = self.trading_config.get(
            'position_management', {})

        self.positions_per_symbol = {}
        self.trailing_stops = {}
        self.trade_history = {}

    async def manage_positions(self):
        """Main position management loop"""
        try:
            # Update positions
            await self.update_positions()

            # Get account info
            account = mt5.account_info()
            if account is None:
                return

            # Check if we need to enter recovery mode
            self._update_recovery_mode(account)

            # Get all current positions
            positions = mt5.positions_get()
            if positions is None:
                return
                
            # Process each position
            for position in positions:
                await self._check_close_conditions(position)

            # Check correlation exposure
            await self.check_correlated_exposure()

            # Consolidate positions if needed
            if len(positions) > self.config['risk_management']['position_limits']['max_positions']:
                await self.consolidate_positions()

        except Exception as e:
            self.logger.error(f"Error managing positions: {str(e)}")

    async def update_positions(self):
        """Update position information and manage trailing stops"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return

            current_positions = {}

            for position in positions:
                # Update position info
                position_info = {
                    'ticket': position.ticket,
                    'symbol': position.symbol,
                    'type': position.type,
                    'volume': position.volume,
                    'open_price': position.price_open,
                    'current_price': position.price_current,
                    'sl': position.sl,
                    'tp': position.tp,
                    'profit': position.profit,
                    'swap': position.swap,
                    'time': position.time
                }

                # Get market data
                symbol_data = await self._get_market_data(position.symbol)
                if symbol_data is not None:
                    # Update trailing stop
                    if self.config['trading_parameters']['exit_parameters']['trailing_stop']['enabled']:
                        self._update_trailing_stop(position)

                    # Check break-even
                    if self.config['trading_parameters']['exit_parameters']['break_even']['enabled']:
                        await self._update_breakeven_stop(position)

                    # Check partial profit taking
                    if self.config['trading_parameters']['exit_parameters']['partial_exit']['enabled']:
                        await self._check_partial_profit(position)

                current_positions[position.ticket] = position_info

            # Update positions dictionary
            self.positions = current_positions
            self.last_update = datetime.now()
                
        except Exception as e:
            import traceback
            self.logger.error(f"Error updating positions: {str(e)}\n{traceback.format_exc()}")
            
    async def _check_close_conditions(self, position):
        """Check if position should be closed based on various conditions"""
        # Python-driven auto-closures are disabled. Positions will only close when hitting SL/TP in MT5.
        return

    def _update_trailing_stop(self, position):
        """Update trailing stop loss based on R-multiples"""
        try:
            # Get trailing stop parameters (in R-multiples)
            trailing_config = self.config.get('trading_parameters', {}).get('exit_parameters', {}).get('trailing_stop', {})
            activation_r = trailing_config.get('activation', 0.5)
            step_r = trailing_config.get('step', 0.1)

            # Calculate Risk (R)
            if position.sl == 0:
                return
            risk = abs(position.price_open - position.sl)
            if risk == 0:
                return
                
            # Calculate current Reward
            current_price = position.price_current
            if position.type == 0:  # Buy
                reward = current_price - position.price_open
            else:  # Sell
                reward = position.price_open - current_price

            # Check if trailing stop should be activated
            if reward > (activation_r * risk):
                # Calculate new stop loss level based on step
                # We want to trail behind the current price by (activation_r - step_r) * risk
                # But a simpler way is: lock in (reward - step_r*risk)
                
                trail_distance = step_r * risk
                
                if position.type == 0:  # Buy position
                    new_sl = current_price - trail_distance
                    if position.sl < new_sl < current_price:
                        self._modify_sl_tp(position.ticket, new_sl, position.tp)
                        
                else:  # Sell position
                    new_sl = current_price + trail_distance
                    if position.sl > new_sl > current_price or position.sl == 0:
                        self._modify_sl_tp(position.ticket, new_sl, position.tp)

        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {str(e)}")

    async def check_correlated_exposure(self):
        """Check and manage correlated positions"""
        try:
            positions = mt5.positions_get()
            if positions is None or len(positions) < 2:
                return

            # Calculate correlation matrix
            symbols = list(set(pos.symbol for pos in positions))
            correlation_matrix = await self._calculate_correlation_matrix(symbols)

            # Check for high correlations
            max_correlation = self.config['risk_management']['position_limits']['max_correlation']
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if correlation_matrix[i][j] > max_correlation:
                        # Close least profitable position
                        pos1 = [p for p in positions if p.symbol == symbol1]
                        pos2 = [p for p in positions if p.symbol == symbol2]

                        if pos1[0].profit < pos2[0].profit:
                            await self._close_position(pos1[0].ticket)
                        else:
                            await self._close_position(pos2[0].ticket)

        except Exception as e:
            self.logger.error(f"Error checking correlated exposure: {str(e)}")

    async def _calculate_correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """Calculate correlation matrix for given symbols"""
        try:
            prices = {}
            for symbol in symbols:
                data = await self._get_market_data(symbol)
                if data is not None:
                    prices[symbol] = data['close']

            if not prices:
                return np.array([])

            # Create price DataFrame
            df = pd.DataFrame(prices)

            # Calculate correlation matrix
            return df.corr().values

        except Exception as e:
            self.logger.error(
                f"Error calculating correlation matrix: {str(e)}")
            return np.array([])

    async def close_all_positions(self):
        """Close all open positions"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return

            for position in positions:
                try:
                    await self._close_position(position.ticket)
                    self.logger.info(
                        f"Closed position {position.ticket} during shutdown")
                except Exception as e:
                    self.logger.error(
                        f"Error closing position {position.ticket} during shutdown: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error closing all positions: {str(e)}")

    def _get_filling_mode(self, symbol: str) -> int:
        """Get the supported filling mode for a symbol dynamically"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return mt5.ORDER_FILLING_IOC
            
        filling_mode = symbol_info.filling_mode
        # 1 = SYMBOL_FILLING_FOK, 2 = SYMBOL_FILLING_IOC
        if filling_mode & 1:
            return mt5.ORDER_FILLING_FOK
        elif filling_mode & 2:
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURN

    async def _close_position(self, ticket: int) -> bool:
        """Close a position"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False

            position = position[0]

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "volume": position.volume,
                "deviation": 10,
                "magic": 234000,
                "comment": "close_by_manager",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._get_filling_mode(position.symbol),
            }

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(
                    f"Failed to close position - Error: {result.retcode}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return False

    async def _close_worst_positions(self):
        """Close positions with the worst performance"""
        try:
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                return

            # Sort positions by profit
            sorted_positions = sorted(positions, key=lambda x: x.profit)

            # Close worst performing positions until within limits
            max_positions = self.config['risk_management']['position_limits']['max_positions']
            positions_to_close = len(positions) - max_positions

            if positions_to_close > 0:
                for position in sorted_positions[:positions_to_close]:
                    await self._close_position(position.ticket)

        except Exception as e:
            self.logger.error(f"Error closing worst positions: {str(e)}")
            return False

        return True

    async def _reduce_directional_exposure(self, long_count: int, short_count: int):
        """Reduce exposure in the dominant direction"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return
                
            # Sort positions by profit (close worst performing first)
            positions = sorted(positions, key=lambda x: x.profit)
            
            # Determine which direction to reduce
            reduce_longs = long_count > short_count
            target_count = min(long_count, short_count)
            
            for pos in positions:
                if (reduce_longs and pos.type == 0) or (not reduce_longs and pos.type == 1):
                    await self._close_position(pos.ticket)
                    if reduce_longs:
                        long_count -= 1
                    else:
                        short_count -= 1
                        
                    if long_count <= target_count and short_count <= target_count:
                        break
                        
        except Exception as e:
            self.logger.error(f"Error reducing directional exposure: {str(e)}")
            
    def _update_recovery_mode(self, account):
        """Update recovery mode status based on performance"""
        try:
            risk_config = self.config.get(
                'trading', {}).get('risk_management', {})
            recovery_config = risk_config.get('recovery_mode', {
                'enabled': False,
                'threshold_percent': 5.0
            })

            if not recovery_config['enabled']:
                return
                
            daily_loss = (account.balance - account.equity) / \
                account.balance * 100
            threshold = recovery_config['threshold_percent']

            if daily_loss > threshold and not self.recovery_mode:
                self.logger.warning(
                    f"Entering recovery mode - Daily loss: {daily_loss:.2f}%")
                self.recovery_mode = True
            elif daily_loss <= 0 and self.recovery_mode:
                self.logger.info("Exiting recovery mode - Losses recovered")
                self.recovery_mode = False
                
        except Exception as e:
            self.logger.error(f"Error updating recovery mode: {str(e)}")
            
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta > 0) * delta
            loss = (delta < 0) * -delta
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return 50.0
            
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate current volatility"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            
            return float(atr.iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0
            
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        try:
            # Calculate EMAs
            ema20 = df['close'].ewm(span=20).mean()
            ema50 = df['close'].ewm(span=50).mean()
            
            # Calculate ADX-like trend strength
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Directional movement
            plus_dm = np.append(0, np.maximum(high[1:] - high[:-1], 0))
            minus_dm = np.append(0, np.maximum(low[:-1] - low[1:], 0))
            
            # Smooth the movements
            tr = self._calculate_volatility(df)
            plus_di = np.mean(plus_dm[-14:]) / tr
            minus_di = np.mean(minus_dm[-14:]) / tr
            
            # Combine indicators
            ema_trend = 1 if ema20.iloc[-1] > ema50.iloc[-1] else -1
            di_trend = plus_di - minus_di
            
            # Normalize to 0-1 range
            trend_strength = (abs(di_trend) + (0.5 * abs(ema_trend))) / 1.5
            
            return float(min(1.0, max(0.0, trend_strength)))
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.5
            
    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate price momentum"""
        try:
            # Calculate rate of change
            roc = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
            
            # Normalize to -1 to 1 range
            momentum = np.tanh(roc / 10)  # Divide by 10 to scale appropriately
            
            return float(momentum)
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum: {str(e)}")
            return 0.0
            
    def _calculate_risk_reward(self, position) -> float:
        """Calculate current risk/reward ratio"""
        try:
            if position.sl == 0:
                return 0.0
                
            risk = abs(position.price_open - position.sl)
            if risk == 0:
                return 0.0
                
            current_price = mt5.symbol_info_tick(
                position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask
                
            if position.type == 0:  # Buy
                reward = current_price - position.price_open
            else:  # Sell
                reward = position.price_open - current_price
            
            return reward / risk
            
        except Exception as e:
            self.logger.error(f"Error calculating R/R ratio: {str(e)}")
            return 0.0
            
    def _is_session_ending(self, symbol: str) -> bool:
        """Check if current session is ending soon"""
        try:
            current_time = datetime.now().time()
            # 15 minutes before session end
            buffer_minutes = timedelta(minutes=15)
            
            for session in self.config['trading']['sessions'].values():
                applies = True
                if 'pairs' in session:
                    applies = symbol in session['pairs']
                
                if applies:
                    end_time = datetime.strptime(
                        session['end'], '%H:%M').time()
                    end_dt = datetime.combine(datetime.now().date(), end_time)
                    current_dt = datetime.combine(
                        datetime.now().date(), current_time)
                    
                    if 0 <= (end_dt - current_dt).total_seconds() <= buffer_minutes.total_seconds():
                        return True
                        
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking session end: {str(e)}")
            return False
            
    def _modify_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        """Modify stop loss and take profit levels"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": sl,
                "tp": tp
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(
                    f"Failed to modify SL/TP - Error: {result.retcode}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error modifying SL/TP: {str(e)}")
            return False
            
    def _check_trailing_stop(self, position, tick):
        """Check if trailing stop is hit"""
        try:
            # Get trailing stop settings
            # Start trailing after 200 points profit
            trailing_start = self.config.get('trailing_start_points', 200)
            trailing_step = self.config.get(
                'trailing_step_points', 50)    # Move SL every 50 points

            # Calculate current profit in points
            point = mt5.symbol_info(position.symbol).point
            if position.type == 0:  # Buy
                current_profit_points = (
                    tick.bid - position.price_open) / point
            else:  # Sell
                current_profit_points = (
                    position.price_open - tick.ask) / point

            # Check if trailing stop is hit
            return current_profit_points <= -trailing_start or current_profit_points >= trailing_start
            
        except Exception as e:
            self.logger.error(f"Error checking trailing stop: {str(e)}")
            return False
            
    async def close_profitable_positions(self):
        """Close positions that have reached target profit"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return
                
            for pos in positions:
                # Close if profit is more than 1% of position value
                if pos.profit > (pos.volume * pos.price_open * 0.01):
                    await self._close_position(pos.ticket)
                    self.logger.info(
                        f"Closing profitable position - "
                        f"Symbol: {pos.symbol}, "
                        f"Profit: {pos.profit:.2f}, "
                        f"Volume: {pos.volume}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error closing profitable positions: {str(e)}")

    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on recovery mode"""
        if self.recovery_mode:
            return self.config['trading']['risk_management']['recovery_mode']['position_size_reduce']
        return 1.0
            
    def can_open_position(self, symbol: str) -> bool:
        """Check if we can open a new position for this symbol"""
        max_positions = self.config.get('max_positions_per_symbol', 2)
        current_positions = self.positions_per_symbol.get(symbol, 0)
        return current_positions < max_positions
        
    def _is_restricted_time(self) -> bool:
        """Check if current time is in restricted trading hours"""
        try:
            current_time = datetime.now().strftime('%H:%M')
            restricted_hours = self.config['trading']['risk_management']['time_filters']['restricted_hours']
            
            for period in restricted_hours:
                start, end = period.split('-')
                if start <= current_time <= end:
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking restricted time: {str(e)}")
            return False

    async def consolidate_positions(self):
        """Consolidate multiple positions in the same symbol"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return
                
            # Group positions by symbol and direction
            position_groups = {}
            for pos in positions:
                key = (pos.symbol, pos.type)
                if key not in position_groups:
                    position_groups[key] = []
                position_groups[key].append(pos)
                
            # Process each group
            for (symbol, pos_type), group in position_groups.items():
                if len(group) <= 1:
                    continue
                    
                # Calculate weighted average entry
                total_volume = sum(pos.volume for pos in group)
                avg_entry = sum(pos.price_open *
                                pos.volume for pos in group) / total_volume
                
                # Calculate total profit
                total_profit = sum(pos.profit for pos in group)
                
                # Check if consolidation is needed
                if len(group) > self.config['trading']['position_management']['consolidation']['max_positions_per_symbol']:
                    self.logger.info(
                        f"Consolidating {len(group)} positions for {symbol}")
                    
                    # Close all positions
                    for pos in group:
                        await self._close_position(pos.ticket)
                        
                    # Open new consolidated position
                    if total_profit >= 0:  # Only reopen if in profit
                        await self._open_consolidated_position(symbol, pos_type, total_volume, avg_entry)
                        
        except Exception as e:
            self.logger.error(f"Error consolidating positions: {str(e)}")
            
    async def _open_consolidated_position(self, symbol: str, pos_type: int, volume: float, entry_price: float):
        """Open a new consolidated position"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return
                
            # Calculate new SL/TP based on ATR
            atr = self._calculate_atr(symbol)
            
            if pos_type == 0:  # Buy
                sl = tick.bid - (3.0 * atr)
                tp = tick.bid + (5.0 * atr)
            else:  # Sell
                sl = tick.ask + (3.0 * atr)
                tp = tick.ask - (5.0 * atr)
                
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": pos_type,
                "price": tick.ask if pos_type == 0 else tick.bid,
                "sl": sl,
                "tp": tp,
                "deviation": 10,
                "magic": 234000,
                "comment": "consolidated_position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._get_filling_mode(symbol),
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(
                    f"Failed to open consolidated position: {result.comment}")
                
        except Exception as e:
            self.logger.error(f"Error opening consolidated position: {str(e)}")
            
    def _calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            rates = mt5.copy_rates_from_pos(
                symbol, mt5.TIMEFRAME_H1, 0, period + 1)
            if rates is None or len(rates) < period + 1:
                return 0
                
            rates = np.array(rates)
            high = rates['high']
            low = rates['low']
            close = rates['close']
            
            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = np.mean(tr)
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return 0
            
    async def _get_market_data(self, symbol: str):
        """Get market data for a given symbol"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 50)
            if rates is None:
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df

        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return None

    async def _update_breakeven_stop(self, position):
        """Move stop loss to breakeven when profit target is reached"""
        try:
            # Get breakeven stop settings (in R-multiples)
            breakeven_config = self.config.get('trading_parameters', {}).get('exit_parameters', {}).get('break_even', {})
            activation_r = breakeven_config.get('activation', 1.0)
            
            # Calculate Risk (R)
            if position.sl == 0:
                return
            risk = abs(position.price_open - position.sl)
            if risk == 0:
                return
                
            # Calculate current Reward
            current_price = position.price_current
            if position.type == 0:  # Buy
                reward = current_price - position.price_open
            else:  # Sell
                reward = position.price_open - current_price
                
            # If current reward >= activation_r * risk, move SL to breakeven
            if reward >= (activation_r * risk):
                # Calculate breakeven price (entry price + spread buffer)
                point = mt5.symbol_info(position.symbol).point
                spread = mt5.symbol_info(position.symbol).spread * point
                
                if position.type == 0:  # Buy
                    breakeven_price = position.price_open + (spread * 1.5)
                    if position.sl < breakeven_price < current_price:
                        self._modify_sl_tp(position.ticket, breakeven_price, position.tp)
                        self.logger.info(f"Moved SL to breakeven for Buy position {position.ticket}")
                else:  # Sell
                    breakeven_price = position.price_open - (spread * 1.5)
                    if position.sl > breakeven_price > current_price:
                        self._modify_sl_tp(position.ticket, breakeven_price, position.tp)
                        self.logger.info(f"Moved SL to breakeven for Sell position {position.ticket}")

        except Exception as e:
            self.logger.error(f"Error updating breakeven stop: {str(e)}")

    async def _check_partial_profit(self, position):
        """Check partial profit taking conditions"""
        try:
            # Get partial profit taking settings
            positions = mt5.positions_get()
            if not positions:
                return
                
            # Calculate correlations between symbols
            symbols = list(set(pos.symbol for pos in positions))
            if len(symbols) < 2:
                return
                
            # Get price data
            data = {}
            for symbol in symbols:
                rates = mt5.copy_rates_from_pos(
                    symbol, mt5.TIMEFRAME_H1, 0, 100)
                if rates is not None:
                    data[symbol] = np.array([rate['close'] for rate in rates])
                    
            # Calculate correlation matrix
            corr_matrix = {}
            for i, sym1 in enumerate(symbols):
                for sym2 in symbols[i+1:]:
                    if sym1 in data and sym2 in data:
                        correlation = np.corrcoef(data[sym1], data[sym2])[0, 1]
                        corr_matrix[(sym1, sym2)] = correlation
                        
            # Check for highly correlated positions
            # Check for highly correlated positions
            pos_management = self.config.get('trading', {}).get('position_management', {}).get('consolidation', {})
            threshold = pos_management.get('correlation_threshold', 0.8)
            max_correlated = pos_management.get('max_positions_correlated', 2)
            
            # Group correlated positions
            correlated_groups = []
            for (sym1, sym2), corr in corr_matrix.items():
                if abs(corr) >= threshold:
                    # Find or create group
                    added = False
                    for group in correlated_groups:
                        if sym1 in group or sym2 in group:
                            group.update([sym1, sym2])
                            added = True
                            break
                    if not added:
                        correlated_groups.append({sym1, sym2})
                        
            # Manage correlated positions
            for group in correlated_groups:
                group_positions = [
                    pos for pos in positions if pos.symbol in group]
                if len(group_positions) > max_correlated:
                    self.logger.warning(
                        f"Too many correlated positions in group: {group}")
                    # Sort by profit and keep the best performing ones
                    group_positions.sort(key=lambda x: x.profit, reverse=True)
                    for pos in group_positions[max_correlated:]:
                        await self._close_position(pos.ticket)
                        
        except Exception as e:
            self.logger.error(f"Error checking correlated exposure: {str(e)}")
