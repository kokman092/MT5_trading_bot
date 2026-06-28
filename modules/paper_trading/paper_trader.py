from typing import Dict, Optional, List
import MetaTrader5 as mt5
from datetime import datetime
import asyncio
import logging
from ..market_data import MarketDataFetcher
from ..strategies.base_strategy import BaseStrategy

class PaperTrader:
    def __init__(self, config: Dict, strategy_class):
        self.config = config
        self.strategy = strategy_class(config)
        self.market_data = MarketDataFetcher(config['SYMBOL'], config['TIMEFRAME'])
        self.initial_balance = config['INITIAL_BALANCE']
        self.current_balance = self.initial_balance
        self.positions = []
        self.trades_history = []
        self.commission = 0.0001  # 0.01% commission per trade
        
    async def start(self):
        """Start paper trading"""
        logging.info("Starting paper trading session...")
        logging.info(f"Initial balance: ${self.initial_balance}")
        
        try:
            while True:
                await self._process_tick()
                await asyncio.sleep(1)  # Check every second
                
        except Exception as e:
            logging.error(f"Paper trading error: {str(e)}")
            
    async def _process_tick(self):
        """Process current market tick"""
        try:
            # Get current market data
            current_data = await self.market_data.get_historical_data(100)
            if current_data is None:
                return
                
            current_price = current_data['close'].iloc[-1]
            
            # Update open positions
            await self._update_positions(current_price)
            
            # Check for new trading opportunities
            if len(self.positions) < self.config['MAX_POSITIONS']:
                signal = self.strategy.analyze(current_data)
                if signal:
                    await self._open_position(signal, current_price)
                    
        except Exception as e:
            logging.error(f"Error processing tick: {str(e)}")
            
    async def _open_position(self, signal: Dict, current_price: float):
        """Open new paper trading position"""
        try:
            # Calculate position size
            risk_amount = self.current_balance * (self.config['RISK_PERCENTAGE'] / 100)
            price_distance = abs(current_price - signal['stop_loss'])
            position_size = risk_amount / price_distance
            
            # Check if we have enough balance
            margin_required = position_size * current_price * 0.01  # 1% margin requirement
            if margin_required > self.current_balance:
                logging.warning("Insufficient balance for trade")
                return
                
            # Create position
            position = {
                'type': signal['action'],
                'entry_price': current_price,
                'size': position_size,
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'entry_time': datetime.now(),
                'margin': margin_required
            }
            
            self.positions.append(position)
            self.current_balance -= margin_required
            
            logging.info(f"Opened {position['type']} position: {position['size']} units at {position['entry_price']}")
            
        except Exception as e:
            logging.error(f"Error opening position: {str(e)}")
            
    async def _update_positions(self, current_price: float):
        """Update open positions"""
        try:
            for position in self.positions[:]:  # Copy list to allow removal during iteration
                # Calculate unrealized P&L
                if position['type'] == 'BUY':
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) * position['size']
                    
                # Check for stop loss or take profit
                should_close = False
                if position['type'] == 'BUY':
                    if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                        should_close = True
                else:
                    if current_price >= position['stop_loss'] or current_price <= position['take_profit']:
                        should_close = True
                        
                if should_close:
                    await self._close_position(position, current_price)
                    
        except Exception as e:
            logging.error(f"Error updating positions: {str(e)}")
            
    async def _close_position(self, position: Dict, current_price: float):
        """Close paper trading position"""
        try:
            # Calculate P&L
            if position['type'] == 'BUY':
                pnl = (current_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - current_price) * position['size']
                
            # Apply commission
            commission = current_price * position['size'] * self.commission
            pnl -= commission
            
            # Update balance
            self.current_balance += position['margin'] + pnl
            
            # Record trade
            trade = {
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'type': position['type'],
                'size': position['size'],
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'pnl': pnl,
                'commission': commission
            }
            
            self.trades_history.append(trade)
            self.positions.remove(position)
            
            logging.info(f"Closed position with P&L: ${pnl:.2f}")
            logging.info(f"Current balance: ${self.current_balance:.2f}")
            
        except Exception as e:
            logging.error(f"Error closing position: {str(e)}")
            
    def get_statistics(self) -> Dict:
        """Get paper trading statistics"""
        try:
            if not self.trades_history:
                return {}
                
            total_trades = len(self.trades_history)
            winning_trades = len([t for t in self.trades_history if t['pnl'] > 0])
            total_pnl = sum(t['pnl'] for t in self.trades_history)
            
            return {
                'initial_balance': self.initial_balance,
                'current_balance': self.current_balance,
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'average_trade': total_pnl / total_trades if total_trades > 0 else 0,
                'open_positions': len(self.positions)
            }
            
        except Exception as e:
            logging.error(f"Error calculating statistics: {str(e)}")
            return {}
