import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Type
from datetime import datetime, timedelta
import logging
from ..strategies.base_strategy import BaseStrategy
from ..market_data import MarketDataFetcher

class BacktestEngine:
    def __init__(self, config: Dict, strategy_class: Type[BaseStrategy]):
        self.config = config
        self.strategy_class = strategy_class
        self.strategy = strategy_class(config)
        self.market_data = MarketDataFetcher(config['SYMBOL'], config['TIMEFRAME'])
        self.initial_balance = config['INITIAL_BALANCE']
        self.commission = 0.0001  # 0.01% commission per trade
        
    async def run_backtest(self, start_date: datetime, end_date: datetime) -> Dict:
        """Run backtest over specified period"""
        try:
            # Fetch historical data
            historical_data = await self._fetch_historical_data(start_date, end_date)
            if historical_data is None or len(historical_data) < 50:
                raise ValueError("Insufficient historical data")
                
            # Initialize backtest variables
            balance = self.initial_balance
            positions = []
            trades = []
            equity_curve = []
            
            # Run through each candle
            for i in range(50, len(historical_data)):
                current_data = historical_data.iloc[:i+1]
                timestamp = current_data.index[-1]
                
                # Update open positions
                balance, positions, closed_trades = self._update_positions(
                    balance, positions, current_data, trades
                )
                trades.extend(closed_trades)
                
                # Generate new signals
                if len(positions) < self.config['MAX_POSITIONS']:
                    signal = self.strategy.analyze(current_data)
                    if signal:
                        position = self._open_position(signal, balance, current_data)
                        if position:
                            positions.append(position)
                            balance -= position['margin']
                            
                # Record equity
                equity = balance + sum(pos['unrealized_pnl'] for pos in positions)
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': equity,
                    'balance': balance
                })
                
            # Close any remaining positions
            final_data = historical_data.iloc[-1:]
            balance, _, closed_trades = self._update_positions(
                balance, positions, final_data, trades, force_close=True
            )
            trades.extend(closed_trades)
            
            # Calculate performance metrics
            metrics = self._calculate_metrics(trades, equity_curve, historical_data)
            
            return {
                'trades': trades,
                'equity_curve': equity_curve,
                'metrics': metrics
            }
            
        except Exception as e:
            logging.error(f"Backtest error: {str(e)}")
            return None
            
    async def _fetch_historical_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch and prepare historical data for backtesting"""
        # Calculate number of candles needed
        timeframe_minutes = int(self.config['TIMEFRAME'].replace('M', ''))
        total_minutes = (end_date - start_date).total_seconds() / 60
        num_candles = int(total_minutes / timeframe_minutes)
        
        data = await self.market_data.get_historical_data(num_candles)
        if data is None:
            return None
            
        # Filter data for date range
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        return data
        
    def _open_position(self, signal: Dict, balance: float, data: pd.DataFrame) -> Optional[Dict]:
        """Open new position based on signal"""
        current_price = data['close'].iloc[-1]
        position_size = self._calculate_position_size(signal, balance, current_price)
        
        if position_size is None:
            return None
            
        margin_required = position_size * current_price * self.config['MARGIN_REQUIREMENT']
        if margin_required > balance:
            return None
            
        return {
            'type': signal['action'],
            'entry_price': current_price,
            'size': position_size,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'margin': margin_required,
            'unrealized_pnl': 0
        }
        
    def _update_positions(
        self, 
        balance: float, 
        positions: List[Dict], 
        data: pd.DataFrame, 
        trades: List[Dict],
        force_close: bool = False
    ) -> tuple:
        """Update open positions and return closed trades"""
        current_price = data['close'].iloc[-1]
        closed_trades = []
        active_positions = []
        
        for position in positions:
            # Calculate unrealized P&L
            if position['type'] == 'BUY':
                pnl = (current_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - current_price) * position['size']
                
            position['unrealized_pnl'] = pnl
            
            # Check for stop loss or take profit
            should_close = force_close
            if not should_close:
                if position['type'] == 'BUY':
                    if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                        should_close = True
                else:
                    if current_price >= position['stop_loss'] or current_price <= position['take_profit']:
                        should_close = True
                        
            if should_close:
                # Close position and record trade
                commission = current_price * position['size'] * self.commission
                realized_pnl = position['unrealized_pnl'] - commission
                balance += position['margin'] + realized_pnl
                
                closed_trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'size': position['size'],
                    'pnl': realized_pnl,
                    'type': position['type'],
                    'commission': commission
                })
            else:
                active_positions.append(position)
                
        return balance, active_positions, closed_trades
        
    def _calculate_metrics(
        self, 
        trades: List[Dict], 
        equity_curve: List[Dict], 
        data: pd.DataFrame
    ) -> Dict:
        """Calculate performance metrics"""
        if not trades:
            return {}
            
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()
        
        # Calculate Sharpe Ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = returns - daily_rf
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate drawdown
        rolling_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': sum(t['pnl'] for t in trades if t['pnl'] > 0) / abs(sum(t['pnl'] for t in trades if t['pnl'] < 0)),
            'average_trade': total_pnl / total_trades,
            'average_win': sum(t['pnl'] for t in trades if t['pnl'] > 0) / winning_trades if winning_trades > 0 else 0,
            'average_loss': sum(t['pnl'] for t in trades if t['pnl'] < 0) / (total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0
        }
