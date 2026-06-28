from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from ..deployment.error_handler import ErrorHandler
from ..strategies.strategy_manager import StrategyManager
from ..risk.risk_manager import RiskManager

class Backtester:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('backtester')
        
        # Initialize components
        self.strategy_manager = StrategyManager(config)
        self.risk_manager = RiskManager(config)
        
        # Initialize tracking
        self.results = {
            'trades': [],
            'metrics': {},
            'equity_curve': []
        }
        
        # Set parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize backtesting parameters"""
        self.params = {
            'initial_balance': self.config.get('INITIAL_BALANCE', 10000),
            'commission': self.config.get('COMMISSION', 0.001),
            'slippage': self.config.get('SLIPPAGE', 0.0001),
            'leverage': self.config.get('LEVERAGE', 1),
            'margin_call': self.config.get('MARGIN_CALL', 0.5),
            'stop_out': self.config.get('STOP_OUT', 0.3)
        }
        
    async def run_backtest(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        strategy_name: str,
        timeframe: str = 'H1',
        **strategy_params
    ) -> Dict:
        """Run backtest for a specific strategy"""
        try:
            # Get historical data
            data = await self._get_historical_data(
                symbol,
                start_date,
                end_date,
                timeframe
            )
            
            if data.empty:
                raise ValueError("No historical data available")
                
            # Initialize account
            account = {
                'balance': self.params['initial_balance'],
                'equity': self.params['initial_balance'],
                'margin': 0,
                'free_margin': self.params['initial_balance'],
                'positions': {}
            }
            
            # Run simulation
            for index, row in data.iterrows():
                # Update market data
                market_data = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['tick_volume'],
                    'time': index
                }
                
                # Generate signals
                signals = await self.strategy_manager.generate_signals(
                    strategy_name,
                    market_data,
                    **strategy_params
                )
                
                # Process signals
                if signals:
                    await self._process_signals(
                        signals,
                        market_data,
                        account,
                        symbol
                    )
                    
                # Update positions
                await self._update_positions(account, market_data, symbol)
                
                # Record equity
                self.results['equity_curve'].append({
                    'timestamp': index,
                    'equity': account['equity'],
                    'balance': account['balance']
                })
                
            # Calculate final metrics
            metrics = await self._calculate_metrics()
            self.results['metrics'] = metrics
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Backtest error: {str(e)}")
            return {}
            
    async def _get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> pd.DataFrame:
        """Get historical data from MT5"""
        try:
            # Convert timeframe string to MT5 timeframe
            timeframes = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = timeframes.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Get rates
            rates = mt5.copy_rates_range(
                symbol,
                mt5_timeframe,
                start_date,
                end_date
            )
            
            if rates is None:
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Historical data fetch error: {str(e)}")
            return pd.DataFrame()
            
    async def _process_signals(
        self,
        signals: Dict,
        market_data: Dict,
        account: Dict,
        symbol: str
    ):
        """Process trading signals"""
        try:
            for signal in signals:
                # Check if we can trade
                if not await self._can_trade(account, signal, market_data):
                    continue
                    
                # Calculate position size
                size = await self.risk_manager.calculate_position_size(
                    signal,
                    account['balance'],
                    market_data
                )
                
                # Apply slippage
                price = market_data['close']
                if signal['direction'] == 'buy':
                    price *= (1 + self.params['slippage'])
                else:
                    price *= (1 - self.params['slippage'])
                    
                # Open position
                position = {
                    'symbol': symbol,
                    'type': signal['direction'],
                    'volume': size,
                    'price': price,
                    'sl': signal.get('stop_loss'),
                    'tp': signal.get('take_profit'),
                    'swap': 0,
                    'profit': 0,
                    'time': market_data['time']
                }
                
                # Update account
                margin = size * price / self.params['leverage']
                account['margin'] += margin
                account['free_margin'] = account['equity'] - account['margin']
                account['positions'][len(account['positions'])] = position
                
                # Record trade
                self.results['trades'].append({
                    'timestamp': market_data['time'],
                    'type': signal['direction'],
                    'price': price,
                    'volume': size,
                    'sl': signal.get('stop_loss'),
                    'tp': signal.get('take_profit')
                })
                
        except Exception as e:
            self.logger.error(f"Signal processing error: {str(e)}")
            
    async def _can_trade(
        self,
        account: Dict,
        signal: Dict,
        market_data: Dict
    ) -> bool:
        """Check if we can trade"""
        try:
            # Check margin level
            if account['margin'] > 0:
                margin_level = account['equity'] / account['margin']
                if margin_level <= self.params['margin_call']:
                    return False
                    
            # Check free margin
            required_margin = (
                signal.get('volume', 0) *
                market_data['close'] /
                self.params['leverage']
            )
            
            if required_margin > account['free_margin']:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Trade validation error: {str(e)}")
            return False
            
    async def _update_positions(
        self,
        account: Dict,
        market_data: Dict,
        symbol: str
    ):
        """Update open positions"""
        try:
            closed_positions = []
            
            for pos_id, position in account['positions'].items():
                if position['symbol'] != symbol:
                    continue
                    
                # Calculate profit
                price_diff = 0
                if position['type'] == 'buy':
                    price_diff = market_data['close'] - position['price']
                else:
                    price_diff = position['price'] - market_data['close']
                    
                position['profit'] = (
                    price_diff *
                    position['volume'] *
                    self.params['leverage']
                )
                
                # Check stop loss
                if position['sl']:
                    if (position['type'] == 'buy' and
                        market_data['low'] <= position['sl']):
                        closed_positions.append(pos_id)
                    elif (position['type'] == 'sell' and
                          market_data['high'] >= position['sl']):
                        closed_positions.append(pos_id)
                        
                # Check take profit
                if position['tp']:
                    if (position['type'] == 'buy' and
                        market_data['high'] >= position['tp']):
                        closed_positions.append(pos_id)
                    elif (position['type'] == 'sell' and
                          market_data['low'] <= position['tp']):
                        closed_positions.append(pos_id)
                        
            # Close positions
            for pos_id in closed_positions:
                position = account['positions'][pos_id]
                account['balance'] += position['profit']
                account['margin'] -= (
                    position['volume'] *
                    position['price'] /
                    self.params['leverage']
                )
                del account['positions'][pos_id]
                
            # Update equity
            account['equity'] = account['balance']
            for position in account['positions'].values():
                account['equity'] += position['profit']
                
            # Check stop out
            if (account['margin'] > 0 and
                account['equity'] / account['margin'] <= self.params['stop_out']):
                # Close all positions
                for pos_id, position in account['positions'].items():
                    account['balance'] += position['profit']
                account['positions'] = {}
                account['margin'] = 0
                
        except Exception as e:
            self.logger.error(f"Position update error: {str(e)}")
            
    async def _calculate_metrics(self) -> Dict:
        """Calculate backtest metrics"""
        try:
            if not self.results['trades']:
                return {}
                
            trades_df = pd.DataFrame(self.results['trades'])
            equity_df = pd.DataFrame(self.results['equity_curve'])
            
            # Basic metrics
            total_trades = len(trades_df)
            profitable_trades = len(trades_df[trades_df['profit'] > 0])
            
            # Profit metrics
            total_profit = equity_df['equity'].iloc[-1] - self.params['initial_balance']
            profit_factor = (
                abs(trades_df[trades_df['profit'] > 0]['profit'].sum()) /
                abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
                if len(trades_df[trades_df['profit'] < 0]) > 0
                else float('inf')
            )
            
            # Risk metrics
            equity_series = equity_df['equity']
            drawdown = (equity_series.cummax() - equity_series) / equity_series.cummax()
            max_drawdown = drawdown.max()
            
            # Return metrics
            returns = equity_series.pct_change()
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
            
            return {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': profitable_trades / total_trades,
                'total_profit': total_profit,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'return_pct': (total_profit / self.params['initial_balance']) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Metrics calculation error: {str(e)}")
            return {}
