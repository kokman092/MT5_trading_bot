from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
import MetaTrader5 as mt5
from ..deployment.error_handler import ErrorHandler
from ..strategies.strategy_manager import StrategyManager
from ..risk.risk_manager import RiskManager

class ForwardTester:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('forward_tester')
        
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
        """Initialize forward testing parameters"""
        self.params = {
            'demo_account': self.config.get('DEMO_ACCOUNT', True),
            'max_positions': self.config.get('MAX_POSITIONS', 5),
            'max_drawdown': self.config.get('MAX_DRAWDOWN', 0.1),
            'trade_timeout': self.config.get('TRADE_TIMEOUT', 60),
            'check_interval': self.config.get('CHECK_INTERVAL', 1)
        }
        
    async def start_forward_test(
        self,
        symbol: str,
        strategy_name: str,
        duration: int,
        **strategy_params
    ) -> Dict:
        """Start forward testing"""
        try:
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration)
            
            # Initialize MT5 demo account if needed
            if not await self._initialize_mt5():
                return {}
                
            self.logger.info(f"Starting forward test for {symbol}")
            
            while datetime.now() < end_time:
                # Get current market data
                market_data = await self._get_market_data(symbol)
                if not market_data:
                    continue
                    
                # Generate signals
                signals = await self.strategy_manager.generate_signals(
                    strategy_name,
                    market_data,
                    **strategy_params
                )
                
                # Process signals
                if signals:
                    await self._process_signals(signals, market_data, symbol)
                    
                # Update positions
                await self._update_positions(symbol)
                
                # Record equity
                await self._record_equity()
                
                # Check risk limits
                if not await self._check_risk_limits():
                    self.logger.warning("Risk limits exceeded, stopping test")
                    break
                    
                # Wait for next check
                await asyncio.sleep(self.params['check_interval'])
                
            # Calculate final metrics
            metrics = await self._calculate_metrics()
            self.results['metrics'] = metrics
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Forward test error: {str(e)}")
            return {}
            
    async def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                raise Exception("MT5 initialization failed")
                
            # Switch to demo account if needed
            if self.params['demo_account']:
                demo_accounts = mt5.account_info()
                if demo_accounts is None:
                    raise Exception("No demo account available")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 initialization error: {str(e)}")
            return False
            
    async def _get_market_data(self, symbol: str) -> Dict:
        """Get current market data"""
        try:
            # Get last tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {}
                
            # Get recent rates
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
            if rates is None:
                return {}
                
            df = pd.DataFrame(rates)
            
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': datetime.fromtimestamp(tick.time),
                'open': df['open'].iloc[-1],
                'high': df['high'].iloc[-1],
                'low': df['low'].iloc[-1],
                'close': df['close'].iloc[-1],
                'history': df
            }
            
        except Exception as e:
            self.logger.error(f"Market data fetch error: {str(e)}")
            return {}
            
    async def _process_signals(
        self,
        signals: Dict,
        market_data: Dict,
        symbol: str
    ):
        """Process trading signals"""
        try:
            for signal in signals:
                # Check position limits
                if len(mt5.positions_get()) >= self.params['max_positions']:
                    continue
                    
                # Calculate position size
                account_info = mt5.account_info()
                if account_info is None:
                    continue
                    
                size = await self.risk_manager.calculate_position_size(
                    signal,
                    account_info.balance,
                    market_data
                )
                
                # Prepare trade request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": size,
                    "type": mt5.ORDER_TYPE_BUY if signal['direction'] == 'buy'
                    else mt5.ORDER_TYPE_SELL,
                    "price": market_data['ask'] if signal['direction'] == 'buy'
                    else market_data['bid'],
                    "deviation": 10,
                    "magic": 234000,
                    "comment": "forward_test",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                if signal.get('stop_loss'):
                    request["sl"] = signal['stop_loss']
                if signal.get('take_profit'):
                    request["tp"] = signal['take_profit']
                    
                # Send order
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.logger.warning(f"Order failed: {result.comment}")
                    continue
                    
                # Record trade
                self.results['trades'].append({
                    'timestamp': market_data['time'],
                    'type': signal['direction'],
                    'price': result.price,
                    'volume': size,
                    'sl': signal.get('stop_loss'),
                    'tp': signal.get('take_profit'),
                    'order_id': result.order
                })
                
        except Exception as e:
            self.logger.error(f"Signal processing error: {str(e)}")
            
    async def _update_positions(self, symbol: str):
        """Update open positions"""
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                return
                
            for position in positions:
                # Check timeout
                if (datetime.now() - datetime.fromtimestamp(position.time)
                    ).total_seconds() > self.params['trade_timeout']:
                    # Close position
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": position.volume,
                        "type": mt5.ORDER_TYPE_SELL if position.type == 0
                        else mt5.ORDER_TYPE_BUY,
                        "position": position.ticket,
                        "price": mt5.symbol_info_tick(symbol).bid
                        if position.type == 0
                        else mt5.symbol_info_tick(symbol).ask,
                        "deviation": 10,
                        "magic": 234000,
                        "comment": "timeout",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        self.logger.warning(f"Position close failed: {result.comment}")
                        
        except Exception as e:
            self.logger.error(f"Position update error: {str(e)}")
            
    async def _record_equity(self):
        """Record current equity"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return
                
            self.results['equity_curve'].append({
                'timestamp': datetime.now(),
                'equity': account_info.equity,
                'balance': account_info.balance,
                'margin': account_info.margin
            })
            
        except Exception as e:
            self.logger.error(f"Equity recording error: {str(e)}")
            
    async def _check_risk_limits(self) -> bool:
        """Check if risk limits are exceeded"""
        try:
            if not self.results['equity_curve']:
                return True
                
            equity_df = pd.DataFrame(self.results['equity_curve'])
            
            # Calculate drawdown
            equity_series = equity_df['equity']
            drawdown = (equity_series.cummax() - equity_series) / equity_series.cummax()
            current_drawdown = drawdown.iloc[-1]
            
            return current_drawdown <= self.params['max_drawdown']
            
        except Exception as e:
            self.logger.error(f"Risk check error: {str(e)}")
            return False
            
    async def _calculate_metrics(self) -> Dict:
        """Calculate forward test metrics"""
        try:
            if not self.results['trades']:
                return {}
                
            trades_df = pd.DataFrame(self.results['trades'])
            equity_df = pd.DataFrame(self.results['equity_curve'])
            
            # Basic metrics
            total_trades = len(trades_df)
            profitable_trades = len(trades_df[trades_df['profit'] > 0])
            
            # Profit metrics
            initial_balance = equity_df['balance'].iloc[0]
            final_equity = equity_df['equity'].iloc[-1]
            total_profit = final_equity - initial_balance
            
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
                'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
                'total_profit': total_profit,
                'return_pct': (total_profit / initial_balance) * 100,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'avg_trade_duration': trades_df['duration'].mean()
                if 'duration' in trades_df.columns else 0
            }
            
        except Exception as e:
            self.logger.error(f"Metrics calculation error: {str(e)}")
            return {}
