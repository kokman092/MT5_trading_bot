from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime
import MetaTrader5 as mt5
from ..market_data import MarketDataFetcher
from ..strategies.base_strategy import BaseStrategy
from ..deployment.error_handler import ErrorHandler
from ..deployment.health_monitor import HealthMonitor
from ..deployment.scaling_manager import ScalingManager

class TradingFlow:
    def __init__(self, config: Dict, strategy_class):
        self.config = config
        self.strategy = strategy_class(config)
        self.market_data = MarketDataFetcher(config['SYMBOL'], config['TIMEFRAME'])
        self.error_handler = ErrorHandler(config)
        self.health_monitor = HealthMonitor(config)
        self.scaling_manager = ScalingManager(config)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup custom logger for trading flow"""
        logger = logging.getLogger('trading_flow')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('logs/trading_flow.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    @error_handler.error_handler_decorator("market_data_fetch")
    async def fetch_market_data(self) -> Optional[Dict]:
        """Step 1: Fetch market data and indicators"""
        self.logger.info("Fetching market data...")
        
        # Get latest market data
        data = await self.market_data.get_historical_data(
            lookback=self.config['LOOKBACK_PERIOD']
        )
        if data is None:
            raise ValueError("Failed to fetch market data")
            
        # Calculate technical indicators
        indicators = await self.market_data.calculate_indicators(data)
        
        # Log market conditions
        self.logger.info(f"Current price: {data['close'].iloc[-1]}")
        self.logger.info(f"Market indicators calculated: {list(indicators.keys())}")
        
        return {
            'data': data,
            'indicators': indicators,
            'timestamp': datetime.now()
        }
        
    @error_handler.error_handler_decorator("market_analysis")
    async def analyze_market(self, market_data: Dict) -> Optional[Dict]:
        """Step 2: Analyze market and generate signals"""
        self.logger.info("Analyzing market conditions...")
        
        # Get trading signals
        signal = self.strategy.analyze(
            market_data['data'],
            market_data['indicators']
        )
        
        if signal:
            self.logger.info(f"Signal generated: {signal['action']}")
            # Validate signal
            if await self._validate_signal(signal):
                return signal
                
        return None
        
    async def _validate_signal(self, signal: Dict) -> bool:
        """Validate trading signal"""
        try:
            # Check trading hours
            if not self._is_trading_allowed():
                self.logger.info("Trading not allowed at current time")
                return False
                
            # Check risk limits
            if not await self._check_risk_limits(signal):
                self.logger.info("Risk limits would be exceeded")
                return False
                
            # Check technical validation
            if not self._validate_technical_conditions(signal):
                self.logger.info("Technical validation failed")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {str(e)}")
            return False
            
    @error_handler.error_handler_decorator("trade_execution")
    async def execute_trade(self, signal: Dict) -> Optional[Dict]:
        """Step 3: Execute trading order"""
        self.logger.info(f"Executing {signal['action']} order...")
        
        try:
            # Calculate position size
            position_size = self.scaling_manager.calculate_position_size(
                signal['price'],
                signal['stop_loss']
            )
            
            # Prepare order
            order = {
                'action': signal['action'],
                'symbol': self.config['SYMBOL'],
                'volume': position_size,
                'price': signal['price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'timestamp': datetime.now()
            }
            
            # Execute order
            if self.config.get('PAPER_TRADING', True):
                result = await self._execute_paper_trade(order)
            else:
                result = await self._execute_live_trade(order)
                
            if result:
                self.logger.info(f"Order executed: {result}")
                return result
                
        except Exception as e:
            self.logger.error(f"Trade execution error: {str(e)}")
            
        return None
        
    @error_handler.error_handler_decorator("position_monitoring")
    async def monitor_positions(self) -> None:
        """Step 4: Monitor open positions"""
        self.logger.info("Monitoring open positions...")
        
        try:
            positions = mt5.positions_get()
            if positions:
                for position in positions:
                    # Check stop loss and take profit
                    await self._check_exit_conditions(position)
                    
                    # Update trailing stop if enabled
                    if self.config.get('TRAILING_STOP_ENABLED', False):
                        await self._update_trailing_stop(position)
                        
        except Exception as e:
            self.logger.error(f"Position monitoring error: {str(e)}")
            
    @error_handler.error_handler_decorator("trade_logging")
    async def log_and_update(self, trade_result: Dict) -> None:
        """Step 5: Log trade and update account metrics"""
        self.logger.info("Updating trade records...")
        
        try:
            # Log trade details
            self._log_trade(trade_result)
            
            # Update account metrics
            await self._update_account_metrics()
            
            # Update performance metrics
            await self._update_performance_metrics(trade_result)
            
        except Exception as e:
            self.logger.error(f"Trade logging error: {str(e)}")
            
    async def run_trading_cycle(self):
        """Run complete trading cycle"""
        while True:
            try:
                # Step 1: Fetch market data
                market_data = await self.fetch_market_data()
                if not market_data:
                    continue
                    
                # Step 2: Analyze market
                signal = await self.analyze_market(market_data)
                if not signal:
                    continue
                    
                # Step 3: Execute trade
                trade_result = await self.execute_trade(signal)
                if trade_result:
                    # Step 4: Monitor positions
                    await self.monitor_positions()
                    
                    # Step 5: Log and update
                    await self.log_and_update(trade_result)
                    
                # Wait for next cycle
                await asyncio.sleep(self.config.get('CYCLE_INTERVAL', 1))
                
            except Exception as e:
                self.logger.error(f"Trading cycle error: {str(e)}")
                await asyncio.sleep(self.config.get('ERROR_WAIT_TIME', 5))
                
    def _is_trading_allowed(self) -> bool:
        """Check if trading is allowed based on time and conditions"""
        current_time = datetime.now().time()
        trading_hours = self.config.get('TRADING_HOURS', {})
        
        if trading_hours:
            start_time = datetime.strptime(trading_hours['start'], '%H:%M').time()
            end_time = datetime.strptime(trading_hours['end'], '%H:%M').time()
            return start_time <= current_time <= end_time
            
        return True
        
    async def _check_risk_limits(self, signal: Dict) -> bool:
        """Check if trade complies with risk management rules"""
        try:
            # Check maximum positions
            if len(mt5.positions_get()) >= self.config['MAX_POSITIONS']:
                return False
                
            # Check daily loss limit
            if await self._check_daily_loss_limit():
                return False
                
            # Check position size limits
            if not self._check_position_size_limits(signal):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Risk check error: {str(e)}")
            return False
            
    async def _update_account_metrics(self):
        """Update account performance metrics"""
        try:
            account_info = mt5.account_info()
            if account_info:
                self.scaling_manager.update_account_metrics(
                    balance=account_info.balance,
                    equity=account_info.equity,
                    profit=account_info.profit,
                    trades_count=len(mt5.positions_get())
                )
        except Exception as e:
            self.logger.error(f"Error updating account metrics: {str(e)}")
            
    def _log_trade(self, trade_result: Dict):
        """Log trade details to database and file"""
        try:
            # Log to file
            self.logger.info(f"Trade completed: {trade_result}")
            
            # Log to database (implement database logging here)
            pass
            
        except Exception as e:
            self.logger.error(f"Trade logging error: {str(e)}")
            
    async def _update_performance_metrics(self, trade_result: Dict):
        """Update performance metrics for monitoring"""
        try:
            # Update Prometheus metrics
            self.health_monitor.update_trade_metrics(trade_result)
            
            # Update scaling manager metrics
            await self.scaling_manager.update_performance_metrics(trade_result)
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")
