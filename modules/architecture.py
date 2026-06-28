from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
import asyncio
import MetaTrader5 as mt5

# Base Module Interface
class TradingModule(ABC):
    """Base interface for all trading modules"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the module"""
        pass
        
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup module resources"""
        pass
        
    @abstractmethod
    async def validate_health(self) -> Dict:
        """Check module health status"""
        pass

# Data Module Interface
class DataModule(TradingModule):
    """Interface for market data handling"""
    
    @abstractmethod
    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict:
        """Fetch historical market data"""
        pass
        
    @abstractmethod
    async def subscribe_to_live_feed(
        self,
        symbol: str,
        callback: callable
    ) -> bool:
        """Subscribe to live market data"""
        pass
        
    @abstractmethod
    async def unsubscribe_from_live_feed(self, symbol: str) -> bool:
        """Unsubscribe from live market data"""
        pass

# Strategy Module Interface
class StrategyModule(TradingModule):
    """Interface for trading strategies"""
    
    @abstractmethod
    async def analyze_market(self, market_data: Dict) -> Dict:
        """Analyze market conditions"""
        pass
        
    @abstractmethod
    async def generate_signals(self, analysis: Dict) -> List[Dict]:
        """Generate trading signals"""
        pass
        
    @abstractmethod
    async def validate_signals(self, signals: List[Dict]) -> List[Dict]:
        """Validate generated signals"""
        pass

# Execution Module Interface
class ExecutionModule(TradingModule):
    """Interface for order execution"""
    
    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None
    ) -> Dict:
        """Place a trading order"""
        pass
        
    @abstractmethod
    async def modify_order(
        self,
        order_id: int,
        new_price: Optional[float] = None,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None
    ) -> bool:
        """Modify an existing order"""
        pass
        
    @abstractmethod
    async def cancel_order(self, order_id: int) -> bool:
        """Cancel an existing order"""
        pass

# Risk Management Module Interface
class RiskModule(TradingModule):
    """Interface for risk management"""
    
    @abstractmethod
    async def validate_trade(
        self,
        symbol: str,
        volume: float,
        direction: str
    ) -> bool:
        """Validate trade against risk rules"""
        pass
        
    @abstractmethod
    async def calculate_position_size(
        self,
        symbol: str,
        risk_per_trade: float
    ) -> float:
        """Calculate safe position size"""
        pass
        
    @abstractmethod
    async def monitor_risk_metrics(self) -> Dict:
        """Monitor current risk exposure"""
        pass

# Monitoring Module Interface
class MonitoringModule(TradingModule):
    """Interface for system monitoring"""
    
    @abstractmethod
    async def log_trade(self, trade_data: Dict) -> None:
        """Log trade information"""
        pass
        
    @abstractmethod
    async def monitor_performance(self) -> Dict:
        """Monitor trading performance"""
        pass
        
    @abstractmethod
    async def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = 'info'
    ) -> bool:
        """Send monitoring alert"""
        pass

# Module Factory
class ModuleFactory:
    """Factory for creating trading modules"""
    
    @staticmethod
    def create_module(module_type: str, config: Dict) -> TradingModule:
        """Create a new module instance"""
        module_map = {
            'data': lambda: DataModule(config),
            'strategy': lambda: StrategyModule(config),
            'execution': lambda: ExecutionModule(config),
            'risk': lambda: RiskModule(config),
            'monitoring': lambda: MonitoringModule(config)
        }
        
        if module_type not in module_map:
            raise ValueError(f"Unknown module type: {module_type}")
            
        return module_map[module_type]()

# Trading Bot Core
class TradingBot:
    """Core trading bot that orchestrates all modules"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('trading_bot')
        self.modules = {}
        self._initialize_modules()
        
    def _initialize_modules(self):
        """Initialize all required modules"""
        try:
            factory = ModuleFactory()
            
            # Create modules
            self.modules['data'] = factory.create_module('data', self.config)
            self.modules['strategy'] = factory.create_module('strategy', self.config)
            self.modules['execution'] = factory.create_module('execution', self.config)
            self.modules['risk'] = factory.create_module('risk', self.config)
            self.modules['monitoring'] = factory.create_module('monitoring', self.config)
            
        except Exception as e:
            self.logger.error(f"Module initialization error: {str(e)}")
            raise
            
    async def start(self):
        """Start the trading bot"""
        try:
            # Initialize all modules
            for module in self.modules.values():
                await module.initialize()
                
            # Start main trading loop
            await self._trading_loop()
            
        except Exception as e:
            self.logger.error(f"Bot startup error: {str(e)}")
            await self.stop()
            
    async def stop(self):
        """Stop the trading bot"""
        try:
            # Cleanup all modules
            for module in self.modules.values():
                await module.cleanup()
                
        except Exception as e:
            self.logger.error(f"Bot shutdown error: {str(e)}")
            
    async def _trading_loop(self):
        """Main trading loop"""
        try:
            while True:
                # Get market data
                market_data = await self.modules['data'].fetch_historical_data(
                    self.config['symbol'],
                    self.config['timeframe'],
                    datetime.now(),
                    datetime.now()
                )
                
                # Analyze market
                analysis = await self.modules['strategy'].analyze_market(market_data)
                
                # Generate signals
                signals = await self.modules['strategy'].generate_signals(analysis)
                
                # Validate signals
                valid_signals = await self.modules['strategy'].validate_signals(signals)
                
                # Execute valid signals
                for signal in valid_signals:
                    # Validate risk
                    if await self.modules['risk'].validate_trade(
                        signal['symbol'],
                        signal['volume'],
                        signal['direction']
                    ):
                        # Place order
                        order_result = await self.modules['execution'].place_order(
                            signal['symbol'],
                            signal['type'],
                            signal['volume'],
                            signal.get('price')
                        )
                        
                        # Log trade
                        await self.modules['monitoring'].log_trade(order_result)
                        
                # Monitor performance
                performance = await self.modules['monitoring'].monitor_performance()
                
                # Check risk metrics
                risk_metrics = await self.modules['risk'].monitor_risk_metrics()
                
                # Sleep for interval
                await asyncio.sleep(self.config.get('loop_interval', 1))
                
        except Exception as e:
            self.logger.error(f"Trading loop error: {str(e)}")
            await self.modules['monitoring'].send_alert(
                'error',
                f"Trading loop error: {str(e)}",
                'critical'
            )
            raise
