import logging
import asyncio
from typing import Optional
from .trading_bot import TradingBot

class BotManager:
    _instance = None
    _bot: Optional[TradingBot] = None
    _task: Optional[asyncio.Task] = None
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.broker = None
        self.performance_monitor = None
        self.risk_manager = None
        self.regime_detector = None
        self._init_lock = asyncio.Lock()
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def initialize(
        self,
        config: dict,
        broker,
        performance_monitor,
        risk_manager,
        regime_detector
    ) -> bool:
        """Initialize bot manager with all required components"""
        async with self._init_lock:
            try:
                self.config = config
                self.broker = broker
                self.performance_monitor = performance_monitor
                self.risk_manager = risk_manager
                self.regime_detector = regime_detector
                
                if self.risk_manager:
                    self.risk_manager.broker = self.broker
                
                # Validate components
                if not all([
                    self.config,
                    self.broker,
                    self.performance_monitor,
                    self.risk_manager,
                    self.regime_detector
                ]):
                    self.logger.error("Missing required components")
                    return False
                    
                self.logger.info("Bot manager initialized successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Error initializing bot manager: {str(e)}")
                return False
    async def process_symbol(self, symbol: str) -> bool:
        """Delegate symbol execution to the active bot instance"""
        if self._bot and self.is_running:
            await self._bot.process_symbol(symbol)
            return True
        return False
        
    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()
    
    async def start_bot(self, config: dict) -> bool:
        """Start the trading bot if it's not already running"""
        try:
            if self.is_running:
                self.logger.warning("Trading bot is already running")
                return False
                
            self._bot = TradingBot(config)
            self._task = asyncio.create_task(self._bot.start())
            
            self.logger.info("Trading bot started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting trading bot: {str(e)}")
            return False
    
    async def stop_bot(self) -> bool:
        """Stop the trading bot if it's running"""
        try:
            if not self.is_running:
                self.logger.warning("Trading bot is not running")
                return False
                
            if self._bot:
                await self._bot.stop()
                
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None
                
            self._bot = None
            self.logger.info("Trading bot stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping trading bot: {str(e)}")
            return False
    
    def get_bot_status(self) -> dict:
        """Get current bot status"""
        return {
            'running': self.is_running,
            'has_bot': self._bot is not None,
            'task_status': str(self._task.get_state()) if self._task else None
        }
        
    async def run_trading_cycle(self):
        """Run the trading cycle"""
        try:
            # Initialize bot if not already initialized
            if not self._bot:
                self._bot = TradingBot(self.config)
                await self._bot.start()
                self.logger.info("Trading bot initialized successfully")

            while True:
                try:
                    # Process each symbol
                    for symbol in self.config['trading']['symbols']:
                        if not self.is_running:
                            break
                        await self._bot._process_symbol(symbol)
                        
                    # Manage existing positions
                    try:
                        await self._bot.position_manager.manage_positions()
                    except Exception as e:
                        self.logger.error(f"Error managing positions: {str(e)}")
                        
                    # Update risk metrics
                    try:
                        acc_info = await self._bot.broker.get_account_info()
                        self._bot.risk_manager.update_risk_metrics(acc_info['balance'], acc_info['equity'])
                    except Exception as e:
                        self.logger.error(f"Error updating risk metrics: {str(e)}")
                        
                    # Sleep for interval
                    await asyncio.sleep(self.config.get('trading', {}).get('update_interval', 5))
                    
                except Exception as e:
                    self.logger.error(f"Error in trading cycle: {str(e)}")
                    await asyncio.sleep(5)  # Wait before retrying
                    
        except Exception as e:
            self.logger.error(f"Fatal error in trading cycle: {str(e)}")
            await self.shutdown()
            
    async def shutdown(self):
        """Shutdown the bot manager and cleanup resources"""
        try:
            # Stop the trading bot
            await self.stop_bot()
            
            # Cleanup resources
            self.config = None
            self.broker = None
            self.performance_monitor = None
            self.risk_manager = None
            self.regime_detector = None
            
            self.logger.info("Bot manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}") 
            
    def cleanup(self):
        """Synchronous cleanup wrapper of bot manager resources"""
        try:
            if self._task and not self._task.done():
                self._task.cancel()
            self._bot = None
            self.logger.info("Bot manager cleanup complete")
        except Exception as e:
            self.logger.error(f"Error during bot manager cleanup: {str(e)}")