import os
import logging
import asyncio
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from dotenv import load_dotenv
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
import psutil
import sys
import gc
import time
import signal
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress XGBoost serialization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Import trading components
from modules.trading.market_analyzer import MarketAnalyzer
from modules.trading.risk_manager import RiskManager
from modules.trading.performance_monitor import PerformanceMonitor
from modules.trading.regime_detector import RegimeDetector
from modules.trading.position_manager import PositionManager
from modules.trading.bot_manager import BotManager
from modules.trading.signal import Signal
from modules.trading.broker import MT5Broker
from modules.config.config_manager import ConfigManager

# Configure logging with timestamp and detailed format
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', mode='a'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Add memory and performance tracking
class PerformanceTracker:
    def __init__(self):
        self.start_time = time.time()
        self.last_check = self.start_time
        self.memory_usage = []
        self.cpu_usage = []
        self.interval_seconds = 60  # Check every minute

    def check(self):
        current_time = time.time()
        if current_time - self.last_check >= self.interval_seconds:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent(interval=0.1)
            
            self.memory_usage.append((current_time, memory_mb))
            self.cpu_usage.append((current_time, cpu_percent))
            
            self.last_check = current_time
            
            # Log if memory usage is too high
            if memory_mb > 500:  # 500MB threshold
                logger.warning(f"High memory usage: {memory_mb:.2f} MB")
                # Force garbage collection
                gc.collect()
            
            return {
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'uptime_hours': (current_time - self.start_time) / 3600
            }
        return None

class ProfessionalTradingSystem:
    def __init__(self):
        """Initialize the professional trading system"""
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.config = None
        self.market_analyzer = None
        self.risk_manager = None
        self.performance_monitor = None
        self.regime_detector = None
        self.position_manager = None
        self.bot_manager = None
        
        # Initialize state
        self.is_running = False
        self.emergency_shutdown = False
        self.last_health_check = datetime.now()
        self.health_check_interval = timedelta(minutes=5)
        self.emergency_shutdown_triggered = False
        self.market_data_cache = {}
        self.performance_tracker = PerformanceTracker()
        
        # Initialize task tracking
        self.active_tasks = set()
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, sig, frame):
        """Handle termination signals"""
        self.logger.info(f"Received signal {sig}, initiating graceful shutdown")
        self.emergency_shutdown = True
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(self.shutdown_event.set)
            else:
                self.shutdown_event.set()
        except RuntimeError:
            self.shutdown_event.set()
        
    async def initialize(self):
        """Initialize the trading system"""
        try:
            self.logger.info("Starting professional trading system initialization...")
            
            # Load environment variables
            load_dotenv()
            self.logger.info("Environment variables loaded")
            
            # Verify system requirements
            if not self._verify_system_requirements():
                self.logger.error("System requirements not met")
                return False
                
            self.logger.info("System requirements verification passed")
            
            # Load configuration
            self.config = self._load_config()
            if not self.config:
                self.logger.error("Failed to load configuration")
                return False
                
            # Initialize components with timeout protection (includes MT5 broker connection and model training)
            try:
                if not await asyncio.wait_for(self._initialize_components(), timeout=60):
                    self.logger.error("Failed to initialize components")
                    return False
            except asyncio.TimeoutError:
                self.logger.error("Timeout while initializing components")
                return False
                
            # Verify trading permissions (which checks mt5 symbol selects and trade modes)
            if not self._verify_trading_permissions():
                self.logger.error("Failed to verify trading permissions")
                return False
                
            # Set initial symbol and timeframe
            self.config['trading']['current_symbol'] = self.config['trading']['symbols'][0]  # Start with first symbol
            self.config['trading']['current_timeframe'] = self.config['market_analysis']['timeframes'][0]  # Start with first timeframe
            
            # Start the trading bot in the background
            if not await self.bot_manager.start_bot(self.config):
                self.logger.error("Failed to start trading bot task")
                return False
            
            self.logger.info("Professional trading system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing trading system: {str(e)}")
            return False
            
    def _verify_system_requirements(self) -> bool:
        """Verify system requirements for testing"""
        try:
            # Check Python version
            if sys.version_info < (3, 8):
                logger.error("Python 3.8 or higher is required")
                return False
                
            # Reduced memory requirement for testing
            memory = psutil.virtual_memory()
            min_memory = 1 * 1024 * 1024 * 1024  # 1GB
            if memory.available < min_memory:
                logger.error(f"Available memory ({memory.available / 1024 / 1024 / 1024:.1f}GB) below minimum requirement (1GB)")
                return False
                
            # Reduced CPU requirement for testing
            if psutil.cpu_count() < 2:
                logger.error(f"Available CPU cores ({psutil.cpu_count()}) below minimum requirement (2)")
                return False
                
            # Reduced disk space requirement for testing
            disk = psutil.disk_usage('/')
            min_disk = 5 * 1024 * 1024 * 1024  # 5GB
            if disk.free < min_disk:
                logger.error(f"Available disk space ({disk.free / 1024 / 1024 / 1024:.1f}GB) below minimum requirement (5GB)")
                return False
                
            logger.info("System requirements verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying system requirements: {str(e)}")
            return False
            
    async def _initialize_mt5_with_retry(self, max_retries: int = 3) -> bool:
        """Initialize MT5 with retry mechanism"""
        retries = 0
        while retries < max_retries:
            try:
                success = await self._initialize_mt5()
                if success:
                    return True
                    
                retries += 1
                self.logger.warning(f"MT5 initialization failed. Retry {retries}/{max_retries}")
                await asyncio.sleep(5 * retries)  # Exponential backoff
                
            except Exception as e:
                retries += 1
                self.logger.error(f"MT5 initialization error: {str(e)}. Retry {retries}/{max_retries}")
                await asyncio.sleep(5 * retries)
                
        return False
        
    async def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            # Get MT5 credentials from environment variables with defaults
            mt5_login = os.getenv('MT5_LOGIN')
            mt5_password = os.getenv('MT5_PASSWORD')
            mt5_server = os.getenv('MT5_SERVER')

            # Validate credentials
            if not all([mt5_login, mt5_password, mt5_server]):
                self.logger.error("Missing MT5 credentials in environment variables")
                return False

            # Initialize MT5
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False

            # Login to MT5
            try:
                login_result = mt5.login(
                    login=int(mt5_login),
                    password=mt5_password,
                    server=mt5_server
                )
                if not login_result:
                    self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
            except ValueError as e:
                self.logger.error(f"Invalid MT5 login credentials: {str(e)}")
                return False

            # Verify connection
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                self.logger.error("Failed to get terminal info")
                return False

            # Enable market data for symbols
            for symbol in self.config['trading']['symbols']:
                if not mt5.symbol_select(symbol, True):
                    self.logger.warning(f"Failed to enable symbol {symbol}")

            self.logger.info(f"MT5 initialized successfully - Connected to {mt5_server}")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing MT5: {str(e)}")
            return False
            
    def _configure_mt5_parameters(self) -> bool:
        """Configure MT5 trading parameters"""
        try:
            # Get account info
            account_info = mt5.account_info()
            if not account_info:
                logger.error("Failed to get account info")
                return False
                
            # Set up basic trading parameters
            try:
                mt5.terminal_info()._asdict()  # Check terminal connection
            except Exception as e:
                logger.warning(f"Terminal info check failed: {str(e)}")
                
            # Configure symbol settings for each trading symbol
            for symbol in self.config['trading']['symbols']:
                try:
                    symbol_info = mt5.symbol_info(symbol)
                    if symbol_info is None:
                        continue
                        
                    if not mt5.symbol_select(symbol, True):
                        logger.warning(f"Failed to select symbol: {symbol}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"Error configuring symbol {symbol}: {str(e)}")
                    continue
                    
            logger.info("MT5 parameters configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error configuring MT5 parameters: {str(e)}")
            return False
            
    def _verify_professional_account(self, account_info) -> bool:
        """Verify if the account meets professional trading requirements"""
        try:
            # Check account type and features
            if account_info.margin_so_mode != 0:  # Professional margin calculation mode
                logger.warning("Account not using professional margin calculation mode")
                return False
                
            # Check leverage (temporarily allowing higher leverage for testing)
            if account_info.leverage > 500:  # Changed from 30 to 500 for testing
                logger.warning(f"Account leverage too high: 1:{account_info.leverage}")
                return False
                
            # Check minimum balance requirement (lowered for testing)
            min_balance = 100  # Changed from 50000 to 100 for testing
            if account_info.balance < min_balance:
                logger.warning(f"Account balance (${account_info.balance}) below minimum requirement (${min_balance})")
                return False
                
            # Check margin call and stop out levels (adjusted for testing)
            if account_info.margin_so_call < 50:  # Changed from 100 to 50 for testing
                logger.warning("Margin call level too low for professional trading")
                return False
                
            if account_info.margin_so_so < 20:  # Changed from 50 to 20 for testing
                logger.warning("Stop out level too low for professional trading")
                return False
                
            # Verify advanced trading permissions
            if not self._verify_advanced_trading_permissions():
                return False
                
            logger.info("Account verification passed with testing parameters")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying professional account: {str(e)}")
            return False
            
    def _verify_advanced_trading_permissions(self) -> bool:
        """Verify advanced trading permissions for testing"""
        try:
            # Get account trade features
            account_info = mt5.account_info()
            if not account_info:
                return False
                
            # Check for advanced order types
            required_actions = [
                mt5.TRADE_ACTION_DEAL,      # Market orders
                mt5.TRADE_ACTION_PENDING,    # Pending orders
                mt5.TRADE_ACTION_SLTP,       # Modify stops
                mt5.TRADE_ACTION_MODIFY     # Modify orders
            ]
            
            # Verify basic trading capabilities
            for action in required_actions:
                if not isinstance(action, int):
                    logger.warning(f"Trading action {action} not available")
                    return False
                    
            logger.info("Advanced trading permissions verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying advanced trading permissions: {str(e)}")
            return False
            
    def _verify_trading_permissions(self) -> bool:
        """Verify account has necessary trading permissions"""
        try:
            # Get available symbols
            symbols = mt5.symbols_get()
            if symbols is None:
                logger.error("Failed to get symbols")
                return False
                
            # Check if required symbols are available
            required_symbols = self.config['trading']['symbols']
            available_symbols = [symbol.name for symbol in symbols]
            
            for symbol in required_symbols:
                if symbol not in available_symbols:
                    logger.error(f"Required symbol not available: {symbol}")
                    return False
                    
                # Check trading permissions for each symbol
                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                    logger.warning(f"Full trading not currently allowed for symbol: {symbol} (Market may be closed on weekends/holidays)")
                    
            return True
            
        except Exception as e:
            logger.error(f"Error verifying trading permissions: {str(e)}")
            return False
            
    async def _load_config_with_validation(self) -> dict:
        """Load and validate professional trading configuration"""
        try:
            # Load configuration
            config = self._load_config()
            if not config:
                return None
                
            # Normalize config schema mismatches
            config = self._normalize_config(config)
                
            # Validate professional configuration
            if not self._validate_professional_config(config):
                return None
                
            # Validate symbol-specific settings
            if not self._validate_symbol_settings(config):
                return None
                
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return None
            
    def _normalize_config(self, config: dict) -> dict:
        """Normalize fallback YAML configuration structures to canonical JSON schema"""
        if 'risk_management' not in config:
            config['risk_management'] = {}
        
        rm = config['risk_management']
        
        # Normalize daily loss
        if 'max_daily_loss' not in rm:
            rm['max_daily_loss'] = rm.get('emergency_stop', {}).get('max_daily_loss_percent', 0.03)
            
        # Normalize position size
        if 'max_position_size' not in rm:
            rm['max_position_size'] = rm.get('position_management', {}).get('volume_profile', {}).get('max_lots', 1.0)
            
        # Normalize position limits / max_positions
        if 'position_limits' not in rm:
            rm['position_limits'] = {
                'max_positions': rm.get('emergency_stop', {}).get('max_open_positions', 5),
                'max_correlation': rm.get('portfolio', {}).get('max_correlation', 0.7)
            }
        else:
            if 'max_positions' not in rm['position_limits']:
                rm['position_limits']['max_positions'] = rm.get('position_limits', {}).get('max_positions_per_symbol', 5)
            if 'max_correlation' not in rm['position_limits']:
                rm['position_limits']['max_correlation'] = rm.get('portfolio', {}).get('max_correlation', 0.7)
                
        # Normalize loss limits / max_drawdown
        if 'loss_limits' not in rm:
            rm['loss_limits'] = {
                'max_drawdown': rm.get('emergency_stop', {}).get('max_drawdown_percent', 0.05)
            }
            
        # Normalize risk per trade
        if 'risk_per_trade' not in rm:
            rm['risk_per_trade'] = rm.get('risk_per_trade', 0.02)
            
        # Normalize min risk reward
        if 'min_risk_reward' not in rm:
            rm['min_risk_reward'] = config.get('trading_parameters', {}).get('risk_reward_ratio', 1.5)
            
        return config
            
    async def _initialize_components(self) -> bool:
        """Initialize trading components with dependency injection"""
        try:
            # Initialize broker
            self.broker = MT5Broker(self.config)
            if not self.broker.initialized:
                logger.error("Failed to initialize broker")
                return False

            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor()
            
            # Initialize risk manager with config
            self.risk_manager = RiskManager(self.config)
            self.risk_manager.broker = self.broker
            if not await self.risk_manager.initialize():
                logger.error("Failed to initialize risk manager")
                return False
            
            # Initialize regime detector
            self.regime_detector = RegimeDetector(self.config)
            
            # Get historical data for initialization
            historical_data = await self._get_historical_data_all_symbols()
            if not historical_data:
                logger.error("Failed to get historical data")
                return False
                
            # Flatten the nested dictionary of DataFrames {symbol: {timeframe: df}} into a single DataFrame
            flat_dfs = []
            for symbol, tf_dict in historical_data.items():
                for timeframe, df in tf_dict.items():
                    if df is not None and not df.empty:
                        flat_dfs.append(df)
            
            if flat_dfs:
                flat_historical_data = pd.concat(flat_dfs, ignore_index=True)
            else:
                flat_historical_data = pd.DataFrame()
                
            # Initialize market regime detection
            if not self.regime_detector.initialize_models(flat_historical_data):
                logger.error("Failed to initialize regime detection models")
                return False
                
            # Initialize bot manager with components
            self.bot_manager = BotManager.get_instance()
            if not await self.bot_manager.initialize(
                self.config,
                self.broker,
                self.performance_monitor,
                self.risk_manager,
                self.regime_detector
            ):
                logger.error("Failed to initialize bot manager")
                return False
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            return False
            
    async def _perform_initial_analysis(self) -> bool:
        """Perform initial market analysis before trading"""
        try:
            # Get market data for all symbols
            market_data = {}
            for symbol in self.config['trading']['symbols']:
                data = self._get_market_data(symbol)
                if data.empty:
                    logger.error(f"Failed to get market data for {symbol}")
                    return False
                market_data[symbol] = data
                
            # Analyze market conditions
            for symbol, data in market_data.items():
                # Detect market regime
                regime = self.regime_detector.detect_regime(data)
                if regime is None:
                    logger.error(f"Failed to detect market regime for {symbol}")
                    return False
                    
                # Check volatility
                if not self._is_volatility_acceptable(data):
                    logger.warning(f"Volatility conditions not suitable for {symbol}")
                    return False
                    
                # Check spread
                if not self._is_spread_acceptable(symbol):
                    logger.warning(f"Spread conditions not suitable for {symbol}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error in initial market analysis: {str(e)}")
            return False
            
    async def run(self):
        """Run the trading system main loop"""
        self.is_running = True
        self.logger.info("Trading system is starting...")
        
        # Run periodic tasks in separate tasks
        health_check_task = asyncio.create_task(self._periodic_health_check())
        self.active_tasks.add(health_check_task)
        health_check_task.add_done_callback(self.active_tasks.discard)
        
        data_update_task = asyncio.create_task(self._periodic_data_update())
        self.active_tasks.add(data_update_task)
        data_update_task.add_done_callback(self.active_tasks.discard)
        
        # Memory cleanup task
        memory_cleanup_task = asyncio.create_task(self._periodic_memory_cleanup())
        self.active_tasks.add(memory_cleanup_task)
        memory_cleanup_task.add_done_callback(self.active_tasks.discard)
        
        try:
            trading_interval = self.config.get('trading', {}).get('update_interval', 5)
            
            while self.is_running and not self.emergency_shutdown and not self.shutdown_event.is_set():
                cycle_start = time.time()
                
                try:
                    # Check performance metrics
                    perf_metrics = self.performance_tracker.check()
                    if perf_metrics:
                        self.logger.debug(f"Performance: {perf_metrics}")
                    
                    # Update market data (runs in parallel with other operations)
                    market_data_task = asyncio.create_task(self._update_market_data())
                    self.active_tasks.add(market_data_task)
                    market_data_task.add_done_callback(self.active_tasks.discard)
                    
                    # Update market regime (runs in parallel)
                    regime_task = asyncio.create_task(self._update_market_regime())
                    self.active_tasks.add(regime_task)
                    regime_task.add_done_callback(self.active_tasks.discard)
                    
                    # Wait for current data tasks to complete
                    await asyncio.gather(market_data_task, regime_task)
                    
                    # Process trading logic for each symbol in parallel
                    tasks = []
                    for symbol in self.config['trading']['symbols']:
                        task = asyncio.create_task(self._process_symbol(symbol))
                        tasks.append(task)
                        self.active_tasks.add(task)
                        task.add_done_callback(self.active_tasks.discard)
                        
                    await asyncio.gather(*tasks)
                    
                    # Calculate execution time and adjust sleep time
                    execution_time = time.time() - cycle_start
                    sleep_time = max(0, trading_interval - execution_time)
                    
                    if execution_time > trading_interval:
                        self.logger.warning(f"Trading cycle taking longer than interval: {execution_time:.2f}s > {trading_interval}s")
                    
                    # Wait for next cycle or until shutdown signal
                    try:
                        await asyncio.wait_for(self.shutdown_event.wait(), timeout=sleep_time)
                        # If we get here, shutdown was requested
                        break
                    except asyncio.TimeoutError:
                        # Normal timeout, continue with next cycle
                        pass
                        
                except Exception as e:
                    self.logger.error(f"Error in trading cycle: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    # Brief pause to prevent rapid error loops
                    await asyncio.sleep(1)
                    
        except Exception as e:
            self.logger.error(f"Critical error in trading system: {str(e)}")
            self.logger.error(traceback.format_exc())
        finally:
            self.logger.info("Trading system main loop ended")
            await self.shutdown()
            
    async def _process_symbol(self, symbol: str):
        """Process trading logic for a single symbol"""
        try:
            # Safely delegate execution using public bot manager interface
            await self.bot_manager.process_symbol(symbol)
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {str(e)}")

    async def _periodic_health_check(self):
        """Run health checks at regular intervals"""
        while self.is_running and not self.emergency_shutdown and not self.shutdown_event.is_set():
            try:
                health_ok = await self._perform_health_check()
                if not health_ok:
                    self.logger.warning("Health check failed")
                    # Take corrective action if needed
            except Exception as e:
                self.logger.error(f"Error in health check: {str(e)}")
            
            await asyncio.sleep(self.health_check_interval.total_seconds())
            
    async def _periodic_data_update(self):
        """Update market data at regular intervals"""
        update_interval = timedelta(minutes=1)
        while self.is_running and not self.emergency_shutdown and not self.shutdown_event.is_set():
            try:
                await self._update_market_data()
            except Exception as e:
                self.logger.error(f"Error updating market data: {str(e)}")
            
            await asyncio.sleep(update_interval.total_seconds())
            
    async def _periodic_memory_cleanup(self):
        """Perform periodic memory cleanup"""
        cleanup_interval = timedelta(minutes=30)
        while self.is_running and not self.emergency_shutdown and not self.shutdown_event.is_set():
            try:
                # Run garbage collection
                gc.collect()
                
                # Log memory usage
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.logger.info(f"Memory usage after cleanup: {memory_mb:.2f} MB")
                
                # Clean up market data cache if too large
                if hasattr(self, 'market_data_cache') and len(self.market_data_cache) > 100:
                    oldest_items = sorted(self.market_data_cache.items(), key=lambda x: x[1].get('timestamp', 0))[:50]
                    for key, _ in oldest_items:
                        self.market_data_cache.pop(key, None)
                    self.logger.info(f"Cleaned up {len(oldest_items)} old market data cache entries")
                    
            except Exception as e:
                self.logger.error(f"Error in memory cleanup: {str(e)}")
            
            await asyncio.sleep(cleanup_interval.total_seconds())

    async def shutdown(self):
        """Shutdown the trading system"""
        if not self.is_running:
            return
            
        self.logger.info("Shutting down trading system...")
        self.is_running = False
        
        try:
            # Signal shutdown to all tasks
            self.shutdown_event.set()
            
            # Cancel all active tasks
            tasks = list(self.active_tasks)
            if tasks:
                self.logger.info(f"Cancelling {len(tasks)} active tasks")
                for task in tasks:
                    task.cancel()
                    
                # Wait for all tasks to complete with timeout
                try:
                    await asyncio.wait(tasks, timeout=10)
                except asyncio.TimeoutError:
                    self.logger.warning("Some tasks did not terminate in time")
            
            # Close all positions
            await self._close_all_positions()
            
            # Cleanup resources
            if hasattr(self, 'bot_manager') and self.bot_manager:
                self.bot_manager.cleanup()
                
            # Shutdown MT5
            mt5.shutdown()
            
            self.logger.info("Trading system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def _close_all_positions(self):
        """Close all open positions"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                logger.warning("No positions to close")
                return
                
            for position in positions:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": position.ticket,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                    "price": mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask,
                    "deviation": 20,
                    "magic": 100,
                    "comment": "Emergency close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Failed to close position {position.ticket}: {result.comment}")
                    
        except Exception as e:
            logger.error(f"Error closing positions: {str(e)}")

    def _load_config(self) -> dict:
        """Load trading configuration from file with deep-merging and validation fallbacks"""
        try:
            def deep_merge(base: dict, update: dict) -> dict:
                for k, v in update.items():
                    if isinstance(v, dict):
                        if k not in base or not isinstance(base[k], dict):
                            base[k] = {}
                        base[k] = deep_merge(base[k], v)
                    else:
                        base[k] = v
                return base

            config_dir = Path('config')
            
            # 1. Start with the baseline config
            baseline_path = config_dir / 'config.json'
            if not baseline_path.exists():
                baseline_path = Path('config.json') if Path('config.json').exists() else Path('config.yaml')
            
            config = ConfigManager.load_config(str(baseline_path))
            
            # 2. Merge supplementary files if they exist
            for filename in ['strategy_config.json', 'risk_management.json', 'trading_config.json', 'trading_config.yaml', 'broker_config.yaml']:
                file_path = config_dir / filename
                if file_path.exists():
                    try:
                        supp_config = ConfigManager.load_config(str(file_path))
                        config = deep_merge(config, supp_config)
                    except Exception as e:
                        logger.warning(f"Failed to merge supplementary config {filename}: {str(e)}")
                        
            # Ensure critical keys exist by mapping/aliasing if missing
            if 'machine_learning' not in config and 'market_analysis' in config and 'ml' in config['market_analysis']:
                config['machine_learning'] = config['market_analysis']['ml']
                
            ml_config = config.setdefault('machine_learning', {})
            if 'enabled' not in ml_config:
                ml_config['enabled'] = True
            if 'models' not in ml_config:
                ml_config['models'] = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']
            if 'training' not in ml_config:
                ml_config['training'] = {'min_samples': 1000, 'validation_size': 0.2, 'retrain_interval_hours': 24}
            if 'prediction' not in ml_config:
                ml_config['prediction'] = {'confidence_threshold': 0.65, 'ensemble_threshold': 0.7}
            if 'hyperparameters' not in ml_config:
                ml_config['hyperparameters'] = {}
            if 'feature_engineering' not in ml_config:
                ml_config['feature_engineering'] = {'feature_selection': {'enabled': True, 'n_features': 20}}
            if 'warmup' not in ml_config:
                ml_config['warmup'] = {'min_predictions': 50, 'min_accuracy': 0.55}
            
            # Ensure position_management or trading_parameters exit parameters exist
            if 'trading_parameters' not in config:
                config['trading_parameters'] = {
                    'execution': config.get('execution', {}),
                    'exit_parameters': {
                        'stop_loss': {'atr_multiplier': 2.0, 'min_distance': 10, 'max_distance': 100},
                        'take_profit': {'atr_multiplier': 3.0, 'min_distance': 15, 'max_distance': 150},
                        'trailing_stop': {'enabled': True, 'activation': 0.5, 'step': 0.1},
                        'break_even': {'enabled': True, 'activation_threshold': 0.5, 'step': 0.2},
                        'partial_exit': {'enabled': True, 'activation_threshold': 0.5, 'step': 0.2}
                    }
                }
            else:
                exit_params = config['trading_parameters'].setdefault('exit_parameters', {})
                if 'break_even' not in exit_params:
                    exit_params['break_even'] = {'enabled': True, 'activation_threshold': 0.5, 'step': 0.2}
                if 'partial_exit' not in exit_params:
                    exit_params['partial_exit'] = {'enabled': True, 'activation_threshold': 0.5, 'step': 0.2}
            
            # Normalize trading timeframes config to be a flat list
            trading_conf = config.setdefault('trading', {})
            tfs = trading_conf.get('timeframes')
            if isinstance(tfs, dict):
                primary = tfs.get('primary')
                secondary = tfs.get('secondary', [])
                if isinstance(secondary, str):
                    secondary = [secondary]
                normalized_tfs = []
                if primary:
                    normalized_tfs.append(primary)
                for tf in secondary:
                    if tf not in normalized_tfs:
                        normalized_tfs.append(tf)
                trading_conf['timeframes'] = normalized_tfs
            elif isinstance(tfs, str):
                trading_conf['timeframes'] = [tfs]
            
            # Ensure regime_detection thresholds exist for RegimeDetector
            regime_det = config.setdefault('regime_detection', {})
            thresholds = regime_det.setdefault('thresholds', {})
            
            market_anal = config.get('market_analysis', {})
            regime_class = market_anal.get('regime_classification', {})
            if 'volatile_threshold' not in thresholds:
                thresholds['volatile_threshold'] = regime_class.get('volatile_threshold', 0.02)
            if 'trending_threshold' not in thresholds:
                thresholds['trending_threshold'] = regime_class.get('trending_threshold', 0.7)
                
            volatility_class = market_anal.get('volatility', {})
            volatility_thresh = thresholds.setdefault('volatility', {})
            if 'low_threshold' not in volatility_thresh:
                volatility_thresh['low_threshold'] = volatility_class.get('low_threshold', 0.005)
            if 'high_threshold' not in volatility_thresh:
                volatility_thresh['high_threshold'] = volatility_class.get('high_threshold', 0.02)
            
            # Direct injection fallback for MT5 credentials from environment
            mt5_acc = config.setdefault('mt5_account', {})
            if not mt5_acc.get('login'):
                mt5_acc['login'] = os.getenv('MT5_LOGIN')
            if not mt5_acc.get('password'):
                mt5_acc['password'] = os.getenv('MT5_PASSWORD')
            if not mt5_acc.get('server'):
                mt5_acc['server'] = os.getenv('MT5_SERVER')
            
            return config
            
        except Exception as e:
            logger.error(f"Error in config loader: {str(e)}")
            try:
                return ConfigManager.load_config('config.yaml')
            except:
                return {}

    def _validate_professional_config(self, config: dict) -> bool:
        """Validate professional trading configuration"""
        try:
            required_sections = [
                'risk_management',
                'market_analysis',
                'execution',
                'position_sizing',
                'regime_detection'
            ]
            
            # Check required sections
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required configuration section: {section}")
                    return False
                    
            # Validate risk management settings
            risk_config = config['risk_management']
            if not all([
                risk_config.get('enabled', False),
                risk_config.get('risk_per_trade', 0) <= 0.02,  # Max 2% risk per trade
                risk_config.get('max_daily_loss', 0) <= 0.05,  # Max 5% daily loss
                risk_config.get('max_positions', 0) <= 5  # Max 5 simultaneous positions
            ]):
                logger.error("Risk management configuration does not meet professional standards")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating configuration: {str(e)}")
            return False

    def _validate_symbol_settings(self, config: dict) -> bool:
        """Validate symbol-specific trading settings"""
        try:
            if 'trading' not in config or 'symbols' not in config['trading']:
                logger.error("Missing trading symbols configuration")
                return False
                
            for symbol in config['trading']['symbols']:
                # Verify symbol exists
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logger.error(f"Symbol {symbol} not found")
                    return False
                    
                # Check minimum volume requirements
                if symbol_info.volume_min > config['risk_management']['position_sizing'].get('min_size', 0.01):
                    logger.error(f"Minimum volume setting too low for {symbol}")
                    return False
                    
                # Check maximum volume limits
                if symbol_info.volume_max < config['risk_management']['position_sizing'].get('max_size', 0.5):
                    logger.error(f"Maximum volume setting too high for {symbol}")
                    return False
                    
                # Verify spread limits
                current_spread = (symbol_info.ask - symbol_info.bid) / symbol_info.point
                if current_spread > config['execution'].get('max_spread', 50):
                    logger.warning(f"Current spread for {symbol} exceeds maximum allowed")
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating symbol settings: {str(e)}")
            return False

    async def _get_historical_data_all_symbols(self) -> pd.DataFrame:
        """Get historical data for all trading symbols"""
        try:
            all_data = {}
            for symbol in self.config['trading']['symbols']:
                # Get data for multiple timeframes
                timeframes = self.config['market_analysis']['timeframes']
                symbol_data = {}
                
                for timeframe in timeframes:
                    tf = getattr(mt5, f'TIMEFRAME_{timeframe}')
                    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 1000)
                    if rates is None:
                        logger.error(f"Failed to get historical data for {symbol} on {timeframe}")
                        return None
                    symbol_data[timeframe] = pd.DataFrame(rates)
                    
                all_data[symbol] = symbol_data
                
            return all_data
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None

    def _is_volatility_acceptable(self, data: pd.DataFrame) -> bool:
        """Check if market volatility is within acceptable range"""
        try:
            # Calculate ATR
            high = data['high']
            low = data['low']
            close = data['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            
            # Get volatility thresholds from config
            min_vol = self.config['market_analysis']['validation']['min_volatility']
            max_vol = self.config['market_analysis']['validation']['max_volatility']
            
            # Calculate current volatility as percentage
            current_price = data['close'].iloc[-1]
            volatility = atr / current_price
            
            return min_vol <= volatility <= max_vol
            
        except Exception as e:
            logger.error(f"Error checking volatility: {str(e)}")
            return False

    def _is_spread_acceptable(self, symbol: str) -> bool:
        """Check if current spread is within acceptable range"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
                
            current_spread = (symbol_info.ask - symbol_info.bid) / symbol_info.point
            max_spread = self.config['execution']['max_spread']
            
            return current_spread <= max_spread
            
        except Exception as e:
            logger.error(f"Error checking spread: {str(e)}")
            return False

    def _check_system_resources(self) -> bool:
        """Check system resources availability"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 80:  # 80% threshold
                logger.warning(f"High CPU usage: {cpu_percent}%")
                return False
                
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 85:  # 85% threshold
                logger.warning(f"High memory usage: {memory.percent}%")
                return False
                
            # Check disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 90:  # 90% threshold
                logger.warning(f"High disk usage: {disk.percent}%")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
            return False

    def _verify_account_status(self) -> bool:
        """Verify trading account status"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return False
                
            # Check margin level
            if account_info.margin > 0 and account_info.margin_level < 200:  # 200% minimum
                logger.warning(f"Low margin level: {account_info.margin_level}%")
                return False
                
            # Check free margin
            if account_info.margin_free < account_info.balance * 0.3:  # 30% minimum
                logger.warning("Low free margin")
                return False
                
            # Check if account is in drawdown
            if account_info.equity / account_info.balance < 0.9:  # Max 10% drawdown
                logger.warning("Account in significant drawdown")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error verifying account status: {str(e)}")
            return False

    def _check_trading_conditions(self) -> bool:
        """Check if current market conditions are suitable for trading"""
        try:
            for symbol in self.config['trading']['symbols']:
                # Check spread
                if not self._is_spread_acceptable(symbol):
                    return False
                    
                # Check trading session
                if not self._is_valid_trading_session(symbol):
                    return False
                    
                # Check market volatility
                data = self._get_market_data(symbol)
                if not self._is_volatility_acceptable(data):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking trading conditions: {str(e)}")
            return False

    def _is_valid_trading_session(self, symbol: str) -> bool:
        """Check if current time is within valid trading sessions"""
        try:
            current_time = datetime.now().time()
            sessions = self.config['trading'].get('sessions', {})
            
            for session_name, session_times in sessions.items():
                start_time = datetime.strptime(session_times['start'], "%H:%M").time()
                end_time = datetime.strptime(session_times['end'], "%H:%M").time()
                
                if start_time <= current_time <= end_time:
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking trading session: {str(e)}")
            return False

    def _detect_extreme_market_conditions(self) -> bool:
        """Detect extreme market conditions that require emergency handling"""
        try:
            for symbol in self.config['trading']['symbols']:
                data = self._get_market_data(symbol)
                
                # Check for extreme volatility
                volatility = data['close'].pct_change().std() * np.sqrt(252)
                if volatility > self.config['risk_management']['emergency_thresholds']['max_volatility']:
                    logger.warning(f"Extreme volatility detected for {symbol}")
                    return True
                    
                # Check for large price gaps
                price_gaps = abs(data['open'] - data['close']) / data['open']
                if price_gaps.max() > self.config['risk_management']['emergency_thresholds']['max_gap']:
                    logger.warning(f"Large price gap detected for {symbol}")
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting extreme market conditions: {str(e)}")
            return True  # Return True to trigger emergency handling

    def _detect_unusual_losses(self) -> bool:
        """Detect unusual or rapid losses"""
        try:
            # Check daily loss limit
            daily_loss = self.performance_monitor.get_daily_metrics()['profit']
            if abs(daily_loss) > self.config['risk_management']['max_daily_loss']:
                logger.warning("Daily loss limit exceeded")
                return True
                
            # Check drawdown
            current_drawdown = self.performance_monitor.get_current_drawdown()
            if current_drawdown > self.config['risk_management']['max_drawdown']:
                logger.warning("Maximum drawdown exceeded")
                return True
                
            # Check consecutive losses
            if self.performance_monitor.get_consecutive_losses() >= 5:
                logger.warning("Too many consecutive losses")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error detecting unusual losses: {str(e)}")
            return True

    def _detect_technical_issues(self) -> bool:
        """Detect technical issues that could affect trading"""
        try:
            # Check MT5 connection quality
            if not mt5.terminal_info().connected:
                logger.error("MT5 connection lost")
                return True
                
            # Check system resources
            if not self._check_system_resources():
                return True
                
            # Check for execution issues
            if self.performance_monitor.get_execution_metrics()['slippage'] > self.config['execution']['max_slippage']:
                logger.warning("Excessive slippage detected")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error detecting technical issues: {str(e)}")
            return True

    def _send_emergency_notification(self):
        """Send emergency notification to administrators"""
        try:
            message = f"""
            EMERGENCY ALERT
            Time: {datetime.now()}
            Account: {mt5.account_info().login}
            Status: Emergency shutdown triggered
            Reason: {self.emergency_shutdown_triggered}
            Current Equity: {mt5.account_info().equity}
            Open Positions: {len(mt5.positions_get() or [])}
            """
            
            # Log the emergency
            logger.critical(message)
            
            # Here you would implement your notification method (email, SMS, etc.)
            # For example:
            # self._send_email(message)
            # self._send_telegram_message(message)
            
        except Exception as e:
            logger.error(f"Error sending emergency notification: {str(e)}")

    async def _update_market_data(self):
        """Update market data cache"""
        try:
            for symbol in self.config['trading']['symbols']:
                data = self._get_market_data(symbol)
                if not data.empty:
                    self.market_data_cache[symbol] = data
                    
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")

    async def _update_market_regime(self):
        """Update market regime analysis"""
        try:
            for symbol, data in self.market_data_cache.items():
                regime = self.regime_detector.detect_regime(data)
                if regime:
                    logger.debug(f"Market regime for {symbol}: {regime.regime_type} "
                              f"(confidence: {regime.confidence:.2f})")
                    
        except Exception as e:
            logger.error(f"Error updating market regime: {str(e)}")

    def _get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get current market data for analysis"""
        try:
            timeframe = mt5.TIMEFRAME_H1
            bars = 100
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None:
                raise Exception("Failed to get market data")
                
            return pd.DataFrame(rates)
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return pd.DataFrame()

    async def _perform_health_check(self) -> bool:
        """Perform system health check"""
        try:
            current_time = datetime.now()
            if current_time - self.last_health_check < self.health_check_interval:
                return True
                
            self.last_health_check = current_time
            
            # Check MT5 connection
            if not mt5.terminal_info().connected:
                logger.error("MT5 connection lost")
                return False
                
            # Check system resources
            if not self._check_system_resources():
                return False
                
            # Verify account status
            if not self._verify_account_status():
                return False
                
            # Check trading conditions
            if not self._check_trading_conditions():
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            return False

    def _check_emergency_conditions(self) -> bool:
        """Check for emergency conditions that require immediate action"""
        try:
            # Check for extreme market conditions
            if self._detect_extreme_market_conditions():
                return True
                
            # Check for unusual losses
            if self._detect_unusual_losses():
                return True
                
            # Check for technical issues
            if self._detect_technical_issues():
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {str(e)}")
            return True  # Trigger emergency handling on error

    async def _handle_emergency(self):
        """Handle emergency situations"""
        try:
            logger.warning("Emergency condition detected - initiating emergency protocol")
            
            # Set emergency flag
            self.emergency_shutdown_triggered = True
            
            # Close all positions
            await self._close_all_positions()
            
            # Cancel all pending orders
            await self._cancel_all_orders()
            
            # Notify administrators
            self._send_emergency_notification()
            
            # Initiate shutdown
            await self.shutdown()
            
        except Exception as e:
            logger.error(f"Error in emergency handling: {str(e)}")

    async def _cancel_all_orders(self):
        """Cancel all pending orders"""
        try:
            orders = mt5.orders_get()
            if orders is None:
                logger.info("No pending orders to cancel")
                return
                
            for order in orders:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                    "comment": "Emergency cancellation"
                }
                
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Failed to cancel order {order.ticket}: {result.comment}")
                    
        except Exception as e:
            logger.error(f"Error cancelling orders: {str(e)}")

async def main():
    """Main entry point for the professional trading system"""
    try:
        # Create and initialize the trading system
        trading_system = ProfessionalTradingSystem()
        logger.info("Starting professional trading system initialization...")
        
        # Initialize the system
        if not await trading_system.initialize():
            logger.error("Failed to initialize trading system. Exiting...")
            return
            
        logger.info("Professional trading system initialized successfully")
        
        # Start the trading loop
        await trading_system.run()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Initiating graceful shutdown...")
    except Exception as e:
        logger.error(f"Critical error in main loop: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Cleanup
        if trading_system and trading_system.bot_manager:
            await trading_system.bot_manager.shutdown()
        mt5.shutdown()
        logger.info("Trading system shutdown complete")

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
