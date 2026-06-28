from typing import Dict, List
import psutil
import logging
import asyncio
from datetime import datetime, timedelta
from prometheus_client import Gauge, Counter, start_http_server
import MetaTrader5 as mt5

class HealthMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_metrics()
        self.system_checks = {}
        self.last_api_check = datetime.now()
        self.api_health = True
        
    def setup_metrics(self):
        """Initialize Prometheus metrics"""
        # System metrics
        self.cpu_usage = Gauge('system_cpu_usage', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_usage', 'Memory usage percentage')
        self.disk_usage = Gauge('system_disk_usage', 'Disk usage percentage')
        
        # Trading metrics
        self.account_balance = Gauge('trading_account_balance', 'Current account balance')
        self.open_positions = Gauge('trading_open_positions', 'Number of open positions')
        self.daily_pnl = Gauge('trading_daily_pnl', 'Daily profit/loss')
        self.api_latency = Gauge('trading_api_latency', 'API response time in milliseconds')
        
        # Error metrics
        self.api_errors = Counter('trading_api_errors', 'Number of API errors')
        self.system_errors = Counter('trading_system_errors', 'Number of system errors')
        
    async def start_monitoring(self):
        """Start the monitoring service"""
        try:
            # Start Prometheus metrics server
            start_http_server(self.config.get('PROMETHEUS_PORT', 8000))
            
            # Start monitoring loops
            await asyncio.gather(
                self.monitor_system_health(),
                self.monitor_trading_health(),
                self.monitor_api_health()
            )
            
        except Exception as e:
            logging.error(f"Error starting monitoring: {str(e)}")
            
    async def monitor_system_health(self):
        """Monitor system resources"""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent()
                self.cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.disk_usage.set(disk.percent)
                
                # Check resource thresholds
                if cpu_percent > 80 or memory.percent > 80 or disk.percent > 80:
                    await self._handle_resource_warning()
                    
            except Exception as e:
                logging.error(f"Error monitoring system health: {str(e)}")
                self.system_errors.inc()
                
            await asyncio.sleep(60)  # Check every minute
            
    async def monitor_trading_health(self):
        """Monitor trading metrics"""
        while True:
            try:
                # Get account info
                account_info = mt5.account_info()
                if account_info:
                    self.account_balance.set(account_info.balance)
                    
                # Get positions
                positions = mt5.positions_get()
                if positions is not None:
                    self.open_positions.set(len(positions))
                    
                # Calculate daily P&L
                today_start = datetime.now().replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                history = mt5.history_deals_get(today_start, datetime.now())
                if history is not None:
                    daily_pnl = sum(deal.profit for deal in history)
                    self.daily_pnl.set(daily_pnl)
                    
            except Exception as e:
                logging.error(f"Error monitoring trading health: {str(e)}")
                self.system_errors.inc()
                
            await asyncio.sleep(60)  # Check every minute
            
    async def monitor_api_health(self):
        """Monitor API health and latency"""
        while True:
            try:
                start_time = datetime.now()
                
                # Check MT5 connection
                if not mt5.initialize():
                    self.api_health = False
                    self.api_errors.inc()
                    await self._handle_api_error("MT5 initialization failed")
                else:
                    self.api_health = True
                    
                # Measure latency
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self.api_latency.set(latency)
                
                # Check latency threshold
                if latency > 1000:  # 1 second
                    await self._handle_api_warning(f"High API latency: {latency}ms")
                    
                self.last_api_check = datetime.now()
                
            except Exception as e:
                logging.error(f"Error monitoring API health: {str(e)}")
                self.api_errors.inc()
                self.api_health = False
                
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def _handle_resource_warning(self):
        """Handle system resource warnings"""
        warning_msg = "System resource usage high:\n"
        warning_msg += f"CPU: {psutil.cpu_percent()}%\n"
        warning_msg += f"Memory: {psutil.virtual_memory().percent}%\n"
        warning_msg += f"Disk: {psutil.disk_usage('/').percent}%"
        
        logging.warning(warning_msg)
        # Implement notification logic here
        
    async def _handle_api_error(self, error_msg: str):
        """Handle API errors"""
        logging.error(f"API Error: {error_msg}")
        # Implement notification logic here
        
    async def _handle_api_warning(self, warning_msg: str):
        """Handle API warnings"""
        logging.warning(f"API Warning: {warning_msg}")
        # Implement notification logic here
        
    def get_health_status(self) -> Dict:
        """Get current health status"""
        return {
            'system_health': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            },
            'api_health': {
                'status': self.api_health,
                'last_check': self.last_api_check.isoformat(),
                'latency': self.api_latency._value.get()
            },
            'trading_health': {
                'account_balance': self.account_balance._value.get(),
                'open_positions': self.open_positions._value.get(),
                'daily_pnl': self.daily_pnl._value.get()
            },
            'errors': {
                'api_errors': self.api_errors._value.get(),
                'system_errors': self.system_errors._value.get()
            }
        }
