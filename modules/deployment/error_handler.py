import logging
import traceback
from typing import Dict, Optional, Callable
from datetime import datetime
import asyncio
import telegram
from prometheus_client import Counter, Gauge
from functools import wraps

class ErrorHandler:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_metrics()
        self.telegram_bot = self._setup_telegram() if config.get('TELEGRAM_BOT_TOKEN') else None
        self.error_count = 0
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
    def setup_metrics(self):
        """Initialize Prometheus metrics"""
        self.error_counter = Counter(
            'trading_errors_total',
            'Total number of trading errors',
            ['error_type']
        )
        self.error_rate = Gauge(
            'trading_error_rate',
            'Rate of trading errors per minute'
        )
        
    def _setup_telegram(self) -> Optional[telegram.Bot]:
        """Setup Telegram bot for notifications"""
        try:
            return telegram.Bot(token=self.config['TELEGRAM_BOT_TOKEN'])
        except Exception as e:
            logging.error(f"Failed to initialize Telegram bot: {str(e)}")
            return None
            
    async def handle_error(
        self,
        error: Exception,
        context: str,
        retry_func: Optional[Callable] = None
    ) -> bool:
        """
        Handle trading errors with retry logic and notifications
        Returns: True if error was handled successfully
        """
        error_type = type(error).__name__
        self.error_counter.labels(error_type=error_type).inc()
        self.error_count += 1
        
        error_msg = f"Error in {context}: {str(error)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        
        # Send notification
        await self.send_notification(error_msg, is_error=True)
        
        # Check if we should retry
        if retry_func and self.error_count <= self.max_retries:
            logging.info(f"Retrying {context} (Attempt {self.error_count}/{self.max_retries})")
            await asyncio.sleep(self.retry_delay * self.error_count)
            try:
                await retry_func()
                self.error_count = 0
                return True
            except Exception as retry_error:
                logging.error(f"Retry failed: {str(retry_error)}")
                
        # Check if we need to emergency stop
        if self.should_emergency_stop():
            await self.emergency_stop()
            return False
            
        return False
        
    def should_emergency_stop(self) -> bool:
        """Determine if trading should be stopped due to errors"""
        # Stop if too many errors in short time
        if self.error_count >= self.max_retries:
            return True
            
        # Stop if critical error types
        critical_errors = [
            'InsufficientFunds',
            'AccountLocked',
            'APIError',
            'NetworkError'
        ]
        return any(err in str(traceback.format_exc()) for err in critical_errors)
        
    async def emergency_stop(self):
        """Emergency stop all trading activities"""
        error_msg = "EMERGENCY STOP triggered due to critical errors"
        logging.critical(error_msg)
        await self.send_notification(error_msg, is_error=True)
        
        # Close all positions
        try:
            # Implement position closing logic here
            pass
        except Exception as e:
            logging.error(f"Error during emergency stop: {str(e)}")
            
    async def send_notification(self, message: str, is_error: bool = False):
        """Send notification via configured channels"""
        try:
            if self.telegram_bot and self.config.get('TELEGRAM_CHAT_ID'):
                prefix = "🚨 ERROR" if is_error else "ℹ️ INFO"
                message = f"{prefix}: {message}"
                await self.telegram_bot.send_message(
                    chat_id=self.config['TELEGRAM_CHAT_ID'],
                    text=message
                )
        except Exception as e:
            logging.error(f"Failed to send notification: {str(e)}")
            
    def error_handler_decorator(self, context: str):
        """Decorator for error handling"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    await self.handle_error(
                        e,
                        context,
                        lambda: func(*args, **kwargs)
                    )
                    return None
            return wrapper
        return decorator
        
    async def check_api_health(self) -> bool:
        """Check if API connections are healthy"""
        try:
            # Implement API health checks here
            return True
        except Exception as e:
            await self.handle_error(e, "API Health Check")
            return False
            
    def log_metric(self, metric_name: str, value: float):
        """Log custom metric to Prometheus"""
        try:
            gauge = Gauge(f'trading_{metric_name}', f'Custom metric: {metric_name}')
            gauge.set(value)
        except Exception as e:
            logging.error(f"Error logging metric: {str(e)}")
            
    async def monitor_error_rate(self):
        """Monitor and update error rate metric"""
        while True:
            try:
                self.error_rate.set(self.error_count)
                self.error_count = 0
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logging.error(f"Error monitoring error rate: {str(e)}")
                await asyncio.sleep(60)
