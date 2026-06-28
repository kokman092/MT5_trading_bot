import logging
from typing import Dict, Optional, Callable, Any
import functools
import asyncio
from datetime import datetime, timedelta
import traceback
import MetaTrader5 as mt5
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import wraps

logger = logging.getLogger('trading_bot')

class MT5Error(Exception):
    """Custom exception for MT5 errors"""
    pass

class ErrorHandler:
    def __init__(self, config: Dict):
        # Add default config if missing
        if 'error_handling' not in config:
            config['error_handling'] = {
                'max_retries': 3,
                'backoff_factor': 2,
                'recovery_timeout': 30,
                'notification_threshold': 5
            }
        if 'email' not in config:
            config['email'] = {
                'enabled': False
            }
            
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}
        self.last_errors = {}
        self.recovery_attempts = {}
        self.max_retries = config['error_handling'].get('max_retries', 3)
        self.backoff_factor = config['error_handling'].get('backoff_factor', 2)
        self.recovery_timeout = config['error_handling'].get('recovery_timeout', 30)
        
    @staticmethod
    def handle_trading_error(func: Callable) -> Callable:
        """
        A decorator to handle trading errors with improved logging and recovery
        """
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except MT5Error as e:
                error_key = f"{func.__name__}_{type(e).__name__}"
                self.logger.error(f"MT5 error in {func.__name__}: {str(e)}")
                
                if await self._should_attempt_recovery(error_key):
                    await self._recover_from_error(error_key, e, *args, **kwargs)
                    # Retry the function after recovery
                    try:
                        return await func(self, *args, **kwargs)
                    except Exception as retry_e:
                        self.logger.error(f"Retry failed after recovery: {str(retry_e)}")
                        
                await self._send_error_notification(error_key, e)
                return False
            except Exception as e:
                self.logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return False
        return wrapper
        
    async def _should_attempt_recovery(self, error_key: str) -> bool:
        """Check if we should attempt recovery for this error"""
        try:
            # Get error count and last attempt time
            count = self.error_counts.get(error_key, 0)
            last_attempt = self.last_errors.get(error_key)
            
            # If we've exceeded max retries, don't attempt recovery
            if count > self.max_retries:
                return False
                
            # If this is the first error, attempt recovery
            if not last_attempt:
                return True
                
            # Calculate time since last attempt
            time_since_last = datetime.now() - last_attempt
            required_wait = timedelta(seconds=self.backoff_factor ** (count - 1))
            
            return time_since_last >= required_wait
            
        except Exception as e:
            self.logger.error(f"Error in _should_attempt_recovery: {str(e)}")
            return False
            
    async def _recover_from_error(self, error_key: str, error: Exception, *args, **kwargs) -> None:
        """Attempt to recover from an error"""
        try:
            # Increment recovery attempts
            self.recovery_attempts[error_key] = self.recovery_attempts.get(error_key, 0) + 1
            
            # Log recovery attempt
            self.logger.info(f"Attempting recovery for error: {error_key}")
            
            # Basic recovery steps
            if isinstance(error, MT5Error):
                # For MT5 errors, try to reinitialize the connection
                if not mt5.initialize():
                    self.logger.error("Failed to reinitialize MT5 connection")
                    return
                    
                # Try to relogin if needed
                if 'account' in self.config:
                    account_info = self.config['account']
                    if not mt5.login(
                        login=int(account_info['login']),
                        password=account_info['password'],
                        server=account_info['server']
                    ):
                        self.logger.error("Failed to relogin to MT5")
                        return
                        
            # Add more recovery steps as needed
            
        except Exception as e:
            self.logger.error(f"Error in recovery attempt: {str(e)}")
            
    async def _send_error_notification(self, error_key: str, error: Exception) -> None:
        """Send email notification about the error"""
        try:
            if not self.config['email']['enabled']:
                return
                
            email_config = self.config['email']
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = email_config['to']
            msg['Subject'] = f"Trading Bot Error: {error_key}"
            
            body = f"""
            Error occurred in trading bot:
            
            Error Key: {error_key}
            Error Message: {str(error)}
            Time: {datetime.now()}
            Stack Trace:
            {traceback.format_exc()}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
                
        except Exception as e:
            self.logger.error(f"Failed to send error notification: {str(e)}")

    async def recover_connection(self) -> bool:
        """Attempt to recover MT5 connection"""
        try:
            # Shutdown existing connection
            mt5.shutdown()
            await asyncio.sleep(1)
            
            # Reinitialize
            if not mt5.initialize():
                self.logger.error("Failed to reinitialize MT5")
                return False
                
            # Relogin if needed
            if 'account' in self.config:
                account = self.config['account']
                if not mt5.login(
                    login=int(account['login']),
                    password=account['password'],
                    server=account['server']
                ):
                    self.logger.error("Failed to relogin to MT5")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Recovery error: {str(e)}")
            return False
