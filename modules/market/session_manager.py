import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, time, timedelta
import logging
import aiohttp
import pytz
import MetaTrader5 as mt5
from dataclasses import dataclass
from .news_provider import NewsProvider

@dataclass
class NewsEvent:
    timestamp: datetime
    currency: str
    impact: str  # 'High', 'Medium', 'Low'
    title: str
    forecast: Optional[float]
    previous: Optional[float]

class SessionManager:
    """Manages trading sessions and market hours"""
    
    def __init__(self, config: Dict):
        """Initialize session manager with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get timezone from config or use UTC
        self.timezone = pytz.timezone(config.get('server_timezone', 'UTC'))
        
        # Initialize session times
        self.sessions = config.get('trading', {}).get('sessions', {})
        
    def is_trading_session(self) -> bool:
        """Check if current time is within any trading session"""
        try:
            # Get current time in configured timezone
            current_time = datetime.now(self.timezone).time()
            
            # Check each session
            for session_name, session_info in self.sessions.items():
                start_str = session_info.get('start', '00:00')
                end_str = session_info.get('end', '23:59')
                
                # Convert string times to time objects
                start_time = datetime.strptime(start_str, '%H:%M').time()
                end_time = datetime.strptime(end_str, '%H:%M').time()
                
                # Check if current time is within session
                if self._is_time_between(current_time, start_time, end_time):
                    self.logger.info(f"Currently in {session_name} session")
                    return True
                    
            self.logger.info("Not currently in any trading session")
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking trading session: {str(e)}")
            return False
            
    def _is_time_between(self, current: time, start: time, end: time) -> bool:
        """Check if current time is between start and end times"""
        if start <= end:
            return start <= current <= end
        else:  # Over midnight
            return start <= current or current <= end
            
    def get_active_pairs(self) -> List[str]:
        """Get active pairs for current session"""
        try:
            # Get current time in configured timezone
            current_time = datetime.now(self.timezone).time()
            active_pairs = set()
            
            # Check each session
            for session_name, session_info in self.sessions.items():
                start_str = session_info.get('start', '00:00')
                end_str = session_info.get('end', '23:59')
                
                # Convert string times to time objects
                start_time = datetime.strptime(start_str, '%H:%M').time()
                end_time = datetime.strptime(end_str, '%H:%M').time()
                
                # If current time is in this session, add its pairs
                if self._is_time_between(current_time, start_time, end_time):
                    session_pairs = session_info.get('pairs', [])
                    active_pairs.update(session_pairs)
                    self.logger.info(f"Currently in {session_name} session")
                    
            return list(active_pairs)
            
        except Exception as e:
            self.logger.error(f"Error getting active pairs: {str(e)}")
            return []
            
class MarketSessionManager:
    def __init__(self, config: Dict):
        """Initialize market session manager"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.last_news_update = None
        self.news_events = []
        self.forex_factory_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        self.news_provider = NewsProvider(config)  # Initialize news provider
        
        # Initialize MT5
        if not mt5.initialize():
            self.logger.error("Failed to initialize MT5")
            raise RuntimeError("MT5 initialization failed")
            
        # Enable market data
        symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
            'NZDUSD', 'USDCHF', 'EURJPY', 'GBPJPY', 'EURGBP'
        ]
        for symbol in symbols:
            if not mt5.symbol_select(symbol, True):
                self.logger.warning(f"Failed to select {symbol}")
                
        self.session_manager = SessionManager(config)
        
    def __del__(self):
        """Cleanup MT5 connection"""
        mt5.shutdown()
        
    def is_trading_session(self, symbol: str) -> bool:
        """Check if current time is within allowed trading sessions"""
        try:
            current_time = datetime.now(pytz.UTC)
            current_weekday = current_time.weekday()
            
            # Don't trade on weekends
            if current_weekday >= 5:  # Saturday = 5, Sunday = 6
                return False
                
            # Convert current time to server timezone
            server_time = current_time.astimezone(
                pytz.timezone(self.config['server_timezone'])
            )
            current_time = server_time.time()
            
            # Check each trading session
            sessions = self.config['trading']['sessions']
            for session_name, session_times in sessions.items():
                start_time = datetime.strptime(session_times['start'], '%H:%M').time()
                end_time = datetime.strptime(session_times['end'], '%H:%M').time()
                
                # Handle sessions that cross midnight
                if start_time > end_time:
                    if current_time >= start_time or current_time <= end_time:
                        return True
                else:
                    if start_time <= current_time <= end_time:
                        return True
                        
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking trading session: {str(e)}")
            return False
            
    def is_high_liquidity(self, symbol: str) -> bool:
        """Check if current time is a high liquidity period for the symbol"""
        try:
            # Get currency pairs from symbol
            base_currency = symbol[:3]
            quote_currency = symbol[3:]
            
            # Define peak liquidity hours for major currency centers
            liquidity_centers = {
                'USD': {'start': '13:00', 'end': '21:00'},  # New York
                'GBP': {'start': '08:00', 'end': '16:00'},  # London
                'EUR': {'start': '08:00', 'end': '16:00'},  # London/Frankfurt
                'JPY': {'start': '00:00', 'end': '08:00'},  # Tokyo
                'AUD': {'start': '22:00', 'end': '06:00'},  # Sydney
                'NZD': {'start': '22:00', 'end': '06:00'}   # Wellington
            }
            
            # Check if either currency is in peak hours
            current_time = datetime.now(pytz.UTC).time()
            
            for currency in [base_currency, quote_currency]:
                if currency in liquidity_centers:
                    center = liquidity_centers[currency]
                    start_time = datetime.strptime(center['start'], '%H:%M').time()
                    end_time = datetime.strptime(center['end'], '%H:%M').time()
                    
                    if start_time <= current_time <= end_time:
                        return True
                        
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking liquidity: {str(e)}")
            return False
            
    def update_news_events(self):
        """Update economic news events"""
        try:
            self.news_events = self.news_provider.get_news_events()
            self.last_news_update = datetime.now()
        except Exception as e:
            self.logger.error(f"Error updating news events: {str(e)}")
            
    def is_news_event_soon(self, symbol: str, minutes_before: int = 30, minutes_after: int = 30) -> bool:
        """Check if there's a high-impact news event soon for the symbol's currencies"""
        try:
            if not self.news_events:
                return False
                
            # Get currencies from symbol
            base_currency = symbol[:3]
            quote_currency = symbol[3:]
            
            current_time = datetime.now(pytz.UTC)
            window_start = current_time - timedelta(minutes=minutes_before)
            window_end = current_time + timedelta(minutes=minutes_after)
            
            # Check each news event
            for event in self.news_events:
                if event['impact'] != 'High':
                    continue
                    
                if event['currency'] in [base_currency, quote_currency]:
                    event_time = datetime.fromtimestamp(event['timestamp'])
                    if window_start <= event_time <= window_end:
                        self.logger.warning(
                            f"High impact news event soon for {symbol}: "
                            f"{event['event']} at {event_time}"
                        )
                        return True
                        
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking news events: {str(e)}")
            return True  # Be safe and avoid trading if there's an error
            
    def get_volatility_adjustment(self, symbol: str) -> float:
        """Get position size adjustment based on session volatility"""
        try:
            # Default adjustment
            adjustment = 1.0
            
            # Reduce position size during Asian session (typically lower volatility)
            if self.session_manager.is_trading_session():
                session = self.session_manager.get_active_pairs()
                adjustment *= 1.0
                
            # Increase position size during London/NY overlap (higher volatility)
            if self.session_manager.get_active_pairs():
                adjustment *= 1.3
                
            # Reduce position size near news events
            if self.is_news_event_soon(symbol, minutes_before=15, minutes_after=15):
                adjustment *= 0.5
                
            return min(max(adjustment, 0.5), 1.5)  # Cap between 0.5 and 1.5
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility adjustment: {str(e)}")
            return 1.0  # Return default adjustment on error
            
    def is_asian_session(self) -> bool:
        """Check if current time is in Asian session"""
        return 'EURUSD' in self.session_manager.get_active_pairs()
        
    def is_london_session(self) -> bool:
        """Check if current time is in London session"""
        return 'GBPUSD' in self.session_manager.get_active_pairs()
        
    def is_newyork_session(self) -> bool:
        """Check if current time is in New York session"""
        return 'USDJPY' in self.session_manager.get_active_pairs()
        
    def is_london_ny_overlap(self) -> bool:
        """Check if current time is in London/NY overlap period"""
        return self.is_london_session() and self.is_newyork_session()
        
    def should_trade(self, current_time: datetime) -> bool:
        """Check if we should trade at the current time"""
        try:
            # Don't trade on weekends
            if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                self.logger.info("Not trading on weekend")
                return False
                
            # Check if within trading session
            if not self.session_manager.is_trading_session():
                self.logger.debug("Outside trading session")
                return False
                
            # Update news events if needed
            if (not self.last_news_update or 
                current_time - self.last_news_update > timedelta(hours=1)):
                self.update_news_events()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking if should trade: {str(e)}")
            return False
            
    def get_market_state(self, symbol: str) -> Dict:
        """Get current market state for a symbol"""
        try:
            # Get current time
            now = datetime.now()
            
            # Get market data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
            if rates is None:
                self.logger.error(f"Failed to get rates for {symbol}")
                return {}
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Calculate volatility
            df['returns'] = df['close'].pct_change()
            volatility = df['returns'].std() * np.sqrt(252)
            
            # Calculate trend
            sma20 = df['close'].rolling(window=20).mean().iloc[-1]
            sma50 = df['close'].rolling(window=50).mean().iloc[-1]
            trend = 'up' if sma20 > sma50 else 'down'
            
            # Get spread
            symbol_info = mt5.symbol_info(symbol)
            spread = symbol_info.spread if symbol_info else 0
            
            # Get session info
            current_session = None
            if self.is_asian_session():
                current_session = 'asian'
            elif self.is_london_session():
                current_session = 'london'
            elif self.is_newyork_session():
                current_session = 'new_york'
                
            # Get news events
            has_news = self.is_news_event_soon(symbol)
            
            return {
                'symbol': symbol,
                'timestamp': now,
                'volatility': volatility,
                'trend': trend,
                'spread': spread,
                'session': current_session,
                'has_news': has_news,
                'high_liquidity': self.is_high_liquidity(symbol)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market state: {str(e)}")
            return {}
