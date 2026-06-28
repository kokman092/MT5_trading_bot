import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class NewsProvider:
    def __init__(self, config: Dict):
        """Initialize news provider"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.forex_factory_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        self.last_update = None
        self.cache_duration = timedelta(hours=1)
        self.cached_news = []
        
    def get_news_events(self) -> List[Dict]:
        """Get economic news events"""
        try:
            # Check if we need to update
            now = datetime.now()
            if (self.last_update and 
                now - self.last_update < self.cache_duration and 
                self.cached_news):
                return self.cached_news
                
            # Fetch news from Forex Factory
            response = requests.get(self.forex_factory_url)
            if response.status_code == 200:
                data = response.json()
                
                # Process news events
                processed_events = []
                for event in data:
                    try:
                        # Convert timestamp to datetime
                        if isinstance(event.get('timestamp'), (int, float)):
                            timestamp = int(event['timestamp'])
                        else:
                            # Try to parse date string
                            date_str = event.get('date')
                            if not date_str:
                                continue
                                
                            try:
                                dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                                timestamp = int(dt.timestamp())
                            except ValueError:
                                continue
                                
                        # Create processed event
                        processed_event = {
                            'timestamp': timestamp,
                            'title': event.get('title', ''),
                            'impact': event.get('impact', 'Low'),
                            'forecast': event.get('forecast'),
                            'previous': event.get('previous'),
                            'currency': event.get('currency', '')
                        }
                        processed_events.append(processed_event)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing news event: {str(e)}")
                        continue
                        
                # Update cache
                self.cached_news = processed_events
                self.last_update = now
                return processed_events
                
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching news events: {str(e)}")
            return []
