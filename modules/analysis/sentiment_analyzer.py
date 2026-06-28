import logging
import json
from datetime import datetime, timedelta
import aiohttp
from textblob import TextBlob
import asyncio
from typing import Dict, List, Optional

class SentimentAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_duration = timedelta(minutes=30)
        
    async def get_sentiment(self, symbol: str) -> Dict:
        """Get sentiment analysis for a symbol"""
        try:
            # Check cache first
            if symbol in self.cache:
                timestamp, sentiment = self.cache[symbol]
                if datetime.now() - timestamp < self.cache_duration:
                    return sentiment
                    
            # Fetch news and analyze sentiment
            news = await self._fetch_news(symbol)
            if not news:
                return {'score': 0, 'magnitude': 0, 'confidence': 0}
                
            # Analyze sentiment using TextBlob
            combined_text = ' '.join([item['title'] + ' ' + item['description'] for item in news])
            blob = TextBlob(combined_text)
            
            # Calculate sentiment metrics
            sentiment = {
                'score': blob.sentiment.polarity,  # Range: -1 to 1
                'magnitude': abs(blob.sentiment.polarity),  # Range: 0 to 1
                'confidence': blob.sentiment.subjectivity,  # Range: 0 to 1
                'news_count': len(news)
            }
            
            # Cache the results
            self.cache[symbol] = (datetime.now(), sentiment)
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
            return {'score': 0, 'magnitude': 0, 'confidence': 0}
            
    async def _fetch_news(self, symbol: str) -> List[Dict]:
        """Fetch financial news for a symbol"""
        try:
            base_currency = symbol[:3]
            quote_currency = symbol[3:]
            
            # Use multiple news sources for better coverage
            news = []
            
            # Fetch from ForexFactory
            forex_factory_news = await self._fetch_forex_factory(base_currency, quote_currency)
            if forex_factory_news:
                news.extend(forex_factory_news)
                
            # Add more news sources here if needed
            
            return news
            
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return []
            
    async def _fetch_forex_factory(self, base_currency: str, quote_currency: str) -> List[Dict]:
        """Fetch news from Forex Factory"""
        try:
            url = f"https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Filter relevant news
                        relevant_news = []
                        for item in data:
                            if (base_currency.lower() in item.get('currency', '').lower() or 
                                quote_currency.lower() in item.get('currency', '').lower()):
                                news_item = {
                                    'title': item.get('title', ''),
                                    'description': item.get('description', ''),
                                    'impact': item.get('impact', ''),
                                    'timestamp': item.get('date', '')
                                }
                                relevant_news.append(news_item)
                                
                        return relevant_news
                        
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching from Forex Factory: {str(e)}")
            return []
            
    def _calculate_weighted_sentiment(self, news: List[Dict]) -> float:
        """Calculate weighted sentiment based on news impact"""
        if not news:
            return 0
            
        total_weight = 0
        weighted_sentiment = 0
        
        for item in news:
            # Get sentiment using TextBlob
            blob = TextBlob(item['title'] + ' ' + item['description'])
            sentiment = blob.sentiment.polarity
            
            # Weight based on impact
            impact_weights = {
                'high': 3.0,
                'medium': 2.0,
                'low': 1.0
            }
            
            weight = impact_weights.get(item['impact'].lower(), 1.0)
            weighted_sentiment += sentiment * weight
            total_weight += weight
            
        return weighted_sentiment / total_weight if total_weight > 0 else 0
