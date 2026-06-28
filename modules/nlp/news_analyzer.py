import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
import re

class NewsAnalyzer:
    def __init__(self, config: Dict):
        """Initialize news analyzer"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    async def fetch_news(self, symbol: str) -> List[Dict]:
        """Fetch news for a currency pair"""
        try:
            # Convert symbol to currency names
            currencies = self._symbol_to_currencies(symbol)
            if not currencies:
                return []
                
            news_items = []
            async with aiohttp.ClientSession(headers=self.headers) as session:
                # Fetch from multiple sources
                tasks = [
                    self._fetch_forex_factory(session, currencies),
                    self._fetch_investing_com(session, currencies),
                    self._fetch_fxstreet(session, currencies)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Combine results
                for result in results:
                    if isinstance(result, list):
                        news_items.extend(result)
                        
            return self._deduplicate_news(news_items)
            
        except Exception as e:
            self.logger.error(f"Error fetching news: {str(e)}")
            return []
            
    def _symbol_to_currencies(self, symbol: str) -> List[str]:
        """Convert MT5 symbol to currency names"""
        currency_names = {
            'USD': 'dollar',
            'EUR': 'euro',
            'GBP': 'pound',
            'JPY': 'yen',
            'AUD': 'australian dollar',
            'NZD': 'new zealand dollar',
            'CAD': 'canadian dollar',
            'CHF': 'swiss franc'
        }
        
        if len(symbol) != 6:
            return []
            
        base = symbol[:3]
        quote = symbol[3:]
        
        currencies = []
        if base in currency_names:
            currencies.append(currency_names[base])
        if quote in currency_names:
            currencies.append(currency_names[quote])
            
        return currencies
        
    async def _fetch_forex_factory(self, session: aiohttp.ClientSession, currencies: List[str]) -> List[Dict]:
        """Fetch news from ForexFactory"""
        try:
            url = "https://www.forexfactory.com/news"
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                    
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                
                news_items = []
                for item in soup.select('.news_item'):
                    title = item.select_one('.title')
                    if not title:
                        continue
                        
                    # Check if news is related to our currencies
                    title_text = title.text.lower()
                    if not any(currency in title_text for currency in currencies):
                        continue
                        
                    news_items.append({
                        'title': title_text,
                        'source': 'ForexFactory',
                        'timestamp': datetime.now(),
                        'url': f"https://www.forexfactory.com{title.get('href', '')}"
                    })
                    
                return news_items
                
        except Exception as e:
            self.logger.error(f"Error fetching from ForexFactory: {str(e)}")
            return []
            
    async def _fetch_investing_com(self, session: aiohttp.ClientSession, currencies: List[str]) -> List[Dict]:
        """Fetch news from Investing.com"""
        try:
            url = "https://www.investing.com/currencies/streaming-forex-news"
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                    
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                
                news_items = []
                for item in soup.select('.articleItem'):
                    title = item.select_one('.title')
                    if not title:
                        continue
                        
                    title_text = title.text.lower()
                    if not any(currency in title_text for currency in currencies):
                        continue
                        
                    news_items.append({
                        'title': title_text,
                        'source': 'Investing.com',
                        'timestamp': datetime.now(),
                        'url': f"https://www.investing.com{title.get('href', '')}"
                    })
                    
                return news_items
                
        except Exception as e:
            self.logger.error(f"Error fetching from Investing.com: {str(e)}")
            return []
            
    async def _fetch_fxstreet(self, session: aiohttp.ClientSession, currencies: List[str]) -> List[Dict]:
        """Fetch news from FXStreet"""
        try:
            url = "https://www.fxstreet.com/news"
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                    
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                
                news_items = []
                for item in soup.select('.fxs_article_title'):
                    title_text = item.text.lower()
                    if not any(currency in title_text for currency in currencies):
                        continue
                        
                    news_items.append({
                        'title': title_text,
                        'source': 'FXStreet',
                        'timestamp': datetime.now(),
                        'url': item.get('href', '')
                    })
                    
                return news_items
                
        except Exception as e:
            self.logger.error(f"Error fetching from FXStreet: {str(e)}")
            return []
            
    def _deduplicate_news(self, news_items: List[Dict]) -> List[Dict]:
        """Remove duplicate news items"""
        seen_titles = set()
        unique_items = []
        
        for item in news_items:
            title = item['title'].lower()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_items.append(item)
                
        return unique_items
        
    def filter_recent_news(self, news_items: List[Dict], hours: int = 24) -> List[Dict]:
        """Filter news items from last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [item for item in news_items if item['timestamp'] > cutoff]
