import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from textblob import TextBlob
import tweepy
import newsapi
from transformers import pipeline
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class EnhancedSentimentAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize API clients
        self._init_apis()
        
        # Initialize ML models
        self._init_models()
        
        # Sentiment cache
        self.sentiment_cache = {}
        self.cache_duration = timedelta(minutes=15)
        
    def _init_apis(self):
        """Initialize API clients for different data sources"""
        try:
            # Twitter API
            self.twitter_client = tweepy.Client(
                bearer_token=self.config['twitter_bearer_token'],
                consumer_key=self.config['twitter_api_key'],
                consumer_secret=self.config['twitter_api_secret'],
                access_token=self.config['twitter_access_token'],
                access_token_secret=self.config['twitter_access_token_secret']
            )
            
            # News API
            self.news_client = newsapi.NewsApiClient(
                api_key=self.config['news_api_key']
            )
            
            # Reddit API (using PRAW)
            self.reddit_client = None  # Initialize based on config
            
        except Exception as e:
            self.logger.error(f"Error initializing APIs: {str(e)}")
            
    def _init_models(self):
        """Initialize ML models for sentiment analysis"""
        try:
            # FinBERT for financial sentiment
            self.finbert = pipeline("sentiment-analysis", 
                                  model="ProsusAI/finbert")
            
            # VADER for social media sentiment
            self.vader = SentimentIntensityAnalyzer()
            
            # Custom models can be added here
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")

    async def analyze_market_sentiment(self, symbol: str, 
                                    timeframe: str = '1d') -> Dict:
        """
        Comprehensive market sentiment analysis
        """
        try:
            # Check cache
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.sentiment_cache:
                cached_result = self.sentiment_cache[cache_key]
                if datetime.now() - cached_result['timestamp'] < self.cache_duration:
                    return cached_result['data']
            
            # Gather data from multiple sources
            news_sentiment = await self._analyze_news_sentiment(symbol)
            social_sentiment = await self._analyze_social_sentiment(symbol)
            technical_sentiment = self._analyze_technical_sentiment(symbol)
            market_sentiment = await self._analyze_market_indicators(symbol)
            
            # Combine sentiments with weights
            combined_sentiment = self._combine_sentiments(
                news_sentiment,
                social_sentiment,
                technical_sentiment,
                market_sentiment
            )
            
            # Cache results
            self.sentiment_cache[cache_key] = {
                'timestamp': datetime.now(),
                'data': combined_sentiment
            }
            
            return combined_sentiment
            
        except Exception as e:
            self.logger.error(f"Error in market sentiment analysis: {str(e)}")
            return None

    async def _analyze_news_sentiment(self, symbol: str) -> Dict:
        """
        Analyze news sentiment using multiple sources
        """
        try:
            # Get company info
            ticker = yf.Ticker(symbol)
            company_name = ticker.info.get('longName', symbol)
            
            # Fetch news from different sources
            news_articles = []
            
            # News API
            news_response = self.news_client.get_everything(
                q=company_name,
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            news_articles.extend(news_response['articles'])
            
            # Process articles
            sentiments = []
            for article in news_articles:
                # FinBERT analysis
                finbert_result = self.finbert(article['title'] + " " + article['description'])[0]
                
                # VADER analysis
                vader_result = self.vader.polarity_scores(
                    article['title'] + " " + article['description']
                )
                
                sentiments.append({
                    'source': article['source']['name'],
                    'timestamp': article['publishedAt'],
                    'finbert_sentiment': finbert_result['label'],
                    'finbert_score': finbert_result['score'],
                    'vader_compound': vader_result['compound']
                })
            
            # Aggregate sentiments
            return {
                'overall_sentiment': self._aggregate_news_sentiments(sentiments),
                'source_breakdown': self._analyze_source_credibility(sentiments),
                'temporal_analysis': self._analyze_temporal_patterns(sentiments)
            }
            
        except Exception as e:
            self.logger.error(f"Error in news sentiment analysis: {str(e)}")
            return None

    async def _analyze_social_sentiment(self, symbol: str) -> Dict:
        """
        Analyze social media sentiment
        """
        try:
            sentiments = {
                'twitter': await self._analyze_twitter_sentiment(symbol),
                'reddit': await self._analyze_reddit_sentiment(symbol),
                'stocktwits': await self._analyze_stocktwits_sentiment(symbol)
            }
            
            # Combine social sentiments with platform-specific weights
            weights = {
                'twitter': 0.4,
                'reddit': 0.3,
                'stocktwits': 0.3
            }
            
            weighted_sentiment = sum(
                sentiments[platform]['score'] * weights[platform]
                for platform in sentiments
                if sentiments[platform] is not None
            )
            
            return {
                'overall_sentiment': weighted_sentiment,
                'platform_breakdown': sentiments,
                'trending_topics': self._extract_trending_topics(sentiments)
            }
            
        except Exception as e:
            self.logger.error(f"Error in social sentiment analysis: {str(e)}")
            return None

    def _analyze_technical_sentiment(self, symbol: str) -> Dict:
        """
        Analyze technical indicators for sentiment
        """
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='6mo')
            
            # Calculate technical indicators
            indicators = {
                'rsi': self._calculate_rsi(hist['Close']),
                'macd': self._calculate_macd(hist['Close']),
                'bollinger': self._calculate_bollinger_bands(hist['Close']),
                'adx': self._calculate_adx(hist)
            }
            
            # Determine technical sentiment
            sentiment_scores = {
                'rsi': self._interpret_rsi(indicators['rsi']),
                'macd': self._interpret_macd(indicators['macd']),
                'bollinger': self._interpret_bollinger(indicators['bollinger']),
                'adx': self._interpret_adx(indicators['adx'])
            }
            
            return {
                'overall_sentiment': np.mean(list(sentiment_scores.values())),
                'indicator_breakdown': sentiment_scores,
                'raw_indicators': indicators
            }
            
        except Exception as e:
            self.logger.error(f"Error in technical sentiment analysis: {str(e)}")
            return None

    async def _analyze_market_indicators(self, symbol: str) -> Dict:
        """
        Analyze broader market indicators
        """
        try:
            # Get market data
            market_data = {
                'vix': self._get_vix_data(),
                'sector_performance': self._get_sector_performance(symbol),
                'market_breadth': self._calculate_market_breadth(),
                'institutional_flows': self._get_institutional_flows(symbol)
            }
            
            # Calculate market sentiment
            sentiment = self._interpret_market_indicators(market_data)
            
            return {
                'overall_sentiment': sentiment['score'],
                'market_conditions': sentiment['conditions'],
                'risk_indicators': sentiment['risks']
            }
            
        except Exception as e:
            self.logger.error(f"Error in market indicators analysis: {str(e)}")
            return None

    def _combine_sentiments(self, news: Dict, social: Dict, 
                          technical: Dict, market: Dict) -> Dict:
        """
        Combine different sentiment sources with dynamic weights
        """
        try:
            # Base weights
            weights = {
                'news': 0.3,
                'social': 0.2,
                'technical': 0.3,
                'market': 0.2
            }
            
            # Adjust weights based on data quality and market conditions
            weights = self._adjust_weights(weights, news, social, 
                                        technical, market)
            
            # Calculate weighted sentiment
            sentiment_score = (
                news['overall_sentiment'] * weights['news'] +
                social['overall_sentiment'] * weights['social'] +
                technical['overall_sentiment'] * weights['technical'] +
                market['overall_sentiment'] * weights['market']
            )
            
            return {
                'overall_sentiment': sentiment_score,
                'component_breakdown': {
                    'news': news,
                    'social': social,
                    'technical': technical,
                    'market': market
                },
                'weights': weights,
                'timestamp': datetime.now(),
                'confidence_score': self._calculate_confidence_score(
                    news, social, technical, market
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error combining sentiments: {str(e)}")
            return None

    def _calculate_confidence_score(self, *components) -> float:
        """
        Calculate confidence score based on data quality
        """
        try:
            scores = []
            for component in components:
                if component is not None:
                    # Add quality metrics here
                    scores.append(1.0)
                    
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.0
