import logging
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class SentimentAnalyzer:
    def __init__(self, config: Dict):
        """Initialize sentiment analyzer"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK data
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            self.sia = SentimentIntensityAnalyzer()
            self.stop_words = set(stopwords.words('english'))
            
        except Exception as e:
            self.logger.error(f"Error initializing NLTK: {str(e)}")
            
    def analyze_text(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        try:
            if not text:
                return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
                
            # Get sentiment scores
            scores = self.sia.polarity_scores(text)
            
            # Extract key phrases
            sentences = sent_tokenize(text)
            tokens = [word_tokenize(sentence) for sentence in sentences]
            pos_tags = [nltk.pos_tag(token) for token in tokens]
            
            # Find market-related phrases
            market_phrases = []
            for sentence_tags in pos_tags:
                for i in range(len(sentence_tags)-1):
                    if sentence_tags[i][1].startswith('JJ') and sentence_tags[i+1][1].startswith('NN'):
                        phrase = f"{sentence_tags[i][0]} {sentence_tags[i+1][0]}"
                        market_phrases.append(phrase.lower())
            
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'key_phrases': market_phrases[:5]  # Top 5 phrases
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing text: {str(e)}")
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
            
    def analyze_news(self, news_items: List[Dict]) -> Dict:
        """Analyze sentiment of multiple news items"""
        try:
            if not news_items:
                return {'sentiment': 0, 'confidence': 0}
                
            # Analyze each news item
            sentiments = []
            for item in news_items:
                title_sentiment = self.analyze_text(item.get('title', ''))
                content_sentiment = self.analyze_text(item.get('content', ''))
                
                # Weight title more heavily than content
                combined_sentiment = title_sentiment['compound'] * 0.6 + content_sentiment['compound'] * 0.4
                sentiments.append(combined_sentiment)
                
            # Calculate overall sentiment
            avg_sentiment = np.mean(sentiments)
            sentiment_std = np.std(sentiments)
            confidence = 1 - min(sentiment_std, 0.5)  # Lower std = higher confidence
            
            return {
                'sentiment': avg_sentiment,
                'confidence': confidence,
                'count': len(news_items)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing news: {str(e)}")
            return {'sentiment': 0, 'confidence': 0}
            
    def get_market_bias(self, sentiment_data: Dict) -> Dict:
        """Convert sentiment to market bias"""
        try:
            sentiment = sentiment_data.get('sentiment', 0)
            confidence = sentiment_data.get('confidence', 0)
            
            # Convert sentiment to market direction
            if abs(sentiment) < 0.2:
                direction = 0  # Neutral
            else:
                direction = 1 if sentiment > 0 else -1
                
            # Scale strength based on sentiment and confidence
            strength = abs(sentiment) * confidence
            
            return {
                'direction': direction,
                'strength': min(strength, 1.0),
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market bias: {str(e)}")
            return {'direction': 0, 'strength': 0, 'confidence': 0}
