import logging
from typing import Dict, Optional, List
import asyncio
from .sentiment_analyzer import SentimentAnalyzer
from .news_analyzer import NewsAnalyzer
from .advanced_nlp import AdvancedNLP
from .market_events import MarketEventAnalyzer

class MarketSentiment:
    def __init__(self, config: Dict):
        """Initialize market sentiment analyzer"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.sentiment_analyzer = SentimentAnalyzer(config)
        self.news_analyzer = NewsAnalyzer(config)
        self.advanced_nlp = AdvancedNLP(config)
        self.event_analyzer = MarketEventAnalyzer(config)
        
    async def analyze_sentiment(self, symbol: str) -> Dict:
        """Analyze market sentiment for a symbol"""
        try:
            # Fetch recent news
            news_items = await self.news_analyzer.fetch_news(symbol)
            recent_news = self.news_analyzer.filter_recent_news(news_items, hours=24)
            
            if not recent_news:
                self.logger.warning(f"No recent news found for {symbol}")
                return {
                    'sentiment': 0,
                    'confidence': 0,
                    'bias': {'direction': 0, 'strength': 0}
                }
                
            # Analyze news sentiment (traditional)
            basic_sentiment = self.sentiment_analyzer.analyze_news(recent_news)
            
            # Get advanced NLP analysis
            advanced_results = []
            for news in recent_news[:5]:  # Analyze top 5 recent news
                title = news.get('title', '')
                content = news.get('content', '')
                full_text = f"{title} {content}"
                
                # Get FinBERT sentiment
                finbert_sentiment = self.advanced_nlp.get_financial_sentiment(full_text)
                
                # Get market entities
                entities = self.advanced_nlp.extract_market_entities(full_text)
                
                # Get dependency patterns
                patterns = self.advanced_nlp.analyze_dependencies(full_text)
                
                advanced_results.append({
                    'finbert': finbert_sentiment,
                    'entities': entities,
                    'patterns': patterns
                })
                
            # Analyze market events
            event_analysis = await self.event_analyzer.analyze_market_events(recent_news)
            
            # Get event summary
            event_summary = self.event_analyzer.get_event_summary()
            
            # Combine all analyses
            combined_sentiment = self._combine_sentiment_scores(
                basic_sentiment,
                advanced_results,
                event_analysis
            )
            
            # Get market bias
            market_bias = self._calculate_market_bias(
                combined_sentiment,
                event_analysis,
                advanced_results
            )
            
            return {
                'sentiment': combined_sentiment['sentiment'],
                'confidence': combined_sentiment['confidence'],
                'bias': market_bias,
                'news_count': len(recent_news),
                'event_summary': event_summary,
                'entities': self._aggregate_entities(advanced_results),
                'patterns': self._aggregate_patterns(advanced_results),
                'latest_news': {
                    'title': recent_news[0].get('title', ''),
                    'source': recent_news[0].get('source', ''),
                    'url': recent_news[0].get('url', ''),
                    'sentiment': advanced_results[0]['finbert'] if advanced_results else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {str(e)}")
            return {
                'sentiment': 0,
                'confidence': 0,
                'bias': {'direction': 0, 'strength': 0}
            }
            
    def _combine_sentiment_scores(self, basic: Dict, advanced: List[Dict], events: Dict) -> Dict:
        """Combine different sentiment scores"""
        try:
            # Get average FinBERT sentiment
            finbert_sentiments = []
            for result in advanced:
                sentiment = result['finbert']
                score = sentiment['scores']['positive'] - sentiment['scores']['negative']
                finbert_sentiments.append(score * sentiment['confidence'])
                
            avg_finbert = sum(finbert_sentiments) / len(finbert_sentiments) if finbert_sentiments else 0
            
            # Combine with weights
            final_sentiment = (
                basic['sentiment'] * 0.3 +  # Basic VADER sentiment
                avg_finbert * 0.4 +         # FinBERT sentiment
                events['impact'] * 0.3       # Event impact
            )
            
            # Calculate confidence
            confidence = (
                basic['confidence'] * 0.3 +
                sum(r['finbert']['confidence'] for r in advanced) / len(advanced) * 0.4 +
                events['confidence'] * 0.3
            ) if advanced else basic['confidence']
            
            return {
                'sentiment': final_sentiment,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error combining sentiment scores: {str(e)}")
            return {'sentiment': 0, 'confidence': 0}
            
    def _calculate_market_bias(self, sentiment: Dict, events: Dict, advanced: List[Dict]) -> Dict:
        """Calculate market bias from all analyses"""
        try:
            # Base direction from sentiment
            direction = 1 if sentiment['sentiment'] > 0.2 else -1 if sentiment['sentiment'] < -0.2 else 0
            
            # Adjust strength based on multiple factors
            base_strength = abs(sentiment['sentiment'])
            
            # Adjust for event impact
            event_factor = abs(events['impact'])
            
            # Adjust for entity presence
            entity_boost = 0
            for result in advanced:
                if result['entities'].get('ORG') or result['entities'].get('MONEY'):
                    entity_boost += 0.1
                    
            # Calculate final strength
            strength = min(base_strength * (1 + event_factor) * (1 + entity_boost), 1.0)
            
            return {
                'direction': direction,
                'strength': strength,
                'confidence': sentiment['confidence'],
                'event_impact': events['impact'],
                'event_count': events['event_count']
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating market bias: {str(e)}")
            return {'direction': 0, 'strength': 0}
            
    def _aggregate_entities(self, results: List[Dict]) -> Dict:
        """Aggregate entities from multiple analyses"""
        try:
            aggregated = {'ORG': set(), 'MONEY': set(), 'PERCENT': set()}
            
            for result in results:
                for entity_type, entities in result['entities'].items():
                    aggregated[entity_type].update(entities)
                    
            return {k: list(v) for k, v in aggregated.items()}
            
        except Exception as e:
            self.logger.error(f"Error aggregating entities: {str(e)}")
            return {}
            
    def _aggregate_patterns(self, results: List[Dict]) -> List[Dict]:
        """Aggregate patterns from multiple analyses"""
        try:
            all_patterns = []
            for result in results:
                all_patterns.extend(result['patterns'])
            return all_patterns[:10]  # Return top 10 patterns
            
        except Exception as e:
            self.logger.error(f"Error aggregating patterns: {str(e)}")
            return []
            
    def adjust_signal(self, original_signal: Dict, sentiment_data: Dict) -> Dict:
        """Adjust trading signal based on sentiment"""
        try:
            if not original_signal or not sentiment_data:
                return original_signal
                
            # Get sentiment bias
            bias = sentiment_data.get('bias', {})
            sentiment_direction = bias.get('direction', 0)
            sentiment_strength = bias.get('strength', 0)
            event_impact = bias.get('event_impact', 0)
            
            # Get original signal parameters
            signal_direction = original_signal.get('direction', 0)
            signal_strength = original_signal.get('strength', 0)
            
            # Calculate alignment score
            alignment = (
                1.2 if sentiment_direction == signal_direction and abs(signal_direction) > 0
                else 0.8 if sentiment_direction != 0 and sentiment_direction != signal_direction
                else 1.0
            )
            
            # Adjust strength based on sentiment and events
            event_factor = 1 + abs(event_impact) * 0.3  # Max 30% impact from events
            new_strength = min(
                signal_strength * alignment * event_factor * (1 + sentiment_strength * 0.5),
                1.0
            )
            
            # Update signal
            adjusted_signal = original_signal.copy()
            adjusted_signal['strength'] = new_strength
            adjusted_signal['sentiment_aligned'] = sentiment_direction == signal_direction
            adjusted_signal['sentiment_data'] = {
                'direction': sentiment_direction,
                'strength': sentiment_strength,
                'event_impact': event_impact,
                'entities': sentiment_data.get('entities', {}),
                'latest_news': sentiment_data.get('latest_news', {})
            }
            
            return adjusted_signal
            
        except Exception as e:
            self.logger.error(f"Error adjusting signal: {str(e)}")
            return original_signal
