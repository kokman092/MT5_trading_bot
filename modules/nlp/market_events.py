import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from .advanced_nlp import AdvancedNLP

class MarketEventAnalyzer:
    def __init__(self, config: Dict):
        """Initialize market event analyzer"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.nlp = AdvancedNLP(config)
        self.event_memory = pd.DataFrame(columns=['timestamp', 'event_type', 'impact', 'description'])
        
    async def analyze_market_events(self, news_items: List[Dict]) -> Dict:
        """Analyze market events from news"""
        try:
            if not news_items:
                return {'events': [], 'impact': 0, 'confidence': 0}
                
            events = []
            total_impact = 0
            confidence_sum = 0
            
            for item in news_items:
                # Get financial sentiment
                sentiment = self.nlp.get_financial_sentiment(item.get('title', ''))
                
                # Extract market entities
                entities = self.nlp.extract_market_entities(item.get('content', ''))
                
                # Get market events
                market_events = self.nlp.extract_market_events(item.get('content', ''))
                
                # Analyze text complexity
                complexity = self.nlp.analyze_text_complexity(item.get('content', ''))
                
                # Calculate event impact
                for event in market_events:
                    impact = self._calculate_event_impact(
                        event,
                        sentiment['scores'],
                        complexity.get('complexity_score', 0),
                        entities
                    )
                    
                    event_data = {
                        'timestamp': item.get('timestamp', datetime.now()),
                        'type': event['type'],
                        'trigger': event['trigger'],
                        'subject': event['subject'],
                        'sentiment': sentiment['sentiment'],
                        'impact': impact,
                        'confidence': sentiment['confidence'],
                        'entities': entities
                    }
                    
                    events.append(event_data)
                    total_impact += impact
                    confidence_sum += sentiment['confidence']
                    
                    # Store event in memory
                    self._store_event(event_data)
                    
            # Calculate aggregate impact
            num_events = len(events) or 1
            avg_impact = total_impact / num_events
            avg_confidence = confidence_sum / num_events
            
            return {
                'events': events,
                'impact': avg_impact,
                'confidence': avg_confidence,
                'event_count': num_events
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market events: {str(e)}")
            return {'events': [], 'impact': 0, 'confidence': 0}
            
    def _calculate_event_impact(self, event: Dict, sentiment_scores: Dict,
                              complexity: float, entities: Dict) -> float:
        """Calculate the market impact of an event"""
        try:
            # Base impact from sentiment
            base_impact = (sentiment_scores['positive'] - sentiment_scores['negative'])
            
            # Adjust based on event type
            type_multipliers = {
                'announcement': 1.2,
                'policy': 1.5,
                'economic': 1.3
            }
            type_mult = type_multipliers.get(event['type'], 1.0)
            
            # Adjust for presence of important entities
            entity_impact = 0
            if entities.get('ORG'):
                entity_impact += 0.2
            if entities.get('MONEY') or entities.get('PERCENT'):
                entity_impact += 0.3
                
            # Adjust for text complexity
            complexity_factor = min(complexity * 0.5, 0.5)  # Max 50% boost from complexity
            
            # Calculate final impact
            impact = base_impact * type_mult * (1 + entity_impact) * (1 + complexity_factor)
            
            return max(min(impact, 1.0), -1.0)  # Clamp between -1 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating event impact: {str(e)}")
            return 0.0
            
    def _store_event(self, event: Dict):
        """Store event in memory for future reference"""
        try:
            new_event = pd.DataFrame([{
                'timestamp': event['timestamp'],
                'event_type': event['type'],
                'impact': event['impact'],
                'description': f"{event['trigger']} - {event['subject']}"
            }])
            
            self.event_memory = pd.concat([self.event_memory, new_event])
            
            # Keep only last 7 days of events
            cutoff = datetime.now() - timedelta(days=7)
            self.event_memory = self.event_memory[self.event_memory['timestamp'] > cutoff]
            
        except Exception as e:
            self.logger.error(f"Error storing event: {str(e)}")
            
    def get_historical_impact(self, event_type: str, lookback_hours: int = 24) -> float:
        """Get historical impact of similar events"""
        try:
            cutoff = datetime.now() - timedelta(hours=lookback_hours)
            relevant_events = self.event_memory[
                (self.event_memory['event_type'] == event_type) &
                (self.event_memory['timestamp'] > cutoff)
            ]
            
            if len(relevant_events) == 0:
                return 0.0
                
            return relevant_events['impact'].mean()
            
        except Exception as e:
            self.logger.error(f"Error getting historical impact: {str(e)}")
            return 0.0
            
    def get_event_summary(self) -> Dict:
        """Get summary of recent market events"""
        try:
            recent_events = self.event_memory[
                self.event_memory['timestamp'] > datetime.now() - timedelta(hours=24)
            ]
            
            return {
                'total_events': len(recent_events),
                'event_types': recent_events['event_type'].value_counts().to_dict(),
                'avg_impact': recent_events['impact'].mean(),
                'max_impact_event': recent_events.loc[recent_events['impact'].abs().idxmax()].to_dict()
                if len(recent_events) > 0 else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting event summary: {str(e)}")
            return {}
