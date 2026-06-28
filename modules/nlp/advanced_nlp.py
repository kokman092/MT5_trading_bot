import logging
from typing import Dict, List, Optional, Tuple
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import numpy as np
from collections import defaultdict
import re

class AdvancedNLP:
    def __init__(self, config: Dict):
        """Initialize advanced NLP processor"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        try:
            # Load SpaCy model for dependency parsing
            self.nlp = spacy.load("en_core_web_sm")
            
            # Load FinBERT for financial sentiment
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            
            # Move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            
            # Named Entity Recognition for financial entities
            self.ner = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english")
            
        except Exception as e:
            self.logger.error(f"Error initializing NLP models: {str(e)}")
            
    def extract_market_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract market-related entities using NER"""
        try:
            entities = defaultdict(list)
            ner_results = self.ner(text)
            
            for ent in ner_results:
                if ent['score'] > 0.85:  # High confidence only
                    entity_text = ent['word'].strip()
                    if ent['entity'] in ['ORG', 'MONEY', 'PERCENT']:
                        entities[ent['entity']].append(entity_text)
                        
            return dict(entities)
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return {}
            
    def analyze_dependencies(self, text: str) -> List[Dict]:
        """Analyze syntactic dependencies for market relationships"""
        try:
            doc = self.nlp(text)
            market_patterns = []
            
            for token in doc:
                # Look for price movement patterns
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    if self._is_market_related(token.text):
                        pattern = {
                            'subject': token.text,
                            'action': token.head.text,
                            'type': 'movement'
                        }
                        
                        # Get magnitude if present
                        for child in token.head.children:
                            if child.dep_ in ["nummod", "quantmod"]:
                                pattern['magnitude'] = child.text
                                
                        market_patterns.append(pattern)
                        
            return market_patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing dependencies: {str(e)}")
            return []
            
    def get_financial_sentiment(self, text: str) -> Dict:
        """Get financial sentiment using FinBERT"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            predictions = predictions.cpu().numpy()
            
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            confidence = float(torch.max(predictions))
            
            return {
                'sentiment': sentiment_map[predictions[0].argmax()],
                'confidence': confidence,
                'scores': {
                    'negative': float(predictions[0][0]),
                    'neutral': float(predictions[0][1]),
                    'positive': float(predictions[0][2])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting financial sentiment: {str(e)}")
            return {'sentiment': 'neutral', 'confidence': 0}
            
    def extract_market_events(self, text: str) -> List[Dict]:
        """Extract market events and their attributes"""
        try:
            doc = self.nlp(text)
            events = []
            
            # Define market event triggers
            event_triggers = {
                'announcement': ['announce', 'release', 'report'],
                'policy': ['raise', 'cut', 'maintain', 'change'],
                'economic': ['grow', 'decline', 'expand', 'contract']
            }
            
            for sent in doc.sents:
                event = self._find_event(sent, event_triggers)
                if event:
                    events.append(event)
                    
            return events
            
        except Exception as e:
            self.logger.error(f"Error extracting market events: {str(e)}")
            return []
            
    def _find_event(self, sent, triggers: Dict[str, List[str]]) -> Optional[Dict]:
        """Find market events in a sentence"""
        for token in sent:
            for event_type, trigger_words in triggers.items():
                if token.lemma_ in trigger_words:
                    event = {
                        'type': event_type,
                        'trigger': token.text,
                        'subject': self._get_subject(token),
                        'time': self._get_time_expression(sent),
                        'magnitude': self._get_magnitude(token)
                    }
                    return event
        return None
        
    def _get_subject(self, token) -> str:
        """Get the subject of an event"""
        for child in token.children:
            if child.dep_ == "nsubj":
                return child.text
        return ""
        
    def _get_time_expression(self, sent) -> str:
        """Extract time expressions"""
        time_patterns = r'\b(today|yesterday|tomorrow|\d{1,2}:\d{2}|\d{1,2}/\d{1,2}|\d{4})\b'
        matches = re.findall(time_patterns, sent.text)
        return matches[0] if matches else ""
        
    def _get_magnitude(self, token) -> str:
        """Get magnitude expressions"""
        for child in token.children:
            if child.dep_ in ["nummod", "quantmod"] or child.pos_ == "NUM":
                return child.text
        return ""
        
    def _is_market_related(self, text: str) -> bool:
        """Check if text is market-related"""
        market_terms = {
            'price', 'market', 'stock', 'bond', 'currency', 'rate',
            'index', 'forex', 'dollar', 'euro', 'yen', 'pound'
        }
        return text.lower() in market_terms
        
    def analyze_text_complexity(self, text: str) -> Dict:
        """Analyze complexity of market text"""
        try:
            doc = self.nlp(text)
            
            # Calculate various complexity metrics
            avg_word_length = sum(len(token.text) for token in doc) / len(doc)
            sentence_lengths = [len(sent) for sent in doc.sents]
            avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
            
            # Count financial terms
            financial_term_count = sum(1 for token in doc if self._is_market_related(token.text))
            
            return {
                'avg_word_length': float(avg_word_length),
                'avg_sentence_length': float(avg_sentence_length),
                'financial_term_density': financial_term_count / len(doc),
                'complexity_score': (avg_word_length * 0.3 + 
                                   avg_sentence_length * 0.4 +
                                   (financial_term_count / len(doc)) * 0.3)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing text complexity: {str(e)}")
            return {}
