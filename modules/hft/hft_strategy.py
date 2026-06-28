import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from .market_making import HFTMarketMaker
from .stat_arb import StatisticalArbitrage
from ..ml.feature_engineering import FinancialFeatureEngineering
from ..nlp.market_sentiment import MarketSentiment

class HFTStrategy:
    """High-frequency trading strategy combining market making, statistical arbitrage,
    and sentiment analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.market_maker = HFTMarketMaker(config)
        self.stat_arb = StatisticalArbitrage(config)
        self.feature_engineering = FinancialFeatureEngineering(config)
        self.sentiment_analyzer = MarketSentiment(config)
        
        # Strategy parameters
        self.params = {
            'min_spread': config.get('min_spread', 0.0001),
            'max_position': config.get('max_position', 1.0),
            'risk_limit': config.get('risk_limit', 0.02),
            'sentiment_threshold': config.get('sentiment_threshold', 0.5),
            'mm_weight': config.get('mm_weight', 0.4),
            'sa_weight': config.get('sa_weight', 0.4),
            'sentiment_weight': config.get('sentiment_weight', 0.2)
        }
        
    async def analyze_market(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze market data and generate trading signals"""
        try:
            results = {}
            
            # Get primary symbol data
            primary_symbol = list(market_data.keys())[0]
            primary_data = market_data[primary_symbol]
            
            # Market making analysis
            mm_analysis = await self.market_maker.analyze_market(primary_data)
            
            # Statistical arbitrage analysis
            sa_analysis = await self.stat_arb.analyze_pairs(market_data)
            
            # Sentiment analysis
            sentiment = await self.sentiment_analyzer.analyze_sentiment(primary_symbol)
            
            # Combine analyses
            combined_signals = self._combine_signals(
                mm_analysis,
                sa_analysis,
                sentiment
            )
            
            # Generate trading decisions
            decisions = self._generate_trading_decisions(
                combined_signals,
                primary_data
            )
            
            return {
                'signals': combined_signals,
                'decisions': decisions,
                'market_making': mm_analysis,
                'stat_arb': sa_analysis,
                'sentiment': sentiment
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market for HFT: {str(e)}")
            return {}
            
    def _combine_signals(self, mm_analysis: Dict,
                        sa_analysis: Dict,
                        sentiment: Dict) -> Dict:
        """Combine signals from different strategies"""
        try:
            signals = {}
            
            # Extract market making signals
            mm_signals = mm_analysis.get('signals', {})
            quotes = mm_analysis.get('quotes', {})
            
            # Extract stat arb signals
            sa_signals = sa_analysis.get('pair_results', {})
            cointegration = sa_analysis.get('cointegration', {})
            
            # Extract sentiment signals
            sentiment_score = sentiment.get('sentiment', 0)
            sentiment_bias = sentiment.get('bias', {})
            
            # Combine position signals
            signals['position'] = self._combine_position_signals(
                mm_signals.get('inventory', 0),
                sa_signals,
                sentiment_bias
            )
            
            # Combine price signals
            signals['price'] = self._combine_price_signals(
                quotes,
                sa_signals,
                sentiment_score
            )
            
            # Combine risk signals
            signals['risk'] = self._combine_risk_signals(
                mm_signals.get('toxicity', 0),
                sa_signals,
                sentiment.get('confidence', 0)
            )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error combining signals: {str(e)}")
            return {}
            
    def _generate_trading_decisions(self, signals: Dict,
                                  market_data: pd.DataFrame) -> Dict:
        """Generate trading decisions based on combined signals"""
        try:
            current_price = market_data['close'].iloc[-1]
            
            # Extract signals
            position_signal = signals['position']
            price_signals = signals['price']
            risk_signals = signals['risk']
            
            # Calculate optimal position
            target_position = self._calculate_target_position(
                position_signal,
                risk_signals
            )
            
            # Calculate optimal quotes
            optimal_quotes = self._calculate_optimal_quotes(
                current_price,
                price_signals,
                risk_signals
            )
            
            # Generate orders
            orders = self._generate_orders(
                target_position,
                optimal_quotes,
                current_price
            )
            
            return {
                'target_position': float(target_position),
                'optimal_quotes': optimal_quotes,
                'orders': orders,
                'risk_score': float(risk_signals.get('total', 0))
            }
            
        except Exception as e:
            self.logger.error(f"Error generating trading decisions: {str(e)}")
            return {}
            
    def _combine_position_signals(self, mm_inventory: float,
                                sa_signals: Dict,
                                sentiment_bias: Dict) -> Dict:
        """Combine position signals from different strategies"""
        try:
            # Market making component
            mm_signal = -mm_inventory  # Mean reversion to target
            
            # Stat arb component
            sa_opportunities = []
            for symbol, data in sa_signals.items():
                opps = data.get('opportunities', {})
                if opps.get('long_entry'):
                    sa_opportunities.append(opps.get('position_size', 0))
                elif opps.get('short_entry'):
                    sa_opportunities.append(-opps.get('position_size', 0))
                    
            sa_signal = np.mean(sa_opportunities) if sa_opportunities else 0
            
            # Sentiment component
            sentiment_direction = sentiment_bias.get('direction', 0)
            sentiment_strength = sentiment_bias.get('strength', 0)
            sentiment_signal = sentiment_direction * sentiment_strength
            
            # Combine signals with weights
            combined_signal = (
                self.params['mm_weight'] * mm_signal +
                self.params['sa_weight'] * sa_signal +
                self.params['sentiment_weight'] * sentiment_signal
            )
            
            return {
                'total': float(combined_signal),
                'market_making': float(mm_signal),
                'stat_arb': float(sa_signal),
                'sentiment': float(sentiment_signal)
            }
            
        except Exception as e:
            self.logger.error(f"Error combining position signals: {str(e)}")
            return {'total': 0.0}
            
    def _combine_price_signals(self, mm_quotes: Dict,
                             sa_signals: Dict,
                             sentiment_score: float) -> Dict:
        """Combine price signals from different strategies"""
        try:
            # Extract market making quotes
            mm_bid = mm_quotes.get('bid', {}).get('price', 0)
            mm_ask = mm_quotes.get('ask', {}).get('price', 0)
            
            # Extract stat arb price levels
            sa_prices = []
            for symbol, data in sa_signals.items():
                opps = data.get('opportunities', {})
                if opps:
                    sa_prices.append(opps.get('position_size', 0))
                    
            sa_price = np.mean(sa_prices) if sa_prices else 0
            
            # Adjust quotes based on sentiment
            sentiment_adjustment = sentiment_score * self.params['sentiment_threshold']
            
            return {
                'bid': float(mm_bid * (1 - sentiment_adjustment)),
                'ask': float(mm_ask * (1 + sentiment_adjustment)),
                'stat_arb_price': float(sa_price),
                'sentiment_adjustment': float(sentiment_adjustment)
            }
            
        except Exception as e:
            self.logger.error(f"Error combining price signals: {str(e)}")
            return {}
            
    def _combine_risk_signals(self, mm_toxicity: float,
                            sa_signals: Dict,
                            sentiment_confidence: float) -> Dict:
        """Combine risk signals from different strategies"""
        try:
            # Market making risk
            mm_risk = mm_toxicity
            
            # Stat arb risk
            sa_risks = []
            for symbol, data in sa_signals.items():
                opps = data.get('opportunities', {})
                if opps:
                    sa_risks.append(1 - opps.get('confidence', 0))
                    
            sa_risk = np.mean(sa_risks) if sa_risks else 0
            
            # Sentiment risk
            sentiment_risk = 1 - sentiment_confidence
            
            # Combine risks with weights
            total_risk = (
                self.params['mm_weight'] * mm_risk +
                self.params['sa_weight'] * sa_risk +
                self.params['sentiment_weight'] * sentiment_risk
            )
            
            return {
                'total': float(total_risk),
                'market_making': float(mm_risk),
                'stat_arb': float(sa_risk),
                'sentiment': float(sentiment_risk)
            }
            
        except Exception as e:
            self.logger.error(f"Error combining risk signals: {str(e)}")
            return {'total': 1.0}
            
    def _calculate_target_position(self, position_signal: Dict,
                                 risk_signals: Dict) -> float:
        """Calculate target position size"""
        try:
            # Get base position from signal
            base_position = position_signal.get('total', 0)
            
            # Risk adjustment
            risk_score = risk_signals.get('total', 1)
            risk_adjustment = 1 - risk_score
            
            # Calculate final position
            target_position = base_position * risk_adjustment
            
            # Apply position limits
            return max(
                -self.params['max_position'],
                min(target_position, self.params['max_position'])
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating target position: {str(e)}")
            return 0.0
            
    def _calculate_optimal_quotes(self, current_price: float,
                                price_signals: Dict,
                                risk_signals: Dict) -> Dict:
        """Calculate optimal quotes for market making"""
        try:
            # Get base quotes
            base_bid = price_signals.get('bid', current_price)
            base_ask = price_signals.get('ask', current_price)
            
            # Risk adjustment
            risk_score = risk_signals.get('total', 0)
            spread_adjustment = 1 + risk_score
            
            # Calculate spread
            base_spread = base_ask - base_bid
            adjusted_spread = max(
                base_spread * spread_adjustment,
                self.params['min_spread']
            )
            
            # Recalculate quotes
            mid_price = (base_bid + base_ask) / 2
            bid_price = mid_price - adjusted_spread / 2
            ask_price = mid_price + adjusted_spread / 2
            
            return {
                'bid': float(bid_price),
                'ask': float(ask_price),
                'spread': float(adjusted_spread),
                'risk_adjustment': float(spread_adjustment)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal quotes: {str(e)}")
            return {}
            
    def _generate_orders(self, target_position: float,
                        optimal_quotes: Dict,
                        current_price: float) -> List[Dict]:
        """Generate orders based on trading decisions"""
        try:
            orders = []
            
            # Market making orders
            if optimal_quotes:
                bid_price = optimal_quotes.get('bid')
                ask_price = optimal_quotes.get('ask')
                
                if bid_price and ask_price:
                    # Bid order
                    orders.append({
                        'type': 'limit',
                        'side': 'buy',
                        'price': float(bid_price),
                        'size': float(self.params['max_position'] / 10)
                    })
                    
                    # Ask order
                    orders.append({
                        'type': 'limit',
                        'side': 'sell',
                        'price': float(ask_price),
                        'size': float(self.params['max_position'] / 10)
                    })
                    
            # Position targeting order
            if abs(target_position) > 0:
                orders.append({
                    'type': 'market',
                    'side': 'buy' if target_position > 0 else 'sell',
                    'size': float(abs(target_position))
                })
                
            return orders
            
        except Exception as e:
            self.logger.error(f"Error generating orders: {str(e)}")
            return []
