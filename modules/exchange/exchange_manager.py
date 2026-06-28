from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import asyncio
import ccxt.async_support as ccxt
import MetaTrader5 as mt5
from ..deployment.error_handler import ErrorHandler

class ExchangeManager:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('exchange_manager')
        self.exchanges = {}
        self.market_data = {}
        self.order_books = {}
        self.exchange_status = {}
        
        # Initialize exchanges
        self._initialize_exchanges()
        
    def _initialize_exchanges(self):
        """Initialize connection to multiple exchanges"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
            else:
                self.exchanges['mt5'] = {'type': 'mt5', 'status': 'active'}
                
            # Initialize CCXT exchanges
            exchange_configs = self.config.get('EXCHANGES', {})
            for exchange_id, credentials in exchange_configs.items():
                if exchange_id == 'mt5':
                    continue
                    
                try:
                    # Get exchange class
                    exchange_class = getattr(ccxt, exchange_id)
                    
                    # Initialize exchange
                    exchange = exchange_class({
                        'apiKey': credentials.get('API_KEY'),
                        'secret': credentials.get('API_SECRET'),
                        'password': credentials.get('API_PASSWORD'),
                        'enableRateLimit': True,
                    })
                    
                    self.exchanges[exchange_id] = {
                        'instance': exchange,
                        'type': 'crypto',
                        'status': 'active'
                    }
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize {exchange_id}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Exchange initialization error: {str(e)}")
            
    async def fetch_market_data(self, symbol: str) -> Dict:
        """Fetch market data from all available exchanges"""
        try:
            market_data = {}
            
            # Fetch from MT5
            if 'mt5' in self.exchanges:
                mt5_data = await self._fetch_mt5_data(symbol)
                if mt5_data:
                    market_data['mt5'] = mt5_data
                    
            # Fetch from crypto exchanges
            tasks = []
            for exchange_id, exchange_info in self.exchanges.items():
                if exchange_info['type'] == 'crypto':
                    tasks.append(self._fetch_crypto_data(
                        exchange_info['instance'],
                        symbol,
                        exchange_id
                    ))
                    
            if tasks:
                crypto_data = await asyncio.gather(*tasks, return_exceptions=True)
                for data in crypto_data:
                    if isinstance(data, dict):
                        market_data.update(data)
                        
            # Store market data
            self.market_data[symbol] = {
                'data': market_data,
                'timestamp': datetime.now()
            }
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Market data fetch error: {str(e)}")
            return {}
            
    async def _fetch_mt5_data(self, symbol: str) -> Dict:
        """Fetch market data from MT5"""
        try:
            # Get last tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {}
                
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': datetime.fromtimestamp(tick.time)
            }
            
        except Exception as e:
            self.logger.error(f"MT5 data fetch error: {str(e)}")
            return {}
            
    async def _fetch_crypto_data(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        exchange_id: str
    ) -> Dict:
        """Fetch market data from crypto exchange"""
        try:
            # Fetch ticker
            ticker = await exchange.fetch_ticker(symbol)
            
            return {
                exchange_id: {
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'last': ticker['last'],
                    'volume': ticker['baseVolume'],
                    'time': datetime.fromtimestamp(ticker['timestamp'] / 1000)
                }
            }
            
        except Exception as e:
            self.logger.error(f"{exchange_id} data fetch error: {str(e)}")
            return {}
            
    async def fetch_order_books(self, symbol: str) -> Dict:
        """Fetch order books from all exchanges"""
        try:
            order_books = {}
            
            # Fetch from MT5
            if 'mt5' in self.exchanges:
                mt5_book = await self._fetch_mt5_order_book(symbol)
                if mt5_book:
                    order_books['mt5'] = mt5_book
                    
            # Fetch from crypto exchanges
            tasks = []
            for exchange_id, exchange_info in self.exchanges.items():
                if exchange_info['type'] == 'crypto':
                    tasks.append(self._fetch_crypto_order_book(
                        exchange_info['instance'],
                        symbol,
                        exchange_id
                    ))
                    
            if tasks:
                crypto_books = await asyncio.gather(*tasks, return_exceptions=True)
                for book in crypto_books:
                    if isinstance(book, dict):
                        order_books.update(book)
                        
            # Store order books
            self.order_books[symbol] = {
                'data': order_books,
                'timestamp': datetime.now()
            }
            
            return order_books
            
        except Exception as e:
            self.logger.error(f"Order book fetch error: {str(e)}")
            return {}
            
    async def _fetch_mt5_order_book(self, symbol: str) -> Dict:
        """Fetch order book from MT5"""
        try:
            book = mt5.market_book_get(symbol)
            if book is None:
                return {}
                
            bids = []
            asks = []
            
            for item in book:
                if item.type == mt5.BOOK_TYPE_SELL:
                    asks.append([item.price, item.volume])
                else:
                    bids.append([item.price, item.volume])
                    
            return {
                'bids': bids,
                'asks': asks,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"MT5 order book fetch error: {str(e)}")
            return {}
            
    async def _fetch_crypto_order_book(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        exchange_id: str
    ) -> Dict:
        """Fetch order book from crypto exchange"""
        try:
            order_book = await exchange.fetch_order_book(symbol)
            
            return {
                exchange_id: {
                    'bids': order_book['bids'],
                    'asks': order_book['asks'],
                    'timestamp': datetime.fromtimestamp(order_book['timestamp'] / 1000)
                }
            }
            
        except Exception as e:
            self.logger.error(f"{exchange_id} order book fetch error: {str(e)}")
            return {}
            
    async def find_arbitrage_opportunities(self, symbol: str) -> List[Dict]:
        """Find arbitrage opportunities across exchanges"""
        try:
            opportunities = []
            
            # Fetch latest order books
            order_books = await self.fetch_order_books(symbol)
            
            # Compare prices across exchanges
            exchanges = list(order_books.keys())
            for i in range(len(exchanges)):
                for j in range(i + 1, len(exchanges)):
                    exchange1 = exchanges[i]
                    exchange2 = exchanges[j]
                    
                    book1 = order_books[exchange1]
                    book2 = order_books[exchange2]
                    
                    # Check for opportunities
                    if book1['asks'] and book2['bids']:
                        best_ask1 = book1['asks'][0][0]
                        best_bid2 = book2['bids'][0][0]
                        
                        spread = best_bid2 - best_ask1
                        if spread > 0:
                            opportunities.append({
                                'buy_exchange': exchange1,
                                'sell_exchange': exchange2,
                                'buy_price': best_ask1,
                                'sell_price': best_bid2,
                                'spread': spread,
                                'spread_percent': (spread / best_ask1) * 100
                            })
                            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Arbitrage calculation error: {str(e)}")
            return []
            
    async def execute_arbitrage(
        self,
        opportunity: Dict,
        amount: float
    ) -> Dict:
        """Execute arbitrage trades"""
        try:
            results = {}
            
            # Execute buy order
            buy_exchange = opportunity['buy_exchange']
            buy_price = opportunity['buy_price']
            
            if buy_exchange == 'mt5':
                buy_result = await self._execute_mt5_trade(
                    symbol=opportunity['symbol'],
                    order_type='buy',
                    price=buy_price,
                    volume=amount
                )
            else:
                buy_result = await self._execute_crypto_trade(
                    exchange=self.exchanges[buy_exchange]['instance'],
                    symbol=opportunity['symbol'],
                    order_type='buy',
                    price=buy_price,
                    amount=amount
                )
                
            results['buy'] = buy_result
            
            # Execute sell order
            sell_exchange = opportunity['sell_exchange']
            sell_price = opportunity['sell_price']
            
            if sell_exchange == 'mt5':
                sell_result = await self._execute_mt5_trade(
                    symbol=opportunity['symbol'],
                    order_type='sell',
                    price=sell_price,
                    volume=amount
                )
            else:
                sell_result = await self._execute_crypto_trade(
                    exchange=self.exchanges[sell_exchange]['instance'],
                    symbol=opportunity['symbol'],
                    order_type='sell',
                    price=sell_price,
                    amount=amount
                )
                
            results['sell'] = sell_result
            
            return results
            
        except Exception as e:
            self.logger.error(f"Arbitrage execution error: {str(e)}")
            return {}
            
    async def _execute_mt5_trade(
        self,
        symbol: str,
        order_type: str,
        price: float,
        volume: float
    ) -> Dict:
        """Execute trade on MT5"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": 10,
                "magic": 234000,
                "comment": "arbitrage",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise Exception(f"MT5 order failed: {result.comment}")
                
            return {
                'order_id': result.order,
                'executed_price': result.price,
                'executed_volume': result.volume,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"MT5 trade execution error: {str(e)}")
            return {}
            
    async def _execute_crypto_trade(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        order_type: str,
        price: float,
        amount: float
    ) -> Dict:
        """Execute trade on crypto exchange"""
        try:
            if order_type == 'buy':
                order = await exchange.create_limit_buy_order(symbol, amount, price)
            else:
                order = await exchange.create_limit_sell_order(symbol, amount, price)
                
            return {
                'order_id': order['id'],
                'executed_price': order['price'],
                'executed_amount': order['amount'],
                'timestamp': datetime.fromtimestamp(order['timestamp'] / 1000)
            }
            
        except Exception as e:
            self.logger.error(f"Crypto trade execution error: {str(e)}")
            return {}
            
    async def check_exchange_status(self) -> Dict:
        """Check status of all connected exchanges"""
        try:
            status = {}
            
            # Check MT5
            if 'mt5' in self.exchanges:
                mt5_status = mt5.terminal_info()
                status['mt5'] = {
                    'connected': mt5_status is not None,
                    'trade_allowed': mt5_status.trade_allowed if mt5_status else False,
                    'balance': mt5.account_info().balance if mt5_status else 0
                }
                
            # Check crypto exchanges
            for exchange_id, exchange_info in self.exchanges.items():
                if exchange_info['type'] == 'crypto':
                    try:
                        await exchange_info['instance'].load_markets()
                        status[exchange_id] = {
                            'connected': True,
                            'trade_allowed': True,
                            'markets': len(exchange_info['instance'].markets)
                        }
                    except Exception as e:
                        status[exchange_id] = {
                            'connected': False,
                            'error': str(e)
                        }
                        
            self.exchange_status = status
            return status
            
        except Exception as e:
            self.logger.error(f"Status check error: {str(e)}")
            return {}
            
    def get_exchange_metrics(self) -> Dict:
        """Get exchange performance metrics"""
        try:
            metrics = {}
            for exchange_id, exchange_info in self.exchanges.items():
                metrics[exchange_id] = {
                    'status': exchange_info.get('status', 'unknown'),
                    'last_update': datetime.now().isoformat(),
                    'market_count': len(self.market_data.get(exchange_id, {}))
                }
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrics calculation error: {str(e)}")
            return {}
