from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5
from dataclasses import dataclass
import yfinance as yf
from scipy.optimize import minimize
from statsmodels.tsa.stattools import coint
import ccxt
from ..analytics.market_analyzer import MarketAnalyzer
from ..deployment.error_handler import ErrorHandler
from ..risk.position_manager import PositionManager

@dataclass
class AssetAllocation:
    asset_type: str  # Type of asset
    weight: float  # Portfolio weight
    risk_contribution: float  # Risk contribution
    expected_return: float  # Expected return
    volatility: float  # Asset volatility

@dataclass
class ArbitrageOpportunity:
    asset_pair: Tuple[str, str]  # Pair of assets
    spread: float  # Price spread
    correlation: float  # Price correlation
    execution_cost: float  # Total execution cost
    expected_profit: float  # Expected profit

@dataclass
class VolatilitySignal:
    vix_level: float  # Current VIX level
    regime: str  # Volatility regime
    hedge_ratio: float  # Recommended hedge ratio
    option_strategy: str  # Recommended options strategy
    risk_adjustment: float  # Risk adjustment factor

class PortfolioDiversifier:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('portfolio_diversifier')
        self.market_analyzer = MarketAnalyzer(config)
        self.position_manager = PositionManager(config)
        
        # Initialize parameters and connections
        self._init_portfolio_parameters()
        self._init_exchange_connections()
        
    def _init_portfolio_parameters(self):
        """Initialize portfolio parameters"""
        # Asset allocation parameters
        self.allocation_params = {
            'min_weight': 0.05,  # Minimum asset weight
            'max_weight': 0.30,  # Maximum asset weight
            'risk_free_rate': 0.02,  # Risk-free rate
            'rebalance_threshold': 0.1,  # Rebalance threshold
            'asset_types': {
                'forex': {
                    'target_weight': 0.3,
                    'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
                },
                'crypto': {
                    'target_weight': 0.2,
                    'pairs': ['BTC/USD', 'ETH/USD', 'XRP/USD']
                },
                'stocks': {
                    'target_weight': 0.2,
                    'symbols': ['AAPL', 'MSFT', 'GOOGL']
                },
                'commodities': {
                    'target_weight': 0.15,
                    'symbols': ['XAUUSD', 'XAGUSD', 'WTIUSD']
                },
                'etfs': {
                    'target_weight': 0.15,
                    'symbols': ['SPY', 'QQQ', 'IWM']
                }
            }
        }
        
        # Arbitrage parameters
        self.arbitrage_params = {
            'min_spread': 0.001,  # Minimum spread (0.1%)
            'min_correlation': 0.8,  # Minimum correlation
            'max_execution_cost': 0.0005,  # Maximum cost (0.05%)
            'min_profit': 0.0015,  # Minimum profit (0.15%)
            'lookback_period': '30d',  # Correlation lookback
            'check_interval': 60  # Check every 60 seconds
        }
        
        # Volatility parameters
        self.volatility_params = {
            'vix_threshold_low': 15,  # Low volatility threshold
            'vix_threshold_high': 25,  # High volatility threshold
            'hedge_ratio_low': 0.1,  # Low volatility hedge
            'hedge_ratio_high': 0.3,  # High volatility hedge
            'option_strategies': {
                'low_vol': 'short_puts',
                'medium_vol': 'iron_condor',
                'high_vol': 'long_puts'
            }
        }
        
    def _init_exchange_connections(self):
        """Initialize exchange connections"""
        try:
            # Initialize MT5 connection
            if not mt5.initialize():
                raise Exception("MT5 initialization failed")
                
            # Initialize crypto exchange
            self.crypto_exchange = ccxt.binance({
                'apiKey': self.config.get('BINANCE_API_KEY'),
                'secret': self.config.get('BINANCE_SECRET')
            })
            
            # Initialize stock data connection
            self.stock_data = yf.Tickers(' '.join(
                self.allocation_params['asset_types']['stocks']['symbols']
            ))
            
        except Exception as e:
            self.logger.error(f"Exchange connection error: {str(e)}")
            
    async def get_optimal_allocation(
        self,
        risk_tolerance: float = 0.5
    ) -> List[AssetAllocation]:
        """Calculate optimal portfolio allocation"""
        try:
            allocations = []
            
            # Get asset data
            asset_data = await self._get_asset_data()
            
            # Calculate correlation matrix
            correlation = asset_data.corr()
            
            # Calculate returns and volatilities
            returns = await self._calculate_returns(asset_data)
            volatilities = await self._calculate_volatilities(asset_data)
            
            # Optimize portfolio
            weights = await self._optimize_portfolio(
                returns,
                volatilities,
                correlation,
                risk_tolerance
            )
            
            # Create allocation objects
            for asset, weight in weights.items():
                allocations.append(
                    AssetAllocation(
                        asset_type=self._get_asset_type(asset),
                        weight=weight,
                        risk_contribution=self._calculate_risk_contribution(
                            weight,
                            volatilities[asset],
                            correlation[asset]
                        ),
                        expected_return=returns[asset],
                        volatility=volatilities[asset]
                    )
                )
                
            return allocations
            
        except Exception as e:
            self.logger.error(f"Portfolio allocation error: {str(e)}")
            return []
            
    async def find_arbitrage_opportunities(
        self
    ) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities across markets"""
        try:
            opportunities = []
            
            # Get asset prices
            prices = await self._get_asset_prices()
            
            # Find correlations
            correlations = await self._calculate_correlations(prices)
            
            # Find opportunities
            for pair, corr in correlations.items():
                if corr >= self.arbitrage_params['min_correlation']:
                    spread = await self._calculate_spread(
                        prices[pair[0]],
                        prices[pair[1]]
                    )
                    
                    if spread >= self.arbitrage_params['min_spread']:
                        cost = await self._calculate_execution_cost(pair)
                        profit = spread - cost
                        
                        if profit >= self.arbitrage_params['min_profit']:
                            opportunities.append(
                                ArbitrageOpportunity(
                                    asset_pair=pair,
                                    spread=spread,
                                    correlation=corr,
                                    execution_cost=cost,
                                    expected_profit=profit
                                )
                            )
                            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Arbitrage opportunity search error: {str(e)}")
            return []
            
    async def get_volatility_signals(self) -> VolatilitySignal:
        """Get volatility trading signals"""
        try:
            # Get VIX data
            vix_data = await self._get_vix_data()
            
            # Determine volatility regime
            regime = await self._determine_volatility_regime(vix_data)
            
            # Calculate hedge ratio
            hedge_ratio = await self._calculate_hedge_ratio(regime)
            
            # Get option strategy
            option_strategy = self.volatility_params['option_strategies'][regime]
            
            # Calculate risk adjustment
            risk_adjustment = await self._calculate_risk_adjustment(
                vix_data,
                regime
            )
            
            return VolatilitySignal(
                vix_level=vix_data['current'],
                regime=regime,
                hedge_ratio=hedge_ratio,
                option_strategy=option_strategy,
                risk_adjustment=risk_adjustment
            )
            
        except Exception as e:
            self.logger.error(f"Volatility signal error: {str(e)}")
            return None
            
    async def _optimize_portfolio(
        self,
        returns: pd.Series,
        volatilities: pd.Series,
        correlation: pd.DataFrame,
        risk_tolerance: float
    ) -> Dict[str, float]:
        """Optimize portfolio weights"""
        try:
            def objective(weights):
                portfolio_return = np.sum(returns * weights)
                portfolio_vol = np.sqrt(
                    np.dot(weights.T, np.dot(correlation * np.outer(
                        volatilities,
                        volatilities
                    ), weights))
                )
                return -(portfolio_return - (1 - risk_tolerance) * portfolio_vol)
                
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: x - self.allocation_params['min_weight']},
                {'type': 'ineq', 'fun': lambda x: self.allocation_params['max_weight'] - x}
            ]
            
            n_assets = len(returns)
            result = minimize(
                objective,
                np.array([1/n_assets] * n_assets),
                method='SLSQP',
                constraints=constraints
            )
            
            return dict(zip(returns.index, result.x))
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization error: {str(e)}")
            return {}
            
    async def _calculate_correlations(
        self,
        prices: pd.DataFrame
    ) -> Dict[Tuple[str, str], float]:
        """Calculate correlations between assets"""
        try:
            correlations = {}
            assets = list(prices.columns)
            
            for i in range(len(assets)):
                for j in range(i + 1, len(assets)):
                    asset1, asset2 = assets[i], assets[j]
                    correlation = prices[asset1].corr(prices[asset2])
                    correlations[(asset1, asset2)] = correlation
                    
            return correlations
            
        except Exception as e:
            self.logger.error(f"Correlation calculation error: {str(e)}")
            return {}
            
    async def _determine_volatility_regime(
        self,
        vix_data: Dict
    ) -> str:
        """Determine current volatility regime"""
        try:
            vix_level = vix_data['current']
            
            if vix_level <= self.volatility_params['vix_threshold_low']:
                return 'low_vol'
            elif vix_level >= self.volatility_params['vix_threshold_high']:
                return 'high_vol'
            else:
                return 'medium_vol'
                
        except Exception as e:
            self.logger.error(f"Volatility regime determination error: {str(e)}")
            return 'medium_vol'
            
    async def _calculate_hedge_ratio(
        self,
        regime: str
    ) -> float:
        """Calculate appropriate hedge ratio"""
        try:
            if regime == 'low_vol':
                return self.volatility_params['hedge_ratio_low']
            elif regime == 'high_vol':
                return self.volatility_params['hedge_ratio_high']
            else:
                return (
                    self.volatility_params['hedge_ratio_low'] +
                    self.volatility_params['hedge_ratio_high']
                ) / 2
                
        except Exception as e:
            self.logger.error(f"Hedge ratio calculation error: {str(e)}")
            return 0.2  # Default hedge ratio
            
    async def _get_asset_data(self) -> pd.DataFrame:
        """Get historical data for all assets"""
        try:
            data = pd.DataFrame()
            
            # Get forex data
            for pair in self.allocation_params['asset_types']['forex']['pairs']:
                rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_D1, 0, 100)
                if rates is not None:
                    df = pd.DataFrame(rates)
                    data[pair] = df['close']
                    
            # Get crypto data
            for pair in self.allocation_params['asset_types']['crypto']['pairs']:
                ohlcv = self.crypto_exchange.fetch_ohlcv(
                    pair,
                    '1d',
                    limit=100
                )
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                    data[pair] = df['close']
                    
            # Get stock data
            stock_data = self.stock_data.history(period='100d')
            if not stock_data.empty:
                data = pd.concat([data, stock_data['Close']], axis=1)
                
            return data
            
        except Exception as e:
            self.logger.error(f"Asset data retrieval error: {str(e)}")
            return pd.DataFrame()

class PortfolioAnalyzer:
    def __init__(self, config):
        self.config = config
        self.correlation_threshold = config.get('CORRELATION_THRESHOLD', 0.7)
        self.volatility_window = config.get('VOLATILITY_WINDOW', 20)
        self.min_trades = config.get('MIN_TRADES_FOR_ANALYSIS', 30)
        self.risk_free_rate = config.get('RISK_FREE_RATE', 0.02)
        self.max_correlation = config.get('MAX_CORRELATION', 0.8)
        self.min_sharpe = config.get('MIN_SHARPE_RATIO', 0.5)
        
        self.logger = logging.getLogger(__name__)
        self.correlation_matrix = None
        self.last_update = None
        self.update_interval = timedelta(hours=1)
        
    def calculate_correlation_matrix(self, symbols):
        """Calculate correlation matrix for given symbols"""
        try:
            # Get historical data for all symbols
            data = {}
            for symbol in symbols:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1000)
                if rates is not None:
                    df = pd.DataFrame(rates)
                    data[symbol] = df['close'].pct_change()
            
            # Create correlation matrix
            returns_df = pd.DataFrame(data)
            correlation_matrix = returns_df.corr()
            
            self.correlation_matrix = correlation_matrix
            self.last_update = datetime.now()
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {str(e)}")
            return None
            
    def get_correlation(self, symbol1, symbol2):
        """Get correlation between two symbols"""
        try:
            # Update correlation matrix if needed
            if (self.correlation_matrix is None or 
                datetime.now() - self.last_update > self.update_interval):
                self.calculate_correlation_matrix([symbol1, symbol2])
            
            if self.correlation_matrix is not None:
                return self.correlation_matrix.loc[symbol1, symbol2]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting correlation: {str(e)}")
            return None
            
    def calculate_portfolio_metrics(self, positions):
        """Calculate portfolio risk metrics"""
        try:
            if len(positions) < 2:
                return None
                
            # Calculate position weights
            total_value = sum(pos['value'] for pos in positions)
            weights = np.array([pos['value'] / total_value for pos in positions])
            
            # Get returns for each position
            returns = []
            for pos in positions:
                rates = mt5.copy_rates_from_pos(
                    pos['symbol'],
                    mt5.TIMEFRAME_H1,
                    0,
                    self.volatility_window
                )
                if rates is not None:
                    df = pd.DataFrame(rates)
                    returns.append(df['close'].pct_change().dropna())
            
            if not returns:
                return None
                
            # Calculate portfolio metrics
            returns_matrix = pd.concat(returns, axis=1)
            correlation_matrix = returns_matrix.corr()
            covariance_matrix = returns_matrix.cov()
            
            # Portfolio volatility
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Portfolio return
            portfolio_return = np.sum(weights * returns_matrix.mean())
            
            # Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            # Diversification ratio
            diversification_ratio = portfolio_volatility / np.sum(weights * returns_matrix.std())
            
            return {
                'volatility': float(portfolio_volatility),
                'return': float(portfolio_return),
                'sharpe_ratio': float(sharpe_ratio),
                'diversification_ratio': float(diversification_ratio),
                'max_correlation': float(correlation_matrix.max().max())
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return None
            
    def check_position_limits(self, new_symbol, current_positions):
        """Check if new position meets portfolio constraints"""
        try:
            # Calculate correlation with existing positions
            for position in current_positions:
                correlation = self.get_correlation(new_symbol, position['symbol'])
                if correlation is not None and abs(correlation) > self.max_correlation:
                    self.logger.warning(
                        f"High correlation ({correlation:.2f}) between {new_symbol} "
                        f"and {position['symbol']}"
                    )
                    return False
            
            # Calculate portfolio metrics with new position
            test_positions = current_positions + [{
                'symbol': new_symbol,
                'value': 1.0  # Dummy value for testing
            }]
            
            metrics = self.calculate_portfolio_metrics(test_positions)
            if metrics:
                if metrics['sharpe_ratio'] < self.min_sharpe:
                    self.logger.warning(
                        f"Adding {new_symbol} would reduce Sharpe ratio to {metrics['sharpe_ratio']:.2f}"
                    )
                    return False
                    
                if metrics['max_correlation'] > self.max_correlation:
                    self.logger.warning(
                        f"Adding {new_symbol} would increase max correlation to {metrics['max_correlation']:.2f}"
                    )
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking position limits: {str(e)}")
            return False
            
    def get_portfolio_report(self, positions):
        """Generate detailed portfolio analysis report"""
        try:
            metrics = self.calculate_portfolio_metrics(positions)
            if not metrics:
                return None
                
            # Calculate additional risk metrics
            symbols = [pos['symbol'] for pos in positions]
            correlation_matrix = self.calculate_correlation_matrix(symbols)
            
            report = {
                'metrics': metrics,
                'correlations': correlation_matrix.to_dict() if correlation_matrix is not None else None,
                'position_count': len(positions),
                'total_value': sum(pos['value'] for pos in positions),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add risk warnings
            warnings = []
            if metrics['sharpe_ratio'] < self.min_sharpe:
                warnings.append(f"Low Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
            if metrics['max_correlation'] > self.max_correlation:
                warnings.append(f"High correlation: {metrics['max_correlation']:.2f}")
            if metrics['diversification_ratio'] < 1.2:
                warnings.append(f"Poor diversification: {metrics['diversification_ratio']:.2f}")
                
            report['warnings'] = warnings
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio report: {str(e)}")
            return None
