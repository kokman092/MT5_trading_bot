from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import MetaTrader5 as mt5
from ..deployment.error_handler import ErrorHandler

class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('risk_manager')
        self.position_risks = {}
        self.hedge_pairs = self._init_hedge_pairs()
        self.risk_metrics = {}
        
    def _init_hedge_pairs(self) -> Dict:
        """Initialize correlated pairs for hedging"""
        return {
            'EURUSD': ['GBPUSD', 'USDCHF'],
            'BTCUSD': ['ETHUSD', 'LTCUSD'],
            'XAUUSD': ['XAGUSD', 'USOIL'],
            # Add more pairs as needed
        }
        
    async def calculate_position_risk(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        account_balance: float
    ) -> Dict:
        """Calculate comprehensive risk metrics for a position"""
        try:
            # Get market data
            market_data = await self._get_market_data(symbol)
            
            # Calculate volatility-based metrics
            volatility = await self._calculate_volatility(market_data)
            
            # Calculate risk amount
            risk_amount = position_size * entry_price
            risk_percentage = (risk_amount / account_balance) * 100
            
            # Calculate dynamic stop-loss
            stop_loss = await self._calculate_dynamic_stop(
                symbol, entry_price, volatility
            )
            
            # Calculate trailing stop parameters
            trailing_params = await self._calculate_trailing_params(
                symbol, entry_price, volatility
            )
            
            # Get hedge recommendations
            hedge_options = await self._get_hedge_options(symbol, risk_amount)
            
            risk_metrics = {
                'risk_amount': risk_amount,
                'risk_percentage': risk_percentage,
                'stop_loss': stop_loss,
                'trailing_params': trailing_params,
                'volatility': volatility,
                'hedge_options': hedge_options,
                'max_loss': abs(entry_price - stop_loss) * position_size,
                'timestamp': datetime.now()
            }
            
            # Store risk metrics
            self.risk_metrics[symbol] = risk_metrics
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Risk calculation error: {str(e)}")
            return {}
            
    async def _get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get market data for risk calculations"""
        try:
            # Get recent price data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
            if rates is None:
                return pd.DataFrame()
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            self.logger.error(f"Market data fetch error: {str(e)}")
            return pd.DataFrame()
            
    async def _calculate_volatility(self, data: pd.DataFrame) -> Dict:
        """Calculate various volatility metrics"""
        try:
            if data.empty:
                return {}
                
            # Calculate ATR
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            # Calculate historical volatility
            returns = np.log(data['close'] / data['close'].shift(1))
            hist_vol = returns.std() * np.sqrt(252 * 288)  # Annualized (5-minute data)
            
            # Calculate volatility regime
            vol_sma = returns.rolling(window=20).std().mean()
            current_vol = returns.rolling(window=20).std().iloc[-1]
            regime = 'high' if current_vol > vol_sma * 1.2 else 'low' if current_vol < vol_sma * 0.8 else 'normal'
            
            return {
                'atr': atr,
                'hist_vol': hist_vol,
                'current_vol': current_vol,
                'regime': regime
            }
            
        except Exception as e:
            self.logger.error(f"Volatility calculation error: {str(e)}")
            return {}
            
    async def _calculate_dynamic_stop(
        self,
        symbol: str,
        entry_price: float,
        volatility: Dict
    ) -> float:
        """Calculate dynamic stop-loss level"""
        try:
            # Base stop-loss percentage
            base_stop = self.config.get('BASE_STOP_LOSS_PERCENT', 1.0) / 100
            
            # Adjust based on volatility regime
            vol_multiplier = {
                'low': 0.8,
                'normal': 1.0,
                'high': 1.2
            }.get(volatility.get('regime', 'normal'), 1.0)
            
            # Adjust based on ATR
            atr = volatility.get('atr', 0)
            atr_multiplier = 1.5  # ATR multiplier for stop-loss
            
            # Calculate final stop-loss distance
            stop_distance = max(
                base_stop * entry_price * vol_multiplier,
                atr * atr_multiplier
            )
            
            # Apply minimum stop-loss distance
            min_stop = self.config.get('MIN_STOP_LOSS_DISTANCE', 0.1)
            stop_distance = max(stop_distance, min_stop)
            
            return entry_price - stop_distance
            
        except Exception as e:
            self.logger.error(f"Dynamic stop calculation error: {str(e)}")
            return entry_price * 0.99  # Default 1% stop-loss
            
    async def _calculate_trailing_params(
        self,
        symbol: str,
        entry_price: float,
        volatility: Dict
    ) -> Dict:
        """Calculate trailing stop parameters"""
        try:
            # Base trailing distance
            atr = volatility.get('atr', 0)
            regime = volatility.get('regime', 'normal')
            
            # Adjust trailing distance based on volatility
            regime_multiplier = {
                'low': 1.5,
                'normal': 2.0,
                'high': 2.5
            }.get(regime, 2.0)
            
            trailing_distance = atr * regime_multiplier
            
            # Calculate activation threshold
            min_profit = self.config.get('MIN_PROFIT_ACTIVATION', 0.5) / 100
            activation_price = entry_price * (1 + min_profit)
            
            # Calculate step size
            step_size = trailing_distance / 4  # Smaller steps for smoother trailing
            
            return {
                'trailing_distance': trailing_distance,
                'activation_price': activation_price,
                'step_size': step_size
            }
            
        except Exception as e:
            self.logger.error(f"Trailing params calculation error: {str(e)}")
            return {}
            
    async def _get_hedge_options(self, symbol: str, risk_amount: float) -> List[Dict]:
        """Get hedging recommendations"""
        try:
            hedge_options = []
            
            # Get correlated symbols
            correlated_symbols = self.hedge_pairs.get(symbol, [])
            
            for hedge_symbol in correlated_symbols:
                # Get correlation
                correlation = await self._calculate_correlation(symbol, hedge_symbol)
                
                if abs(correlation) > 0.5:  # Significant correlation
                    # Calculate hedge ratio
                    hedge_ratio = await self._calculate_hedge_ratio(
                        symbol, hedge_symbol, correlation
                    )
                    
                    # Calculate hedge position size
                    hedge_size = risk_amount * hedge_ratio
                    
                    hedge_options.append({
                        'symbol': hedge_symbol,
                        'correlation': correlation,
                        'hedge_ratio': hedge_ratio,
                        'position_size': hedge_size,
                        'direction': 'sell' if correlation > 0 else 'buy'
                    })
                    
            return hedge_options
            
        except Exception as e:
            self.logger.error(f"Hedge options calculation error: {str(e)}")
            return []
            
    async def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        try:
            # Get price data
            data1 = await self._get_market_data(symbol1)
            data2 = await self._get_market_data(symbol2)
            
            if data1.empty or data2.empty:
                return 0
                
            # Calculate returns
            returns1 = data1['close'].pct_change()
            returns2 = data2['close'].pct_change()
            
            # Calculate correlation
            correlation = returns1.corr(returns2)
            return correlation
            
        except Exception as e:
            self.logger.error(f"Correlation calculation error: {str(e)}")
            return 0
            
    async def _calculate_hedge_ratio(
        self,
        symbol1: str,
        symbol2: str,
        correlation: float
    ) -> float:
        """Calculate optimal hedge ratio"""
        try:
            # Get volatility data
            data1 = await self._get_market_data(symbol1)
            data2 = await self._get_market_data(symbol2)
            
            if data1.empty or data2.empty:
                return 1
                
            # Calculate volatilities
            vol1 = data1['close'].pct_change().std()
            vol2 = data2['close'].pct_change().std()
            
            # Calculate hedge ratio using correlation and volatilities
            hedge_ratio = correlation * (vol1 / vol2)
            
            return abs(hedge_ratio)
            
        except Exception as e:
            self.logger.error(f"Hedge ratio calculation error: {str(e)}")
            return 1
            
    async def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        position_data: Dict
    ) -> Optional[float]:
        """Update trailing stop level"""
        try:
            if symbol not in self.risk_metrics:
                return None
                
            trailing_params = self.risk_metrics[symbol]['trailing_params']
            current_stop = position_data.get('sl', 0)
            
            # Check if trailing stop should be activated
            if current_price >= trailing_params['activation_price']:
                # Calculate new stop level
                new_stop = current_price - trailing_params['trailing_distance']
                
                # Move stop only if it's higher than current stop
                if new_stop > current_stop:
                    # Round to step size
                    step_size = trailing_params['step_size']
                    new_stop = round(new_stop / step_size) * step_size
                    return new_stop
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Trailing stop update error: {str(e)}")
            return None
            
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics summary"""
        try:
            metrics = {}
            for symbol, risk_data in self.risk_metrics.items():
                metrics[symbol] = {
                    'risk_percentage': risk_data['risk_percentage'],
                    'volatility_regime': risk_data.get('volatility', {}).get('regime', 'unknown'),
                    'stop_loss': risk_data['stop_loss'],
                    'max_loss': risk_data['max_loss'],
                    'hedge_count': len(risk_data.get('hedge_options', [])),
                    'timestamp': risk_data['timestamp'].isoformat()
                }
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrics calculation error: {str(e)}")
            return {}

    async def validate_strategy_robustness(self, strategy_metrics: Dict) -> Dict:
        """Validate strategy robustness to prevent overfitting"""
        try:
            robustness_metrics = {
                'is_robust': True,
                'warnings': []
            }
            
            # Check for overfit indicators
            if strategy_metrics.get('in_sample_sharpe', 0) > strategy_metrics.get('out_of_sample_sharpe', 0) * 1.5:
                robustness_metrics['warnings'].append('High in-sample vs out-of-sample performance difference')
                robustness_metrics['is_robust'] = False
                
            if strategy_metrics.get('max_consecutive_wins', 0) > 10:
                robustness_metrics['warnings'].append('Suspiciously long winning streak')
                
            if strategy_metrics.get('profit_factor', 0) > 3:
                robustness_metrics['warnings'].append('Unusually high profit factor')
                
            return robustness_metrics
            
        except Exception as e:
            self.logger.error(f"Strategy robustness validation error: {str(e)}")
            return {'is_robust': False, 'warnings': ['Validation error occurred']}
            
    async def validate_technical_health(self) -> Dict:
        """Check technical system health"""
        try:
            health_status = {
                'is_healthy': True,
                'issues': []
            }
            
            # Check MT5 connection
            if not mt5.initialize():
                health_status['issues'].append('MT5 connection failed')
                health_status['is_healthy'] = False
                
            # Check available margin
            account_info = mt5.account_info()
            if account_info:
                margin_level = account_info.margin_level
                if account_info.margin > 0 and margin_level < 150:  # Less than 150% margin level
                    health_status['issues'].append(f'Low margin level: {margin_level}%')
                    health_status['is_healthy'] = False
                    
            # Check system resources
            try:
                import psutil
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                if cpu_percent > 80:
                    health_status['issues'].append(f'High CPU usage: {cpu_percent}%')
                if memory_percent > 80:
                    health_status['issues'].append(f'High memory usage: {memory_percent}%')
                    
            except ImportError:
                health_status['issues'].append('Unable to check system resources')
                
            return health_status
            
        except Exception as e:
            self.logger.error(f"Technical health check error: {str(e)}")
            return {'is_healthy': False, 'issues': ['Health check error occurred']}
            
    async def create_system_backup(self) -> Dict:
        """Create backup of critical trading data"""
        try:
            backup_status = {
                'success': True,
                'backup_data': {}
            }
            
            # Backup trading configuration
            backup_status['backup_data']['config'] = self.config
            
            # Backup risk metrics
            backup_status['backup_data']['risk_metrics'] = self.risk_metrics
            
            # Backup position data
            positions = mt5.positions_get()
            if positions:
                backup_status['backup_data']['positions'] = [
                    {
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'volume': pos.volume,
                        'price_open': pos.price_open,
                        'sl': pos.sl,
                        'tp': pos.tp,
                        'profit': pos.profit
                    }
                    for pos in positions
                ]
                
            # Backup orders
            orders = mt5.orders_get()
            if orders:
                backup_status['backup_data']['orders'] = [
                    {
                        'ticket': order.ticket,
                        'symbol': order.symbol,
                        'type': order.type,
                        'volume': order.volume_current,
                        'price': order.price_open
                    }
                    for order in orders
                ]
                
            return backup_status
            
        except Exception as e:
            self.logger.error(f"Backup creation error: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    async def validate_portfolio_diversification(self) -> Dict:
        """Validate portfolio diversification"""
        try:
            diversification_metrics = {
                'is_diversified': True,
                'warnings': [],
                'metrics': {}
            }
            
            positions = mt5.positions_get()
            if not positions:
                return diversification_metrics
                
            # Analyze position distribution
            symbols = [pos.symbol for pos in positions]
            volumes = [pos.volume for pos in positions]
            total_volume = sum(volumes)
            
            # Check concentration
            for symbol, volume in zip(symbols, volumes):
                concentration = (volume / total_volume) * 100
                if concentration > 20:  # More than 20% in single instrument
                    diversification_metrics['warnings'].append(
                        f'High concentration in {symbol}: {concentration:.1f}%'
                    )
                    diversification_metrics['is_diversified'] = False
                    
            # Check correlation between positions
            correlations = []
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    corr = await self._calculate_correlation(symbol1, symbol2)
                    correlations.append(abs(corr))
                    
            if correlations:
                avg_correlation = sum(correlations) / len(correlations)
                if avg_correlation > 0.7:  # High average correlation
                    diversification_metrics['warnings'].append(
                        f'High average correlation between positions: {avg_correlation:.2f}'
                    )
                    diversification_metrics['is_diversified'] = False
                    
            # Calculate diversification metrics
            diversification_metrics['metrics'] = {
                'position_count': len(positions),
                'unique_symbols': len(set(symbols)),
                'avg_correlation': avg_correlation if correlations else 0,
                'max_concentration': max((v / total_volume) * 100 for v in volumes)
            }
            
            return diversification_metrics
            
        except Exception as e:
            self.logger.error(f"Portfolio diversification validation error: {str(e)}")
            return {
                'is_diversified': False,
                'warnings': ['Validation error occurred'],
                'metrics': {}
            }

    def calculate_position_size(self, symbol: str, stop_loss_points: float, confidence: float = 1.0) -> float:
        """Calculate position size based on risk parameters and current conditions"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                self.logger.error("Failed to get account info")
                return 0.0
                
            # Get risk amount based on account balance
            risk_percent = self.config['risk_per_trade']
            risk_amount = account_info.balance * risk_percent
            
            # Adjust risk for cryptocurrencies
            if symbol in ['BTCUSD', 'ETHUSD', 'XRPUSD', 'DOTUSD', 'ADAUSD', 'SOLUSD']:
                # Reduce risk for crypto due to higher volatility
                risk_amount *= 0.5
                # Apply volatility adjustment
                volatility_adj = self.config['trading']['crypto_settings']['volatility_adjustment']
                stop_loss_points *= volatility_adj
                
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Failed to get symbol info for {symbol}")
                return 0.0
                
            point = symbol_info.point
            position_size = risk_amount / (stop_loss_points * point)
            
            # Apply confidence adjustment
            position_size *= confidence
            
            # Apply crypto-specific volume limits
            if symbol in ['BTCUSD', 'ETHUSD', 'XRPUSD', 'DOTUSD', 'ADAUSD', 'SOLUSD']:
                max_volume = self.config['trading']['crypto_settings']['max_volume_btc']
                min_volume = self.config['trading']['crypto_settings']['min_volume_btc']
            else:
                max_volume = self.config['trading']['max_volume']
                min_volume = self.config['trading']['min_volume']
                
            position_size = min(position_size, max_volume)
            position_size = max(position_size, min_volume)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
            
    def check_margin_requirements(self, symbol: str, volume: float) -> bool:
        """Check if there's enough margin for the trade"""
        try:
            # Get account info
            account_info = mt5.account_info()
            if not account_info:
                self.logger.error("Failed to get account info")
                return False
                
            # Calculate required margin
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Failed to get symbol info for {symbol}")
                return False
                
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.logger.error(f"Failed to get tick data for {symbol}")
                return False
                
            price = (tick.ask + tick.bid) / 2
            margin_requirement = symbol_info.margin_initial * volume * price
            
            # Increase margin requirement for crypto
            if symbol in ['BTCUSD', 'ETHUSD', 'XRPUSD', 'DOTUSD', 'ADAUSD', 'SOLUSD']:
                margin_requirement *= self.config['trading']['crypto_settings']['increased_margin_requirement']
                
            # Check if we have enough free margin
            free_margin = account_info.margin_free
            required_margin_level = 150  # Minimum 150% margin level required
            
            if free_margin < margin_requirement:
                self.logger.warning(f"Insufficient free margin. Required: {margin_requirement}, Available: {free_margin}")
                return False
                
            # Calculate margin level after potential trade
            new_margin_level = ((account_info.equity - margin_requirement) / 
                              (account_info.margin + margin_requirement)) * 100
                              
            if new_margin_level < required_margin_level:
                self.logger.warning(f"Trade would result in unsafe margin level: {new_margin_level}%")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking margin requirements: {str(e)}")
            return False
