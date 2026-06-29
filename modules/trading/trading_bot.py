import logging
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import asyncio
from .signal import Signal
from .market_analyzer import MarketAnalyzer
from .trade_executor import TradeExecutor
from .position_manager import PositionManager
from .risk_manager import RiskManager
from .ml_analyzer import MLAnalyzer
from .broker import MT5Broker

class TradingBot:
    def __init__(self, config: Dict):
        """Initialize the trading bot with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.market_analyzer = MarketAnalyzer(config)
        self.broker = MT5Broker(config)
        self.risk_manager = RiskManager(config)
        self.risk_manager.broker = self.broker
        self.trade_executor = TradeExecutor(config, self.broker, self.risk_manager)
        self.position_manager = PositionManager(config)
        self.ml_analyzer = MLAnalyzer(config)
        self.running = False
        self.symbols = self._get_available_symbols()
        # Trade cooldown tracking: prevent spam trading the same symbol
        self._last_trade_time: Dict[str, datetime] = {}
        self._trade_cooldown = timedelta(seconds=config.get('trading', {}).get('trade_cooldown_seconds', 60))
        # Minimum confidence to execute a trade
        self._min_confidence = config.get('trading', {}).get('entry_conditions', {}).get('min_signal_strength', 0.6)
        
    def _get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols based on configuration"""
        try:
            # Get configured symbols from config
            configured_symbols = self.config['trading']['symbols']
            available_symbols = []
            
            # Validate each configured symbol
            for symbol in configured_symbols:
                if mt5.symbol_select(symbol, True):
                    available_symbols.append(symbol)
                else:
                    self.logger.warning(f"Symbol {symbol} not available in MT5")
                    
            self.logger.info(f"Found {len(available_symbols)} available symbols: {available_symbols}")
            return available_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {str(e)}")
            return []
            
    async def start(self):
        """Start the trading bot"""
        try:
            self.running = True
            self.logger.info("Trading bot started")
            
            while self.running:
                try:
                    # Process each symbol
                    for symbol in self.symbols:
                        if not self.running:
                            break
                        await self._process_symbol(symbol)
                        
                    # Manage existing positions
                    try:
                        await self.position_manager.manage_positions()
                    except Exception as e:
                        self.logger.error(f"Error managing positions: {str(e)}")
                        
                    # Update risk metrics
                    try:
                        acc_info = await self.broker.get_account_info()
                        self.risk_manager.update_risk_metrics(acc_info['balance'], acc_info['equity'])
                    except Exception as e:
                        self.logger.error(f"Error updating risk metrics: {str(e)}")
                        
                except Exception as e:
                    self.logger.error(f"Error in main trading loop: {str(e)}")
                finally:
                    # Prevent CPU lockup in any circumstance (e.g. market closed, API failures)
                    await asyncio.sleep(self.config.get('trading', {}).get('update_interval', 5))
                    
        except Exception as e:
            self.logger.error(f"Fatal error in trading bot: {str(e)}")
        finally:
            await self.stop()
            
    async def _process_symbol(self, symbol: str):
        """Process a single symbol for trading opportunities"""
        try:
            # Check if market is open for this symbol
            if not self._is_market_open(symbol):
                self.logger.debug(f"{symbol}: Market closed, skipping")
                return
                
            # Get market data and analyze for each timeframe
            timeframe = self.config['trading']['timeframes'][0]  # Use primary timeframe
            market_data = await self.market_analyzer.get_market_data(symbol, timeframe)
            if market_data is None:
                self.logger.debug(f"{symbol}: No market data available")
                return
                
            # Analyze market data
            market_analysis = await self.market_analyzer.analyze_market(symbol, timeframe)
            if market_analysis is None:
                self.logger.debug(f"{symbol}: Market analysis returned None")
                return
                
            # Generate trading signals
            try:
                signals = await self.market_analyzer.generate_signals(symbol, timeframe)
            except Exception as e:
                self.logger.error(f"Error generating signals for {symbol}: {str(e)}")
                return
                
            # Get ML predictions
            try:
                ml_result = await self.ml_analyzer.predict(market_data, symbol, timeframe)
                if ml_result:
                    ml_signals = {
                        'direction': ml_result.get('signal'),
                        'confidence': ml_result.get('confidence', 0.0)
                    }
                else:
                    ml_signals = None
            except Exception as e:
                self.logger.error(f"Error getting ML signals for {symbol}: {str(e)}")
                ml_signals = None
                
            # Combine signals
            final_signal = self._combine_signals(signals, ml_signals)
            
            if final_signal and final_signal.direction != 'none':
                self.logger.info(f"{symbol}: Signal detected — direction={final_signal.direction}, "
                               f"confidence={final_signal.confidence:.2f}, strength={final_signal.strength}")
                
                # Check minimum confidence threshold
                if final_signal.confidence < self._min_confidence:
                    self.logger.info(f"{symbol}: Signal too weak ({final_signal.confidence:.2f} < {self._min_confidence}), skipping")
                    return
                    
                # Check per-symbol trade cooldown
                last_trade = self._last_trade_time.get(symbol)
                if last_trade and (datetime.now() - last_trade) < self._trade_cooldown:
                    remaining = (self._trade_cooldown - (datetime.now() - last_trade)).seconds
                    self.logger.debug(f"{symbol}: Trade cooldown active, {remaining}s remaining")
                    return
                # Calculate position size
                try:
                    account_info = await self.broker.get_account_info()
                    balance = account_info.get('balance', 10000.0) if account_info else 10000.0
                    # Build a dictionary of market metrics for risk manager adjustments
                    regime = 'unknown'
                    volatility = 1.0
                    if market_analysis and 'market_structure' in market_analysis:
                        volatility = market_analysis['market_structure'].get('volatility', 1.0)
                        
                    market_info_dict = {
                        'regime': {'regime': regime} if isinstance(regime, str) else regime,
                        'confidence': ml_result.get('confidence', 0.5) if ml_result else 0.5,
                        'volatility': volatility,
                        'signal_strength': getattr(final_signal, 'confidence', 0.5),
                        'strategy_type': 'trend_following' if regime in ['bull', 'bear'] else 'mean_reversion'
                    }

                    pos_size, risk_info = await self.risk_manager.calculate_position_size(
                        symbol=symbol,
                        entry_price=final_signal.entry_price,
                        stop_loss=final_signal.stop_loss,
                        account_balance=balance,
                        market_data=market_info_dict
                    )
                    final_signal.position_size = pos_size
                    self.logger.info(f"{symbol}: Position size calculated = {pos_size}")
                except Exception as e:
                    self.logger.error(f"Error calculating position size for {symbol}: {str(e)}")
                    final_signal.position_size = 0.01

                # Validate trading conditions
                try:
                    if await self.risk_manager.validate_trade(symbol, final_signal):
                        self.logger.info(f"{symbol}: Trade validated, executing...")
                        # Execute trade
                        await self.trade_executor.execute_trade(final_signal)
                        # Record trade time for cooldown
                        self._last_trade_time[symbol] = datetime.now()
                    else:
                        self.logger.info(f"{symbol}: Trade rejected by risk manager")
                except Exception as e:
                    self.logger.error(f"Error validating/executing trade for {symbol}: {str(e)}")
            else:
                self.logger.debug(f"{symbol}: No actionable signal this cycle")
                    
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {str(e)}")
            
    async def process_symbol(self, symbol: str):
        """Public method to process trading logic for a symbol"""
        await self._process_symbol(symbol)
            
    def _is_market_open(self, symbol: str) -> bool:
        """Check if market is open for trading"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
                
            return symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL
            
        except Exception as e:
            self.logger.error(f"Error checking market status for {symbol}: {str(e)}")
            return False
            
    def _combine_signals(self, technical_signals: List[Signal], ml_signals: Dict) -> Optional[Signal]:
        """Combine technical and ML signals into final trading decision"""
        try:
            if not technical_signals:
                return None
                
            # Get the strongest technical signal
            final_signal = max(technical_signals, key=lambda x: x.confidence)
            
            # Adjust signal confidence based on ML prediction
            new_confidence = final_signal.confidence
            if ml_signals:
                if ml_signals.get('direction') == final_signal.direction:
                    new_confidence *= (1 + ml_signals.get('confidence', 0))
                else:
                    new_confidence *= 0.5
            
            # Ensure confidence stays within valid range [0, 1]
            new_confidence = min(1.0, max(0.0, new_confidence))
            
            final_signal.update_confidence(new_confidence)
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Error combining signals: {str(e)}")
            return None
            
    async def stop(self):
        """Stop the trading bot"""
        try:
            self.running = False
            
            # Close all positions if configured
            if self.config.get('trading', {}).get('close_positions_on_stop', True):
                try:
                    await self.position_manager.close_all_positions()
                except Exception as e:
                    self.logger.error(f"Error closing positions on stop: {str(e)}")
                    
            self.logger.info("Trading bot stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading bot: {str(e)}") 