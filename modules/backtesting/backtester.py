import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import ta
import ta.trend
import ta.momentum
import ta.volatility
from typing import Dict, List, Optional

class Backtester:
    def __init__(self, config: Dict):
        self.config = config
        self.timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        
    def run_backtest(self, params: Dict) -> Dict:
        """Run backtest with specified parameters"""
        try:
            # Parse parameters
            symbol = params['symbol']
            timeframe = params['timeframe']
            start_date = datetime.strptime(params['start_date'], '%Y-%m-%d %H:%M')
            end_date = datetime.strptime(params['end_date'], '%Y-%m-%d %H:%M')
            initial_balance = float(params['initial_balance'])
            
            # Get historical data
            df = self._get_historical_data(symbol, timeframe, start_date, end_date)
            if df is None or df.empty:
                return {
                    'success': False,
                    'error': f'No data available for {symbol} from {start_date} to {end_date}'
                }
                
            # Add technical indicators
            df = self._add_indicators(df)
            
            # Run simulation
            results = self._simulate_trades(df, initial_balance)
            
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def _get_historical_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get historical data from MT5"""
        try:
            mt5_timeframe = self.timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception:
            return None
            
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # Volatility indicators
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        # Trend indicators
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
        
        # Stochastic
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        
        return df
        
    def _simulate_trades(self, df: pd.DataFrame, initial_balance: float) -> Dict:
        """Simulate trades with risk management"""
        balance = initial_balance
        trades = []
        equity_curve = [{'time': df['time'].iloc[0], 'equity': balance}]
        open_position = None
        consecutive_losses = 0
        max_consecutive_losses = 0
        daily_returns = []
        total_trades = 0
        winning_trades = 0
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            current_time = df['time'].iloc[i]
            
            # Update open position if exists
            if open_position:
                # Calculate profit/loss
                if open_position['type'] == 'buy':
                    profit = (current_price - open_position['entry_price']) * open_position['size']
                else:
                    profit = (open_position['entry_price'] - current_price) * open_position['size']
                    
                # Check stop loss and take profit
                if profit <= -open_position['stop_loss'] or profit >= open_position['take_profit']:
                    balance += profit
                    trades.append({
                        'entry_time': open_position['entry_time'],
                        'exit_time': current_time,
                        'type': open_position['type'],
                        'entry_price': open_position['entry_price'],
                        'exit_price': current_price,
                        'profit': profit,
                        'balance': balance
                    })
                    
                    if profit > 0:
                        winning_trades += 1
                        consecutive_losses = 0
                    else:
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        
                    total_trades += 1
                    open_position = None
                    
            # Look for new trade signals
            if not open_position and self._check_entry_conditions(df, i):
                # Calculate position size with risk management
                risk_amount = balance * self.config['risk_management']['risk_per_trade']
                stop_loss_pips = df['atr'].iloc[i] * self.config['risk_management']['stop_loss_atr_multiplier']
                position_size = risk_amount / stop_loss_pips
                
                # Open new position
                open_position = {
                    'type': 'buy' if df['ema_50'].iloc[i] > df['ema_200'].iloc[i] else 'sell',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'size': position_size,
                    'stop_loss': stop_loss_pips,
                    'take_profit': stop_loss_pips * self.config['risk_management']['risk_reward_ratio']
                }
                
            # Update equity curve
            if current_time.date() != equity_curve[-1]['time'].date():
                daily_returns.append((balance / equity_curve[-1]['equity']) - 1)
                
            equity_curve.append({
                'time': current_time,
                'equity': balance + (profit if open_position else 0)
            })
            
        # Calculate performance metrics
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        
        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': ((balance / initial_balance) - 1) * 100,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'max_consecutive_losses': max_consecutive_losses,
            'trades': trades,
            'equity_curve': equity_curve
        }
        
    def _check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if entry conditions are met"""
        # Trend alignment
        trend_aligned = (
            df['ema_50'].iloc[index] > df['ema_200'].iloc[index] and
            df['close'].iloc[index] > df['ema_50'].iloc[index]
        ) or (
            df['ema_50'].iloc[index] < df['ema_200'].iloc[index] and
            df['close'].iloc[index] < df['ema_50'].iloc[index]
        )
        
        # Momentum confirmation
        momentum_confirmed = (
            df['macd'].iloc[index] > df['macd_signal'].iloc[index] and
            df['rsi'].iloc[index] > 50
        ) or (
            df['macd'].iloc[index] < df['macd_signal'].iloc[index] and
            df['rsi'].iloc[index] < 50
        )
        
        # Volatility check
        volatility_suitable = (
            df['atr_percent'].iloc[index] > self.config['market_analysis']['volatility']['min_threshold'] and
            df['atr_percent'].iloc[index] < self.config['market_analysis']['volatility']['max_threshold']
        )
        
        return trend_aligned and momentum_confirmed and volatility_suitable
        
    def _calculate_max_drawdown(self, equity_curve: List[Dict]) -> float:
        """Calculate maximum drawdown"""
        equity_values = [point['equity'] for point in equity_curve]
        peak = equity_values[0]
        max_dd = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd * 100
        
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0
            
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if len(returns_array) < 2:
            return 0
            
        return np.sqrt(252) * (np.mean(excess_returns) / np.std(returns_array, ddof=1)) 