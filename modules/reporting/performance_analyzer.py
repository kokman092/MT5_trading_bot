import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class PerformanceAnalyzer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create directories if they don't exist
        Path("data").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        Path("charts").mkdir(exist_ok=True)
        
        self.trade_history_file = config['reporting']['TRADE_HISTORY_FILE']
        self.performance_report_file = config['reporting']['PERFORMANCE_REPORT_FILE']
        self.trades_df = self.load_trade_history()
        
    def load_trade_history(self):
        """Load trade history from CSV file"""
        try:
            return pd.read_csv(self.trade_history_file)
        except FileNotFoundError:
            return pd.DataFrame(columns=[
                'time', 'symbol', 'type', 'entry', 'exit_price', 'size',
                'profit', 'close_reason', 'holding_time'
            ])
        
    def add_trade(self, trade):
        """Add new trade to history"""
        try:
            trade_df = pd.DataFrame([trade])
            self.trades_df = pd.concat([self.trades_df, trade_df], ignore_index=True)
            self.trades_df.to_csv(self.trade_history_file, index=False)
        except Exception as e:
            self.logger.error(f"Error adding trade to history: {str(e)}")
    
    def calculate_metrics(self, timeframe='all'):
        """Calculate performance metrics for specified timeframe"""
        try:
            df = self.trades_df.copy()
            df['time'] = pd.to_datetime(df['time'])
            
            if timeframe == 'today':
                df = df[df['time'].dt.date == datetime.now().date()]
            elif timeframe == 'week':
                one_week_ago = datetime.now() - timedelta(days=7)
                df = df[df['time'] >= one_week_ago]
            elif timeframe == 'month':
                one_month_ago = datetime.now() - timedelta(days=30)
                df = df[df['time'] >= one_month_ago]
            
            if len(df) == 0:
                return None
            
            # Basic metrics
            total_trades = len(df)
            winning_trades = len(df[df['profit'] > 0])
            losing_trades = len(df[df['profit'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit metrics
            total_profit = df['profit'].sum()
            avg_profit = df['profit'].mean()
            max_profit = df['profit'].max()
            max_loss = df['profit'].min()
            
            # Calculate drawdown
            cumulative_returns = (1 + df['profit']).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Risk metrics
            profit_factor = abs(df[df['profit'] > 0]['profit'].sum()) / abs(df[df['profit'] < 0]['profit'].sum()) if losing_trades > 0 else float('inf')
            sharpe_ratio = np.sqrt(252) * (df['profit'].mean() / df['profit'].std()) if len(df) > 1 else 0
            
            # Trade analysis
            avg_holding_time = pd.to_timedelta(df['holding_time']).mean()
            
            # Exit analysis
            exit_reasons = df['close_reason'].value_counts().to_dict()
            
            metrics = {
                'timeframe': timeframe,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': float(total_profit),
                'average_profit': float(avg_profit),
                'max_profit': float(max_profit),
                'max_loss': float(max_loss),
                'max_drawdown': float(max_drawdown),
                'profit_factor': float(profit_factor),
                'sharpe_ratio': float(sharpe_ratio),
                'average_holding_time': str(avg_holding_time),
                'exit_reasons': exit_reasons,
                'timestamp': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return None
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        try:
            report = {
                'all_time': self.calculate_metrics('all'),
                'monthly': self.calculate_metrics('month'),
                'weekly': self.calculate_metrics('week'),
                'today': self.calculate_metrics('today')
            }
            
            # Save report
            with open(self.performance_report_file, 'w') as f:
                json.dump(report, f, indent=4)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return None
    
    def plot_equity_curve(self):
        """Plot equity curve and drawdown"""
        try:
            df = self.trades_df.copy()
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')
            
            # Calculate cumulative returns
            df['cumulative_returns'] = (1 + df['profit']).cumprod()
            
            # Calculate drawdown
            df['peak'] = df['cumulative_returns'].expanding().max()
            df['drawdown'] = (df['cumulative_returns'] - df['peak']) / df['peak']
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
            
            # Plot equity curve
            ax1.plot(df['time'], df['cumulative_returns'], label='Equity Curve')
            ax1.set_title('Equity Curve')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Equity')
            ax1.grid(True)
            
            # Plot drawdown
            ax2.fill_between(df['time'], df['drawdown'], 0, color='red', alpha=0.3)
            ax2.set_title('Drawdown')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Drawdown')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('charts/equity_curve.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {str(e)}")
    
    def plot_trade_analysis(self):
        """Plot trade analysis charts"""
        try:
            df = self.trades_df.copy()
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Profit distribution
            sns.histplot(data=df, x='profit', bins=50, ax=ax1)
            ax1.set_title('Profit Distribution')
            ax1.set_xlabel('Profit')
            ax1.set_ylabel('Count')
            
            # Profit by exit reason
            df_grouped = df.groupby('close_reason')['profit'].agg(['mean', 'count'])
            df_grouped.plot(kind='bar', y='mean', ax=ax2)
            ax2.set_title('Average Profit by Exit Reason')
            ax2.set_xlabel('Exit Reason')
            ax2.set_ylabel('Average Profit')
            
            # Win rate over time
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df['win'] = df['profit'] > 0
            df['win_rate'] = df['win'].rolling(window=20).mean()
            df['win_rate'].plot(ax=ax3)
            ax3.set_title('Win Rate (20-trade Moving Average)')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Win Rate')
            
            # Trade duration analysis
            df['holding_time'] = pd.to_timedelta(df['holding_time'])
            df['holding_hours'] = df['holding_time'].dt.total_seconds() / 3600
            sns.scatterplot(data=df, x='holding_hours', y='profit', ax=ax4)
            ax4.set_title('Profit vs Holding Time')
            ax4.set_xlabel('Holding Time (hours)')
            ax4.set_ylabel('Profit')
            
            plt.tight_layout()
            plt.savefig('charts/trade_analysis.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting trade analysis: {str(e)}")
    
    def generate_performance_charts(self):
        """Generate all performance charts"""
        try:
            self.plot_equity_curve()
            self.plot_trade_analysis()
        except Exception as e:
            self.logger.error(f"Error generating performance charts: {str(e)}")
