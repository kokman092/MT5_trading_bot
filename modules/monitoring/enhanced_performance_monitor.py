import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import json
import requests
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

class EnhancedPerformanceMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Monitoring parameters
        self.params = {
            'metrics_update_interval': config.get('metrics_update_interval', 60),  # seconds
            'alert_thresholds': config.get('alert_thresholds', {
                'drawdown': 0.05,
                'consecutive_losses': 3,
                'profit_target': 0.02,
                'risk_exposure': 0.8
            }),
            'dashboard_refresh': config.get('dashboard_refresh', 5),  # seconds
            'history_retention': config.get('history_retention', '30d')
        }
        
        # Initialize InfluxDB client
        self.influx_client = InfluxDBClient(
            url=config['influx_url'],
            token=config['influx_token'],
            org=config['influx_org']
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        
        # Initialize metrics storage
        self.current_metrics = {}
        self.alerts = []
        self.last_update = None

    def update_performance_metrics(self, 
                                 account_info: Dict,
                                 positions: List[Dict],
                                 trades: List[Dict],
                                 strategy_performance: Dict) -> None:
        """
        Update and store performance metrics
        """
        try:
            # Calculate current metrics
            metrics = self._calculate_metrics(
                account_info, positions, trades, strategy_performance
            )
            
            # Store metrics in InfluxDB
            self._store_metrics(metrics)
            
            # Check for alerts
            self._check_alerts(metrics)
            
            # Update current metrics
            self.current_metrics = metrics
            self.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")

    def _calculate_metrics(self, account_info: Dict, positions: List[Dict],
                         trades: List[Dict], strategy_performance: Dict) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        # Account metrics
        account_metrics = {
            'balance': account_info['balance'],
            'equity': account_info['equity'],
            'margin': account_info['margin'],
            'free_margin': account_info['free_margin'],
            'margin_level': account_info['margin_level']
        }
        
        # Calculate drawdown
        peak_equity = max(trade['equity'] for trade in trades) if trades else account_info['balance']
        current_drawdown = (peak_equity - account_info['equity']) / peak_equity
        
        # Trading metrics
        trading_metrics = self._calculate_trading_metrics(trades)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(positions, account_info['equity'])
        
        # Strategy metrics
        strategy_metrics = self._calculate_strategy_metrics(strategy_performance)
        
        return {
            'account': account_metrics,
            'trading': trading_metrics,
            'risk': risk_metrics,
            'strategy': strategy_metrics,
            'drawdown': current_drawdown,
            'timestamp': datetime.now()
        }

    def _calculate_trading_metrics(self, trades: List[Dict]) -> Dict:
        """
        Calculate trading performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'average_profit': 0,
                'average_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'consecutive_wins': 0,
                'consecutive_losses': 0
            }
            
        # Calculate basic metrics
        profitable_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] <= 0]
        
        win_rate = len(profitable_trades) / len(trades)
        
        total_profit = sum(t['profit'] for t in profitable_trades)
        total_loss = abs(sum(t['profit'] for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in trades:
            if trade['profit'] > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                    
            max_win_streak = max(max_win_streak, current_streak)
            max_loss_streak = min(max_loss_streak, current_streak)
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_profit': total_profit / len(profitable_trades) if profitable_trades else 0,
            'average_loss': total_loss / len(losing_trades) if losing_trades else 0,
            'largest_win': max(t['profit'] for t in trades),
            'largest_loss': min(t['profit'] for t in trades),
            'consecutive_wins': max_win_streak,
            'consecutive_losses': abs(max_loss_streak)
        }

    def _calculate_risk_metrics(self, positions: List[Dict], 
                              equity: float) -> Dict:
        """
        Calculate risk metrics
        """
        total_exposure = sum(pos['margin'] for pos in positions)
        risk_exposure = total_exposure / equity if equity > 0 else 0
        
        position_sizes = [pos['margin'] / equity for pos in positions]
        
        return {
            'total_exposure': total_exposure,
            'risk_exposure': risk_exposure,
            'largest_position': max(position_sizes) if position_sizes else 0,
            'position_count': len(positions),
            'average_position_size': np.mean(position_sizes) if position_sizes else 0
        }

    def _calculate_strategy_metrics(self, strategy_performance: Dict) -> Dict:
        """
        Calculate strategy-specific metrics
        """
        strategy_metrics = {}
        
        for strategy, performance in strategy_performance.items():
            strategy_metrics[strategy] = {
                'profit': performance.get('profit', 0),
                'trades': performance.get('trades', 0),
                'win_rate': performance.get('win_rate', 0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0)
            }
            
        return strategy_metrics

    def _store_metrics(self, metrics: Dict) -> None:
        """
        Store metrics in InfluxDB
        """
        points = []
        
        # Create points for each metric category
        for category, category_metrics in metrics.items():
            if isinstance(category_metrics, dict):
                for name, value in category_metrics.items():
                    if isinstance(value, (int, float)):
                        point = Point("trading_metrics") \
                            .tag("category", category) \
                            .tag("metric", name) \
                            .field("value", value) \
                            .time(datetime.utcnow())
                        points.append(point)
                        
        # Write points to InfluxDB
        self.write_api.write(
            bucket=self.config['influx_bucket'],
            record=points
        )

    def _check_alerts(self, metrics: Dict) -> None:
        """
        Check for alert conditions
        """
        alerts = []
        
        # Check drawdown
        if metrics['drawdown'] > self.params['alert_thresholds']['drawdown']:
            alerts.append({
                'type': 'DRAWDOWN',
                'severity': 'HIGH',
                'message': f"Drawdown exceeded threshold: {metrics['drawdown']:.2%}"
            })
            
        # Check consecutive losses
        if metrics['trading']['consecutive_losses'] >= \
           self.params['alert_thresholds']['consecutive_losses']:
            alerts.append({
                'type': 'CONSECUTIVE_LOSSES',
                'severity': 'MEDIUM',
                'message': f"Consecutive losses: {metrics['trading']['consecutive_losses']}"
            })
            
        # Check risk exposure
        if metrics['risk']['risk_exposure'] > \
           self.params['alert_thresholds']['risk_exposure']:
            alerts.append({
                'type': 'RISK_EXPOSURE',
                'severity': 'HIGH',
                'message': f"High risk exposure: {metrics['risk']['risk_exposure']:.2%}"
            })
            
        # Store and notify alerts
        self.alerts.extend(alerts)
        self._notify_alerts(alerts)

    def _notify_alerts(self, alerts: List[Dict]) -> None:
        """
        Send alert notifications
        """
        if not alerts:
            return
            
        # Implement notification logic (e.g., email, Slack, etc.)
        for alert in alerts:
            self.logger.warning(f"Alert: {alert['type']} - {alert['message']}")
            
            # Send to notification service if configured
            if 'notification_url' in self.config:
                try:
                    requests.post(
                        self.config['notification_url'],
                        json=alert
                    )
                except Exception as e:
                    self.logger.error(f"Error sending notification: {str(e)}")

    def get_current_metrics(self) -> Dict:
        """
        Get current performance metrics
        """
        return self.current_metrics

    def get_alerts(self, severity: str = None) -> List[Dict]:
        """
        Get current alerts, optionally filtered by severity
        """
        if severity:
            return [alert for alert in self.alerts 
                   if alert['severity'] == severity]
        return self.alerts

    def clear_alerts(self) -> None:
        """
        Clear all alerts
        """
        self.alerts = []
