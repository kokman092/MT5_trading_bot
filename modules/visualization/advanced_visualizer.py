import logging
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional
from datetime import datetime

class AdvancedVisualizer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.chart_settings = config.get('visualization', {}).get('chart_settings', {
            'plot_width': 1200,
            'plot_height': 800,
            'background_color': '#1a1a1a',
            'grid_color': '#333333',
            'text_color': '#ffffff'
        })
        
    def create_candlestick_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create an interactive candlestick chart"""
        try:
            fig = go.Figure(data=[
                go.Candlestick(
                    x=df['time'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close']
                )
            ])
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Price Chart',
                yaxis_title='Price',
                template='plotly_dark',
                width=self.chart_settings['plot_width'],
                height=self.chart_settings['plot_height']
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating candlestick chart: {str(e)}")
            return None
            
    def add_indicators(self, fig: go.Figure, df: pd.DataFrame, indicators: List[Dict]) -> go.Figure:
        """Add technical indicators to the chart"""
        try:
            for indicator in indicators:
                if indicator['type'] == 'MA':
                    fig.add_trace(
                        go.Scatter(
                            x=df['time'],
                            y=df[indicator['column']],
                            name=indicator['name'],
                            line=dict(color=indicator.get('color', '#00ff00'))
                        )
                    )
                elif indicator['type'] == 'VOLUME':
                    fig.add_trace(
                        go.Bar(
                            x=df['time'],
                            y=df['tick_volume'],
                            name='Volume',
                            yaxis='y2'
                        )
                    )
                    
            return fig
            
        except Exception as e:
            self.logger.error(f"Error adding indicators: {str(e)}")
            return fig
            
    def save_chart(self, fig: go.Figure, filename: str) -> bool:
        """Save chart to file"""
        try:
            fig.write_html(filename)
            return True
        except Exception as e:
            self.logger.error(f"Error saving chart: {str(e)}")
            return False
            
    def create_performance_dashboard(self, performance_metrics: Dict) -> go.Figure:
        """Create a performance dashboard"""
        try:
            # Create subplots
            fig = go.Figure()
            
            # Add performance metrics
            for metric_name, metric_value in performance_metrics.items():
                fig.add_trace(
                    go.Indicator(
                        mode="number+delta",
                        value=metric_value['current'],
                        delta={'reference': metric_value['previous']},
                        title={'text': metric_name},
                        domain={'row': 0, 'column': 0}
                    )
                )
                
            # Update layout
            fig.update_layout(
                title='Trading Performance Dashboard',
                template='plotly_dark',
                width=self.chart_settings['plot_width'],
                height=self.chart_settings['plot_height']
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating performance dashboard: {str(e)}")
            return None 