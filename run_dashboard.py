import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
import os

class TradingDashboard:
    def __init__(self):
        self.data_path = "data"
        self.log_path = "logs"
        
    def load_trading_data(self):
        """Load trading data from log files"""
        try:
            # Load trade history
            trades_file = os.path.join(self.data_path, "trades.json")
            if os.path.exists(trades_file):
                with open(trades_file, "r") as f:
                    trades = json.load(f)
                trades_df = pd.DataFrame(trades)
                trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
                # Calculate cumulative profit
                trades_df["cumulative_profit"] = trades_df["profit"].cumsum()
                return trades_df
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading trading data: {str(e)}")
            return pd.DataFrame()
    
    def create_equity_chart(self, trades_df):
        """Create equity curve chart"""
        if trades_df.empty:
            return None
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades_df["timestamp"],
            y=trades_df["cumulative_profit"],  # Now using pre-calculated cumulative profit
            mode="lines",
            name="Equity Curve"
        ))
        
        fig.update_layout(
            title="Account Equity Curve",
            xaxis_title="Date",
            yaxis_title="Profit/Loss",
            template="plotly_dark"
        )
        return fig
    
    def calculate_performance_metrics(self, trades_df):
        """Calculate key performance metrics"""
        if trades_df.empty:
            return {
                "Total Trades": 0,
                "Win Rate": 0,
                "Profit Factor": 0,
                "Average Win": 0,
                "Average Loss": 0,
                "Max Drawdown": 0
            }
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["profit"] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        winning_trades_df = trades_df[trades_df["profit"] > 0]
        losing_trades_df = trades_df[trades_df["profit"] < 0]
        
        avg_win = winning_trades_df["profit"].mean() if not winning_trades_df.empty else 0
        avg_loss = abs(losing_trades_df["profit"].mean()) if not losing_trades_df.empty else 0
        
        total_profit = winning_trades_df["profit"].sum() if not winning_trades_df.empty else 0
        total_loss = abs(losing_trades_df["profit"].sum()) if not losing_trades_df.empty else 0
        profit_factor = total_profit / total_loss if total_loss != 0 else 0
        
        # Calculate max drawdown
        cumulative = trades_df["cumulative_profit"]
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative - rolling_max
        max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0
        
        return {
            "Total Trades": total_trades,
            "Win Rate": f"{win_rate:.2f}%",
            "Profit Factor": f"{profit_factor:.2f}",
            "Average Win": f"${avg_win:.2f}",
            "Average Loss": f"${avg_loss:.2f}",
            "Max Drawdown": f"${max_drawdown:.2f}"
        }
    
    def run(self):
        """Run the dashboard"""
        st.set_page_config(
            page_title="Trading Bot Dashboard",
            page_icon="📈",
            layout="wide"
        )
        
        st.title("Trading Bot Dashboard")
        
        # Load data
        trades_df = self.load_trading_data()
        
        # Display metrics
        metrics = self.calculate_performance_metrics(trades_df)
        cols = st.columns(len(metrics))
        for col, (metric, value) in zip(cols, metrics.items()):
            col.metric(metric, value)
        
        # Display equity chart
        equity_chart = self.create_equity_chart(trades_df)
        if equity_chart:
            st.plotly_chart(equity_chart, use_container_width=True)
        
        # Display recent trades
        st.subheader("Recent Trades")
        if not trades_df.empty:
            recent_trades = trades_df.sort_values("timestamp", ascending=False).head(10)
            st.dataframe(recent_trades[["timestamp", "symbol", "type", "profit", "pips"]])
        else:
            st.info("No trades available")
        
        # System status
        st.subheader("System Status")
        status_cols = st.columns(3)
        status_cols[0].metric("Bot Status", "Running")
        status_cols[1].metric("Open Positions", "0")
        status_cols[2].metric("Last Update", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run() 