from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import ta
import ta.trend
import ta.momentum
import ta.volatility
import asyncio
from asyncio import WebSocketDisconnect
from websockets import WebSocket
from modules.web.shared import initialize_mt5, get_account_info, get_open_positions, calculate_daily_pl, validate_trading_conditions
from modules.backtesting.backtester import Backtester

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
socketio = SocketIO(app)

# Global variables for storing trading data
trading_data = {
    'balance': 0.0,
    'open_positions': 0,
    'daily_pl': 0.0,
    'win_rate': 0.0,
    'balance_history': [],
    'performance_data': [],
    'active_trades': []
}

class DashboardWebSocket:
    def __init__(self):
        self.clients = set()
        self.last_data = None
    
    async def register(self, websocket):
        self.clients.add(websocket)
        if self.last_data:
            await websocket.send(json.dumps(self.last_data))
    
    async def unregister(self, websocket):
        self.clients.remove(websocket)
    
    async def broadcast(self, data):
        self.last_data = data
        if self.clients:
            await asyncio.gather(
                *[client.send(json.dumps(data)) for client in self.clients]
            )

dashboard_ws = DashboardWebSocket()

def update_trading_data():
    """Update trading data periodically"""
    while True:
        account_info = get_account_info()
        if account_info:
            trading_data['balance'] = account_info['balance']
            positions = get_open_positions()
            trading_data['open_positions'] = len(positions)
            trading_data['daily_pl'] = calculate_daily_pl()
            
            # Update balance history
            trading_data['balance_history'].append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'balance': account_info['balance']
            })
            if len(trading_data['balance_history']) > 100:
                trading_data['balance_history'].pop(0)
            
            # Update performance data
            performance_by_symbol = {}
            for pos in positions:
                symbol = pos['symbol']
                if symbol not in performance_by_symbol:
                    performance_by_symbol[symbol] = 0
                performance_by_symbol[symbol] += pos['profit']
            
            trading_data['performance_data'] = [
                {'symbol': symbol, 'profit': profit}
                for symbol, profit in performance_by_symbol.items()
            ]
            
            # Update active trades
            trading_data['active_trades'] = positions
            
            # Emit updated data to all connected clients
            socketio.emit('update_data', trading_data)
        
        time.sleep(1)  # Update every second

@app.route('/')
def index():
    """Render dashboard template"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    emit('update_data', trading_data)

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """Run backtest with specified parameters"""
    try:
        data = request.get_json()
        backtester = Backtester(app.config)
        result = backtester.run_backtest(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/market_regime')
def get_market_regime():
    """Get current market regime analysis"""
    try:
        symbol = request.args.get('symbol', 'EURUSD')
        timeframe = request.args.get('timeframe', 'H1')
        
        # Get market data
        market_data = get_market_data(symbol, timeframe)
        if market_data is None:
            return jsonify({'error': 'Failed to get market data'})
            
        # Analyze regime
        regime = analyze_market_regime(market_data)
        return jsonify(regime)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/model_performance')
def get_model_performance():
    """Get ML model performance metrics"""
    try:
        symbol = request.args.get('symbol', 'EURUSD')
        timeframe = request.args.get('timeframe', 'H1')
        
        # Get model metrics
        metrics = get_model_metrics(symbol, timeframe)
        return jsonify(metrics)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/predictions')
def get_predictions():
    """Get current market predictions"""
    try:
        symbol = request.args.get('symbol', 'EURUSD')
        timeframe = request.args.get('timeframe', 'H1')
        
        # Get predictions
        predictions = get_market_predictions(symbol, timeframe)
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/parameters')
def get_parameters():
    """Get current trading parameters"""
    try:
        return jsonify({
            'symbols': app.config['trading']['symbols'],
            'timeframes': app.config['trading']['timeframes'],
            'risk_management': app.config['risk_management'],
            'technical_indicators': app.config['market_analysis']['technical_indicators']
        })
    except Exception as e:
        return jsonify({'error': str(e)})

async def update_dashboard():
    """Update dashboard data periodically"""
    while True:
        try:
            # Get updated data
            account_info = get_account_info()
            positions = get_open_positions()
            daily_pl = calculate_daily_pl()
            
            # Prepare dashboard data
            dashboard_data = {
                'account': account_info,
                'positions': positions,
                'daily_pl': daily_pl,
                'timestamp': datetime.now().isoformat()
            }
            
            # Broadcast to all clients
            await dashboard_ws.broadcast(dashboard_data)
            
        except Exception as e:
            logging.error(f"Error updating dashboard: {str(e)}")
        
        await asyncio.sleep(1)

@app.websocket('/ws')
async def websocket_endpoint():
    """WebSocket endpoint for real-time updates"""
    try:
        websocket = WebSocket()
    await dashboard_ws.register(websocket)
        
    try:
        while True:
                data = await websocket.receive_text()
                # Handle incoming messages if needed
                
    except WebSocketDisconnect:
        await dashboard_ws.unregister(websocket)

    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")

@app.before_first_request
def start_dashboard_update():
    """Start dashboard update task"""
    threading.Thread(target=lambda: asyncio.run(update_dashboard()), daemon=True).start()

def start_dashboard():
    """Start the dashboard application"""
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    start_dashboard()
