import asyncio
import websockets
import json
import logging
from datetime import datetime
import random  # For demo data, remove in production

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketServer:
    def __init__(self):
        self.clients = set()
        self.last_prices = {
            'EURUSD': 1.2150,
            'GBPUSD': 1.3850,
            'USDJPY': 110.50,
            'AUDUSD': 0.7450,
        }
        
    async def register(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        logger.info(f"New client connected. Total clients: {len(self.clients)}")
        
    async def unregister(self, websocket):
        """Unregister a client"""
        self.clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        
    def generate_mock_data(self):
        """Generate mock trading data for demo purposes"""
        # Update mock prices
        for symbol in self.last_prices:
            change = random.uniform(-0.0010, 0.0010)
            self.last_prices[symbol] = round(self.last_prices[symbol] + change, 4)
            
        return {
            'timestamp': datetime.now().isoformat(),
            'account_balance': round(random.uniform(9800, 12000), 2),
            'equity': round(random.uniform(9800, 12000), 2),
            'open_positions': random.randint(1, 5),
            'daily_profit': round(random.uniform(-100, 300), 2),
            'prices': self.last_prices,
            'positions': [
                {
                    'symbol': 'EURUSD',
                    'type': 'BUY',
                    'volume': 0.1,
                    'entry_price': 1.2150,
                    'current_price': self.last_prices['EURUSD'],
                    'profit': round((self.last_prices['EURUSD'] - 1.2150) * 10000, 1)
                },
                {
                    'symbol': 'GBPUSD',
                    'type': 'SELL',
                    'volume': 0.1,
                    'entry_price': 1.3850,
                    'current_price': self.last_prices['GBPUSD'],
                    'profit': round((1.3850 - self.last_prices['GBPUSD']) * 10000, 1)
                }
            ]
        }
        
    async def broadcast_updates(self):
        """Broadcast updates to all connected clients"""
        while True:
            if not self.clients:
                await asyncio.sleep(1)
                continue
                
            # Generate mock data (replace with real MT5 data in production)
            data = self.generate_mock_data()
            
            # Broadcast to all connected clients
            websockets_to_remove = set()
            for websocket in self.clients:
                try:
                    await websocket.send(json.dumps(data))
                except websockets.exceptions.ConnectionClosed:
                    websockets_to_remove.add(websocket)
                    
            # Remove disconnected clients
            for websocket in websockets_to_remove:
                await self.unregister(websocket)
                
            await asyncio.sleep(1)  # Update every second
            
    async def handle_client(self, websocket, path):
        """Handle individual client connections"""
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get('type') == 'subscribe':
                        # Handle subscription requests
                        response = {'type': 'subscribed', 'channels': data.get('channels', [])}
                        await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
        finally:
            await self.unregister(websocket)
            
async def main():
    server = WebSocketServer()
    
    # Start the broadcast task
    broadcast_task = asyncio.create_task(server.broadcast_updates())
    
    async with websockets.serve(server.handle_client, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
