import logging
from typing import Dict, Optional, List
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import json
import sqlite3
import os

class Monitor:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.setup_database()
        
    def setup_logging(self):
        """Configure logging"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"trading_{datetime.now().strftime('%Y%m%d')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def setup_database(self):
        """Initialize SQLite database"""
        try:
            self.conn = sqlite3.connect('trading_history.db')
            cursor = self.conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    action TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    volume REAL,
                    profit REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT
                )
            ''')
            
            # Create market_data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    timeframe TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL
                )
            ''')
            
            self.conn.commit()
            
        except Exception as e:
            logging.error(f"Database setup error: {str(e)}")
            
    def log_trade(self, trade_data: Dict):
        """Log trade to database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO trades (
                    symbol, action, entry_price, volume,
                    stop_loss, take_profit, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data['symbol'],
                trade_data['action'],
                trade_data['entry_price'],
                trade_data['volume'],
                trade_data['stop_loss'],
                trade_data['take_profit'],
                'OPEN'
            ))
            self.conn.commit()
            
        except Exception as e:
            logging.error(f"Error logging trade: {str(e)}")
            
    def update_trade(self, trade_id: int, exit_price: float, profit: float):
        """Update trade with exit information"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE trades
                SET exit_price = ?, profit = ?, status = 'CLOSED'
                WHERE id = ?
            ''', (exit_price, profit, trade_id))
            self.conn.commit()
            
        except Exception as e:
            logging.error(f"Error updating trade: {str(e)}")
            
    def log_market_data(self, market_data: Dict):
        """Log market data to database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO market_data (
                    symbol, timeframe, open, high, low, close, volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                market_data['symbol'],
                market_data['timeframe'],
                market_data['open'],
                market_data['high'],
                market_data['low'],
                market_data['close'],
                market_data['volume']
            ))
            self.conn.commit()
            
        except Exception as e:
            logging.error(f"Error logging market data: {str(e)}")
            
    def send_notification(self, subject: str, message: str):
        """Send email notification"""
        try:
            if not hasattr(self.config, 'EMAIL_SETTINGS'):
                return
                
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = self.config['EMAIL_SETTINGS']['from']
            msg['To'] = self.config['EMAIL_SETTINGS']['to']
            
            with smtplib.SMTP_SSL(self.config['EMAIL_SETTINGS']['smtp_server']) as server:
                server.login(
                    self.config['EMAIL_SETTINGS']['username'],
                    self.config['EMAIL_SETTINGS']['password']
                )
                server.send_message(msg)
                
        except Exception as e:
            logging.error(f"Error sending notification: {str(e)}")
            
    def get_trading_statistics(self) -> Dict:
        """Calculate trading statistics"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(profit) as total_profit,
                    AVG(profit) as avg_profit,
                    MAX(profit) as max_profit,
                    MIN(profit) as max_loss
                FROM trades
                WHERE status = 'CLOSED'
            ''')
            
            row = cursor.fetchone()
            if not row:
                return {}
                
            total_trades = row[0]
            win_rate = (row[1] / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': row[1],
                'losing_trades': row[2],
                'win_rate': round(win_rate, 2),
                'total_profit': round(row[3], 2),
                'average_profit': round(row[4], 2),
                'max_profit': round(row[5], 2),
                'max_loss': round(row[6], 2)
            }
            
        except Exception as e:
            logging.error(f"Error calculating statistics: {str(e)}")
            return {}
