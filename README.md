# Professional MT5 Trading Bot

A sophisticated algorithmic trading system for MetaTrader 5, combining technical analysis, machine learning, and professional risk management.

## Features

### 🎯 Core Capabilities
- Multi-timeframe analysis (M5, M15, H1)
- Machine learning-based predictions using LSTM networks
- Advanced risk management system
- Market regime detection and adaptation
- Real-time performance monitoring

### 📊 Technical Analysis
- Multiple technical indicators:
  - MACD (Moving Average Convergence Divergence)          
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - ATR (Average True Range)
  - Moving Averages (EMA/SMA)

### 🤖 Machine Learning Features
- LSTM-based price prediction
- Feature engineering with technical indicators
- Dynamic model retraining
- Confidence-based signal filtering

### ⚠️ Risk Management
- Position sizing based on volatility
- Maximum risk per trade: 2%
- Maximum daily loss: 5%
- Maximum drawdown protection: 10%
- Correlation-based position filtering

### 🔄 Trading Pairs
- EURUSD
- GBPUSD
- USDJPY
- AUDUSD

## Requirements

### System Requirements
- Python 3.8 or higher
- MetaTrader 5 platform
- Minimum 8GB RAM
- Stable internet connection

### Python Dependencies
```
tensorflow>=2.0.0
numpy>=1.19.0
pandas>=1.0.0
MetaTrader5>=5.0.0
ta>=0.7.0
scikit-learn>=0.24.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mt5-trading-bot.git
cd mt5-trading-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your MT5 credentials in `.env`:
```env
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
```

The bot is highly configurable through `config/config.json`:

### Selecting Symbols to Trade (e.g., Gold Only)
To configure the symbols the bot is allowed to trade, modify the `"symbols"` array under the `"trading"` section inside [config/config.json](file:///e:/trading%20bot%20for%20MT5/config/config.json) (and [config/trading_config.json](file:///e:/trading%20bot%20for%20MT5/config/trading_config.json) if it is present):

* **To only trade Gold (XAUUSD)**:
  ```json
  "trading": {
      "symbols": ["XAUUSD"],
      "timeframes": ["M5", "M15", "H1"]
  }
  ```
* **To trade multiple currency pairs and gold**:
  ```json
  "trading": {
      "symbols": ["EURUSD", "GBPUSD", "XAUUSD"],
      "timeframes": ["M5", "M15", "H1"]
  }
  ```

### Market Analysis Settings
```json
"market_analysis": {
    "timeframes": ["M5", "M15", "H1"],
    "technical_indicators": {
        "rsi": {"period": 14},
        "macd": {
            "fast": 12,
            "slow": 26,
            "signal": 9
        }
    }
}
```

### Risk Management Settings
```json
"risk_management": {
    "risk_per_trade": 0.02,
    "max_daily_loss": 0.05,
    "max_positions": 5
}
```

## Usage

1. Start the trading bot:
```bash
python run_trader.py
```

2. Monitor logs in `logs/trading_{datetime}.log`

3. View performance metrics in the web dashboard:
```bash
python run_dashboard.py
streamlit run run_dashboard.py

```

## Safety Features

- Emergency shutdown on excessive losses
- Spread monitoring and filtering
- Slippage protection
- Position size limits
- Correlation checks between positions

## Performance Monitoring

The bot includes comprehensive monitoring:
- Real-time performance metrics
- Trade analysis and statistics
- Risk exposure monitoring
- System health checks

## Architecture

```
trading_bot/
├── modules/
│   ├── trading/
│   │   ├── market_analyzer.py
│   │   ├── ml_analyzer.py
│   │   ├── risk_manager.py
│   │   └── position_manager.py
│   └── config/
├── config/
│   ├── config.json
│   └── risk_management.json
└── logs/
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

Trading forex carries significant risk. This bot is for educational purposes only. Always test thoroughly on a demo account first.

## Support

For support, please open an issue in the GitHub repository
