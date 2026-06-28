import os
import json
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Centralized configuration management for the trading bot
    Supports loading from JSON, YAML, and environment variables
    """
    
    @staticmethod
    def load_config(config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from file or environment variables
        
        Args:
            config_path (str, optional): Path to config file. 
                                         Supports .json and .yaml/.yml extensions
        
        Returns:
            Dict[str, Any]: Loaded configuration
        """
        # Default configuration
        default_config = {
            'account': {
                'login': os.getenv('MT5_LOGIN', ''),
                'password': os.getenv('MT5_PASSWORD', ''),
                'server': os.getenv('MT5_SERVER', ''),
                'timeout': 60000
            },
            'trading': {
                'max_volume': 1.0,
                'min_volume': 0.01,
                'risk_per_trade': 0.02,
                'max_daily_trades': 10,
                'max_concurrent_trades': 5,
                'crypto_settings': {
                    'max_volume_btc': 0.1,
                    'min_volume_btc': 0.001,
                    'increased_margin_requirement': 2.0
                }
            },
            'symbol_prefixes': ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF'],
            'max_spread': 0.0003,
            'update_interval': 1,
            'risk_management': {
                'max_daily_loss': 5.0,
                'max_drawdown': 20.0,
                'volatility_threshold': 0.002,
                'correlation_threshold': 0.7,
                'max_position_size': 0.5,
                'risk_per_trade': 1.0,
                'min_margin_level': 150,
                'max_open_positions': 5
            },
            'technical_indicators': {
                'rsi': {
                    'period': 14,
                    'overbought': 70,
                    'oversold': 30
                },
                'moving_averages': {
                    'fast': 10,
                    'slow': 30
                },
                'atr': {
                    'period': 14
                },
                'bollinger': {
                    'period': 20,
                    'std_dev': 2
                },
                'macd': {
                    'fast_period': 12,
                    'slow_period': 26,
                    'signal_period': 9
                }
            },
            'analysis': {
                'timeframes': ['M5', 'M15', 'H1'],
                'min_data_points': 100,
                'ml_settings': {
                    'min_signal_strength': 0.7,
                    'prediction_threshold': 0.65,
                    'training_period': 1000,
                    'retrain_interval': 168
                }
            },
            'strategies': {
                'trend_following': {
                    'enabled': True,
                    'weight': 0.4,
                    'parameters': {
                        'ma_fast': 10,
                        'ma_slow': 30,
                        'atr_period': 14
                    }
                },
                'mean_reversion': {
                    'enabled': True,
                    'weight': 0.3,
                    'parameters': {
                        'lookback': 20,
                        'std_dev': 2.0
                    }
                },
                'momentum': {
                    'enabled': True,
                    'weight': 0.3,
                    'parameters': {
                        'rsi_period': 14,
                        'macd_fast': 12,
                        'macd_slow': 26,
                        'macd_signal': 9
                    }
                }
            },
            'logging': {
                'level': 'INFO',
                'file_path': 'logs/trading_bot.log',
                'max_file_size': 10 * 1024 * 1024,
                'backup_count': 5
            }
        }
        
        # If no config path provided, try environment variable
        if config_path is None:
            config_path = os.getenv('TRADING_BOT_CONFIG')
        
        # If still no config, return default
        if not config_path:
            logger.warning("No configuration file specified. Using default configuration.")
            return default_config
        
        try:
            # Load from file
            if not os.path.exists(config_path):
                logger.error(f"Configuration file not found: {config_path}")
                return default_config
            
            # Determine file type
            _, ext = os.path.splitext(config_path)
            
            with open(config_path, 'r') as config_file:
                if ext.lower() in ['.yaml', '.yml']:
                    user_config = yaml.safe_load(config_file)
                elif ext.lower() == '.json':
                    user_config = json.load(config_file)
                else:
                    logger.error(f"Unsupported configuration file type: {ext}")
                    return default_config
            
            # Merge user config with default config
            merged_config = ConfigManager._deep_merge(default_config, user_config)
            
            logger.info(f"Configuration loaded successfully from {config_path}")
            return merged_config
        
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return default_config
    
    @staticmethod
    def _deep_merge(base: Dict, update: Dict) -> Dict:
        """
        Recursively merge two dictionaries
        
        Args:
            base (Dict): Default configuration
            update (Dict): User-provided configuration
        
        Returns:
            Dict: Merged configuration
        """
        for key, value in update.items():
            if isinstance(value, dict):
                # Recursively merge nested dictionaries
                base[key] = ConfigManager._deep_merge(base.get(key, {}), value)
            else:
                base[key] = value
        return base
    
    @staticmethod
    def validate_config(config: Dict) -> bool:
        """
        Validate configuration for required fields
        
        Args:
            config (Dict): Configuration to validate
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        required_keys = [
            'account.login', 
            'account.password', 
            'account.server',
            'trading.max_volume',
            'trading.min_volume',
            'trading.risk_per_trade',
            'risk_management.max_daily_loss',
            'risk_management.max_drawdown',
            'risk_management.volatility_threshold',
            'risk_management.correlation_threshold',
            'risk_management.max_position_size',
            'risk_management.min_margin_level',
            'risk_management.max_open_positions',
            'technical_indicators.rsi.period',
            'technical_indicators.rsi.overbought',
            'technical_indicators.rsi.oversold',
            'technical_indicators.moving_averages.fast',
            'technical_indicators.moving_averages.slow',
            'technical_indicators.atr.period',
            'technical_indicators.bollinger.period',
            'technical_indicators.bollinger.std_dev',
            'technical_indicators.macd.fast_period',
            'technical_indicators.macd.slow_period',
            'technical_indicators.macd.signal_period',
            'analysis.timeframes',
            'analysis.min_data_points',
            'analysis.ml_settings.min_signal_strength',
            'analysis.ml_settings.prediction_threshold',
            'analysis.ml_settings.training_period',
            'analysis.ml_settings.retrain_interval',
            'strategies.trend_following.enabled',
            'strategies.trend_following.weight',
            'strategies.trend_following.parameters.ma_fast',
            'strategies.trend_following.parameters.ma_slow',
            'strategies.trend_following.parameters.atr_period',
            'strategies.mean_reversion.enabled',
            'strategies.mean_reversion.weight',
            'strategies.mean_reversion.parameters.lookback',
            'strategies.mean_reversion.parameters.std_dev',
            'strategies.momentum.enabled',
            'strategies.momentum.weight',
            'strategies.momentum.parameters.rsi_period',
            'strategies.momentum.parameters.macd_fast',
            'strategies.momentum.parameters.macd_slow',
            'strategies.momentum.parameters.macd_signal',
            'logging.level',
            'logging.file_path',
            'logging.max_file_size',
            'logging.backup_count'
        ]
        
        for key in required_keys:
            # Split nested keys
            keys = key.split('.')
            current = config
            
            # Navigate through nested dictionary
            for subkey in keys:
                if not isinstance(current, dict) or subkey not in current:
                    logger.error(f"Missing required configuration key: {key}")
                    return False
                current = current[subkey]
        
        return True

    @staticmethod
    def get_strategy_config(strategy_name: str) -> Dict:
        """Get configuration for a specific strategy"""
        config = ConfigManager.load_config()
        return config['strategies'].get(strategy_name, {})

# Market Configuration
MARKETS = {
    'FOREX': {
        'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
        'timeframes': {
            'M5': 'M5',
            'M15': 'M15',
            'H1': 'H1'
        },
        'pip_value': 0.0001,
        'leverage': 30,  # Default leverage 1:30
        'trading_hours': {
            'start': '09:00',
            'end': '17:00'
        }
    },
    'CRYPTO': {
        'symbols': ['BTCUSD', 'ETHUSD', 'LTCUSD'],
        'timeframes': {
            'M5': 'M5',
            'M15': 'M15',
            'H1': 'H1'
        },
        'pip_value': 0.01,  # Different pip value for crypto
        'leverage': 2,  # Lower leverage for crypto
        'trading_hours': {
            'start': '00:00',  # 24/7 trading
            'end': '23:59'
        }
    }
}

# Trading Parameters
INITIAL_BALANCE = 10.0  # Starting balance in USD
RISK_PERCENTAGE = 2.0   # Risk per trade (%)
TAKE_PROFIT_RATIO = 2.0  # Risk:Reward ratio

# Default Market Settings
ACTIVE_MARKET = 'FOREX'  # Can be 'FOREX' or 'CRYPTO'
SYMBOL = MARKETS[ACTIVE_MARKET]['symbols'][0]  # Default to first symbol in active market
TIMEFRAME = 'M5'        # Default timeframe (5 minutes)

# Account Configuration
ACCOUNT_NUMBER = None  # Replace with your MT5 account number
PASSWORD = None       # Replace with your MT5 password
SERVER = None        # Replace with your MT5 server name

# Strategy Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MA_FAST_PERIOD = 20
MA_SLOW_PERIOD = 50

# Market-Specific Strategy Parameters
MARKET_PARAMS = {
    'FOREX': {
        'min_spread': 2.0,  # Maximum acceptable spread in pips
        'slippage': 2,      # Maximum acceptable slippage in pips
        'min_volume': 0.1,  # Minimum trade volume
        'volume_step': 0.1  # Volume step size
    },
    'CRYPTO': {
        'min_spread': 10.0, # Crypto typically has higher spreads
        'slippage': 5,      # Higher slippage tolerance for crypto
        'min_volume': 0.001,# Minimum trade volume
        'volume_step': 0.001# Volume step size
    }
}

# Risk Management Parameters
MAX_DAILY_TRADES = 5    # Maximum number of trades per day
MAX_DAILY_LOSS = 5.0    # Maximum daily loss as percentage of balance
TRAILING_STOP = True    # Enable trailing stop loss
TRAILING_STOP_START = 1.5  # Start trailing after 1.5x risk reached
