import json
import yaml
import logging
import os
from typing import Dict, Optional
from pathlib import Path

class ConfigManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = {}

    @staticmethod
    def load_config(file_path: str) -> Dict:
        """Load configuration from JSON or YAML file"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {file_path}")

            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    config = json.load(f)
                elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    raise ValueError("Unsupported configuration file format")
                    
            # Replace environment variables in MT5 account section
            if 'mt5_account' in config:
                config['mt5_account']['login'] = os.getenv('MT5_LOGIN')
                config['mt5_account']['password'] = os.getenv('MT5_PASSWORD')
                config['mt5_account']['server'] = os.getenv('MT5_SERVER')

            return config

        except Exception as e:
            raise Exception(f"Error loading configuration: {str(e)}")

    @staticmethod
    def validate_config(config: Dict) -> bool:
        """Validate trading configuration"""
        try:
            # Required configuration sections
            required_sections = [
                'mt5_account', 'trading', 'risk_management', 
                'market_analysis', 'execution', 'regime_detection'
            ]

            # Check required sections
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required configuration section: {section}")

            # Validate MT5 account configuration
            mt5_config = config['mt5_account']
            if not all(key in mt5_config for key in ['login', 'password', 'server']):
                raise ValueError("Invalid MT5 account configuration")

            # Validate trading configuration
            trading_config = config['trading']
            if not trading_config.get('symbols') or not trading_config.get('timeframes'):
                raise ValueError("Trading symbols and timeframes must be specified")
            
            # Validate position management
            if 'position_management' not in trading_config:
                raise ValueError("Missing position_management section in trading configuration")
            
            position_mgmt = trading_config['position_management']
            if not all(key in position_mgmt for key in ['risk_limits', 'order_types', 'timeframes', 'volume_profile']):
                raise ValueError("Invalid position management configuration")

            # Validate risk management configuration
            risk_config = config['risk_management']
            required_risk_params = [
                'risk_per_trade', 'max_daily_loss', 'max_position_size',
                'position_sizing', 'loss_limits', 'position_limits'
            ]
            if not all(param in risk_config for param in required_risk_params):
                raise ValueError("Invalid risk management configuration")

            # Validate market analysis configuration
            analysis_config = config['market_analysis']
            required_analysis_params = [
                'timeframes', 'technical_indicators', 'validation',
                'regime_thresholds'
            ]
            if not all(param in analysis_config for param in required_analysis_params):
                raise ValueError("Invalid market analysis configuration")

            # Validate execution configuration
            execution_config = config['execution']
            required_execution_params = [
                'max_spread', 'slippage_tolerance', 'retry_attempts',
                'min_volume', 'max_volume'
            ]
            if not all(param in execution_config for param in required_execution_params):
                raise ValueError("Invalid execution configuration")
            
            # Validate regime detection configuration
            regime_config = config['regime_detection']
            required_regime_params = [
                'enabled', 'update_interval', 'min_data_points',
                'methods', 'features', 'thresholds'
            ]
            if not all(param in regime_config for param in required_regime_params):
                raise ValueError("Invalid regime detection configuration")

            return True

        except Exception as e:
            logging.error(f"Configuration validation error: {str(e)}")
            return False

    @staticmethod
    def get_default_config() -> Dict:
        """Get default trading configuration"""
        return {
            'mt5_account': {
                'login': None,
                'password': None,
                'server': None,
                'timeout': 60000
            },
            'trading': {
                'symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
                'timeframes': ['M5', 'M15', 'H1'],
                'sessions': {
                    'london': {'start': "08:00", 'end': "16:00"},
                    'newyork': {'start': "13:00", 'end': "21:00"},
                    'tokyo': {'start': "00:00", 'end': "08:00"}
                }
            },
            'risk_management': {
                'enabled': True,
                'risk_per_trade': 0.02,
                'max_daily_loss': 0.05,
                'max_position_size': 0.5,
                'position_sizing': {
                    'max_size': 0.01,
                    'min_size': 0.001,
                    'size_increment': 0.001
                },
                'loss_limits': {
                    'max_daily_loss': 0.05,
                    'max_drawdown': 0.10,
                    'trailing_drawdown': 0.05
                },
                'position_limits': {
                    'max_positions': 5,
                    'max_positions_per_symbol': 2,
                    'max_correlation': 0.7
                }
            },
            'market_analysis': {
                'timeframes': ['M5', 'M15', 'H1'],
                'technical_indicators': {
                    'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
                    'moving_averages': {'fast': 10, 'slow': 30},
                    'bollinger': {'period': 20, 'std_dev': 2},
                    'atr': {'period': 14}
                },
                'validation': {
                    'min_data_points': 100,
                    'max_gap': 5,
                    'min_volatility': 0.0002
                },
                'regime_thresholds': {
                    'trend_strength': 25,
                    'volatility': 0.002,
                    'momentum': 0.5
                }
            },
            'execution': {
                'max_spread': 0.0003,
                'slippage_tolerance': 0.0001,
                'min_volume': 0.01,
                'max_volume': 0.1,
                'retry_attempts': 3,
                'retry_delay': 1
            }
        }

    def save_config(self, config: Dict, file_path: str) -> bool:
        """Save configuration to file"""
        try:
            with open(file_path, 'w') as f:
                if file_path.endswith('.json'):
                    json.dump(config, f, indent=4)
                elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    raise ValueError("Unsupported configuration file format")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            return False

    def update_config(self, updates: Dict) -> bool:
        """Update configuration with new values"""
        try:
            def deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = deep_update(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d

            self.config = deep_update(self.config, updates)
            return True
        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            return False 