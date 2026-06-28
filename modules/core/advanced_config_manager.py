import logging
from typing import Dict, Any, Optional
import json
import os
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradingConfig:
    # Account settings
    account: Dict[str, Any]
    
    # Trading parameters
    symbols: list
    timeframes: list
    trading_sessions: Dict[str, Dict]
    
    # Risk management
    risk_management: Dict[str, Any]
    
    # Execution settings
    execution: Dict[str, Any]
    
    # Technical indicators
    indicators: Dict[str, Dict]
    
    # ML/AI settings
    ml_settings: Dict[str, Any]
    
    # Performance monitoring
    monitoring: Dict[str, Any]

class AdvancedConfigManager:
    def __init__(self, config_path: str):
        """Initialize configuration manager"""
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.load_config()
        
    def load_config(self) -> None:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
                
            # Validate and process configuration
            self.config = self._process_config(config_data)
            
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
            
    def _process_config(self, config_data: Dict) -> TradingConfig:
        """Process and validate configuration data"""
        try:
            # Validate required sections
            required_sections = [
                'account', 'symbols', 'timeframes', 'trading_sessions',
                'risk_management', 'execution', 'indicators', 'ml_settings',
                'monitoring'
            ]
            
            for section in required_sections:
                if section not in config_data:
                    raise ValueError(f"Missing required configuration section: {section}")
                    
            # Process account settings
            account = self._validate_account_settings(config_data['account'])
            
            # Process trading parameters
            symbols = self._validate_symbols(config_data['symbols'])
            timeframes = self._validate_timeframes(config_data['timeframes'])
            trading_sessions = self._validate_trading_sessions(config_data['trading_sessions'])
            
            # Process risk management settings
            risk_management = self._validate_risk_settings(config_data['risk_management'])
            
            # Process execution settings
            execution = self._validate_execution_settings(config_data['execution'])
            
            # Process technical indicators
            indicators = self._validate_indicators(config_data['indicators'])
            
            # Process ML settings
            ml_settings = self._validate_ml_settings(config_data['ml_settings'])
            
            # Process monitoring settings
            monitoring = self._validate_monitoring_settings(config_data['monitoring'])
            
            return TradingConfig(
                account=account,
                symbols=symbols,
                timeframes=timeframes,
                trading_sessions=trading_sessions,
                risk_management=risk_management,
                execution=execution,
                indicators=indicators,
                ml_settings=ml_settings,
                monitoring=monitoring
            )
            
        except Exception as e:
            self.logger.error(f"Error processing configuration: {str(e)}")
            raise
            
    def _validate_account_settings(self, account_settings: Dict) -> Dict:
        """Validate account settings"""
        required_fields = ['login', 'password', 'server', 'type']
        
        for field in required_fields:
            if field not in account_settings:
                raise ValueError(f"Missing required account field: {field}")
                
        return account_settings
        
    def _validate_symbols(self, symbols: list) -> list:
        """Validate trading symbols"""
        if not symbols:
            raise ValueError("No trading symbols specified")
            
        # Convert to uppercase
        return [symbol.upper() for symbol in symbols]
        
    def _validate_timeframes(self, timeframes: list) -> list:
        """Validate trading timeframes"""
        valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
        
        for tf in timeframes:
            if tf not in valid_timeframes:
                raise ValueError(f"Invalid timeframe: {tf}")
                
        return timeframes
        
    def _validate_trading_sessions(self, sessions: Dict) -> Dict:
        """Validate trading sessions"""
        required_sessions = ['asian', 'european', 'american']
        
        for session in required_sessions:
            if session not in sessions:
                raise ValueError(f"Missing trading session: {session}")
                
            if 'start' not in sessions[session] or 'end' not in sessions[session]:
                raise ValueError(f"Invalid session times for: {session}")
                
        return sessions
        
    def _validate_risk_settings(self, risk_settings: Dict) -> Dict:
        """Validate risk management settings"""
        required_settings = [
            'max_risk_per_trade',
            'max_daily_loss',
            'max_positions',
            'max_correlation',
            'position_sizing',
            'stop_loss',
            'take_profit'
        ]
        
        for setting in required_settings:
            if setting not in risk_settings:
                raise ValueError(f"Missing risk management setting: {setting}")
                
        # Validate values
        if not 0 < risk_settings['max_risk_per_trade'] <= 0.02:
            raise ValueError("Invalid max risk per trade (should be <= 2%)")
            
        if not 0 < risk_settings['max_daily_loss'] <= 0.05:
            raise ValueError("Invalid max daily loss (should be <= 5%)")
            
        return risk_settings
        
    def _validate_execution_settings(self, execution_settings: Dict) -> Dict:
        """Validate execution settings"""
        required_settings = [
            'max_slippage',
            'retry_attempts',
            'retry_delay',
            'execution_styles',
            'smart_routing',
            'order_types'
        ]
        
        for setting in required_settings:
            if setting not in execution_settings:
                raise ValueError(f"Missing execution setting: {setting}")
                
        return execution_settings
        
    def _validate_indicators(self, indicators: Dict) -> Dict:
        """Validate technical indicators"""
        required_indicators = [
            'moving_averages',
            'oscillators',
            'volatility',
            'volume',
            'momentum'
        ]
        
        for indicator in required_indicators:
            if indicator not in indicators:
                raise ValueError(f"Missing indicator group: {indicator}")
                
        return indicators
        
    def _validate_ml_settings(self, ml_settings: Dict) -> Dict:
        """Validate machine learning settings"""
        required_settings = [
            'models',
            'features',
            'training',
            'prediction',
            'optimization'
        ]
        
        for setting in required_settings:
            if setting not in ml_settings:
                raise ValueError(f"Missing ML setting: {setting}")
                
        return ml_settings
        
    def _validate_monitoring_settings(self, monitoring_settings: Dict) -> Dict:
        """Validate monitoring settings"""
        required_settings = [
            'health_check_interval',
            'state_update_interval',
            'performance_update_interval',
            'logging_level',
            'alerts'
        ]
        
        for setting in required_settings:
            if setting not in monitoring_settings:
                raise ValueError(f"Missing monitoring setting: {setting}")
                
        return monitoring_settings
        
    def get_config(self) -> TradingConfig:
        """Get current configuration"""
        if self.config is None:
            raise ValueError("Configuration not loaded")
        return self.config
        
    def update_config(self, new_config: Dict) -> None:
        """Update configuration"""
        try:
            # Process and validate new configuration
            updated_config = self._process_config(new_config)
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(new_config, f, indent=4)
                
            # Update current config
            self.config = updated_config
            
            self.logger.info("Configuration updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            raise
            
    def get_section(self, section: str) -> Optional[Dict]:
        """Get specific configuration section"""
        if self.config is None:
            raise ValueError("Configuration not loaded")
            
        return getattr(self.config, section, None)
        
    def validate_config(self) -> bool:
        """Validate entire configuration"""
        try:
            if self.config is None:
                return False
                
            # Validate all sections
            self._validate_account_settings(self.config.account)
            self._validate_symbols(self.config.symbols)
            self._validate_timeframes(self.config.timeframes)
            self._validate_trading_sessions(self.config.trading_sessions)
            self._validate_risk_settings(self.config.risk_management)
            self._validate_execution_settings(self.config.execution)
            self._validate_indicators(self.config.indicators)
            self._validate_ml_settings(self.config.ml_settings)
            self._validate_monitoring_settings(self.config.monitoring)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {str(e)}")
            return False
            
    def backup_config(self) -> None:
        """Create backup of current configuration"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config_path}.backup_{timestamp}"
            
            with open(self.config_path, 'r') as src, open(backup_path, 'w') as dst:
                json.dump(json.load(src), dst, indent=4)
                
            self.logger.info(f"Configuration backup created: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating configuration backup: {str(e)}")
            raise 