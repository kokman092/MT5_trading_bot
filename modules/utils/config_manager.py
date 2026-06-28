import os
import json
from cryptography.fernet import Fernet
from typing import Dict, Any
import logging

class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        """Initialize configuration manager"""
        self.logger = logging.getLogger(__name__)
        self.config_dir = config_dir
        self.key_file = os.path.join(config_dir, ".key")
        self.encrypted_config_file = os.path.join(config_dir, "secure_config.enc")
        self._init_encryption()
        
    def _init_encryption(self):
        """Initialize or load encryption key"""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            
        if os.path.exists(self.key_file):
            with open(self.key_file, "rb") as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(self.key)
                
        self.cipher = Fernet(self.key)
        
    def save_secure_config(self, config: Dict[str, Any]):
        """Save configuration securely"""
        try:
            # Extract sensitive data
            sensitive_data = {
                "mt5_account": config.pop("mt5_account", {}),
                "apis": config.pop("apis", {})
            }
            
            # Save non-sensitive config normally
            with open(os.path.join(self.config_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
                
            # Encrypt and save sensitive data
            encrypted_data = self.cipher.encrypt(json.dumps(sensitive_data).encode())
            with open(self.encrypted_config_file, "wb") as f:
                f.write(encrypted_data)
                
            self.logger.info("Configuration saved securely")
            
        except Exception as e:
            self.logger.error(f"Error saving secure config: {str(e)}")
            raise
            
    def load_secure_config(self) -> Dict[str, Any]:
        """Load configuration including decrypted sensitive data"""
        try:
            # Load non-sensitive config
            with open(os.path.join(self.config_dir, "config.json"), "r") as f:
                config = json.load(f)
                
            # Load and decrypt sensitive data
            if os.path.exists(self.encrypted_config_file):
                with open(self.encrypted_config_file, "rb") as f:
                    encrypted_data = f.read()
                decrypted_data = json.loads(self.cipher.decrypt(encrypted_data))
                config.update(decrypted_data)
                
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading secure config: {str(e)}")
            raise
            
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters"""
        required_fields = {
            "mt5_account": ["login", "password", "server"],
            "risk_management": ["risk_per_trade", "max_daily_loss", "max_open_positions"],
            "trading_parameters": ["symbols", "timeframes"],
            "technical_indicators": ["rsi", "moving_averages"]
        }
        
        try:
            for section, fields in required_fields.items():
                if section not in config:
                    self.logger.error(f"Missing required section: {section}")
                    return False
                    
                for field in fields:
                    if field not in config[section]:
                        self.logger.error(f"Missing required field: {section}.{field}")
                        return False
                        
            # Validate specific parameters
            risk_mgmt = config["risk_management"]
            if not (0 < risk_mgmt["risk_per_trade"] <= 10):
                self.logger.error("risk_per_trade must be between 0 and 10 percent")
                return False
                
            if not (0 < risk_mgmt["max_daily_loss"] <= 20):
                self.logger.error("max_daily_loss must be between 0 and 20 percent")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating config: {str(e)}")
            return False
