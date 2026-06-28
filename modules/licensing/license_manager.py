import hashlib
import jwt
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import json
import os

class LicenseTier(Enum):
    BASIC = "basic"      # Paper trading only
    PRO = "pro"         # Live trading with basic features
    ENTERPRISE = "enterprise"  # All features + priority support

@dataclass
class LicenseFeatures:
    paper_trading: bool = True
    live_trading: bool = False
    backtesting: bool = True
    max_symbols: int = 3
    priority_support: bool = False
    custom_strategies: bool = False
    advanced_analytics: bool = False
    email_alerts: bool = False
    api_access: bool = False
    max_concurrent_trades: int = 1

class LicenseManager:
    def __init__(self, config: Dict):
        """Initialize license manager"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._load_license_key()
        self.features = self._get_default_features()
        
    def _load_license_key(self):
        """Load license key from config"""
        try:
            license_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'license.key')
            if os.path.exists(license_file):
                with open(license_file, 'r') as f:
                    self.license_key = f.read().strip()
            else:
                self.license_key = None
                self.logger.warning("No license key found")
        except Exception as e:
            self.logger.error(f"Error loading license: {str(e)}")
            self.license_key = None
            
    def _get_default_features(self) -> LicenseFeatures:
        """Get default features (Basic tier)"""
        return LicenseFeatures()
        
    def _get_pro_features(self) -> LicenseFeatures:
        """Get Pro tier features"""
        return LicenseFeatures(
            paper_trading=True,
            live_trading=True,
            backtesting=True,
            max_symbols=10,
            priority_support=False,
            custom_strategies=True,
            advanced_analytics=True,
            email_alerts=True,
            api_access=False,
            max_concurrent_trades=5
        )
        
    def _get_enterprise_features(self) -> LicenseFeatures:
        """Get Enterprise tier features"""
        return LicenseFeatures(
            paper_trading=True,
            live_trading=True,
            backtesting=True,
            max_symbols=30,
            priority_support=True,
            custom_strategies=True,
            advanced_analytics=True,
            email_alerts=True,
            api_access=True,
            max_concurrent_trades=15
        )
        
    def validate_license(self) -> bool:
        """Validate license key with server"""
        if not self.license_key:
            return False
            
        try:
            # In production, this would make an API call to your license server
            # For now, we'll use a simple JWT validation
            try:
                payload = jwt.decode(self.license_key, self.config['LICENSE_SECRET'], algorithms=['HS256'])
                if payload['exp'] < datetime.utcnow().timestamp():
                    self.logger.error("License expired")
                    return False
                    
                # Update features based on license tier
                tier = LicenseTier(payload['tier'])
                if tier == LicenseTier.PRO:
                    self.features = self._get_pro_features()
                elif tier == LicenseTier.ENTERPRISE:
                    self.features = self._get_enterprise_features()
                else:
                    self.features = self._get_default_features()
                    
                return True
                
            except jwt.InvalidTokenError:
                self.logger.error("Invalid license key")
                return False
                
        except Exception as e:
            self.logger.error(f"License validation error: {str(e)}")
            return False
            
    def check_feature_access(self, feature: str) -> bool:
        """Check if current license has access to a feature"""
        return getattr(self.features, feature, False)
        
    def generate_machine_id(self) -> str:
        """Generate unique machine ID for license binding"""
        try:
            # Get system-specific information
            import platform
            system_info = {
                'processor': platform.processor(),
                'machine': platform.machine(),
                'node': platform.node()
            }
            
            # Create a unique hash
            machine_id = hashlib.sha256(
                json.dumps(system_info, sort_keys=True).encode()
            ).hexdigest()
            
            return machine_id
            
        except Exception as e:
            self.logger.error(f"Error generating machine ID: {str(e)}")
            return ""
            
    def generate_trial_license(self) -> Optional[str]:
        """Generate a 30-day trial license"""
        try:
            machine_id = self.generate_machine_id()
            payload = {
                'machine_id': machine_id,
                'tier': LicenseTier.BASIC.value,
                'exp': (datetime.utcnow() + timedelta(days=30)).timestamp(),
                'trial': True
            }
            
            trial_key = jwt.encode(payload, self.config['LICENSE_SECRET'], algorithm='HS256')
            return trial_key
            
        except Exception as e:
            self.logger.error(f"Error generating trial license: {str(e)}")
            return None
            
    def get_license_info(self) -> Dict:
        """Get current license information"""
        if not self.license_key:
            return {
                'status': 'inactive',
                'tier': LicenseTier.BASIC.value,
                'features': vars(self._get_default_features())
            }
            
        try:
            payload = jwt.decode(self.license_key, self.config['LICENSE_SECRET'], algorithms=['HS256'])
            return {
                'status': 'active',
                'tier': payload['tier'],
                'expiration': datetime.fromtimestamp(payload['exp']).isoformat(),
                'features': vars(self.features),
                'trial': payload.get('trial', False)
            }
            
        except jwt.InvalidTokenError:
            return {
                'status': 'invalid',
                'tier': LicenseTier.BASIC.value,
                'features': vars(self._get_default_features())
            }
