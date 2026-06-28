from typing import Dict, Optional
import logging
from .license_manager import LicenseManager, LicenseTier
from functools import wraps

def requires_license(feature: str):
    """Decorator to check feature access"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.license_manager.check_feature_access(feature):
                raise FeatureNotAvailableError(
                    f"This feature requires a higher license tier. "
                    f"Please upgrade your license to access {feature}."
                )
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

class FeatureNotAvailableError(Exception):
    """Raised when a feature is not available in current license tier"""
    pass

class FeatureManager:
    def __init__(self, config: Dict):
        """Initialize feature manager"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.license_manager = LicenseManager(config)
        
    def get_available_features(self) -> Dict:
        """Get all available features for current license"""
        return {
            'trading': self._get_trading_features(),
            'analysis': self._get_analysis_features(),
            'support': self._get_support_features(),
            'automation': self._get_automation_features()
        }
        
    def _get_trading_features(self) -> Dict:
        """Get available trading features"""
        features = {
            'paper_trading': True,  # Available in all tiers
            'live_trading': self.license_manager.check_feature_access('live_trading'),
            'max_symbols': self.license_manager.features.max_symbols,
            'max_concurrent_trades': self.license_manager.features.max_concurrent_trades,
            'custom_strategies': self.license_manager.check_feature_access('custom_strategies')
        }
        return features
        
    def _get_analysis_features(self) -> Dict:
        """Get available analysis features"""
        features = {
            'basic_indicators': True,  # Available in all tiers
            'advanced_analytics': self.license_manager.check_feature_access('advanced_analytics'),
            'backtesting': self.license_manager.check_feature_access('backtesting'),
            'performance_reports': True,  # Available in all tiers
            'market_analysis': self.license_manager.check_feature_access('advanced_analytics')
        }
        return features
        
    def _get_support_features(self) -> Dict:
        """Get available support features"""
        features = {
            'email_support': True,  # Available in all tiers
            'priority_support': self.license_manager.check_feature_access('priority_support'),
            'custom_solutions': self.license_manager.check_feature_access('priority_support'),
            'training_sessions': self.license_manager.check_feature_access('priority_support')
        }
        return features
        
    def _get_automation_features(self) -> Dict:
        """Get available automation features"""
        features = {
            'automated_trading': self.license_manager.check_feature_access('live_trading'),
            'email_alerts': self.license_manager.check_feature_access('email_alerts'),
            'api_access': self.license_manager.check_feature_access('api_access'),
            'custom_integrations': self.license_manager.check_feature_access('api_access')
        }
        return features
        
    @requires_license('live_trading')
    def execute_live_trade(self, trade_params: Dict) -> bool:
        """Execute a live trade"""
        # Implementation for live trading
        return True
        
    @requires_license('advanced_analytics')
    def run_advanced_analysis(self, analysis_params: Dict) -> Dict:
        """Run advanced market analysis"""
        # Implementation for advanced analysis
        return {}
        
    @requires_license('api_access')
    def get_api_credentials(self) -> Optional[Dict]:
        """Get API credentials for external integration"""
        # Implementation for API access
        return None
        
    def validate_feature_access(self, feature_name: str) -> bool:
        """Validate access to a specific feature"""
        try:
            # Check if feature exists in any category
            all_features = self.get_available_features()
            for category in all_features.values():
                if feature_name in category:
                    return category[feature_name]
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating feature access: {str(e)}")
            return False
            
    def get_upgrade_recommendations(self) -> Dict:
        """Get recommendations for license upgrade"""
        current_tier = self.license_manager.get_license_info()['tier']
        
        if current_tier == LicenseTier.BASIC.value:
            return {
                'recommended_tier': LicenseTier.PRO.value,
                'benefits': [
                    'Live Trading Access',
                    'Advanced Analytics',
                    'Email Alerts',
                    'Custom Strategies',
                    'Up to 10 Trading Symbols',
                    'Up to 5 Concurrent Trades'
                ],
                'upgrade_link': 'https://your-domain.com/upgrade/pro'
            }
        elif current_tier == LicenseTier.PRO.value:
            return {
                'recommended_tier': LicenseTier.ENTERPRISE.value,
                'benefits': [
                    'Priority Support',
                    'API Access',
                    'Custom Integrations',
                    'Up to 30 Trading Symbols',
                    'Up to 15 Concurrent Trades',
                    'Training Sessions'
                ],
                'upgrade_link': 'https://your-domain.com/upgrade/enterprise'
            }
        return {}
