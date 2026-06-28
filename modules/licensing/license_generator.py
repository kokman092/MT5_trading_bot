import jwt
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import hashlib
import json
from .license_manager import LicenseTier

class LicenseGenerator:
    def __init__(self, config: Dict):
        """Initialize license generator"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.secret = config['licensing']['LICENSE_SECRET']
        
    def generate_license(
        self,
        email: str,
        tier: LicenseTier,
        duration_days: int,
        machine_id: Optional[str] = None
    ) -> Optional[str]:
        """Generate a new license key"""
        try:
            # Create unique customer ID
            customer_id = self._generate_customer_id(email)
            
            # Create payload
            payload = {
                'customer_id': customer_id,
                'email': email,
                'tier': tier.value,
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(days=duration_days)
            }
            
            # Add machine ID if provided (for machine-locked licenses)
            if machine_id:
                payload['machine_id'] = machine_id
                
            # Generate JWT token
            license_key = jwt.encode(
                payload,
                self.secret,
                algorithm='HS256'
            )
            
            # Store license in database
            self._store_license(license_key, payload)
            
            return license_key
            
        except Exception as e:
            self.logger.error(f"Error generating license: {str(e)}")
            return None
            
    def validate_license(self, license_key: str, machine_id: Optional[str] = None) -> Dict:
        """Validate a license key"""
        try:
            # Decode JWT token
            payload = jwt.decode(
                license_key,
                self.secret,
                algorithms=['HS256']
            )
            
            # Check expiration
            if datetime.fromtimestamp(payload['exp']) < datetime.utcnow():
                return {
                    'valid': False,
                    'reason': 'License expired',
                    'expires': datetime.fromtimestamp(payload['exp']).isoformat()
                }
                
            # Check machine ID if present
            if 'machine_id' in payload and machine_id:
                if payload['machine_id'] != machine_id:
                    return {
                        'valid': False,
                        'reason': 'Invalid machine ID'
                    }
                    
            # Check if license is revoked
            if self._is_license_revoked(license_key):
                return {
                    'valid': False,
                    'reason': 'License revoked'
                }
                
            return {
                'valid': True,
                'tier': payload['tier'],
                'email': payload['email'],
                'expires': datetime.fromtimestamp(payload['exp']).isoformat()
            }
            
        except jwt.InvalidTokenError:
            return {
                'valid': False,
                'reason': 'Invalid license key'
            }
        except Exception as e:
            self.logger.error(f"Error validating license: {str(e)}")
            return {
                'valid': False,
                'reason': 'Validation error'
            }
            
    def revoke_license(self, license_key: str) -> bool:
        """Revoke a license key"""
        try:
            # Decode license key
            payload = jwt.decode(
                license_key,
                self.secret,
                algorithms=['HS256']
            )
            
            # Add to revoked licenses
            self._revoke_license(license_key, payload)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error revoking license: {str(e)}")
            return False
            
    def _generate_customer_id(self, email: str) -> str:
        """Generate unique customer ID"""
        hash_input = f"{email}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
    def _store_license(self, license_key: str, payload: Dict):
        """Store license in database"""
        try:
            # In a real implementation, this would store the license in a database
            # For now, we'll store it in a JSON file
            import os
            
            licenses_file = os.path.join(
                os.path.dirname(__file__),
                'data',
                'licenses.json'
            )
            
            os.makedirs(os.path.dirname(licenses_file), exist_ok=True)
            
            licenses = {}
            if os.path.exists(licenses_file):
                with open(licenses_file, 'r') as f:
                    licenses = json.load(f)
                    
            licenses[payload['customer_id']] = {
                'license_key': license_key,
                'payload': {
                    k: str(v) if isinstance(v, datetime) else v
                    for k, v in payload.items()
                },
                'created_at': datetime.utcnow().isoformat(),
                'revoked': False
            }
            
            with open(licenses_file, 'w') as f:
                json.dump(licenses, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error storing license: {str(e)}")
            
    def _is_license_revoked(self, license_key: str) -> bool:
        """Check if license is revoked"""
        try:
            # In a real implementation, this would check a database
            licenses_file = os.path.join(
                os.path.dirname(__file__),
                'data',
                'licenses.json'
            )
            
            if not os.path.exists(licenses_file):
                return False
                
            with open(licenses_file, 'r') as f:
                licenses = json.load(f)
                
            # Find license by key
            for license_data in licenses.values():
                if license_data['license_key'] == license_key:
                    return license_data.get('revoked', False)
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking revoked license: {str(e)}")
            return False
            
    def _revoke_license(self, license_key: str, payload: Dict):
        """Mark license as revoked"""
        try:
            licenses_file = os.path.join(
                os.path.dirname(__file__),
                'data',
                'licenses.json'
            )
            
            if not os.path.exists(licenses_file):
                return
                
            with open(licenses_file, 'r') as f:
                licenses = json.load(f)
                
            # Find and revoke license
            customer_id = payload['customer_id']
            if customer_id in licenses:
                licenses[customer_id]['revoked'] = True
                licenses[customer_id]['revoked_at'] = datetime.utcnow().isoformat()
                
            with open(licenses_file, 'w') as f:
                json.dump(licenses, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error revoking license: {str(e)}")
            
    def generate_trial_license(self, email: str, duration_days: int = 30) -> Optional[str]:
        """Generate a trial license"""
        return self.generate_license(
            email,
            LicenseTier.BASIC,
            duration_days,
            machine_id=None  # Trial licenses are not machine-locked
        )
