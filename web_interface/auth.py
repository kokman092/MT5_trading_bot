from functools import wraps
from flask import session, redirect, url_for, flash
import jwt
from datetime import datetime, timedelta
import hashlib
import os
import json

class Auth:
    def __init__(self, config):
        self.config = config
        self.users_file = os.path.join(
            os.path.dirname(__file__),
            'data',
            'users.json'
        )
        self._ensure_users_file()
        
    def _ensure_users_file(self):
        """Ensure users file exists"""
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
                
    def _hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
        
    def _generate_token(self, user_id):
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(days=1)
        }
        return jwt.encode(payload, self.config['web_interface']['secret_key'], algorithm='HS256')
        
    def register(self, email, password, name):
        """Register a new user"""
        with open(self.users_file, 'r') as f:
            users = json.load(f)
            
        if email in users:
            return False, "Email already registered"
            
        users[email] = {
            'password': self._hash_password(password),
            'name': name,
            'created_at': datetime.utcnow().isoformat(),
            'subscription': None,
            'license_key': None
        }
        
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=4)
            
        return True, "Registration successful"
        
    def login(self, email, password):
        """Login user"""
        with open(self.users_file, 'r') as f:
            users = json.load(f)
            
        user = users.get(email)
        if not user or user['password'] != self._hash_password(password):
            return False, "Invalid email or password"
            
        token = self._generate_token(email)
        return True, token
        
    def get_user(self, email):
        """Get user details"""
        with open(self.users_file, 'r') as f:
            users = json.load(f)
            return users.get(email)
            
    def update_user(self, email, updates):
        """Update user details"""
        with open(self.users_file, 'r') as f:
            users = json.load(f)
            
        if email not in users:
            return False
            
        users[email].update(updates)
        
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=4)
            
        return True
        
    def verify_token(self, token):
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.config['web_interface']['secret_key'], algorithms=['HS256'])
            return True, payload['user_id']
        except jwt.ExpiredSignatureError:
            return False, "Token expired"
        except jwt.InvalidTokenError:
            return False, "Invalid token"
            
def login_required(f):
    """Decorator for routes that require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = session.get('token')
        if not token:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
            
        from flask import current_app
        auth = Auth(current_app.config)
        valid, user_id = auth.verify_token(token)
        
        if not valid:
            flash('Session expired. Please log in again', 'warning')
            return redirect(url_for('login'))
            
        return f(*args, **kwargs)
    return decorated_function
