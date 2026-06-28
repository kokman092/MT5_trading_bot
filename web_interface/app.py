import os
import sys

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from datetime import datetime
import json
from modules.licensing.license_manager import LicenseManager, LicenseTier
from modules.payment.payment_processor import PaymentProcessor
from modules.support.support_manager import SupportManager
from modules.payment.mock_stripe import MockStripe
from modules.web.shared import initialize_mt5, get_account_info, get_open_positions, calculate_daily_pl, validate_trading_conditions
from auth import Auth, login_required
import logging
import threading
import asyncio
from modules.config.config_manager import ConfigManager
from modules.trading.market_analyzer import MarketAnalyzer
from modules.trading.ml_analyzer import MLAnalyzer
from modules.trading.risk_manager import RiskManager
from modules.trading.trade_executor import TradeExecutor
from modules.trading.position_manager import PositionManager
from modules.backtesting.backtester import Backtester
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_socketio import SocketIO
from modules.auth.user_manager import UserManager
from modules.support.ticket_manager import TicketManager
from modules.trading.bot_manager import BotManager

class MockMT5:
    """Mock MT5 for testing without MetaTrader5 installed"""
    def __init__(self):
        self.positions = []
        self.bot_running = False
        
    def initialize(self, login=None, server=None, password=None):
        if login and server and password:
            self.bot_running = True
            return True
        return False
        
    def shutdown(self):
        self.bot_running = False
        return True
        
    def account_info(self):
        class AccountInfo:
            def __init__(self):
                self.balance = 10000.0
                self.equity = 10500.0
                self.margin_level = 200.0
                
        return AccountInfo()
        
    def positions_total(self):
        return len(self.positions)
        
    def positions_get(self, symbol=None):
        if symbol:
            return [pos for pos in self.positions if pos.symbol == symbol]
        return self.positions
        
    def symbol_info(self, symbol):
        class SymbolInfo:
            def __init__(self):
                self.trade_mode = 1  # SYMBOL_TRADE_MODE_FULL
                
        return SymbolInfo()
        
    def symbol_info_tick(self, symbol):
        class Tick:
            def __init__(self):
                self.bid = 1.2150
                self.ask = 1.2152
                
        return Tick()

# Use mock MT5
mt5 = MockMT5()

app = Flask(__name__)
app.secret_key = 'dev-secret-key-123'
socketio = SocketIO(app)

# Initialize managers
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

user_manager = UserManager()
license_manager = LicenseManager()
ticket_manager = TicketManager()

# Load configuration
config_path = os.path.join(parent_dir, 'config', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Use mock Stripe in development
stripe = MockStripe()

# Initialize components
auth = Auth(config)
payment_processor = PaymentProcessor(config)
support_manager = SupportManager(config)

# Make config available to templates
app.config.update(
    stripe_public_key=config['stripe']['public_key'],
    domain=config['web_interface']['domain']
)

# Initialize bot manager
bot_manager = BotManager.get_instance()

@login_manager.user_loader
def load_user(user_id):
    return user_manager.get_user(user_id)

@app.route('/')
def home():
    """Home page with pricing plans"""
    return render_template('home.html', pricing=config['pricing'])

@app.route('/purchase/<tier>')
def purchase(tier):
    """Purchase page for a specific tier"""
    try:
        if tier not in config['pricing']:
            raise ValueError("Invalid tier")
            
        plan_info = config['pricing'][tier]
        plan_info['tier'] = tier
        
        return render_template(
            'purchase.html',
            plan=plan_info,
            stripe_public_key=app.config['stripe_public_key']
        )
    except ValueError:
        flash('Invalid subscription tier')
        return redirect(url_for('home'))

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    """Create checkout session"""
    try:
        email = request.form.get('email')
        tier = request.form.get('tier', 'basic')
        billing_cycle = request.form.get('billing_cycle', 'monthly')
        
        # Create checkout session with mock Stripe
        checkout_session = stripe.create_checkout_session(
            customer_email=email,
            metadata={
                'tier': tier,
                'billing_cycle': billing_cycle
            }
        )
        
        return jsonify({'sessionId': checkout_session.id})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/license/validate', methods=['POST'])
def validate_license():
    """Validate a license key"""
    try:
        license_key = request.form['license_key']
        machine_id = request.form.get('machine_id')
        
        result = license_manager.validate_license(license_key, machine_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/support/ticket', methods=['GET', 'POST'])
async def support_ticket():
    """Create or view support ticket"""
    if request.method == 'POST':
        try:
            user_id = request.form['email']
            subject = request.form['subject']
            description = request.form['description']
            license_key = request.form['license_key']
            
            # Validate license
            license_info = license_manager.validate_license(license_key)
            if not license_info['valid']:
                flash('Invalid license key')
                return redirect(url_for('support_ticket'))
                
            # Create ticket
            ticket_id = await support_manager.create_ticket(
                user_id,
                subject,
                description,
                LicenseTier(license_info['tier'])
            )
            
            if ticket_id:
                flash(f'Ticket created successfully: {ticket_id}')
                return redirect(url_for('view_ticket', ticket_id=ticket_id))
            else:
                flash('Failed to create ticket')
                
        except Exception as e:
            flash(f'Error: {str(e)}')
            
    return render_template('support_ticket.html')

@app.route('/support/ticket/<ticket_id>')
def view_ticket(ticket_id):
    """View a specific support ticket"""
    ticket = support_manager.get_ticket(ticket_id)
    if ticket:
        return render_template('view_ticket.html', ticket=ticket)
    flash('Ticket not found')
    return redirect(url_for('support_ticket'))

@app.route('/docs')
def documentation():
    """View documentation"""
    return render_template('documentation.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = user_manager.authenticate(email, password)
        if user:
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')
        
        if user_manager.register(email, password, name):
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        else:
            flash('Registration failed. Email may already be in use.')
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Render user dashboard"""
    account_info = get_account_info()
    positions = get_open_positions()
    daily_pl = calculate_daily_pl()
    
    return render_template('dashboard.html',
                         account_info=account_info,
                         positions=positions,
                         daily_pl=daily_pl)

@app.route('/api/account')
@login_required
def get_account_data():
    """Get current account information"""
    try:
        account_info = get_account_info()
        positions = get_open_positions()
        daily_pl = calculate_daily_pl()
        
        return jsonify({
            'account_info': account_info,
            'positions': positions,
            'daily_pl': daily_pl,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/trading/start', methods=['POST'])
@login_required
async def start_trading():
    """Start automated trading"""
    try:
        if not license_manager.validate_license(current_user.id):
            return jsonify({
                'success': False,
                'error': 'Invalid or expired license'
            })
        
        if not validate_trading_conditions():
            return jsonify({
                'success': False,
                'error': 'Trading conditions not met'
            })
            
        # Check if bot is already running
        if bot_manager.is_running:
            return jsonify({
                'success': False,
                'error': 'Trading bot is already running'
            })
        
        # Start the bot
        success = await bot_manager.start_bot(config)
        if success:
            return jsonify({
                'success': True,
                'message': 'Trading started successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to start trading bot'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/trading/stop', methods=['POST'])
@login_required
async def stop_trading():
    """Stop automated trading"""
    try:
        # Check if bot is running
        if not bot_manager.is_running:
            return jsonify({
                'success': False,
                'error': 'Trading bot is not running'
            })
        
        # Stop the bot
        success = await bot_manager.stop_bot()
        if success:
            return jsonify({
                'success': True,
                'message': 'Trading stopped successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to stop trading bot'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/trading/status')
@login_required
def get_trading_status():
    """Get current trading bot status"""
    try:
        status = bot_manager.get_bot_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/backtest', methods=['POST'])
@login_required
def run_backtest():
    """Run trading strategy backtest"""
    try:
        if not license_manager.validate_license(current_user.id):
            return jsonify({
                'success': False,
                'error': 'Invalid or expired license'
            })
        
        data = request.get_json()
        backtester = Backtester(app.config)
        result = backtester.run_backtest(data)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/license/purchase', methods=['POST'])
@login_required
def purchase_license():
    """Handle license purchase"""
    try:
        plan_id = request.form.get('plan_id')
        payment_token = request.form.get('payment_token')
        
        success = license_manager.process_purchase(
            user_id=current_user.id,
            plan_id=plan_id,
            payment_token=payment_token
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'License purchased successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'License purchase failed'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/license/status')
@login_required
def get_license_status():
    """Get current license status"""
    try:
        status = license_manager.get_license_status(current_user.id)
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/support/ticket', methods=['POST'])
@login_required
def create_ticket():
    """Create support ticket"""
    try:
        subject = request.form.get('subject')
        description = request.form.get('description')
        priority = request.form.get('priority', 'medium')
        
        ticket = ticket_manager.create_ticket(
            user_id=current_user.id,
            subject=subject,
            description=description,
            priority=priority
        )
        
        return jsonify({
            'success': True,
            'ticket_id': ticket.id
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/support/tickets')
@login_required
def get_tickets():
    """Get user's support tickets"""
    try:
        tickets = ticket_manager.get_user_tickets(current_user.id)
        return jsonify({
            'tickets': [ticket.to_dict() for ticket in tickets]
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/support/ticket/<ticket_id>')
@login_required
def get_ticket(ticket_id):
    """Get specific support ticket"""
    try:
        ticket = ticket_manager.get_ticket(ticket_id)
        if ticket and ticket.user_id == current_user.id:
            return jsonify(ticket.to_dict())
        else:
            return jsonify({'error': 'Ticket not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/support/ticket/<ticket_id>/reply', methods=['POST'])
@login_required
def reply_ticket(ticket_id):
    """Reply to support ticket"""
    try:
        message = request.form.get('message')
        
        success = ticket_manager.add_reply(
            ticket_id=ticket_id,
            user_id=current_user.id,
            message=message
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Reply added successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to add reply'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

def start_web_interface():
    """Start the web interface application"""
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)

if __name__ == '__main__':
    start_web_interface()
