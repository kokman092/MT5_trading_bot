import stripe
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import json
from ..licensing.license_manager import LicenseTier
from dataclasses import dataclass

@dataclass
class PricingPlan:
    tier: LicenseTier
    monthly_price: float
    annual_price: float
    description: str
    features: list

class PaymentProcessor:
    def __init__(self, config: Dict):
        """Initialize payment processor"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        stripe.api_key = config['stripe']['secret_key']
        self.pricing_plans = self._init_pricing_plans()
        
    def _init_pricing_plans(self) -> Dict[LicenseTier, PricingPlan]:
        """Initialize pricing plans"""
        return {
            LicenseTier.BASIC: PricingPlan(
                tier=LicenseTier.BASIC,
                monthly_price=0,
                annual_price=0,
                description="Basic paper trading features",
                features=[
                    "Paper Trading",
                    "Basic Technical Indicators",
                    "Up to 3 Symbols",
                    "Email Support"
                ]
            ),
            LicenseTier.PRO: PricingPlan(
                tier=LicenseTier.PRO,
                monthly_price=49.99,
                annual_price=499.99,
                description="Professional trading features",
                features=[
                    "Live Trading",
                    "Advanced Analytics",
                    "Up to 10 Symbols",
                    "Email Alerts",
                    "Custom Strategies",
                    "24h Support"
                ]
            ),
            LicenseTier.ENTERPRISE: PricingPlan(
                tier=LicenseTier.ENTERPRISE,
                monthly_price=199.99,
                annual_price=1999.99,
                description="Enterprise-grade trading solution",
                features=[
                    "Priority Support",
                    "API Access",
                    "Custom Integrations",
                    "Up to 30 Symbols",
                    "24/7 Support",
                    "Training Sessions"
                ]
            )
        }
        
    async def create_checkout_session(
        self,
        tier: LicenseTier,
        billing_cycle: str,
        user_email: str
    ) -> Optional[str]:
        """Create Stripe checkout session"""
        try:
            plan = self.pricing_plans[tier]
            price = plan.annual_price if billing_cycle == 'annual' else plan.monthly_price
            
            # Create Stripe session
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': f'Trading Bot {tier.value.title()} Plan',
                            'description': plan.description
                        },
                        'unit_amount': int(price * 100),  # Convert to cents
                    },
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=f'{self.config["website"]["domain"]}/payment/success?session_id={{CHECKOUT_SESSION_ID}}',
                cancel_url=f'{self.config["website"]["domain"]}/payment/cancel',
                customer_email=user_email,
                metadata={
                    'tier': tier.value,
                    'billing_cycle': billing_cycle
                }
            )
            
            return session.id
            
        except Exception as e:
            self.logger.error(f"Error creating checkout session: {str(e)}")
            return None
            
    async def process_webhook(self, event_data: Dict) -> bool:
        """Process Stripe webhook events"""
        try:
            event_type = event_data['type']
            
            if event_type == 'checkout.session.completed':
                return await self._handle_successful_payment(event_data['data']['object'])
            elif event_type == 'customer.subscription.deleted':
                return await self._handle_subscription_cancelled(event_data['data']['object'])
            elif event_type == 'invoice.payment_failed':
                return await self._handle_payment_failed(event_data['data']['object'])
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing webhook: {str(e)}")
            return False
            
    async def _handle_successful_payment(self, session: Dict) -> bool:
        """Handle successful payment"""
        try:
            # Get customer details
            customer_email = session['customer_email']
            tier = session['metadata']['tier']
            billing_cycle = session['metadata']['billing_cycle']
            
            # Calculate license duration
            duration = 365 if billing_cycle == 'annual' else 30
            
            # Generate license key
            from ..licensing.license_generator import LicenseGenerator
            license_generator = LicenseGenerator(self.config)
            license_key = license_generator.generate_license(
                customer_email,
                LicenseTier(tier),
                duration
            )
            
            if not license_key:
                raise Exception("Failed to generate license key")
                
            # Send welcome email with license key
            await self._send_welcome_email(
                customer_email,
                tier,
                license_key,
                billing_cycle
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling successful payment: {str(e)}")
            return False
            
    async def _handle_subscription_cancelled(self, subscription: Dict) -> bool:
        """Handle subscription cancellation"""
        try:
            customer_email = subscription['customer_email']
            
            # Revoke license
            from ..licensing.license_manager import LicenseManager
            license_manager = LicenseManager(self.config)
            if not license_manager.revoke_license(customer_email):
                raise Exception("Failed to revoke license")
                
            # Send cancellation email
            await self._send_cancellation_email(customer_email)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling subscription cancellation: {str(e)}")
            return False
            
    async def _handle_payment_failed(self, invoice: Dict) -> bool:
        """Handle failed payment"""
        try:
            customer_email = invoice['customer_email']
            
            # Send payment failed email
            await self._send_payment_failed_email(
                customer_email,
                invoice['hosted_invoice_url']
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling payment failure: {str(e)}")
            return False
            
    async def _send_welcome_email(
        self,
        email: str,
        tier: str,
        license_key: str,
        billing_cycle: str
    ):
        """Send welcome email with license key"""
        try:
            from ..utils.email_sender import EmailSender
            email_sender = EmailSender(self.config)
            
            subject = f"Welcome to Trading Bot {tier.title()} Plan"
            body = f"""
            Thank you for choosing our Trading Bot!
            
            Your subscription details:
            - Plan: {tier.title()}
            - Billing Cycle: {billing_cycle}
            - License Key: {license_key}
            
            Getting Started:
            1. Install the trading bot (see documentation)
            2. Configure your license key
            3. Start trading!
            
            Need help? Contact our support team.
            """
            
            await email_sender.send_email(email, subject, body)
            
        except Exception as e:
            self.logger.error(f"Error sending welcome email: {str(e)}")
            
    async def _send_cancellation_email(self, email: str):
        """Send subscription cancellation email"""
        try:
            from ..utils.email_sender import EmailSender
            email_sender = EmailSender(self.config)
            
            subject = "Trading Bot Subscription Cancelled"
            body = """
            We're sorry to see you go!
            
            Your subscription has been cancelled and your license will be deactivated at the end of the current billing period.
            
            If you change your mind, you can resubscribe anytime.
            
            Thank you for trying our Trading Bot!
            """
            
            await email_sender.send_email(email, subject, body)
            
        except Exception as e:
            self.logger.error(f"Error sending cancellation email: {str(e)}")
            
    async def _send_payment_failed_email(self, email: str, invoice_url: str):
        """Send payment failed email"""
        try:
            from ..utils.email_sender import EmailSender
            email_sender = EmailSender(self.config)
            
            subject = "Trading Bot Payment Failed"
            body = f"""
            We were unable to process your payment for the Trading Bot subscription.
            
            Please update your payment information here:
            {invoice_url}
            
            Your subscription will be paused if payment is not received within 3 days.
            
            Need help? Contact our support team.
            """
            
            await email_sender.send_email(email, subject, body)
            
        except Exception as e:
            self.logger.error(f"Error sending payment failed email: {str(e)}")
            
    def get_pricing_info(self) -> Dict:
        """Get pricing information for all tiers"""
        return {
            tier.value: {
                'monthly_price': plan.monthly_price,
                'annual_price': plan.annual_price,
                'description': plan.description,
                'features': plan.features
            }
            for tier, plan in self.pricing_plans.items()
        }
