import stripe
import json
import os
from datetime import datetime
from modules.licensing.license_manager import LicenseManager, LicenseTier
from modules.utils.email_sender import EmailSender

class StripeWebhookHandler:
    def __init__(self, config):
        self.config = config
        self.stripe = stripe
        self.stripe.api_key = config['stripe']['secret_key']
        self.webhook_secret = config['stripe']['webhook_secret']
        self.license_manager = LicenseManager(config)
        self.email_sender = EmailSender(config)
        
    def handle_webhook(self, payload, sig_header):
        """Handle Stripe webhook events"""
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )
            
            # Handle different event types
            if event.type == 'checkout.session.completed':
                return self._handle_checkout_completed(event.data.object)
            elif event.type == 'customer.subscription.deleted':
                return self._handle_subscription_deleted(event.data.object)
            elif event.type == 'invoice.payment_failed':
                return self._handle_payment_failed(event.data.object)
            elif event.type == 'customer.subscription.updated':
                return self._handle_subscription_updated(event.data.object)
                
            return {'status': 'success', 'message': f'Unhandled event type: {event.type}'}
            
        except stripe.error.SignatureVerificationError:
            return {'status': 'error', 'message': 'Invalid signature'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def _handle_checkout_completed(self, session):
        """Handle successful checkout"""
        try:
            # Get customer details
            customer = stripe.Customer.retrieve(session.customer)
            subscription = stripe.Subscription.retrieve(session.subscription)
            
            # Get subscription details
            tier = session.metadata.get('tier', 'basic')
            billing_cycle = session.metadata.get('billing_cycle', 'monthly')
            duration = 365 if billing_cycle == 'annual' else 30
            
            # Generate license key
            license_key = self.license_manager.generate_license(
                customer.email,
                LicenseTier(tier),
                duration
            )
            
            if not license_key:
                raise Exception("Failed to generate license key")
                
            # Store subscription info
            self._store_subscription_info(
                customer.email,
                subscription.id,
                tier,
                license_key,
                billing_cycle
            )
            
            # Send welcome email
            self._send_welcome_email(
                customer.email,
                tier,
                license_key,
                billing_cycle
            )
            
            return {
                'status': 'success',
                'message': 'Checkout completed successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def _handle_subscription_deleted(self, subscription):
        """Handle subscription cancellation"""
        try:
            # Get customer details
            customer = stripe.Customer.retrieve(subscription.customer)
            
            # Revoke license
            self.license_manager.revoke_license(customer.email)
            
            # Send cancellation email
            self._send_cancellation_email(customer.email)
            
            return {
                'status': 'success',
                'message': 'Subscription cancelled successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def _handle_payment_failed(self, invoice):
        """Handle failed payment"""
        try:
            # Get customer details
            customer = stripe.Customer.retrieve(invoice.customer)
            
            # Send payment failed email
            self._send_payment_failed_email(
                customer.email,
                invoice.hosted_invoice_url
            )
            
            return {
                'status': 'success',
                'message': 'Payment failure handled'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def _handle_subscription_updated(self, subscription):
        """Handle subscription updates"""
        try:
            # Get customer details
            customer = stripe.Customer.retrieve(subscription.customer)
            
            # Update license if tier changed
            current_tier = self._get_subscription_tier(subscription)
            stored_info = self._get_stored_subscription_info(subscription.id)
            
            if stored_info and stored_info['tier'] != current_tier:
                # Generate new license for updated tier
                duration = 365 if stored_info['billing_cycle'] == 'annual' else 30
                new_license = self.license_manager.generate_license(
                    customer.email,
                    LicenseTier(current_tier),
                    duration
                )
                
                if new_license:
                    # Update stored info
                    stored_info['tier'] = current_tier
                    stored_info['license_key'] = new_license
                    self._update_subscription_info(subscription.id, stored_info)
                    
                    # Send update email
                    self._send_tier_update_email(
                        customer.email,
                        current_tier,
                        new_license
                    )
                    
            return {
                'status': 'success',
                'message': 'Subscription updated successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def _store_subscription_info(self, email, subscription_id, tier, license_key, billing_cycle):
        """Store subscription information"""
        try:
            subscriptions_file = os.path.join(
                os.path.dirname(__file__),
                'data',
                'subscriptions.json'
            )
            
            os.makedirs(os.path.dirname(subscriptions_file), exist_ok=True)
            
            subscriptions = {}
            if os.path.exists(subscriptions_file):
                with open(subscriptions_file, 'r') as f:
                    subscriptions = json.load(f)
                    
            subscriptions[subscription_id] = {
                'email': email,
                'tier': tier,
                'license_key': license_key,
                'billing_cycle': billing_cycle,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            with open(subscriptions_file, 'w') as f:
                json.dump(subscriptions, f, indent=4)
                
        except Exception as e:
            print(f"Error storing subscription info: {str(e)}")
            
    def _get_stored_subscription_info(self, subscription_id):
        """Get stored subscription information"""
        try:
            subscriptions_file = os.path.join(
                os.path.dirname(__file__),
                'data',
                'subscriptions.json'
            )
            
            if not os.path.exists(subscriptions_file):
                return None
                
            with open(subscriptions_file, 'r') as f:
                subscriptions = json.load(f)
                return subscriptions.get(subscription_id)
                
        except Exception as e:
            print(f"Error getting subscription info: {str(e)}")
            return None
            
    def _update_subscription_info(self, subscription_id, info):
        """Update stored subscription information"""
        try:
            subscriptions_file = os.path.join(
                os.path.dirname(__file__),
                'data',
                'subscriptions.json'
            )
            
            if not os.path.exists(subscriptions_file):
                return
                
            with open(subscriptions_file, 'r') as f:
                subscriptions = json.load(f)
                
            if subscription_id in subscriptions:
                info['updated_at'] = datetime.utcnow().isoformat()
                subscriptions[subscription_id] = info
                
                with open(subscriptions_file, 'w') as f:
                    json.dump(subscriptions, f, indent=4)
                    
        except Exception as e:
            print(f"Error updating subscription info: {str(e)}")
            
    def _get_subscription_tier(self, subscription):
        """Get subscription tier from Stripe subscription"""
        try:
            # This would need to be customized based on your Stripe product/price structure
            price_id = subscription.items.data[0].price.id
            
            # Map price IDs to tiers
            price_tier_map = {
                self.config['stripe']['price_ids']['basic_monthly']: 'basic',
                self.config['stripe']['price_ids']['basic_annual']: 'basic',
                self.config['stripe']['price_ids']['pro_monthly']: 'pro',
                self.config['stripe']['price_ids']['pro_annual']: 'pro',
                self.config['stripe']['price_ids']['enterprise_monthly']: 'enterprise',
                self.config['stripe']['price_ids']['enterprise_annual']: 'enterprise'
            }
            
            return price_tier_map.get(price_id, 'basic')
            
        except Exception as e:
            print(f"Error getting subscription tier: {str(e)}")
            return 'basic'
            
    async def _send_welcome_email(self, email, tier, license_key, billing_cycle):
        """Send welcome email"""
        await self.email_sender.send_welcome_email(email, tier, license_key)
        
    async def _send_cancellation_email(self, email):
        """Send cancellation email"""
        subject = "Trading Bot Subscription Cancelled"
        body = """
        We're sorry to see you go!
        
        Your subscription has been cancelled and your license will be deactivated at the end of the current billing period.
        
        If you change your mind, you can resubscribe anytime.
        
        Thank you for trying our Trading Bot!
        """
        
        await self.email_sender.send_email(email, subject, body)
        
    async def _send_payment_failed_email(self, email, invoice_url):
        """Send payment failed email"""
        subject = "Trading Bot Payment Failed"
        body = f"""
        We were unable to process your payment for the Trading Bot subscription.
        
        Please update your payment information here:
        {invoice_url}
        
        Your subscription will be paused if payment is not received within 3 days.
        
        Need help? Contact our support team.
        """
        
        await self.email_sender.send_email(email, subject, body)
        
    async def _send_tier_update_email(self, email, new_tier, new_license):
        """Send tier update email"""
        subject = f"Trading Bot Subscription Updated to {new_tier.title()}"
        body = f"""
        Your Trading Bot subscription has been updated to the {new_tier.title()} tier!
        
        Your new license key is: {new_license}
        
        Please update your configuration with the new license key to access your upgraded features.
        
        Need help? Contact our support team.
        """
        
        await self.email_sender.send_email(email, subject, body)
