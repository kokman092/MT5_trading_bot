import uuid
from datetime import datetime, timedelta

class MockStripe:
    def __init__(self):
        self.customers = {}
        self.sessions = {}
        self.subscriptions = {}
        
    def create_checkout_session(self, **kwargs):
        """Create a mock checkout session"""
        session_id = f"cs_{uuid.uuid4().hex}"
        session = {
            'id': session_id,
            'customer_email': kwargs.get('customer_email'),
            'metadata': kwargs.get('metadata', {}),
            'payment_status': 'paid',
            'subscription': f"sub_{uuid.uuid4().hex}",
            'customer': f"cus_{uuid.uuid4().hex}",
            'created': int(datetime.now().timestamp())
        }
        self.sessions[session_id] = session
        return MockObject(session)
        
    def Customer(self):
        """Mock Stripe Customer class"""
        class CustomerClass:
            @staticmethod
            def retrieve(customer_id):
                return MockObject({
                    'id': customer_id,
                    'email': 'test@example.com',
                    'name': 'Test Customer'
                })
        return CustomerClass
        
    def Subscription(self):
        """Mock Stripe Subscription class"""
        class SubscriptionClass:
            @staticmethod
            def retrieve(subscription_id):
                return MockObject({
                    'id': subscription_id,
                    'customer': f"cus_{uuid.uuid4().hex}",
                    'status': 'active',
                    'current_period_end': int((datetime.now() + timedelta(days=30)).timestamp()),
                    'items': MockObject({
                        'data': [{
                            'price': MockObject({
                                'id': 'price_test_basic_monthly'
                            })
                        }]
                    })
                })
        return SubscriptionClass
        
    def Webhook(self):
        """Mock Stripe Webhook class"""
        class WebhookClass:
            @staticmethod
            def construct_event(payload, sig_header, secret):
                return MockObject({
                    'type': 'checkout.session.completed',
                    'data': MockObject({
                        'object': MockObject({
                            'customer': f"cus_{uuid.uuid4().hex}",
                            'subscription': f"sub_{uuid.uuid4().hex}",
                            'metadata': {
                                'tier': 'basic',
                                'billing_cycle': 'monthly'
                            }
                        })
                    })
                })
        return WebhookClass

class MockObject:
    """Mock Stripe object that allows attribute access"""
    def __init__(self, data):
        self._data = data
        
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'MockObject' has no attribute '{name}'")
        
    def to_dict(self):
        return self._data
