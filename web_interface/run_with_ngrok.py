from pyngrok import ngrok
import json
import os
import sys
import subprocess

def setup_ngrok():
    # Get the public URL
    http_tunnel = ngrok.connect(5000)
    public_url = http_tunnel.public_url
    
    # Update config.json with the ngrok URL
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['web_interface']['domain'] = public_url
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"\nNgrok URL: {public_url}")
    print("Updated config.json with ngrok URL")
    print("\nNext steps:")
    print("1. Go to https://stripe.com and sign up")
    print("2. Get your API keys from Stripe Dashboard > Developers > API keys")
    print("3. Create products and prices in Stripe Dashboard > Products")
    print("4. Set up webhook in Stripe Dashboard > Developers > Webhooks")
    print("   - Webhook URL: {}/webhook".format(public_url))
    print("   - Events to listen for:")
    print("     * checkout.session.completed")
    print("     * customer.subscription.deleted")
    print("     * customer.subscription.updated")
    print("     * invoice.payment_failed")
    print("\nUpdate config.json with your Stripe keys and price IDs")
    
if __name__ == '__main__':
    try:
        setup_ngrok()
        
        # Start Flask app
        subprocess.run([sys.executable, "app.py"], 
                      cwd=os.path.dirname(os.path.abspath(__file__)))
                      
    except KeyboardInterrupt:
        print("\nShutting down...")
        ngrok.kill()
