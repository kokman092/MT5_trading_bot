import smtplib
import logging
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
from datetime import datetime

class EmailSender:
    def __init__(self, config: Dict):
        """Initialize email sender"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.smtp_config = config['smtp']
        
    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        attachments: Optional[List[str]] = None,
        html_content: Optional[str] = None
    ) -> bool:
        """Send email with optional attachments"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add plain text body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add HTML content if provided
            if html_content:
                msg.attach(MIMEText(html_content, 'html'))
                
            # Add attachments if provided
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
                            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
                            msg.attach(part)
                            
            # Send email
            with smtplib.SMTP_SSL(
                self.smtp_config['host'],
                self.smtp_config['port']
            ) as server:
                server.login(
                    self.smtp_config['username'],
                    self.smtp_config['password']
                )
                server.send_message(msg)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email: {str(e)}")
            return False
            
    async def send_welcome_email(
        self,
        to_email: str,
        tier: str,
        license_key: str
    ) -> bool:
        """Send welcome email to new customer"""
        subject = f"Welcome to Trading Bot {tier.title()} Plan!"
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .footer {{ text-align: center; padding: 20px; color: #666; }}
                .button {{ background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to Trading Bot!</h1>
                </div>
                <div class="content">
                    <h2>Thank you for choosing our {tier.title()} Plan</h2>
                    <p>Here's your license key:</p>
                    <pre>{license_key}</pre>
                    
                    <h3>Getting Started:</h3>
                    <ol>
                        <li>Download and install the trading bot</li>
                        <li>Configure your license key</li>
                        <li>Connect to your MT5 account</li>
                        <li>Start trading!</li>
                    </ol>
                    
                    <p>
                        <a href="https://your-domain.com/docs" class="button">View Documentation</a>
                    </p>
                </div>
                <div class="footer">
                    <p>Need help? Contact our support team at {self.config['licensing']['support_email']}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return await self.send_email(
            to_email,
            subject,
            "Welcome to Trading Bot! Your license key is: " + license_key,
            html_content=html_content
        )
        
    async def send_license_expiry_notice(
        self,
        to_email: str,
        days_remaining: int,
        renewal_url: str
    ) -> bool:
        """Send license expiry notification"""
        subject = f"Your Trading Bot License Expires in {days_remaining} Days"
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #ff9800; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .footer {{ text-align: center; padding: 20px; color: #666; }}
                .button {{ background-color: #ff9800; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>License Expiry Notice</h1>
                </div>
                <div class="content">
                    <h2>Your Trading Bot license expires in {days_remaining} days</h2>
                    <p>To ensure uninterrupted trading, please renew your license before it expires.</p>
                    
                    <p>
                        <a href="{renewal_url}" class="button">Renew Now</a>
                    </p>
                </div>
                <div class="footer">
                    <p>Questions? Contact our support team at {self.config['licensing']['support_email']}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return await self.send_email(
            to_email,
            subject,
            f"Your Trading Bot license expires in {days_remaining} days. Renew at: {renewal_url}",
            html_content=html_content
        )
        
    async def send_error_notification(
        self,
        to_email: str,
        error_type: str,
        error_details: str,
        stack_trace: Optional[str] = None
    ) -> bool:
        """Send error notification to user"""
        subject = f"Trading Bot Error: {error_type}"
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #f44336; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .error-details {{ background-color: #f8f8f8; padding: 15px; border-radius: 5px; }}
                .stack-trace {{ font-family: monospace; white-space: pre-wrap; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Trading Bot Error</h1>
                </div>
                <div class="content">
                    <h2>{error_type}</h2>
                    <div class="error-details">
                        <p><strong>Time:</strong> {datetime.utcnow().isoformat()}</p>
                        <p><strong>Details:</strong></p>
                        <p>{error_details}</p>
                    </div>
                    
                    {f'<h3>Stack Trace:</h3><pre class="stack-trace">{stack_trace}</pre>' if stack_trace else ''}
                    
                    <p>Our support team has been notified and will investigate this issue.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return await self.send_email(
            to_email,
            subject,
            f"Trading Bot Error: {error_type}\n\nDetails: {error_details}",
            html_content=html_content
        )
