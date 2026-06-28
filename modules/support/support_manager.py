import logging
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
import os
from ..licensing.license_manager import LicenseTier

class SupportTicket:
    def __init__(self, user_id: str, subject: str, description: str, priority: str):
        self.ticket_id = self._generate_ticket_id()
        self.user_id = user_id
        self.subject = subject
        self.description = description
        self.priority = priority
        self.status = "open"
        self.created_at = datetime.utcnow()
        self.updated_at = self.created_at
        self.responses: List[Dict] = []
        
    def _generate_ticket_id(self) -> str:
        """Generate unique ticket ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        return f"TICKET-{timestamp}"
        
    def to_dict(self) -> Dict:
        """Convert ticket to dictionary"""
        return {
            'ticket_id': self.ticket_id,
            'user_id': self.user_id,
            'subject': self.subject,
            'description': self.description,
            'priority': self.priority,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'responses': self.responses
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'SupportTicket':
        """Create ticket from dictionary"""
        ticket = cls(
            user_id=data['user_id'],
            subject=data['subject'],
            description=data['description'],
            priority=data['priority']
        )
        ticket.ticket_id = data['ticket_id']
        ticket.status = data['status']
        ticket.created_at = datetime.fromisoformat(data['created_at'])
        ticket.updated_at = datetime.fromisoformat(data['updated_at'])
        ticket.responses = data['responses']
        return ticket

class SupportManager:
    def __init__(self, config: Dict):
        """Initialize support manager"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tickets_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'support_tickets.json')
        self.tickets: Dict[str, SupportTicket] = {}
        self._load_tickets()
        
    def _load_tickets(self):
        """Load existing tickets from file"""
        try:
            if os.path.exists(self.tickets_file):
                with open(self.tickets_file, 'r') as f:
                    tickets_data = json.load(f)
                    self.tickets = {
                        ticket_id: SupportTicket.from_dict(ticket_data)
                        for ticket_id, ticket_data in tickets_data.items()
                    }
        except Exception as e:
            self.logger.error(f"Error loading tickets: {str(e)}")
            
    def _save_tickets(self):
        """Save tickets to file"""
        try:
            os.makedirs(os.path.dirname(self.tickets_file), exist_ok=True)
            with open(self.tickets_file, 'w') as f:
                tickets_data = {
                    ticket_id: ticket.to_dict()
                    for ticket_id, ticket in self.tickets.items()
                }
                json.dump(tickets_data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving tickets: {str(e)}")
            
    def create_ticket(
        self,
        user_id: str,
        subject: str,
        description: str,
        license_tier: LicenseTier
    ) -> Optional[str]:
        """Create new support ticket"""
        try:
            # Set priority based on license tier
            priority = {
                LicenseTier.ENTERPRISE: "high",
                LicenseTier.PRO: "medium",
                LicenseTier.BASIC: "low"
            }.get(license_tier, "low")
            
            ticket = SupportTicket(user_id, subject, description, priority)
            self.tickets[ticket.ticket_id] = ticket
            self._save_tickets()
            
            # Send email notification
            self._send_notification(ticket)
            
            return ticket.ticket_id
            
        except Exception as e:
            self.logger.error(f"Error creating ticket: {str(e)}")
            return None
            
    def get_ticket(self, ticket_id: str) -> Optional[Dict]:
        """Get ticket by ID"""
        ticket = self.tickets.get(ticket_id)
        return ticket.to_dict() if ticket else None
        
    def update_ticket(
        self,
        ticket_id: str,
        response: str,
        is_staff: bool = False
    ) -> bool:
        """Update existing ticket"""
        try:
            ticket = self.tickets.get(ticket_id)
            if not ticket:
                return False
                
            ticket.responses.append({
                'response': response,
                'is_staff': is_staff,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            ticket.updated_at = datetime.utcnow()
            self._save_tickets()
            
            # Send notification
            self._send_notification(ticket, is_update=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating ticket: {str(e)}")
            return False
            
    def close_ticket(self, ticket_id: str) -> bool:
        """Close support ticket"""
        try:
            ticket = self.tickets.get(ticket_id)
            if not ticket:
                return False
                
            ticket.status = "closed"
            ticket.updated_at = datetime.utcnow()
            self._save_tickets()
            
            # Send closure notification
            self._send_notification(ticket, is_closed=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing ticket: {str(e)}")
            return False
            
    def _send_notification(
        self,
        ticket: SupportTicket,
        is_update: bool = False,
        is_closed: bool = False
    ):
        """Send email notification"""
        try:
            smtp_config = self.config.get('smtp', {})
            if not smtp_config:
                return
                
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = self.config['support_email']
            
            if is_closed:
                subject = f"Ticket Closed: {ticket.ticket_id}"
                body = f"Ticket {ticket.ticket_id} has been closed.\n\nSubject: {ticket.subject}"
            elif is_update:
                subject = f"Ticket Updated: {ticket.ticket_id}"
                body = f"Ticket {ticket.ticket_id} has been updated.\n\nSubject: {ticket.subject}\n\nLatest Response:\n{ticket.responses[-1]['response']}"
            else:
                subject = f"New Support Ticket: {ticket.ticket_id}"
                body = f"New support ticket created.\n\nTicket ID: {ticket.ticket_id}\nSubject: {ticket.subject}\nDescription: {ticket.description}"
                
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP_SSL(smtp_config['host'], smtp_config['port']) as server:
                server.login(smtp_config['username'], smtp_config['password'])
                server.send_message(msg)
                
        except Exception as e:
            self.logger.error(f"Error sending notification: {str(e)}")
            
    def get_user_tickets(self, user_id: str) -> List[Dict]:
        """Get all tickets for a user"""
        return [
            ticket.to_dict()
            for ticket in self.tickets.values()
            if ticket.user_id == user_id
        ]
        
    def get_priority_tickets(self) -> List[Dict]:
        """Get high priority tickets"""
        return [
            ticket.to_dict()
            for ticket in self.tickets.values()
            if ticket.priority == "high" and ticket.status == "open"
        ]
