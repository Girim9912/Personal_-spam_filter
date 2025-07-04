import os
import pickle
import base64
import re
from datetime import datetime
from typing import List, Dict, Tuple

# Gmail API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Hugging Face transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Data processing
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class GmailSpamFilter:
    def __init__(self):
        # Gmail API setup
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
        self.service = None
        
        # Initialize Hugging Face models
        self.spam_classifier = None
        self.sentiment_analyzer = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize pre-trained models from Hugging Face"""
        print("Loading Hugging Face models...")
        
        # Option 1: Use a pre-trained spam classifier
        try:
            self.spam_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",  # Can detect spam-like content
                device=0 if torch.cuda.is_available() else -1
            )
        except:
            # Fallback to a general classification model
            self.spam_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
        
        # Option 2: Load a dedicated email classification model
        # You can fine-tune this on your own spam dataset
        try:
            model_name = "microsoft/DialoGPT-medium"  # Replace with spam-specific model
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased"
            )
        except Exception as e:
            print(f"Error loading custom model: {e}")
    
    def authenticate_gmail(self, credentials_path: str = "credentials.json"):
        """Authenticate with Gmail API"""
        creds = None
        
        # Load existing token
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('gmail', 'v1', credentials=creds)
        print("Gmail API authenticated successfully!")
    
    def get_emails(self, query: str = "", max_results: int = 100) -> List[Dict]:
        """Fetch emails from Gmail"""
        try:
            # Get list of messages
            results = self.service.users().messages().list(
                userId='me', q=query, maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for message in messages:
                # Get full message details
                msg = self.service.users().messages().get(
                    userId='me', id=message['id'], format='full'
                ).execute()
                
                email_data = self.parse_email(msg)
                emails.append(email_data)
            
            return emails
            
        except Exception as e:
            print(f"Error fetching emails: {e}")
            return []
    
    def parse_email(self, message: Dict) -> Dict:
        """Parse Gmail message into structured data"""
        headers = message['payload'].get('headers', [])
        
        # Extract headers
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
        date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
        
        # Extract body
        body = self.extract_body(message['payload'])
        
        return {
            'id': message['id'],
            'subject': subject,
            'sender': sender,
            'date': date,
            'body': body,
            'labels': message.get('labelIds', [])
        }
    
    def extract_body(self, payload: Dict) -> str:
        """Extract email body text"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body += base64.urlsafe_b64decode(data).decode('utf-8')
        elif payload['mimeType'] == 'text/plain':
            data = payload['body']['data']
            body = base64.urlsafe_b64decode(data).decode('utf-8')
        
        return body
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess email text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove email addresses and URLs for privacy
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        text = re.sub(r'http\S+', '[URL]', text)
        
        return text
    
    def extract_features(self, email: Dict) -> Dict:
        """Extract additional features for spam detection"""
        features = {}
        
        # Text-based features
        subject = email['subject'].lower()
        body = email['body'].lower()
        
        # Spam indicators
        features['has_urgent_words'] = any(word in subject + body for word in 
                                         ['urgent', 'act now', 'limited time', 'expires'])
        features['has_money_words'] = any(word in subject + body for word in 
                                        ['free', 'money', 'cash', 'prize', 'winner'])
        features['excessive_caps'] = sum(1 for c in subject if c.isupper()) > len(subject) * 0.3
        features['excessive_punctuation'] = subject.count('!') > 2
        
        # Sender-based features
        sender = email['sender'].lower()
        features['sender_suspicious'] = '@' not in sender or 'noreply' in sender
        
        # Length features
        features['subject_length'] = len(email['subject'])
        features['body_length'] = len(email['body'])
        
        return features
    
    def predict_spam(self, email: Dict) -> Tuple[bool, float]:
        """Predict if email is spam using Hugging Face model"""
        # Combine subject and body
        text = f"{email['subject']} {email['body']}"
        text = self.preprocess_text(text)
        
        # Truncate text to model's max length
        text = text[:512]  # BERT models typically have 512 token limit
        
        try:
            # Get prediction from Hugging Face model
            result = self.spam_classifier(text)
            
            # Interpret results (depends on model)
            if isinstance(result, list) and len(result) > 0:
                prediction = result[0]
                
                # For toxic-bert or similar models
                if prediction['label'] in ['TOXIC', 'NEGATIVE']:
                    is_spam = True
                    confidence = prediction['score']
                else:
                    is_spam = False
                    confidence = 1 - prediction['score']
            else:
                is_spam = False
                confidence = 0.5
            
            # Combine with rule-based features
            features = self.extract_features(email)
            spam_indicators = sum([
                features['has_urgent_words'],
                features['has_money_words'],
                features['excessive_caps'],
                features['excessive_punctuation'],
                features['sender_suspicious']
            ])
            
            # Adjust confidence based on features
            if spam_indicators >= 3:
                confidence = min(confidence + 0.3, 1.0)
                is_spam = True
            elif spam_indicators >= 2:
                confidence = min(confidence + 0.1, 1.0)
            
            return is_spam, confidence
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return False, 0.5
    
    def apply_spam_filter(self, emails: List[Dict], confidence_threshold: float = 0.7):
        """Apply spam filter to emails and take action"""
        results = []
        
        for email in emails:
            is_spam, confidence = self.predict_spam(email)
            
            result = {
                'email_id': email['id'],
                'subject': email['subject'],
                'sender': email['sender'],
                'is_spam': is_spam,
                'confidence': confidence,
                'action_taken': None
            }
            
            # Take action if confidence is high enough
            if is_spam and confidence >= confidence_threshold:
                try:
                    # Add spam label and remove from inbox
                    self.service.users().messages().modify(
                        userId='me',
                        id=email['id'],
                        body={
                            'addLabelIds': ['SPAM'],
                            'removeLabelIds': ['INBOX']
                        }
                    ).execute()
                    
                    result['action_taken'] = 'Moved to spam'
                    
                except Exception as e:
                    print(f"Error moving email to spam: {e}")
                    result['action_taken'] = 'Error'
            
            results.append(result)
        
        return results
    
    def train_custom_model(self, training_data: List[Dict]):
        """Fine-tune model on your specific spam data"""
        # This is a placeholder for custom training
        # You would implement fine-tuning here with your labeled data
        print("Custom model training not implemented yet")
        print("Consider using datasets like Enron Spam or SpamAssassin for training")
    
    def generate_report(self, results: List[Dict]) -> str:
        """Generate a report of spam filtering results"""
        df = pd.DataFrame(results)
        
        report = f"""
        Spam Filtering Report
        ====================
        
        Total emails processed: {len(results)}
        Emails marked as spam: {sum(df['is_spam'])}
        Emails moved to spam: {sum(df['action_taken'] == 'Moved to spam')}
        
        Average confidence: {df['confidence'].mean():.2f}
        
        High-confidence spam subjects:
        {df[df['confidence'] > 0.8]['subject'].head(5).to_string()}
        """
        
        return report

# Usage example
def main():
    # Initialize the spam filter
    spam_filter = GmailSpamFilter()
    
    # Authenticate with Gmail
    spam_filter.authenticate_gmail("path/to/your/credentials.json")
    
    # Get recent emails
    emails = spam_filter.get_emails(query="is:unread", max_results=50)
    
    # Apply spam filter
    results = spam_filter.apply_spam_filter(emails, confidence_threshold=0.7)
    
    # Generate report
    report = spam_filter.generate_report(results)
    print(report)

if __name__ == "__main__":
    main()