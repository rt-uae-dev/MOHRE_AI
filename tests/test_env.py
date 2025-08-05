#!/usr/bin/env python3
"""
Test environment variables loading
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Environment Variables Test")
print("=" * 40)

# Check Document AI variables
project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
processor_id = os.getenv('DOCUMENT_AI_PROCESSOR_ID')

print(f"GOOGLE_CLOUD_PROJECT_ID: {project_id}")
print(f"DOCUMENT_AI_PROCESSOR_ID: {processor_id}")

# Check other variables
print(f"EMAIL_ADDRESS: {os.getenv('EMAIL_ADDRESS', 'Not set')}")
print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}")

# Check if credentials file exists
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if credentials_path:
    if os.path.exists(credentials_path):
        print(f"✅ Credentials file exists: {credentials_path}")
    else:
        print(f"❌ Credentials file missing: {credentials_path}")
else:
    print("❌ GOOGLE_APPLICATION_CREDENTIALS not set") 