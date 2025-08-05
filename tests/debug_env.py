#!/usr/bin/env python3
"""
Debug Environment Variables
"""

import os
from dotenv import load_dotenv

print("Debug Environment Variables")
print("=" * 40)

# Try loading from different locations
print("1. Trying to load from config/.env:")
try:
    load_dotenv('config/.env')
    print("   ✅ Loaded from config/.env")
except Exception as e:
    print(f"   ❌ Failed to load from config/.env: {e}")

print("\n2. Trying to load from .env:")
try:
    load_dotenv('.env')
    print("   ✅ Loaded from .env")
except Exception as e:
    print(f"   ❌ Failed to load from .env: {e}")

print("\n3. Current Environment Variables:")
print("-" * 30)

# Check all relevant variables
variables = [
    'EMAIL_ADDRESS',
    'GOOGLE_APPLICATION_CREDENTIALS', 
    'GOOGLE_CLOUD_PROJECT_ID',
    'DOCUMENT_AI_PROCESSOR_ID'
]

for var in variables:
    value = os.getenv(var)
    if value:
        print(f"   {var}: {value}")
    else:
        print(f"   {var}: None")

print("\n4. Checking if .env files exist:")
print("-" * 30)

import os
if os.path.exists('config/.env'):
    print("   ✅ config/.env exists")
else:
    print("   ❌ config/.env does not exist")

if os.path.exists('.env'):
    print("   ✅ .env exists")
else:
    print("   ❌ .env does not exist") 