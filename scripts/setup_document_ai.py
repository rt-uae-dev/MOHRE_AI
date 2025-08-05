#!/usr/bin/env python3
"""
Document AI Setup Script
Helps users configure Document AI for the MOHRE pipeline
"""

import os
import sys
from dotenv import load_dotenv

def check_environment():
    """Check current environment configuration"""
    print("🔍 Checking Document AI Configuration...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check required variables
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
    processor_id = os.getenv('DOCUMENT_AI_PROCESSOR_ID')
    credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    print(f"📋 Google Cloud Project ID: {'✅ Set' if project_id else '❌ Missing'}")
    if project_id:
        print(f"   Value: {project_id}")
    
    print(f"🔧 Document AI Processor ID: {'✅ Set' if processor_id else '❌ Missing'}")
    if processor_id:
        print(f"   Value: {processor_id}")
    
    print(f"🔑 Google Credentials: {'✅ Set' if credentials else '❌ Missing'}")
    if credentials:
        print(f"   Path: {credentials}")
        if os.path.exists(credentials):
            print(f"   File exists: ✅")
        else:
            print(f"   File exists: ❌")
    
    return project_id, processor_id, credentials

def test_document_ai_connection():
    """Test Document AI connection"""
    print(f"\n🧪 Testing Document AI Connection...")
    print("=" * 50)
    
    try:
        # Import Document AI processor
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from document_ai_processor import DOCUMENT_AI_PROCESSOR
        
        if DOCUMENT_AI_PROCESSOR.enabled:
            print("✅ Document AI is properly configured!")
            print(f"   Processor: {DOCUMENT_AI_PROCESSOR.processor_id}")
            print(f"   Project: {DOCUMENT_AI_PROCESSOR.project_id}")
            return True
        else:
            print("❌ Document AI is not properly configured")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Document AI: {e}")
        return False

def provide_setup_instructions():
    """Provide step-by-step setup instructions"""
    print(f"\n📚 Document AI Setup Instructions")
    print("=" * 50)
    
    print("""
1. 🏗️ Google Cloud Console Setup:
   - Go to: https://console.cloud.google.com/
   - Select your project (or create one)
   - Enable Document AI API: APIs & Services > Library > Document AI API > Enable

2. 🔧 Create Document AI Processor:
   - Go to: Document AI > Processors
   - Click "Create Processor"
   - Choose "Document OCR Processor"
   - Name it: "document-ocr-processor"
   - Note the Processor ID

3. 🔑 Service Account Setup:
   - Go to: IAM & Admin > Service Accounts
   - Find your existing service account (used for Google Vision)
   - Add these permissions:
     * Document AI Document Processor
     * Document AI User

4. 📝 Environment Variables:
   Add these to your .env file:
   
   GOOGLE_CLOUD_PROJECT_ID=your-project-id
   DOCUMENT_AI_PROCESSOR_ID=your-processor-id
   GOOGLE_APPLICATION_CREDENTIALS=config/GOOGLEAPI.json

5. 🧪 Test Configuration:
   Run: python test_document_ai_integration.py

6. 🚀 Use in Pipeline:
   Document AI is now integrated as the primary OCR method!
   Run: python main.py
""")

def create_env_template():
    """Create or update .env template"""
    env_template = """# Email Configuration
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
IMAP_SERVER=imap.gmail.com

# Google Cloud Vision API
GOOGLE_APPLICATION_CREDENTIALS=config/GOOGLEAPI.json

# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Google Document AI Configuration
GOOGLE_CLOUD_PROJECT_ID=your_project_id_here
DOCUMENT_AI_PROCESSOR_ID=your_processor_id_here
"""
    
    with open('.env', 'w') as f:
        f.write(env_template)
    
    print("✅ Created .env template file")
    print("   Please update it with your actual values")

def main():
    """Main setup function"""
    print("🚀 Document AI Setup for MOHRE Pipeline")
    print("=" * 60)
    
    # Check current configuration
    project_id, processor_id, credentials = check_environment()
    
    # Test connection if configured
    if project_id and processor_id and credentials:
        connection_ok = test_document_ai_connection()
        if connection_ok:
            print(f"\n🎉 Document AI is ready to use!")
            print("   You can now run: python main.py")
            return
        else:
            print(f"\n⚠️ Configuration issues detected")
    else:
        print(f"\n❌ Document AI is not configured")
    
    # Provide setup instructions
    provide_setup_instructions()
    
    # Ask if user wants to create .env template
    response = input(f"\n🤔 Would you like to create a .env template file? (y/n): ")
    if response.lower() in ['y', 'yes']:
        create_env_template()
    
    print(f"\n📖 For detailed setup guide, see: document_ai_setup_complete.md")

if __name__ == "__main__":
    main() 