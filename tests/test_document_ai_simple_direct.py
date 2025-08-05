#!/usr/bin/env python3
"""
Simple Document AI Test - Direct Testing
Tests Document AI without Google Vision dependencies
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from root directory
load_dotenv('.env')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_document_ai_direct():
    """Test Document AI directly without other dependencies"""
    
    print("Document AI Direct Test")
    print("=" * 40)
    
    # Test environment variables
    print("1. Checking Environment Variables:")
    print("-" * 30)
    
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
    processor_id = os.getenv('DOCUMENT_AI_PROCESSOR_ID')
    
    print(f"GOOGLE_CLOUD_PROJECT_ID: {project_id}")
    print(f"DOCUMENT_AI_PROCESSOR_ID: {processor_id}")
    
    if not project_id or not processor_id:
        print("❌ Document AI variables not set")
        print("Please add to your .env file:")
        print("GOOGLE_CLOUD_PROJECT_ID=842132862003")
        print("DOCUMENT_AI_PROCESSOR_ID=c36524c4040096d1")
        return
    
    print("✅ Document AI variables found")
    
    # Test Document AI import
    print("\n2. Testing Document AI Import:")
    print("-" * 30)
    
    try:
        from document_ai_processor import DOCUMENT_AI_PROCESSOR
        print("✅ Document AI module imported successfully")
        
        # Test initialization
        if DOCUMENT_AI_PROCESSOR.enabled:
            print("✅ Document AI processor initialized")
            print(f"   Project ID: {DOCUMENT_AI_PROCESSOR.project_id}")
            print(f"   Processor ID: {DOCUMENT_AI_PROCESSOR.processor_id}")
        else:
            print("❌ Document AI processor not enabled")
            
    except Exception as e:
        print(f"❌ Document AI import failed: {e}")
        return
    
    # Test with a sample image
    print("\n3. Testing with Sample Image:")
    print("-" * 30)
    
    # Look for a passport image to test with
    passport_path = "data/processed/COMPLETED/For Visa Cancellation   Yogeshkumar Sant/YOGESHKUMAR_passport_1.jpg"
    
    if os.path.exists(passport_path):
        print(f"Found test image: {os.path.basename(passport_path)}")
        
        try:
            result = DOCUMENT_AI_PROCESSOR.process_document(passport_path)
            
            if "error" in result:
                print(f"❌ Document AI processing failed: {result['error']}")
            else:
                print("✅ Document AI processing successful!")
                print(f"   Text length: {len(result.get('full_text', ''))} characters")
                print(f"   Confidence: {result.get('confidence', 0):.2f}")
                print(f"   Pages: {result.get('pages', 0)}")
                
                # Show some extracted text
                text = result.get('full_text', '')
                if text:
                    print(f"   Text preview: {text[:200]}...")
                    
                    # Test field extraction
                    print("\n4. Testing Field Extraction:")
                    print("-" * 30)
                    
                    extracted_fields = DOCUMENT_AI_PROCESSOR.extract_fields_by_document_type(text)
                    for field_name, field_value in extracted_fields.items():
                        print(f"   {field_name}: {field_value}")
                
        except Exception as e:
            print(f"❌ Document AI test failed: {e}")
    else:
        print(f"❌ Test image not found: {passport_path}")
    
    print("\n5. Next Steps:")
    print("-" * 30)
    print("If Document AI is working, you can:")
    print("1. Integrate it into your main pipeline")
    print("2. Replace Google Vision for passport processing")
    print("3. Use it for all document types")

if __name__ == "__main__":
    test_document_ai_direct() 