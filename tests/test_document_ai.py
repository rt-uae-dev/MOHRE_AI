#!/usr/bin/env python3
"""
Google Document AI vs Google Vision Test
Tests Document AI's ability to extract structured data from passports
"""

import os
import sys
import json
from google.cloud import documentai_v1 as documentai
from google.cloud import vision

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from yolo_crop_ocr_pipeline import run_google_vision_ocr

def test_document_ai_on_passport(image_path: str):
    """Test Google Document AI on passport image"""
    
    print(f"🔍 Testing Google Document AI on: {os.path.basename(image_path)}")
    print("=" * 60)
    
    try:
        # Initialize Document AI client
        client = documentai.DocumentProcessorServiceClient()
        
        # Use the IDENTITY_DOCUMENT processor for passports
        # You'll need to create this processor in Google Cloud Console
        processor_name = f"projects/{os.getenv('GOOGLE_CLOUD_PROJECT_ID')}/locations/us/processors/{os.getenv('DOCUMENT_AI_PROCESSOR_ID')}"
        
        # Read the image
        with open(image_path, "rb") as image:
            image_content = image.read()
        
        # Create the document
        document = {"content": image_content, "mime_type": "image/jpeg"}
        
        # Process the document
        request = {"name": processor_name, "document": document}
        result = client.process_document(request=request)
        document = result.document
        
        print("✅ Document AI Results:")
        print(f"📝 Text length: {len(document.text)} characters")
        print(f"📄 Full text:\n{document.text}")
        
        # Extract structured data
        print(f"\n🔍 Structured Data from Document AI:")
        
        # Look for specific entities
        entities = document.entities
        for entity in entities:
            print(f"📍 {entity.type}: {entity.mention_text} (confidence: {entity.confidence:.2f})")
        
        # Check for specific fields
        issue_place_found = False
        passport_number_found = False
        name_found = False
        
        for entity in entities:
            if entity.type in ["place_of_issue", "issuing_authority"]:
                print(f"✅ Found issue place: {entity.mention_text}")
                issue_place_found = True
            elif entity.type in ["document_number", "passport_number"]:
                print(f"✅ Found passport number: {entity.mention_text}")
                passport_number_found = True
            elif entity.type in ["given_names", "full_name"]:
                print(f"✅ Found name: {entity.mention_text}")
                name_found = True
        
        if not issue_place_found:
            print("❌ No issue place found")
        if not passport_number_found:
            print("❌ No passport number found")
        if not name_found:
            print("❌ No name found")
            
        return document.text, entities
        
    except Exception as e:
        print(f"❌ Document AI failed: {e}")
        return "", []

def test_google_vision_on_passport(image_path: str):
    """Test Google Vision on passport image"""
    
    print(f"\n🔍 Testing Google Vision on: {os.path.basename(image_path)}")
    print("=" * 60)
    
    try:
        result = run_google_vision_ocr(image_path)
        vision_text = result['ocr_text']
        
        print("✅ Google Vision Results:")
        print(f"📝 Text length: {len(vision_text)} characters")
        print(f"📄 Full text:\n{vision_text}")
        
        # Check for specific fields
        print(f"\n🔍 Looking for specific fields:")
        
        issue_keywords = ['place', 'issue', 'authority', 'issued', 'duba', 'dubai', 'mumbai', 'delhi']
        found_issue = []
        for line in vision_text.split('\n'):
            line_lower = line.lower()
            for keyword in issue_keywords:
                if keyword in line_lower:
                    found_issue.append(line)
                    break
        
        if found_issue:
            print(f"📍 Issue place candidates: {found_issue}")
        else:
            print(f"❌ No issue place fields found")
        
        # Check for passport number
        if '25547821' in vision_text or 'z5547821' in vision_text.lower():
            print("✅ Found passport number")
        else:
            print("❌ Passport number not found")
        
        # Check for name
        if 'yogeshkumar' in vision_text.lower() or 'yogesh' in vision_text.lower():
            print("✅ Found name")
        else:
            print("❌ Name not found")
            
        return vision_text
        
    except Exception as e:
        print(f"❌ Google Vision failed: {e}")
        return ""

def main():
    """Main test function"""
    print("🧪 Google Document AI vs Google Vision Test")
    print("=" * 60)
    
    # Test on the passport image
    image_path = "data/temp/original_passport.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Test Document AI
    doc_ai_text, doc_ai_entities = test_document_ai_on_passport(image_path)
    
    # Test Google Vision
    vision_text = test_google_vision_on_passport(image_path)
    
    # Comparison
    print(f"\n📊 Comparison Summary:")
    print(f"Document AI text length: {len(doc_ai_text)}")
    print(f"Google Vision text length: {len(vision_text)}")
    print(f"Document AI entities found: {len(doc_ai_entities)}")
    
    if len(doc_ai_text) > len(vision_text):
        print("✅ Document AI captured more text!")
    elif len(vision_text) > len(doc_ai_text):
        print("✅ Google Vision captured more text!")
    else:
        print("ℹ️ Both captured similar amount of text")

if __name__ == "__main__":
    main() 