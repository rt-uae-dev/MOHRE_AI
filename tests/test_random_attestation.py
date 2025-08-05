#!/usr/bin/env python3
"""
Test script for Gemini Vision attestation number extraction on random dataset images
"""

import os
import sys
import random
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def test_random_attestation_image():
    """Test attestation number extraction with a random image from the dataset."""
    try:
        from resnet18_classifier import extract_attestation_numbers_with_gemini_vision, extract_document_data_with_gemini_vision
        
        # Path to certificate attestation dataset
        dataset_path = "data/dataset/certificate_attestation"
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return False
        
        # Get list of all images
        image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"❌ No image files found in dataset")
            return False
        
        # Select a random image
        random_image = random.choice(image_files)
        image_path = os.path.join(dataset_path, random_image)
        
        print(f"🔍 Testing Gemini Vision Attestation Number Extraction")
        print("=" * 60)
        print(f"📄 Testing with random image: {random_image}")
        
        # Test attestation number extraction
        print(f"\n🤖 Testing Attestation Number Extraction:")
        print("-" * 50)
        
        try:
            attestation_analysis = extract_attestation_numbers_with_gemini_vision(image_path)
            
            print(f"📋 Attestation Extraction Results:")
            print(f"   Attestation Number 1: {attestation_analysis.get('attestation_number_1', 'null')}")
            print(f"   Attestation Number 2: {attestation_analysis.get('attestation_number_2', 'null')}")
            print(f"   Receipt Number: {attestation_analysis.get('receipt_number', 'null')}")
            print(f"   Label Number: {attestation_analysis.get('label_number', 'null')}")
            print(f"   Certificate Number: {attestation_analysis.get('certificate_number', 'null')}")
            print(f"   Document ID: {attestation_analysis.get('document_id', 'null')}")
            print(f"   Institution Reference: {attestation_analysis.get('institution_reference', 'null')}")
            print(f"   Confidence: {attestation_analysis.get('confidence', 'unknown')}")
            print(f"   Notes: {attestation_analysis.get('extraction_notes', [])}")
            print(f"   Recommendations: {attestation_analysis.get('recommendations', [])}")
            
        except Exception as e:
            print(f"❌ Attestation extraction failed: {e}")
            return False
        
        # Test comprehensive document data extraction
        print(f"\n🤖 Testing Comprehensive Document Data Extraction:")
        print("-" * 60)
        
        try:
            comprehensive_analysis = extract_document_data_with_gemini_vision(image_path)
            
            print(f"📋 Comprehensive Extraction Results:")
            
            # Attestation numbers
            attestation_nums = comprehensive_analysis.get('attestation_numbers', {})
            print(f"   Primary Attestation: {attestation_nums.get('primary', 'null')}")
            print(f"   Secondary Attestation: {attestation_nums.get('secondary', 'null')}")
            print(f"   Receipt Number: {attestation_nums.get('receipt', 'null')}")
            print(f"   Label Number: {attestation_nums.get('label', 'null')}")
            
            # Personal info
            personal_info = comprehensive_analysis.get('personal_info', {})
            print(f"   Full Name: {personal_info.get('full_name', 'null')}")
            print(f"   Date of Birth: {personal_info.get('date_of_birth', 'null')}")
            print(f"   Nationality: {personal_info.get('nationality', 'null')}")
            
            # Institution
            institution = comprehensive_analysis.get('institution', {})
            print(f"   Institution: {institution.get('name', 'null')}")
            print(f"   Faculty: {institution.get('faculty', 'null')}")
            print(f"   Department: {institution.get('department', 'null')}")
            
            # Qualification
            qualification = comprehensive_analysis.get('qualification', {})
            print(f"   Degree: {qualification.get('degree', 'null')}")
            print(f"   Major: {qualification.get('major', 'null')}")
            print(f"   Grade: {qualification.get('grade', 'null')}")
            
            # Document info
            doc_info = comprehensive_analysis.get('document_info', {})
            print(f"   Document Type: {doc_info.get('type', 'null')}")
            print(f"   Issue Date: {doc_info.get('issue_date', 'null')}")
            print(f"   Expiry Date: {doc_info.get('expiry_date', 'null')}")
            print(f"   Has Official Stamp: {doc_info.get('has_official_stamp', False)}")
            
            print(f"   Confidence: {comprehensive_analysis.get('confidence', 'unknown')}")
            print(f"   Notes: {comprehensive_analysis.get('extraction_notes', [])}")
            print(f"   Recommendations: {comprehensive_analysis.get('recommendations', [])}")
            
        except Exception as e:
            print(f"❌ Comprehensive extraction failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Random attestation test failed: {e}")
        return False

def main():
    """Run Gemini Vision attestation extraction test on random dataset image."""
    print("🧪 Gemini Vision Attestation Number Extraction - Random Dataset Test")
    print("=" * 80)
    
    success = test_random_attestation_image()
    
    if success:
        print(f"\n🎉 Test completed successfully!")
    else:
        print(f"\n❌ Test failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 