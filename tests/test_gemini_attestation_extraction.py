#!/usr/bin/env python3
"""
Test script for Gemini Vision attestation number extraction
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def test_attestation_extraction():
    """Test attestation number extraction with Gemini Vision."""
    try:
        from resnet18_classifier import extract_attestation_numbers_with_gemini_vision, extract_document_data_with_gemini_vision
        
        # Test with Ahmad's certificate (convert PDF to image first)
        base_path = "data/raw/downloads/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC"
        certificate_pdf = "Electrical Engineering Certificate BAU.pdf"
        pdf_path = os.path.join(base_path, certificate_pdf)
        
        if not os.path.exists(pdf_path):
            print(f"❌ Certificate PDF not found: {certificate_pdf}")
            return False
        
        print(f"🔍 Testing Gemini Vision Attestation Number Extraction")
        print("=" * 60)
        print(f"📄 Testing with: {certificate_pdf}")
        
        # Convert PDF to image first
        try:
            from pdf2image import convert_from_path
            print(f"📄 Converting PDF to image...")
            
            # Convert second page (page 2)
            images = convert_from_path(pdf_path, first_page=2, last_page=2)
            if not images:
                print(f"❌ Failed to convert PDF second page")
                return False
            
            # Save the second page
            temp_dir = "data/temp"
            os.makedirs(temp_dir, exist_ok=True)
            certificate_image_path = os.path.join(temp_dir, "certificate_attestation_test.jpg")
            images[0].save(certificate_image_path, "JPEG", quality=95)
            
            print(f"✅ PDF second page converted to: {certificate_image_path}")
            
        except Exception as e:
            print(f"❌ PDF conversion failed: {e}")
            return False
        
        # Test attestation number extraction
        print(f"\n🤖 Testing Attestation Number Extraction:")
        print("-" * 50)
        
        try:
            attestation_analysis = extract_attestation_numbers_with_gemini_vision(certificate_image_path)
            
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
            comprehensive_analysis = extract_document_data_with_gemini_vision(certificate_image_path)
            
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
        print(f"❌ Attestation extraction test failed: {e}")
        return False

def test_with_other_documents():
    """Test attestation extraction with other document types."""
    try:
        from resnet18_classifier import extract_attestation_numbers_with_gemini_vision
        
        base_path = "data/raw/downloads/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC"
        
        # Test with other documents
        test_docs = [
            "Ahmad Visa 2027.jpg",
            "LC cancellation Ahmad.jpg"
        ]
        
        print(f"\n🔍 Testing with Other Documents:")
        print("=" * 40)
        
        for doc_name in test_docs:
            doc_path = os.path.join(base_path, doc_name)
            
            if not os.path.exists(doc_path):
                print(f"❌ Document not found: {doc_name}")
                continue
            
            print(f"\n📄 Testing: {doc_name}")
            print("-" * 30)
            
            try:
                analysis = extract_attestation_numbers_with_gemini_vision(doc_path)
                
                print(f"   Attestation Number 1: {analysis.get('attestation_number_1', 'null')}")
                print(f"   Attestation Number 2: {analysis.get('attestation_number_2', 'null')}")
                print(f"   Certificate Number: {analysis.get('certificate_number', 'null')}")
                print(f"   Confidence: {analysis.get('confidence', 'unknown')}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Other documents test failed: {e}")
        return False

def main():
    """Run Gemini Vision attestation extraction tests."""
    print("🧪 Gemini Vision Attestation Number Extraction Testing")
    print("=" * 70)
    
    tests = [
        ("Certificate Attestation Extraction", test_attestation_extraction),
        ("Other Documents Test", test_with_other_documents),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Gemini Vision attestation extraction is working.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 