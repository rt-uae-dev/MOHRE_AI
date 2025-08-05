#!/usr/bin/env python3
"""
Document AI Integration Test
Comprehensive test of Document AI integration for all document types
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from document_ai_processor import DOCUMENT_AI_PROCESSOR
from yolo_crop_ocr_pipeline import run_google_vision_ocr, run_enhanced_ocr
from document_processing_pipeline import classify_and_ocr

def test_document_ai_integration():
    """Comprehensive test of Document AI integration"""
    
    print("🔍 Document AI Integration Test")
    print("=" * 60)
    
    # Check Document AI availability
    if not DOCUMENT_AI_PROCESSOR.enabled:
        print("❌ Document AI not configured. Please set up Document AI first.")
        print("   Follow the setup guide in document_ai_setup_complete.md")
        return False
    
    print("✅ Document AI is configured and ready")
    
    # Test with different document types
    test_documents = [
        {
            "name": "Passport",
            "path": "data/dataset/passport_1/01. Passport Copy _Unaise_page_1_1_1_1.jpg",
            "expected_fields": ["Passport Number", "Passport Issue Place", "Full Name", "Date of Birth"]
        },
        {
            "name": "Emirates ID",
            "path": "data/dataset/emirates_id/02.ra_EID_Abhishek2034_page_1_1_1_1.jpg",
            "expected_fields": ["EID Number", "Full Name", "Date of Birth", "Nationality"]
        },
        {
            "name": "Certificate",
            "path": "data/dataset/certificate/03. Diploma In Mechanical with UAE Attestation_page_2_1.jpg",
            "expected_fields": ["Degree/Qualification", "Institution", "Attestation Number"]
        },
        {
            "name": "Certificate Attestation",
            "path": "data/dataset/certificate_attestation/03. Diploma In Mechanical with UAE Attestation_page_3_1.jpg",
            "expected_fields": ["Attestation Number", "Authority", "Issue Date"]
        }
    ]
    
    results = {
        "document_ai_success": 0,
        "google_vision_success": 0,
        "document_ai_better": 0,
        "total_tests": 0
    }
    
    for doc_info in test_documents:
        print(f"\n{'='*20} Testing {doc_info['name']} {'='*20}")
        
        # Check if document exists
        if not os.path.exists(doc_info["path"]):
            print(f"❌ Document not found: {doc_info['path']}")
            print(f"   Looking for alternative files...")
            
            # Try to find similar files
            base_dir = os.path.dirname(doc_info["path"])
            if os.path.exists(base_dir):
                files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    doc_info["path"] = os.path.join(base_dir, files[0])
                    print(f"   Using: {files[0]}")
                else:
                    print(f"   No image files found in {base_dir}")
                    continue
            else:
                print(f"   Directory not found: {base_dir}")
                continue
        
        print(f"Testing on: {os.path.basename(doc_info['path'])}")
        print("-" * 40)
        
        results["total_tests"] += 1
        
        # Test Google Vision (current approach)
        print("1. Testing Google Vision (current approach)...")
        try:
            start_time = time.time()
            vision_result = run_google_vision_ocr(doc_info["path"])
            vision_time = time.time() - start_time
            
            vision_text = vision_result.get("ocr_text", "")
            print(f"   ⏱️ Processing time: {vision_time:.2f}s")
            print(f"   📄 Text length: {len(vision_text)} characters")
            print(f"   📝 Text preview: {vision_text[:200]}...")
            
            # Check for expected fields in Vision text
            vision_fields_found = 0
            for field in doc_info["expected_fields"]:
                if field.lower().replace(" ", "") in vision_text.lower().replace(" ", ""):
                    print(f"   ✅ Found {field} in Vision text")
                    vision_fields_found += 1
                else:
                    print(f"   ❌ No {field} found in Vision text")
            
            results["google_vision_success"] += 1
                    
        except Exception as e:
            print(f"   ❌ Google Vision failed: {e}")
            vision_result = {"ocr_text": ""}
            vision_fields_found = 0
            vision_time = 0
        
        # Test Document AI Document OCR Processor
        print("\n2. Testing Document AI Document OCR Processor...")
        try:
            start_time = time.time()
            doc_ai_result = DOCUMENT_AI_PROCESSOR.process_document(doc_info["path"])
            doc_ai_time = time.time() - start_time
            
            if "error" in doc_ai_result:
                print(f"   ❌ Document AI failed: {doc_ai_result['error']}")
                doc_ai_fields_found = 0
                doc_ai_time = 0
            else:
                print(f"   ✅ Document AI processed successfully")
                print(f"   ⏱️ Processing time: {doc_ai_time:.2f}s")
                
                full_text = doc_ai_result.get('full_text', '')
                confidence = doc_ai_result.get('confidence', 0.0)
                pages = doc_ai_result.get('pages', 0)
                
                print(f"   📄 Text length: {len(full_text)} characters")
                print(f"   🎯 Overall confidence: {confidence:.2f}")
                print(f"   📄 Pages processed: {pages}")
                
                # Show OCR data
                ocr_data = doc_ai_result.get("ocr_data", {})
                print(f"   📝 Text blocks found: {len(ocr_data.get('text_blocks', []))}")
                
                # Extract fields based on document type
                document_type = DOCUMENT_AI_PROCESSOR.get_document_type(full_text)
                extracted_fields = DOCUMENT_AI_PROCESSOR.extract_fields_by_document_type(full_text)
                
                print(f"   📋 Document Type: {document_type}")
                print(f"   🔍 Extracted Fields: {len(extracted_fields)}")
                
                # Show extracted fields
                for field_name, field_value in extracted_fields.items():
                    if field_value and field_value != "N/A":
                        print(f"      - {field_name}: {field_value}")
                
                # Check for expected fields
                doc_ai_fields_found = 0
                for field in doc_info["expected_fields"]:
                    field_found = False
                    # Check in extracted fields first
                    for extracted_field, value in extracted_fields.items():
                        if field.lower() in extracted_field.lower() and value and value != "N/A":
                            print(f"   ✅ Found {field} in Document AI extracted fields: {value}")
                            doc_ai_fields_found += 1
                            field_found = True
                            break
                    
                    # Check in full text if not found in extracted fields
                    if not field_found and field.lower().replace(" ", "") in full_text.lower().replace(" ", ""):
                        print(f"   ✅ Found {field} in Document AI text")
                        doc_ai_fields_found += 1
                    elif not field_found:
                        print(f"   ❌ No {field} found in Document AI")
                
                results["document_ai_success"] += 1
                
                # Compare results
                if doc_ai_fields_found > vision_fields_found:
                    print(f"   🏆 Document AI found more fields ({doc_ai_fields_found} vs {vision_fields_found})")
                    results["document_ai_better"] += 1
                elif doc_ai_fields_found < vision_fields_found:
                    print(f"   📊 Google Vision found more fields ({vision_fields_found} vs {doc_ai_fields_found})")
                else:
                    print(f"   🤝 Both methods found same number of fields ({doc_ai_fields_found})")
                
                # Compare processing times
                if doc_ai_time > 0 and vision_time > 0:
                    if doc_ai_time < vision_time:
                        print(f"   ⚡ Document AI was faster ({doc_ai_time:.2f}s vs {vision_time:.2f}s)")
                    else:
                        print(f"   🐌 Google Vision was faster ({vision_time:.2f}s vs {doc_ai_time:.2f}s)")
                
        except Exception as e:
            print(f"   ❌ Document AI error: {e}")
            doc_ai_fields_found = 0
            doc_ai_time = 0
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {results['total_tests']}")
    print(f"Document AI successful: {results['document_ai_success']}/{results['total_tests']}")
    print(f"Google Vision successful: {results['google_vision_success']}/{results['total_tests']}")
    print(f"Document AI better: {results['document_ai_better']}/{results['total_tests']}")
    
    if results['document_ai_better'] > results['total_tests'] / 2:
        print("✅ Document AI integration is working well!")
        return True
    else:
        print("⚠️ Document AI needs improvement or configuration")
        return False

def test_enhanced_ocr_pipeline():
    """Test the enhanced OCR pipeline with Document AI integration"""
    
    print(f"\n{'='*60}")
    print("🔧 TESTING ENHANCED OCR PIPELINE")
    print(f"{'='*60}")
    
    # Test with a sample document
    test_path = "data/dataset/passport_1/01. Passport Copy _Unaise_page_1_1_1_1.jpg"
    
    if not os.path.exists(test_path):
        print(f"❌ Test document not found: {test_path}")
        return False
    
    print(f"Testing enhanced OCR pipeline with: {os.path.basename(test_path)}")
    
    try:
        # Test the full pipeline
        rotated_path, vision_data, label = classify_and_ocr(test_path, "data/temp")
        
        print(f"✅ Pipeline completed successfully")
        print(f"📋 Final label: {label}")
        print(f"🔧 OCR method: {vision_data.get('ocr_method', 'unknown')}")
        print(f"📄 Document type: {vision_data.get('document_type', 'unknown')}")
        print(f"🎯 Confidence: {vision_data.get('confidence', 0.0):.2f}")
        
        # Show extracted fields
        extracted_fields = vision_data.get('extracted_fields', {})
        if extracted_fields:
            print(f"📝 Extracted fields:")
            for field_name, field_value in extracted_fields.items():
                if field_value and field_value != "N/A":
                    print(f"   - {field_name}: {field_value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Document AI Integration Tests...")
    
    # Test Document AI integration
    integration_success = test_document_ai_integration()
    
    # Test enhanced OCR pipeline
    pipeline_success = test_enhanced_ocr_pipeline()
    
    print(f"\n{'='*60}")
    print("🎯 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Document AI Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
    print(f"Enhanced OCR Pipeline: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
    
    if integration_success and pipeline_success:
        print("\n🎉 All tests passed! Document AI integration is ready for production.")
    else:
        print("\n⚠️ Some tests failed. Please check the configuration and try again.") 