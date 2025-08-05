#!/usr/bin/env python3
"""
Test script for Enhanced Document Processor with trained YOLO8 and ResNet models
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def test_enhanced_processor():
    """Test the enhanced document processor with trained models."""
    try:
        from enhanced_document_processor import EnhancedDocumentProcessor
        
        print("🧪 Enhanced Document Processor Testing")
        print("=" * 60)
        
        # Initialize processor
        processor = EnhancedDocumentProcessor()
        
        # Test with a simple test image
        test_image = "data/raw/downloads/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC/Ahmad photo.jpg"
        
        if not os.path.exists(test_image):
            print(f"❌ Test image not found: {test_image}")
            return False
        
        print(f"📄 Testing with: Ahmad photo.jpg")
        print("-" * 40)
        
        # Process the document
        result = processor.process_document(test_image)
        
        # Display results
        print(f"\n📋 Processing Results:")
        print("=" * 40)
        
        if "error" in result:
            print(f"❌ Processing failed: {result['error']}")
            return False
        
        # Display metadata
        metadata = result.get("metadata", {})
        print(f"📋 Document Type: {metadata.get('document_type', 'unknown')}")
        print(f"📊 Classification Confidence: {metadata.get('classification_confidence', 0):.3f}")
        print(f"🎯 Detected Labels: {metadata.get('detected_labels', 0)}")
        print(f"🔧 Processing Method: {metadata.get('processing_method', 'unknown')}")
        
        # Display attestation numbers
        attestation_nums = result.get("attestation_numbers", {})
        print(f"\n🔢 Attestation Numbers:")
        print(f"   Primary: {attestation_nums.get('primary', 'null')}")
        print(f"   Secondary: {attestation_nums.get('secondary', 'null')}")
        print(f"   Receipt: {attestation_nums.get('receipt', 'null')}")
        print(f"   Label: {attestation_nums.get('label', 'null')}")
        
        # Display document info
        doc_info = result.get("document_info", {})
        print(f"\n📄 Document Info:")
        print(f"   Type: {doc_info.get('type', 'null')}")
        print(f"   Issue Date: {doc_info.get('issue_date', 'null')}")
        print(f"   Expiry Date: {doc_info.get('expiry_date', 'null')}")
        print(f"   Has Official Stamp: {doc_info.get('has_official_stamp', False)}")
        
        # Display confidence and notes
        print(f"\n📊 Confidence: {result.get('confidence', 'unknown')}")
        print(f"📝 Notes: {result.get('extraction_notes', [])}")
        print(f"💡 Recommendations: {result.get('recommendations', [])}")
        
        # Save results to file
        output_file = "enhanced_processor_results.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Cleanup
        processor.cleanup_temp_files()
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced processor test failed: {e}")
        return False

def test_model_loading():
    """Test if the trained models can be loaded correctly."""
    try:
        from enhanced_document_processor import EnhancedDocumentProcessor
        
        print("🔧 Testing Model Loading")
        print("=" * 40)
        
        processor = EnhancedDocumentProcessor()
        
        # Check if models are loaded
        yolo_loaded = processor.yolo_model is not None
        classifier_loaded = processor.classifier_model is not None
        doc_ai_loaded = processor.document_ai_client is not None
        
        print(f"🤖 YOLO8 Model: {'✅ Loaded' if yolo_loaded else '❌ Failed'}")
        print(f"📋 ResNet Classifier: {'✅ Loaded' if classifier_loaded else '❌ Failed'}")
        print(f"📄 Document AI Client: {'✅ Loaded' if doc_ai_loaded else '❌ Failed'}")
        
        return yolo_loaded and classifier_loaded
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def main():
    """Run all tests for the enhanced document processor."""
    print("🚀 Enhanced Document Processor - Complete Testing Suite")
    print("=" * 80)
    
    tests = [
        ("Model Loading Test", test_model_loading),
        ("Enhanced Processor Test", test_enhanced_processor),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 40)
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced processor is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 