#!/usr/bin/env python3
"""
Test script for Enhanced Document Processor with integrated simple text-based orientation detection.
This demonstrates how the complex AI vision approach has been replaced with a simpler, more reliable method.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.enhanced_document_processor import EnhancedDocumentProcessor

def test_enhanced_processor_with_simple_orientation():
    """
    Test the enhanced document processor with integrated simple orientation detection.
    """
    print("🧪 ENHANCED DOCUMENT PROCESSOR WITH GOOGLE VISION ORIENTATION DETECTION")
    print("=" * 80)
    print("This test demonstrates the integration of simple Google Vision orientation detection")
    print("into the main document processing pipeline. Just asks: Does this image need rotation?\n")
    
    # Initialize the enhanced document processor
    print("🔧 Initializing Enhanced Document Processor with Google Vision...")
    processor = EnhancedDocumentProcessor()
    
    # Test images from different document types
    test_cases = [
        {
            "path": "data/dataset/passport_1/01. Passport Copy _Unaise_page_1_1_1_1.jpg",
            "type": "Passport Page 1",
            "description": "Should process with orientation detection"
        },
        {
            "path": "data/dataset/certificate/03. Diploma In Mechanical with UAE Attestation_page_2_1.jpg", 
            "type": "Certificate",
            "description": "Should process with orientation detection"
        },
        {
            "path": "data/dataset/emirates_id/02.ra_EID_Abhishek2034_page_1_1_1_1.jpg",
            "type": "Emirates ID",
            "description": "Should process with orientation detection"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📄 TEST {i}: {test_case['type']}")
        print(f"   File: {os.path.basename(test_case['path'])}")
        print(f"   Description: {test_case['description']}")
        print("-" * 60)
        
        if os.path.exists(test_case['path']):
            try:
                # Process document with integrated orientation detection
                result = processor.process_document(test_case['path'])
                
                # Check if processing was successful
                if "error" not in result:
                    print(f"✅ Processing successful!")
                    print(f"📊 Results:")
                    print(f"   Document type: {result.get('metadata', {}).get('document_type', 'unknown')}")
                    print(f"   Classification confidence: {result.get('metadata', {}).get('classification_confidence', 0):.3f}")
                    print(f"   Detected labels: {result.get('metadata', {}).get('detected_labels', 0)}")
                    print(f"   Processing method: {result.get('metadata', {}).get('processing_method', 'unknown')}")
                    print(f"   Orientation corrected: {result.get('metadata', {}).get('orientation_corrected', False)}")
                    
                    # Store successful result
                    results.append({
                        "type": test_case['type'],
                        "file": os.path.basename(test_case['path']),
                        "success": True,
                        "document_type": result.get('metadata', {}).get('document_type', 'unknown'),
                        "orientation_corrected": result.get('metadata', {}).get('orientation_corrected', False)
                    })
                    
                else:
                    print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                    results.append({
                        "type": test_case['type'],
                        "file": os.path.basename(test_case['path']),
                        "success": False,
                        "error": result.get('error', 'Unknown error')
                    })
                    
            except Exception as e:
                print(f"❌ Exception during processing: {e}")
                results.append({
                    "type": test_case['type'],
                    "file": os.path.basename(test_case['path']),
                    "success": False,
                    "error": str(e)
                })
                
        else:
            print(f"❌ File not found: {test_case['path']}")
            results.append({
                "type": test_case['type'],
                "file": os.path.basename(test_case['path']),
                "success": False,
                "error": "File not found"
            })
    
    # Print summary
    print(f"\n📋 SUMMARY")
    print("=" * 80)
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    orientation_corrected = sum(1 for r in results if r.get('orientation_corrected', False))
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    print(f"Images with orientation correction: {orientation_corrected}")
    
    print(f"\nDetailed results:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        if result['success']:
            orientation_status = "ROTATED" if result.get('orientation_corrected', False) else "KEPT"
            print(f"  {status} {result['type']}: {orientation_status} (type: {result.get('document_type', 'unknown')})")
        else:
            print(f"  {status} {result['type']}: FAILED ({result.get('error', 'Unknown error')})")
    
    print(f"\n🎯 INTEGRATION BENEFITS:")
    print("  ✅ Simple Google Vision orientation detection integrated into main pipeline")
    print("  ✅ Just asks one question: Does this image need rotation?")
    print("  ✅ Fast and reliable orientation detection")
    print("  ✅ Seamless integration with existing YOLO8 + ResNet + Document AI pipeline")
    print("  ✅ Clear metadata showing orientation correction status")

def test_orientation_detection_only():
    """
    Test just the orientation detection functionality.
    """
    print(f"\n🔄 TESTING ORIENTATION DETECTION ONLY")
    print("=" * 50)
    
    processor = EnhancedDocumentProcessor()
    
    # Test with one image
    test_image = "data/dataset/passport_1/01. Passport Copy _Unaise_page_1_1_1_1.jpg"
    
    if os.path.exists(test_image):
        print(f"Testing orientation detection on: {os.path.basename(test_image)}")
        corrected_path = processor.detect_and_correct_orientation(test_image)
        print(f"✅ Orientation detection completed. Result: {corrected_path}")
    else:
        print(f"❌ Test image not found: {test_image}")

if __name__ == "__main__":
    test_enhanced_processor_with_simple_orientation()
    test_orientation_detection_only() 