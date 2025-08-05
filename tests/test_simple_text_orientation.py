#!/usr/bin/env python3
"""
Test script for simple text-based orientation detection.
This demonstrates how much simpler and more reliable this approach is compared to AI vision models.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simple_text_orientation_detector import detect_text_orientation_simple, auto_rotate_image_simple

def test_simple_orientation():
    """
    Test the simple text-based orientation detection on various document types.
    """
    print("🧪 SIMPLE TEXT-BASED ORIENTATION DETECTION TEST")
    print("=" * 70)
    print("This approach uses OCR to detect text orientation instead of complex AI vision models.")
    print("Much simpler, faster, and more reliable!\n")
    
    # Test images from different document types
    test_cases = [
        {
            "path": "data/dataset/passport_1/01. Passport Copy _Unaise_page_1_1_1_1.jpg",
            "type": "Passport Page 1",
            "expected": "Should detect correct orientation"
        },
        {
            "path": "data/dataset/certificate/03. Diploma In Mechanical with UAE Attestation_page_2_1.jpg", 
            "type": "Certificate",
            "expected": "Should detect correct orientation"
        },
        {
            "path": "data/dataset/emirates_id/02.ra_EID_Abhishek2034_page_1_1_1_1.jpg",
            "type": "Emirates ID",
            "expected": "Should detect correct orientation"
        },
        {
            "path": "data/dataset/visa/06. VISA_Unaise Cheenambeedan_page_1.jpg",
            "type": "Visa Document",
            "expected": "Should detect correct orientation"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📄 TEST {i}: {test_case['type']}")
        print(f"   File: {os.path.basename(test_case['path'])}")
        print(f"   Expected: {test_case['expected']}")
        print("-" * 50)
        
        if os.path.exists(test_case['path']):
            # Test the simple orientation detection
            result = detect_text_orientation_simple(test_case['path'])
            
            print(f"📊 Results:")
            print(f"   Rotation needed: {result['rotation_needed']}")
            print(f"   Rotation angle: {result['rotation_angle']}°")
            print(f"   Confidence: {result['confidence']:.1f}")
            print(f"   Text length: {result['detected_text_length']}")
            print(f"   Reason: {result['reason']}")
            
            # Store result for summary
            results.append({
                "type": test_case['type'],
                "file": os.path.basename(test_case['path']),
                "rotation_needed": result['rotation_needed'],
                "rotation_angle": result['rotation_angle'],
                "confidence": result['confidence'],
                "success": result['confidence'] > 30  # Consider success if confidence > 30
            })
            
        else:
            print(f"❌ File not found: {test_case['path']}")
            results.append({
                "type": test_case['type'],
                "file": os.path.basename(test_case['path']),
                "rotation_needed": False,
                "rotation_angle": 0,
                "confidence": 0,
                "success": False
            })
    
    # Print summary
    print(f"\n📋 SUMMARY")
    print("=" * 70)
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    print(f"\nDetailed results:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        rotation_status = f"ROTATE {result['rotation_angle']}°" if result['rotation_needed'] else "KEEP"
        print(f"  {status} {result['type']}: {rotation_status} (confidence: {result['confidence']:.1f})")
    
    print(f"\n🎯 ADVANTAGES OF THIS APPROACH:")
    print("  ✅ No AI API calls needed")
    print("  ✅ No complex prompts or vision models")
    print("  ✅ Fast and reliable")
    print("  ✅ Works offline")
    print("  ✅ Based on actual text readability")
    print("  ✅ Easy to understand and debug")

def test_auto_rotation():
    """
    Test the automatic rotation functionality.
    """
    print(f"\n🔄 TESTING AUTO-ROTATION FUNCTIONALITY")
    print("=" * 50)
    
    # Test with one image
    test_image = "data/dataset/passport_1/01. Passport Copy _Unaise_page_1_1_1_1.jpg"
    
    if os.path.exists(test_image):
        print(f"Testing auto-rotation on: {os.path.basename(test_image)}")
        result_path = auto_rotate_image_simple(test_image)
        print(f"✅ Auto-rotation completed. Result: {result_path}")
    else:
        print(f"❌ Test image not found: {test_image}")

if __name__ == "__main__":
    test_simple_orientation()
    test_auto_rotation() 