#!/usr/bin/env python3
"""
Test script for Gemini Vision features:
1. Vision-based document classification
2. Image orientation checking
3. Auto-rotation for passport photos and certificates
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def test_gemini_vision_classification():
    """Test Gemini Vision-based document classification."""
    try:
        from resnet18_classifier import classify_image_with_gemini_vision
        
        # Test with a sample image (you can replace this with an actual image path)
        test_image_path = "data/test_samples/test_document.jpg"  # Replace with actual path
        
        if not os.path.exists(test_image_path):
            print(f"⚠️ Test image not found: {test_image_path}")
            print("Please provide a test image path or create a test image")
            return False
        
        print(f"🔍 Testing Gemini Vision classification with: {test_image_path}")
        result = classify_image_with_gemini_vision(test_image_path)
        
        print(f"✅ Gemini Vision classification result: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini Vision classification test failed: {e}")
        return False

def test_orientation_checking():
    """Test image orientation checking."""
    try:
        from resnet18_classifier import check_image_orientation
        
        # Test with a sample image
        test_image_path = "data/test_samples/test_photo.jpg"  # Replace with actual path
        
        if not os.path.exists(test_image_path):
            print(f"⚠️ Test image not found: {test_image_path}")
            print("Please provide a test image path or create a test image")
            return False
        
        print(f"🔍 Testing orientation checking with: {test_image_path}")
        analysis = check_image_orientation(test_image_path)
        
        print("✅ Orientation analysis result:")
        print(f"   Document type: {analysis.get('document_type', 'unknown')}")
        print(f"   Orientation correct: {analysis.get('orientation_correct', 'unknown')}")
        print(f"   Clarity: {analysis.get('clarity', 'unknown')}")
        print(f"   Issues: {analysis.get('issues', [])}")
        print(f"   Recommendations: {analysis.get('recommendations', [])}")
        print(f"   Dimensions: {analysis.get('image_dimensions', {})}")
        print(f"   Aspect ratio: {analysis.get('aspect_ratio', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Orientation checking test failed: {e}")
        return False

def test_auto_rotation():
    """Test automatic image rotation."""
    try:
        from resnet18_classifier import auto_rotate_image_if_needed, check_image_orientation
        
        # Test with a sample image
        test_image_path = "data/test_samples/test_rotated.jpg"  # Replace with actual path
        
        if not os.path.exists(test_image_path):
            print(f"⚠️ Test image not found: {test_image_path}")
            print("Please provide a test image path or create a test image")
            return False
        
        print(f"🔍 Testing auto-rotation with: {test_image_path}")
        
        # First check orientation
        analysis = check_image_orientation(test_image_path)
        
        # Then try to auto-rotate if needed
        corrected_path = auto_rotate_image_if_needed(test_image_path, analysis)
        
        if corrected_path != test_image_path:
            print(f"✅ Image was auto-corrected: {corrected_path}")
        else:
            print("✅ No rotation needed")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-rotation test failed: {e}")
        return False

def test_integration():
    """Test the integrated pipeline with new features."""
    try:
        from document_processing_pipeline import classify_and_ocr
        
        # Test with a sample image
        test_image_path = "data/test_samples/test_document.jpg"  # Replace with actual path
        temp_dir = "data/temp"
        
        if not os.path.exists(test_image_path):
            print(f"⚠️ Test image not found: {test_image_path}")
            print("Please provide a test image path or create a test image")
            return False
        
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"🔍 Testing integrated pipeline with: {test_image_path}")
        
        rotated_path, vision_data, final_label = classify_and_ocr(test_image_path, temp_dir)
        
        print("✅ Integrated pipeline result:")
        print(f"   Final label: {final_label}")
        print(f"   Rotated path: {rotated_path}")
        print(f"   OCR method: {vision_data.get('ocr_method', 'unknown')}")
        print(f"   Document type: {vision_data.get('document_type', 'unknown')}")
        
        # Check for orientation analysis
        if "orientation_analysis" in vision_data:
            print("   Orientation analysis: Available")
            analysis = vision_data["orientation_analysis"]
            print(f"     - Correct: {analysis.get('orientation_correct', 'unknown')}")
            print(f"     - Issues: {analysis.get('issues', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Gemini Vision Features")
    print("=" * 50)
    
    tests = [
        ("Gemini Vision Classification", test_gemini_vision_classification),
        ("Orientation Checking", test_orientation_checking),
        ("Auto-Rotation", test_auto_rotation),
        ("Integrated Pipeline", test_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Gemini Vision features are working.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\n💡 Note: Some tests require actual test images.")
        print("   Create test images in data/test_samples/ or update the paths in the test script.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 