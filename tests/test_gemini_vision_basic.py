#!/usr/bin/env python3
"""
Basic test script for Gemini Vision features - tests imports and basic functionality
without requiring actual test images.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def test_imports():
    """Test that all new functions can be imported."""
    try:
        from resnet18_classifier import (
            classify_image_with_gemini_vision,
            check_image_orientation,
            auto_rotate_image_if_needed
        )
        print("✅ All new functions imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_gemini_api_key():
    """Test if Gemini API key is configured."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("✅ GEMINI_API_KEY found")
        return True
    else:
        print("❌ GEMINI_API_KEY not found")
        print("Please set GEMINI_API_KEY in your .env file")
        return False

def test_gemini_vision_connection():
    """Test basic Gemini Vision connection."""
    try:
        import google.generativeai as genai
        
        # Configure API
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        # Test with a simple text prompt (no image)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello, please respond with 'Gemini Vision is working!'")
        
        if "Gemini Vision is working" in response.text:
            print("✅ Gemini Vision API connection successful")
            return True
        else:
            print(f"⚠️ Gemini responded but not as expected: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Gemini Vision connection failed: {e}")
        return False

def test_function_signatures():
    """Test that functions have the correct signatures."""
    try:
        from resnet18_classifier import (
            classify_image_with_gemini_vision,
            check_image_orientation,
            auto_rotate_image_if_needed
        )
        
        # Test function signatures by checking they can be called with expected parameters
        import inspect
        
        # Check classify_image_with_gemini_vision
        sig = inspect.signature(classify_image_with_gemini_vision)
        if len(sig.parameters) == 1 and 'image_path' in sig.parameters:
            print("✅ classify_image_with_gemini_vision signature correct")
        else:
            print("❌ classify_image_with_gemini_vision signature incorrect")
            return False
        
        # Check check_image_orientation
        sig = inspect.signature(check_image_orientation)
        if len(sig.parameters) == 1 and 'image_path' in sig.parameters:
            print("✅ check_image_orientation signature correct")
        else:
            print("❌ check_image_orientation signature incorrect")
            return False
        
        # Check auto_rotate_image_if_needed
        sig = inspect.signature(auto_rotate_image_if_needed)
        if len(sig.parameters) == 2 and 'image_path' in sig.parameters and 'analysis' in sig.parameters:
            print("✅ auto_rotate_image_if_needed signature correct")
        else:
            print("❌ auto_rotate_image_if_needed signature incorrect")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Function signature test failed: {e}")
        return False

def test_document_processing_pipeline_import():
    """Test that the updated document processing pipeline can be imported."""
    try:
        from document_processing_pipeline import classify_and_ocr
        print("✅ Updated document processing pipeline imported successfully")
        return True
    except Exception as e:
        print(f"❌ Document processing pipeline import failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🧪 Basic Testing of Gemini Vision Features")
    print("=" * 50)
    
    tests = [
        ("Function Imports", test_imports),
        ("API Key Configuration", test_gemini_api_key),
        ("Gemini Vision Connection", test_gemini_vision_connection),
        ("Function Signatures", test_function_signatures),
        ("Pipeline Import", test_document_processing_pipeline_import),
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
        print("🎉 All basic tests passed! Gemini Vision features are ready.")
        print("\n💡 To test with actual images:")
        print("   1. Place test images in data/test_samples/")
        print("   2. Run: python test_gemini_vision_features.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 