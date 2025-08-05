#!/usr/bin/env python3
"""
Test script to verify Google Gemini integration.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def test_gemini_api_key():
    """Test if Gemini API key is configured."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("Please add your Gemini API key to your .env file:")
        print("GEMINI_API_KEY=your_gemini_api_key_here")
        return False
    
    print("✅ GEMINI_API_KEY found")
    return True

def test_gemini_import():
    """Test if Gemini library can be imported."""
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI library imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import google.generativeai: {e}")
        print("Please install the library: pip install google-generativeai")
        return False

def test_gemini_connection():
    """Test if Gemini API can be accessed."""
    try:
        import google.generativeai as genai
        
        # Configure API
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        # Test with a simple prompt
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello, please respond with 'Gemini is working!'")
        
        if "Gemini is working" in response.text:
            print("✅ Gemini API connection successful")
            return True
        else:
            print(f"⚠️ Gemini responded but not as expected: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Gemini API connection failed: {e}")
        return False

def test_structure_with_gemini():
    """Test the structure_with_gemini function."""
    try:
        from structure_with_gemini import structure_with_gemini
        
        # Test with minimal data
        test_data = structure_with_gemini(
            passport_ocr_1="Test passport OCR",
            passport_ocr_2="",
            emirates_id_ocr="",
            emirates_id_2_ocr="",
            employee_info="",
            certificate_ocr="",
            salary_data={},
            email_text="",
            resnet_label="passport_1",
            google_metadata={}
        )
        
        print("✅ structure_with_gemini function imported and called successfully")
        return True
        
    except Exception as e:
        print(f"❌ structure_with_gemini test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Google Gemini Integration")
    print("=" * 40)
    
    tests = [
        ("API Key Configuration", test_gemini_api_key),
        ("Library Import", test_gemini_import),
        ("API Connection", test_gemini_connection),
        ("Function Import", test_structure_with_gemini),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Gemini integration is ready.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 