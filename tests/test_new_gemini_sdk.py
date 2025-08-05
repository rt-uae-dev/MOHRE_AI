#!/usr/bin/env python3
"""
Test script for the new Google AI SDK approach.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_new_gemini_sdk():
    """Test the Google Generative AI approach."""
    try:
        import google.generativeai as genai
        
        # Configure the API key from the environment variable `GEMINI_API_KEY`.
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Explain how AI works in a few words")
        
        print("✅ Google Generative AI test successful!")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Google Generative AI test failed: {e}")
        return False

def test_api_key():
    """Test if the API key is set."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("✅ GEMINI_API_KEY found")
        return True
    else:
        print("❌ GEMINI_API_KEY not found")
        print("Please set GEMINI_API_KEY in your .env file")
        return False

def main():
    """Run the test."""
    print("🧪 Testing Google Generative AI")
    print("=" * 40)
    
    # Test API key
    if not test_api_key():
        return False
    
    # Test SDK
    if not test_new_gemini_sdk():
        return False
    
    print("\n🎉 All tests passed! Google Generative AI is working.")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 