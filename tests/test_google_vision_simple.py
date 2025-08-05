#!/usr/bin/env python3
"""
Simple test for Google Vision orientation detector.
Just asks: "Does this image need to be rotated?"
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.google_vision_orientation_detector import rotate_if_needed

def test_google_vision_simple():
    """
    Test the simple Google Vision orientation detection.
    """
    print("🧪 GOOGLE VISION SIMPLE ORIENTATION TEST")
    print("=" * 50)
    print("Just asking Google Vision: Does this image need rotation?\n")
    
    # Test images
    test_images = [
        "data/dataset/passport_1/01. Passport Copy _Unaise_page_1_1_1_1.jpg",
        "data/dataset/certificate/03. Diploma In Mechanical with UAE Attestation_page_2_1.jpg",
        "data/dataset/emirates_id/02.ra_EID_Abhishek2034_page_1_1_1_1.jpg"
    ]
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n📄 TEST {i}: {os.path.basename(image_path)}")
        print("-" * 40)
        
        if os.path.exists(image_path):
            result = rotate_if_needed(image_path)
            print(f"✅ Result: {result}")
        else:
            print(f"❌ File not found: {image_path}")

if __name__ == "__main__":
    test_google_vision_simple() 