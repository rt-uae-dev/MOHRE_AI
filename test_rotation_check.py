#!/usr/bin/env python3
"""
Test script to check if Yogeshkumar's passport page 2 was rotated incorrectly.
"""

import os
import sys
sys.path.append('src')
from google_ai_rotation import detect_orientation_with_gemini

def test_yogeshkumar_passport_rotation():
    """
    Test if Yogeshkumar's passport page 2 was rotated incorrectly.
    """
    
    # Path to the original passport page 2
    original_path = "data/temp/Yogeshkumar Sant  Passport copy_page2.jpg"
    
    # Path to the rotated passport page 2
    rotated_path = "data/temp/Yogeshkumar Sant  Passport copy_page2_rotated_90deg.jpg"
    
    print("🔍 Testing Yogeshkumar's Passport Page 2 Rotation")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(original_path):
        print(f"❌ Original file not found: {original_path}")
        return
    
    if not os.path.exists(rotated_path):
        print(f"❌ Rotated file not found: {rotated_path}")
        return
    
    print(f"📄 Original file: {original_path}")
    print(f"📄 Rotated file: {rotated_path}")
    print()
    
    # Test 1: Check if original needs rotation
    print("🔍 TEST 1: Checking if ORIGINAL passport page 2 needs rotation...")
    original_result = detect_orientation_with_gemini(original_path, "passport_2")
    print(f"Original result: {original_result}")
    print()
    
    # Test 2: Check if rotated version needs rotation
    print("🔍 TEST 2: Checking if ROTATED passport page 2 needs rotation...")
    rotated_result = detect_orientation_with_gemini(rotated_path, "passport_2")
    print(f"Rotated result: {rotated_result}")
    print()
    
    # Analysis
    print("📊 ANALYSIS:")
    print("-" * 30)
    
    if original_result.get("rotation_needed", False):
        print("✅ Original image NEEDED rotation - Google AI was correct")
    else:
        print("❌ Original image did NOT need rotation - Google AI was wrong")
    
    if rotated_result.get("rotation_needed", False):
        print("❌ Rotated image still needs rotation - rotation was incorrect")
    else:
        print("✅ Rotated image is now correct - rotation was successful")
    
    print()
    print("🎯 CONCLUSION:")
    if original_result.get("rotation_needed", False) and not rotated_result.get("rotation_needed", False):
        print("✅ Rotation was CORRECT - Google AI properly detected and fixed the orientation")
    elif not original_result.get("rotation_needed", False):
        print("❌ Rotation was INCORRECT - Google AI should not have rotated the original image")
    elif rotated_result.get("rotation_needed", False):
        print("❌ Rotation was INCOMPLETE - Google AI still detects rotation needed after rotation")
    else:
        print("❓ UNCLEAR - Both original and rotated images appear correct to Google AI")

if __name__ == "__main__":
    test_yogeshkumar_passport_rotation() 