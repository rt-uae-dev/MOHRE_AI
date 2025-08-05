#!/usr/bin/env python3
"""
Simple test for improved orientation detection
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def test_improved_orientation():
    """Test the improved orientation detection."""
    try:
        from resnet18_classifier import check_image_orientation
        
        # Test with an existing image file
        base_path = "data/raw/downloads/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC"
        test_image = "Ahmad photo.jpg"
        image_path = os.path.join(base_path, test_image)
        
        if not os.path.exists(image_path):
            print(f"❌ Test image not found: {test_image}")
            return False
        
        print(f"🔍 Testing improved orientation detection with: {test_image}")
        
        # Test the improved orientation detection
        analysis = check_image_orientation(image_path)
        
        print(f"📋 Results:")
        print(f"   Document type: {analysis.get('document_type', 'unknown')}")
        print(f"   Current orientation: {analysis.get('current_orientation', 'unknown')}")
        print(f"   Rotation needed: {analysis.get('rotation_needed', 0)}°")
        print(f"   Rotation description: {analysis.get('rotation_description', 'Unknown')}")
        print(f"   Orientation correct: {analysis.get('orientation_correct', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Simple Orientation Test")
    print("=" * 30)
    
    if test_improved_orientation():
        print("✅ Test passed!")
    else:
        print("❌ Test failed!") 