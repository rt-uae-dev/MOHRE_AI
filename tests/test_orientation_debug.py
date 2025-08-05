#!/usr/bin/env python3
"""
Debug script to test orientation detection and show detailed analysis
"""

import os
import sys
from dotenv import load_dotenv
from PIL import Image
import json

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def analyze_image_properties(image_path):
    """Analyze basic image properties to understand orientation."""
    try:
        image = Image.open(image_path)
        width, height = image.size
        aspect_ratio = width / height
        
        print(f"📐 Image Properties for: {os.path.basename(image_path)}")
        print(f"   Dimensions: {width} x {height}")
        print(f"   Aspect ratio: {aspect_ratio:.2f}")
        print(f"   Format: {image.format}")
        print(f"   Mode: {image.mode}")
        
        # Check EXIF orientation if available
        try:
            exif = image._getexif()
            if exif:
                orientation = exif.get(274)  # Orientation tag
                print(f"   EXIF Orientation: {orientation}")
                if orientation:
                    orientation_meanings = {
                        1: "Normal (0°)",
                        2: "Mirrored horizontally",
                        3: "Rotated 180°",
                        4: "Mirrored vertically",
                        5: "Mirrored horizontally and rotated 90° CCW",
                        6: "Rotated 90° CW",
                        7: "Mirrored horizontally and rotated 90° CW",
                        8: "Rotated 90° CCW"
                    }
                    print(f"   EXIF Meaning: {orientation_meanings.get(orientation, 'Unknown')}")
            else:
                print(f"   EXIF Orientation: None")
        except:
            print(f"   EXIF Orientation: Could not read")
        
        return {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "format": image.format,
            "mode": image.mode
        }
        
    except Exception as e:
        print(f"❌ Failed to analyze image properties: {e}")
        return None

def test_orientation_detection():
    """Test orientation detection on Ahmad's documents."""
    try:
        from resnet18_classifier import check_image_orientation, auto_rotate_image_if_needed
        
        base_path = "data/raw/downloads/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC"
        
        # Test documents
        test_docs = [
            "Ahmad photo.jpg",
            "Ahmad Visa 2027.jpg",
            "LC cancellation Ahmad.jpg"
        ]
        
        print("🔍 Testing Orientation Detection and Analysis")
        print("=" * 60)
        
        for doc_name in test_docs:
            doc_path = os.path.join(base_path, doc_name)
            
            if not os.path.exists(doc_path):
                print(f"❌ Document not found: {doc_name}")
                continue
            
            print(f"\n📄 Testing: {doc_name}")
            print("-" * 40)
            
            # Analyze basic properties
            properties = analyze_image_properties(doc_path)
            
            # Test Gemini orientation detection
            print(f"\n🤖 Gemini Orientation Analysis:")
            try:
                analysis = check_image_orientation(doc_path)
                print(f"   Document type: {analysis.get('document_type', 'unknown')}")
                print(f"   Orientation correct: {analysis.get('orientation_correct', 'unknown')}")
                print(f"   Clarity: {analysis.get('clarity', 'unknown')}")
                print(f"   Issues: {analysis.get('issues', [])}")
                print(f"   Recommendations: {analysis.get('recommendations', [])}")
                
                # Test auto-rotation
                print(f"\n🔄 Testing Auto-Rotation:")
                corrected_path = auto_rotate_image_if_needed(doc_path, analysis)
                if corrected_path != doc_path:
                    print(f"   ✅ Image was corrected: {os.path.basename(corrected_path)}")
                    # Analyze the corrected image
                    corrected_analysis = check_image_orientation(corrected_path)
                    print(f"   Corrected orientation: {corrected_analysis.get('orientation_correct', 'unknown')}")
                else:
                    print(f"   ℹ️ No correction needed")
                    
            except Exception as e:
                print(f"   ❌ Orientation analysis failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Orientation detection test failed: {e}")
        return False

def test_manual_rotation():
    """Test manual rotation to see if we can detect orientation issues."""
    try:
        from resnet18_classifier import check_image_orientation
        
        base_path = "data/raw/downloads/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC"
        
        # Test with a specific document
        doc_path = os.path.join(base_path, "Ahmad photo.jpg")
        
        if not os.path.exists(doc_path):
            print(f"❌ Document not found: Ahmad photo.jpg")
            return False
        
        print(f"🔄 Testing Manual Rotation Analysis")
        print("=" * 50)
        
        # Load original image
        original_image = Image.open(doc_path)
        
        # Create rotated versions
        rotations = [90, 180, 270]
        
        for angle in rotations:
            print(f"\n🔄 Testing {angle}° rotation:")
            
            # Create rotated image
            rotated_image = original_image.rotate(angle, expand=True)
            
            # Save temporarily
            temp_path = f"data/temp/test_rotation_{angle}.jpg"
            os.makedirs("data/temp", exist_ok=True)
            rotated_image.save(temp_path, "JPEG", quality=95)
            
            # Analyze the rotated image
            try:
                analysis = check_image_orientation(temp_path)
                print(f"   Orientation correct: {analysis.get('orientation_correct', 'unknown')}")
                print(f"   Document type: {analysis.get('document_type', 'unknown')}")
                print(f"   Issues: {len(analysis.get('issues', []))} found")
                
                # Clean up
                os.remove(temp_path)
                
            except Exception as e:
                print(f"   ❌ Analysis failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual rotation test failed: {e}")
        return False

def main():
    """Run orientation debugging tests."""
    print("🧪 Orientation Detection Debugging")
    print("=" * 50)
    
    tests = [
        ("Image Properties Analysis", test_orientation_detection),
        ("Manual Rotation Test", test_manual_rotation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Debug Results: {passed}/{total} tests completed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 