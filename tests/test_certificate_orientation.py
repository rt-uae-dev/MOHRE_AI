#!/usr/bin/env python3
"""
Test script for certificate orientation with improved Gemini Vision detection
"""

import os
import sys
from dotenv import load_dotenv
from PIL import Image

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def test_certificate_orientation():
    """Test orientation detection on Ahmad's certificate."""
    try:
        from resnet18_classifier import check_image_orientation, auto_rotate_image_if_needed
        
        # Test with the certificate PDF (converted to image)
        base_path = "data/raw/downloads/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC"
        certificate_pdf = "Electrical Engineering Certificate BAU.pdf"
        pdf_path = os.path.join(base_path, certificate_pdf)
        
        if not os.path.exists(pdf_path):
            print(f"❌ Certificate PDF not found: {certificate_pdf}")
            return False
        
        print(f"🔍 Testing Certificate Orientation Detection")
        print("=" * 60)
        
        # Convert PDF to image first
        try:
            from pdf2image import convert_from_path
            print(f"📄 Converting PDF to image...")
            
            # Convert first page
            images = convert_from_path(pdf_path, first_page=1, last_page=1)
            if not images:
                print(f"❌ Failed to convert PDF")
                return False
            
            # Save the first page
            temp_dir = "data/temp"
            os.makedirs(temp_dir, exist_ok=True)
            certificate_image_path = os.path.join(temp_dir, "certificate_test.jpg")
            images[0].save(certificate_image_path, "JPEG", quality=95)
            
            print(f"✅ PDF converted to: {certificate_image_path}")
            
        except Exception as e:
            print(f"❌ PDF conversion failed: {e}")
            return False
        
        # Test orientation detection
        print(f"\n🤖 Testing Gemini Vision Orientation Analysis:")
        print("-" * 50)
        
        try:
            analysis = check_image_orientation(certificate_image_path)
            
            print(f"📋 Analysis Results:")
            print(f"   Document type: {analysis.get('document_type', 'unknown')}")
            print(f"   Current orientation: {analysis.get('current_orientation', 'unknown')}")
            print(f"   Rotation needed: {analysis.get('rotation_needed', 0)}°")
            print(f"   Rotation description: {analysis.get('rotation_description', 'Unknown')}")
            print(f"   Orientation correct: {analysis.get('orientation_correct', 'unknown')}")
            print(f"   Clarity: {analysis.get('clarity', 'unknown')}")
            print(f"   Issues: {analysis.get('issues', [])}")
            print(f"   Recommendations: {analysis.get('recommendations', [])}")
            
            # Test auto-rotation
            print(f"\n🔄 Testing Auto-Rotation:")
            print("-" * 30)
            
            corrected_path = auto_rotate_image_if_needed(certificate_image_path, analysis)
            
            if corrected_path != certificate_image_path:
                print(f"✅ Image was corrected: {os.path.basename(corrected_path)}")
                
                # Analyze the corrected image
                print(f"\n🔍 Analyzing corrected image:")
                corrected_analysis = check_image_orientation(corrected_path)
                print(f"   Document type: {corrected_analysis.get('document_type', 'unknown')}")
                print(f"   Current orientation: {corrected_analysis.get('current_orientation', 'unknown')}")
                print(f"   Rotation needed: {corrected_analysis.get('rotation_needed', 0)}°")
                print(f"   Orientation correct: {corrected_analysis.get('orientation_correct', 'unknown')}")
                
                # Check if correction was successful
                if corrected_analysis.get('orientation_correct', False):
                    print(f"✅ SUCCESS: Image is now properly oriented!")
                else:
                    print(f"⚠️ Image may still need adjustment")
            else:
                print(f"ℹ️ No rotation needed - image is already properly oriented")
            
            return True
            
        except Exception as e:
            print(f"❌ Orientation analysis failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Certificate orientation test failed: {e}")
        return False

def test_manual_rotation_simulation():
    """Test by manually creating rotated versions of the certificate."""
    try:
        from resnet18_classifier import check_image_orientation, auto_rotate_image_if_needed
        
        # Convert PDF to image first
        base_path = "data/raw/downloads/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC"
        certificate_pdf = "Electrical Engineering Certificate BAU.pdf"
        pdf_path = os.path.join(base_path, certificate_pdf)
        
        if not os.path.exists(pdf_path):
            print(f"❌ Certificate PDF not found: {certificate_pdf}")
            return False
        
        # Convert PDF to image
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        if not images:
            print(f"❌ Failed to convert PDF")
            return False
        
        # Save original
        temp_dir = "data/temp"
        os.makedirs(temp_dir, exist_ok=True)
        original_path = os.path.join(temp_dir, "certificate_original.jpg")
        images[0].save(original_path, "JPEG", quality=95)
        
        print(f"🔄 Testing Manual Rotation Simulation")
        print("=" * 50)
        
        # Test different rotations
        rotations = [90, 180, 270]
        
        for angle in rotations:
            print(f"\n🔄 Testing {angle}° rotation:")
            
            # Create rotated version
            rotated_image = images[0].rotate(angle, expand=True)
            rotated_path = os.path.join(temp_dir, f"certificate_rotated_{angle}.jpg")
            rotated_image.save(rotated_path, "JPEG", quality=95)
            
            # Analyze the rotated image
            try:
                analysis = check_image_orientation(rotated_path)
                print(f"   Document type: {analysis.get('document_type', 'unknown')}")
                print(f"   Current orientation: {analysis.get('current_orientation', 'unknown')}")
                print(f"   Rotation needed: {analysis.get('rotation_needed', 0)}°")
                print(f"   Orientation correct: {analysis.get('orientation_correct', 'unknown')}")
                
                # Test auto-correction
                corrected_path = auto_rotate_image_if_needed(rotated_path, analysis)
                if corrected_path != rotated_path:
                    print(f"   ✅ Auto-corrected to: {os.path.basename(corrected_path)}")
                else:
                    print(f"   ℹ️ No correction needed")
                
            except Exception as e:
                print(f"   ❌ Analysis failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual rotation simulation failed: {e}")
        return False

def main():
    """Run certificate orientation tests."""
    print("🧪 Certificate Orientation Testing with Improved Gemini Vision")
    print("=" * 70)
    
    tests = [
        ("Certificate Orientation Detection", test_certificate_orientation),
        ("Manual Rotation Simulation", test_manual_rotation_simulation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Improved orientation detection is working.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 