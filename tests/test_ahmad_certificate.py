#!/usr/bin/env python3
"""
Test script for Ahmad's certificate using Gemini Vision features:
1. Vision-based document classification
2. Image orientation checking
3. Auto-rotation for certificate
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def test_ahmad_certificate_classification():
    """Test Gemini Vision classification with Ahmad's certificate."""
    try:
        from resnet18_classifier import classify_image_with_gemini_vision
        
        # Ahmad's certificate path
        certificate_path = "data/processed/COMPLETED/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC/Unknown_certificate.jpg"
        
        if not os.path.exists(certificate_path):
            print(f"❌ Certificate not found: {certificate_path}")
            return False
        
        print(f"🔍 Testing Gemini Vision classification with Ahmad's certificate...")
        result = classify_image_with_gemini_vision(certificate_path)
        
        print(f"✅ Gemini Vision classification result: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini Vision classification test failed: {e}")
        return False

def test_ahmad_certificate_orientation():
    """Test orientation checking with Ahmad's certificate."""
    try:
        from resnet18_classifier import check_image_orientation
        
        # Ahmad's certificate path
        certificate_path = "data/processed/COMPLETED/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC/Unknown_certificate.jpg"
        
        if not os.path.exists(certificate_path):
            print(f"❌ Certificate not found: {certificate_path}")
            return False
        
        print(f"🔍 Testing orientation checking with Ahmad's certificate...")
        analysis = check_image_orientation(certificate_path)
        
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

def test_ahmad_certificate_auto_rotation():
    """Test auto-rotation with Ahmad's certificate."""
    try:
        from resnet18_classifier import auto_rotate_image_if_needed, check_image_orientation
        
        # Ahmad's certificate path
        certificate_path = "data/processed/COMPLETED/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC/Unknown_certificate.jpg"
        
        if not os.path.exists(certificate_path):
            print(f"❌ Certificate not found: {certificate_path}")
            return False
        
        print(f"🔍 Testing auto-rotation with Ahmad's certificate...")
        
        # First check orientation
        analysis = check_image_orientation(certificate_path)
        
        # Then try to auto-rotate if needed
        corrected_path = auto_rotate_image_if_needed(certificate_path, analysis)
        
        if corrected_path != certificate_path:
            print(f"✅ Certificate was auto-corrected: {corrected_path}")
            print(f"   Original: {certificate_path}")
            print(f"   Corrected: {corrected_path}")
        else:
            print("✅ No rotation needed for certificate")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-rotation test failed: {e}")
        return False

def test_ahmad_certificate_attestation():
    """Test with Ahmad's certificate attestation."""
    try:
        from resnet18_classifier import classify_image_with_gemini_vision, check_image_orientation
        
        # Ahmad's certificate attestation path
        attestation_path = "data/processed/COMPLETED/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC/Unknown_certificate_attestation.jpg"
        
        if not os.path.exists(attestation_path):
            print(f"❌ Certificate attestation not found: {attestation_path}")
            return False
        
        print(f"🔍 Testing with Ahmad's certificate attestation...")
        
        # Test classification
        classification_result = classify_image_with_gemini_vision(attestation_path)
        print(f"   Classification: {classification_result}")
        
        # Test orientation
        orientation_analysis = check_image_orientation(attestation_path)
        print(f"   Orientation correct: {orientation_analysis.get('orientation_correct', 'unknown')}")
        print(f"   Issues: {orientation_analysis.get('issues', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Certificate attestation test failed: {e}")
        return False

def test_integrated_pipeline_with_ahmad():
    """Test the integrated pipeline with Ahmad's certificate."""
    try:
        from document_processing_pipeline import classify_and_ocr
        
        # Ahmad's certificate path
        certificate_path = "data/processed/COMPLETED/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC/Unknown_certificate.jpg"
        temp_dir = "data/temp"
        
        if not os.path.exists(certificate_path):
            print(f"❌ Certificate not found: {certificate_path}")
            return False
        
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"🔍 Testing integrated pipeline with Ahmad's certificate...")
        
        rotated_path, vision_data, final_label = classify_and_ocr(certificate_path, temp_dir)
        
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
            print(f"     - Recommendations: {analysis.get('recommendations', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated pipeline test failed: {e}")
        return False

def main():
    """Run all tests with Ahmad's certificate."""
    print("🧪 Testing Gemini Vision Features with Ahmad's Certificate")
    print("=" * 60)
    
    tests = [
        ("Certificate Classification", test_ahmad_certificate_classification),
        ("Certificate Orientation Check", test_ahmad_certificate_orientation),
        ("Certificate Auto-Rotation", test_ahmad_certificate_auto_rotation),
        ("Certificate Attestation", test_ahmad_certificate_attestation),
        ("Integrated Pipeline", test_integrated_pipeline_with_ahmad),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Gemini Vision features work great with Ahmad's certificate.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 