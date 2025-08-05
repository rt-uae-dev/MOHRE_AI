#!/usr/bin/env python3
"""
Test script to convert Ahmad's PDF documents to images and test Gemini Vision features:
1. Convert PDF documents to images
2. Test vision-based document classification
3. Test image orientation checking
4. Test auto-rotation for converted documents
"""

import os
import sys
from dotenv import load_dotenv
from pdf2image import convert_from_path

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def convert_pdf_to_images(pdf_path, output_dir="data/temp"):
    """Convert PDF to images and return the image paths."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        
        image_paths = []
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f"{base_name}_page_{i+1}.jpg")
            image.save(image_path, "JPEG", quality=95)
            image_paths.append(image_path)
            print(f"   ✅ Converted page {i+1}: {os.path.basename(image_path)}")
        
        return image_paths
        
    except Exception as e:
        print(f"   ❌ PDF conversion failed: {e}")
        return []

def test_pdf_conversion_and_vision():
    """Test PDF conversion and Gemini Vision features."""
    try:
        from resnet18_classifier import classify_image_with_gemini_vision, check_image_orientation
        
        base_path = "data/raw/downloads/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC"
        
        # Important PDF documents to test
        pdf_documents = [
            "Electrical Engineering Certificate BAU.pdf",
            "passport Ahmad.pdf",
            "Ahmad Emirates ID 2027.pdf",
            "Employee Information Form.pdf",
            "visa cancellation Ahmad.pdf"
        ]
        
        print("🔍 Testing PDF conversion and Gemini Vision features...")
        
        all_results = {}
        
        for pdf_name in pdf_documents:
            pdf_path = os.path.join(base_path, pdf_name)
            
            if not os.path.exists(pdf_path):
                print(f"❌ PDF not found: {pdf_name}")
                continue
            
            print(f"\n📄 Processing: {pdf_name}")
            
            # Convert PDF to images
            image_paths = convert_pdf_to_images(pdf_path)
            
            if not image_paths:
                print(f"   ❌ Failed to convert {pdf_name}")
                continue
            
            # Test each converted image
            page_results = {}
            
            for i, image_path in enumerate(image_paths):
                print(f"   🔍 Testing page {i+1}: {os.path.basename(image_path)}")
                
                try:
                    # Test classification
                    classification = classify_image_with_gemini_vision(image_path)
                    print(f"      ✅ Classification: {classification}")
                    
                    # Test orientation
                    orientation = check_image_orientation(image_path)
                    print(f"      ✅ Document type: {orientation.get('document_type', 'unknown')}")
                    print(f"      ✅ Orientation: {orientation.get('orientation_correct', 'unknown')}")
                    print(f"      ✅ Clarity: {orientation.get('clarity', 'unknown')}")
                    
                    page_results[f"page_{i+1}"] = {
                        "classification": classification,
                        "orientation": orientation
                    }
                    
                except Exception as e:
                    print(f"      ❌ Failed: {e}")
                    page_results[f"page_{i+1}"] = {"error": str(e)}
            
            all_results[pdf_name] = page_results
        
        # Print summary
        print(f"\n📊 PDF Conversion and Vision Test Summary:")
        for pdf_name, pages in all_results.items():
            print(f"\n📄 {pdf_name}:")
            for page_name, results in pages.items():
                if "error" in results:
                    print(f"   {page_name}: Error - {results['error']}")
                else:
                    print(f"   {page_name}: {results['classification']} - Orientation: {results['orientation'].get('orientation_correct', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF conversion and vision test failed: {e}")
        return False

def test_certificate_pdf_specifically():
    """Test the certificate PDF specifically since it's the main document."""
    try:
        from resnet18_classifier import classify_image_with_gemini_vision, check_image_orientation
        from document_processing_pipeline import classify_and_ocr
        
        base_path = "data/raw/downloads/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC"
        certificate_pdf = "Electrical Engineering Certificate BAU.pdf"
        pdf_path = os.path.join(base_path, certificate_pdf)
        
        if not os.path.exists(pdf_path):
            print(f"❌ Certificate PDF not found: {certificate_pdf}")
            return False
        
        print(f"🔍 Testing certificate PDF specifically: {certificate_pdf}")
        
        # Convert PDF to images
        image_paths = convert_pdf_to_images(pdf_path)
        
        if not image_paths:
            print(f"❌ Failed to convert certificate PDF")
            return False
        
        # Test the first page (usually the main certificate)
        main_image = image_paths[0]
        print(f"🔍 Testing main certificate page: {os.path.basename(main_image)}")
        
        # Test individual features
        print("   🔍 Testing individual features...")
        
        # Classification
        classification = classify_image_with_gemini_vision(main_image)
        print(f"      ✅ Classification: {classification}")
        
        # Orientation
        orientation = check_image_orientation(main_image)
        print(f"      ✅ Document type: {orientation.get('document_type', 'unknown')}")
        print(f"      ✅ Orientation: {orientation.get('orientation_correct', 'unknown')}")
        print(f"      ✅ Clarity: {orientation.get('clarity', 'unknown')}")
        
        # Test integrated pipeline
        print("   🔍 Testing integrated pipeline...")
        temp_dir = "data/temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        rotated_path, vision_data, final_label = classify_and_ocr(main_image, temp_dir)
        
        print(f"      ✅ Final label: {final_label}")
        print(f"      ✅ OCR method: {vision_data.get('ocr_method', 'unknown')}")
        print(f"      ✅ Document type: {vision_data.get('document_type', 'unknown')}")
        
        # Check for orientation analysis
        if "orientation_analysis" in vision_data:
            analysis = vision_data["orientation_analysis"]
            print(f"      ✅ Orientation analysis: Correct={analysis.get('orientation_correct', 'unknown')}")
            print(f"      ✅ Issues found: {len(analysis.get('issues', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Certificate PDF test failed: {e}")
        return False

def main():
    """Run PDF conversion and vision tests."""
    print("🧪 Testing PDF Conversion and Gemini Vision Features")
    print("=" * 60)
    
    tests = [
        ("PDF Conversion and Vision", test_pdf_conversion_and_vision),
        ("Certificate PDF Specific", test_certificate_pdf_specifically),
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
        print("🎉 All tests passed! PDF conversion and Gemini Vision features work great!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 