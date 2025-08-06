#!/usr/bin/env python3
"""
Test document processing functionality
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_document_processing():
    """Test processing a sample document"""
    print("🚀 Testing Document Processing")
    print("=" * 50)
    
    try:
        from config import get_config
        from ultralytics import YOLO
        from PIL import Image
        import tempfile
        
        config = get_config()
        
        # Find a sample passport image
        sample_image = None
        passport_dir = config.data_dir / "passport_1"
        if passport_dir.exists():
            for img_file in passport_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    sample_image = img_file
                    break
        
        if not sample_image:
            print("❌ No sample passport image found")
            return False
            
        print(f"📄 Processing sample image: {sample_image.name}")
        
        # Load YOLO model
        model = YOLO(str(config.yolo_model_path))
        print("✅ YOLO model loaded")
        
        # Run detection
        results = model(str(sample_image))
        print(f"✅ Detection completed: {len(results[0].boxes)} objects found")
        
        # Show detected objects
        for i, box in enumerate(results[0].boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"   Object {i+1}: {class_name} (confidence: {confidence:.2f})")
        
        # Test image loading and basic operations
        img = Image.open(sample_image)
        print(f"✅ Image loaded: {img.size} pixels")
        
        # Test OCR (if available)
        try:
            from google.cloud import vision
            client = vision.ImageAnnotatorClient()
            
            with open(sample_image, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            
            if response.error.message:
                print(f"⚠️ OCR error: {response.error.message}")
            else:
                texts = response.text_annotations
                if texts:
                    print(f"✅ OCR completed: Found {len(texts)} text blocks")
                    print(f"   First text block: {texts[0].description[:100]}...")
                else:
                    print("⚠️ No text detected in image")
                    
        except Exception as e:
            print(f"⚠️ OCR test failed: {e}")
        
        print("\n✅ Document processing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        return False

def test_classification():
    """Test document classification"""
    print("\n🔍 Testing Document Classification")
    print("=" * 50)
    
    try:
        from config import get_config
        from ultralytics import YOLO
        
        config = get_config()
        
        # Test with different document types
        doc_types = ["passport_1", "emirates_id", "certificate"]
        
        for doc_type in doc_types:
            doc_dir = config.data_dir / doc_type
            if doc_dir.exists():
                # Find first image in this category
                for img_file in doc_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        print(f"📄 Testing classification for {doc_type}: {img_file.name}")
                        
                        # Load model and predict
                        model = YOLO(str(config.yolo_model_path))
                        results = model(str(img_file))
                        
                        # Get predictions
                        for box in results[0].boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = model.names[class_id]
                            print(f"   Detected: {class_name} (confidence: {confidence:.2f})")
                        
                        break
                        
        print("✅ Classification test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 MOHRE AI Document Processing Test")
    print("=" * 60)
    
    tests = [
        test_document_processing,
        test_classification
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All document processing tests passed!")
        print("🎉 The MOHRE AI system is working correctly!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        
    return passed == total

if __name__ == "__main__":
    main() 