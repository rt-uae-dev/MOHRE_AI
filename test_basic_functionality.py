#!/usr/bin/env python3
"""
Basic functionality test for MOHRE AI system
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_basic_imports():
    """Test if basic modules can be imported"""
    print("🔍 Testing basic imports...")
    
    try:
        from config import get_config
        print("✅ Config module imported successfully")
        
        config = get_config()
        print(f"✅ Config loaded: {config}")
        
        # Test if models exist
        if config.yolo_model_path.exists():
            print(f"✅ YOLO model found: {config.yolo_model_path}")
        else:
            print(f"❌ YOLO model not found: {config.yolo_model_path}")
            
        if config.model_save_path.exists():
            print(f"✅ Classifier model found: {config.model_save_path}")
        else:
            print(f"❌ Classifier model not found: {config.model_save_path}")
            
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_access():
    """Test if we can access the dataset"""
    print("\n🔍 Testing data access...")
    
    try:
        from config import get_config
        config = get_config()
        
        if config.data_dir.exists():
            print(f"✅ Data directory found: {config.data_dir}")
            
            # List available document types
            doc_types = [d for d in config.data_dir.iterdir() if d.is_dir()]
            print(f"✅ Found {len(doc_types)} document types:")
            for doc_type in doc_types[:5]:  # Show first 5
                print(f"   - {doc_type.name}")
            if len(doc_types) > 5:
                print(f"   ... and {len(doc_types) - 5} more")
                
            return True
        else:
            print(f"❌ Data directory not found: {config.data_dir}")
            return False
    except Exception as e:
        print(f"❌ Data access failed: {e}")
        return False

def test_model_loading():
    """Test if models can be loaded"""
    print("\n🔍 Testing model loading...")
    
    try:
        # Test YOLO model
        from ultralytics import YOLO
        from config import get_config
        
        config = get_config()
        if config.yolo_model_path.exists():
            model = YOLO(str(config.yolo_model_path))
            print("✅ YOLO model loaded successfully")
        else:
            print("❌ YOLO model file not found")
            return False
            
        # Test classifier model
        try:
            import torch
            if config.model_save_path.exists():
                classifier = torch.load(str(config.model_save_path), map_location='cpu', weights_only=True)
                print("✅ Classifier model loaded successfully")
            else:
                print("⚠️ Classifier model not found, but this is optional")
        except Exception as e:
            print(f"⚠️ Classifier model loading failed: {e}")
            
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_google_apis():
    """Test Google API connectivity"""
    print("\n🔍 Testing Google APIs...")
    
    try:
        from config import get_config
        config = get_config()
        
        if config.google_application_credentials.exists():
            print(f"✅ Google credentials found: {config.google_application_credentials}")
            
            # Test if we can import Google Vision
            try:
                from google.cloud import vision
                print("✅ Google Vision API imported successfully")
            except ImportError:
                print("⚠️ Google Vision API not installed")
                
            # Test if we can import Document AI
            try:
                from google.cloud import documentai
                print("✅ Google Document AI imported successfully")
            except ImportError:
                print("⚠️ Google Document AI not installed")
                
            return True
        else:
            print(f"❌ Google credentials not found: {config.google_application_credentials}")
            return False
    except Exception as e:
        print(f"❌ Google API test failed: {e}")
        return False

def test_sample_processing():
    """Test processing a sample image"""
    print("\n🔍 Testing sample processing...")
    
    try:
        from config import get_config
        config = get_config()
        
        # Find a sample image
        sample_image = None
        for doc_type in config.data_dir.iterdir():
            if doc_type.is_dir():
                for img_file in doc_type.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        sample_image = img_file
                        break
                if sample_image:
                    break
                    
        if sample_image:
            print(f"✅ Found sample image: {sample_image.name}")
            
            # Test basic image operations
            try:
                from PIL import Image
                img = Image.open(sample_image)
                print(f"✅ Image loaded: {img.size} pixels, {img.mode} mode")
                
                # Test YOLO detection
                try:
                    from ultralytics import YOLO
                    model = YOLO(str(config.yolo_model_path))
                    results = model(str(sample_image))
                    print(f"✅ YOLO detection completed: {len(results[0].boxes)} objects detected")
                except Exception as e:
                    print(f"⚠️ YOLO detection failed: {e}")
                    
                return True
            except Exception as e:
                print(f"❌ Image processing failed: {e}")
                return False
        else:
            print("❌ No sample images found")
            return False
    except Exception as e:
        print(f"❌ Sample processing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 MOHRE AI Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_data_access,
        test_model_loading,
        test_google_apis,
        test_sample_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! The system is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        
    return passed == total

if __name__ == "__main__":
    main() 