#!/usr/bin/env python3
"""
Simple test script for ResNet classifier only
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from resnet18_classifier import classify_image_resnet

def test_classifier():
    """Test the ResNet classifier with sample images from dataset"""
    
    print("🧪 Testing ResNet Classifier Only")
    print("=" * 50)
    
    # Test with sample images from the dataset directory
    dataset_dir = "data/dataset"
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    print(f"🔍 Looking for sample images in dataset: {dataset_dir}")
    
    # Find sample images from different classes
    test_images = []
    class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    # Take one sample from each of the first 5 classes
    for class_name in class_dirs[:5]:
        class_path = os.path.join(dataset_dir, class_name)
        class_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if class_files:
            # Take the first image from this class
            sample_image = os.path.join(class_path, class_files[0])
            test_images.append((sample_image, class_name))
            print(f"✅ Found sample from {class_name}: {class_files[0]}")
    
    if not test_images:
        print("❌ No sample images found in dataset")
        return
    
    print(f"\n🔍 Testing {len(test_images)} sample images from dataset")
    print("\n📁 Image paths for reference:")
    for i, (img_path, expected_class) in enumerate(test_images, 1):
        print(f"{i}. {img_path}")
    
    print("\n" + "=" * 50)
    
    for i, (img_path, expected_class) in enumerate(test_images, 1):
        print(f"\n🔍 Test {i}: {os.path.basename(img_path)}")
        print(f"📁 Full path: {img_path}")
        print(f"🎯 Expected class: {expected_class}")
        print("-" * 50)
        
        try:
            result = classify_image_resnet(img_path)
            print(f"✅ Classification: {result}")
            
            # Check if classification matches expected class
            if result == expected_class:
                print(f"🎯 PERFECT MATCH! Expected: {expected_class}, Got: {result}")
            else:
                print(f"⚠️ Mismatch - Expected: {expected_class}, Got: {result}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Classifier test completed!")
    print("\n💡 To view the images, you can:")
    print("1. Navigate to the paths shown above")
    print("2. Open the images in your file explorer")
    print("3. Or use: explorer data/dataset")

if __name__ == "__main__":
    test_classifier() 