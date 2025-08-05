#!/usr/bin/env python3
"""
Test script to compare different compression methods
"""

import os
import sys
sys.path.append('src')

from image_compression_utils import compress_image_to_jpg as compress_pil
from image_utils import compress_image_to_jpg as compress_cv2

def test_compression_methods():
    """Test both compression methods on sample images"""
    
    # Test images from the dataset (using actual files that exist)
    test_images = [
        "data/dataset/passport_1/Yogeshkumar Sant  Passport copy_page_1_1_1_1.jpg",
        "data/dataset/emirates_id/Yogeshkumar  EID copy 2026_page_1_1_1_1.jpg", 
        "data/dataset/certificate/Electrical Engineering Certificate BAU_page_1_1_1_1.jpg",
        "data/dataset/personal_photo/Ahmad photo.jpg"
    ]
    
    print("🔍 Testing compression methods...")
    print("=" * 60)
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"⚠️ Test image not found: {image_path}")
            continue
            
        print(f"\n📄 Testing: {os.path.basename(image_path)}")
        print("-" * 40)
        
        # Get original file size
        original_size = os.path.getsize(image_path) / 1024
        print(f"📏 Original size: {original_size:.1f}KB")
        
        # Test PIL compression (current method)
        try:
            pil_output = f"data/temp/{os.path.splitext(os.path.basename(image_path))[0]}_pil_compressed.jpg"
            compress_pil(image_path, pil_output, max_kb=250)
            pil_size = os.path.getsize(pil_output) / 1024
            print(f"🖼️  PIL compressed: {pil_size:.1f}KB")
        except Exception as e:
            print(f"❌ PIL compression failed: {e}")
            pil_size = 0
        
        # Test OpenCV compression (previous method)
        try:
            cv2_output = f"data/temp/{os.path.splitext(os.path.basename(image_path))[0]}_cv2_compressed.jpg"
            compress_cv2(image_path, cv2_output, max_kb=110)  # Use max_kb=110 for fair comparison
            cv2_size = os.path.getsize(cv2_output) / 1024
            print(f"📷 OpenCV compressed: {cv2_size:.1f}KB")
        except Exception as e:
            print(f"❌ OpenCV compression failed: {e}")
            cv2_size = 0
        
        # Compare results
        if pil_size > 0 and cv2_size > 0:
            pil_ratio = pil_size / original_size
            cv2_ratio = cv2_size / original_size
            print(f"📊 Compression ratios:")
            print(f"   PIL: {pil_ratio:.2f} ({pil_ratio*100:.1f}% of original)")
            print(f"   OpenCV: {cv2_ratio:.2f} ({cv2_ratio*100:.1f}% of original)")
            
            if pil_size < cv2_size:
                print(f"✅ PIL is more compressed (smaller file)")
            elif cv2_size < pil_size:
                print(f"✅ OpenCV is more compressed (smaller file)")
            else:
                print(f"⚖️  Both methods produced similar sizes")
        
        print()

if __name__ == "__main__":
    # Create temp directory
    os.makedirs("data/temp", exist_ok=True)
    
    test_compression_methods()
    
    print("🎯 Test completed! Check the compressed files in data/temp/")
    print("📁 You can manually inspect the quality of both methods.") 