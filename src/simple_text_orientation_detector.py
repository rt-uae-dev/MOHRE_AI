#!/usr/bin/env python3
"""
Simple Text-Based Orientation Detector
Uses OCR to detect if text is sideways instead of complex AI vision models.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

def detect_text_orientation_simple(image_path: str) -> dict:
    """
    Simple text-based orientation detection using OCR.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict with keys:
        - rotation_needed: bool
        - rotation_angle: int (0, 90, 180, 270)
        - confidence: float
        - reason: str
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {
                "rotation_needed": False,
                "rotation_angle": 0,
                "confidence": 0.0,
                "reason": "Failed to read image"
            }
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Test OCR in different orientations
        orientations = [
            (0, "0° (original)"),
            (90, "90° clockwise"),
            (180, "180° clockwise"), 
            (270, "270° clockwise")
        ]
        
        best_orientation = 0
        best_confidence = 0.0
        best_text = ""
        
        for angle, description in orientations:
            # Rotate image
            if angle == 0:
                rotated_image = image
            elif angle == 90:
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_image = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Convert to PIL for tesseract
            pil_image = Image.fromarray(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
            
            # Extract text with confidence
            try:
                # Use tesseract with confidence scoring
                data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
                
                # Calculate average confidence for detected text
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                text_parts = [text for text in data['text'] if text.strip()]
                
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    text_content = ' '.join(text_parts)
                    
                    print(f"  {description}: Confidence={avg_confidence:.1f}, Text length={len(text_content)}")
                    
                    # Update best if this orientation has higher confidence
                    if avg_confidence > best_confidence and len(text_content) > 10:
                        best_confidence = avg_confidence
                        best_orientation = angle
                        best_text = text_content
                        
                else:
                    print(f"  {description}: No text detected")
                    
            except Exception as e:
                print(f"  {description}: OCR error - {e}")
                continue
        
        # Determine if rotation is needed
        rotation_needed = best_orientation != 0
        rotation_angle = best_orientation
        
        # Create reason
        if rotation_needed:
            reason = f"Text detected with higher confidence ({best_confidence:.1f}) at {rotation_angle}° rotation"
        else:
            reason = f"Text detected with highest confidence ({best_confidence:.1f}) at original orientation"
        
        return {
            "rotation_needed": rotation_needed,
            "rotation_angle": rotation_angle,
            "confidence": best_confidence,
            "reason": reason,
            "detected_text_length": len(best_text)
        }
        
    except Exception as e:
        return {
            "rotation_needed": False,
            "rotation_angle": 0,
            "confidence": 0.0,
            "reason": f"Error: {str(e)}"
        }

def rotate_image_simple(image_path: str, angle: int) -> str:
    """
    Rotate image by specified angle and save to new file.
    
    Args:
        image_path: Path to original image
        angle: Rotation angle (90, 180, 270)
        
    Returns:
        Path to rotated image
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to read image")
        
        # Rotate image
        if angle == 90:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            raise ValueError(f"Invalid rotation angle: {angle}")
        
        # Create output path
        base_name = os.path.splitext(image_path)[0]
        extension = os.path.splitext(image_path)[1]
        output_path = f"{base_name}_rotated_{angle}deg{extension}"
        
        # Save rotated image
        cv2.imwrite(output_path, rotated_image)
        
        print(f"✅ Image rotated {angle}° and saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error rotating image: {e}")
        return image_path

def auto_rotate_image_simple(image_path: str) -> str:
    """
    Automatically detect and rotate image if needed using simple text-based approach.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Path to the corrected image (original if no rotation needed)
    """
    print(f"🔍 Analyzing text orientation for: {os.path.basename(image_path)}")
    
    # Detect orientation
    result = detect_text_orientation_simple(image_path)
    
    print(f"📊 Results:")
    print(f"  Rotation needed: {result['rotation_needed']}")
    print(f"  Rotation angle: {result['rotation_angle']}°")
    print(f"  Confidence: {result['confidence']:.1f}")
    print(f"  Reason: {result['reason']}")
    
    # Rotate if needed
    if result['rotation_needed'] and result['rotation_angle'] > 0:
        print(f"🔄 Rotating image {result['rotation_angle']}°...")
        return rotate_image_simple(image_path, result['rotation_angle'])
    else:
        print(f"✅ No rotation needed - keeping original image")
        return image_path

def test_simple_orientation_detection():
    """
    Test the simple text-based orientation detection on sample images.
    """
    test_images = [
        "data/dataset/passport_1/01. Passport Copy _Unaise_page_1_1_1_1.jpg",
        "data/dataset/certificate/03. Diploma In Mechanical with UAE Attestation_page_2_1.jpg",
        "data/dataset/emirates_id/02.ra_EID_Abhishek2034_page_1_1_1_1.jpg"
    ]
    
    print("🧪 TESTING SIMPLE TEXT-BASED ORIENTATION DETECTION")
    print("=" * 60)
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n📄 Testing: {os.path.basename(image_path)}")
            result = auto_rotate_image_simple(image_path)
            print(f"✅ Final result: {result}")
        else:
            print(f"❌ File not found: {image_path}")

if __name__ == "__main__":
    test_simple_orientation_detection() 