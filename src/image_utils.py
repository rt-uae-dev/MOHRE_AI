#!/usr/bin/env python3
"""
Image utility functions for the MOHRE document processing pipeline.
"""

import cv2
import os
import numpy as np

def compress_image_to_jpg(image_path: str, output_path: str, max_kb: int = 110) -> str:
    """
    Compress an image to JPG format targeting a maximum file size.
    
    Args:
        image_path: Path to the input image
        output_path: Path for the output compressed image
        max_kb: Maximum file size in KB (default: 110KB)
    
    Returns:
        Path to the compressed image
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Get original file size
        original_size = os.path.getsize(image_path) / 1024  # KB
        
        # Start with high quality and reduce until we meet the target
        quality = 95
        current_image = image.copy()
        compressed_size = float('inf')  # Start with infinity
        
        # First try: reduce quality more gradually
        while quality > 15 and compressed_size > max_kb:
            # Compress and save
            cv2.imwrite(output_path, current_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # Get compressed file size
            compressed_size = os.path.getsize(output_path) / 1024  # KB
            
            # If we're under the target, we're done
            if compressed_size <= max_kb:
                break
            
            # Reduce quality more gradually (5 instead of 10)
            quality -= 5
        
        # If still too large, resize the image
        if compressed_size > max_kb:
            height, width = current_image.shape[:2]
            scale_factor = 0.9
            
            # Calculate minimum scale factor to reach 600x600
            min_scale_for_600 = min(600 / width, 600 / height)
            
            while compressed_size > max_kb and scale_factor > min_scale_for_600:
                # Resize image
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # Ensure minimum dimensions of 600x600
                new_width = max(new_width, 600)
                new_height = max(new_height, 600)
                
                current_image = cv2.resize(current_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Try compressing again with higher quality
                cv2.imwrite(output_path, current_image, [cv2.IMWRITE_JPEG_QUALITY, 20])
                compressed_size = os.path.getsize(output_path) / 1024
                
                # If still too large, reduce quality further
                if compressed_size > max_kb:
                    cv2.imwrite(output_path, current_image, [cv2.IMWRITE_JPEG_QUALITY, 10])
                    compressed_size = os.path.getsize(output_path) / 1024
                    quality = 10
                else:
                    quality = 20
                    break
                
                scale_factor -= 0.1
        
        # Final check - if still too large, force it to be under target
        if compressed_size > max_kb:
            # Use very aggressive compression
            cv2.imwrite(output_path, current_image, [cv2.IMWRITE_JPEG_QUALITY, 1])
            compressed_size = os.path.getsize(output_path) / 1024
            quality = 1
        
        print(f"✅ Compressed {os.path.basename(image_path)} to {compressed_size:.1f}KB (quality={quality})")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error compressing {image_path}: {e}")
        return image_path

def crop_passport_1_to_passport_2(passport_1_path: str, output_dir: str) -> str:
    """
    Crop the top 50% of passport_1 image to create a passport_2 image.
    
    Args:
        passport_1_path: Path to the passport_1 image
        output_dir: Directory to save the cropped image
    
    Returns:
        Path to the cropped passport_2 image
    """
    try:
        # Read the image
        image = cv2.imread(passport_1_path)
        if image is None:
            raise ValueError(f"Could not read image: {passport_1_path}")
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Crop top 50% of the image
        cropped_height = height // 2
        cropped_image = image[:cropped_height, :]
        
        # Save the cropped image
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(passport_1_path))[0]}_passport_2_cropped.jpg")
        cv2.imwrite(output_path, cropped_image)
        
        print(f"✂️ Successfully cropped passport_1 to create passport_2: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error cropping passport_1: {e}")
        return None

def create_passport_2_from_passport_1(passport_1_path: str, temp_dir: str) -> tuple:
    """
    Create a passport_2 image by cropping passport_1 and return both the path and OCR data.
    
    Args:
        passport_1_path: Path to the passport_1 image
        temp_dir: Temporary directory for processing
    
    Returns:
        Tuple of (passport_2_path, passport_2_ocr_text)
    """
    try:
        # Crop passport_1 to create passport_2
        passport_2_path = crop_passport_1_to_passport_2(passport_1_path, temp_dir)
        
        if not passport_2_path:
            return None, ""
        
        # Process the cropped passport_2 with OCR
        try:
            from document_processing_pipeline import classify_and_ocr
            
            # Process the cropped passport_2 image
            rotated_path, vision_data, final_label = classify_and_ocr(passport_2_path, temp_dir)
            
            # Get OCR text
            ocr_text = vision_data.get("ocr_text", "")
            
            print(f"📄 Created passport_2 from passport_1 with OCR text length: {len(ocr_text)}")
            return passport_2_path, ocr_text
            
        except ImportError:
            print("⚠️ Could not import classify_and_ocr, skipping OCR for passport_2")
            return passport_2_path, ""
            
    except Exception as e:
        print(f"❌ Error creating passport_2 from passport_1: {e}")
        return None, ""

def resize_image(image_path: str, output_path: str, max_width: int = 1920, max_height: int = 1080) -> str:
    """
    Resize an image to fit within specified dimensions while maintaining aspect ratio.
    
    Args:
        image_path: Path to the input image
        output_path: Path for the output resized image
        max_width: Maximum width
        max_height: Maximum height
    
    Returns:
        Path to the resized image
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Get original dimensions
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale_x = max_width / width
        scale_y = max_height / height
        scale = min(scale_x, scale_y, 1.0)  # Don't upscale
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Save the resized image
        cv2.imwrite(output_path, resized_image)
        
        print(f"📏 Resized {os.path.basename(image_path)} to {new_width}x{new_height}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error resizing {image_path}: {e}")
        return image_path

def enhance_image_quality(image_path: str, output_path: str) -> str:
    """
    Enhance image quality for better OCR results.
    
    Args:
        image_path: Path to the input image
        output_path: Path for the output enhanced image
    
    Returns:
        Path to the enhanced image
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Save the enhanced image
        cv2.imwrite(output_path, thresh)
        
        print(f"✨ Enhanced image quality: {os.path.basename(image_path)}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error enhancing {image_path}: {e}")
        return image_path 