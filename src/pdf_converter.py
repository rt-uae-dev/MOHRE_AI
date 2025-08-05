#!/usr/bin/env python3
"""
PDF conversion utilities for the MOHRE document processing pipeline.
"""

import os
from pdf2image import convert_from_path

def convert_pdf_to_jpg(pdf_path: str, temp_dir: str) -> list:
    """
    Convert PDF to JPG images.
    
    Args:
        pdf_path: Path to the PDF file
        temp_dir: Directory to save the converted images
    
    Returns:
        List of paths to the converted JPG images
    """
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        image_paths = []
        
        for i, image in enumerate(images):
            # Create output filename
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            jpg_filename = f"{base_name}_page{i+1}.jpg"
            jpg_path = os.path.join(temp_dir, jpg_filename)
            
            # Save the image
            image.save(jpg_path, "JPEG", quality=95)
            image_paths.append(jpg_path)
            
            print(f"📄 Converted page {i+1}: {jpg_filename}")
            
        return image_paths
        
    except Exception as e:
        print(f"❌ Error converting PDF {pdf_path}: {e}")
        return []

def convert_pdf_to_jpg_with_quality(pdf_path: str, temp_dir: str, quality: int = 95) -> list:
    """
    Convert PDF to JPG images with specified quality.
    
    Args:
        pdf_path: Path to the PDF file
        temp_dir: Directory to save the converted images
        quality: JPEG quality (1-100)
    
    Returns:
        List of paths to the converted JPG images
    """
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        image_paths = []
        
        for i, image in enumerate(images):
            # Create output filename
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            jpg_filename = f"{base_name}_page{i+1}.jpg"
            jpg_path = os.path.join(temp_dir, jpg_filename)
            
            # Save the image with specified quality
            image.save(jpg_path, "JPEG", quality=quality)
            image_paths.append(jpg_path)
            
            print(f"📄 Converted page {i+1}: {jpg_filename} (quality={quality})")
            
        return image_paths
        
    except Exception as e:
        print(f"❌ Error converting PDF {pdf_path}: {e}")
        return []

def get_pdf_page_count(pdf_path: str) -> int:
    """
    Get the number of pages in a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Number of pages in the PDF
    """
    try:
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        # Get page count by trying to access all pages
        page_count = 0
        while True:
            try:
                images = convert_from_path(pdf_path, first_page=page_count + 1, last_page=page_count + 1)
                if not images:
                    break
                page_count += 1
            except:
                break
        return page_count
    except Exception as e:
        print(f"❌ Error getting page count for {pdf_path}: {e}")
        return 0 