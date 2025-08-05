#!/usr/bin/env python3
"""
Simple EasyOCR Test on Passport - Both Pages
"""

import os
import easyocr
import cv2
import numpy as np
from pdf2image import convert_from_path

def test_easyocr_on_passport():
    """Test EasyOCR on both pages of the passport"""
    
    # Try original PDF
    pdf_path = "data/raw/downloads/For Visa Cancellation   Yogeshkumar Sant/Yogeshkumar Sant  Passport copy.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    print(f"📄 Converting PDF to images...")
    images = convert_from_path(pdf_path)
    
    # Initialize EasyOCR reader
    print("🔄 Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Test both pages
    for page_num, image in enumerate(images, 1):
        print(f"\n{'='*60}")
        print(f"🔍 Testing Page {page_num}")
        print(f"{'='*60}")
        
        # Save page as image
        image_path = f"data/temp/passport_page_{page_num}.jpg"
        image.save(image_path, 'JPEG', quality=95)
        
        try:
            # Test on this page
            results = reader.readtext(image_path)
            
            print(f"📝 Found {len(results)} text blocks")
            print(f"📄 All detected text:")
            print("-" * 40)
            
            all_text = []
            for i, (bbox, text, prob) in enumerate(results):
                if prob > 0.3:  # Lower confidence threshold
                    print(f"{i+1:2d}. [{prob:.2f}] {text}")
                    all_text.append(text)
            
            full_text = " ".join(all_text)
            print(f"\n📝 Total text length: {len(full_text)} characters")
            
            # Look for specific fields
            print(f"\n🔍 Looking for specific fields on page {page_num}:")
            
            # Check for issue place related text - look more broadly
            issue_keywords = ['place', 'issue', 'authority', 'issued', 'duba', 'dubai', 'mumbai', 'delhi', 'office', 'passport office']
            found_issue = []
            
            # Check each text block individually
            for text in all_text:
                text_lower = text.lower()
                for keyword in issue_keywords:
                    if keyword in text_lower:
                        found_issue.append(text)
                        break
            
            if found_issue:
                print(f"📍 Issue place candidates: {found_issue}")
            else:
                print(f"❌ No issue place fields found")
            
            # Check for passport number
            if '25547821' in full_text or 'z5547821' in full_text.lower():
                print("✅ Found passport number")
            else:
                print("❌ Passport number not found")
            
            # Check for name
            if 'yogeshkumar' in full_text.lower() or 'yogesh' in full_text.lower():
                print("✅ Found name")
            else:
                print("❌ Name not found")
            
            # Check for dates
            if '01/08/2021' in full_text:
                print("✅ Found issue date: 01/08/2021")
            if '31/07/2051' in full_text:
                print("✅ Found expiry date: 31/07/2051")
                
        except Exception as e:
            print(f"❌ EasyOCR failed on page {page_num}: {e}")

if __name__ == "__main__":
    test_easyocr_on_passport() 