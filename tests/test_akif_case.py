#!/usr/bin/env python3
"""
Test simulating Akif's case with validation logic
"""

import re

def validate_passport_in_certificate(ocr_text: str) -> bool:
    """Check if document might be a passport"""
    if not ocr_text:
        return False
    
    ocr_lower = ocr_text.lower()
    
    # Passport-specific keywords
    passport_indicators = [
        "passport", "جواز سفر", "pasaport", "паспорт", "पासपोर्ट",
        "passport no", "passport number", "رقم الجواز",
        "date of birth", "place of birth", "nationality",
        "given name", "surname", "family name",
        "date of issue", "place of issue", "authority",
        "expiry date", "valid until", "expires",
        "republic of", "ministry of", "government of",
        "machine readable", "mrz", "icao"
    ]
    
    # Passport patterns
    passport_patterns = [
        r'\b[A-Z]\d{8}\b',  # Single letter + 8 digits
        r'\b[A-Z]{2}\d{7}\b',  # 2 letters + 7 digits  
        r'\b\d{9}\b',  # 9 digits
        r'\b[A-Z]\d{7}\b',  # Single letter + 7 digits
    ]
    
    # Count indicators
    indicator_count = sum(1 for indicator in passport_indicators if indicator in ocr_lower)
    pattern_matches = sum(1 for pattern in passport_patterns if re.search(pattern, ocr_text))
    total_indicators = indicator_count + pattern_matches
    
    print(f"🔍 Found {indicator_count} keywords + {pattern_matches} patterns = {total_indicators} total")
    
    return total_indicators >= 3

def validate_document_misclassification(resnet_label: str, ocr_text: str) -> str:
    """Validate and correct misclassifications"""
    if not ocr_text:
        return resnet_label
    
    print(f"🔍 Running validation for '{resnet_label}'...")
    
    if resnet_label == "certificate":
        if validate_passport_in_certificate(ocr_text):
            print("⚠️ Strong passport indicators found - switching to passport_1")
            return "passport_1"
        else:
            print("✅ Certificate classification confirmed")
    
    return resnet_label

# Simulate Akif's case
print("=== Simulating Akif's Case ===")
print()

# Step 1: ResNet18 misclassifies as certificate
resnet_classification = "certificate"
print(f"🤖 ResNet18 classification: {resnet_classification}")

# Step 2: OCR text from Akif's passport
akif_ocr_text = """
REPUBLIC OF AZERBAIJAN
PASSPORT
Passport No: C03002770
Surname: MANAFOV
Given Names: AKIF ANAR OĞLU
Nationality: AZERBAIJAN
Date of Birth: 25 DEC 2001
Place of Birth: AZERBAIJAN
Date of Issue: 15 JAN 2020
Place of Issue: MINISTRY OF INTERNAL AFFAIRS
Authority: REPUBLIC OF AZERBAIJAN
Date of Expiry: 15 JAN 2030
"""

print("📄 OCR Text extracted:")
print(akif_ocr_text)
print()

# Step 3: Apply validation logic
final_classification = validate_document_misclassification(resnet_classification, akif_ocr_text)

print(f"✅ Final classification: {final_classification}")
print()

if final_classification == "passport_1":
    print("🎉 SUCCESS: Validation logic corrected the misclassification!")
    print("   Akif's document will now be processed as a passport instead of certificate.")
else:
    print("❌ FAILED: Validation logic did not correct the misclassification.")

print()
print("=== Test Complete ===") 