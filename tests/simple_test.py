#!/usr/bin/env python3
"""
Simple test for validation logic
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
    
    # Passport number patterns
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
    
    print(f"Found {indicator_count} keywords + {pattern_matches} patterns = {total_indicators} total")
    
    return total_indicators >= 3

# Test with Akif's passport-like text
test_text = """
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

print("Testing passport validation...")
result = validate_passport_in_certificate(test_text)
print(f"Result: {'PASSPORT' if result else 'NOT PASSPORT'}")

# Test with certificate text
cert_text = """
UNIVERSITY OF TECHNOLOGY
CERTIFICATE OF COMPLETION
This is to certify that
JOHN DOE
has successfully completed the course in
COMPUTER SCIENCE
Date: 15/06/2023
Certificate Number: CS-2023-001
"""

print("\nTesting certificate validation...")
result2 = validate_passport_in_certificate(cert_text)
print(f"Result: {'PASSPORT' if result2 else 'NOT PASSPORT'}")

print("\n✅ Validation logic test complete!") 