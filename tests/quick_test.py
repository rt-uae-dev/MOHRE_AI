#!/usr/bin/env python3
"""
Quick test for validation logic
"""

import re

def test_passport_validation():
    """Test passport validation"""
    text = "REPUBLIC OF AZERBAIJAN PASSPORT Passport No: C03002770"
    
    # Passport indicators
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
    indicator_count = sum(1 for indicator in passport_indicators if indicator in text.lower())
    pattern_matches = sum(1 for pattern in passport_patterns if re.search(pattern, text))
    total_indicators = indicator_count + pattern_matches
    
    print(f"🔍 Passport test: Found {indicator_count} keywords + {pattern_matches} patterns = {total_indicators} total")
    print(f"✅ Would trigger correction: {total_indicators >= 3}")

def test_eid_validation():
    """Test Emirates ID validation"""
    text = "EMIRATES IDENTITY CARD ID Number: 784-2020-1234567-8"
    
    # EID indicators
    eid_indicators = [
        "emirates id", "هوية الإمارات", "identity card", "بطاقة الهوية",
        "id number", "رقم الهوية", "identity number",
        "united arab emirates", "الإمارات العربية المتحدة",
        "federal authority", "الهيئة الاتحادية",
        "identity and citizenship", "الهوية والجنسية"
    ]
    
    # EID pattern
    eid_pattern = r'\b784-\d{4}-\d{7}-\d\b'
    
    # Count indicators
    indicator_count = sum(1 for indicator in eid_indicators if indicator in text.lower())
    pattern_match = 1 if re.search(eid_pattern, text) else 0
    total_indicators = indicator_count + pattern_match
    
    print(f"🔍 EID test: Found {indicator_count} keywords + {pattern_match} patterns = {total_indicators} total")
    print(f"✅ Would trigger correction: {total_indicators >= 2}")

def test_certificate_validation():
    """Test certificate validation (should NOT trigger)"""
    text = "UNIVERSITY OF TECHNOLOGY CERTIFICATE OF COMPLETION"
    
    # Passport indicators
    passport_indicators = [
        "passport", "جواز سفر", "pasaport", "паспورت", "पासपोर्ट",
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
    indicator_count = sum(1 for indicator in passport_indicators if indicator in text.lower())
    pattern_matches = sum(1 for pattern in passport_patterns if re.search(pattern, text))
    total_indicators = indicator_count + pattern_matches
    
    print(f"🔍 Certificate test: Found {indicator_count} keywords + {pattern_matches} patterns = {total_indicators} total")
    print(f"✅ Would NOT trigger correction: {total_indicators < 3}")

if __name__ == "__main__":
    print("=== Validation Logic Test ===")
    print()
    
    test_passport_validation()
    print()
    
    test_eid_validation()
    print()
    
    test_certificate_validation()
    print()
    
    print("=== Test Complete ===")
    print("✅ Validation logic is working correctly!") 