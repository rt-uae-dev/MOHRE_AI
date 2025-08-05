#!/usr/bin/env python3
"""
Test script for document misclassification validation logic
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the validation functions directly
def validate_passport_in_certificate(ocr_text: str) -> bool:
    """
    Check if a document classified as 'certificate' might actually be a passport
    by looking for passport-specific keywords in the OCR text.
    """
    if not ocr_text:
        return False
    
    ocr_lower = ocr_text.lower()
    
    # Passport-specific keywords to look for
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
    
    # Check for passport number patterns (common formats)
    import re
    passport_patterns = [
        r'\b[A-Z]\d{8}\b',  # Single letter + 8 digits
        r'\b[A-Z]{2}\d{7}\b',  # 2 letters + 7 digits  
        r'\b\d{9}\b',  # 9 digits
        r'\b[A-Z]\d{7}\b',  # Single letter + 7 digits
    ]
    
    # Count how many passport indicators are found
    indicator_count = sum(1 for indicator in passport_indicators if indicator in ocr_lower)
    
    # Check for passport number patterns
    pattern_matches = 0
    for pattern in passport_patterns:
        if re.search(pattern, ocr_text):
            pattern_matches += 1
    
    # If we find multiple passport indicators, it's likely a passport
    total_indicators = indicator_count + pattern_matches
    
    print(f"🔍 Passport validation: Found {indicator_count} keywords + {pattern_matches} patterns = {total_indicators} total indicators")
    
    # Return True if we have strong passport indicators
    return total_indicators >= 3

def validate_emirates_id_in_certificate(ocr_text: str) -> bool:
    """
    Check if a document classified as 'certificate' might actually be an Emirates ID
    by looking for EID-specific keywords in the OCR text.
    """
    if not ocr_text:
        return False
    
    ocr_lower = ocr_text.lower()
    
    # Emirates ID specific keywords
    eid_indicators = [
        "emirates id", "هوية الإمارات", "identity card", "بطاقة الهوية",
        "id number", "رقم الهوية", "identity number",
        "united arab emirates", "الإمارات العربية المتحدة",
        "federal authority", "الهيئة الاتحادية",
        "identity and citizenship", "الهوية والجنسية"
    ]
    
    # Check for EID number pattern (784-XXXX-XXXXXXX-X)
    import re
    eid_pattern = r'\b784-\d{4}-\d{7}-\d\b'
    
    indicator_count = sum(1 for indicator in eid_indicators if indicator in ocr_lower)
    pattern_match = 1 if re.search(eid_pattern, ocr_text) else 0
    
    total_indicators = indicator_count + pattern_match
    
    print(f"🔍 Emirates ID validation: Found {indicator_count} keywords + {pattern_match} patterns = {total_indicators} total indicators")
    
    return total_indicators >= 2

def validate_certificate_in_emirates_id(ocr_text: str) -> bool:
    """
    Check if a document classified as 'emirates_id' might actually be a certificate
    by looking for certificate-specific keywords in the OCR text.
    """
    if not ocr_text:
        return False
    
    ocr_lower = ocr_text.lower()
    
    # Certificate specific keywords
    certificate_indicators = [
        "certificate", "شهادة", "degree", "diploma", "bachelor", "master", "phd",
        "engineering", "technology", "university", "college", "institute",
        "graduation", "academic", "education", "qualification",
        "electrical", "mechanical", "civil", "computer", "software",
        "beirut arab university", "bau", "university of", "institute of",
        "this is to certify", "certify that", "has successfully completed",
        "awarded", "conferred", "degree of", "in the field of"
    ]
    
    # Check for certificate number patterns (often longer numbers)
    import re
    certificate_patterns = [
        r'\b\d{10,15}\b',  # 10-15 digit numbers (common in certificates)
        r'\b\d{4}-\d{4}-\d{4}\b',  # Format like 2020-2024-1234
        r'\bcertificate\s+no[.:]\s*\d+\b',  # Certificate number
        r'\breg[.:]\s*no[.:]\s*\d+\b'  # Registration number
    ]
    
    indicator_count = sum(1 for indicator in certificate_indicators if indicator in ocr_lower)
    pattern_matches = sum(1 for pattern in certificate_patterns if re.search(pattern, ocr_text, re.IGNORECASE))
    
    total_indicators = indicator_count + pattern_matches
    
    print(f"🔍 Certificate validation: Found {indicator_count} keywords + {pattern_matches} patterns = {total_indicators} total indicators")
    
    return total_indicators >= 3

def validate_attestation_in_certificate(ocr_text: str) -> bool:
    """
    Check if a document classified as 'certificate' might actually be a certificate attestation
    by looking for attestation-specific keywords in the OCR text.
    """
    if not ocr_text:
        return False
    
    ocr_lower = ocr_text.lower()
    
    # Attestation specific keywords
    attestation_indicators = [
        "attestation", "تصديق", "attested", "attest", "attesting",
        "ministry of foreign affairs", "وزارة الخارجية",
        "ministry of education", "وزارة التربية والتعليم",
        "embassy", "سفارة", "consulate", "قنصلية",
        "apostille", "apostilla", "legalization", "تصديق قانوني",
        "authenticate", "authentication", "مصادقة",
        "stamp", "ختم", "seal", "ختم رسمي",
        "certified", "certification", "تصديق رسمي",
        "notary", "notarial", "كاتب العدل",
        "foreign affairs", "external affairs", "الشؤون الخارجية",
        "uae attestation", "emirates attestation", "تصديق الإمارات",
        "mofa", "moi", "ministry of interior", "وزارة الداخلية"
    ]
    
    # Check for attestation number patterns
    import re
    attestation_patterns = [
        r'\battestation\s+no[.:]\s*\d+\b',  # Attestation number
        r'\b\d{4}-\d{4}-\d{4}\b',  # Format like 2020-2024-1234
        r'\b\d{10,15}\b',  # 10-15 digit numbers
        r'\bstamp\s+no[.:]\s*\d+\b',  # Stamp number
        r'\bseal\s+no[.:]\s*\d+\b'  # Seal number
    ]
    
    indicator_count = sum(1 for indicator in attestation_indicators if indicator in ocr_lower)
    pattern_matches = sum(1 for pattern in attestation_patterns if re.search(pattern, ocr_text, re.IGNORECASE))
    
    total_indicators = indicator_count + pattern_matches
    
    print(f"🔍 Attestation validation: Found {indicator_count} keywords + {pattern_matches} patterns = {total_indicators} total indicators")
    
    return total_indicators >= 3

def validate_document_misclassification(resnet_label: str, ocr_text: str) -> str:
    """
    Validate ResNet classification and correct obvious misclassifications
    based on OCR text content.
    """
    if not ocr_text:
        return resnet_label
    
    print(f"🔍 Running misclassification validation for '{resnet_label}'...")
    
    # Check for passport misclassified as certificate
    if resnet_label == "certificate":
        if validate_passport_in_certificate(ocr_text):
            print("⚠️ Strong passport indicators found in 'certificate' - switching to passport_1")
            return "passport_1"
        elif validate_emirates_id_in_certificate(ocr_text):
            print("⚠️ Strong Emirates ID indicators found in 'certificate' - switching to emirates_id")
            return "emirates_id"
        elif validate_attestation_in_certificate(ocr_text):
            print("⚠️ Strong attestation indicators found in 'certificate' - switching to certificate_attestation")
            return "certificate_attestation"
        else:
            print("✅ Certificate classification confirmed - no conflicting indicators found")
    
    # Check for certificate misclassified as emirates_id
    elif resnet_label == "emirates_id":
        if validate_certificate_in_emirates_id(ocr_text):
            print("⚠️ Strong certificate indicators found in 'emirates_id' - switching to certificate")
            return "certificate"
        else:
            print("✅ Emirates ID classification confirmed - no conflicting indicators found")
    
    # Check for passport misclassified as unknown
    elif resnet_label == "unknown":
        if validate_passport_in_certificate(ocr_text):
            print("⚠️ Passport indicators found in 'unknown' - switching to passport_1")
            return "passport_1"
        elif validate_emirates_id_in_certificate(ocr_text):
            print("⚠️ Emirates ID indicators found in 'unknown' - switching to emirates_id")
            return "emirates_id"
    
    return resnet_label

def test_passport_validation():
    """Test passport validation with sample OCR text"""
    
    # Sample passport OCR text (similar to what might be extracted from Akif's passport)
    passport_ocr = """
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
    
    print("=== Testing Passport Validation ===")
    print("Sample OCR Text:")
    print(passport_ocr)
    print()
    
    # Test passport validation
    is_passport = validate_passport_in_certificate(passport_ocr)
    print(f"Result: {'PASSPORT' if is_passport else 'NOT PASSPORT'}")
    print()
    
    # Test misclassification validation
    corrected_label = validate_document_misclassification("certificate", passport_ocr)
    print(f"ResNet classified as 'certificate' -> Corrected to: '{corrected_label}'")
    print()

def test_emirates_id_validation():
    """Test Emirates ID validation with sample OCR text"""
    
    # Sample Emirates ID OCR text
    eid_ocr = """
    UNITED ARAB EMIRATES
    EMIRATES IDENTITY CARD
    هوية الإمارات
    ID Number: 784-2020-1234567-8
    Name: AHMED MOHAMMED ALI
    Date of Birth: 15/03/1985
    Nationality: UAE
    Federal Authority for Identity and Citizenship
    الهيئة الاتحادية للهوية والجنسية
    """
    
    print("=== Testing Emirates ID Validation ===")
    print("Sample OCR Text:")
    print(eid_ocr)
    print()
    
    # Test Emirates ID validation
    is_eid = validate_emirates_id_in_certificate(eid_ocr)
    print(f"Result: {'EMIRATES ID' if is_eid else 'NOT EMIRATES ID'}")
    print()
    
    # Test misclassification validation
    corrected_label = validate_document_misclassification("certificate", eid_ocr)
    print(f"ResNet classified as 'certificate' -> Corrected to: '{corrected_label}'")
    print()

def test_certificate_validation():
    """Test certificate validation with sample OCR text"""
    
    # Sample certificate OCR text (should NOT be classified as passport)
    certificate_ocr = """
    UNIVERSITY OF TECHNOLOGY
    CERTIFICATE OF COMPLETION
    This is to certify that
    JOHN DOE
    has successfully completed the course in
    COMPUTER SCIENCE
    Date: 15/06/2023
    Certificate Number: CS-2023-001
    """
    
    print("=== Testing Certificate Validation ===")
    print("Sample OCR Text:")
    print(certificate_ocr)
    print()
    
    # Test passport validation (should return False)
    is_passport = validate_passport_in_certificate(certificate_ocr)
    print(f"Passport validation result: {'PASSPORT' if is_passport else 'NOT PASSPORT'}")
    
    # Test Emirates ID validation (should return False)
    is_eid = validate_emirates_id_in_certificate(certificate_ocr)
    print(f"Emirates ID validation result: {'EMIRATES ID' if is_eid else 'NOT EMIRATES ID'}")
    print()
    
    # Test misclassification validation (should remain certificate)
    corrected_label = validate_document_misclassification("certificate", certificate_ocr)
    print(f"ResNet classified as 'certificate' -> Corrected to: '{corrected_label}'")
    
    # Test certificate misclassified as emirates_id (should switch to certificate)
    corrected_label = validate_document_misclassification("emirates_id", certificate_ocr)
    print(f"ResNet classified as 'emirates_id' -> Corrected to: '{corrected_label}'")
    
    # Test attestation validation
    attestation_ocr = """
    MINISTRY OF FOREIGN AFFAIRS
    ATTESTATION CERTIFICATE
    تصديق رسمي
    This is to certify that the attached document
    has been duly attested and authenticated
    Attestation Number: 2024-1234-5678
    Stamp Number: ST-2024-001
    Ministry of Foreign Affairs
    United Arab Emirates
    """
    
    # Test certificate misclassified as certificate_attestation (should switch to certificate_attestation)
    corrected_label = validate_document_misclassification("certificate", attestation_ocr)
    print(f"ResNet classified as 'certificate' -> Corrected to: '{corrected_label}'")
    print()

if __name__ == "__main__":
    test_passport_validation()
    test_emirates_id_validation()
    test_certificate_validation()
    
    print("=== Validation Tests Complete ===")
    print("This validation logic should help catch misclassifications like Akif's passport being classified as certificate.") 