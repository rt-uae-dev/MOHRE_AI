import os
from PIL import Image
from yolo_crop_ocr_pipeline import run_yolo_crop, run_enhanced_ocr
from image_rotation_utils import rotate_image_if_needed
from resnet18_classifier import (
    classify_image_resnet, 
    classify_image_from_text, 
    classify_image_with_gemini_vision,
    check_image_orientation,
    auto_rotate_image_if_needed
)

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

def validate_attestation_label_detection(yolo_labels: list, ocr_text: str) -> bool:
    """
    Check if YOLO detected attestation_label and validate with OCR text.
    If YOLO labels it as attestation_label, it should be classified as certificate_attestation.
    """
    if not yolo_labels:
        return False
    
    # Check if YOLO detected attestation_label
    attestation_yolo_indicators = ["attestation_label", "attestation", "attestation_certificate"]
    yolo_detected_attestation = any(label.lower() in [l.lower() for l in yolo_labels] for label in attestation_yolo_indicators)
    
    if yolo_detected_attestation:
        print(f"🔍 YOLO detected attestation labels: {yolo_labels}")
        # Additional validation with OCR text
        if validate_attestation_in_certificate(ocr_text):
            print("✅ YOLO attestation detection confirmed by OCR text")
            return True
        else:
            print("⚠️ YOLO detected attestation but OCR text doesn't confirm")
            return False
    
    return False

def validate_document_misclassification(resnet_label: str, ocr_text: str, yolo_labels: list = None) -> str:
    """
    Validate ResNet classification and correct obvious misclassifications
    based on OCR text content and YOLO detections.
    """
    if not ocr_text:
        return resnet_label
    
    print(f"🔍 Running misclassification validation for '{resnet_label}'...")
    
    # PRIORITY 1: Check for YOLO attestation_label detection
    if yolo_labels and validate_attestation_label_detection(yolo_labels, ocr_text):
        print("🎯 YOLO detected attestation_label - switching to attestation_label")
        return "attestation_label"
    
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
        elif validate_attestation_in_certificate(ocr_text):
            print("⚠️ Strong attestation indicators found in 'emirates_id' - switching to certificate_attestation")
            return "certificate_attestation"
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
        elif validate_attestation_in_certificate(ocr_text):
            print("⚠️ Attestation indicators found in 'unknown' - switching to certificate_attestation")
            return "certificate_attestation"
    
    return resnet_label

def classify_and_ocr(image_path: str, temp_dir: str):
    """
    Full image pipeline: crop with YOLO, OCR with Document AI (primary) + Google Vision (fallback),
    rotate if needed, classify with ResNet, fallback to GPT if needed.
    Returns:
        - rotated image path
        - vision_data: {ocr_text, angle, labels, document_type, extracted_fields, confidence}
        - final label (resnet or GPT fallback)
    """
    print(f"\n🔁 Processing: {os.path.basename(image_path)}")

    # Step 1: Crop the image
    cropped_path = run_yolo_crop(image_path, temp_dir)
    print(f"✂️ Cropped to: {cropped_path}")

    # Step 2: Enhanced OCR (Document AI + Google Vision fallback)
    vision_data = run_enhanced_ocr(cropped_path)
    ocr_text = vision_data.get("ocr_text", "").strip()
    ocr_method = vision_data.get("ocr_method", "unknown")
    document_type = vision_data.get("document_type", "unknown")
    confidence = vision_data.get("confidence", 0.0)
    extracted_fields = vision_data.get("extracted_fields", {})
    
    print(f"📄 OCR Method: {ocr_method}")
    print(f"📋 Document Type: {document_type}")
    print(f"🎯 Confidence: {confidence:.2f}")
    
    # Show extracted fields if any
    if extracted_fields:
        print(f"📝 Extracted Fields:")
        for field_name, field_value in extracted_fields.items():
            print(f"   - {field_name}: {field_value}")

    # Step 3: Classify with ResNet
    resnet_label = classify_image_resnet(cropped_path)
    print(f"🏷️ ResNet label: {resnet_label}")

    # Step 4: Validation logic for certificate misclassification
    yolo_labels = vision_data.get("labels", [])
    final_label = validate_document_misclassification(resnet_label, ocr_text, yolo_labels)

    # Step 5: Enhanced fallback to Gemini Vision for unknown documents only
    if final_label in ["unknown", "certificate"]:
        print(f"🔍 Using Gemini Vision fallback for {final_label} classification...")
        gemini_vision_label = classify_image_with_gemini_vision(cropped_path)
        if gemini_vision_label != "unknown":
            print(f"🤖 Gemini Vision classified as: {gemini_vision_label}")
            final_label = gemini_vision_label
        elif ocr_text:
            # Fallback to text-based classification if vision fails
            gemini_text_label = classify_image_from_text(ocr_text)
            print(f"🤖 Gemini text fallback label: {gemini_text_label}")
            final_label = gemini_text_label

    # Step 6: Check orientation for passport photos, personal photos, certificates, and attestation labels
    orientation_analysis = None
    if final_label in ["passport_photo", "personal_photo", "certificate", "certificate_attestation", "attestation_label"]:
        print(f"🔍 Checking orientation for {final_label}...")
        orientation_analysis = check_image_orientation(cropped_path)
        
        if not orientation_analysis.get("orientation_correct", True):
            print(f"⚠️ Orientation issues detected: {orientation_analysis.get('issues', [])}")
            print(f"💡 Recommendations: {orientation_analysis.get('recommendations', [])}")
            
            # Auto-rotate if needed
            corrected_path = auto_rotate_image_if_needed(cropped_path, orientation_analysis)
            if corrected_path != cropped_path:
                cropped_path = corrected_path
                print(f"✅ Image auto-corrected and saved as: {corrected_path}")

    # Step 7: Rotate if needed (existing logic)
    angle_data = vision_data.get("angle", [])
    rotated_path = rotate_image_if_needed(cropped_path, angle_data, final_label)

    # Add orientation analysis to vision_data if available
    if orientation_analysis:
        vision_data["orientation_analysis"] = orientation_analysis
    
    return rotated_path, vision_data, final_label
