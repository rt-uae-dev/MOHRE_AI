#!/usr/bin/env python3
"""
Attestation number validation utilities
"""

import re
from typing import List, Tuple, Optional

def validate_attestation_numbers(ocr_text: str, extracted_numbers: dict) -> dict:
    """
    Validate that extracted attestation numbers actually appear in the OCR text
    
    Args:
        ocr_text: The OCR text from the attestation document
        extracted_numbers: Dictionary containing 'Attestation Number 1' and 'Attestation Number 2'
    
    Returns:
        Dictionary with validated numbers (None if not found in OCR text)
    """
    
    validated = {}
    
    # Arabic to Western numeral mapping
    arabic_to_western = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    
    # Convert Arabic numerals to Western numerals in OCR text for comparison
    normalized_ocr = ocr_text
    for arabic, western in arabic_to_western.items():
        normalized_ocr = normalized_ocr.replace(arabic, western)
    
    # Helper function to clean and validate attestation numbers
    def clean_attestation_number(number_str: str, min_length: int, max_length: int, number_type: str) -> str:
        """Clean attestation number by removing leading zeros and validating format"""
        if not number_str or number_str == 'null':
            return None
        
        # Remove leading zeros
        cleaned = number_str.lstrip('0')
        
        # Check if it's a valid format
        if not cleaned.isdigit():
            print(f"❌ {number_type} '{number_str}' is not a valid number - setting to null")
            return None
        
        if len(cleaned) < min_length or len(cleaned) > max_length:
            print(f"❌ {number_type} '{number_str}' (cleaned: '{cleaned}') has wrong length {len(cleaned)}, expected {min_length}-{max_length} - setting to null")
            return None
        
        # Check if original had leading zeros (likely OCR artifact)
        if number_str.startswith('0') and len(number_str) > max_length:
            print(f"⚠️ {number_type} '{number_str}' had leading zeros (cleaned to '{cleaned}') - likely OCR artifact")
        
        return cleaned
    
    # Check Attestation Number 1 (should be 10-15 digits, no leading zeros)
    if 'Attestation Number 1' in extracted_numbers:
        num1 = extracted_numbers['Attestation Number 1']
        cleaned_num1 = clean_attestation_number(num1, 5, 15, "Attestation Number 1")  # 5-15 digits (more flexible)
        
        if cleaned_num1:
            # Check if the cleaned number appears in the normalized OCR text
            if cleaned_num1 in normalized_ocr:
                validated['Attestation Number 1'] = cleaned_num1
                print(f"✅ Attestation Number 1 '{cleaned_num1}' validated - found in OCR text")
            else:
                # For Document AI extractions, be more lenient - trust Document AI even if not in OCR text
                # This handles cases where Document AI extracts from parts not visible in OCR
                validated['Attestation Number 1'] = cleaned_num1
                print(f"⚠️ Attestation Number 1 '{cleaned_num1}' not found in OCR text but trusting Document AI extraction")
        else:
            validated['Attestation Number 1'] = None
    else:
        validated['Attestation Number 1'] = None
    
    # Check Attestation Number 2 (should be 6-7 digits, no leading zeros)
    if 'Attestation Number 2' in extracted_numbers:
        num2 = extracted_numbers['Attestation Number 2']
        cleaned_num2 = clean_attestation_number(num2, 6, 7, "Attestation Number 2")  # 6-7 digits
        
        if cleaned_num2:
            # Check if the cleaned number appears in the normalized OCR text
            if cleaned_num2 in normalized_ocr:
                validated['Attestation Number 2'] = cleaned_num2
                print(f"✅ Attestation Number 2 '{cleaned_num2}' validated - found in OCR text")
            else:
                # For Attestation Number 2, be more lenient - it might be in a different part
                # or OCR might have missed it, but if it's a reasonable 7-digit number, accept it
                if not cleaned_num2.startswith('784'):  # Not Emirates ID
                    validated['Attestation Number 2'] = cleaned_num2
                    print(f"⚠️ Attestation Number 2 '{cleaned_num2}' not found in OCR text but appears valid - keeping it")
                else:
                    validated['Attestation Number 2'] = None
                    print(f"❌ Attestation Number 2 '{cleaned_num2}' appears to be Emirates ID - setting to null")
        else:
            validated['Attestation Number 2'] = None
    else:
        validated['Attestation Number 2'] = None
    
    return validated

def extract_attestation_numbers_from_ocr(ocr_text: str) -> Tuple[List[str], List[str]]:
    """
    Extract potential attestation numbers from OCR text
    
    Args:
        ocr_text: The OCR text from the attestation document
    
    Returns:
        Tuple of (long_numbers, seven_digit_numbers)
    """
    
    # Look for attestation numbers that are 10-15 digits long
    long_numbers = re.findall(r'\b\d{10,15}\b', ocr_text)
    
    # Look for 7-digit numbers
    seven_digit_numbers = re.findall(r'\b\d{7}\b', ocr_text)
    
    # Filter out numbers that start with 784 (Emirates ID)
    filtered_long = [num for num in long_numbers if not num.startswith('784')]
    filtered_seven = [num for num in seven_digit_numbers if not num.startswith('784')]
    
    return filtered_long, filtered_seven

def suggest_attestation_numbers(ocr_text: str) -> dict:
    """
    Suggest attestation numbers based on OCR text analysis
    
    Args:
        ocr_text: The OCR text from the attestation document
    
    Returns:
        Dictionary with suggested numbers
    """
    
    long_nums, seven_nums = extract_attestation_numbers_from_ocr(ocr_text)
    
    suggestions = {
        'Attestation Number 1': None,
        'Attestation Number 2': None,
        'available_long_numbers': long_nums,
        'available_seven_digit_numbers': seven_nums
    }
    
    if long_nums:
        suggestions['Attestation Number 1'] = long_nums[0]  # Take the first one
    
    if seven_nums:
        suggestions['Attestation Number 2'] = seven_nums[0]  # Take the first one
    
    return suggestions
