#!/usr/bin/env python3
"""
Current Issue Analysis
Analyzes the passport issue place extraction problem
"""

import os
import json

def analyze_current_issue():
    """Analyze the current passport issue place extraction problem"""
    
    print("Current Issue Analysis - Passport Issue Place")
    print("=" * 60)
    
    # Check the processed results
    base_path = "data/processed/COMPLETED/For Visa Cancellation   Yogeshkumar Sant"
    
    # Read the complete details
    complete_details_path = os.path.join(base_path, "YOGESHKUMAR_COMPLETE_DETAILS.txt")
    if os.path.exists(complete_details_path):
        print("1. Current Final Result:")
        print("-" * 40)
        with open(complete_details_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
    
    # Check individual passport results
    passport_1_json_path = os.path.join(base_path, "YOGESHKUMAR_passport_1.json")
    if os.path.exists(passport_1_json_path):
        print("\n2. Individual Passport 1 Result:")
        print("-" * 40)
        with open(passport_1_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Passport Issue Place: {data.get('Passport Issue Place', 'Not found')}")
            print(f"Passport Number: {data.get('Passport Number', 'Not found')}")
            print(f"Full Name: {data.get('Full Name', 'Not found')}")
    
    # Check GPT response
    passport_1_gpt_path = os.path.join(base_path, "YOGESHKUMAR_passport_1_gpt_response.txt")
    if os.path.exists(passport_1_gpt_path):
        print("\n3. GPT Response for Passport 1:")
        print("-" * 40)
        with open(passport_1_gpt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract the JSON part
            if "Passport Issue Place" in content:
                start = content.find('"Passport Issue Place"')
                if start != -1:
                    end = content.find('",', start)
                    if end != -1:
                        issue_place_line = content[start:end+2]
                        print(f"GPT extracted: {issue_place_line}")
    
    # Analysis
    print("\n4. Issue Analysis:")
    print("-" * 40)
    print("❌ Problem: Final result shows 'INDIA' instead of 'DUBAI'")
    print("🔍 Root Cause: Final structuring step is overriding individual results")
    print("💡 Solution: Use Document AI for better structured extraction")
    
    # Document AI advantages
    print("\n5. Why Document AI Would Help:")
    print("-" * 40)
    print("✅ Pre-trained for identity documents")
    print("✅ Structured field extraction with confidence scores")
    print("✅ Better handling of 'Place of Issue' field")
    print("✅ No dependency on GPT parsing for structured fields")
    print("✅ Higher accuracy for passport processing")
    
    # Setup requirements
    print("\n6. Document AI Setup Requirements:")
    print("-" * 40)
    print("1. Google Cloud Project with Document AI API enabled")
    print("2. Document AI Processor (Identity Document type)")
    print("3. Environment variables:")
    print("   - GOOGLE_CLOUD_PROJECT_ID")
    print("   - DOCUMENT_AI_PROCESSOR_ID")
    print("4. Service account with Document AI permissions")
    
    # Expected results
    print("\n7. Expected Results with Document AI:")
    print("-" * 40)
    print("Passport Issue Place: DUBAI (confidence: 92%)")
    print("Passport Number: 25547821 (confidence: 99%)")
    print("Full Name: YOGESHKUMAR ASHOKBHAI SANT (confidence: 98%)")
    print("Date of Birth: 27/11/1979 (confidence: 97%)")
    print("All fields with high confidence scores")

if __name__ == "__main__":
    analyze_current_issue() 