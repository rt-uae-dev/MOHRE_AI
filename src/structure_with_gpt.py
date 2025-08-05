import openai
import os
import json
from attestation_utils import validate_attestation_numbers

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def structure_with_gpt(
    passport_ocr_1: str,
    passport_ocr_2: str,
    emirates_id_ocr: str,
    emirates_id_2_ocr: str,
    employee_info: str,
    certificate_ocr: str,
    salary_data: dict,
    email_text: str,
    resnet_label: str,
    google_metadata: dict
) -> dict:
    prompt = f"""
You are an expert MOHRE document structuring assistant.
Use the provided OCR text, email body, classification labels, salary data, and Document AI extracted fields to generate a clean JSON structure.

IMPORTANT: Prioritize Document AI extracted fields over raw OCR text for accuracy.
CRITICAL: If Document AI has extracted specific fields (like passport numbers, issue places, etc.), use those values directly.

IMPORTANT: Prioritize data from Emirates ID (EID) and Passport over all other sources for accuracy.

CRITICAL RULE: Passport Issue Place must be the specific place where the passport was issued. Look for fields like "Place of Issue", "Authority", "जारी करने का स्थान", "Oude and Place" in the passport. If no specific issue place is found, then use the issuing country.

Follow these rules:
1. **Full Name**: 
   - FIRST PRIORITY: Document AI extracted "Full Name" field (if available)
   - SECOND PRIORITY: Passport page 1 - look for "Given Name(s)" or "Name" field
   - THIRD PRIORITY: Emirates ID (EID) - look for "Name:" field (only if passport name is incomplete)
   - FOURTH PRIORITY: Employee info form
   - SPECIAL RULE FOR INDIAN PASSPORTS: If nationality is INDIAN and Father's Name is found (from passport page 2 or employee info), append Father's Name to Full Name (e.g., "YOGESHKUMAR ASHOKBHAI SANT" becomes "YOGESHKUMAR ASHOKBHAI SANT ASHOKBHAI BABURAO")
   - For other nationalities, never change the Full Name.
   - Use EID to supplement any missing parts of the name from passport.

2. **Date of Birth**: 
   - FIRST PRIORITY: Document AI extracted "Date of Birth" field (if available)
   - SECOND PRIORITY: Emirates ID (EID) - look for "Date of Birth" or "تاريخ الميلاد"
   - THIRD PRIORITY: Passport page 1 - look for "Date of Birth" field
   - FOURTH PRIORITY: Employee info form
   - Must be in DD/MM/YYYY or DD-MM-YYYY format. If incomplete (like "11/1979"), try to find complete date from other sources.

3. **Nationality**: 
   - FIRST PRIORITY: Document AI extracted "Nationality" field (if available)
   - SECOND PRIORITY: Passport page 1 - look for "Nationality" field
   - THIRD PRIORITY: Emirates ID (EID)
   - FOURTH PRIORITY: Employee info form

4. **Passport Number**: 
   - FIRST PRIORITY: Document AI extracted "Passport Number" field (if available)
   - SECOND PRIORITY: Passport page 1 - look for "Passport No" or "رقم الجواز"
   - THIRD PRIORITY: Emirates ID (EID)
   - FOURTH PRIORITY: Employee info form

5. **EID Number (Emirates ID Number)**: 
   - FIRST PRIORITY: Document AI extracted "EID Number" field (if available)
   - SECOND PRIORITY: Emirates ID (EID) - look for "ID Number" or "رقم الهوية" (format: 784-XXXX-XXXXXXX-X)
   - THIRD PRIORITY: Employee info form - look for "EID" or "Emirates ID Number"
   - FOURTH PRIORITY: Any other document with EID/Identity number
   - NOTE: EID numbers always start with "784"
   - IMPORTANT: Only EID numbers start with "784"

6. **Identity Number/File Number**: 
   - FIRST PRIORITY: Document AI extracted "Identity Number" field (if available)
   - SECOND PRIORITY: Look for numbers in format "XXX/YYYY/ZZZZZZZ" (e.g., "101/2019/3892898")
   - THIRD PRIORITY: Residence cancellation documents - look for "Identity No" or "رقم الهوية"
   - FOURTH PRIORITY: Any document with file numbers in this format
   - FIFTH PRIORITY: Employee info form or other official documents
   - This is different from EID number - it's a government file reference

7. **U.I.D Number**: 
   - FIRST PRIORITY: Document AI extracted "UID Number" field (if available)
   - SECOND PRIORITY: Look for numbers in format "XXXXXXX" (e.g., "2188862")
   - THIRD PRIORITY: Residence cancellation documents - look for "U.I.D No" or "U.I.D"
   - FOURTH PRIORITY: Any document with U.I.D references
   - FIFTH PRIORITY: Employee info form or other official documents
   - This is different from EID and Identity numbers

8. **Father's Name**: 
   - FIRST PRIORITY: Document AI extracted "Father's Name" field (if available)
   - SECOND PRIORITY: Passport page 2 - look for "Father" or "Guardian" field
   - THIRD PRIORITY: Employee info form

9. **Mother's Name**: 
   - FIRST PRIORITY: Document AI extracted "Mother's Name" field (if available)
   - SECOND PRIORITY: Passport page 2 - look for "Mother" field
   - THIRD PRIORITY: Employee info form
   - Must be one word (first found).

10. **Place of Birth**: 
    - FIRST PRIORITY: Document AI extracted "Place of Birth" field (if available)
    - SECOND PRIORITY: Passport page 1 - look for "Place of Birth" field
    - THIRD PRIORITY: Employee info form
    - FOURTH PRIORITY: Use nationality as fallback
    - CRITICAL: This is where the person was born, NOT where the passport was issued

11. **Passport Issue Place**: 
    - FIRST PRIORITY: Document AI extracted "Passport Issue Place" field (if available)
    - SECOND PRIORITY: Passport page 1 or 2 - look for "Place of Issue", "Authority", "जारी करने का स्थान", "Oude and Place", or similar fields
    - THIRD PRIORITY: Employee info form
    - FOURTH PRIORITY: Extract the issuing country from passport text (look for country names like "Republic of Lebanon", "Republic of India", "Arab Republic of Egypt", etc.)
    - FIFTH PRIORITY: Use passport country code (LBN = Lebanon, IND = India, EGY = Egypt, etc.)
    - SIXTH PRIORITY: Use nationality as fallback
    - CRITICAL: Never use UAE addresses, place of birth, or other location data as passport issue place
    - EXAMPLES: 
      * Lebanese passport with "الجمهورية اللبنانية / Republic of Lebanon" → "LEBANON"
      * Indian passport with "भारत गणराज्य REPUBLIC OF INDIA" → "INDIA"
      * Egyptian passport with "جمهورية مصر العربية / Arab Republic of Egypt" → "EGYPT"
      * Passport with country code "LBN" → "LEBANON"
      * Passport with country code "EGY" → "EGYPT"
      * Passport with country code "IND" → "INDIA"
      * Indian passport with "Oude and Place: DUBAI" → "DUBAI"

12. **Passport Issue Date**: 
    - FIRST PRIORITY: Document AI extracted "Issue Date" field (if available)
    - SECOND PRIORITY: Passport page 1 - look for "Date of Issue", "Issue Date", "Date d'émission", "تاريخ الإصدار", or similar fields
    - THIRD PRIORITY: Passport page 2 - look for issue date fields
    - FOURTH PRIORITY: Employee info form - look for passport issue date
    - Must be in DD/MM/YYYY or DD-MM-YYYY format
    - If found in different format, convert to DD/MM/YYYY

13. **Passport Expiry Date**: 
    - FIRST PRIORITY: Document AI extracted "Expiry Date" field (if available)
    - SECOND PRIORITY: Passport page 1 - look for "Date of Expiry", "Expiry Date", "Date d'expiration", "تاريخ انتهاء الصلاحية", or similar fields
    - THIRD PRIORITY: Passport page 2 - look for expiry date fields
    - FOURTH PRIORITY: Employee info form - look for passport expiry date
    - Must be in DD/MM/YYYY or DD-MM-YYYY format
    - If found in different format, convert to DD/MM/YYYY

14. **Home Phone Number**: 
    - Look for phone numbers in any document
    - If phone number starts with "+", replace "+" with "00"
    - Example: "+971501234567" becomes "00971501234567"

15. **Home Address**: 
    - FIRST PRIORITY: Passport page 2 - look for address section
    - SECOND PRIORITY: Employee info form - look for "Address" field
    - Extract only: Area, City, Province/State
    - Example: "E 108 CHANDRANAGAR COLONY BEHIND SUVARNALAXMI APARTMENTS WAGHODIA ROAD, VADODARA PIN:390019,GUJARAT, INDIA" becomes "VADODARA, GUJARAT"
    - Remove PIN codes, street details, building numbers

16. **UAE Address**: 
    - FIRST PRIORITY: Employee info form - look for "Primary Address In UAE" or address fields
    - SECOND PRIORITY: Look for UAE addresses in any document
    - Extract: Area, City/Emirate (both parts)
    - Example: "Madinat Zayed, Al Yam street, B841 building, F202" becomes "Madinat Zayed, Abu Dhabi"
    - Example: "B841 building, F202, Abu Dhabi" becomes "Abu Dhabi"
    - Example: "Al Ain, Abu Dhabi" becomes "Al Ain, Abu Dhabi"
    - Example: "Dubai Marina, Dubai" becomes "Dubai Marina, Dubai"
    - Remove building numbers, apartment numbers, street details
    - Include both the specific area and the city/emirate when available

17. **Job Title**: 
    - FIRST PRIORITY: Employee info form - look for "Profession" or "Job Title"
    - SECOND PRIORITY: Certificate OCR
    - THIRD PRIORITY: Email text

18. **Salary**: Use salary_data JSON if provided.

19. **Document Type**: Use ResNet prediction and Google Vision metadata.

20. **Arabic Translations**: 
    - Translate Full Name, Place of Birth, and Passport Issue Place to Arabic
    - IMPORTANT: For Indian passports, translate the COMPLETE name (including appended father's name) to Arabic
    - If missing, use nationality as fallback

21. **Attestation** (only if certificate_attestation is detected):
   - ONLY extract from certificate_attestation OCR (same document), never from other documents
   - Look for attestation numbers that are 10-15 digits long (e.g., "201400642961", "20183787892730") - this is Attestation Number 1
   - Look for any 7-digit number (e.g., "2468761") - this is Attestation Number 2
   - NEVER use Emirates ID numbers (they start with 784)
   - NEVER use identity numbers (they have / format like 101/2019/179119)
   - NEVER use receipt numbers or numbers from other document types
   - NEVER use numbers that don't appear in the actual OCR text
   - ONLY use numbers that are clearly attestation numbers from the attestation sticker/certificate
   - CRITICAL: Verify that any extracted number actually appears in the OCR text before using it
   - If both are not found, set them to null

Return output as JSON with these fields:
  "Full Name", "Full Name (AR)", "Father's Name", "Mother's Name", "Date of Birth",
  "Nationality", "Nationality (AR)", "Passport Number", "EID_Number", "Identity_Number", "UID_Number",
  "Place of Birth", "Place of Birth (AR)", "Passport Issue Place", "Passport Issue Place (AR)",
  "Passport Issue Date", "Passport Expiry Date",
  "Home Phone Number", "Home Address", "UAE Address", "Job Title", "Salary", "Document Type",
  "Attestation Number 1", "Attestation Number 2"

VALIDATION: Before returning, verify that:
- Passport Issue Place is the country that issued the passport (not where person lives/works)
- Place of Birth is where the person was born (not where passport was issued)
- UAE Address is where the person lives in UAE (not passport issue place)

=== PASSPORT PAGE 1 OCR ===
{passport_ocr_1}

=== PASSPORT PAGE 2 OCR ===
{passport_ocr_2}

=== EMIRATES ID FRONT OCR ===
{emirates_id_ocr}

=== EMIRATES ID BACK OCR ===
{emirates_id_2_ocr}

=== EMPLOYEE INFO OCR ===
{employee_info}

=== CERTIFICATE OCR ===
{certificate_ocr}

=== EMAIL TEXT ===
{email_text}

=== SALARY TABLE ===
{json.dumps(salary_data, indent=2)}

=== RESNET CLASSIFICATION ===
{resnet_label}

=== DOCUMENT AI EXTRACTED FIELDS ===
{json.dumps(google_metadata.get('passport_1_fields', {}), indent=2) if 'passport_1_fields' in google_metadata else '{}'}
{json.dumps(google_metadata.get('passport_2_fields', {}), indent=2) if 'passport_2_fields' in google_metadata else '{}'}
{json.dumps(google_metadata.get('emirates_id_fields', {}), indent=2) if 'emirates_id_fields' in google_metadata else '{}'}
{json.dumps(google_metadata.get('emirates_id_2_fields', {}), indent=2) if 'emirates_id_2_fields' in google_metadata else '{}'}
{json.dumps(google_metadata.get('certificate_fields', {}), indent=2) if 'certificate_fields' in google_metadata else '{}'}
{json.dumps(google_metadata.get('employee_info_fields', {}), indent=2) if 'employee_info_fields' in google_metadata else '{}'}

=== GOOGLE VISION METADATA ===
{json.dumps(google_metadata, indent=2)}

Return only the structured JSON.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content

        if not content or not content.strip():
            print("⚠️ GPT returned empty response.")
            return {}

        print("🔍 GPT RAW RESPONSE:\n", content)  # Debug log
        
        # Store original response for text file
        original_response = content
        
        # Remove markdown wrapping if present
        if content.strip().startswith("```json"):
            content = content.strip().removeprefix("```json").removesuffix("```").strip()
        elif content.strip().startswith("```"):
            content = content.strip().removeprefix("```").removesuffix("```").strip()
        
        # Parse the JSON response
        structured_data = json.loads(content)
        
        # Validate attestation numbers if certificate_attestation OCR is available
        if certificate_ocr and ('Attestation Number 1' in structured_data or 'Attestation Number 2' in structured_data):
            print("🔍 Validating attestation numbers...")
            
            # Check if Document AI extracted attestation numbers are available in metadata
            document_ai_attestation_1 = None
            document_ai_attestation_2 = None
            if google_metadata and 'certificate_fields' in google_metadata:
                doc_ai_fields = google_metadata['certificate_fields']
                if 'Attestation Number 1' in doc_ai_fields:
                    document_ai_attestation_1 = doc_ai_fields['Attestation Number 1']
                    print(f"📋 Document AI extracted Attestation Number 1: {document_ai_attestation_1}")
                if 'Attestation Number 2' in doc_ai_fields:
                    document_ai_attestation_2 = doc_ai_fields['Attestation Number 2']
                    print(f"📋 Document AI extracted Attestation Number 2: {document_ai_attestation_2}")
            
            # If Document AI extracted attestation numbers, trust them over OCR validation
            if document_ai_attestation_1 and document_ai_attestation_1 != "N/A":
                print(f"✅ Using Document AI extracted Attestation Number 1: {document_ai_attestation_1}")
                structured_data['Attestation Number 1'] = document_ai_attestation_1
            
            if document_ai_attestation_2 and document_ai_attestation_2 != "N/A":
                print(f"✅ Using Document AI extracted Attestation Number 2: {document_ai_attestation_2}")
                structured_data['Attestation Number 2'] = document_ai_attestation_2
            
            # Fall back to OCR text validation for any missing numbers
            if not document_ai_attestation_1 or not document_ai_attestation_2:
                validated_attestation = validate_attestation_numbers(certificate_ocr, structured_data)
                
                # Update the structured data with validated numbers (only if Document AI didn't provide them)
                if not document_ai_attestation_1 and 'Attestation Number 1' in validated_attestation:
                    structured_data['Attestation Number 1'] = validated_attestation['Attestation Number 1']
                if not document_ai_attestation_2 and 'Attestation Number 2' in validated_attestation:
                    structured_data['Attestation Number 2'] = validated_attestation['Attestation Number 2']
        
        return structured_data, original_response

    except json.JSONDecodeError as e:
        print("⚠️ GPT returned invalid JSON:", e)
        print("Raw content was:\n", content)
        return {}, content

    except Exception as e:
        print("❌ GPT structuring failed:", e)
        return {}, ""
def structure_document_with_gpt(label: str, ocr_text: str, email_text: str, salary_data: dict) -> dict:
    """
    Lightweight GPT structuring for a single document's OCR + label context.
    """
    if not ocr_text.strip():
        print(f"⚠️ Skipping GPT — empty OCR text.")
        return {}

    prompt = f"""
You are an expert in understanding MOHRE document content. Your job is to extract whatever structured data you can from this OCR text.

Document Type: {label}

OCR TEXT:
{ocr_text}

EMAIL TEXT:
{email_text}

SALARY DATA:
{json.dumps(salary_data, indent=2)}

Return as many of these fields as possible:
"Full Name", "Father's Name", "Mother's Name", "Date of Birth",
"Nationality", "Passport Number", "Place of Birth", "Passport Issue Place",
"Passport Issue Date", "Passport Expiry Date",
"Job Title", "Salary", "Address (Final)", "Attestation Number 1", "Attestation Number 2"
Return null for anything that isn't found. Return JSON only.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            print("⚠️ GPT returned empty response.")
            return {}

        # 🔽 Remove markdown wrapping if present
        if content.strip().startswith("```json"):
            content = content.strip().removeprefix("```json").removesuffix("```").strip()
        elif content.strip().startswith("```"):
            content = content.strip().removeprefix("```").removesuffix("```").strip()

        return json.loads(content)

    except Exception as e:
        print(f"❌ GPT failed to structure single document: {e}")
        return {}
def merge_structured_documents(structured_docs):
    """
    Combines all partial structured docs into a single folder-level structure.
    structured_docs = [(dict, image_path), ...]
    """
    final = {
        "Full Name": None,
        "Full Name (AR)": None,
        "Father's Name": None,
        "Mother's Name": None,
        "Date of Birth": None,
        "Nationality": None,
        "Nationality (AR)": None,
        "Passport Number": None,
        "EID_Number": None,
        "Identity_Number": None,
        "UID_Number": None,
        "Place of Birth": None,
        "Place of Birth (AR)": None,
        "Passport Issue Place": None,
        "Passport Issue Place (AR)": None,
        "Passport Issue Date": None,
        "Passport Expiry Date": None,
        "Home Phone Number": None,
        "Home Address": None,
        "UAE Address": None,
        "Job Title": None,
        "Salary": None,
        "Document Type": None,
        "Attestation Number 1": None,
        "Attestation Number 2": None
    }

    for struct, _ in structured_docs:
        for key in final:
            if not final[key] and struct.get(key):
                final[key] = struct[key]

    return final
