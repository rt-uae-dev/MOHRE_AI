#!/usr/bin/env python3
"""
Google Document AI Processor
Handles structured document processing using Document OCR Processor
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from config import get_config

class DocumentAIProcessor:
    """Google Document AI processor using Document OCR Processor"""
    
    def __init__(self):
        """Initialize Document AI client and processor"""
        # Get processor details from configuration
        config = get_config()
        self.project_id = config.google_cloud_project_id
        self.processor_id = config.document_ai_processor_id
        self.location = "us"  # Default location
        
        if not self.project_id or not self.processor_id:
            print("⚠️ Document AI not configured. Set GOOGLE_CLOUD_PROJECT_ID and DOCUMENT_AI_PROCESSOR_ID")
            self.enabled = False
            self.client = None
        else:
            try:
                from google.cloud import documentai_v1 as documentai
                self.client = documentai.DocumentProcessorServiceClient()
                self.processor_name = f"projects/{self.project_id}/locations/{self.location}/processors/{self.processor_id}"
                self.enabled = True
                print(f"✅ Document AI initialized with Document OCR Processor: {self.processor_id}")
            except Exception as e:
                print(f"⚠️ Document AI initialization failed: {e}")
                print("   This is expected if Document AI is not set up yet")
                self.enabled = False
                self.client = None
    
    def process_document(self, image_path: str) -> Dict:
        """
        Process document image with Document AI Document OCR Processor
        Returns OCR text with confidence scores
        """
        if not self.enabled or not self.client:
            return {"error": "Document AI not configured or not available"}
        
        try:
            # Read the image
            with open(image_path, "rb") as image:
                image_content = image.read()
            
            # Create the document
            document = {"content": image_content, "mime_type": "image/jpeg"}
            
            # Process the document - fix the request format
            request = {
                "name": self.processor_name, 
                "raw_document": document  # Changed from "document" to "raw_document"
            }
            result = self.client.process_document(request=request)
            document = result.document
            
            # Extract text and confidence data
            ocr_data = self._extract_ocr_data(document)
            
            return {
                "success": True,
                "full_text": document.text,
                "ocr_data": ocr_data,
                "confidence": getattr(document, 'confidence', 0.0),  # Use getattr for confidence
                "pages": len(document.pages)
            }
            
        except Exception as e:
            print(f"❌ Document AI processing failed: {e}")
            return {"error": str(e)}
    
    def _extract_ocr_data(self, document) -> Dict:
        """Extract OCR data from Document AI response"""
        ocr_data = {
            "text_blocks": [],
            "confidence_scores": [],
            "page_layout": []
        }
        
        # Extract text blocks with confidence scores
        for page in document.pages:
            page_data = {
                "page_number": page.page_number,
                "width": getattr(page, 'width', 0),  # Use getattr to handle missing attributes
                "height": getattr(page, 'height', 0),
                "text_blocks": []
            }
            
            # Handle different page structures
            if hasattr(page, 'blocks'):
                blocks = page.blocks
            elif hasattr(page, 'paragraphs'):
                blocks = page.paragraphs
            else:
                blocks = []
            
            for block in blocks:
                try:
                    block_text = self._get_text_from_layout(block.layout, document.text)
                    confidence = getattr(block.layout, 'confidence', 0.0)
                    
                    # Get bounding box if available
                    bounding_box = {}
                    if hasattr(block.layout, 'bounding_poly') and block.layout.bounding_poly.vertices:
                        vertices = block.layout.bounding_poly.vertices
                        if len(vertices) >= 4:
                            bounding_box = {
                                "x": vertices[0].x,
                                "y": vertices[0].y,
                                "width": vertices[2].x - vertices[0].x,
                                "height": vertices[2].y - vertices[0].y
                            }
                    
                    page_data["text_blocks"].append({
                        "text": block_text,
                        "confidence": confidence,
                        "bounding_box": bounding_box
                    })
                    
                    ocr_data["text_blocks"].append(block_text)
                    ocr_data["confidence_scores"].append(confidence)
                    
                except Exception as e:
                    print(f"Warning: Error processing block: {e}")
                    continue
            
            ocr_data["page_layout"].append(page_data)
        
        return ocr_data
    
    def _get_text_from_layout(self, layout, full_text: str) -> str:
        """Extract text from layout using text anchor"""
        if not layout.text_anchor.text_segments:
            return ""
        
        text_segments = []
        for segment in layout.text_anchor.text_segments:
            start_index = segment.start_index
            end_index = segment.end_index
            text_segments.append(full_text[start_index:end_index])
        
        return "".join(text_segments)
    
    def get_structured_result(self, image_path: str) -> Dict:
        """
        Get structured result in the same format as current pipeline
        """
        result = self.process_document(image_path)
        
        if "error" in result:
            return result
        
        # Convert to the format expected by the pipeline
        structured_data = {
            "ocr_text": result["full_text"],
            "confidence": result["confidence"],
            "text_blocks": result["ocr_data"]["text_blocks"],
            "page_count": result["pages"]
        }
        
        # Add metadata
        structured_data["Document Type"] = "document_ai_processed"
        structured_data["_document_ai_confidence"] = result["confidence"]
        structured_data["_document_ai_ocr_data"] = result["ocr_data"]
        
        return structured_data
    
    def is_better_than_vision(self, doc_ai_result: Dict, vision_result: Dict) -> bool:
        """
        Compare Document AI result with Google Vision result
        Returns True if Document AI is better
        """
        if "error" in doc_ai_result:
            return False
        
        # Check overall confidence
        doc_ai_confidence = doc_ai_result.get("confidence", 0)
        if doc_ai_confidence > 0.9:
            print(f"✅ Document AI has high confidence: {doc_ai_confidence:.2f}")
            return True
        
        # Check text length
        doc_ai_text = doc_ai_result.get("full_text", "")
        vision_text = vision_result.get("ocr_text", "")
        
        if len(doc_ai_text) > len(vision_text) * 1.2:  # 20% more text
            print(f"✅ Document AI captured more text: {len(doc_ai_text)} vs {len(vision_text)}")
            return True
        
        # Check for specific fields that Vision might have missed
        doc_ai_text_lower = doc_ai_text.lower()
        vision_text_lower = vision_text.lower()
        
        # Look for issue place keywords
        issue_keywords = ['dubai', 'duba', 'place of issue', 'authority']
        for keyword in issue_keywords:
            if keyword in doc_ai_text_lower and keyword not in vision_text_lower:
                print(f"✅ Document AI found '{keyword}' that Vision missed")
                return True
        
        return False
    
    def extract_specific_fields(self, ocr_text: str) -> Dict:
        """
        Extract specific fields from OCR text using comprehensive parsing
        Enhanced to handle UAE documents with structured information
        """
        fields = {}
        text_lower = ocr_text.lower()
        
        import re
        
        # Extract UID Number (Unified Identity Number)
        uid_patterns = [
            r'U\.I\.D\s*No[:\s]*(\d+)',
            r'U\.I\.D\s*Number[:\s]*(\d+)',
            r'الرقم الموحد[:\s]*(\d+)',
            r'unified\s*identity\s*number[:\s]*(\d+)'
        ]
        for pattern in uid_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                fields["UID_Number"] = matches[0]
                break
        
        # Extract Identity Number (File Number)
        identity_patterns = [
            r'Identity\s*No[:\s]*(\d+)',
            r'رقم الهوية[:\s]*(\d+)',
            r'identity\s*number[:\s]*(\d+)'
        ]
        for pattern in identity_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                fields["Identity_Number"] = matches[0]
                break
        
        # Extract Residence Number (complex format like 101/2019/2/179119)
        residence_patterns = [
            r'Residence\s*No[:\s]*([0-9/]+)',
            r'رقم الإقامة[:\s]*([0-9/]+)',
            r'residence\s*number[:\s]*([0-9/]+)'
        ]
        for pattern in residence_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                fields["Residence_Number"] = matches[0]
                break
        
        # Extract Emirates ID Number
        eid_patterns = [
            r'Emirates\s*ID[:\s]*(\d+)',
            r'رقم الهوية[:\s]*(\d+)',
            r'emirates\s*id\s*number[:\s]*(\d+)',
            r'(\d{3}-\d{4}-\d{7}-\d)',  # Format: 784-1991-6903171-5
            r'(\d{15})'  # 15-digit Emirates ID
        ]
        for pattern in eid_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                # Clean up the number (remove dashes and spaces)
                clean_number = re.sub(r'[-\s]', '', matches[0])
                fields["EID_Number"] = clean_number
                break
        
        # Extract Passport Number
        passport_patterns = [
            r'Passport\s*No[:\s]*([A-Z0-9]+)',
            r'رقم الجواز[:\s]*([A-Z0-9]+)',
            r'passport\s*number[:\s]*([A-Z0-9]+)',
            r'\b([A-Z]\d{7})\b',  # Z5547821 format
            r'\b(\d{8})\b'        # 25547821 format
        ]
        for pattern in passport_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                fields["Passport Number"] = matches[0]
                break
        
        # Extract Full Name
        name_patterns = [
            r'Full\s*Name[:\s]*([A-Z\s]+)',
            r'الاسم الكامل[:\s]*([^\n]+)',
            r'full\s*name[:\s]*([A-Z\s]+)',
            r'name[:\s]*([A-Z\s]+)'
        ]
        for pattern in name_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                # Clean up the name - remove extra words and clean up
                name = matches[0].strip()
                # Remove common extra words that might be captured
                name = re.sub(r'\b(Profession|Employer|Cancel Date|Date of Birth|Full Name|Name)\b', '', name, flags=re.IGNORECASE)
                name = re.sub(r'\s+', ' ', name).strip()  # Clean up extra spaces
                if len(name) > 3 and not name.lower().startswith(('profession', 'employer', 'cancel', 'date')):  # Only use if it's a reasonable name
                    fields["Full Name"] = name
                    break
        
        # Extract Profession/Job Title
        profession_patterns = [
            r'Profession[:\s]*([^\n]+)',
            r'المهنة[:\s]*([^\n]+)',
            r'profession[:\s]*([^\n]+)',
            r'job\s*title[:\s]*([^\n]+)'
        ]
        for pattern in profession_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                profession = matches[0].strip()
                # Clean up profession - remove extra words
                profession = re.sub(r'\b(Employer|Cancel Date|Date of Birth|Full Name|Name)\b', '', profession, flags=re.IGNORECASE)
                profession = re.sub(r'\s+', ' ', profession).strip()  # Clean up extra spaces
                if len(profession) > 2 and not profession.lower().startswith(('employer', 'cancel', 'date')):
                    fields["Job Title"] = profession
                    break
        
        # Extract Employer
        employer_patterns = [
            r'Employer[:\s]*([^\n]+)',
            r'صاحب العمل[:\s]*([^\n]+)',
            r'employer[:\s]*([^\n]+)'
        ]
        for pattern in employer_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                employer = matches[0].strip()
                if len(employer) > 2:
                    fields["Employer"] = employer
                    break
        
        # Extract Issue Place
        issue_patterns = [
            r'Place\s*of\s*Issue[:\s]*([^\n]+)',
            r'جهة الإصدار[:\s]*([^\n]+)',
            r'place\s*of\s*issue[:\s]*([^\n]+)'
        ]
        for pattern in issue_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                place = matches[0].strip()
                if len(place) > 2:
                    fields["Passport Issue Place"] = place
                    break
        
        # Extract dates
        date_patterns = [
            r'\b\d{2}/\d{2}/\d{4}\b',  # DD/MM/YYYY
            r'\b\d{2}-\d{2}-\d{4}\b'   # DD-MM-YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, ocr_text)
            if matches:
                fields["Dates Found"] = matches
        
        return fields
    
    def extract_certificate_fields(self, ocr_text: str) -> Dict:
        """
        Extract certificate-specific fields from OCR text
        Handles both regular certificates and attestation certificates
        """
        fields = {}
        text_lower = ocr_text.lower()
        
        import re
        
        # Extract attestation numbers - look for specific patterns from UAE attestation certificates
        # Based on the actual document structure, we need to find:
        # 1. Application Number (رقم الطلب) - typically 12 digits
        # 2. Large prominent number - typically 7 digits
        
        # Look for Application Number (رقم الطلب)
        application_patterns = [
            r'رقم الطلب[:\s]*([0-9]{12})',  # Arabic: Application Number
            r'application\s*number[:\s]*([0-9]{12})',  # English
            r'رقم الطلب\s*([0-9]{12})',  # Arabic without colon
            r'application\s*no[.:\s]*([0-9]{12})',  # English abbreviation
        ]
        
        # Look for large prominent numbers (typically 7 digits)
        prominent_number_patterns = [
            r'\b([0-9]{7})\b',  # 7-digit numbers at word boundaries
            r'([0-9]{7})(?=\s|$)',  # 7-digit numbers at end of line
        ]
        
        # Look for barcode numbers (12 digits with leading zeros)
        barcode_patterns = [
            r'\b([0-9]{12})\b',  # 12-digit numbers
            r'([0-9]{12})(?=\s|$)',  # 12-digit numbers at end of line
        ]
        
        # First, exclude Emirates ID numbers (they start with 784)
        emirates_id_pattern = r'784[-\s]*[0-9]{3}[-\s]*[0-9]{7}[-\s]*[0-9]'
        emirates_id_matches = re.findall(emirates_id_pattern, ocr_text, re.IGNORECASE)
        
        # Also exclude identity numbers (they have / format like 101/2019/179119)
        identity_pattern = r'[0-9]{3}/[0-9]{4}/[0-9]+'
        identity_matches = re.findall(identity_pattern, ocr_text, re.IGNORECASE)
        
        # Extract Application Number (Attestation Number 1)
        application_found = False
        for pattern in application_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                for match in matches:
                    # Skip if it's an Emirates ID number
                    if any(emirates_id in match for emirates_id in emirates_id_matches):
                        continue
                    # Skip if it's an identity number
                    if any(identity in match for identity in identity_matches):
                        continue
                    # Skip if it starts with 784 (Emirates ID pattern)
                    if match.startswith('784'):
                        continue
                    # Skip if it contains / (identity number pattern)
                    if '/' in match:
                        continue
                    
                    fields["Attestation Number 1"] = match
                    application_found = True
                    break
                if application_found:
                    break
        
        # Extract large prominent number (Attestation Number 2)
        prominent_found = False
        for pattern in prominent_number_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                for match in matches:
                    # Skip if it's an Emirates ID number
                    if any(emirates_id in match for emirates_id in emirates_id_matches):
                        continue
                    # Skip if it's an identity number
                    if any(identity in match for identity in identity_matches):
                        continue
                    # Skip if it starts with 784 (Emirates ID pattern)
                    if match.startswith('784'):
                        continue
                    # Skip if it contains / (identity number pattern)
                    if '/' in match:
                        continue
                    
                    fields["Attestation Number 2"] = match
                    prominent_found = True
                    break
                if prominent_found:
                    break
        
        # Fallback: if no specific patterns found, look for any 12-digit number as Attestation Number 1
        if not application_found:
            for pattern in barcode_patterns:
                matches = re.findall(pattern, ocr_text, re.IGNORECASE)
                if matches:
                    for match in matches:
                        # Skip if it's an Emirates ID number
                        if any(emirates_id in match for emirates_id in emirates_id_matches):
                            continue
                        # Skip if it's an identity number
                        if any(identity in match for identity in identity_matches):
                            continue
                        # Skip if it starts with 784 (Emirates ID pattern)
                        if match.startswith('784'):
                            continue
                        # Skip if it contains / (identity number pattern)
                        if '/' in match:
                            continue
                        
                        fields["Attestation Number 1"] = match
                        break
                    break
        
        # Extract degree/qualification
        degree_patterns = [
            r'bachelor[:\s]*([A-Za-z\s]+)',
            r'master[:\s]*([A-Za-z\s]+)',
            r'phd[:\s]*([A-Za-z\s]+)',
            r'diploma[:\s]*([A-Za-z\s]+)',
            r'degree[:\s]*([A-Za-z\s]+)'
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                fields["Degree/Qualification"] = matches[0].strip()
                break
        
        # Extract institution/university name
        institution_patterns = [
            r'university[:\s]*([A-Za-z\s]+)',
            r'institute[:\s]*([A-Za-z\s]+)',
            r'college[:\s]*([A-Za-z\s]+)',
            r'school[:\s]*([A-Za-z\s]+)'
        ]
        
        for pattern in institution_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                fields["Institution"] = matches[0].strip()
                break
        
        # Extract issuing authority
        authority_patterns = [
            r'ministry[:\s]*([A-Za-z\s]+)',
            r'authority[:\s]*([A-Za-z\s]+)',
            r'department[:\s]*([A-Za-z\s]+)',
            r'government[:\s]*([A-Za-z\s]+)'
        ]
        
        for pattern in authority_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                fields["Issuing Authority"] = matches[0].strip()
                break
        
        # Extract dates (issue date, expiry date)
        date_patterns = [
            r'issue[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})',
            r'date[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})',
            r'expiry[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})',
            r'valid[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                if 'expiry' in pattern or 'valid' in pattern:
                    fields["Expiry Date"] = matches[0]
                else:
                    fields["Issue Date"] = matches[0]
        
        # Extract grade/score
        grade_patterns = [
            r'grade[:\s]*([A-Za-z0-9\s]+)',
            r'score[:\s]*([A-Za-z0-9\s]+)',
            r'gpa[:\s]*([0-9.]+)',
            r'cgpa[:\s]*([0-9.]+)'
        ]
        
        for pattern in grade_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                fields["Grade/Score"] = matches[0].strip()
                break
        
        # Check for official stamps/seals
        stamp_keywords = ['stamp', 'seal', 'official', 'authenticated', 'verified']
        for keyword in stamp_keywords:
            if keyword in text_lower:
                fields["Has Official Stamp"] = "Yes"
                break
        else:
            fields["Has Official Stamp"] = "No"
        
        return fields
    
    def extract_emirates_id_fields(self, ocr_text: str) -> Dict:
        """
        Extract Emirates ID specific fields from OCR text
        """
        fields = {}
        text_lower = ocr_text.lower()
        
        import re
        
        # Extract Emirates ID number - must start with 784
        id_patterns = [
            r'emirates\s*id[:\s]*([0-9]{3}[-\s]*[0-9]{3}[-\s]*[0-9]{7}[-\s]*[0-9])',
            r'id\s*number[:\s]*([0-9]{3}[-\s]*[0-9]{3}[-\s]*[0-9]{7}[-\s]*[0-9])',
            r'رقم\s*الهوية[:\s]*([0-9]{3}[-\s]*[0-9]{3}[-\s]*[0-9]{7}[-\s]*[0-9])',
            # Look for 784 pattern specifically
            r'(784[-\s]*[0-9]{3}[-\s]*[0-9]{7}[-\s]*[0-9])',
        ]
        
        for pattern in id_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                # Clean up the number (remove spaces and dashes)
                clean_number = re.sub(r'[-\s]', '', matches[0])
                # Verify it starts with 784
                if clean_number.startswith('784'):
                    fields["Emirates ID Number"] = clean_number
                    break
        
        # Extract full name
        name_patterns = [
            r'name[:\s]*([A-Za-z\s]+)',
            r'full[:\s]*name[:\s]*([A-Za-z\s]+)',
            r'اسم[:\s]*([A-Za-z\s]+)'
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                fields["Full Name"] = matches[0].strip()
                break
        
        # Extract nationality
        nationality_patterns = [
            r'nationality[:\s]*([A-Za-z\s]+)',
            r'citizen[:\s]*([A-Za-z\s]+)',
            r'الجنسية[:\s]*([A-Za-z\s]+)'
        ]
        
        for pattern in nationality_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                fields["Nationality"] = matches[0].strip()
                break
        
        # Extract dates
        date_patterns = [
            r'issue[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})',
            r'expiry[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})',
            r'birth[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})',
            r'تاريخ\s*الميلاد[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                if 'birth' in pattern or 'الميلاد' in pattern:
                    fields["Date of Birth"] = matches[0]
                elif 'expiry' in pattern:
                    fields["Expiry Date"] = matches[0]
                else:
                    fields["Issue Date"] = matches[0]
        
        return fields
    
    def get_document_type(self, ocr_text: str) -> str:
        """
        Determine document type based on OCR text content
        """
        text_lower = ocr_text.lower()
        
        # Check for passport indicators
        if any(keyword in text_lower for keyword in ['passport', 'passport no', 'passport number']):
            return "passport"
        
        # Check for Emirates ID indicators
        if any(keyword in text_lower for keyword in ['emirates id', 'emirates identity', 'uae id']):
            return "emirates_id"
        
        # Check for attestation certificate indicators
        if any(keyword in text_lower for keyword in ['attestation', 'attested', 'ministry', 'government']):
            return "attestation_certificate"
        
        # Check for regular certificate indicators
        if any(keyword in text_lower for keyword in ['certificate', 'degree', 'diploma', 'university', 'college']):
            return "certificate"
        
        return "unknown"
    
    def extract_fields_by_document_type(self, ocr_text: str) -> Dict:
        """
        Extract fields based on detected document type
        """
        document_type = self.get_document_type(ocr_text)
        
        if document_type == "passport":
            return self.extract_specific_fields(ocr_text)
        elif document_type == "emirates_id":
            return self.extract_emirates_id_fields(ocr_text)
        elif document_type == "attestation_certificate":
            return self.extract_certificate_fields(ocr_text)
        elif document_type == "certificate":
            return self.extract_certificate_fields(ocr_text)
        else:
            # Try all extractors for unknown documents
            all_fields = {}
            all_fields.update(self.extract_specific_fields(ocr_text))
            all_fields.update(self.extract_certificate_fields(ocr_text))
            all_fields.update(self.extract_emirates_id_fields(ocr_text))
            return all_fields

# Global instance
DOCUMENT_AI_PROCESSOR = DocumentAIProcessor() 