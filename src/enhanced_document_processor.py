#!/usr/bin/env python3
"""
Enhanced Document Processor using trained YOLO8 and ResNet models
and Google Document AI/Vision for OCR.
"""

import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from google.cloud import documentai_v1 as documentai
from config import get_config

# Import simple Google Vision orientation detection
from google_vision_orientation_detector import rotate_if_needed
from yolo_crop_ocr_pipeline import run_google_vision_ocr

# Load configuration
config = get_config()

# Set Google Application Credentials for Document AI
google_creds_path = config.google_application_credentials
if google_creds_path and google_creds_path.exists():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(google_creds_path)
    print(f"✅ Document AI credentials set: {google_creds_path}")
else:
    print(f"⚠️ Document AI credentials not found at: {google_creds_path}")

class EnhancedDocumentProcessor:
    def __init__(self):
        """Initialize the enhanced document processor with trained models."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained models
        self.yolo_model = self.load_yolo_model()
        self.classifier_model = self.load_classifier_model()
        
        # Initialize Document AI client
        self.document_ai_client = self.initialize_document_ai()
        

        
        # Image transforms for classifier
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Document type mapping (update based on your 22 classes)
        self.document_types = {
            0: "certificate_attestation",
            1: "certificate_original", 
            2: "passport_photo",
            3: "emirates_id",
            4: "visa_document",
            5: "salary_document",
            6: "contract",
            7: "offer_letter",
            8: "employment_agreement",
            9: "other",
            # Add more mappings for your 22 classes as needed
            # You can map multiple class IDs to the same document type
        }
        
        # YOLO class mapping (update based on your training)
        self.yolo_classes = {


            1: "receipt_label", 
            2: "application_number",
            3: "label_number",
            4: "old_indian_stamp"
        }
    
    def load_yolo_model(self):
        """Load the trained YOLO8 model."""
        try:
            model_path = "models/yolo8_best.pt"  # Use the correct path
            if os.path.exists(model_path):
                model = YOLO(model_path)
                print(f"✅ YOLO8 model loaded from: {model_path}")
                return model
            else:
                print(f"❌ YOLO8 model not found at: {model_path}")
                return None
        except Exception as e:
            print(f"❌ Failed to load YOLO8 model: {e}")
            return None
    
    def load_classifier_model(self):
        """Load the trained ResNet classifier model."""
        try:
            model_path = "models/resnet_classifier.pt"  # Use the correct path
            if os.path.exists(model_path):
                # Load the model state dict
                state_dict = torch.load(model_path, map_location=self.device)
                
                # Create ResNet model (assuming ResNet18, adjust as needed)
                from torchvision import models
                model = models.resnet18(weights=None)  # Use weights=None instead of pretrained=False
                
                # Determine the number of classes from the state dict
                fc_weight = state_dict.get('fc.weight', None)
                if fc_weight is not None:
                    num_classes = fc_weight.shape[0]
                    print(f"📊 Detected {num_classes} classes in model")
                else:
                    num_classes = 23  # Default fallback
                    print(f"⚠️ Could not detect classes, using default: {num_classes}")
                
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                
                # Load state dict
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                
                print(f"✅ ResNet classifier loaded from: {model_path} with {num_classes} classes")
                return model
            else:
                print(f"❌ ResNet classifier not found at: {model_path}")
                return None
        except Exception as e:
            print(f"❌ Failed to load ResNet classifier: {e}")
            return None
    
    def initialize_document_ai(self):
        """Initialize Google Document AI client."""
        try:
            # For now, let's skip Document AI and use a simpler OCR approach
            print("⚠️ Document AI authentication requires service account credentials")
            print("📝 Using fallback OCR method...")
            return None
        except Exception as e:
            print(f"❌ Failed to initialize Document AI: {e}")
            return None
    
    def detect_and_correct_orientation(self, image_path):
        """
        Detect and correct image orientation using improved Gemini AI approach.
        Uses conservative logic for passport pages to prevent over-rotation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path to the corrected image (original if no rotation needed)
        """
        try:
            print(f"🔄 Asking Gemini AI if image needs rotation...")
            
            # Use the improved Gemini AI orientation detection
            corrected_image_path = rotate_if_needed(image_path)
            
            if corrected_image_path != image_path:
                print(f"✅ Image orientation corrected: {os.path.basename(corrected_image_path)}")
            else:
                print(f"✅ Image orientation is already correct")
                
            return corrected_image_path
            
        except Exception as e:
            print(f"❌ Orientation detection failed: {e}")
            return image_path
    
    def classify_document_type(self, image_path):
        """Classify document type using trained ResNet model."""
        try:
            if self.classifier_model is None:
                return "unknown", 0.0
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.classifier_model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            document_type = self.document_types.get(predicted_class, f"class_{predicted_class}")
            
            print(f"📋 Document classified as: {document_type} (confidence: {confidence:.3f})")
            return document_type, confidence
            
        except Exception as e:
            print(f"❌ Document classification failed: {e}")
            return "unknown", 0.0
    
    def detect_attestation_labels(self, image_path, document_type):
        """Detect and crop attestation labels using trained YOLO8 model."""
        try:
            if self.yolo_model is None:
                return []
            
            # Run YOLO detection
            results = self.yolo_model(image_path)
            
            cropped_labels = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.yolo_classes.get(class_id, f"class_{class_id}")
                        
                        # Crop the detected region
                        image = cv2.imread(image_path)
                        cropped = image[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Save cropped image
                        crop_path = f"data/temp/cropped_{class_name}_{len(cropped_labels)}.jpg"
                        os.makedirs(os.path.dirname(crop_path), exist_ok=True)
                        cv2.imwrite(crop_path, cropped)
                        
                        cropped_labels.append({
                            "class": class_name,
                            "class_id": class_id,
                            "confidence": float(confidence),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "crop_path": crop_path
                        })
            
            print(f"🎯 Detected {len(cropped_labels)} labels with YOLO8")
            return cropped_labels
            
        except Exception as e:
            print(f"❌ YOLO detection failed: {e}")
            return []
    
    def extract_text_with_document_ai(self, image_path):
        """Extract text from image using Google Document AI or Google Vision OCR."""
        try:
            if self.document_ai_client is None:
                # Use Google Vision OCR as fallback
                return run_google_vision_ocr(image_path)["ocr_text"]

            # Read image file
            with open(image_path, "rb") as image:
                image_content = image.read()

            # Configure the process request
            name = (
                f"projects/{config.google_cloud_project_id}/locations/us/"
                f"processors/{config.document_ai_processor_id}"
            )

            raw_document = documentai.RawDocument(content=image_content, mime_type="image/jpeg")
            request = documentai.ProcessRequest(name=name, raw_document=raw_document)

            # Process the document
            result = self.document_ai_client.process_document(request=request)
            document = result.document

            # Extract text
            text = document.text

            print(f"📄 Document AI extracted {len(text)} characters")
            return text

        except Exception as e:
            print(f"❌ Document AI extraction failed: {e}")
            return run_google_vision_ocr(image_path)["ocr_text"]
    
    def extract_salary_details(self, image_path):
        """Extract comprehensive salary details from document image."""
        try:
            print(f"💰 Extracting salary details from: {os.path.basename(image_path)}")
            
            # Use the enhanced salary extractor
            salary_data = self.salary_extractor.extract_salary_from_file(image_path)
            
            if salary_data:
                # Calculate total if missing
                salary_data = self.salary_extractor.calculate_total_if_missing(salary_data)
                
                # Format into summary
                salary_summary = self.salary_extractor.format_salary_summary(salary_data)
                
                print(f"✅ Salary extraction completed")
                return salary_summary
            else:
                print(f"⚠️ No salary data found in document")
                return {}
                
        except Exception as e:
            print(f"❌ Salary extraction failed: {e}")
            return {}
    
    def process_document(self, image_path):
        """Main document processing pipeline."""
        try:
            print(f"🔍 Processing document: {os.path.basename(image_path)}")
            print("=" * 60)
            
            # Step 1: Detect and correct image orientation
            corrected_image_path = self.detect_and_correct_orientation(image_path)
            
            # Step 2: Classify document type
            document_type, classification_confidence = self.classify_document_type(corrected_image_path)
            
            # Step 3: Extract salary details (if applicable)
            salary_details = {}
            if document_type in ["salary_document", "contract", "offer_letter", "employment_agreement"]:
                salary_details = self.extract_salary_details(corrected_image_path)
            
            # Step 4: Detect relevant labels
            detected_labels = self.detect_attestation_labels(corrected_image_path, document_type)
            
            # Step 5: Extract text from cropped labels
            extracted_data = {}
            for label in detected_labels:
                text = self.extract_text_with_document_ai(label["crop_path"])
                extracted_data[label["class"]] = {
                    "text": text,
                    "confidence": label["confidence"],
                    "bbox": label["bbox"]
                }
            
            # Step 6: Parse and structure the data
            structured_data = self.parse_extracted_data(extracted_data, document_type)
            
            # Add salary details to structured data
            if salary_details:
                structured_data["salary_details"] = salary_details
            
            # Add metadata
            structured_data["metadata"] = {
                "document_type": document_type,
                "classification_confidence": classification_confidence,
                "detected_labels": len(detected_labels),
                "processing_method": "Gemini AI + YOLO8 + ResNet + Document AI + Enhanced Salary Extractor",
                "orientation_corrected": corrected_image_path != image_path,
                "salary_extracted": bool(salary_details)
            }
            
            return structured_data
            
        except Exception as e:
            print(f"❌ Document processing failed: {e}")
            return {"error": str(e)}
    
    def parse_extracted_data(self, extracted_data, document_type):
        """Parse extracted text data into structured format."""
        structured_data = {
            "attestation_numbers": {
                "primary": None,
                "secondary": None,
                "receipt": None,
                "label": None
            },
            "document_info": {
                "type": document_type,
                "issue_date": None,
                "expiry_date": None,
                "has_official_stamp": False
            },
            "confidence": "medium",
            "extraction_notes": [],
            "recommendations": []
        }
        
        # Parse based on document type
        if document_type == "certificate_attestation":
            self.parse_attestation_data(extracted_data, structured_data)
        elif document_type == "certificate_original":
            self.parse_certificate_data(extracted_data, structured_data)
        # Add more document type parsers as needed
        
        return structured_data
    
    def parse_attestation_data(self, extracted_data, structured_data):
        """Parse attestation document data."""
        # Parse attestation seals
        if "attestation_seal" in extracted_data:
            text = extracted_data["attestation_seal"]["text"]
            # Extract numbers from text using regex or other methods
            numbers = self.extract_numbers_from_text(text)
            if numbers:
                structured_data["attestation_numbers"]["primary"] = numbers[0]
                if len(numbers) > 1:
                    structured_data["attestation_numbers"]["secondary"] = numbers[1]
        
        # Parse receipt labels
        if "receipt_label" in extracted_data:
            text = extracted_data["receipt_label"]["text"]
            numbers = self.extract_numbers_from_text(text)
            if numbers:
                structured_data["attestation_numbers"]["receipt"] = numbers[0]
        
        # Parse label numbers
        if "label_number" in extracted_data:
            text = extracted_data["label_number"]["text"]
            numbers = self.extract_numbers_from_text(text)
            if numbers:
                structured_data["attestation_numbers"]["label"] = numbers[0]
    
    def parse_certificate_data(self, extracted_data, structured_data):
        """Parse original certificate data."""
        # Implementation for certificate parsing
        pass
    
    def extract_numbers_from_text(self, text):
        """Extract numbers from text using regex."""
        import re
        numbers = re.findall(r'\d+', text)
        return numbers if numbers else []
    
    def cleanup_temp_files(self):
        """Clean up temporary cropped images."""
        try:
            temp_dir = "data/temp"
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    if file.startswith("cropped_"):
                        os.remove(os.path.join(temp_dir, file))
                print("🧹 Temporary files cleaned up")
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")

# Example usage
if __name__ == "__main__":
    processor = EnhancedDocumentProcessor()
    
    # Test with a sample image
    test_image = "data/dataset/passport_1/01. Passport Copy _Unaise_page_1_1_1_1.jpg"
    if os.path.exists(test_image):
        result = processor.process_document(test_image)
        print(json.dumps(result, indent=2))
    else:
        print("❌ Test image not found")
    
    # Cleanup
    processor.cleanup_temp_files() 