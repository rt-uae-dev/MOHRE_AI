# resnet18_classifier.py
import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import google.generativeai as genai
from pathlib import Path
from config import get_config

# Configure Google Gemini API
config = get_config()
genai.configure(api_key=config.gemini_api_key)

# === CONFIG ===
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = str(config.dataset_dir)
MODEL_PATH = str(config.model_path)

# === Auto-detect class names from folders ===
CLASS_NAMES = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

# === Class mapping no longer needed - model has all 23 classes ===

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dynamically determine number of classes ===
if os.path.exists(DATASET_DIR):
    ORIGINAL_CLASSES = len([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    print(f"🔍 Dynamically determined {ORIGINAL_CLASSES} classes from dataset")
else:
    ORIGINAL_CLASSES = 23  # Fallback to 23 classes if dataset directory doesn't exist
    print(f"⚠️ Using fallback: {ORIGINAL_CLASSES} classes")

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, ORIGINAL_CLASSES)

# Try to load the model
model_loaded_successfully = False
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    print(f"✅ Model loaded successfully with {ORIGINAL_CLASSES} classes")
    model_loaded_successfully = True
except RuntimeError as e:
    if "size mismatch" in str(e):
        print(f"⚠️ Model size mismatch. Expected {ORIGINAL_CLASSES} classes but model has different size.")
        print("🔄 Creating new model with correct size...")
        # Create a new model with the correct number of classes
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, ORIGINAL_CLASSES)
        print(f"✅ Created new model with {ORIGINAL_CLASSES} classes")
        print("⚠️ WARNING: This is an untrained model and will make random predictions!")
        print("🔄 Will use Gemini Vision as fallback for classification")
    else:
        raise e
except FileNotFoundError:
    print(f"⚠️ Model file not found: {MODEL_PATH}")
    print("🔄 Creating new model with correct size...")
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, ORIGINAL_CLASSES)
    print(f"✅ Created new model with {ORIGINAL_CLASSES} classes")
    print("⚠️ WARNING: This is an untrained model and will make random predictions!")
    print("🔄 Will use Gemini Vision as fallback for classification")

model.to(device)
model.eval()

# === Image transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Classify image ===
def classify_image_resnet(image_path: str) -> str:
    # If model wasn't loaded successfully, use Gemini Vision as fallback
    if not model_loaded_successfully:
        print(f"🔄 Using Gemini Vision fallback for {os.path.basename(image_path)} (ResNet model not trained)")
        return classify_image_with_gemini_vision(image_path)
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Failed to load image: {image_path} - {e}")
        return "unknown"

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Dynamically get class names from dataset directory
        # This ensures we always have the correct class names in alphabetical order
        if os.path.exists(DATASET_DIR):
            original_class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
        else:
            # Fallback to hardcoded names if dataset directory doesn't exist
            original_class_names = [
                'ID_Reg', 'Job_offer', 'Submit_work_permit_cancellation', 'certificate', 
                'certificate_attestation', 'contract', 'emirates_id', 'emirates_id_2',
                'employee_info_form', 'form_signature_page', 'forms', 'ica-issue_residence',
                'national_ID', 'passport_1', 'passport_2', 'personal_photo', 'preapproval_workpermit',
                'residence_cancellation', 'salary_documents', 'step1_mohre', 'unknown', 'visa',
                'work_permit_cancellation'
            ]
        
        predicted_original_class = original_class_names[predicted.item()]
        confidence_value = confidence.item()
        
        # Only show detailed output if confidence is low
        if confidence_value < 0.4:
            print(f"🔍 Classification for {os.path.basename(image_path)}:")
            print(f"   Predicted (original): {predicted_original_class}")
            print(f"   Confidence: {confidence_value:.3f}")
            
            # Show top 3 predictions
            top3_conf, top3_indices = torch.topk(probabilities, 3, dim=1)
            print(f"   Top 3 predictions:")
            for i in range(3):
                class_name = original_class_names[top3_indices[0][i].item()]
                conf = top3_conf[0][i].item()
                print(f"     {i+1}. {class_name}: {conf:.3f}")
        
        # If confidence is too low, use Gemini Vision as fallback
        if confidence_value < 0.4:
            print(f"⚠️ Low confidence ({confidence_value:.3f}), using Gemini Vision fallback")
            return classify_image_with_gemini_vision(image_path)
        
        return predicted_original_class
    
    
def classify_image_from_text(ocr_text: str) -> str:
    """
    Classify image based on OCR text using Gemini (text-only, not vision).
    """
    if not ocr_text.strip():
        print("⚠️ Skipping Gemini — OCR text is empty.")
        return "unknown"

    prompt = (
        "Given the OCR text from a scanned document, classify the type of document.\n\n"
        f"OCR Text:\n{ocr_text}\n\n"
        "Return ONLY one of the following class names:\n"
        f"{', '.join(CLASS_NAMES)}\n\n"
        "Answer with only the class name."
    )

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        label = response.text.strip().lower()
        if label in CLASS_NAMES:
            print(f"✅ Gemini fallback classified as: {label}")
            return label
        else:
            print(f"⚠️ Gemini returned unknown class: {label}")
            return "unknown"
    except Exception as e:
        print(f"❌ Gemini classification failed: {e}")
        return "unknown"


def classify_image_with_gemini_vision(image_path: str) -> str:
    """
    Classify image using Gemini Vision API (image + text analysis).
    This is especially useful for attestation documents and certificates.
    """
    try:
        # Load and prepare the image
        image = Image.open(image_path).convert("RGB")
        
        # Create Gemini model with vision capabilities
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Analyze this document image and classify it into one of the following categories:
        {', '.join(CLASS_NAMES)}
        
        Look for visual indicators like:
        - Official stamps and seals
        - Document layouts and structures
        - Text formatting and positioning
        - Official logos or watermarks
        - Document type indicators
        
        Return ONLY the class name, nothing else.
        """
        
        # Generate content with image
        response = model.generate_content([prompt, image])
        label = response.text.strip().lower()
        
        if label in CLASS_NAMES:
            print(f"✅ Gemini Vision classified as: {label}")
            return label
        else:
            print(f"⚠️ Gemini Vision returned unknown class: {label}")
            return "unknown"
            
    except Exception as e:
        print(f"❌ Gemini Vision classification failed: {e}")
        return "unknown"


def check_image_orientation(image_path: str) -> dict:
    """
    Check if image orientation is correct for passport photos and certificates.
    Returns orientation analysis and recommendations.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        aspect_ratio = width / height
        
        # Create Gemini model for vision analysis
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """
        Analyze this document image and determine the exact rotation needed to make it properly oriented.
        
        Look for:
        1. Text direction (should be readable from left to right, top to bottom)
        2. Document orientation (headings at top, content flowing down)
        3. Any upside-down or sideways text
        4. Proper document alignment
        
        Return your analysis in this JSON format:
        {
            "document_type": "passport_photo/certificate/emirates_id/passport/other",
            "current_orientation": "correct/upside_down/sideways_left/sideways_right",
            "rotation_needed": 0,
            "rotation_description": "No rotation needed/Needs 180° rotation/Needs 90° clockwise/Needs 90° counterclockwise",
            "clarity": "good/fair/poor",
            "issues": ["list of specific orientation issues found"],
            "recommendations": ["list of recommendations"]
        }
        
        For rotation_needed, use:
        - 0: No rotation needed
        - 90: Rotate 90° clockwise
        - 180: Rotate 180°
        - 270: Rotate 90° counterclockwise (or 270° clockwise)
        """
        
        response = model.generate_content([prompt, image])
        content = response.text.strip()
        
        # Try to parse JSON response
        try:
            import json
            # Remove markdown wrapping if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            analysis = json.loads(content)
            
            # Add basic metadata
            analysis["image_dimensions"] = {"width": width, "height": height}
            analysis["aspect_ratio"] = round(aspect_ratio, 2)
            
            # Determine if orientation is correct based on rotation needed
            analysis["orientation_correct"] = (analysis.get("rotation_needed", 0) == 0)
            
            print(f"✅ Orientation analysis completed for {os.path.basename(image_path)}")
            print(f"   Rotation needed: {analysis.get('rotation_needed', 0)}°")
            print(f"   Description: {analysis.get('rotation_description', 'Unknown')}")
            return analysis
            
        except json.JSONDecodeError:
            # Fallback analysis based on image properties
            print(f"⚠️ Could not parse Gemini response, using fallback analysis")
            return {
                "document_type": "unknown",
                "current_orientation": "unknown",
                "rotation_needed": 0,
                "rotation_description": "Could not determine",
                "orientation_correct": True,  # Assume correct if we can't determine
                "clarity": "unknown",
                "issues": ["Could not analyze with AI"],
                "recommendations": ["Manual review recommended"],
                "image_dimensions": {"width": width, "height": height},
                "aspect_ratio": round(aspect_ratio, 2)
            }
            
    except Exception as e:
        print(f"❌ Orientation check failed: {e}")
        return {
            "document_type": "unknown",
            "current_orientation": "unknown",
            "rotation_needed": 0,
            "rotation_description": "Analysis failed",
            "orientation_correct": False,
            "clarity": "unknown",
            "issues": [f"Analysis failed: {str(e)}"],
            "recommendations": ["Manual review required"],
            "image_dimensions": {"width": 0, "height": 0},
            "aspect_ratio": 0
        }


def auto_rotate_image_if_needed(image_path: str, analysis: dict = None) -> str:
    """
    Automatically rotate image based on orientation analysis.
    Returns the path to the corrected image.
    """
    try:
        if not analysis:
            analysis = check_image_orientation(image_path)
        
        rotation_needed = analysis.get("rotation_needed", 0)
        
        if rotation_needed == 0:
            print(f"✅ Image orientation is correct, no rotation needed")
            return image_path
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply the specific rotation needed
        rotated_image = image.rotate(rotation_needed, expand=True)
        
        # Save rotated image
        base_name = os.path.splitext(image_path)[0]
        rotated_path = f"{base_name}_rotated_{rotation_needed}deg.jpg"
        rotated_image.save(rotated_path, "JPEG", quality=95)
        
        print(f"✅ Image rotated {rotation_needed}° and saved as: {rotated_path}")
        return rotated_path
        
    except Exception as e:
        print(f"❌ Auto-rotation failed: {e}")
        return image_path


def extract_attestation_numbers_with_gemini_vision(image_path: str) -> dict:
    """
    Extract attestation numbers from certificate images using Gemini Vision.
    This provides more accurate extraction than OCR alone.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = """
        Analyze this UAE attestation document image and extract all attestation numbers, document numbers, and important identification numbers.
        
        IMPORTANT UAE ATTESTATION DOCUMENT STRUCTURE:
        1. Application Number (left side) = Attestation Number 1 - Look for this number positioned to the LEFT of the receipt number
        2. Receipt Number (right side) = Receipt Number - This is usually on the right side
        3. Label Number = Big numbers at the bottom of the label/stamp
        4. Multiple UAE Seals: If multiple seals found, prefer the one with AED currency
        5. Old Seal (stamp with Indian numerals): If found, use same number for both Attestation 1 and 2
        
        CRITICAL: Look for TWO separate numbers:
        - APPLICATION NUMBER: Usually smaller, positioned to the LEFT of the receipt number
        - RECEIPT NUMBER: Usually larger, positioned to the RIGHT of the application number
        
        Look for:
        - Application numbers positioned to the LEFT of receipt numbers
        - Receipt numbers positioned to the RIGHT of application numbers
        - Large numbers at the bottom of official labels/stamps
        - UAE Ministry of Foreign Affairs seals and stamps
        - Currency indicators (AED preferred)
        - Indian numeral stamps (older format)
        - Any other official identification numbers
        
        Pay special attention to:
        - Numbers that appear in pairs (application + receipt)
        - Relative positioning (left vs right)
        - Size differences between application and receipt numbers
        
        Return your analysis in this JSON format:
        {
            "attestation_number_1": "application number (left side) or null",
            "attestation_number_2": "number or null", 
            "receipt_number": "receipt number (right side) or null",
            "label_number": "number or null",
            "certificate_number": "number or null",
            "document_id": "number or null",
            "institution_reference": "number or null",
            "confidence": "high/medium/low",
            "extraction_notes": ["list of notes about what was found"],
            "recommendations": ["list of recommendations"]
        }
        
        For UAE attestation documents:
        - Attestation Number 1: Application number (LEFT side, usually smaller)
        - Attestation Number 2: Same as Attestation 1 if old Indian numeral stamp found, otherwise different number
        - Receipt Number: Receipt number (RIGHT side, usually larger)
        - Label Number: Large numbers at bottom of official label/stamp
        - Prefer AED currency seals over others
        """
        
        response = model.generate_content([prompt, image])
        content = response.text.strip()
        
        # Remove markdown wrapping if present
        if isinstance(content, str):
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        
        # Parse JSON response
        if isinstance(content, str):
            analysis = json.loads(content)
        else:
            analysis = content
            
        print(f"✅ Gemini Vision attestation extraction completed for {os.path.basename(image_path)}")
        return analysis
        
    except Exception as e:
        print(f"❌ Gemini Vision attestation extraction failed: {e}")
        return {
            "attestation_number_1": None,
            "attestation_number_2": None,
            "receipt_number": None,
            "label_number": None,
            "certificate_number": None,
            "document_id": None,
            "institution_reference": None,
            "confidence": "low",
            "extraction_notes": [f"Extraction failed: {e}"],
            "recommendations": ["Check image quality and try again"]
        }


def extract_document_data_with_gemini_vision(image_path: str) -> dict:
    """
    Extract comprehensive document data using Gemini Vision.
    This includes attestation numbers, personal info, and document details.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = """
        Analyze this UAE attestation document image and extract all important information including attestation numbers, personal details, and document information.
        
        IMPORTANT UAE ATTESTATION DOCUMENT STRUCTURE:
        1. Application Number (left side) = Attestation Number 1 - Look for this number positioned to the LEFT of the receipt number
        2. Receipt Number (right side) = Receipt Number - This is usually on the right side
        3. Label Number = Big numbers at the bottom of the label/stamp
        4. Multiple UAE Seals: If multiple seals found, prefer the one with AED currency
        5. Old Seal (stamp with Indian numerals): If found, use same number for both Attestation 1 and 2
        
        CRITICAL: Look for TWO separate numbers:
        - APPLICATION NUMBER: Usually smaller, positioned to the LEFT of the receipt number
        - RECEIPT NUMBER: Usually larger, positioned to the RIGHT of the application number
        
        Look for:
        - Application numbers positioned to the LEFT of receipt numbers
        - Receipt numbers positioned to the RIGHT of application numbers
        - Large numbers at the bottom of official labels/stamps
        - UAE Ministry of Foreign Affairs seals and stamps
        - Currency indicators (AED preferred)
        - Indian numeral stamps (older format)
        - Personal information (name, date of birth, nationality)
        - Institution details (university, faculty, department)
        - Qualification information (degree, major, grade)
        - Document dates and validity information
        
        Pay special attention to:
        - Numbers that appear in pairs (application + receipt)
        - Relative positioning (left vs right)
        - Size differences between application and receipt numbers
        
        Return your analysis in this JSON format:
        {
            "attestation_numbers": {
                "primary": "application number (left side) or null",
                "secondary": "number or null",
                "receipt": "receipt number (right side) or null",
                "label": "number or null"
            },
            "personal_info": {
                "full_name": "name or null",
                "date_of_birth": "date or null",
                "nationality": "nationality or null"
            },
            "institution": {
                "name": "institution name or null",
                "faculty": "faculty or null",
                "department": "department or null"
            },
            "qualification": {
                "degree": "degree or null",
                "major": "major or null",
                "grade": "grade or null"
            },
            "document_info": {
                "type": "document type or null",
                "issue_date": "date or null",
                "expiry_date": "date or null",
                "has_official_stamp": true/false
            },
            "confidence": "high/medium/low",
            "extraction_notes": ["list of notes about what was found"],
            "recommendations": ["list of recommendations"]
        }
        
        For UAE attestation documents:
        - Primary Attestation: Application number (LEFT side, usually smaller)
        - Secondary Attestation: Same as primary if old Indian numeral stamp found, otherwise different number
        - Receipt Number: Receipt number (RIGHT side, usually larger)
        - Label Number: Large numbers at bottom of official label/stamp
        - Prefer AED currency seals over others
        """
        
        response = model.generate_content([prompt, image])
        content = response.text.strip()
        
        # Remove markdown wrapping if present
        if isinstance(content, str):
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        
        # Parse JSON response
        if isinstance(content, str):
            analysis = json.loads(content)
        else:
            analysis = content
            
        print(f"✅ Gemini Vision comprehensive extraction completed for {os.path.basename(image_path)}")
        return analysis
        
    except Exception as e:
        print(f"❌ Gemini Vision comprehensive extraction failed: {e}")
        return {
            "attestation_numbers": {
                "primary": None,
                "secondary": None,
                "receipt": None,
                "label": None
            },
            "personal_info": {
                "full_name": None,
                "date_of_birth": None,
                "nationality": None
            },
            "institution": {
                "name": None,
                "faculty": None,
                "department": None
            },
            "qualification": {
                "degree": None,
                "major": None,
                "grade": None
            },
            "document_info": {
                "type": None,
                "issue_date": None,
                "expiry_date": None,
                "has_official_stamp": False
            },
            "confidence": "low",
            "extraction_notes": [f"Extraction failed: {e}"],
            "recommendations": ["Check image quality and try again"]
        }