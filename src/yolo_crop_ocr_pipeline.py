import os
import io
import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
from google.cloud import vision

# Load environment variables first
load_dotenv()

# Import Document AI processor
try:
    from document_ai_processor import DOCUMENT_AI_PROCESSOR
    DOCUMENT_AI_AVAILABLE = True
    print("✅ Document AI processor imported successfully")
except ImportError as e:
    DOCUMENT_AI_AVAILABLE = False
    print(f"⚠️ Document AI not available: {e}")

# === LOAD .env VARIABLES ===
load_dotenv()

# === CONFIG ===
YOLO_MODEL_PATH = "models/yolo8_best.pt"
YOLO_MODEL = YOLO(YOLO_MODEL_PATH)

VISION_CLIENT = vision.ImageAnnotatorClient()

def run_enhanced_ocr(image_path: str) -> dict:
    """
    Run OCR using Google Document AI. No other OCR engines are used.
    Returns comprehensive OCR data with confidence scores.
    """
    filename = os.path.basename(image_path)
    print(f"🔍 Processing {filename} with Document AI (primary)...")
    
    # Run Document AI OCR if available
    if DOCUMENT_AI_AVAILABLE and DOCUMENT_AI_PROCESSOR.enabled:
        try:
            print("   📄 Processing with Document AI Document OCR Processor...")
            doc_ai_result = DOCUMENT_AI_PROCESSOR.process_document(image_path)

            if "error" not in doc_ai_result:
                print("   ✅ Document AI processing successful!")

                # Extract comprehensive fields based on document type
                full_text = doc_ai_result.get('full_text', '')
                document_type = DOCUMENT_AI_PROCESSOR.get_document_type(full_text)
                extracted_fields = DOCUMENT_AI_PROCESSOR.extract_fields_by_document_type(full_text)
                confidence = doc_ai_result.get('confidence', 0.0)

                print(f"   📋 Document Type: {document_type}")
                print(f"   🎯 Confidence: {confidence:.2f}")
                print(f"   📝 Extracted Fields: {len(extracted_fields)}")

                return {
                    "ocr_text": full_text,
                    "confidence": confidence,
                    "document_type": document_type,
                    "extracted_fields": extracted_fields,
                    "ocr_method": "document_ai",
                    "text_blocks": doc_ai_result.get('ocr_data', {}).get('text_blocks', []),
                    "page_count": doc_ai_result.get('pages', 1)
                }
            else:
                print(f"   ❌ Document AI failed: {doc_ai_result['error']}")
        except Exception as e:
            print(f"   ❌ Document AI error: {e}")

    # Document AI unavailable or failed
    return {
        "ocr_text": "",
        "confidence": 0.0,
        "document_type": "unknown",
        "extracted_fields": {},
        "ocr_method": "document_ai_unavailable",
        "text_blocks": [],
        "page_count": 0,
    }

def preprocess_image_for_ocr(image_path: str) -> str:
    """
    Preprocess image to improve OCR accuracy.
    Returns path to preprocessed image.
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return image_path
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply slight blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    
    # Apply adaptive thresholding to improve text contrast
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Save preprocessed image
    preprocessed_path = os.path.splitext(image_path)[0] + "_preprocessed.jpg"
    cv2.imwrite(preprocessed_path, thresh)
    
    return preprocessed_path

def run_yolo_crop(image_path: str, output_dir: str) -> str:
    """
    Runs YOLO on the image and saves the cropped region to output_dir.
    Returns the cropped image path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"⚠️ Could not read image: {image_path}")

    results = YOLO_MODEL(image)[0]
    if not results.boxes:
        raise ValueError(f"⚠️ No objects detected in image: {image_path}")

    # Get all detected objects and their labels
    detected_labels = []
    for box in results.boxes:
        if hasattr(box, 'cls') and box.cls is not None:
            class_id = int(box.cls[0])
            # Get class name from YOLO model
            class_name = YOLO_MODEL.names[class_id] if hasattr(YOLO_MODEL, 'names') else f"class_{class_id}"
            detected_labels.append(class_name)
    
    # Check if attestation_label is detected
    has_attestation_label = any("attestation" in label.lower() for label in detected_labels)
    
    if has_attestation_label:
        # For attestation labels, crop the exact label area for OCR extraction
        print(f"🔍 Attestation label detected - cropping exact label area for OCR")
        x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0])
        
        # Crop exactly what YOLO detected - no margins
        cropped = image[y1:y2, x1:x2]
        print(f"📏 Attestation label cropped: {x2-x1}x{y2-y1} pixels")
    else:
        # For other documents, crop the exact detected area
        x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0])
        
        # Crop exactly what YOLO detected - no margins
        cropped = image[y1:y2, x1:x2]
        
        # Show cropping info
        height, width = image.shape[:2]
        original_area = width * height
        cropped_area = (x2 - x1) * (y2 - y1)
        reduction = ((original_area - cropped_area) / original_area) * 100
        
        print(f"🔍 Cropped to exact YOLO detection area")
        print(f"📏 Original: {width}x{height} ({original_area:,}px)")
        print(f"📏 Cropped: {x2-x1}x{y2-y1} ({cropped_area:,}px)")
        print(f"📉 Size reduction: {reduction:.1f}%")

    # Create a new filename for the cropped image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
    cv2.imwrite(output_path, cropped)
    return output_path

def run_google_vision_ocr(image_path: str) -> dict:
    """
    Enhanced OCR with multiple attempts to capture all text fields.
    Returns comprehensive OCR data.
    """
    # Try original image first
    ocr_result = _perform_ocr(image_path)
    
    # If OCR text is too short or missing key fields, try preprocessed image
    if len(ocr_result["ocr_text"]) < 100 or "passport" in image_path.lower():
        print(f"🔄 Attempting enhanced OCR for {os.path.basename(image_path)}...")
        preprocessed_path = preprocess_image_for_ocr(image_path)
        enhanced_result = _perform_ocr(preprocessed_path)
        
        # Combine results - use longer/more complete text
        if len(enhanced_result["ocr_text"]) > len(ocr_result["ocr_text"]):
            print(f"✅ Enhanced OCR captured more text")
            ocr_result = enhanced_result
        else:
            print(f"ℹ️ Original OCR was sufficient")
        
        # Clean up preprocessed file
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
    
    return ocr_result

def _perform_ocr(image_path: str) -> dict:
    """
    Perform OCR using Google Vision API with multiple detection methods.
    """
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    
    # Try document text detection first (better for structured documents)
    doc_response = VISION_CLIENT.document_text_detection(image=image)
    doc_text = doc_response.full_text_annotation.text if doc_response.full_text_annotation else ""
    
    # Also try regular text detection (might catch different text)
    text_response = VISION_CLIENT.text_detection(image=image)
    text_annotations = text_response.text_annotations if text_response.text_annotations else []
    
    # Combine text from both methods
    combined_text = doc_text
    if text_annotations:
        # Add any additional text found by text detection
        additional_text = " ".join([annotation.description for annotation in text_annotations[1:] if annotation.description])
        if additional_text and additional_text not in combined_text:
            combined_text += " " + additional_text
    
    # Get angle data from document detection
    if doc_response.text_annotations:
        angle = [
            {'x': v.x, 'y': v.y}
            for v in doc_response.text_annotations[0].bounding_poly.vertices
        ]
    else:
        angle = []
    
    # Get labels
    labels = [label.description for label in doc_response.label_annotations] if doc_response.label_annotations else []
    
    return {
        "ocr_text": combined_text,
        "angle": angle,
        "labels": labels
    }
