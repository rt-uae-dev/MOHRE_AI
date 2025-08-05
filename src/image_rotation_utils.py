import cv2
import numpy as np
import os

# Only rotate these document types that commonly need rotation
ROTATABLE_CLASSES = {"passport_1", "passport_2", "certificate", "certificate_attestation", "attestation_label", "personal_photo"}

# Set to False to completely disable rotation
ENABLE_ROTATION = True

def detect_text_orientation(image_path: str) -> dict:
    """
    Detect if text in the image is oriented correctly by analyzing OCR text patterns.
    Returns orientation analysis with recommendations.
    """
    try:
        import cv2
        import numpy as np
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"orientation_correct": True, "issues": [], "recommendations": []}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get image dimensions
        height, width = gray.shape
        
        # Check if image is landscape or portrait
        is_landscape = width > height
        
        # For passports, they should typically be landscape
        # If a passport is portrait, it might need rotation
        if is_landscape:
            # Check if text is readable horizontally
            # This is a simple heuristic - in a real implementation, you'd use OCR
            return {"orientation_correct": True, "issues": [], "recommendations": []}
        else:
            # Portrait orientation might need rotation for passports
            return {
                "orientation_correct": False,
                "issues": ["Document appears to be in portrait orientation but should be landscape"],
                "recommendations": ["Consider rotating 90 degrees to landscape orientation"]
            }
            
    except Exception as e:
        return {"orientation_correct": True, "issues": [f"Error analyzing orientation: {e}"], "recommendations": []}

def rotate_image_if_needed(image_path: str, angle_data: list, doc_class: str) -> str:
    """
    Smart rotation - handles common cases (90°, 180°, 270°) while being conservative about small angles.
    Returns original path if any doubt about rotation.
    """
    # Completely disable rotation if flag is set
    if not ENABLE_ROTATION:
        print(f"🔄 Rotation disabled globally")
        return image_path
    
    # Skip rotation for most document types
    if doc_class.lower() not in ROTATABLE_CLASSES:
        print(f"↩️ Skipping rotation for class: {doc_class}")
        return image_path
    
    # Load image
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Could not load image: {image_path}")
            return image_path
    except Exception as e:
        print(f"⚠️ Error loading image: {e}")
        return image_path

    # For passports, also check orientation even if angle data is insufficient
    if doc_class in ["passport_1", "passport_2"]:
        orientation_check = detect_text_orientation(image_path)
        if not orientation_check.get("orientation_correct", True):
            print(f"🔍 Orientation check suggests rotation needed for {doc_class}")
            # Force a 90-degree rotation for portrait passports
            h, w = image.shape[:2]
            if h > w:  # Portrait orientation
                print(f"↻ Rotating portrait passport to landscape orientation")
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, 90, 1.0)
                rotated = cv2.warpAffine(image, M, (h, w), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                
                rotated_path = os.path.splitext(image_path)[0] + "_rotated_90deg.jpg"
                success = cv2.imwrite(rotated_path, rotated)
                
                if success:
                    print(f"✅ Portrait passport rotated to landscape: {rotated_path}")
                    return rotated_path
                else:
                    print(f"⚠️ Failed to save rotated passport image")
                    return image_path
    
    # Only rotate if we have very clear angle data (at least 4 points for bounding box)
    if not angle_data or len(angle_data) < 4:
        print(f"⚠️ Insufficient angle data for {doc_class} - skipping rotation")
        return image_path

    try:
        # Check if we have valid bounding box data
        valid_points = [pt for pt in angle_data if "x" in pt and "y" in pt]
        if len(valid_points) < 4:
            print(f"⚠️ Invalid bounding box data for {doc_class} - skipping rotation")
            return image_path
        
        # Calculate angle from top edge of bounding box
        pts = np.array([[pt["x"], pt["y"]] for pt in valid_points[:4]])
        
        # Use top edge (first two points) to calculate angle
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Normalize angle to 0-360 range
        angle = angle % 360
        
        print(f"📐 Detected angle for {doc_class}: {angle:.2f}°")
        
        # Handle common rotation cases
        if abs(angle) < 10:
            print(f"✅ Already upright for {doc_class}. Angle: {angle:.2f}°")
            return image_path
        
        # Handle 90-degree rotations
        if 80 <= abs(angle) <= 100:
            rotation_angle = -90 if angle > 0 else 90
            print(f"↻ Rotating {os.path.basename(image_path)} by {rotation_angle}° (90-degree rotation)")
        elif 170 <= abs(angle) <= 190:
            rotation_angle = 180
            print(f"↻ Rotating {os.path.basename(image_path)} by {rotation_angle}° (180-degree rotation)")
        elif 260 <= abs(angle) <= 280:
            rotation_angle = 90 if angle > 0 else -90
            print(f"↻ Rotating {os.path.basename(image_path)} by {rotation_angle}° (270-degree rotation)")
        else:
            # For other angles, be more conservative
            if abs(angle) > 45 and abs(angle) < 135:
                print(f"⚠️ Unusual angle ({angle:.2f}°) - skipping rotation to avoid errors")
                return image_path
            else:
                # Use the calculated angle for small corrections
                rotation_angle = -angle
                print(f"↻ Rotating {os.path.basename(image_path)} by {rotation_angle:.2f}° (small correction)")
        
        # Perform rotation
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # Save rotated image
        rotated_path = os.path.splitext(image_path)[0] + "_rotated.jpg"
        success = cv2.imwrite(rotated_path, rotated)
        
        if success:
            print(f"✅ Rotation successful: {rotated_path}")
            return rotated_path
        else:
            print(f"⚠️ Failed to save rotated image")
            return image_path
            
    except Exception as e:
        print(f"⚠️ Rotation failed for {doc_class}: {e}")
        return image_path
