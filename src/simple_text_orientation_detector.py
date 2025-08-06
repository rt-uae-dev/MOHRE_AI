#!/usr/bin/env python3
"""Simple Text-Based Orientation Detector
Uses Google Vision OCR to detect if text is sideways without complex AI models.
"""

import cv2
import os
import tempfile
from yolo_crop_ocr_pipeline import run_google_vision_ocr


def detect_text_orientation_simple(image_path: str) -> dict:
    """Simple text-based orientation detection using Google Vision OCR.

    Args:
        image_path: Path to the image file

    Returns:
        dict with keys:
        - rotation_needed: bool
        - rotation_angle: int (0, 90, 180, 270)
        - confidence: float
        - reason: str
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {
                "rotation_needed": False,
                "rotation_angle": 0,
                "confidence": 0.0,
                "reason": "Failed to read image",
                "detected_text_length": 0,
            }

        orientations = [
            (0, "0° (original)"),
            (90, "90° clockwise"),
            (180, "180° clockwise"),
            (270, "270° clockwise"),
        ]

        best_orientation = 0
        best_text_len = 0

        for angle, description in orientations:
            if angle == 0:
                temp_path = image_path
            else:
                if angle == 90:
                    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    rotated = cv2.rotate(image, cv2.ROTATE_180)
                elif angle == 270:
                    rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(temp_file.name, rotated)
                temp_path = temp_file.name

            try:
                ocr_result = run_google_vision_ocr(temp_path)
                text_len = len(ocr_result.get("ocr_text", ""))
                print(f"  {description}: Text length={text_len}")
                if text_len > best_text_len:
                    best_text_len = text_len
                    best_orientation = angle
            finally:
                if angle != 0 and os.path.exists(temp_path):
                    os.remove(temp_path)

        rotation_needed = best_orientation != 0
        rotation_angle = best_orientation

        if rotation_needed:
            reason = f"Google Vision OCR detected more text at {rotation_angle}° rotation"
        else:
            reason = "Text appears correctly oriented"

        return {
            "rotation_needed": rotation_needed,
            "rotation_angle": rotation_angle,
            "confidence": float(best_text_len),
            "reason": reason,
            "detected_text_length": best_text_len,
        }

    except Exception as e:
        return {
            "rotation_needed": False,
            "rotation_angle": 0,
            "confidence": 0.0,
            "reason": f"Error: {str(e)}",
            "detected_text_length": 0,
        }


def rotate_image_simple(image_path: str, angle: int) -> str:
    """Rotate image by specified angle and save to new file.

    Args:
        image_path: Path to original image
        angle: Rotation angle (90, 180, 270)

    Returns:
        Path to rotated image
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to read image")

        if angle == 90:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            raise ValueError(f"Invalid rotation angle: {angle}")

        base_name = os.path.splitext(image_path)[0]
        extension = os.path.splitext(image_path)[1]
        output_path = f"{base_name}_rotated_{angle}deg{extension}"

        cv2.imwrite(output_path, rotated_image)

        print(f"✅ Image rotated {angle}° and saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error rotating image: {e}")
        return image_path


def auto_rotate_image_simple(image_path: str) -> str:
    """Automatically detect and rotate image if needed using Google Vision OCR."""
    print(f"🔍 Analyzing text orientation for: {os.path.basename(image_path)}")

    result = detect_text_orientation_simple(image_path)

    print("📊 Results:")
    print(f"  Rotation needed: {result['rotation_needed']}")
    print(f"  Rotation angle: {result['rotation_angle']}°")
    print(f"  Confidence: {result['confidence']:.1f}")
    print(f"  Reason: {result['reason']}")

    if result['rotation_needed'] and result['rotation_angle'] > 0:
        print(f"🔄 Rotating image {result['rotation_angle']}°...")
        return rotate_image_simple(image_path, result['rotation_angle'])
    else:
        print("✅ No rotation needed - keeping original image")
        return image_path


def test_simple_orientation_detection():
    """Test the simple text-based orientation detection on sample images."""
    test_images = [
        "data/dataset/passport_1/01. Passport Copy _Unaise_page_1_1_1_1.jpg",
        "data/dataset/certificate/03. Diploma In Mechanical with UAE Attestation_page_2_1.jpg",
        "data/dataset/emirates_id/02.ra_EID_Abhishek2034_page_1_1_1_1.jpg",
    ]

    for image_path in test_images:
        if os.path.exists(image_path):
            print("\n=== Testing:", os.path.basename(image_path), "===")
            auto_rotate_image_simple(image_path)
        else:
            print(f"⚠️ Test image not found: {image_path}")


if __name__ == "__main__":
    test_simple_orientation_detection()

