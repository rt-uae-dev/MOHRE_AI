#!/usr/bin/env python3
"""
Simple Gemini 2.5 Flash orientation detector.
Just asks: "Does this image need to be rotated?"
"""

import os
import base64
import json
import google.generativeai as genai
import cv2

def ask_gemini_if_needs_rotation(image_path: str) -> dict:
    """
    Ask Gemini 2.5 Flash: "Does this image need to be rotated?"
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict with rotation_needed (bool), rotation_angle (int), and reason (str)
    """
    try:
        # Configure Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        # Use Gemini 2.5 Flash
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        # Create the simple prompt with better instructions for all document types
        prompt = """
        This is a document image (passport, certificate, ID card, etc.). 
        
        IMPORTANT RULES:
        1. Look at the MAIN content (text, photos, official stamps, logos) to determine orientation.
        2. If the main text is upside-down or sideways, the image needs rotation.
        3. If the document title/header is upside-down, the image needs rotation.
        4. If official stamps or seals are upside-down, the image needs rotation.
        5. For certificates: If the university name, degree title, or main text is upside-down, rotate it.
        6. For IDs: If the photo or main text is upside-down, rotate it.
        7. For passports: If the passport photo or main text is upside-down, rotate it.
        8. Only ignore rotation if ALL main content appears correctly oriented.
        
        If rotation is needed, tell me the angle of rotation (90, 180, 270 degrees).
        Only reply with a single number: 0, 90, 180, or 270.
        
        If the image is correctly oriented, reply with 0.
        """
        
        # Generate response from Gemini
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
        
        # Extract the rotation angle from response
        try:
            rotation_angle = int(response.text.strip())
            if rotation_angle not in [0, 90, 180, 270]:
                rotation_angle = 0
        except ValueError:
            rotation_angle = 0
        
        # Determine if rotation is needed
        rotation_needed = rotation_angle != 0
        
        # Create reason based on the angle
        if rotation_angle == 0:
            reason = "Gemini 2.5 Flash detected: Image is correctly oriented"
        else:
            reason = f"Gemini 2.5 Flash detected: Image needs {rotation_angle}° rotation"
        
        print(f"🤖 Gemini 2.5 Flash says: {rotation_angle}° rotation needed")
        
        return {
            'rotation_needed': rotation_needed,
            'rotation_angle': rotation_angle,
            'reason': reason
        }
                
    except Exception as e:
        print(f"❌ Error in Gemini 2.5 Flash orientation detection: {e}")
        return {
            'rotation_needed': False,
            'rotation_angle': 0,
            'reason': f"Error occurred: {str(e)}"
        }

def rotate_if_needed(image_path: str) -> str:
    """
    Ask Gemini 2.5 Flash if image needs rotation and rotate if needed.
    Includes verification step to ensure rotation was correct.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Path to the corrected image (original if no rotation needed)
    """
    print(f"🔍 Asking Gemini 2.5 Flash: Does this image need rotation?")
    print(f"   Image: {os.path.basename(image_path)}")
    
    # Ask Gemini 2.5 Flash
    result = ask_gemini_if_needs_rotation(image_path)
    
    print(f"📊 Gemini 2.5 Flash says:")
    print(f"   Rotation needed: {result['rotation_needed']}")
    print(f"   Reason: {result['reason']}")
    
    # Rotate if needed
    if result['rotation_needed']:
        print(f"🔄 Rotating image by {result['rotation_angle']}°...")
        
        # Read and rotate image
        image = cv2.imread(image_path)
        
        # Apply the correct rotation
        # When Gemini says "X° rotation needed", it means the image needs X° rotation to be correct
        # So we apply the opposite rotation to fix it
        if result['rotation_angle'] == 90:
            print(f"   ↺ Rotating 90° counterclockwise to correct orientation")
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif result['rotation_angle'] == 180:
            print(f"   ↻ Rotating 180° to correct orientation")
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        elif result['rotation_angle'] == 270:
            print(f"   ↺ Rotating 90° counterclockwise to correct orientation")
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            print(f"❌ Invalid rotation angle: {result['rotation_angle']}")
            return image_path
        
        # Save rotated image
        base_name = os.path.splitext(image_path)[0]
        extension = os.path.splitext(image_path)[1]
        output_path = f"{base_name}_rotated{extension}"
        
        cv2.imwrite(output_path, rotated_image)
        print(f"✅ Rotated image saved to: {output_path}")
        
        # 🔄 VERIFICATION STEP: Ask Gemini again to verify the rotation was correct
        print(f"🔄 Verifying rotation was correct...")
        verification_result = ask_gemini_if_needs_rotation(output_path)
        
        if verification_result['rotation_needed']:
            print(f"⚠️ Gemini says rotation is still needed! Trying opposite rotation...")
            
            # If Gemini says it still needs rotation, try the opposite direction
            if result['rotation_angle'] == 90:
                print(f"   ↻ Trying 90° clockwise instead...")
                corrected_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif result['rotation_angle'] == 270:
                print(f"   ↻ Trying 90° clockwise instead...")
                corrected_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif result['rotation_angle'] == 180:
                print(f"   ↻ Trying 180° again (might be the same)...")
                corrected_image = cv2.rotate(image, cv2.ROTATE_180)
            else:
                print(f"❌ Unknown rotation angle, keeping original")
                return image_path
            
            # Save corrected image
            corrected_output_path = f"{base_name}_corrected{extension}"
            cv2.imwrite(corrected_output_path, corrected_image)
            print(f"✅ Corrected image saved to: {corrected_output_path}")
            
            # Final verification
            final_verification = ask_gemini_if_needs_rotation(corrected_output_path)
            if not final_verification['rotation_needed']:
                print(f"✅ Final verification successful - rotation is now correct!")
                return corrected_output_path
            else:
                print(f"⚠️ Still not correct, keeping original image")
                return image_path
        else:
            print(f"✅ Verification successful - rotation was correct!")
            return output_path
    else:
        print(f"✅ No rotation needed - keeping original image")
        return image_path

if __name__ == "__main__":
    # Test with a sample image
    test_image = "data/dataset/passport_1/01. Passport Copy _Unaise_page_1_1_1_1.jpg"
    
    if os.path.exists(test_image):
        print(f"🧪 Testing Gemini 2.5 Flash orientation detection...")
        result = rotate_if_needed(test_image)
        print(f"Final result: {result}")
    else:
        print(f"Test image not found: {test_image}") 