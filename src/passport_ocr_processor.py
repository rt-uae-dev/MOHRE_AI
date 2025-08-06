import os
import io
from google.cloud import vision
import google.generativeai as genai
from pathlib import Path
from config import get_config

# === Step 1: Load configuration ===
config = get_config()

# ✅ Set credentials from environment
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(config.google_application_credentials)
genai.configure(api_key=config.google_api_key)

# === Step 2: Use Google Vision to extract text from image ===
def extract_text_with_google_vision(image_path):
    vision_client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    response = vision_client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        return ""

    return texts[0].description.strip()

# === Step 3: Use Gemini to extract structured passport fields ===
def ask_gemini_to_structure_passport_data(ocr_text):
    prompt = f"""
Given the OCR text of a passport, extract the following fields and return them as a JSON object:

- Full Name
- Passport Number
- Nationality
- Date of Birth
- Sex
- Place of Birth
- Date of Issue
- Date of Expiry
- Place of Issue

OCR TEXT:
{ocr_text}
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# === Step 4: Run everything ===
def main():
    # 🖼️ Image path can be configured via environment variable
    image_path = str(config.passport_image_path)

    print("🔍 Extracting text with Google Vision...")
    ocr_text = extract_text_with_google_vision(image_path)
    print("\n🧾 OCR Result Preview:\n", ocr_text[:500], "\n...")

    print("🤖 Sending to Gemini for field extraction...")
    structured_output = ask_gemini_to_structure_passport_data(ocr_text)
    print("\n✅ Extracted Passport Fields:\n", structured_output)

if __name__ == "__main__":
    main()
