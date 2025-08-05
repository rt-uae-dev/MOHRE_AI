import os
import io
from dotenv import load_dotenv
from google.cloud import vision
from openai import OpenAI

# === Step 1: Load environment variables from .env ===
load_dotenv()

# ✅ Set credentials from environment
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

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

# === Step 3: Ask GPT-4o to extract structured passport fields ===
def ask_gpt_to_structure_passport_data(ocr_text):
    system_message = "You are a smart assistant that extracts structured fields from passport OCR."
    user_prompt = f"""
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

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# === Step 4: Run everything ===
def main():
    # 🖼️ Update this to your image filename
    image_path = r"C:\Users\Husai\Desktop\mohre-email-parser\COMPLETED\passport_1\Alwin John Passport Copy_page_1_1_1_1_crop2.jpg"

    print("🔍 Extracting text with Google Vision...")
    ocr_text = extract_text_with_google_vision(image_path)
    print("\n🧾 OCR Result Preview:\n", ocr_text[:500], "\n...")

    print("🤖 Sending to GPT-4o for field extraction...")
    structured_output = ask_gpt_to_structure_passport_data(ocr_text)
    print("\n✅ Extracted Passport Fields:\n", structured_output)

if __name__ == "__main__":
    main()
