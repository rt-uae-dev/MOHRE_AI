import os
import shutil
from pdf2image import convert_from_path
from PIL import Image, ImageOps, ImageChops
import torch
import torchvision.transforms as transforms
from torchvision import models
import fitz  # from PyMuPDF
from pathlib import Path
from dotenv import load_dotenv


# === CONFIG ===
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_CLASSES_PATH = Path(os.getenv("DATASET_CLASSES_PATH", BASE_DIR / "data" / "dataset"))
INPUT_DIR = Path(os.getenv("INPUT_DIR", BASE_DIR / "data" / "raw" / "downloads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "data" / "processed" / "MOHRE_ready"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "model_classifier.pt"))
TEMP_JPG = "temp.jpg"
IMG_SIZE = 224
MAX_OUTPUT_KB = 110

Image.MAX_IMAGE_PIXELS = None

# === Load AI Classifier ===
CLASS_NAMES = sorted([
    folder for folder in os.listdir(DATASET_CLASSES_PATH)
    if os.path.isdir(os.path.join(DATASET_CLASSES_PATH, folder))
])
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(str(MODEL_PATH), map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Helper Functions ===
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def compress_image(img, out_path):
    quality = 95
    while quality > 20:
        img.save(out_path, format='JPEG', quality=quality, optimize=True)
        if os.path.getsize(out_path) <= MAX_OUTPUT_KB * 1024:
            break
        quality -= 5

def crop_if_needed(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageOps.invert(diff.convert("L"))
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)
    return image

def classify_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return CLASS_NAMES[predicted.item()]

# === Process Documents ===
def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if not file.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(folder_path, file)
            print(f"\n📄 Processing: {file}")
            try:
                # Step 1: Extract text
                ocr_text = extract_text_from_pdf(pdf_path)

                # Step 2: Convert all pages to images
                pages = convert_from_path(pdf_path, dpi=300)

                for i, page in enumerate(pages):
                    temp_image_path = f"temp_page_{i+1}.jpg"
                    page.save(temp_image_path, 'JPEG')

                    # Step 3: AI classify image
                    predicted_class = classify_image(temp_image_path)
                    print(f"→ Page {i+1} classified as: {predicted_class}")

                    # Step 4: Crop if needed
                    image = Image.open(temp_image_path).convert("RGB")
                    if predicted_class in ["passport_1", "passport_2", "personal_photo", "form_signature_page"]:
                        image = crop_if_needed(image)

                    # Step 5: Compress and save
                    final_name = f"{os.path.splitext(file)[0]}_page{i+1}__{predicted_class}.jpg"
                    final_path = os.path.join(output_dir, final_name)
                    compress_image(image, final_path)
                    print(f"✅ Saved: {final_path}")

                    os.remove(temp_image_path)

            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

# === RUN ===
if __name__ == "__main__":
    process_folder(INPUT_DIR, OUTPUT_DIR)
