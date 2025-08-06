import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from pdf2image import convert_from_path
import os
from pathlib import Path
from config import get_config

# === CONFIG ===
config = get_config()
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = str(config.model_path)  # Updated model path
IMG_SIZE = 224
ROOT_DIR = config.root_dir
DATASET_DIR = config.dataset_dir
TEMP_JPG_PATH = "temp_passport_page.jpg"

# === Disable DecompressionBombWarning ===
Image.MAX_IMAGE_PIXELS = None

# === Load Model ===
# Discover classes from training (assumes order matches training)
CLASS_NAMES = sorted([
    folder for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, folder))
])

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Walk through all folders and PDFs ===
for folder in os.listdir(ROOT_DIR):
    folder_path = ROOT_DIR / folder
    if not folder_path.is_dir():
        continue

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            pdf_path = folder_path / file
            try:
                pages = convert_from_path(str(pdf_path), dpi=300, first_page=1, last_page=1)
                pages[0].save(TEMP_JPG_PATH, 'JPEG')

                image = Image.open(TEMP_JPG_PATH).convert("RGB")
                image_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(image_tensor)
                    _, predicted = torch.max(outputs, 1)
                    predicted_class = CLASS_NAMES[predicted.item()]

                print(f"[{file}] → {predicted_class}")

            except Exception as e:
                print(f"Error processing {file}: {e}")

# Clean up temp image if it exists
if os.path.exists(TEMP_JPG_PATH):
    os.remove(TEMP_JPG_PATH)
