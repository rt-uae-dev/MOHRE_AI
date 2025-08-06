from ultralytics import YOLO
import cv2
import os
from config import get_config

# === CONFIGURATION ===
config = get_config()
model_path = config.yolo_model_path
input_root = config.input_root
output_root = config.output_root

# Load model
model = YOLO(str(model_path))

# Recursively walk through subfolders
for root, _, files in os.walk(str(input_root)):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(root, filename)
            rel_path = os.path.relpath(input_path, input_root)
            subfolder = os.path.dirname(rel_path)

            # Output path (mirror structure)
            output_folder = os.path.join(output_root, subfolder)
            os.makedirs(output_folder, exist_ok=True)

            image = cv2.imread(input_path)
            if image is None:
                print(f"⚠️ Could not read image: {input_path}")
                continue

            results = model(image)[0]
            if not results.boxes:
                print(f"ℹ️ No detections in image: {rel_path}")
                continue

            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = image[y1:y2, x1:x2]

                base_name = os.path.splitext(os.path.basename(filename))[0]
                crop_name = f"{base_name}_crop{i+1}.jpg"
                crop_path = os.path.join(output_folder, crop_name)

                cv2.imwrite(crop_path, cropped)

print("✅ Cropping complete. Output saved in:", output_root)
