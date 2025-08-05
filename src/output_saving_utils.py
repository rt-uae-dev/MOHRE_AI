import os
import json
import shutil
# Add timestamp or UUID
from datetime import datetime
import os
import json
import shutil
from datetime import datetime

def save_outputs(jpg_path: str, structured_json: dict, output_dir: str, base_name: str, gemini_response: str = None) -> str:
    """
    Save the final compressed JPG and structured JSON to output directory.
    Returns the full path to the saved JPG.
    """
    os.makedirs(output_dir, exist_ok=True)

    # No timestamp needed - keep base_name as is
    final_jpg_path = os.path.join(output_dir, base_name + ".jpg")
    final_json_path = os.path.join(output_dir, base_name + ".json")

    try:
        shutil.copy2(jpg_path, final_jpg_path)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to copy image: {e}")

    try:
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(structured_json, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to write JSON: {e}")

    return final_jpg_path

def log_processed_file(log_file: str, original_file: str, saved_jpg_path: str, label: str):
    """
    Append log entry to a .txt file for processed documents.
    """
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"Processed File: {original_file}\n")
        log.write(f"Saved JPG:     {saved_jpg_path}\n")
        log.write(f"Document Type: {label}\n")
        log.write("-" * 40 + "\n")
