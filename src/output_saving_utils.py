"""Helpers for saving processed outputs and logging results."""

import os
import json
import shutil
from typing import Any, Dict, Optional


def save_outputs(
    jpg_path: str,
    structured_json: Dict[str, Any],
    output_dir: str,
    base_name: str,
    gemini_response: Optional[str] = None,
) -> str:
    """Save the final compressed JPEG and its structured JSON.

    Args:
        jpg_path: Path to the compressed JPEG file to be saved.
        structured_json: Parsed document data to persist alongside the image.
        output_dir: Destination directory for the outputs.
        base_name: Base name (without extension) for the saved files.
        gemini_response: Optional raw response from Gemini for debugging.

    Returns:
        The full path to the saved JPEG file.

    Raises:
        RuntimeError: If copying the image or writing the JSON fails.
    """
    os.makedirs(output_dir, exist_ok=True)

    final_jpg_path = os.path.join(output_dir, base_name + ".jpg")
    final_json_path = os.path.join(output_dir, base_name + ".json")

    try:
        shutil.copy2(jpg_path, final_jpg_path)
    except Exception as e:  # pragma: no cover - filesystem issues
        raise RuntimeError(f"❌ Failed to copy image: {e}")

    try:
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(structured_json, f, ensure_ascii=False, indent=2)
    except Exception as e:  # pragma: no cover - filesystem issues
        raise RuntimeError(f"❌ Failed to write JSON: {e}")

    if gemini_response:
        debug_path = os.path.join(output_dir, base_name + "_raw.txt")
        with open(debug_path, "w", encoding="utf-8") as debug_file:
            debug_file.write(gemini_response)

    return final_jpg_path


def log_processed_file(log_file: str, original_file: str, saved_jpg_path: str, label: str) -> None:
    """Append a log entry describing a processed document.

    Args:
        log_file: Path to the log file.
        original_file: Original source file that was processed.
        saved_jpg_path: Location of the compressed JPEG on disk.
        label: Detected document label.

    Returns:
        None
    """
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"Processed File: {original_file}\n")
        log.write(f"Saved JPG:     {saved_jpg_path}\n")
        log.write(f"Document Type: {label}\n")
        log.write("-" * 40 + "\n")
