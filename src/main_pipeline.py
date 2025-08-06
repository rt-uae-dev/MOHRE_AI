#!/usr/bin/env python3
"""
Main pipeline for MOHRE document processing.

The pipeline is split into logical steps that can be tested individually:
1. fetch_emails
2. convert_documents
3. classify_images
4. perform_ocr
5. gemini_structuring
6. save_results
"""

import os
import shutil
import subprocess
import platform
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

from email_parser import fetch_and_store_emails
from pdf_converter import convert_pdf_to_jpg
from resnet18_classifier import classify_image_resnet
from yolo_crop_ocr_pipeline import run_yolo_crop, run_enhanced_ocr
from structure_with_gemini import structure_with_gemini
from output_saving_utils import save_outputs, log_processed_file
from image_utils import compress_image_to_jpg
from google_vision_orientation_detector import rotate_if_needed
from parse_salary_docx import parse_salary_docx

INPUT_DIR = "data/raw/downloads"
OUTPUT_DIR = "data/processed/COMPLETED"
TEMP_DIR = "data/temp"
LOG_FILE = "logs/process_log.txt"


@dataclass
class PipelineContext:
    """Shared configuration for the pipeline."""
    input_dir: str = INPUT_DIR
    output_dir: str = OUTPUT_DIR
    temp_dir: str = TEMP_DIR
    log_file: str = LOG_FILE


def open_file_explorer(directory_path: str) -> None:
    try:
        if platform.system() == "Windows":
            subprocess.run(["explorer", directory_path], check=True)
        elif platform.system() == "Darwin":
            subprocess.run(["open", directory_path], check=True)
        else:
            subprocess.run(["xdg-open", directory_path], check=True)
        print(f"📂 Opened file explorer to: {directory_path}")
    except Exception as e:
        print(f"⚠️ Could not open file explorer: {e}")
        print(f"📂 Please manually navigate to: {os.path.abspath(directory_path)}")


def fetch_emails(context: PipelineContext) -> None:
    print("📧 Fetching emails...")
    fetch_and_store_emails()


def convert_documents(context: PipelineContext, subject_path: str) -> List[str]:
    print("🔄 Converting PDFs to JPGs...")
    all_image_paths: List[str] = []
    for filename in os.listdir(subject_path):
        file_path = os.path.join(subject_path, filename)
        if filename.lower().endswith(".pdf"):
            print(f"📄 Converting: {filename}")
            jpg_paths = convert_pdf_to_jpg(file_path, context.temp_dir)
            all_image_paths.extend(jpg_paths)
        elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
            temp_path = os.path.join(context.temp_dir, filename)
            shutil.copy2(file_path, temp_path)
            all_image_paths.append(temp_path)
            print(f"📷 Copied image: {filename}")
    return all_image_paths


def classify_images(context: PipelineContext, image_paths: List[str]) -> List[Dict]:
    print("🏷️ Classifying images with ResNet...")
    classified_images: List[Dict] = []
    for img_path in image_paths:
        try:
            resnet_label = classify_image_resnet(img_path)
            classified_images.append({
                "path": img_path,
                "label": resnet_label,
                "filename": os.path.basename(img_path),
            })
            print(f"✅ {os.path.basename(img_path)} → {resnet_label}")
        except Exception as e:
            print(f"❌ Error classifying {os.path.basename(img_path)}: {e}")

    has_certificate = any(img["label"] == "certificate" for img in classified_images)
    has_attestation = any(img["label"] in ["certificate_attestation", "attestation_label"] for img in classified_images)
    if has_certificate and not has_attestation:
        print("⚠️ Certificate found but no attestation page. Looking for misclassified attestation...")
        for img_data in classified_images:
            if img_data["label"] in ["emirates_id", "emirates_id_2", "unknown"]:
                img_data["label"] = "attestation_label"
                print(f"🔄 Reclassified {img_data['filename']} as attestation_label")

    rotation_check_types = ["passport_1", "passport_2", "personal_photo", "certificate"]
    for img_data in classified_images:
        try:
            if img_data["label"] in rotation_check_types:
                rotated_path = rotate_if_needed(img_data["path"])
                if rotated_path != img_data["path"]:
                    img_data["path"] = rotated_path
                    print(f"✅ Rotated {img_data['filename']} ({img_data['label']})")
            else:
                print(f"⏭️ Skipping rotation for {img_data['filename']} ({img_data['label']})")
        except Exception as e:
            print(f"⚠️ Error rotating {img_data['filename']}: {e}")

    return classified_images


def perform_ocr(context: PipelineContext, classified_images: List[Dict]) -> List[Dict]:
    print("📝 Running OCR for all documents...")
    processed_images: List[Dict] = []
    for img_data in classified_images:
        try:
            cropped_path = run_yolo_crop(img_data["path"], context.temp_dir)
            if cropped_path:
                img_data["cropped_path"] = cropped_path
            ocr_path = img_data.get("cropped_path") or img_data.get("full_page_path") or img_data["path"]
            vision_data = run_enhanced_ocr(ocr_path)
            img_data["ocr_text"] = vision_data.get("ocr_text", "")
            img_data["extracted_fields"] = vision_data.get("extracted_fields", {})
            img_data["document_type"] = vision_data.get("document_type", "unknown")
            img_data["confidence"] = vision_data.get("confidence", 0.0)
            processed_images.append(img_data)
            print(f"✅ OCR completed: {img_data['filename']} ({img_data['label']})")
        except Exception as e:
            print(f"❌ Error processing {img_data['filename']}: {e}")
    return processed_images


def gemini_structuring(context: PipelineContext, processed_images: List[Dict], salary_data: Dict, email_text: str, requested_service: str, service_needed: str) -> Tuple[Dict, str]:
    print("🧠 Running comprehensive Gemini structuring...")
    passport_ocr_1 = ""
    passport_ocr_2 = ""
    emirates_id_ocr = ""
    emirates_id_2_ocr = ""
    employee_info = ""
    certificate_ocr = ""
    google_metadata: Dict = {}
    for img_data in processed_images:
        ocr_text = img_data.get("ocr_text", "")
        extracted_fields = img_data.get("extracted_fields", {})
        label = img_data["label"]
        if label == "passport_1":
            passport_ocr_1 = ocr_text
            if extracted_fields:
                google_metadata["passport_1_fields"] = extracted_fields
        elif label == "passport_2":
            passport_ocr_2 = ocr_text
            if extracted_fields:
                google_metadata["passport_2_fields"] = extracted_fields
        elif label == "emirates_id":
            emirates_id_ocr = ocr_text
            if extracted_fields:
                google_metadata["emirates_id_fields"] = extracted_fields
        elif label == "emirates_id_2":
            emirates_id_2_ocr = ocr_text
            if extracted_fields:
                google_metadata["emirates_id_2_fields"] = extracted_fields
        elif label == "employee_info_form":
            employee_info = ocr_text
            if extracted_fields:
                google_metadata["employee_info_fields"] = extracted_fields
        elif label == "certificate":
            certificate_ocr = ocr_text
            if extracted_fields:
                google_metadata["certificate_fields"] = extracted_fields
        elif label in ["certificate_attestation", "attestation_label"] and ocr_text:
            certificate_ocr = ocr_text
            if extracted_fields:
                google_metadata["certificate_fields"] = extracted_fields

    result = structure_with_gemini(
        passport_ocr_1=passport_ocr_1,
        passport_ocr_2=passport_ocr_2,
        emirates_id_ocr=emirates_id_ocr,
        emirates_id_2_ocr=emirates_id_2_ocr,
        employee_info=employee_info,
        certificate_ocr=certificate_ocr,
        salary_data=salary_data,
        email_text=email_text,
        resnet_label=", ".join([img["label"] for img in processed_images]),
        google_metadata=google_metadata,
    )
    if isinstance(result, tuple):
        final_structured, gemini_response = result
    else:
        final_structured, gemini_response = result, ""

    try:
        final_structured["Requested Service"] = requested_service
    except Exception:
        pass
    final_structured["Service Needed"] = service_needed
    return final_structured, gemini_response


def save_results(context: PipelineContext, subject_folder: str, processed_images: List[Dict], final_structured: Dict, gemini_response: str, salary_data: Dict, service_needed: str, sender_email: str, sender_name: str) -> None:
    subject_output_dir = os.path.join(context.output_dir, subject_folder)
    os.makedirs(subject_output_dir, exist_ok=True)

    full_name = final_structured.get("Full Name", "")
    first_name = full_name.split()[0] if full_name else "Unknown"
    master_text_file = os.path.join(subject_output_dir, f"{first_name}_COMPLETE_DETAILS.txt")
    with open(master_text_file, "w", encoding="utf-8") as f:
        f.write(f"SERVICE NEEDED: {service_needed}\n")
        if sender_name:
            f.write(f"Sender Name: {sender_name}\n")
        if sender_email:
            f.write(f"Email Address: {sender_email}\n")
        f.write("\n")
        for key, value in final_structured.items():
            f.write(f"{key}: {value}\n")

    print(f"📄 Created details file: {master_text_file}")

    for img_data in processed_images:
        doc_type = img_data["label"]
        base = f"{first_name}_{doc_type}"
        save_path = img_data.get("cropped_path") or img_data.get("path")
        final_path = save_outputs(save_path, final_structured, subject_output_dir, base, gemini_response)
        log_processed_file(context.log_file, img_data["filename"], final_path, img_data["label"])

    for img_data in processed_images:
        doc_type = img_data["label"]
        base = f"{first_name}_{doc_type}"
        saved_file = os.path.join(subject_output_dir, f"{base}.jpg")
        if os.path.exists(saved_file):
            compress_image_to_jpg(saved_file, saved_file)


def main() -> None:
    context = PipelineContext()
    fetch_emails(context)

    download_dirs = [context.input_dir, "downloads"]
    processed_folders = set()

    for download_dir in download_dirs:
        if not os.path.exists(download_dir):
            print(f"⚠️ Download directory not found: {download_dir}")
            continue
        print(f"📁 Processing from: {download_dir}")
        for subject_folder in os.listdir(download_dir):
            if subject_folder in processed_folders:
                print(f"⏭️ Skipping already processed folder: {subject_folder}")
                continue
            subject_path = os.path.join(download_dir, subject_folder)
            if not os.path.isdir(subject_path):
                continue

            print(f"\n🔍 Processing folder: {subject_folder}")
            requested_service = "Unknown Service"
            email_text_path = os.path.join(subject_path, "email_body.txt")
            email_text = ""
            service_needed = "N/A"
            sender_email = ""
            sender_name = ""
            if os.path.exists(email_text_path):
                with open(email_text_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if lines:
                    first_line = lines[0].strip()
                    if first_line.lower().startswith("sender:"):
                        sender_email = first_line.split(":", 1)[1].strip()
                        body_lines = lines[1:]
                    else:
                        body_lines = lines
                    email_text = "".join(body_lines)
                match = re.search(r"(?i)service needed[:\-]\s*(.+)", email_text)
                if match:
                    service_needed = match.group(1).strip()
                if sender_email and re.match(r"^[A-Za-z._]+@[A-Za-z0-9.-]+$", sender_email):
                    local_part = sender_email.split("@")[0]
                    name_parts = re.split(r"[._]", local_part)
                    if name_parts and all(part.isalpha() for part in name_parts):
                        sender_name = " ".join(part.capitalize() for part in name_parts)
                try:
                    from service_detector import detect_service_from_email
                    requested_service = detect_service_from_email(email_text)
                except Exception:
                    requested_service = "Unknown Service"

            image_paths = convert_documents(context, subject_path)
            if not image_paths:
                print(f"⚠️ No images found in {subject_folder}")
                continue

            salary_data: Dict = {}
            docx_files = [f for f in os.listdir(subject_path) if f.lower().endswith(".docx") and "salary" in f.lower()]
            for docx_file in docx_files:
                try:
                    docx_path = os.path.join(subject_path, docx_file)
                    parsed_salary = parse_salary_docx(docx_path)
                    if parsed_salary:
                        salary_data.update(parsed_salary)
                except Exception as e:
                    print(f"❌ Error parsing salary from {docx_file}: {e}")

            classified_images = classify_images(context, image_paths)
            processed_images = perform_ocr(context, classified_images)
            if not processed_images:
                print(f"⚠️ No processed images for {subject_folder}. Skipping folder.")
                continue

            final_structured, gemini_response = gemini_structuring(
                context, processed_images, salary_data, email_text, requested_service, service_needed
            )

            save_results(
                context,
                subject_folder,
                processed_images,
                final_structured,
                gemini_response,
                salary_data,
                service_needed,
                sender_email,
                sender_name,
            )

            processed_folders.add(subject_folder)
            print(f"📂 Done with folder: {subject_folder}\n{'-'*40}")

    print("✅ All documents processed.")
    print("\n📂 Opening file explorer to view processed documents...")
    open_file_explorer(os.path.abspath(context.output_dir))


if __name__ == "__main__":
    main()

