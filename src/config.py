from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parents[1]

# Load environment variables once
load_dotenv()


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # API Keys
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    google_cloud_project_id: Optional[str] = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    document_ai_processor_id: Optional[str] = os.getenv("DOCUMENT_AI_PROCESSOR_ID")

    # Paths
    yolo_model_path: Path = Path(os.getenv("YOLO_MODEL_PATH", BASE_DIR / "models" / "yolo8_best.pt"))
    input_root: Path = Path(os.getenv("INPUT_ROOT", BASE_DIR / "data" / "dataset"))
    output_root: Path = Path(os.getenv("OUTPUT_ROOT", BASE_DIR / "data" / "processed" / "COMPLETED"))
    data_dir: Path = Path(os.getenv("DATA_DIR", BASE_DIR / "data" / "dataset"))
    model_save_path: Path = Path(os.getenv("MODEL_SAVE_PATH", BASE_DIR / "models" / "model_classifier.pt"))
    dataset_classes_path: Path = Path(os.getenv("DATASET_CLASSES_PATH", BASE_DIR / "data" / "dataset"))
    input_dir: Path = Path(os.getenv("INPUT_DIR", BASE_DIR / "data" / "raw" / "downloads"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "data" / "processed" / "MOHRE_ready"))
    model_path: Path = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "classifier.pt"))
    root_dir: Path = Path(os.getenv("ROOT_DIR", BASE_DIR / "data" / "raw" / "downloads"))
    dataset_dir: Path = Path(os.getenv("DATASET_DIR", BASE_DIR / "data" / "dataset"))
    source_yolo_model: Path = Path(os.getenv("SOURCE_YOLO_MODEL", BASE_DIR / "models" / "yolo8_best.pt"))
    source_classifier_model: Path = Path(os.getenv("SOURCE_CLASSIFIER_MODEL", BASE_DIR / "models" / "classifier.pt"))
    source_dataset: Path = Path(os.getenv("SOURCE_DATASET", BASE_DIR / "data" / "dataset"))
    google_application_credentials: Path = Path(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS", BASE_DIR / "config" / "GOOGLEAPI.json")
    )
    passport_image_path: Path = Path(
        os.getenv(
            "PASSPORT_IMAGE_PATH",
            BASE_DIR / "data" / "processed" / "COMPLETED" / "passport_1" / "sample_passport.jpg",
        )
    )
    email_address: Optional[str] = os.getenv("EMAIL_ADDRESS")
    email_password: Optional[str] = os.getenv("EMAIL_PASSWORD")
    imap_server: Optional[str] = os.getenv("IMAP_SERVER")
    download_dir: Path = Path(os.getenv("DOWNLOAD_DIR", BASE_DIR / "data" / "raw" / "downloads"))


@lru_cache()
def get_config() -> Config:
    """Return a cached configuration instance."""
    cfg = Config()
    if cfg.google_application_credentials and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cfg.google_application_credentials)
    return cfg

