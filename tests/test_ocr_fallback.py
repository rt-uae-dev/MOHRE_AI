import os
import sys
import types

# Provide lightweight mocks for heavy optional dependencies
sys.modules.setdefault('cv2', types.SimpleNamespace())

class DummyYOLO:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return [types.SimpleNamespace(boxes=[])]

sys.modules.setdefault('ultralytics', types.SimpleNamespace(YOLO=DummyYOLO))

class DummyVisionClient:
    def document_text_detection(self, image):
        return types.SimpleNamespace(full_text_annotation=None, text_annotations=None, label_annotations=None)

    def text_detection(self, image):
        return types.SimpleNamespace(text_annotations=None)

vision_stub = types.SimpleNamespace(
    ImageAnnotatorClient=lambda: DummyVisionClient(),
    Image=lambda **kwargs: types.SimpleNamespace(),
)
sys.modules.setdefault('google', types.SimpleNamespace(cloud=types.SimpleNamespace(vision=vision_stub)))
sys.modules.setdefault('google.cloud', types.SimpleNamespace(vision=vision_stub))
sys.modules.setdefault('google.cloud.vision', vision_stub)

# Ensure src directory is on the path for imports
sys.path.append('src')

from yolo_crop_ocr_pipeline import run_enhanced_ocr


def test_google_vision_fallback(monkeypatch, tmp_path):
    """When Document AI is disabled, run_enhanced_ocr should use Google Vision."""
    # Disable Document AI
    monkeypatch.setattr("yolo_crop_ocr_pipeline.DOCUMENT_AI_AVAILABLE", False)

    # Stub out Google Vision OCR to avoid external API calls
    def fake_vision(path):
        return {"ocr_text": "dummy"}

    monkeypatch.setattr("yolo_crop_ocr_pipeline.run_google_vision_ocr", fake_vision)

    # Call enhanced OCR with a dummy image path
    dummy_path = tmp_path / "dummy.jpg"
    result = run_enhanced_ocr(str(dummy_path))

    assert result["ocr_method"] == "google_vision"
    assert result["ocr_text"] == "dummy"


def test_low_confidence_triggers_google_vision(monkeypatch, tmp_path):
    """Low Document AI confidence should trigger Google Vision fallback."""

    class DummyProcessor:
        enabled = True

        def process_document(self, path):
            return {"full_text": "doc ai text", "confidence": 0.2}

        def get_document_type(self, text):
            return "passport"

        def extract_fields_by_document_type(self, text):
            return {}

    monkeypatch.setattr("yolo_crop_ocr_pipeline.DOCUMENT_AI_AVAILABLE", True)
    monkeypatch.setattr("yolo_crop_ocr_pipeline.DOCUMENT_AI_PROCESSOR", DummyProcessor())

    def fake_vision(path):
        return {"ocr_text": "vision"}

    monkeypatch.setattr("yolo_crop_ocr_pipeline.run_google_vision_ocr", fake_vision)

    dummy_path = tmp_path / "dummy.jpg"
    result = run_enhanced_ocr(str(dummy_path))

    assert result["ocr_method"] == "google_vision"
    assert result["ocr_text"] == "vision"
    assert result["confidence"] == 0.2
