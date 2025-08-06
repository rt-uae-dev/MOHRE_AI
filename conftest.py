import os
import sys
import types
from src.logger import configure_logging

def pytest_configure():
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    configure_logging()

    # Stub out heavy optional dependencies
    class DummyCV2:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None
            return _noop

    sys.modules.setdefault('cv2', DummyCV2())

    class DummyVisionClient:
        def document_text_detection(self, image):
            return types.SimpleNamespace(full_text_annotation=None, text_annotations=None, label_annotations=None)

        def text_detection(self, image):
            return types.SimpleNamespace(text_annotations=None)

    vision_stub = types.SimpleNamespace(
        ImageAnnotatorClient=lambda: DummyVisionClient(),
        Image=lambda **kwargs: types.SimpleNamespace(),
    )

    documentai_stub = types.SimpleNamespace(
        DocumentProcessorServiceClient=lambda: types.SimpleNamespace(process_document=lambda *a, **k: None)
    )

    google_stub = types.SimpleNamespace(
        cloud=types.SimpleNamespace(vision=vision_stub, documentai_v1=documentai_stub),
        auth=types.SimpleNamespace(default=lambda *args, **kwargs: (None, None)),
    )

    sys.modules.setdefault('google', google_stub)
    sys.modules.setdefault('google.cloud', google_stub.cloud)
    sys.modules.setdefault('google.cloud.vision', vision_stub)
    sys.modules.setdefault('google.cloud.documentai_v1', documentai_stub)
