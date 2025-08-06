import sys
import pathlib
import types
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / 'src'))


@pytest.fixture(scope='module')
def validations():
    sys.modules['yolo_crop_ocr_pipeline'] = types.SimpleNamespace(
        run_yolo_crop=lambda *a, **k: None,
        run_enhanced_ocr=lambda *a, **k: None,
    )
    sys.modules['image_rotation_utils'] = types.SimpleNamespace(
        rotate_image_if_needed=lambda *a, **k: None,
    )
    sys.modules['resnet18_classifier'] = types.SimpleNamespace(
        classify_image_resnet=lambda *a, **k: None,
        classify_image_from_text=lambda *a, **k: None,
        classify_image_with_gemini_vision=lambda *a, **k: None,
        check_image_orientation=lambda *a, **k: None,
        auto_rotate_image_if_needed=lambda *a, **k: None,
    )
    import document_processing_pipeline as dpp
    return dpp


@pytest.fixture
def passport_text():
    return 'Passport No A12345678 given name John date of birth 1990'


@pytest.fixture
def eid_text():
    return 'Emirates ID 784-1234-1234567-1 identity card holder'


@pytest.fixture
def certificate_text():
    return 'This is to certify that Jane Doe has completed Bachelor degree certificate no. 1234567890'


@pytest.fixture
def attestation_text():
    return 'Document attestation by Ministry of Foreign Affairs attestation no. 1234567890'


def test_validate_passport_in_certificate(validations, passport_text):
    assert validations.validate_passport_in_certificate(passport_text)
    assert not validations.validate_passport_in_certificate('random text')


def test_validate_emirates_id_in_certificate(validations, eid_text):
    assert validations.validate_emirates_id_in_certificate(eid_text)


def test_validate_certificate_in_emirates_id(validations, certificate_text):
    assert validations.validate_certificate_in_emirates_id(certificate_text)


def test_validate_attestation_in_certificate(validations, attestation_text):
    assert validations.validate_attestation_in_certificate(attestation_text)


def test_validate_attestation_label_detection(validations, attestation_text):
    assert validations.validate_attestation_label_detection(['attestation_label'], attestation_text)
    assert not validations.validate_attestation_label_detection(['attestation_label'], 'no attestation here')
    assert not validations.validate_attestation_label_detection([], attestation_text)
