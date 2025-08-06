import os
import sys
import pathlib
import numpy as np
import cv2
import pytest

# Ensure src is importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / 'src'))

from image_utils import (
    compress_image_to_jpg,
    crop_passport_1_to_passport_2,
    resize_image,
    enhance_image_quality,
)


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / 'sample.jpg'
    img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture
def large_image(tmp_path):
    path = tmp_path / 'large.jpg'
    img = np.full((2000, 3000, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return str(path)


def test_compress_image_to_jpg_reduces_size(sample_image, tmp_path):
    out_path = tmp_path / 'compressed.jpg'
    result = compress_image_to_jpg(sample_image, str(out_path), max_kb=50)
    assert result == str(out_path)
    assert out_path.exists()
    assert os.path.getsize(out_path) / 1024 <= 50


def test_compress_image_to_jpg_invalid_path(tmp_path):
    missing = tmp_path / 'missing.jpg'
    out_path = tmp_path / 'out.jpg'
    result = compress_image_to_jpg(str(missing), str(out_path))
    assert result == str(missing)


def test_crop_passport_1_to_passport_2(sample_image, tmp_path):
    result = crop_passport_1_to_passport_2(sample_image, str(tmp_path))
    assert result is not None
    img = cv2.imread(result)
    assert img.shape[0] == 500
    assert img.shape[1] == 1000


def test_crop_passport_1_to_passport_2_invalid(tmp_path):
    result = crop_passport_1_to_passport_2(str(tmp_path / 'missing.jpg'), str(tmp_path))
    assert result is None


def test_resize_image_downscales(large_image, tmp_path):
    out_path = tmp_path / 'resized.jpg'
    result = resize_image(large_image, str(out_path))
    assert result == str(out_path)
    img = cv2.imread(result)
    assert img.shape[1] <= 1920
    assert img.shape[0] <= 1080


def test_resize_image_invalid_path(tmp_path):
    result = resize_image(str(tmp_path / 'missing.jpg'), str(tmp_path / 'out.jpg'))
    assert result == str(tmp_path / 'missing.jpg')


def test_enhance_image_quality_creates_file(sample_image, tmp_path):
    out_path = tmp_path / 'enhanced.jpg'
    result = enhance_image_quality(sample_image, str(out_path))
    assert result == str(out_path)
    assert out_path.exists()


def test_enhance_image_quality_invalid_path(tmp_path):
    result = enhance_image_quality(str(tmp_path / 'missing.jpg'), str(tmp_path / 'out.jpg'))
    assert result == str(tmp_path / 'missing.jpg')
