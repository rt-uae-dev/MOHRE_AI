import os
import sys
import pathlib
import json
import shutil
import numpy as np
import cv2
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / 'src'))

from output_saving_utils import save_outputs, log_processed_file


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / 'input.jpg'
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return str(path)


def test_save_outputs_success(sample_image, tmp_path):
    output_dir = tmp_path / 'out'
    result = save_outputs(sample_image, {'a': 1}, str(output_dir), 'base')
    assert result == str(output_dir / 'base.jpg')
    assert (output_dir / 'base.json').exists()


def test_save_outputs_copy_failure(monkeypatch, sample_image, tmp_path):
    def broken_copy(src, dst):
        raise OSError('fail')
    monkeypatch.setattr(shutil, 'copy2', broken_copy)
    with pytest.raises(RuntimeError):
        save_outputs(sample_image, {}, str(tmp_path), 'base')


def test_log_processed_file(tmp_path):
    log_file = tmp_path / 'log.txt'
    log_processed_file(str(log_file), 'orig.pdf', 'out.jpg', 'passport')
    text = log_file.read_text()
    assert 'orig.pdf' in text
    assert 'passport' in text
