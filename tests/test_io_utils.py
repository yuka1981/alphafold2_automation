import pytest
import pdb
from scripts.io_utils import prepare_output_dir
from pathlib import Path


def test_prepare_output_dir_creates_folder(tmp_path):
    # Arrange
    output_dir = tmp_path / "output"

    # Act
    prepare_output_dir(output_dir)

    # Assert
    assert output_dir.exists()
    assert output_dir.is_dir()


def test_prepare_output_dir_reuses_existing(tmp_path):
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    dummy = output_dir / "dummy.txt"
    dummy.write_text("data")

    prepare_output_dir(output_dir)

    assert dummy.exists()


def test_prepare_output_dir_raises_if_parent_is_file(tmp_path):
    # 模擬情境：output_dir 的 parent 是一個檔案 → 不能 mkdir
    file_as_parent = tmp_path / "not_a_dir"
    file_as_parent.write_text("I am a file")

    # output_dir 嘗試在檔案底下建立 → 會失敗
    output_dir = file_as_parent / "subdir"
    log_file = tmp_path / "log.txt"

    with pytest.raises(OSError):
        prepare_output_dir(output_dir, log_file)

    content = log_file.read_text()
    assert "Error creating output directory" in content