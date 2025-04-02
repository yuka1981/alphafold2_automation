import shutil
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