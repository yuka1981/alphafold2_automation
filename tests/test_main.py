import os
import pytest
import pdb
from unittest import mock
from unittest.mock import patch
from pathlib import Path
from scripts.log_utils import log_message, check_required_file
from scripts.env_utils import af_info, setup_platform


def test_log_message(tmp_path):
    # Arrange
    log_file = tmp_path / "test.log"
    message = "Test log message to verify the logging function works correctly."

    # Act
    log_message(message, log_file)

    # Assert
    content = log_file.read_text()
    assert message in content


@mock.patch("scripts.env_utils.subprocess.run")
def test_af_info(mock_run, tmp_path):
    # Arrange
    log_file = tmp_path / "env.log"
    fake_output = "jax 0.4.21\njaxlib 0.4.21"
    mock_run.return_value.stdout = fake_output

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

    # Act
    af_info(log_file)

    # Assert
    content = log_file.read_text()
    assert "jax 0.4.21" in content
    assert "jaxlib 0.4.21" in content


@pytest.mark.parametrize(
    "gpu_type, expected_env", [
        ("nvidia", {"JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "0"}),
        ("amd", {"JAX_PLATFORMS": "rocm", "HIP_VISIBLE_DEVICES": "0"}),
    ]
)
def test_setup_platform_gpu_auto_detect(gpu_type, expected_env):
    # Clean up before test
    for value in ["JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCM_PATH"]:
        os.environ.pop(value, None)

    with patch("scripts.gpu_detect.has_nvidia_gpu", return_value=(gpu_type == "nvidia")), \
        patch("scripts.gpu_detect.has_amd_gpu", return_value=(gpu_type == "amd")):
        setup_platform("gpu")
    
    # Assert
    for key, value in expected_env.items():
        assert os.environ[key] == value

    assert os.environ["TF_DETERMINISTIC_OPS"] == "1"


def test_check_required_file_exists(tmp_path, capsys):
    # Arrange
    file = tmp_path / "input.fasta"
    file.write_text("SEQUENCE")

    # Act
    check_required_file(file)

    # Assert
    captured = capsys.readouterr()
    assert f"Fasta file {file} exists." in captured.out


def test_check_required_file_not_exists(tmp_path, capsys):
    # Arrange
    # Create a temporary file path object without creating the file
    file = tmp_path / "input.fasta"

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Missing required file:"):
        check_required_file(file)
