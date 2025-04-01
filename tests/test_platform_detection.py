import subprocess
from unittest.mock import patch
from scripts.main import has_nvidia_gpu, has_amd_gpu

# === NVIDIA GPU test cases ===

@patch("subprocess.check_output")
def test_has_nvidia_gpu_return_true(mock_check_output):
    mock_check_output.return_value = "NVIDIA GH200 480GB"
    assert has_nvidia_gpu() is True


@patch("subprocess.check_output")
def test_has_nvidia_gpu_return_false_when_output_empty(mock_check_output):
    mock_check_output.return_value = ""
    assert has_nvidia_gpu() is False


@patch("subprocess.check_output")
def test_has_nvidia_gpu_return_false_when_file_not_found(mock_check_output):
    mock_check_output.side_effect = FileNotFoundError
    assert has_nvidia_gpu() is False


@patch("subprocess.check_output")
def test_has_nvidia_gpu_general_error(mock_check_output):
    mock_check_output.side_effect = Exception("General error")
    assert has_nvidia_gpu() is False


@patch("subprocess.check_output")
def test_has_nvidia_gpu_called_process_error(mock_check_output):
    mock_check_output.side_effect = subprocess.CalledProcessError(1, "nvidia-smi")
    assert has_nvidia_gpu() is False
