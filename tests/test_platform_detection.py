import subprocess
import pytest
from unittest.mock import patch
from scripts.gpu_detect import has_nvidia_gpu, has_amd_gpu, detect_platform

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


# === AMD GPU test cases ===
@patch("subprocess.check_output")
def test_has_amd_gpu_return_true(mock_check_output):
    mock_check_output.return_value = "Name: gfx90a"
    assert has_amd_gpu() is True


@patch("subprocess.check_output")
def test_has_amd_gpu_return_false_when_output_empty(mock_check_output):
    mock_check_output.return_value = ""
    assert has_amd_gpu() is False


@patch("subprocess.check_output")
def test_has_amd_gpu_return_false_when_file_not_found(mock_check_output):
    mock_check_output.side_effect = FileNotFoundError
    assert has_amd_gpu() is False


@patch("subprocess.check_output")
def test_has_amd_gpu_general_error(mock_check_output):
    mock_check_output.side_effect = Exception("General error")
    assert has_amd_gpu() is False


@patch("subprocess.check_output")
def test_has_amd_gpu_called_process_error(mock_check_output):
    mock_check_output.side_effect = subprocess.CalledProcessError(1, "rocminfo")
    assert has_amd_gpu() is False


# === Platform detection test cases (only check GPU environment) ===
@patch("scripts.gpu_detect.has_nvidia_gpu")
def test_detect_platform_nvidia(mock_has_nvidia_gpu):
    mock_has_nvidia_gpu.return_value = True
    assert detect_platform() == "nvidia"


@patch("scripts.gpu_detect.has_amd_gpu")
def test_detect_platform_amd(mock_has_amd_gpu):
    mock_has_amd_gpu.return_value = True
    assert detect_platform() == "amd"


@patch("scripts.gpu_detect.has_nvidia_gpu", return_value=False)
@patch("scripts.gpu_detect.has_amd_gpu", return_value=False)
def test_detect_platform_no_gpu(mock_has_nvidia_gpu, mock_has_amd_gpu):
    with pytest.raises(RuntimeError) as excinfo:
        detect_platform()
        assert str(excinfo.value) == "No supported GPU platform found."
