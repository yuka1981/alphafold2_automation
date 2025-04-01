import subprocess
from pathlib import Path
from unittest.mock import patch, Mock
from scripts.env_utils import af_info
from scripts.log_utils import log_message


def test_af_info_logs_error_when_pip_fails(tmp_path):
    # Arrange
    log_file = tmp_path / "afinfo.log"
    
    # Mock subprocess.run to simulate pip failure
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, ["pip", "list"])

        # Act
        af_info(log_file)
        
        # Assert
        content = log_file.read_text()
        assert "Error: Unable to run pip list." in content


def test_af_info_logs_jax_version(tmp_path):
    log_file = tmp_path / "afinfo.log"
    fake_pip_output = (
        "Package    Version\n"
        "---------- -----------\n"
        "jax        0.4.21\n"
        "jaxlib     0.4.21\n"
        "numpy      1.23.5\n"
    )

    make_result = Mock()
    make_result.stdout = fake_pip_output

    # Act
    with patch("subprocess.run", return_value=make_result):
        af_info(log_file)

    # Assert
    content = log_file.read_text()
    assert "JAX versions: jax 0.4.21, jaxlib 0.4.21" in content
