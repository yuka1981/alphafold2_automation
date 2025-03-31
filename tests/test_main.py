import os
from unittest import mock
from pathlib import Path
from scripts.main import log_message, af_info

def test_log_message(tmp_path):
    # Arrange
    log_file = tmp_path / "test.log"
    message = "Test log message to verify the logging function works correctly."

    # Act
    log_message(message, log_file)

    # Assert
    content = log_file.read_text()
    assert message in content


@mock.patch("scripts.main.subprocess.run")
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
