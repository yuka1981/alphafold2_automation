from pathlib import Path
from scripts.main import log_message

def test_log_message(tmp_path):
    # Arrange
    log_file = tmp_path / "test.log"
    message = "This is a test message."

    # Act
    log_message(message, log_file)

    # Assert
    content = log_file.read_text()
    assert message in content
