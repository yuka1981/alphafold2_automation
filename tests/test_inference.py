import pytest
from scripts.inference import run_inference
from pathlib import Path

def test_run_inference_success(tmp_path):
    # Arrange
    input_path = Path("tests/data/single_seq.fasta")
    fasta_text = input_path.read_text()
    fasta_file = tmp_path / "single.fasta"
    fasta_file.write_text(fasta_text)
    log_file = tmp_path / "single.txt"

    # Act
    run_inference(fasta_file, log_file)

    # Assert
    assert log_file.exists()
    content = log_file.read_text()
    assert "Mock inference complete" in content
    assert "seq_len=10" in content # exmpale sequence length is 10 


def test_run_inference_multiseq(tmp_path):
    # Arrange
    input_path = Path("tests/data/multi_seq.fasta")
    fasta_text = input_path.read_text()
    fasta_file = tmp_path / "multi.fasta"
    fasta_file.write_text(fasta_text)
    log_file = tmp_path / "multi.log"

    # Act
    run_inference(fasta_file, log_file)

    # Assert
    assert log_file.exists()
    content = log_file.read_text()
    assert "Mock inference complete" in content
    assert "seq_len=51" in content # example sequence length is 51


def test_run_inference_missing_file(tmp_path):
    # Arrange
    fasta_file = tmp_path / "not_exist.fasta"
    log_file = tmp_path / "log.txt"

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Missing required file"):
        run_inference(fasta_file, log_file)
