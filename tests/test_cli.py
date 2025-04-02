import pytest
import pdb
from unittest.mock import patch, call
from pathlib import Path
from scripts.main import main


def test_cli_main_success(tmp_path, monkeypatch, capsys):
    # Arrange
    fasta_file = tmp_path / "input.fasta"
    fasta_file.write_text(">Test sequence\nSEQUENCE")

    log_file = tmp_path / "run.log"

    monkeypatch.setattr("sys.argv", [
        "main.py", 
        "--input", 
        str(fasta_file), 
        "--mode", 
        "cpu", 
        "--log", 
        str(log_file)
    ])

    # Act
    main()

    # Assert
    output = capsys.readouterr().out
    print(output)
    assert "AlphaFold 2 Runner" in output
    assert "Setup complete" in output

    assert log_file.exists()
    assert "End of Environment Information" in log_file.read_text()


def test_cli_missing_input(monkeypatch, capsys):
    # Arrange
    monkeypatch.setattr("sys.argv", ["alphaforld-runner"])

    # Act
    with pytest.raises(SystemExit) as excinfo:
        main()

    # Assert
    # Check exit code, 2 for argument error
    assert excinfo.value.code == 2

    output = capsys.readouterr().err + capsys.readouterr().out
    assert "usage:" in output
    assert "Error: the following arguments are required: --input" in output


def test_cli_file_not_exist(monkeypatch, capsys, tmp_path):
    # Arrange
    fake_file = tmp_path / "non_existent.fasta"
    fake_log_file = tmp_path / "run.log"
    monkeypatch.setattr("sys.argv", [
        "alphafold-runner", 
        "--input", str(fake_file), 
        "--mode", "cpu", 
        "--log", str(fake_log_file)
    ])

    # Act
    with pytest.raises(FileNotFoundError) as excinfo:
        main()

    # Assert
    assert "Error: Missing required file" in str(excinfo.value)


@patch("scripts.inference.subprocess.run")
def test_cli_backend_external(mock_subprocess_run, tmp_path, monkeypatch, capsys):
    # Arrange: 
    input_path = Path("tests/data/single_seq.fasta")
    fasta_text = input_path.read_text()
    fasta_file = tmp_path / "input.fasta"
    fasta_file.write_text(fasta_text)

    af2_py = tmp_path / "fake_af2_python" # external fake python binary
    af2_py.write_text("#!/usr/bin/env python3.10\n")
    af2_script = tmp_path / "fake_af2_script.py"
    af2_script.write_text("# dummy script")

    log_file = tmp_path / "run.log"
    output_dir = tmp_path / "output"

    monkeypatch.setattr("sys.argv", [
        "alphafold-runner",
        "--input", str(fasta_file),
        "--backend", "external",
        "--af2-python", str(af2_py),
        "--af2-script", str(af2_script),
        "--log", str(log_file),
        "--output-dir", str(output_dir)
    ])

    # Act: 執行 CLI main
    main()

    # Assert: subprocess.run
    excepted_call = call([
        str(af2_py),
        str(af2_script),
        "--fasta", str(fasta_file),
        "--output_dir", str(output_dir)
    ], check=True)
    assert excepted_call in mock_subprocess_run.call_args_list

    args_passed = mock_subprocess_run.call_args[0][0]
    assert str(fasta_file) in args_passed
    assert str(af2_py) in args_passed
    assert str(af2_script) in args_passed

    assert log_file.exists()
    content = log_file.read_text()
    assert "✅ AlphaFold2 process completed" in content


@patch("scripts.inference.subprocess.run")
def test_cli_external_missing_af2_python(mock_subprocess_run, tmp_path, monkeypatch):
    # Arrange:
    input_path = Path("tests/data/single_seq.fasta")
    fasta_text = input_path.read_text()
    fasta_file = tmp_path / "input.fasta"
    fasta_file.write_text(fasta_text)

    script = tmp_path / "fake_af2_script.py"
    script.write_text("# dummy")

    monkeypatch.setattr("sys.argv", [
        "alphafold-runner",
        "--input", str(fasta_file),
        "--backend", "external",
        "--af2-script", str(script)
    ])

    # Act & Assert:
    with pytest.raises(ValueError, match="requires --af2-python"):
        main()


@patch("scripts.inference.subprocess.run")
def test_cli_external_missing_af2_script(mock_subprocess_run, tmp_path, monkeypatch):
    # Arrange:
    input_path = Path("tests/data/single_seq.fasta")
    fasta_text = input_path.read_text()
    fasta_file = tmp_path / "input.fasta"
    fasta_file.write_text(fasta_text)

    af2_python = tmp_path / "fake_af2_python"
    af2_python.write_text("# dummy python")

    monkeypatch.setattr("sys.argv", [
        "alphafold-runner",
        "--input", str(fasta_file),
        "--backend", "external",
        "--af2-python", str(af2_python)
    ])

    # Act & Assert:
    with pytest.raises(ValueError, match="requires --af2-python and --af2-script"):
        main()
