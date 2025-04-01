import pytest
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
    output =  capsys.readouterr().out
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