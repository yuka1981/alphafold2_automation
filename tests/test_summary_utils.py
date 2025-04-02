from scripts.summary_utils import print_summary
from pathlib import Path
import pdb

def test_print_summary_outputs_expected_lines(tmp_path, capsys):
    # Arrange
    exmple_json = Path("tests/data") / "example_summary.json"
    summary_text = exmple_json.read_text()
    summary_json = tmp_path / "mock_output.json"
    summary_json.write_text(summary_text)

    # Act
    print_summary(summary_json)

    # Assert
    output = capsys.readouterr().out
    assert "== Inference Summary ==" in output
    assert "Total sequences: 2" in output
    assert "Total length: 51" in output
    assert "Wall time: 00:00:17" in output
