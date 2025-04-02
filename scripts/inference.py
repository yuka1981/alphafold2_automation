from pathlib import Path
import time
import pdb
import subprocess
from scripts.log_utils import log_message
from scripts.log_utils import check_required_file

def run_inference(input_fasta: Path, log_path: Path) -> None:
    """
    Simulate AlphaFold inference process.

    Parameters:
        input_fasta (Path): The path to input FASTA file.
        log_path (Path): The path to log file.
    """
    check_required_file(input_fasta)

    log_message("Simulating AlphaFold inference...", log_path)

    #time.sleep(1)  # simulate computation

    lines = input_fasta.read_text().splitlines()
    seq_lines = [line for line in lines if not line.startswith(">")]
    seq = "".join(seq_lines)
    seq_length = len(seq)

    log_message(f"Mock inference complete (seq_len={seq_length})", log_path)


def run_alphafold_external(
        input_fasta: Path,
        output_dir: Path,
        af2_python: Path,
        af2_script: Path,
        log_path: Path
    ) -> None:
    """
    Simulate invoking AlphaFold 2 via subprocess using a different Python environment.

    Parameters:
        input_fasta (Path): Input FASTA file
        output_dir (Path): Directory to store output
        af2_python (Path): Path to AlphaFold2's python executable (3.10)
        af2_script (Path): Path to run_alphafold.py or main.py
        log_path (Path): Log file path
    """
    if not af2_python.exists():
        raise FileNotFoundError(f"AlphaFold2 Python interpreter not found: {af2_python}")

    if not af2_script.exists():
        raise FileNotFoundError(f"AlphaFold2 script not found: {af2_script}")

    log_message(f"🧪 Calling AlphaFold2 external process...", log_path)
    log_message(f"🧪 Calling AlphaFold2 external process...", log_path)

    try:
        subprocess.run(
            [ str(af2_python), str(af2_script),
                "--fasta", str(input_fasta),
                "--output_dir", str(output_dir)],
            check=True
        )
        log_message("✅ AlphaFold2 process completed.", log_path)
    except subprocess.CalledProcessError as e:
        log_message(f"❌ AlphaFold2 failed with return code {e.returncode}\n{e.stderr.decode()}", log_path)
        raise