import os
import sys
import subprocess
from pathlib import Path


def log_message(message: str, log_path: Path) -> None:
    """
    Write a message to both stdout and append it to a log file.

    Parameters:
        message (str): The message to be logged.
        log_path (Path): The file path where the log should be saved.
    """
    print(message)
    with open(log_path, "a") as f:
        f.write(message + "\n")


def af_info(log_path: Path) -> None:
    """
    Collect and log Python environment information related to AlphaFold.

    Parameters:
        log_path (Path): Path to the log file where information will be written.
    """

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list"],
            capture_output=True, 
            text=True,
            check=True
        )

        jax_lines = [line for line in result.stdout.splitlines() if "jax" in line.lower()]
        jax_versions = []

        for line in jax_lines:
            parts = line.split()
            if len(parts) >= 2:
                jax_versions.append(f"{parts[0]} {parts[1]}")

        jax_summary = ", ".join(jax_versions)
    except FileNotFoundError:
        log_message("Error: pip is not available in this environment.", log_path)
        return

    tf_deterministic_ops = os.environ.get("TF_DETERMINISTIC_OPS", "N/A")
    tf_unified_memory = os.environ.get("TF_FORCE_UNIFIED_MEMORY", "N/A")
    xla_fraction = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "N/A")

    log_message("==== AlphaFold Environment Information: ====", log_path)
    log_message(f"Python version: {sys.version}", log_path)
    log_message(f"JAX versions: {jax_summary}", log_path)
    log_message(f"TF_DETERMINISTIC_OPS: {tf_deterministic_ops}", log_path)
    log_message(f"TF_FORCE_UNIFIED_MEMORY: {tf_unified_memory}", log_path)
    log_message(f"XLA_PYTHON_CLIENT_MEM_FRACTION: {xla_fraction}", log_path)
    log_message(f"Log file path: {log_path}", log_path)
    log_message("AlphaFold environment information collected successfully.", log_path)
    log_message("==== End of Environment Information ====", log_path)
