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

        jax_summary = ", ".join(jax_versions) if jax_versions else "N/A"

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


def has_nvidia_gpu() -> bool:
    """
    Check if the system has an NVIDIA GPU.

    Returns:
        bool: True if an NVIDIA GPU is detected, False otherwise.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True
        )
        # return format: "NVIDIA GH200 480GB"
        return bool(output.strip())

    except (Exception, FileNotFoundError,subprocess.CalledProcessError):
        return False


def has_amd_gpu() -> bool:
    """
    Check if the system has an AMD GPU.

    Returns:
        bool: True if an AMD GPU is detected, False otherwise.
    """
    try:
        output = subprocess.check_output(
            ["rocminfo"],
            stderr=subprocess.DEVNULL,
            text=True
        )
        # return format: "Name: gfx90a ..." (multiple lines)
        return any("gfx" in line.lower() for line in output.splitlines())

    except (Exception, FileNotFoundError, subprocess.CalledProcessError):
        return False


def detect_platform() -> str:
    """
    Detect the platform (AMD or NVIDIA) based on the available GPUs.

    Returns:
        str: The detected platform. Can be "amd", "nvidia", or "cpu".
    """
    if has_nvidia_gpu():
        return "nvidia"
    elif has_amd_gpu():
        return "amd"
    else:
        raise RuntimeError("No supported GPU platform found.")


def setup_platform(mode: str) -> None:
    """
    Setup platform-specific environment variables for AlphaFold based on mode.

    Parameters:
        mode (str): Either "cpu" or "gpu". In "gpu" mode, the system auto-detects NVIDIA or AMD GPU.

    Raises:
        ValueError: If mode is unsupported.
        RuntimeError: If GPU mode is selected but no platform is detected.
    """
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    if mode == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        # Try disabling both GPU type, just in case
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["HIP_VISIBLE_DEVICES"] = "-1"

    elif mode == "gpu":
        platform = detect_platform()
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
        
        if platform == "nvidia":
            os.environ["JAX_PLATFORMS"] = "cuda"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        elif platform == "amd":
            os.environ["JAX_PLATFORMS"] = "rocm"
            os.environ["HIP_VISIBLE_DEVICES"] = "0"
            os.environ["ROCM_PATH"] = "/opt/rocm"
    
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def check_required_file(file_path: Path) -> None:
    """
    Check whether the given file exists. If not, raise an error.

    Parameters:
        file_path (Path): The path to the file that should exist.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")

    print(f"Fasta file {file_path} exists.")
