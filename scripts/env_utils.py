import os
import sys
import subprocess
from pathlib import Path
from scripts.log_utils import log_message
from scripts.gpu_detect import detect_platform


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


def setup_platform(mode: str) -> None:
    """
    Setup platform-specific environment variables for AlphaFold based on mode.

    Parameters:
        mode (str): Either "cpu" or "gpu". In "gpu" mode, the system auto-detects NVIDIA or AMD GPU.

    Raises:
        ValueError: If mode is unsupported, fallback to cpu mode.
    """
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    platform = detect_platform()

    if mode == "gpu" and platform != "cpu":
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
        
        if platform == "nvidia":
            os.environ["JAX_PLATFORMS"] = "cuda"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        elif platform == "amd":
            os.environ["JAX_PLATFORMS"] = "rocm"
            os.environ["HIP_VISIBLE_DEVICES"] = "0"
            os.environ["ROCM_PATH"] = "/opt/rocm"

    elif mode == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        # Try disabling both GPU type, just in case
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    
    else:
        raise ValueError(f"Unsupported mode: {mode}")
