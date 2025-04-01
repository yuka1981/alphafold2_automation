import os
import subprocess


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