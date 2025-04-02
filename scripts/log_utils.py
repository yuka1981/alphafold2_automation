from pathlib import Path
from datetime import datetime


def log_message(message: str, log_path: Path | None = None) -> None:
    """
    Write a message to both stdout and append it to a log file.

    Parameters:
        message (str): The message to be logged.
        log_path (Path): The file path where the log should be saved.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"

    print(full_message)  # always print to stdout

    if log_path is not None:
        with log_path.open("a") as f:
            f.write(full_message + "\n")


def check_required_file(file_path: Path) -> None:
    """
    Check whether the given file exists. If not, raise an error.

    Parameters:
        file_path (Path): The path to the file that should exist.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Error: Missing required file: {file_path}")

    print(f"Fasta file {file_path} exists.")