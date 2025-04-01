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