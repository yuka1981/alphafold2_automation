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
