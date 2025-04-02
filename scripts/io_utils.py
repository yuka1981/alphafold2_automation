from pathlib import Path
from scripts.log_utils import log_message

def prepare_output_dir(output_dir: Path, log_path: Path | None = None) -> None:
    """
    Prepare the output directory for AlphaFold inference.

    If the directory does not exist, it will be created.
    If it exists, it will be emptied (files and subdirectories deleted).
    Optionally logs actions if log_path is provided.

    Parameters:
        output_dir (Path): The directory to prepare.
        log_path (Path | None): Optional path to a log file.
    """
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            log_message(f"Created output directory: {output_dir}", log_path)
        except OSError as e:
            log_message(f"Error creating output directory {output_dir}: {e}", log_path)
            raise
    else:
        # Handle write permissions error
        try:
            log_message(f"Reusing existing output directory: {output_dir}", log_path)
        except OSError as e:
            log_message(f"Error reusing output directory {output_dir}: {e}", log_path)
            raise
