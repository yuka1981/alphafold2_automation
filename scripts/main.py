import argparse
import sys
from pathlib import Path
from typing import NoReturn
from scripts.env_utils import af_info, setup_platform
from scripts.log_utils import check_required_file, log_message


class CustomArgParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_help(sys.stderr)
        self.exit(2, f"Error: {message}\n")


def main() -> None:
    parser = CustomArgParser(
        description="Run AlphaFold workflow",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--input", 
        type=Path, 
        required=True,
        help="Path to the input FASTA file"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["cpu", "gpu"], 
        default="cpu", 
        help="Mode to run: cpu or gpu, default: cpu"
    )

    parser.add_argument(
        "--log", 
        type=Path, 
        default=Path("run.log"), 
        help="Path to the log file, default: ./run.log"
    )

    args = parser.parse_args()

    print("== AlphaFold 2 Runner ==")
    print(f"Input file: {args.input}")
    print(f"Run mode: {args.mode}")
    print(f"Log file: {args.log}")

    # Check if the input file exists
    check_required_file(args.input)

    # Setup the platform
    setup_platform(args.mode)

    # Log the environment information
    af_info(args.log)

    # Log the successful setup
    print("Setup complete. Ready to run AlphaFold (not yet implemented).")