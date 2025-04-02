import argparse
import sys
import datetime
from pathlib import Path
from typing import NoReturn
from scripts.env_utils import af_info, setup_platform
from scripts.log_utils import check_required_file
from scripts.inference import run_inference, run_alphafold_external


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
        default= "output" / Path("run.log"), 
        help="Path to the log file, default: ./run.log"
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["mock", "external"],
        default="mock",
        help="Inference backend: mock (default) or external"
    )

    parser.add_argument(
        "--af2-python",
        type=Path,
        help="Path to Python 3.10 interpreter for AlphaFold2"
    )

    parser.add_argument(
        "--af2-script",
        type=Path,
        help="Path to AlphaFold2 run script"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,  # fallback to current directory
        help="Directory to store AlphaFold2 outputs. Default: ./output/<timestamp>"
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

    if args.backend == "mock":
        run_inference(args.input, args.log)

    elif args.backend == "external":
        if not args.af2_python or not args.af2_script:
            raise ValueError("External mode requires --af2-python and --af2-script")

        # Check if the provided paths exist
        output_dir = args.output_dir
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("output") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        run_alphafold_external(
            input_fasta=args.input,
            output_dir=output_dir,
            af2_python=args.af2_python,
            af2_script=args.af2_script,
            log_path=args.log
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # Log the successful setup
    print("Setup complete. Ready to run AlphaFold (not yet implemented).")