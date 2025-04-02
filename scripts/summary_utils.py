import json
from pathlib import Path

def print_summary(summary_path: Path, env_path: Path | None = None) -> None:
    """
    Print a summary of inference results and optional environment info.

    Parameters:
        summary_path (Path): Path to JSON file with inference results.
        env_path (Path | None): Optional path to environment info JSON.
    """
    if env_path and env_path.exists():
        env_data = json.loads(env_path.read_text())
        print("== Environment ==")
        print(f"CUDA version: {env_data.get('cuda_version', 'N/A')}")
        print(f"Driver version: {env_data.get('driver_version', 'N/A')}")
        print(f"Python version: {env_data.get('python_version', 'N/A')}")
        print(f"TensorFlow: {env_data.get('tensorflow_version', 'N/A')}")
        print(f"JAX: {env_data.get('jax_version', 'N/A')}")
        print(f"JAXlib: {env_data.get('jaxlib_version', 'N/A')}")
        print()

    data = json.loads(summary_path.read_text())
    print("== Inference Summary ==")
    print(f"Total sequences: {data.get('num_sequences', '?')}")
    print(f"Total length: {data.get('total_sequence_length', '?')}")
    print(f"MSA search: {data.get('msa', '?')} s")
    print(f"Inference: {data.get('inference', '?')} s")
    print(f"I/O + others: {data.get('io_others', '?')} s")
    print(f"Model time: {data.get('model_times', [])}")
    print(f"Wall time: {data.get('wall(hh:mm:ss)', '?')} ({data.get('wall(secs)', '?')} sec)")
