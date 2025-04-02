# 🧬 AlphaFold Automation Runner 

🚀 Designed for run AlphaFold 2 across different GPUs environment. Automatically summarize the benchmark data and environment information.

---

## 📦 Features

- ✅ Command-line interface built with `argparse`
- ✅ TDD-first workflow using `pytest`
- ✅ Modular codebase with SOLID principles
- ✅ Supports mock simulation (no AF2 required)
- ✅ Optional support for external AlphaFold2 environment
- ✅ CI-ready with `GitHub Actions` + `uv` + `mypy` + `ruff`

---

## 🛠️ Installation

> Requires Python 3.12+

```bash
git clone https://github.com/yourname/alphafold-runner.git
cd alphafold-runner

# Using uv to manage the project environment
uv pip install -e ".[dev]"
```

---

## 🧪 Example: Running in Mock Mode (No AF2 required)

```bash
cat > input.fasta << EOF
>seq1
ACDEFGHIK
>seq2
LMNPQRSTVWY
EOF
```

```bash
alphafold-runner \
  --input input.fasta \
  --backend mock \
  --log result.log
```

### ✅ Sample Output

```
== AlphaFold 2 Runner ==
Input file: input.fasta
Run mode: cpu
Log file: result.log
🧪 Simulating AlphaFold inference...
✅ Mock inference complete (seq_len=20)
```

---

## 🧾 Log Output

```bash
cat result.log
```

```
[2024-03-31 22:15:01] Fasta file input.fasta exists.
[2024-03-31 22:15:01] 🧪 Simulating AlphaFold inference...
[2024-03-31 22:15:02] ✅ Mock inference complete (seq_len=20)
```

---

## 🧬 Running in External Mode (AlphaFold2 Integration)

> You must provide a Python 3.10 environment and AF2 runner script.

```bash
alphafold-runner \
  --input input.fasta \
  --backend external \
  --af2-python /opt/af2_env/bin/python3.10 \
  --af2-script /home/user/alphafold/run_alphafold.py \
  --output-dir results \
  --log af2.log
```

---

## 🧪 Running Tests

```bash
pytest -v
```

---

## ⚙️ Development Tools

| Tool      | Purpose                |
|-----------|------------------------|
| `pytest`  | Unit testing           |
| `mypy`    | Type checking          |
| `ruff`    | Linting & formatting   |
| `uv`      | Dependency management  |

---

## 📂 Project Structure

```
scripts/
├── main.py           # CLI entry point
├── inference.py      # Inference stubs (mock + external)
├── log_utils.py      # Logging + file checks
├── env_utils.py      # Platform detection
├── gpu_detect.py     # GPU backend detection

tests/
├── test_cli.py
├── test_inference.py
├── test_env_utils.py
└── data/
    └── multiseq.fasta
```

---

## 📄 License

MIT License © 2025 Your Name
