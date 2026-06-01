# CSE 151B Competition — Starter Code

Open **`starter_code_cse151b_comp.ipynb`** to get started.

The notebook covers environment setup, inference with Qwen3-4B-Thinking (INT8), and scoring against the public dataset.

For private-set inference and CSV submission, use **`run_inference.py`** (vLLM, batched, resume-safe).

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Main entry point for public-set baseline + evaluation |
| `run_inference.py` | Private-set inference script (writes JSONL + submission CSV) |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `data/private.jsonl` | Private test set (no answers) |
| `results/` | Output JSONL and CSV files written at runtime |
| `hf_cache/` | Local Hugging Face model cache |

## Running private inference

```bash
python -m venv .venv && source .venv/bin/activate
python run_inference.py --install-deps   # first time only
python run_inference.py --gpu 0
```

Outputs:

- `results/private_results.jsonl` — `{id, is_mcq, response}` per line
- `results/private_submission.csv` — `{id, response}` for leaderboard upload

## Reproducibility requirements

- **GPU used:** NVIDIA **A100** on Google Colab.
- **Approximate inference time:** Running inference on `data/private.jsonl` takes about **3-4 hours**.

### Model weights setup

This repo does **not** include custom fine-tuned weights. We directly use the baseline Hugging Face model (`Qwen/Qwen3-4B-Thinking-2507`).

Model files are downloaded automatically by `run_inference.py` and cached in `hf_cache/`.

1. (Optional) Pre-download or authenticate with Hugging Face if needed for gated models.
2. Keep (or create) the cache directory at:
   - `hf_cache/`
3. Run:

```bash
python run_inference.py --install-deps   # first time only
python run_inference.py --gpu 0
```

If weights are already cached in `hf_cache/`, the script will reuse them.

### How to call `run_inference()` to reproduce results

You can reproduce results either from CLI (recommended) or by importing the function directly.

**CLI call:**

```bash
python run_inference.py --gpu 0
```

**Python call (if importing from code):**

```python
from run_inference import run_inference

run_inference(
    gpu="0",
    install_deps=False,
    skip_preview=False,
)
```
