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
# CSE-151B-Kaggle-Competition
