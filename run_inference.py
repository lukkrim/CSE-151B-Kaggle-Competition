# -*- coding: utf-8 -*-
"""Run vLLM inference locally for the CSE 151B competition."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"

PUBLIC_PATH = DATA_DIR / "public.jsonl"
PRIVATE_PATH = DATA_DIR / "private.jsonl"
PRIVATE_OUTPUT_PATH = RESULTS_DIR / "private_results.jsonl"
PRIVATE_CSV_PATH = RESULTS_DIR / "private_submission.csv"

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
BATCH_SIZE = 30


def install_dependencies() -> None:
    req_file = PROJECT_ROOT / "requirements.txt"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])


def print_gpu_info() -> None:
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi not found (no NVIDIA GPU or driver not installed)")


SYSTEM_PROMPT_MATH = """You are a confident expert mathematician. You trust your computations and commit to your answers.

Solve the problem step-by-step, then put your final answer in \\boxed{}. For multi-part problems, put all sub-answers in one box separated by commas: \\boxed{3, 7}.

Your answer must match the grader's expected format exactly. The grader does literal comparison — `7.797` will NOT match `7.79743547584717`, and `1.363` will NOT match `atan(4.76)`. Choose the form carefully:

**Prefer exact/symbolic forms. Do NOT evaluate to a decimal unless the problem explicitly asks for one.**
- Inverse trig: write `atan(4.76)`, not `1.363`
- Constants: write `pi`, not `3.142`; write `sqrt(2)/2`, not `0.707`
- Logarithms: write `ln(11/8)` or `[ln(0.5)]/[ln(0.96584)]`, not their decimals
- Fractional powers: write `(1/2)^(36/31)`, not `0.447`
- Unevaluated products when the problem is about deriving a formula: `325*(1+325)` rather than `105950`
- Fractions: write `5/8`, not `0.625`

**When a decimal IS required, give the FULL unrounded value (12+ significant figures).**
Round only when the problem explicitly says to. Otherwise:
- Write `7.79743547584717`, not `7.797`
- Write `442.857142857143`, not `442.86`
If you compute 565/9, do not stop at "≈ 62.78" — write out 62.7777777777778.

**Match the rounding rule the problem states.**
- "Round to integer" with x = 14.4375 → 14
- "After how many full years until X happens" — if the event happens during year 14 (between t=14 and t=15), the answer is often 13 (full years completed before it happens), not 14. Read carefully.
- "How many signatures / units needed" → always round UP (ceiling).

**Reasoning style.**
- Solve directly using one clear method. Commit to your approach.
- Keep steps focused and concise; most problems take 200-600 tokens of reasoning.
- Long, exploratory reasoning is a sign of going off-track, not of being thorough.
- Read the problem carefully once at the start — especially units, limits of integration, and what is being asked for.

**Verification.**
- After reaching your answer, do ONE quick sanity check (units, sign, or magnitude).
- Then commit. Do not re-derive, restart, or explore alternative methods.
- If a check reveals a clear error, fix that specific step and continue forward.

**No matching option (multiple choice).**
- If your computed answer doesn't match any option exactly, output \\boxed{NONE}.
- Do NOT pick the closest option.
- Do NOT reinterpret the problem, invent typos, or assume the problem meant something different to force a match.
- Trust your computation over the answer choices.

**Worked examples of correct final answers:**
- Sum of first 325 positive even numbers → \\boxed{325*(1+325)}
- 145°F to Celsius → \\boxed{62.7777777777778}
- All solutions to tan(θ)=4.76 in form θ=a+b·n → \\boxed{atan(4.76), pi}
- Fraction remaining after 36 years, half-life 31 → \\boxed{(1/2)^(36/31)}
- Half-life when daily decay is 3.416% → \\boxed{[ln(0.5)]/[ln(0.96584)]}
- Standard deviation when variance is 60.8 → \\boxed{7.79743547584717}
- Reduce 25/40 → \\boxed{5/8}
- Integral with no matching multiple-choice option → \\boxed{NONE}"""

SYSTEM_PROMPT_MCQ = """You are an expert mathematician. Read the problem and the options, then output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}.

**Procedure:**
1. Solve the problem independently first, without looking at the options.
2. Then match your derived answer to the options.
3. **If your computed answer doesn't match any option, do NOT pick the closest one.** Re-check your computation — you likely made an arithmetic or sign error, or misread the problem. Also check whether an option is mathematically equivalent to your answer in a different form (e.g., your `pi*sqrt(a)` might equal an option written as `sqrt(a)*pi`).
4. Never collapse a symbolic answer (with π, e, √, etc.) to the closest decimal option just because the options are all decimals — that usually means you misread the problem or the options. Re-examine first.

**Reasoning style.** Solve directly, no "this is complex" preambles. Verify briefly. If you spot an error, fix it and move on without dwelling.
"""


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def preview_public_data() -> None:
    data = [json.loads(line) for line in open(PUBLIC_PATH, encoding="utf-8")]

    n_mcq = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options") for d in data)
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    mcq_sample = next(d for d in data if d.get("options"))
    free_sample = next(d for d in data if not d.get("options"))

    print("\n── MCQ sample ──")
    print(json.dumps(mcq_sample, indent=2))
    print("\n── Free-form sample ──")
    print(json.dumps(free_sample, indent=2))

    for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
        _, usr_p = build_prompt(item["question"], item.get("options"))
        print(f"── {label} user prompt (first 200 chars) ──")
        print(usr_p[:200], "...\n")


def load_model(gpu_id: str):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",  # use "float16" if on T4
        enable_prefix_caching=True,
        gpu_memory_utilization=0.85,
        max_model_len=32768,  # lower if OOM
        trust_remote_code=True,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
    )

    sampling_params = SamplingParams(
        max_tokens=24768,  # leaves 8K for the input prompt
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    print("Model loaded.")
    return tokenizer, llm, sampling_params


def run_private_inference(tokenizer, llm, sampling_params) -> None:
    from tqdm import tqdm

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    private_data = [json.loads(line) for line in open(PRIVATE_PATH, encoding="utf-8")]
    print(f"Loaded {len(private_data)} private questions.")

    completed_ids = set()
    if PRIVATE_OUTPUT_PATH.exists():
        with open(PRIVATE_OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    completed_ids.add(row["id"])
                except json.JSONDecodeError:
                    pass

    print(f"Already completed: {len(completed_ids)}")

    remaining_items = [item for item in private_data if item["id"] not in completed_ids]
    print(f"Remaining to run: {len(remaining_items)}")

    for start in tqdm(range(0, len(remaining_items), BATCH_SIZE), desc="Generating"):
        batch_items = remaining_items[start : start + BATCH_SIZE]

        prompts = []
        for item in batch_items:
            system, user = build_prompt(item["question"], item.get("options"))
            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt_text)

        outputs = llm.generate(prompts, sampling_params=sampling_params)
        responses = [out.outputs[0].text.strip() for out in outputs]

        with open(PRIVATE_OUTPUT_PATH, "a", encoding="utf-8") as f:
            for item, response in zip(batch_items, responses):
                result = {
                    "id": item["id"],
                    "is_mcq": bool(item.get("options")),
                    "response": response,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

        print(f"Saved batch {start} to {start + len(batch_items)}")

    print(f"Finished. Results saved to: {PRIVATE_OUTPUT_PATH}")


def export_submission_csv() -> None:
    import pandas as pd

    df = pd.read_json(PRIVATE_OUTPUT_PATH, lines=True)
    df = df[["id", "response"]].sort_values("id")

    df.to_csv(
        PRIVATE_CSV_PATH,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        escapechar=None,
        doublequote=True,
    )

    print(f"Saved CSV to: {PRIVATE_CSV_PATH}")
    print(df.head())
    print("Total rows:", len(df))


def run_inference(
    gpu: str = "0",
    install_deps: bool = False,
    skip_preview: bool = False,
) -> None:
    """Programmatic entry point to reproduce private-set inference."""
    if install_deps:
        install_dependencies()

    print_gpu_info()

    if not skip_preview:
        preview_public_data()

    tokenizer, llm, sampling_params = load_model(gpu)
    run_private_inference(tokenizer, llm, sampling_params)
    export_submission_csv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM inference on the private test set.")
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install packages from requirements.txt before running",
    )
    parser.add_argument("--gpu", default="0", help="CUDA device id (default: 0)")
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip loading and previewing the public dataset",
    )
    args = parser.parse_args()
    run_inference(
        gpu=args.gpu,
        install_deps=args.install_deps,
        skip_preview=args.skip_preview,
    )


if __name__ == "__main__":
    main()
