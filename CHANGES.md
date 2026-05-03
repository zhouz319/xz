# Changes Relative to Upstream verl

This repository is forked from [verl](https://github.com/volcengine/verl) at commit
[`0eb50ec4a33cda97e05ed8caab9c7f17a30c05a9`](https://github.com/volcengine/verl/commit/0eb50ec4a33cda97e05ed8caab9c7f17a30c05a9)
(version `0.7.0.dev`). The list below documents every PA-GRPO addition or modification
relative to that upstream snapshot.

## New files (PA-GRPO specific)

### Reward functions — `my_reward/`

- `my_reward/mcq_llama.py` — Reward function for the MCQ task with Llama models
  (parses `<think>` reasoning tag). Implements:
  - **Preliminary reward** (Eq. 7 in the paper): correctness ± 1, format bonus ± 0.3,
    length regularization ± 0.1.
  - **Cross-permutation unique-mode consistency reward** (Eq. 9, 10).
- `my_reward/mcq_qwen.py` — Same as `mcq_llama.py` but for Qwen3 models, which use the
  `<thinking>` tag.
- `my_reward/judge_llama.py` — Reward function for the pairwise Judge task with Llama
  models. Implements **index-aligned pairwise consistency** (Eq. 8).
- `my_reward/judge_qwen.py` — Same as `judge_llama.py` but for Qwen3 models.

### Training scripts — `scripts/`

- `scripts/run_mcq_llama.sh`, `run_mcq_qwen.sh`, `run_judge_llama.sh`, `run_judge_qwen.sh`
  — Single-entry scripts that launch `verl.trainer.main_ppo` with the PA-GRPO reward
  function and the appropriate Permutation Group dataset.

### Evaluation scripts (project root)

- `evaluate_models.py` — Inference driver. Loads a HF / LoRA / merged checkpoint with
  vLLM (or Transformers fallback), runs the test parquets under `dataset/test/`
  (overridable via `$PAGRPO_DATASET_DIR`), and writes per-sample JSON results.
  Default decoding is greedy (`temperature=0`) to match the deterministic numbers
  reported in the paper.
- `compute_metrics_judge.py` — Computes Accuracy / Consistency / Consistent
  Accuracy plus appendix-only RStd / CKLD for the pairwise (P = 2) Judge
  benchmarks (MT-Bench, JudgeBench, PreferenceBench, RewardBench, ...).
- `compute_metrics_mcq.py` — Same metrics for the 4-option MCQ benchmarks
  (ARC-Challenge, TinyMMLU, GPQA, ...) under P = 24 Full Permutation Expansion;
  per Eq. 10 of the paper, ties (no unique mode) count as fragmentation and are
  scored as inconsistent for CA.

### Pre-processed datasets — `dataset/train/`

Eight parquet files in two task × two variant × two chat-template combinations:

- `mmlu_{raw,filter}_5perm_{think,thinking}.parquet` — MMLU instances expanded
  into Permutation Groups of size P = 5 (Π = {ABCD, BCDA, CDAB, DABC, DCBA}).
- `chatbot_arena_{raw,filter}_2perm_{think,thinking}.parquet` — Chatbot Arena
  pairwise instances expanded into Permutation Groups of size P = 2
  (Π = {AB, BA}).

Variants:
- **`raw`**: the full set of instances whose original base model exhibits
  inconsistent predictions across permutations. Used by `scripts/run_*.sh` by
  default. MMLU: 29,295 rows (5,859 questions × 5 perms); Chatbot Arena:
  12,152 rows (6,076 pairs × 2 perms).
- **`filter`**: a hard subset pre-filtered against a specific base model; each
  row's `extra_info` carries the base model's answer and CoT. MMLU: 5,545 rows
  (1,109 questions × 5 perms); Chatbot Arena: 2,960 rows (1,480 pairs × 2 perms).

Chat-template:
- `_think.parquet` is for Llama (`<think>` tag).
- `_thinking.parquet` is for Qwen3 (`<thinking>` tag).

### Other

- `requirements-lock.txt` — Pinned full-environment `pip freeze` lock used in our experiments
  (Linux, Python 3.12, CUDA 12.4).
- `requirements-verl-upstream.txt` — Verbatim copy of upstream verl's original
  `requirements.txt` (kept for traceability; not used during install).
- `CHANGES.md` — This file.

## Modified files (relative to upstream verl)

### `verl/trainer/ppo/ray_trainer.py`

Two PA-GRPO-specific additions:

1. **`apply_group_baseline_from_returns()`** (around lines 64-120) — A new helper that
   computes the **Cross-Permutation Advantage** A_PA defined in Eq. 11-12 of the paper.
   For each `original_index` group, it computes the permutation-group mean μ_G and
   standard deviation σ_G across all P × N samples, then normalizes each sample's reward
   by `(r - μ_G) / σ_G` (zeroed out when σ_G < δ to avoid amplifying noise).

2. **`uid` re-keying in the training loop** (around lines 1116-1164) — Replaces the
   default per-sample uid assignment with one that derives the uid from
   `original_question_id` (or falls back to `original_index` / `index // 2`) so that
   all permutation variants of the same semantic instance share a uid and are grouped
   together by the GRPO baseline computation.

### `requirements.txt`

Replaced upstream verl's `requirements.txt` with a PA-GRPO-tailored version that pins
the actual versions used in our experiments (e.g. `torch==2.6.0`, `vllm==0.8.5`,
`transformers>=4.57.0`). The original upstream file is preserved verbatim at
`requirements-verl-upstream.txt` for reference.

No other verl files have been modified.
