# PA-GRPO: Permutation-Aware Group Relative Policy Optimization

This repository implements **PA-GRPO** on top of the [verl](https://github.com/volcengine/verl)
framework. PA-GRPO mitigates **selection bias** in LLMs by jointly modeling multiple
permutation variants of the same instance as a Permutation Group, encouraging
permutation-invariant decisions on MCQ and pairwise LLM-as-a-Judge tasks.

> Forked from verl at commit
> [`0eb50ec4`](https://github.com/volcengine/verl/commit/0eb50ec4a33cda97e05ed8caab9c7f17a30c05a9)
> (v0.7.0.dev). Upstream directories (`docker/`, `docs/`, `examples/`, `recipe/`,
> `tests/`, `setup.py`, `pyproject.toml`, `CONTRIBUTING.md`) are kept verbatim
> for traceability and **are not part of the PA-GRPO contribution**.

## Installation

Tested on Python 3.12, PyTorch 2.6.0+cu124, CUDA 12.4.

```bash
git clone https://github.com/ECNU-Text-Computing/PA-GRPO.git
cd PA-GRPO
conda create -n pagrpo python=3.12 && conda activate pagrpo
pip install -r requirements.txt
# Bit-for-bit reproduction of the paper environment:
# pip install -r requirements-lock.txt
```

> **Do NOT run `pip install -e .`** — `setup.py` is inherited from upstream verl
> and would register a system-wide `verl` package that shadows the local
> modifications under `./verl/`. Run scripts directly from the project root.

## Quick Start

```bash
# 1. Set MODEL_PATH inside the script of your choice, then launch training:
bash scripts/run_mcq_llama.sh    # or run_mcq_qwen.sh / run_judge_{llama,qwen}.sh

# 2. Run inference (greedy by default; pass --temperature > 0 to sample):
python evaluation/evaluate_models.py \
    --model_path /path/to/checkpoints/global_step_300 \
    --base_model_path /path/to/Meta-Llama-3.1-8B-Instruct \
    --mode think --batch_size 32

# 3. Score:
python evaluation/compute_metrics_judge.py ./eval_results/<run_dir>/   # P=2 pairwise
python evaluation/compute_metrics_mcq.py   ./eval_results/<run_dir>/   # P=24 MCQ
```

Supported models: Llama-3.1-8B-Instruct, Qwen3-8B, Qwen3-32B (the Qwen scripts
work for both 8B and 32B; just change `MODEL_PATH`).

## Datasets

`dataset/train/` ships eight pre-built parquets (MCQ / Judge × `raw` / `filter`
× `_think` / `_thinking` chat templates). The training scripts default to
`raw`; switch to `filter` by editing `data.train_files` / `data.val_files`.

`dataset/test/` is **not shipped**. Download or build the test parquets for
your benchmarks, expand each instance into Full Permutation Expansion
(P=2 for Judge, P=24 for MCQ), drop them under `dataset/test/`, and the
inference script auto-discovers `*_<mode>.parquet`. Override the location
with `PAGRPO_DATASET_DIR=/your/path`.

## Modifications relative to upstream verl

Two upstream files are modified:

- `verl/trainer/ppo/ray_trainer.py` — adds `apply_group_baseline_from_returns()`
  (Cross-Permutation Advantage, Eq. 11–12 of the paper) and re-keys `uid` so all
  permutation variants of the same instance share a GRPO baseline group.
- `requirements.txt` — pinned to the versions used in our experiments.

All other PA-GRPO-specific additions live under `my_reward/`, `scripts/`,
`evaluation/`, and `dataset/train/`.

## Citation

```bibtex
@article{zheng2026mitigating,
  title   = {Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO},
  author  = {Zheng, Jinquan and Yuan, Jia and Yao, Jiacheng and Gu, Chenyang and Zheng, Pujun and He, Guoxiu},
  journal = {arXiv preprint arXiv:2603.21016},
  year    = {2026},
  note    = {To appear in ACL 2026}
}
```

## License

Based on [verl](https://github.com/volcengine/verl) (© 2023–2024 Bytedance Ltd.)
under Apache 2.0. PA-GRPO modifications are © 2026 East China Normal University,
also under Apache 2.0. See [`Notice.txt`](./Notice.txt) for full attribution.
