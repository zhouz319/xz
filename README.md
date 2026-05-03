# PA-GRPO: Permutation-Aware Group Relative Policy Optimization

This repository provides the implementation of **PA-GRPO (Permutation-Aware Group Relative Policy Optimization)** based on the [verl](https://github.com/volcengine/verl) framework.

> **Note**: This repository is forked from [verl](https://github.com/volcengine/verl) at commit
> [`0eb50ec4`](https://github.com/volcengine/verl/commit/0eb50ec4a33cda97e05ed8caab9c7f17a30c05a9) (v0.7.0.dev).
> Upstream directories (`docker/`, `docs/`, `examples/`, `recipe/`, `tests/`,
> `CONTRIBUTING.md`, `.pre-commit-config.yaml`, `.readthedocs.yaml`, `setup.py`,
> `pyproject.toml`) are kept verbatim for traceability and **are not part of
> the PA-GRPO contribution**. Every PA-GRPO addition or modification is listed
> in [`CHANGES.md`](./CHANGES.md).

## 📋 Overview

PA-GRPO is proposed to mitigate **Selection Bias** in LLMs induced by non-semantic factors (position, labels) in MCQ reasoning and LLM-as-a-Judge pairwise evaluation tasks.

Unlike standard preference optimization that treats each prompt as an independent training sample, PA-GRPO **jointly models multiple permutation variants of the same semantic instance as a Permutation Group**, and explicitly aligns cross-permutation consistency during optimization. This encourages the model to learn **permutation-invariant semantic decision strategies**.

**Main Objectives:**
- Improve **Permutation Invariance** and decision consistency
- Maintain task performance while achieving the above goals

### Supported Tasks

| Task | Model | Description |
|------|-------|-------------|
| MCQ | Llama-3.1-8B / Qwen3-8B / Qwen3-32B | Multiple-choice question answering |
| Judge | Llama-3.1-8B / Qwen3-8B / Qwen3-32B | Pairwise response comparison |

## 🛠️ Installation

### Prerequisites

- Python: 3.12.0
- PyTorch: 2.6.0+cu124
- CUDA: 12.4

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/ECNU-Text-Computing/PA-GRPO.git
cd PA-GRPO

# Create conda environment (recommended)
conda create -n pagrpo python=3.12
conda activate pagrpo

# Install dependencies
pip install -r requirements.txt
# For a bit-for-bit reproduction of the paper environment, use the lock file instead:
# pip install -r requirements-lock.txt
```

> **Do NOT run `pip install -e .`**. The bundled `setup.py` / `pyproject.toml`
> are inherited from upstream verl and would register a *system-wide* `verl`
> package, which can shadow PA-GRPO's local modifications and conflict with
> upstream verl installed elsewhere. Run training and evaluation scripts
> directly from the project root — Python will import the modified `./verl/`
> via normal relative imports.

## 📁 Project Structure

```
PA-GRPO/
├── dataset/                    # Datasets
│   ├── train/                  # Training data (.parquet files, included)
│   └── test/                   # Evaluation data (NOT included — see §Evaluation below)
├── my_reward/                  # Custom reward functions
│   ├── mcq_llama.py           # MCQ reward for Llama (uses <think> tag)
│   ├── mcq_qwen.py            # MCQ reward for Qwen (uses <thinking> tag)
│   ├── judge_llama.py         # Judge reward for Llama (uses <think> tag)
│   └── judge_qwen.py          # Judge reward for Qwen (uses <thinking> tag)
├── scripts/                    # Training scripts
│   ├── run_mcq_llama.sh       # MCQ training with Llama
│   ├── run_mcq_qwen.sh        # MCQ training with Qwen
│   ├── run_judge_llama.sh     # Judge training with Llama
│   └── run_judge_qwen.sh      # Judge training with Qwen
├── evaluation/                 # Inference & metrics
│   ├── evaluate_models.py     # Run inference (vLLM/Transformers), write per-sample JSON
│   ├── compute_metrics_judge.py  # Acc / Con / CA for pairwise (P=2) Judge benchmarks
│   └── compute_metrics_mcq.py    # Acc / Con / CA for MCQ (P=24, full permutation expansion)
├── verl/                       # verl framework source code (modified)
├── requirements.txt            # PA-GRPO direct dependencies (default install)
├── requirements-lock.txt       # Pinned `pip freeze` lock for exact reproduction
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Prepare Your Model

Download or prepare your base model:
- **Llama-3.1-8B-Instruct**: [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- **Qwen3-8B**: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- **Qwen3-32B**: [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)

> **Note**: The Qwen scripts (`run_mcq_qwen.sh`, `run_judge_qwen.sh`) support both Qwen3-8B and Qwen3-32B. Just change the `MODEL_PATH` variable.

### 2. Configure Model Path

Edit the training script to set your model path:

```bash
# Edit scripts/run_mcq_llama.sh (or other scripts)
MODEL_PATH="/path/to/your/Meta-Llama-3.1-8B-Instruct"
```

### 3. Run Training

```bash
# MCQ task with Llama
bash scripts/run_mcq_llama.sh

# MCQ task with Qwen
bash scripts/run_mcq_qwen.sh

# Judge task with Llama
bash scripts/run_judge_llama.sh

# Judge task with Qwen
bash scripts/run_judge_qwen.sh
```

### 4. Monitor Training

```bash
# View training logs
tail -f mcq_llama3_1_8b.log

# Launch TensorBoard
tensorboard --logdir=logs/
```

### 5. Evaluate

Evaluation runs in two stages: inference with `evaluation/evaluate_models.py`, then metrics
with `evaluation/compute_metrics_{judge,mcq}.py`.

**Evaluation data is not shipped with this repository.** Download or build the test
parquets yourself for the seven benchmarks reported in the paper (MT-Bench, JudgeBench,
PreferenceBench, RewardBench, ARC-Challenge, TinyMMLU, GPQA), expand each test instance
into Full Permutation Expansion (P = 2 for Judge, P = 24 for MCQ), and place the parquet
files under `dataset/test/`. (Or set `PAGRPO_DATASET_DIR` to wherever they live.)

```bash
# Run inference (greedy by default; pass --temperature > 0 to sample)
python evaluation/evaluate_models.py \
    --model_path /path/to/checkpoints/global_step_300 \
    --base_model_path /path/to/Meta-Llama-3.1-8B-Instruct \
    --mode think \
    --batch_size 32

# Compute Acc / Con / CA for pairwise Judge benchmarks (P=2)
python evaluation/compute_metrics_judge.py ./eval_results/<run_dir>/

# Compute Acc / Con / CA for MCQ benchmarks (P=24, full permutation expansion)
python evaluation/compute_metrics_mcq.py ./eval_results/<run_dir>/
```

Both `compute_metrics_*` scripts can take a single results JSON or a directory; given a
directory, they recursively find all `*results.json` files and write a per-run Excel
summary alongside them.


## 📊 Datasets

`dataset/train/` ships eight pre-built parquets covering both tasks (MCQ / Judge),
two variants (`raw` = all permutation-inconsistent instances; `filter` = a smaller
hard subset), and two chat-template tags (`_think` for Llama, `_thinking` for Qwen3).
Pick the one matching your task and policy model — the training scripts in
`scripts/run_*.sh` default to `raw`. To use `filter`, edit `data.train_files` /
`data.val_files` in the corresponding script.

## 🔧 Key Modifications

### Custom Reward Functions (`my_reward/`)

- **Answer Extraction**: Parse `<answer>X</answer>` tags
- **Permutation Mapping**: Map permuted answers to canonical positions
- **Consistency Scoring**: Reward permutation-invariant predictions

### Modified verl Files

- **`verl/trainer/ppo/ray_trainer.py`**: Modified to support PA-GRPO's permutation group training logic.

## 📚 Citation

If you find PA-GRPO useful in your research, please cite:

```bibtex
@article{zheng2026mitigating,
  title   = {Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO},
  author  = {Zheng, Jinquan and Yuan, Jia and Yao, Jiacheng and Gu, Chenyang and Zheng, Pujun and He, Guoxiu},
  journal = {arXiv preprint arXiv:2603.21016},
  year    = {2026},
  note    = {To appear in ACL 2026}
}
```

## 📝 License

This project is based on [verl](https://github.com/volcengine/verl) (Copyright 2023-2024 Bytedance Ltd. and/or its affiliates) and follows its Apache 2.0 License.

PA-GRPO modifications are © 2026 East China Normal University, also released under the Apache 2.0 License.
See [`Notice.txt`](./Notice.txt) for full attribution.
