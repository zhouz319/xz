# PA-GRPO: Permutation-Aware Group Relative Policy Optimization

PA-GRPO mitigates **selection bias** in LLMs by jointly modeling permutation
variants of the same instance as a Permutation Group, encouraging
permutation-invariant decisions on MCQ and pairwise LLM-as-a-Judge tasks.

Forked from [verl](https://github.com/volcengine/verl) @ [`0eb50ec4`](https://github.com/volcengine/verl/commit/0eb50ec4a33cda97e05ed8caab9c7f17a30c05a9) (v0.7.0.dev). Upstream files (`docker/`, `docs/`, `examples/`, `recipe/`, `tests/`, `setup.py`, `pyproject.toml`) are **not** part of PA-GRPO.

## Install (Python 3.12, CUDA 12.4)

```bash
git clone https://github.com/ECNU-Text-Computing/PA-GRPO.git && cd PA-GRPO
conda create -n pagrpo python=3.12 -y && conda activate pagrpo
pip install -r requirements.txt   # or requirements-lock.txt for exact reproduction
```
> Do **not** `pip install -e .` — it would shadow the local `./verl/`.

## Run

```bash
# Train — set MODEL_PATH inside the script first
bash scripts/run_mcq_llama.sh   # or run_{mcq,judge}_{llama,qwen}.sh

# Inference (greedy; auto-discovers $PAGRPO_DATASET_DIR/*_<mode>.parquet)
python evaluation/evaluate_models.py --model_path /path/to/ckpt \
    --base_model_path /path/to/base --mode think --batch_size 32

# Score
python evaluation/compute_metrics_judge.py ./eval_results/<run>/   # P=2
python evaluation/compute_metrics_mcq.py   ./eval_results/<run>/   # P=24
```
`dataset/train/` ships eight pre-built parquets; `dataset/test/` is **not** shipped — drop your own `*_<mode>.parquet` files there.

## Modifications vs. upstream verl

`verl/trainer/ppo/ray_trainer.py` adds Cross-Permutation Advantage (`apply_group_baseline_from_returns`, Eq. 11–12) and re-keys `uid` so permutation variants share a GRPO baseline. `requirements.txt` is re-pinned. PA-GRPO additions: `my_reward/`, `scripts/`, `evaluation/`, `dataset/train/`.

## Citation & License

```bibtex
@article{zheng2026mitigating,
  title  = {Mitigating Selection Bias in LLMs via Permutation-Aware GRPO},
  author = {Zheng, Jinquan and Yuan, Jia and Yao, Jiacheng and Gu, Chenyang and Zheng, Pujun and He, Guoxiu},
  year   = {2026}, note = {To appear in ACL 2026}
}
```
Apache 2.0. verl © 2023–2024 Bytedance; PA-GRPO modifications © 2026 ECNU. See [`Notice.txt`](./Notice.txt).
