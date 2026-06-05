# PRISM

Predictive Risk Intelligence and Safety Management.

PRISM is a PyTorch research prototype for future fire-risk prediction and safe closed-loop navigation in a toy fire environment. The current codebase covers supervised risk forecasting, uncertainty propagation, trajectory-level safe-risk constraints, ToyFireEnv dataset generation, and closed-loop evaluation.

## Project Overview

The model consumes the past `obs_window=4` observations:

```text
obs: [B, 4, C, 64, 64]
```

and predicts the next `horizon=5` risk maps:

```text
mu:      [B, 5, 1, 64, 64]
log_var: [B, 5, 1, 64, 64]
```

The core model is:

```text
CNN Encoder -> GRU Temporal Module -> Mean Head / Variance Head
```

ToyFireEnv uses robot/action/trajectory coordinates `[x, y]`. Risk, smoke, fire and obstacle maps use image indexing `[y, x]`.

## Current Status

Implemented stages:

- Stage 1: risk prediction
- Stage 2: uncertainty propagation and safe-risk maps
- Stage 3: trajectory-level safe-risk constraint
- Stage 4: closed-loop ToyFireEnv evaluation
- Stage 4.1: closed-loop stabilization
- Stage 4.2: ToyFireEnv dataset generation
- Stage 4.3: formal toy training and prediction evaluation pipeline

This project intentionally does not include PPO, SAC, CPO, reinforcement-learning training, MPC, or real FDS integration.

## Environment Setup

Use Python 3.10+ and PyTorch 2.x.

[Open PRISM in Colab](https://colab.research.google.com/github/KKKKKK-y/prism/blob/main/PRISM_Colab.ipynb)

For a formal Colab T4 run, use:

```text
FORMAL_TRAIN_EPISODES = 300
FORMAL_VAL_EPISODES = 60
FORMAL_TEST_EPISODES = 60
FORMAL_EPOCHS = 50
FORMAL_EVAL_EPISODES = 100
```

If VRAM or time is tight, use:

```text
FORMAL_TRAIN_EPISODES = 150
FORMAL_VAL_EPISODES = 30
FORMAL_TEST_EPISODES = 30
FORMAL_EPOCHS = 30
FORMAL_EVAL_EPISODES = 50
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

On macOS, the code automatically uses MPS when available. On Ubuntu with an RTX GPU, it automatically uses CUDA when available.

## Quick Smoke Test

MacBook or CPU debug flow:

```bash
python scripts/check_project_ready.py
python scripts/test_env.py
python scripts/test_shapes.py --config configs/smoke.yaml
python scripts/test_uncertainty.py --config configs/smoke.yaml --checkpoint outputs/checkpoints/best.pt
python scripts/test_planner.py --config configs/smoke.yaml
python scripts/train.py --config configs/smoke.yaml --debug
```

Missing checkpoints in uncertainty/visualization scripts are allowed; those scripts warn and continue with random model weights.

## Toy Dataset Generation

Small local debug dataset:

```bash
python scripts/generate_toy_dataset.py --config configs/toy_train.yaml --train_episodes 5 --val_episodes 2 --test_episodes 2
python scripts/test_toy_dataset.py --npz outputs/datasets/toy_fire_train.npz
```

Formal Ubuntu dataset:

```bash
python scripts/generate_toy_dataset.py --config configs/toy_train.yaml --train_episodes 300 --val_episodes 60 --test_episodes 60
```

Generated `.npz` files are written to `outputs/datasets/` and are intentionally ignored by Git.

## Formal Training on Ubuntu

Train the Stage 4.3 ToyFireEnv predictor:

```bash
python scripts/train.py --config configs/toy_train.yaml
```

Resume interrupted training from the latest checkpoint:

```bash
python scripts/train.py \
  --config configs/toy_train.yaml \
  --resume outputs/checkpoints_toy/last.pt
```

Run the formal Stage 4.3 pipeline in debug mode:

```bash
python scripts/run_stage4_3_pipeline.py --config configs/toy_train.yaml --debug
```

The best validation checkpoint is saved to:

```text
outputs/checkpoints_toy/best.pt
```

Training logs are saved to:

```text
outputs/results/stage4_toy_training_log.csv
```

The formal run summary is saved to:

```text
outputs/results/formal_run_summary.txt
```

Package checkpoints, results, visualizations and key config files:

```bash
python scripts/package_results.py --output outputs/prism_formal_results.zip
```

Plot the training curve:

```bash
python scripts/plot_training_curve.py --csv outputs/results/stage4_toy_training_log.csv --output outputs/visualizations/stage4_toy_training_curve.png
```

## Prediction Evaluation

Evaluate all prediction horizons `t+1` through `t+5`:

```bash
python scripts/evaluate_prediction_on_toy.py --config configs/toy_train.yaml --checkpoint outputs/checkpoints_toy/best.pt
```

Plot horizon-level MAE/RMSE:

```bash
python scripts/plot_prediction_metrics.py --csv outputs/results/stage4_toy_prediction_metrics.csv --output outputs/visualizations/stage4_toy_prediction_metrics.png
```

Visualize target, predicted mean, and absolute error for every horizon:

```bash
python scripts/visualize_prediction.py --config configs/toy_train.yaml --checkpoint outputs/checkpoints_toy/best.pt --output outputs/visualizations/stage4_toy_prediction_horizons.png --all_horizons
```

## Closed-Loop Evaluation

Single closed-loop rollout:

```bash
python scripts/run_closed_loop.py --config configs/toy_train.yaml --checkpoint outputs/checkpoints_toy/best.pt
```

Multi-episode closed-loop evaluation:

```bash
python scripts/evaluate_closed_loop.py --config configs/toy_train.yaml --checkpoint outputs/checkpoints_toy/best.pt --episodes 100
```

Optional parameter sweep:

```bash
python scripts/sweep_stage4_params.py --config configs/toy_train.yaml --checkpoint outputs/checkpoints_toy/best.pt --episodes 10
```

## Project Structure

```text
prism/
├── configs/
│   ├── smoke.yaml
│   ├── toy_train.yaml
│   └── prism.yaml
├── datasets/
│   ├── __init__.py
│   └── dataset.py
├── envs/
│   ├── __init__.py
│   └── toy_fire_env.py
├── models/
│   ├── __init__.py
│   └── model.py
├── planners/
│   ├── __init__.py
│   ├── trajectory_sampler.py
│   └── safe_risk_planner.py
├── trainers/
│   ├── __init__.py
│   └── trainer.py
├── utils/
│   ├── __init__.py
│   ├── device.py
│   ├── losses.py
│   ├── metrics.py
│   ├── seed.py
│   └── uncertainty.py
├── scripts/
│   ├── check_project_ready.py
│   ├── test_env.py
│   ├── test_shapes.py
│   ├── test_uncertainty.py
│   ├── test_planner.py
│   ├── generate_toy_dataset.py
│   ├── test_toy_dataset.py
│   ├── train.py
│   ├── evaluate_prediction_on_toy.py
│   ├── evaluate_closed_loop.py
│   ├── visualize_prediction.py
│   ├── visualize_uncertainty.py
│   ├── visualize_planner.py
│   ├── visualize_toy_env.py
│   ├── run_closed_loop.py
│   ├── run_stage4_3_pipeline.py
│   ├── plot_training_curve.py
│   └── plot_prediction_metrics.py
├── outputs/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── config.py
```

## Notes for RTX 5070 Ti

- Install a CUDA-enabled PyTorch build that matches the Ubuntu NVIDIA driver.
- Keep generated datasets, checkpoints and TensorBoard logs under `outputs/`; they are ignored by Git.
- Use `configs/toy_train.yaml` for formal supervised training.
- Start with the small dataset generation command before launching the 300/60/60 episode dataset build.
- If CUDA is available, `select_device()` chooses `cuda`; otherwise it falls back to `mps` or `cpu`.
