# PRISM: Predictive Risk-Informed Safe Motion Planning for Dynamic Fire Environments

PRISM is a PyTorch research prototype for risk-aware robot motion planning in dynamic toy fire environments. The final planner, PRISM-Fusion, combines a current-risk prior with uncertainty-aware predictive safe-risk refinement.

The final planner risk is:

```text
M_safe = mu + lambda_u * sigma
M_final = alpha * M_current + (1 - alpha) * M_safe
```

The default final setting is `alpha = 0.4`, `lambda_u = 0.5`, `uncertainty_alpha = 0.7`, and `num_mc_samples = 5`. In words, the planner uses 40% current-risk prior and 60% predictive safe-risk refinement.

This project intentionally does not include PPO, SAC, CPO, reinforcement-learning training, MPC, or FDS integration.

## Repository Structure

```text
configs/      Experiment, demo, and reproduction configs
datasets/     Dataset loaders
envs/         Toy fire environments
models/       PRISM risk predictor
planners/     Trajectory sampling, safe-risk scoring, and fusion logic
scripts/      Dataset generation, training, evaluation, plotting, and packaging
trainers/     Training loop
utils/        Losses, uncertainty, and device helpers
```

Generated outputs, datasets, checkpoints, and archives are ignored by Git.

## Installation

Use Python 3.10. For the RTX 5070 Ti / CUDA 12.8 environment used in development:

```bash
conda create -n prism python=3.10 -y
conda activate prism
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-no-torch.txt
pip install -e .
```

If your GPU does not use CUDA 12.8, install the appropriate official PyTorch build for your platform first, then install `requirements-no-torch.txt`.

## Smoke Tests

```bash
python scripts/test_env.py
python scripts/test_shapes.py --config configs/smoke.yaml
python scripts/test_planner.py --config configs/smoke.yaml
```

## Reviewer Demo

The reviewer demo regenerates a tiny toy dataset, trains for 3 epochs by default, runs prediction and closed-loop checks, then runs a small PRISM-Fusion planner demo. It does not depend on existing datasets or checkpoints.

```bash
python scripts/run_reviewer_demo.py --config configs/reviewer_demo.yaml
```

For an even faster local check:

```bash
python scripts/run_reviewer_demo.py --config configs/reviewer_demo.yaml --epochs 1
```

Summary output:

```text
outputs/results/reviewer_demo_summary.txt
```

## Full Reproduction

The Level-A reproduction regenerates the dataset, trains a fresh checkpoint, evaluates prediction quality, runs closed-loop evaluation, and runs the final PRISM-Fusion comparison.

```bash
python scripts/run_full_reproduction.py --config configs/reproduction_level_a.yaml
```

Expected outputs:

```text
outputs/results/full_reproduction_summary.txt
outputs/results/full_reproduction_main_table.csv
outputs/visualizations/full_reproduction_main_comparison.png
```

## Paper-Ready Tables And Figures

After Stage-5 result CSVs are available locally, build Stage-6 reviewer tables, figures, summary, and the protected result zip:

```bash
python scripts/run_stage6_finalize_experiments.py
```

The package is written to:

```text
outputs/prism_stage6_paper_ready_results.zip
```

The packager excludes checkpoints, generated datasets, NumPy arrays, and nested zip archives.

## Expected Trend

In the hard dynamic fire benchmark, the Current-Risk Planner achieved approximately success `0.78` and collision `0.17`. PRISM-Fusion achieved approximately success `0.85` and collision `0.13`.

The final PRISM planner uses current-risk prior and uncertainty-aware predictive safe-risk refinement. Compared with the current-risk-only baseline, alpha-fusion improves success rate and reduces collision rate in hard dynamic fire scenarios.

## Artifact Policy

The repository does not include:

- Generated `outputs/`
- Trained checkpoints (`.pt`, `.pth`, `.ckpt`)
- Generated datasets or NumPy arrays (`.npz`, `.npy`)
- Result archives (`.zip`)

These artifacts are regenerated locally by the provided scripts and remain ignored by Git.

## Citation

```bibtex
@article{xiao2026prism,
  title={PRISM: Predictive Risk-Informed Safe Motion Planning for Dynamic Fire Environments},
  author={Xiao, Kaiyan and Cai, Can},
  journal={Under review},
  year={2026}
}
```
