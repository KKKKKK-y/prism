# PRISM

Predictive Risk Intelligence and Safety Management.

## Stage 1 Validation

Run these commands from the PRISM project root:

```bash
pip install -e .
python scripts/test_env.py
python scripts/train.py --config configs/smoke.yaml --debug
python scripts/test_shapes.py --config configs/smoke.yaml
python scripts/visualize_prediction.py --config configs/smoke.yaml --checkpoint prism/outputs/checkpoints/best.pt
```

The visualization script tolerates a missing checkpoint and will use a randomly initialized Stage 1 model with a warning.

## Stage 1 Risk Prediction Training

Train the next-step risk map predictor and write the best validation checkpoint to `prism/outputs/checkpoints/best.pt`:

```bash
python scripts/train_stage1.py
python scripts/visualize_prediction.py
```

`train_stage1.py` uses `DummyFireRiskDataset` and `RiskPredictor`, optimizes only next-step mean risk with MSE loss, and reports train/validation loss for each epoch.

## Stage 2

Stage 2: Uncertainty Propagation and Safe-Risk Transformation.

Run these commands from the PRISM project root:

```bash
python scripts/test_shapes.py --config configs/smoke.yaml
python scripts/test_uncertainty.py --config configs/smoke.yaml --checkpoint prism/outputs/checkpoints/best.pt
python scripts/visualize_uncertainty.py --config configs/smoke.yaml --checkpoint prism/outputs/checkpoints/best.pt
```

Expected success messages:

```text
All PRISM Stage-2 uncertainty propagation checks passed.
Saved uncertainty visualization to: prism/outputs/visualizations/stage2_uncertainty.png
```

## Stage 3

Stage 3: Trajectory-Level Safe-Risk Constraint.

Run these commands from the PRISM project root:

```bash
python scripts/test_planner.py --config configs/smoke.yaml
python scripts/visualize_planner.py --config configs/smoke.yaml --checkpoint prism/outputs/checkpoints/best.pt
```

Expected success messages:

```text
All PRISM Stage-3 trajectory safe-risk checks passed.
Saved planner visualization to: prism/outputs/visualizations/stage3_planner.png
```

Trajectory coordinates use `[x, y]`. Safe-risk maps use image indexing `[y, x]`, where `y` is row and `x` is column.

## Stage 4

Stage 4: Closed-Loop Toy Fire Environment Evaluation.

Run these commands from the PRISM project root:

```bash
python scripts/visualize_toy_env.py --config configs/smoke.yaml
python scripts/run_closed_loop.py --config configs/smoke.yaml --checkpoint prism/outputs/checkpoints/best.pt
python scripts/evaluate_closed_loop.py --config configs/smoke.yaml --checkpoint prism/outputs/checkpoints/best.pt --episodes 20
```

Expected artifacts:

```text
outputs/visualizations/stage4_toy_env_dynamics.png
outputs/visualizations/stage4_closed_loop.png
outputs/results/stage4_closed_loop_results.csv
```

ToyFireEnv uses robot/action coordinates `[x, y]`; fire, smoke, risk and obstacle maps use image indexing `[y, x]`.

## Stage 4.1

Stage 4.1: Closed-Loop Stabilization and Debugging.

The toy environment limits each executed robot step with `env.max_step_size`, while the planner scores trajectories with safe-risk, goal progress and a backtrack penalty.

```bash
python scripts/run_closed_loop.py --config configs/smoke.yaml --checkpoint prism/outputs/checkpoints/best.pt --verbose
python scripts/evaluate_closed_loop.py --config configs/smoke.yaml --checkpoint prism/outputs/checkpoints/best.pt --episodes 20
python scripts/sweep_stage4_params.py --config configs/smoke.yaml --checkpoint prism/outputs/checkpoints/best.pt --episodes 10
```

The sweep writes:

```text
outputs/results/stage4_param_sweep.csv
```

## Stage 4.2

Stage 4.2: ToyFireEnv Dataset Generation and Supervised Training.

Run these commands from the PRISM project root:

```bash
python scripts/generate_toy_dataset.py --config configs/toy_train.yaml --train_episodes 100 --val_episodes 20 --test_episodes 20
python scripts/test_toy_dataset.py --npz outputs/datasets/toy_fire_train.npz
python scripts/train.py --config configs/toy_train.yaml
python scripts/evaluate_prediction_on_toy.py --config configs/toy_train.yaml --checkpoint outputs/checkpoints_toy/best.pt
python scripts/visualize_prediction.py --config configs/toy_train.yaml --checkpoint outputs/checkpoints_toy/best.pt --output outputs/visualizations/stage4_toy_prediction.png
python scripts/evaluate_closed_loop.py --config configs/toy_train.yaml --checkpoint outputs/checkpoints_toy/best.pt --episodes 20
```

For a quick smoke run:

```bash
python scripts/generate_toy_dataset.py --config configs/toy_train.yaml --train_episodes 20 --val_episodes 5 --test_episodes 5
python scripts/train.py --config configs/toy_train.yaml --debug
```

## Stage 4.3

Stage 4.3: Formal Toy Training and Prediction Evaluation.

`configs/toy_train.yaml` trains on the generated ToyFireEnv NPZ dataset with `obs_window=4`, `horizon=5`, `in_channels=5`, and writes the best validation checkpoint to `outputs/checkpoints_toy/best.pt`. Training logs are saved to `outputs/results/stage4_toy_training_log.csv`.

```bash
python scripts/train.py --config configs/toy_train.yaml
python scripts/evaluate_prediction_on_toy.py --config configs/toy_train.yaml --checkpoint outputs/checkpoints_toy/best.pt
python scripts/plot_training_curve.py --csv outputs/results/stage4_toy_training_log.csv --output outputs/visualizations/stage4_toy_training_curve.png
python scripts/plot_prediction_metrics.py --csv outputs/results/stage4_toy_prediction_metrics.csv --output outputs/visualizations/stage4_toy_prediction_metrics.png
python scripts/visualize_prediction.py --config configs/toy_train.yaml --checkpoint outputs/checkpoints_toy/best.pt --output outputs/visualizations/stage4_toy_prediction_horizons.png --all_horizons
```

For a short end-to-end verification run:

```bash
python scripts/train.py --config configs/toy_train.yaml --debug
python scripts/run_stage4_3_pipeline.py --config configs/toy_train.yaml --debug
```

Expected artifacts:

```text
outputs/results/stage4_toy_training_log.csv
outputs/results/stage4_toy_prediction_metrics.csv
outputs/visualizations/stage4_toy_training_curve.png
outputs/visualizations/stage4_toy_prediction_metrics.png
outputs/visualizations/stage4_toy_prediction_horizons.png
```
