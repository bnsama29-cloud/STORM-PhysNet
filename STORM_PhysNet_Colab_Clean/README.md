# STORM-PhysNet — Clean Colab Pipeline

Use **separate scripts in order**. Do not concatenate into one "Run all" cell.

## Drive layout (recommended)

```
MyDrive/storm_physnet/
  ieee_final_fixed.zip      # code: src/, configs/, run_training.py
  datasets.zip              # goes/, omni/, grasp/
  nb1_outputs/              # main multi-seed checkpoints
  nb2_outputs/              # ablations + classical baselines
  ieee_eval_outputs/        # tables, figures, JSON
```

## Order

| Step | Script | GPU | Writes |
|------|--------|-----|--------|
| 1 | `scripts/01_train_main.py` | Yes | `nb1_outputs/` |
| 2 | `scripts/02_train_ablations_baselines.py` | Yes | `nb2_outputs/` |
| 3 | `scripts/03_ieee_eval.py` | Yes (eval) | `ieee_eval_outputs/` |

Optional: skip step 2 if ablations/baselines already exist on Drive.

## Colab setup

1. Runtime → GPU (T4 is enough).
2. Mount Drive.
3. Edit paths at the top of each script (`DRIVE_CODE_ZIP`, `DRIVE_DATA_ZIP`, `DRIVE_OUT`).
4. Run **one script per session** (or sequential cells, but finish each phase before the next).

## Paper numbers

Main PE table must come from **nb1 multi-seed** eval (`storm_bz` vs `transformer`, seeds 42/43/44).

GRASP Table II in the paper came from the dedicated transfer study. If step-3 GRASP fine-tune is weaker, **keep the paper GRASP numbers** and treat step-3 as a reproducibility check only.

## Do not

- Claim MC-dropout bands as calibrated uncertainty in the main paper.
- Overwrite paper PE with single-seed or wrong storm-mask runs.
- Mix old README PE (0.42 / 0.9) into this pipeline.
