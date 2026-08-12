# STORM-PhysNet

**Physics-Informed Multi-Horizon Forecasting of Geostationary Relativistic Electron Flux with Transfer to Indian Longitude**

This repository contains the official implementation of **STORM-PhysNet**, accompanying the papers:

- Conference version (IEEE)
- Extended version (IEEE Access)

STORM-PhysNet is a Transformer-based model for multi-horizon forecasting of >2 MeV electron flux at geostationary orbit (GEO). It combines:

- A standard temporal Transformer encoder
- A learnable L1–Earth propagation delay module
- A \(B_z\)-conditioned physics gate
- Residual multi-horizon heads (45 min / 6 h / 12 h)

The model is evaluated under a rigorous multi-seed protocol and includes zero-shot + fine-tuned transfer experiments to GSAT-19 GRASP (Indian longitude).

---

## Key Results (Summary)

| System | PE<sub>45min</sub> | PE<sub>6h</sub> | PE<sub>12h</sub> | PE<sub>st,6h</sub> |
|-------------------------|--------------------|-----------------|------------------|--------------------|
| Transformer | 0.977 | 0.904 | 0.859 | 0.821 |
| STORM-Bz | **0.986** | 0.897 | 0.851 | 0.827 |
| Ensemble (α*=0.3) | 0.983 | **0.911** | **0.870** | 0.839 |
| STORM bagged (15 seeds) | **0.988** | 0.911 | 0.870 | **0.849** |

- Short-horizon gain is statistically significant (paired *p* = 0.002).
- Ablations show that the gain comes primarily from the overall training protocol rather than any single physics module at inference.
- Fine-tuning on GRASP raises 6 h PE from 0.449 → 0.599 and 12 h PE from 0.182 → 0.517.

---

## Extra Experiments (Access Paper)

The IEEE Access version includes two additional controlled experiments:

1. **Wider Delay Bound Ablation**  
   STORM was retrained with delay upper bounds of 2.0, 2.5, 3.0, 3.5, and 4.0 h (fifteen seeds each).  
   Mean PE<sub>45min</sub> stayed in the narrow range 0.9859–0.9862 and PE<sub>6h</sub> stayed in 0.900–0.902.  
   The original [0.5, 1.5] h constraint is therefore not a performance bottleneck.

2. **Bagged Transformer Control**  
   A sixteen-seed bagged Transformer reached mean PE<sub>45min</sub> ≈ 0.978 and PE<sub>6h</sub> ≈ 0.895.  
   While bagging improves the Transformer, the gap relative to bagged STORM remains.

Result summary (wider-delay ablation + bagged Transformer control) is in:

```text
results/summary.json
```

These experiments can also be reproduced from the master notebook (`notebooks/STORM_PhysNet_Colab.ipynb`).

---

## Repository Structure

```text
STORM-PhysNet/
├── configs/
│   └── config.yaml                 # Model, data, training hyperparameters
│                                   # forecast_horizons: [0.75, 6.0, 12.0]
│
├── datasets/
│   ├── goes/                       # GOES-15 EPEAD >2 MeV electron flux (CDF)
│   │   └── goes15_epead-*.cdf      # ~51 MB CDAWeb science file
│   ├── omni/                       # OMNI solar-wind + geomagnetic indices
│   │   ├── omni2.lst               # Hourly OMNI-2 time series
│   │   └── omni2.fmt               # Format description
│   └── grasp/                      # GSAT-19 GRASP 5-min averages (text)
│       └── grasp_5_min_avg_*.txt
│
├── results/
│   ├── summary.json                # Curated PE summary (matches the papers)
│   ├── all_results.csv             # Seed-level evaluation outputs
│   ├── wider_delay_results.csv     # Wider delay-bound ablation (15/16 seeds)
│   ├── bagged_tf_results.csv       # Bagged Transformer control (16 seeds)
│   ├── ablation_final_table.csv    # Ablation summary table
│   ├── wider_delay_pe6h.png        # Figure used in the Access paper
│   └── README.md                   # What each file is / what is not released
│
├── src/
│   ├── data/
│   │   ├── cdf_reader.py           # GOES CDF + OMNI readers (paper pipeline)
│   │   ├── preprocessor.py         # Feature build, chronological splits
│   │   ├── dataloader.py           # Windows, horizons [0.75, 6, 12], storm sampler
│   │   ├── synthetic_generator.py  # DEV ONLY — not used in paper runs
│   │   └── storm_augmentor.py      # DEV ONLY — not used in paper runs
│   ├── model/
│   │   ├── storm_physnet.py        # STORM-PhysNet (delay + Bz gate + heads)
│   │   ├── baselines.py            # VanillaTransformer, LSTM, MLP, CNN
│   │   ├── propagation_delay.py
│   │   ├── bz_gate.py
│   │   └── forecasting_heads.py
│   ├── training/
│   │   ├── trainer.py              # Main training loop (Adam, early stop, seeds)
│   │   ├── physics_loss.py
│   │   ├── horizon_physics_loss.py
│   │   └── transfer_learning.py    # GRASP fine-tune helpers
│   └── evaluation/
│       └── metrics.py              # PE_clim, PE_pers, RMSE helpers
│
├── notebooks/
│   └── STORM_PhysNet_Colab.ipynb   # Master reproduction scaffold
│
├── requirements.txt
└── README.md
```

**Notes**

- Paper experiments load **real** GOES / OMNI / GRASP files under `datasets/`.
- `synthetic_generator.py` and `storm_augmentor.py` are **not** imported by `Trainer` or the notebook pipeline.
- Full `.pt` checkpoints are **not** stored here (size). Seed-level CSVs under `results/` support the reported tables.

---

## Quick Start (Google Colab – Recommended)

1. Open [`notebooks/STORM_PhysNet_Colab.ipynb`](notebooks/STORM_PhysNet_Colab.ipynb) in Google Colab.
2. Set runtime to **T4 GPU**.
3. Run all cells.

The notebook supports two modes:

- `DEMO_MODE = True` → quick run (few epochs) for testing
- `DEMO_MODE = False` → full paper-level training

---

## Data

The papers use public datasets from NOAA NCEI (GOES-15), NASA OMNIWeb (OMNI), and ISSDC (GSAT-19 GRASP).

Sample files are included under `datasets/` for convenience. For full archives and redistribution terms, please use the official sources:

- GOES-15: https://www.ngdc.noaa.gov/stp/satellite/goes/
- OMNI: https://omniweb.gsfc.nasa.gov/
- GRASP: https://www.issdc.gov.in/

---

## Reproducibility Protocol (as used in the papers)

- **Split**: Purely chronological 70 / 15 / 15 % (no shuffling)
- **Seeds**: 15 independent random initializations (seeds 42–56)
- **Metrics**: PE<sub>clim</sub> (primary) and PE<sub>pers</sub>
- **Baselines**:
  - Transformer baseline (default hyperparameters: d_model=64, 3 layers, 4 heads) — not matched to STORM in width or depth
  - LSTM
- **Ablations**: No-Delay, No-Gate, No-Physics, horizon-restricted physics loss
- Test PE is computed **once** after training and never used for model selection

---

## Reproduction notes

- The Colab notebook is a **scaffold**: set `DEMO_MODE = False` for full 15-seed runs (GPU-heavy).
- Headline PE tables in the papers come from multi-account training; seed-level CSVs are in `results/`.
- `src/data/synthetic_generator.py` and `storm_augmentor.py` are **not** part of the paper pipeline.
- Transformer baseline uses default hyperparameters (`d_model=64`, 3 layers) and is **not** capacity-matched to STORM (`d_model=128`, 2 layers), as stated in the manuscripts.

---

## Citation

If you use this code or the results, please cite:

```bibtex
@article{samarth2026storm,
  title   = {STORM-PhysNet: A Multi-Horizon Transformer for Geostationary Relativistic Electron Flux Forecasting with Physics-Inspired Components and Cross-Satellite Transfer},
  author  = {Samarth BN},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

(Also cite the conference version once it is published.)

---

## License

Code is released under the MIT License for academic and research use.  
See the official data-provider terms for GOES, OMNI, and GRASP redistribution.

---

## Contact

**Samarth BN**  
RV College of Engineering, Bengaluru, India  
Email: samarthbn.ec25@rvce.edu.in
```
