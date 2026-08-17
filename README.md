# STORM-PhysNet

**Physics-Informed Multi-Horizon Forecasting of Geostationary Relativistic Electron Flux with Transfer to Indian Longitude**

This repository contains the official implementation of **STORM-PhysNet**, accompanying the papers:

- Conference version (IEEE)
- Extended version (IEEE Access)

STORM-PhysNet is a Transformer-based model for multi-horizon forecasting of >2 MeV electron flux at geostationary orbit (GEO). It combines:

- A standard temporal Transformer encoder
- A learnable L1–Earth propagation delay module
- A \(B_z\)-conditioned physics gate
- Residual multi-horizon heads (1 h / 6 h / 12 h)

The model is evaluated under a rigorous multi-seed protocol and includes zero-shot + fine-tuned transfer experiments to GSAT-19 GRASP (Indian longitude).



## Key Results (Summary)

**Official GOES PE = Aug 2026 full retrain + eval zip; GRASP = new-weight transfer.**

### GOES seed-mean PE
| Model | PE₁ₕ | PE₆ₕ | PE₁₂ₕ | PE_st,6h | PE_pers 1h |
|-------|------|------|-------|----------|------------|
| LSTM | 0.955 | 0.881 | 0.840 | 0.788 | −1.19 |
| Transformer | 0.978 | 0.895 | 0.845 | 0.797 | −0.08 |
| STORM-Bz | **0.986** | 0.900 | 0.854 | 0.812 | **+0.31** |
| No-Delay | 0.987 | 0.902 | 0.855 | 0.809 | +0.36 |
| No-Physics | 0.986 | 0.899 | 0.850 | 0.807 | +0.31 |
| No-Gate | 0.986 | 0.900 | 0.856 | 0.819 | +0.32 |
| TF matched | 0.980 | 0.895 | 0.845 | 0.809 | −0.01 |
| RDG / RDG-S / SDG | 0.986 | ~0.901 | ~0.855 | ~0.815 | ~0.32 |

### True bagged (n=15)
| Model | PE₁ₕ | PE₆ₕ | PE₁₂ₕ | PE_st,6h |
|-------|------|------|-------|----------|
| STORM-Bz bagged | 0.987 | **0.910** | **0.870** | 0.836 |
| TF matched bagged | 0.984 | 0.908 | 0.861 | 0.831 |
| TF bagged | 0.984 | 0.907 | 0.861 | 0.816 |

- Short-horizon STORM gain vs Transformer (0.986 vs 0.978); PE_pers ≈ +0.31 vs -0.08.
- Ablations show that the gain comes primarily from the overall training protocol rather than any single physics module at inference.
- Fine-tuning on GRASP raises 6 h PE from **0.740 → 0.841** and 12 h PE from **0.567 → 0.762**.
- Architecture-matched Transformer control reaches mean PE (1 h) = 0.980 and PE (6 h) = 0.895; bagging yields 0.984 / 0.908 / 0.861.

---

## Extra Experiments (Access Paper)

The IEEE Access version includes additional controlled experiments:

1. **Architecture-Matched Transformer Control**  
   A Vanilla Transformer with the same encoder hyperparameters ($d_{\mathrm{model}}$, layers, heads) as STORM
   ($d_{\mathrm{model}}=128$, two layers, four heads;
   $\approx1.19\times10^{6}$ parameters) was trained under the same
   chronological split and fifteen seeds (seeds 42–56).
   - Mean: PE<sub>1h</sub>=0.980, PE<sub>6h</sub>=0.895, PE<sub>12h</sub>=0.845
   - Bagged (15 seeds): PE<sub>1h</sub>=0.984, PE<sub>6h</sub>=0.908, PE<sub>12h</sub>=0.861
   - Matching encoder hyperparameters shrinks the short-horizon gap versus STORM-Bz
     and improves multi-horizon PE under bagging.

2. **Wider Delay Bound Ablation**  
   STORM was retrained with delay upper bounds of 2.0, 2.5, 3.0, 3.5, and 4.0 h (fifteen seeds each).  
   Mean PE<sub>1h</sub> stayed in the narrow range 0.9859–0.9862 and PE<sub>6h</sub> stayed in 0.900–0.902.  
   The original [0.5, 1.5] h constraint is therefore not a performance bottleneck.

3. **Supplementary gates (Access)**  
   - **RDG** / **RDG-S** / **SDG**: alternative nonlinearities (experimental variants)
   - Mean PE ≈ 0.986 / 0.901 / 0.855; none replaces STORM-Bz as the primary system
   - Artifacts: `checkpoints/storm_cathode*/`, `checkpoints/storm_radiotrophic/`, and `results/alt_gates_summary.csv`

Result summaries are in:

```text
results/table_main_means.csv
results/table_bagged.csv
results/ensemble_summary.json
results/table_grasp_storm_bz.csv
results/alt_gates_summary.csv
```

These experiments can also be reproduced from the master notebook ([Open in Colab](https://colab.research.google.com/github/bnsama29-cloud/STORM-PhysNet/blob/main/notebooks/STORM_PhysNet_Master.ipynb)).

---

## Repository Structure

```text
STORM-PhysNet/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── config.yaml                 # Model, data, training hyperparameters (STORM)
│   │                               # forecast_horizons: [1.0, 6.0, 12.0]
│   └── config_transformer_baseline.yaml # Parameters for default 64-dim 3-layer TF
├── src/
│   ├── data/
│   │   ├── cdf_reader.py           # GOES CDF + OMNI readers (paper pipeline)
│   │   ├── preprocessor.py         # Feature build, chronological splits
│   │   └── dataloader.py           # Windows, horizons [1.0, 6.0, 12.0], storm sampler
│   ├── model/
│   │   ├── storm_physnet.py        # STORM-PhysNet (delay + Bz gate + heads)
│   │   ├── baselines.py            # VanillaTransformer, LSTM, MLP, CNN
│   │   ├── propagation_delay.py
│   │   ├── bz_gate.py
│   │   └── forecasting_heads.py
│   ├── training/
│   │   ├── trainer.py              # Main training loop (Adam, early stop, seeds)
│   │   ├── physics_loss.py
│   │   └── transfer_learning.py    # GRASP fine-tune helpers
│   └── evaluation/
│       └── metrics.py              # PE_clim, PE_pers, RMSE helpers
├── notebooks/
│   ├── README.md                   # Colab instructions
│   └── STORM_PhysNet_Master.ipynb  # Master reproduction notebook
├── figures/                         # ALL paper figures
├── results/
│   ├── table_main_means.csv         # Mean Performance across 15 seeds for baselines vs STORM
│   ├── table_main_stats.csv         # Standard deviations for the main evaluation metrics
│   ├── table_means_bootstrap_ci.csv # Bootstrap confidence intervals
│   ├── table_bagged.csv             # Ensemble Bagged prediction efficiency
│   ├── table_parameter_counts.csv   # Model parameter sizes
│   ├── table_grasp_storm_bz.csv     # GRASP domain transfer table
│   ├── ablation_final_table.csv     # Delay and physics-informed gates ablation
│   ├── alt_gates_summary.csv        # PE for alternative physical gates
│   ├── ensemble_summary.json        # val-selected α* linear mix (not seed-bagging)
│   ├── grasp_summary.csv            # GRASP metrics summary
│   ├── all_seed_results_full.csv    # The complete 15-seed data block
│   └── README.md
├── checkpoints/
│   ├── README.md
│   └── (All trained checkpoints for seeds 42-56 across all models)
└── datasets/                        # only if you already ship them
    ├── goes/
    ├── omni/
    └── grasp/
```

**Notes**

- Paper experiments load **real** GOES / OMNI / GRASP files under `datasets/`.
- Trained checkpoints for seeds 42–56 are provided under `checkpoints/`.
- Result CSVs and curated PE summaries are under `results/`.
- Supplementary extra experiments (Noise Robustness, Operational Persistence PE, GRASP fine-tuning) are located in `results/` and are fully reproducible via the updated Colab notebook script.
- All paper figures are under `figures/`.

## Experimental code (not in paper tables)
`src/model/experimental/` (analogy gates, spectral head, magnetopause)
and optional SSM/iTransformer paths are exploratory. Reported results
use `backbone: transformer` and the physics Bz gate only.

---

## Quick Start (Google Colab – Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bnsama29-cloud/STORM-PhysNet/blob/main/notebooks/STORM_PhysNet_Master.ipynb)

1. Open [`notebooks/STORM_PhysNet_Master.ipynb`](https://colab.research.google.com/github/bnsama29-cloud/STORM-PhysNet/blob/main/notebooks/STORM_PhysNet_Master.ipynb) in Google Colab.
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
  - **Transformer matched** (d_model=128, 2 layers, 4 heads; ~1.19e6 params) — architecture-matched control, fifteen seeds
  - LSTM
- **Ablations**: No-Delay, No-Gate, No-Physics, horizon-restricted physics loss
- Test PE is computed **once** after training and never used for model selection

---

## Reproduction notes

- **Note:** Headline PE tables come from the August 2026 full Kaggle retrain (15 seeds × 10 systems). The released notebook is a pipeline scaffold, not a one-click 15-seed reproduction. See `results/table_main_means.csv` and `results/all_seed_results_full.csv`.
- Transformer baseline uses default hyperparameters (`d_model=64`, 3 layers) and is **not** architecture-matched to STORM (`d_model=128`, 2 layers), as stated in the manuscripts.
- An architecture-matched Transformer control (`d_model=128`, 2 layers, 4 heads) was trained for fifteen seeds; checkpoints are under `checkpoints/transformer_matched/seed_{42..56}/`.
- Best checkpoints for all main models and seeds 42–56 are included under `checkpoints/`.

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
