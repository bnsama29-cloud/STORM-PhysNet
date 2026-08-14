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

## Reproduction notes

- The Colab notebook is a **scaffold**: set `DEMO_MODE = False` for full 15-seed runs (GPU-heavy).
- Headline PE tables in the papers come from multi-account training; seed-level CSVs are in `results/`.
- `src/data/synthetic_generator.py` and `storm_augmentor.py` are **not** part of the paper pipeline.
- Transformer baseline uses default hyperparameters (`d_model=64`, 3 layers) and is **not** capacity-matched to STORM (`d_model=128`, 2 layers), as stated in the manuscripts.

---

## Key Results (Summary)

| System | PE<sub>45min</sub> | PE<sub>6h</sub> | PE<sub>12h</sub> | PE<sub>st,6h</sub> |
|-------------------------|--------------------|-----------------|------------------|--------------------|
| Transformer | 0.977 | 0.904 | 0.859 | 0.821 |
| **Transformer matched (mean)** | **0.981** | **0.907** | **0.861** | — |
| **Transformer matched (bagged)** | **0.986** | **0.919** | **0.877** | — |
| STORM-Bz | **0.986** | 0.897 | 0.851 | 0.827 |
| Ensemble (α*=0.3) | 0.983 | **0.911** | **0.870** | 0.839 |
| STORM bagged (15 seeds) | **0.988** | 0.911 | 0.870 | **0.849** |

- Short-horizon gain is statistically significant (paired *p* = 0.002).
- Ablations show that the gain comes primarily from the overall training protocol rather than any single physics module at inference.
- Fine-tuning on GRASP raises 6 h PE from 0.449 → 0.599 and 12 h PE from 0.182 → 0.517.
- Capacity-matched Transformer control ($d_{model}=128$, 2 layers) reaches mean PE (45 min) = 0.981 and PE (6 h) = 0.907; bagging yields 0.986 / 0.919 / 0.877.

---

## Extra Experiments (Access Paper)

The IEEE Access version includes additional controlled experiments:

1. **Capacity-Matched Transformer Control**  
   A Vanilla Transformer with the same encoder capacity as STORM
   ($d_{\mathrm{model}}=128$, two layers, four heads;
   $\approx1.19\times10^{6}$ parameters) was trained under the same
   chronological split and fifteen seeds (seeds 42–56).
   - Mean: PE<sub>45min</sub>=0.981, PE<sub>6h</sub>=0.907, PE<sub>12h</sub>=0.861
   - Bagged (15 seeds): PE<sub>45min</sub>=0.986, PE<sub>6h</sub>=0.919, PE<sub>12h</sub>=0.877
   - Matching encoder capacity shrinks the short-horizon gap versus STORM-Bz
     and improves multi-horizon PE under bagging.

2. **Wider Delay Bound Ablation**  
   STORM was retrained with delay upper bounds of 2.0, 2.5, 3.0, 3.5, and 4.0 h (fifteen seeds each).  
   Mean PE<sub>45min</sub> stayed in the narrow range 0.9859–0.9862 and PE<sub>6h</sub> stayed in 0.900–0.902.  
   The original [0.5, 1.5] h constraint is therefore not a performance bottleneck.

Result summaries are in:

```text
results/transformer_matched_summary.json
results/transformer_matched_bagged_pe.json
results/transformer_matched_seed_pe.csv
results/wider_delay_results.csv
```

These experiments can also be reproduced from the master notebook ([Open in Colab](https://colab.research.google.com/github/bnsama29-cloud/STORM-PhysNet/blob/main/notebooks/STORM_PhysNet_Colab.ipynb)).

---

## Repository Structure

```text
STORM-PhysNet/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml                 # Model, data, training hyperparameters
│                                   # forecast_horizons: [0.75, 6.0, 12.0]
├── src/
│   ├── data/
│   │   ├── cdf_reader.py           # GOES CDF + OMNI readers (paper pipeline)
│   │   ├── preprocessor.py         # Feature build, chronological splits
│   │   └── dataloader.py           # Windows, horizons [0.75, 6, 12], storm sampler
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
├── notebooks/
│   └── STORM_PhysNet_Colab.ipynb   # Master reproduction notebook
├── figures/                         # ALL paper figures
├── results/
│   ├── summary.json
│   ├── ablation_final_table.csv
│   ├── all_results.csv
│   ├── wider_delay_results.csv
│   ├── bagged_tf_results.csv
│   ├── transformer_matched_summary.json
│   ├── transformer_matched_bagged_pe.json
│   ├── transformer_matched_seed_pe.csv
│   ├── grasp_metrics_summary.csv    # STORM / main-paper GRASP metrics
│   ├── matched_tf_grasp.csv         # Capacity-matched TF GRASP zero-shot / fine-tune (seeds 42–56)
│   ├── matched_tf_noise.csv         # Capacity-matched TF noise robustness
│   ├── matched_tf_pers.csv          # Capacity-matched TF PE_pers at 45 min
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

---

## Quick Start (Google Colab – Recommended)

1. Open [`notebooks/STORM_PhysNet_Colab.ipynb`](https://colab.research.google.com/github/bnsama29-cloud/STORM-PhysNet/blob/main/notebooks/STORM_PhysNet_Colab.ipynb) in Google Colab.
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
  - **Transformer matched** (d_model=128, 2 layers, 4 heads; ~1.19e6 params) — capacity-matched control, fifteen seeds
  - LSTM
- **Ablations**: No-Delay, No-Gate, No-Physics, horizon-restricted physics loss
- Test PE is computed **once** after training and never used for model selection

---

## Reproduction notes

- The Colab notebook is a **scaffold**: set `DEMO_MODE = False` for full 15-seed runs (GPU-heavy).
- Headline PE tables in the papers come from multi-seed training; seed-level CSVs are in `results/`.
- `src/data/synthetic_generator.py` and `storm_augmentor.py` are **not** part of the paper pipeline.
- Transformer baseline uses default hyperparameters (`d_model=64`, 3 layers) and is **not** capacity-matched to STORM (`d_model=128`, 2 layers), as stated in the manuscripts.
- A capacity-matched Transformer control (`d_model=128`, 2 layers, 4 heads) was trained for fifteen seeds; checkpoints are under `checkpoints/transformer_matched/seed_{42..56}/` and results under `results/transformer_matched_*`.
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
```
