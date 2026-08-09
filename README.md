# STORM-PhysNet

**Physics-Informed Multi-Horizon Forecasting of Geostationary Relativistic Electron Flux with Transfer to Indian Longitude**

This repository contains the official implementation of **STORM-PhysNet**, accompanying the papers:

- Conference version (IEEE)
- Extended version (IEEE Access)

STORM-PhysNet is a Transformer-based model for multi-horizon forecasting of >2 MeV electron flux at geostationary orbit (GEO). It combines:

- A standard temporal Transformer encoder
- A learnable L1\u2013Earth propagation delay module
- A \(B_z\)-conditioned physics gate
- Residual multi-horizon heads (45 min / 6 h / 12 h)

The model is evaluated under a rigorous multi-seed protocol and includes zero-shot + fine-tuned transfer experiments to GSAT-19 GRASP (Indian longitude).

---

## Key Results (Summary)

| System                  | PE<sub>45min</sub> | PE<sub>6h</sub> | PE<sub>12h</sub> | PE<sub>st,6h</sub> |
|-------------------------|--------------------|-----------------|------------------|--------------------|
| Transformer             | 0.977              | 0.904           | 0.859            | 0.821              |
| STORM-Bz                | **0.986**          | 0.897           | 0.851            | 0.827              |
| Ensemble (\u03b1*=0.3)       | 0.983              | **0.911**       | **0.870**        | 0.839              |
| STORM bagged (15 seeds) | **0.988**          | 0.911           | 0.870            | **0.849**          |

- Short-horizon gain is statistically significant (paired *p* = 0.002).
- Ablations show that the gain comes primarily from the overall training protocol rather than any single physics module at inference.
- Fine-tuning on GRASP raises 6 h PE from 0.449 \u2192 0.599 and 12 h PE from 0.182 \u2192 0.517.

---

## Repository Structure

```text
STORM-PhysNet/
\u251c\u2500\u2500 configs/
\u2502   \u2514\u2500\u2500 config.yaml
\u251c\u2500\u2500 datasets/
\u2502   \u251c\u2500\u2500 goes/
\u2502   \u251c\u2500\u2500 omni/
\u2502   \u2514\u2500\u2500 grasp/
\u251c\u2500\u2500 src/
\u2502   \u251c\u2500\u2500 data/
\u2502   \u251c\u2500\u2500 model/
\u2502   \u251c\u2500\u2500 training/
\u2502   \u2514\u2500\u2500 evaluation/
\u251c\u2500\u2500 notebooks/
\u2502   \u2514\u2500\u2500 STORM_PhysNet_Colab.ipynb     \u2190 Master reproduction notebook
\u251c\u2500\u2500 requirements.txt
\u2514\u2500\u2500 README.md
```

---

## Quick Start (Google Colab \u2013 Recommended)

1. Open [`notebooks/STORM_PhysNet_Colab.ipynb`](notebooks/STORM_PhysNet_Colab.ipynb) in Google Colab.
2. Set runtime to **T4 GPU**.
3. Run all cells.

The notebook supports two modes:

- `DEMO_MODE = True` \u2192 quick run (few epochs) for testing
- `DEMO_MODE = False` \u2192 full paper-level training

---

## Data

The papers use the following public datasets, **which are included directly in this repository** under the `datasets/` folder for immediate reproducibility:

| Dataset       | Source                                      | Notes                              | Location in Repo |
|---------------|---------------------------------------------|------------------------------------|------------------|
| GOES-15       | [NOAA NCEI](https://www.ngdc.noaa.gov/stp/satellite/goes/) | >2 MeV electron flux (2012\u20132016) | `datasets/goes/` |
| OMNI          | [NASA OMNIWeb](https://omniweb.gsfc.nasa.gov/) | Solar wind + IMF                 | `datasets/omni/` |
| GSAT-19 GRASP | [ISSDC](https://www.issdc.gov.in/)          | Indian-longitude GEO measurements | `datasets/grasp/` |

---

## Reproducibility Protocol (as used in the papers)

- **Split**: Purely chronological 70 / 15 / 15 % (no shuffling)
- **Seeds**: 15 independent random initializations (seeds 42\u201356)
- **Metrics**: PE<sub>clim</sub> (primary) and PE<sub>pers</sub>
- **Baselines**: Depth-matched Transformer + LSTM
- **Ablations**: No-Delay, No-Gate, No-Physics, horizon-restricted physics loss
- Test PE is computed **once** after training and never used for model selection

---

## Citation

If you use this code or the results, please cite:

```bibtex
@article{samarth2026storm,
  title   = {STORM-PhysNet: A Multi-Horizon Transformer for Geostationary Relativistic Electron Flux Forecasting with Interpretable Physics-Inspired Modules and Cross-Satellite Transfer},
  author  = {Samarth BN},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

(Also cite the conference version once it is published.)

---

## License

This code is released for academic and research use.  
Please contact the author for commercial use.

---

## Contact

**Samarth BN**  
RV College of Engineering, Bengaluru, India  
Email: samarthbn.ec25@rvce.edu.in
