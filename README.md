# 🚀 STORM-PhysNet
**Physics-Informed Multi-Horizon Forecasting of Geostationary Relativistic Electron Flux**

STORM-PhysNet is a domain-aware deep learning framework for forecasting high-energy electron flux ($E > 2$ MeV) at Geostationary Earth Orbit (GEO). It couples a standard Transformer encoder with physics-informed modules to accurately model flux enhancements driven by geomagnetic storms, avoiding the catastrophic instabilities seen in standard black-box models at short operational horizons.

---

## 🛰️ Architecture
The model processes 72 hours of upstream solar wind and geomagnetic indices to simultaneously predict electron flux at **45-min, 6-h, and 12-h horizons**.

```text
Solar Wind (OMNI/GOES, 16 features, 72 h window)
        │
        ▼
┌─────────────────────────────┐
│  Adaptive Propagation Delay │  ← Learns L1-to-Earth transit time (~1.09 h)
└──────────────┬──────────────┘
               │
      ┌────────▼─────────┐
      │ Temporal Encoder │  ← Standard Transformer self-attention backbone
      └────────┬─────────┘
               │
       ┌───────▼────────┐
       │ Bz Physics Gate│  ← Selectively amplifies features during southward IMF
       └───────┬────────┘
               │
    ┌──────────▼───────────┐
    │ Multi-Horizon Heads  │  ← Shared latent space for 45-m, 6-h, 12-h output
    └──────────────────────┘
```

---

## 📊 Interpretations & Results
All figures and tables generated during the rigorous multi-seed evaluation process on the 5-year GOES dataset can be found in the `interpretations/` directory.

### Performance Summary (6-Hour Horizon)
| Model Architecture | Seeds | PE (All) | PE (Storm) | PE (High Flux) | RMSE (All) |
|-------------------|:---:|:---:|:---:|:---:|:---:|
| **STORM-Bz (Ours)** | 3 | **0.669** ±0.030 | **0.674** ±0.017 | **0.745** | **0.251** |
| STORM-NoDelay (Ablation) | 3 | 0.672 ±0.022 | 0.643 ±0.015 | 0.717 | 0.250 |
| STORM-NoPhysics (Ablation) | 3 | 0.677 ±0.019 | 0.651 ±0.029 | 0.719 | 0.248 |
| Transformer (Baseline) | 3 | 0.648 ±0.031 | 0.612 ±0.042 | 0.717 | 0.259 |
| LSTM (Baseline) | 1 | 0.614 | 0.634 | 0.792 | 0.271 |
| MLP (Baseline) | 1 | 0.525 | 0.541 | 0.666 | 0.301 |
| CNN (Baseline) | 1 | 0.033 | 0.164 | 0.187 | 0.429 |

### 1. Multi-Horizon Reliability
At the critical 45-minute horizon, standard Transformers are highly unstable across random seeds. STORM-PhysNet remains strictly reliable and outperforms baselines across all horizons.

![Multi-Horizon PE](interpretations/Figures/fig_horizon_pe.png)

### 2. Ablation & Physics Constraints
Removing the Adaptive Delay costs 3.0 storm PE points; removing the Physics Gate costs 2.3 storm PE points.

![Ablation Results](interpretations/Figures/fig_ablation_6h.png)

The physics-informed networks successfully learn physical phenomena without direct supervision. The propagation delay network learns L1-to-Earth transit times centered precisely around ~1.0 hours:

![Propagation Delay Histogram](interpretations/Figures/fig_physics_tau_hist.png)

The physics gate strictly activates during geomagnetic storm periods (Bz < 0 conditions), acting as an attention amplifier exactly when it matters most:

![Gate Activation](interpretations/Figures/fig_physics_gate_storm_quiet.png)

### 3. Feature Importance & Event Case Studies
A massive permutation importance analysis (over 7,800 random feature shuffles) quantitatively confirms the model fundamentally relies on key solar wind drivers (like Bz and Flow Speed) over autoregressive persistence.

![Permutation Feature Importance](interpretations/Figures/fig_feature_importance.png)

During intense geomagnetic activity, the model tracks rapid flux enhancements successfully while maintaining tight bounds during quiet periods.

![Event Case Studies](interpretations/Figures/fig_case_studies.png)

### 4. Cross-Satellite Transfer (GRASP)
Tested on 14 months of novel Indian **GSAT-19 (GRASP)** data. A frozen-encoder strategy tuning only ~73K parameters recovers robust skill ($PE_{6h} = 0.564$), demonstrating high practical value for newly commissioned space weather missions with scarce data.

![GRASP Transfer Learning](interpretations/Figures/fig_grasp_transfer.png)

### 5. Uncertainty & Residual Diagnostics
Monte-Carlo dropout provides highly reliable 95% uncertainty bounds during varying solar wind conditions.

![MC Dropout Uncertainty](interpretations/Figures/fig_mc_dropout_band.png)

Residual diagnostics demonstrate that STORM-PhysNet produces a tighter, more zero-centered and Gaussian error distribution compared to the standard Transformer baseline.

![Residual Diagnostics](interpretations/Figures/fig_residual_storm_bz.png)

---

## 📁 Repository Structure
* `src/model/` - PyTorch model architecture (Transformer backbone, Delay module, Physics gate).
* `src/data/` - Data loading and preprocessing pipelines.
* `src/training/` - Model training loops and multi-seed logic.
* `src/evaluation/` - Metrics calculation (PE, RMSE) and evaluation scripts.
* `interpretations/` - Complete IEEE final output metrics, JSON statistics, and PNG figures.
* `colab_full_train.py` - The unified master pipeline script.
* `dashboard/` - Interactive Streamlit web app for live forecasting.

*(Note: Latex paper drafts, compiled PDFs, and raw dataset files are excluded from this repository).*

---

## 🚀 Quick Start
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Google Colab Unified Pipeline
To effortlessly train all configurations, extract checkpoints, compute metrics, and generate the full suite of interpretation figures on a T4 GPU, use the master Colab pipeline:
```bash
python colab_full_train.py
```
*Ensure your `datasets.zip` (GOES + OMNI + GRASP) is available as specified in the script.*

### 3. Run the Interactive Dashboard
Launch the Streamlit app for real-time visualization of the trained models:
```bash
streamlit run dashboard/app.py
```

---
**Author:** Samarth BN (RV College of Engineering)  
**Acknowledgment:** The authors thank the providers of GOES, OMNI, and GRASP data products.
