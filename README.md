# 🚀 STORM-PhysNet
**Physics-Informed Multi-Horizon Forecasting of Geostationary Relativistic Electron Flux**

STORM-PhysNet is a domain-aware deep learning framework for forecasting high-energy electron flux ($E > 2$ MeV) at Geostationary Earth Orbit (GEO). It couples a standard Transformer encoder with physics-informed modules to accurately model flux enhancements driven by geomagnetic storms, avoiding the catastrophic instabilities seen in standard black-box models at short operational horizons.

---

## 🛰️ Architecture
The model processes 72 hours of upstream solar wind and geomagnetic indices to simultaneously predict electron flux at **45-min, 6-h, and 12-h horizons**.

```
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

## 📊 Verified Results & Achievements
Evaluated across 5 years of GOES data using chronological splits and rigorous 3-seed reproducibility.

1. **Short-Horizon Reliability:** At the critical 45-minute horizon, standard Transformers are unstable across seeds ($PE$ range: $-0.456$ to $+0.002$). STORM-PhysNet remains highly stable at **PE = $0.34 \pm 0.05$**.
2. **Storm-Time Accuracy:** At 6 hours, STORM-BzGate achieves a storm-time PE of **0.674** (vs 0.612 for Vanilla Transformer).
3. **Ablations:** Removing the Adaptive Delay costs **3.0** storm PE points; removing the Physics Gate costs **2.3** storm PE points.
4. **Cross-Satellite Transfer:** Tested on 14 months of novel Indian **GSAT-19 (GRASP)** data. A frozen-encoder strategy tuning only ~73K parameters recovers robust skill ($PE_{6h} = 0.564$), demonstrating high practical value for newly commissioned missions.

---

## 📁 Repository Structure
* `src/model/` - PyTorch model architecture (Transformer backbone, Delay module, Physics gate).
* `src/data/` - Data loading and preprocessing pipelines.
* `src/training/` - Model training loops and multi-seed logic.
* `src/evaluation/` - Metrics calculation (PE, RMSE) and evaluation scripts.
* `dashboard/` - Interactive Streamlit web app for live forecasting.
* `notebooks/` - Training and evaluation entry points (`01_train_main.py`, `02_train_optional.py`, `03_eval_all.py`).

*(Note: Latex paper drafts, compiled PDFs, and raw dataset files are excluded from this repository).*

---

## 🚀 Quick Start
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Download the raw GOES and OMNI datasets and place them in the `datasets/` directory. The pipeline expects hourly mean/std resampled CDF files.

### 3. Train Models
Run the multi-seed training pipeline:
```bash
python notebooks/01_train_main.py
```

### 4. Evaluate & Plot
Generate metrics and figures from the paper:
```bash
python notebooks/03_eval_all.py
```

### 5. Run the Interactive Dashboard
Launch the Streamlit app for real-time visualization:
```bash
streamlit run dashboard/app.py
```

---
**Author:** Samarth BN (RV College of Engineering)  
**Acknowledgment:** The authors thank the providers of GOES, OMNI, and GRASP data products.
