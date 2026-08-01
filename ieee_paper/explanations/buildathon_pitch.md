# Techverse Solution Buildathon 2026 - Pitch Deck Content
**Project Title:** STORM-PhysNet: AI-Driven Space Weather Forecasting for Satellite Protection

---

## Slide 1: Problem Statement
**Title:** The "Killer Electron" Threat to Space Infrastructure
* **The Core Issue:** Space weather (solar flares, coronal mass ejections) unleashes high-energy (>2 MeV) "killer electrons" into Earth's geosynchronous (GEO) orbit.
* **The Impact:** These electrons penetrate spacecraft shielding, causing deep dielectric charging, short circuits, and multi-million-dollar satellite failures (e.g., Galaxy-15).
* **The Current Gap:** 
  * Traditional physics models are too slow for real-time alerts.
  * Standard Machine Learning models suffer from the "MSE Trap"—they are highly biased toward predicting safe, quiet weather (95% of the time) and completely miss the massive 5% storm spikes that actually destroy satellites.
  * When ISRO or NASA launches a new satellite, standard AI fails due to lack of historical data on that specific hardware.

---

## Slide 2: Proposed Solution
**Title:** STORM-PhysNet: A Physics-Informed Neural Network
* **What it is:** A Deep Learning system that predicts high-energy electron fluxes 6 to 12 hours in advance using upstream solar wind data from the L1 Lagrange point.
* **How it works:** Instead of a black-box AI, STORM-PhysNet forces the neural network to obey astrophysics.
* **The Breakthrough:** 
  * **Dynamic Time Warping:** We created a neural module that actively calculates the physical travel time of solar wind from L1 to Earth (based on variable plasma speeds of 300–1000 km/s) rather than assuming a static 1-hour delay.
  * **Physics-Informed Loss (PINN):** Custom loss functions penalize the AI if it disobeys laws of magnetic reconnection (e.g., predicting a storm when the IMF $B_z$ field is pointing North).

---

## Slide 3: Key Features & Live Results
**Title:** State-of-the-Art Accuracy & Domain Adaptation
* **Unprecedented Storm Accuracy:** Unlike standard models, STORM-PhysNet achieves **0.78 Prediction Efficiency (PE)** during the top 10% of extreme high-flux radiation events.
* **Cross-Satellite Transfer Learning (Few-Shot):**
  * We pre-trained the model on 5 years of American GOES satellite data.
  * Using **Frozen Domain Adaptation**, we transferred the physical knowledge to the **ISRO GSAT-12R (GRASP)** Indian satellite.
  * *Result:* Achieved **65.8% accuracy** on the new satellite using only a few months of its data (beating train-from-scratch models by 18%).
* **Live Dashboard:** We have built a fully functional Streamlit dashboard for real-time space weather monitoring and Bz physics-gate visualization.

---

## Slide 4: Tech Stack
**Title:** Built for High-Performance Deep Learning
* **Core Machine Learning:** PyTorch (for dynamic computational graphs and custom PINN loss functions), Scikit-Learn, Pandas, NumPy.
* **Model Architecture:** iTransformer (for inverted multivariate time-series embeddings), custom Adaptive Delay modules with differentiable `grid_sample` interpolation.
* **Frontend & Visualization:** Streamlit, Plotly (for interactive real-time solar wind monitoring and uncertainty bands).
* **Data Processing:** `cdflib` for NASA Common Data Format (CDF) extraction, handling decades of highly imbalanced space weather data.
* **Compute:** Trained on Google Colab Tesla T4 GPUs with Deep Ensembling for uncertainty calibration.

---

## Slide 5: Impact & Scalability
**Title:** Real-World Application and Viability
* **Operational Satellite Protection:** Provides a crucial 6-12 hour early warning system. Satellite operators can switch to "safe mode" or delay orbital maneuvers, saving hundreds of millions of dollars.
* **Cross-Mission Scalability:** Our Domain Adaptation pipeline proves that STORM-PhysNet can be instantly deployed to **any newly launched satellite** globally (ISRO, ESA, SpaceX), overcoming the "cold start" data problem in aerospace.
* **Innovation:** Merges the reliability of hard plasma physics with the speed of deep learning, creating a highly interpretable, trustworthy AI for mission-critical aerospace applications.

---

## Slide 6: Future Scope
**Title:** The Future of Space Weather Forecasting
* **Multi-Orbit Expansion:** Extending the transfer learning pipeline beyond Geosynchronous (GEO) orbit down to Low Earth Orbit (LEO) to protect the ISS, Starlink constellations, and astronauts from radiation hazards.
* **Real-Time Edge Deployment:** Compressing and deploying the model directly onto satellite edge-compute hardware to allow autonomous self-preservation maneuvers without waiting for ground-station commands.
* **Multi-Modal Inputs:** Integrating solar coronal imagery (Extreme Ultraviolet images) alongside raw solar wind telemetry to predict the storms before they even leave the Sun's surface.
