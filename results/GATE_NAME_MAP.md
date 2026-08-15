# Gate name map (paper ↔ code)

| Paper | Short | `gate_type` / run id | Notes |
|-------|--------|----------------------|--------|
| Physics Bz gate | STORM-Bz | `bz` | Main reported model |
| Rectified Driver Gate | RDG | `cathode_anode` / `storm_cathode` | Analogy gate |
| RDG + Spectral Head | RDG-S | `storm_cathode_spec` | + spectral head |
| Saturating Dose Gate | SDG | `radiotrophic` / `storm_radiotrophic` | Analogy gate |

Do not rename checkpoint folders; papers cite the short names only.
