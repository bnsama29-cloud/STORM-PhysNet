import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

raw_csv = Path(r"f:\Downloads\ieee_final_fixed\drive\storm_physnet\week3_robustness\tables\noise_robustness_raw.csv")
df = pd.read_csv(raw_csv)

new_rows = []

for (seed, noise), group in df.groupby(['seed', 'noise']):
    tf = group[group['system'] == 'transformer'].iloc[0]
    st = group[group['system'] == 'storm_bz'].iloc[0]
    ens4 = group[group['system'] == 'ensemble_0.4']
    if ens4.empty:
        continue
    ens4 = ens4.iloc[0]
    
    r_ens3 = {"seed": seed, "noise": noise, "system": "ensemble_0.3"}
    
    for metric in ['PE_45min', 'PE_6h', 'PE_12h', 'PE_storm_45min', 'PE_storm_6h', 'PE_storm_12h']:
        E_tf = 1 - tf[metric]
        E_st = 1 - st[metric]
        E_ens4 = 1 - ens4[metric]
        
        # E_ens4 = 0.16 * E_st + 0.36 * E_tf + 0.48 * C
        C = (E_ens4 - 0.16 * E_st - 0.36 * E_tf) / 0.48
        
        # E_ens3 = 0.09 * E_st + 0.49 * E_tf + 0.42 * C
        E_ens3 = 0.09 * E_st + 0.49 * E_tf + 0.42 * C
        r_ens3[metric] = 1 - E_ens3
        
    # Copy persistence metrics (they are the same)
    for metric in ['PE_pers_45min', 'PE_pers_6h', 'PE_pers_12h']:
        r_ens3[metric] = tf[metric]
        
    new_rows.append(r_ens3)

df_new = pd.DataFrame(new_rows)
df_combined = pd.concat([df, df_new], ignore_index=True)

# Generate Summary
g = df_combined.groupby(["system", "noise"]).mean().reset_index()

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

pub_labels = {"storm_bz": "STORM-Bz", "transformer": "Transformer", "hybrid": "Hybrid", "ensemble_0.3": r"Ensemble ($\alpha=0.3$)"}
for sys_name in ["storm_bz", "transformer", "hybrid", "ensemble_0.3"]:
    sub = g[g["system"] == sys_name]
    if sub.empty: continue
    
    axes[0].plot(sub["noise"], sub["PE_45min"], marker="o", label=pub_labels.get(sys_name, sys_name), linewidth=2)
    axes[1].plot(sub["noise"], sub["PE_6h"], marker="s", label=pub_labels.get(sys_name, sys_name), linewidth=2)
    
axes[0].set_title("Robustness: PE (45 min) vs Sensor Noise")
axes[1].set_title("Robustness: PE (6 h) vs Sensor Noise")

for ax in axes:
    ax.set_xlabel("Noise Std (Standardized)")
    ax.set_ylabel("PE")
    ax.legend()
    
plt.tight_layout()

out_path = Path(r"f:\Downloads\ieee_final_fixed\ieee_paper\claude\figures\noise_robustness.png")
fig.savefig(out_path, dpi=300)
print(f"Saved exact plot to {out_path}")
