import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Regenerate domain_gap_plot.png with correct x-axis labels
# Using the GRASP transfer results from the paper

# Data from the paper's Table 5 (GRASP transfer results)
horizons = ["45 min", "6 h", "12 h"]
zero = [0.820, 0.449, 0.182]  # Zero-shot PE
ft = [0.827, 0.599, 0.517]    # Fine-tuned PE

x = np.arange(len(horizons))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(x - width/2, zero, width, label='Zero-Shot (GOES Model)', color='#d62728', edgecolor='black')
ax.bar(x + width/2, ft, width, label='Fine-Tuned (GRASP Model)', color='#2ca02c', edgecolor='black')

ax.set_ylabel('Prediction Efficiency (PE)', fontsize=12)
ax.set_title('Cross-Satellite Transfer Learning (GOES → GRASP)', fontsize=14, fontweight='bold')
ax.set_xticks(np.arange(len(horizons)))
ax.set_xticklabels(horizons, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.0])

for i, (z, f) in enumerate(zip(zero, ft)):
    ax.text(i - width/2, z + 0.02, f"{z:.2f}", ha='center', fontweight='bold', fontsize=10)
    ax.text(i + width/2, f + 0.02, f"{f:.2f}", ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(r'F:\Downloads\ieee_final_fixed\ieee_paper\claude\figures\domain_gap_plot.png', dpi=300)
print("Regenerated domain_gap_plot.png with correct x-axis labels")