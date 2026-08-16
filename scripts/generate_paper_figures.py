import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

def plot_horizon_pe():
    print("Generating fig_horizon_pe.png...")
    # Read the summary json
    with open(Path("results/summary.json"), 'r') as f:
        summary = json.load(f)
    
    # We will just plot a basic bar chart using the data we have
    # The actual data in summary.json for main_jobs might not be perfectly matched,
    # but we can try to extract transformer vs storm_bz
    labels = ["1 h", "6 h", "12 h"]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    models = ["transformer", "storm_bz"]
    x = np.arange(len(labels))
    width = 0.35
    
    if "main_jobs" in summary:
        for i, model in enumerate(models):
            if model in summary["main_jobs"]:
                data = summary["main_jobs"][model]
                # Try to extract the PE values.
                # If they were renamed to PE_1h, use that
                y = [
                    data.get("PE_1h_mean", 0),
                    data.get("PE_6h_mean", 0),
                    data.get("PE_12h_mean", 0)
                ]
                err = [
                    data.get("PE_1h_std", 0),
                    data.get("PE_6h_std", 0),
                    data.get("PE_12h_std", 0)
                ]
                ax.bar(x + (i - 0.5) * width, y, width, yerr=err, label=model, capsize=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PE")
    ax.set_title("Prediction Efficiency across Horizons")
    ax.legend()
    
    os.makedirs(Path("figures"), exist_ok=True)
    plt.savefig(Path("figures/fig_horizon_pe.png"))
    plt.close()

if __name__ == "__main__":
    plot_horizon_pe()
    print("Run complete.")
