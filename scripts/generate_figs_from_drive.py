import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Use the data directly from the drive directory
DRIVE_TABLES = r"f:\Downloads\ieee_final_fixed\drive\storm_physnet\nb_final_ieee_eval\Tables"
OUT_DIR = r"f:\Downloads\ieee_final_fixed\figures"

def generate_horizon_pe_fig():
    """Reads benchmark_metrics.csv or ieee_main_table.csv to plot fig_horizon_pe.png"""
    path = os.path.join(DRIVE_TABLES, "ieee_main_table.csv")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
        
    df = pd.read_csv(path)
    
    # Simple bar chart of PE for Horizons
    fig, ax = plt.subplots(figsize=(8, 5))
    horizons = ["1 h", "6 h", "12 h"]
    
    # We will grab transformer and storm_bz if they exist
    models = ["transformer", "storm_bz"]
    x = np.arange(len(horizons))
    width = 0.35
    
    for i, model in enumerate(models):
        subset = df[df.iloc[:, 0].astype(str).str.contains(model, case=False)]
        if not subset.empty:
            # Assumes columns exist for PE_1h, PE_6h, PE_12h (or 45min)
            # Find the columns containing mean values
            cols = subset.columns
            pe_1h_col = next((c for c in cols if '1h' in c or '45min' in c), None)
            pe_6h_col = next((c for c in cols if '6h' in c), None)
            pe_12h_col = next((c for c in cols if '12h' in c), None)
            
            if pe_1h_col and pe_6h_col and pe_12h_col:
                y = [subset[pe_1h_col].values[0], subset[pe_6h_col].values[0], subset[pe_12h_col].values[0]]
                ax.bar(x + (i - 0.5) * width, y, width, label=model)
                
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylabel("Prediction Efficiency")
    ax.set_title("PE across Forecast Horizons")
    ax.legend()
    
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUT_DIR, "fig_horizon_pe.png"))
    plt.close()
    print("Generated fig_horizon_pe.png")

def generate_feature_importance():
    path = os.path.join(DRIVE_TABLES, "permutation_importance.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if df.shape[1] > 1:
        # Assuming col 0 is feature name, col 1 is importance
        df = df.sort_values(by=df.columns[1], ascending=True)
        ax.barh(df.iloc[:, 0], df.iloc[:, 1], color='coral')
        ax.set_xlabel("Permutation Importance")
        ax.set_title("Feature Importance (6 h Forecast)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "fig_feature_importance.png"))
        plt.close()
        print("Generated fig_feature_importance.png")

def main():
    generate_horizon_pe_fig()
    generate_feature_importance()
    print(f"Figures saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
