import pandas as pd
import numpy as np
import glob

res_dir = "f:/Downloads/ieee_final_fixed/ieee_paper/results"

def summarize(prefix):
    files = glob.glob(f"{res_dir}/{prefix}_*.csv")
    if not files:
        print(f"No files for {prefix}")
        return None
    df = pd.concat([pd.read_csv(f) for f in files])
    print(f"\n--- {prefix.upper()} ---")
    
    if prefix == "noise":
        aggs = df.groupby("noise").agg(["mean", "std"])
        for std in aggs.index:
            m45 = aggs.loc[std, ("PE_45min", "mean")]
            s45 = aggs.loc[std, ("PE_45min", "std")]
            m6 = aggs.loc[std, ("PE_6h", "mean")]
            s6 = aggs.loc[std, ("PE_6h", "std")]
            print(f"Noise {std:.1f}: 45m = {m45:.3f} ± {s45:.3f} | 6h = {m6:.3f} ± {s6:.3f}")
            
    elif prefix == "grasp":
        m = df.mean()
        s = df.std()
        print(f"Zero-shot 6h: {m['zs_6h']:.3f} ± {s['zs_6h']:.3f}")
        print(f"Fine-tuned 6h: {m['ft_6h']:.3f} ± {s['ft_6h']:.3f}")
        
    elif prefix == "pers":
        m = df.mean()
        s = df.std()
        print(f"PE_pers 45m: {m['PE_pers_45min']:.3f} ± {s['PE_pers_45min']:.3f}")

summarize("pers")
summarize("noise")
summarize("grasp")
