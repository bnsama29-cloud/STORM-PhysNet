import re
import pandas as pd
import json

tex_path = "f:/Downloads/ieee_final_fixed/ieee_paper/claude/storm_physnet_conference_claude.tex"
with open(tex_path, "r", encoding="utf-8") as f:
    tex_content = f.read()

def extract_tex_numbers(pattern):
    return re.findall(pattern, tex_content)

print("--- CONSISTENCY AUDIT ---")

# 1. Check Capacity-Matched TF Means
matched_tf_45 = extract_tex_numbers(r"PE_\{45\\mathrm\{min\}\}=([0-9\.]+)")
matched_tf_6h = extract_tex_numbers(r"PE_\{6\\mathrm\{h\}\}=([0-9\.]+)")
print(f"TeX Matched TF: 45min={matched_tf_45}, 6h={matched_tf_6h}")

try:
    df_matched = pd.read_csv("f:/Downloads/ieee_final_fixed/results/transformer_matched_seed_pe.csv")
    mean_matched = df_matched.mean()
    print(f"CSV Matched TF Means: 45min={mean_matched['PE_45min']:.3f}, 6h={mean_matched['PE_6h']:.3f}")
except Exception as e:
    print(f"CSV Matched TF Error: {e}")

# 2. Check Bagged Matched TF
bagged_tf = extract_tex_numbers(r"bagging yields ([0-9\.]+) / ([0-9\.]+) / ([0-9\.]+)")
print(f"TeX Bagged TF: {bagged_tf}")
try:
    with open("f:/Downloads/ieee_final_fixed/results/transformer_matched_bagged_pe.json", "r") as f:
        bagged_json = json.load(f)
    print(f"JSON Bagged TF: {bagged_json['PE_45min']:.3f} / {bagged_json['PE_6h']:.3f} / {bagged_json['PE_12h']:.3f}")
except Exception as e:
    print(f"JSON Bagged TF Error: {e}")

# 3. Check GRASP
grasp_storm = extract_tex_numbers(r"raises 6\\,h PE from ([0-9\.]+) to ([0-9\.]+)")
grasp_storm_12h = extract_tex_numbers(r"12\\,h PE from ([0-9\.]+) to ([0-9\.]+)")
print(f"TeX GRASP STORM-Bz 6h: {grasp_storm}, 12h: {grasp_storm_12h}")

grasp_matched = extract_tex_numbers(r"6\\,h PE from \\\(([0-9\.]+) \\pm [0-9\.]+\\\) \(zero-shot\) to \\\(([0-9\.]+) \\pm [0-9\.]+\\\)")
print(f"TeX GRASP Matched TF 6h: {grasp_matched}")

try:
    df_g = pd.concat([
        pd.read_csv("f:/Downloads/ieee_final_fixed/ieee_paper/results/grasp_0.csv"),
        pd.read_csv("f:/Downloads/ieee_final_fixed/ieee_paper/results/grasp_1.csv")
    ])
    mg = df_g.mean()
    print(f"CSV GRASP Matched TF 6h: ZS={mg['zs_6h']:.3f}, FT={mg['ft_6h']:.3f}")
except Exception as e:
    print(f"CSV GRASP Matched TF Error: {e}")

# 4. Check PE_pers and Noise
pe_pers = extract_tex_numbers(r"PE\\(_\{\\mathrm\{pers\}\}\\)[\) ]*of \\\(([0-9\.\-]+) \\pm")
noise_pe = extract_tex_numbers(r"\\(\\(.*\\sigma=1\.0.*\\)\\).*drops to \\\(([0-9\.]+) \\pm")
print(f"TeX PE_pers: {pe_pers}")
print(f"TeX Noise (6h, sigma=1.0): {noise_pe}")

# 5. Notebook Check
nb_path = "f:/Downloads/ieee_final_fixed/notebooks/STORM_PhysNet_Colab.ipynb"
try:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_str = f.read()
    print("Notebook uses synthetic_generator?", "synthetic_generator" in nb_str)
    print("Notebook sets DEMO_MODE = True?", "DEMO_MODE = True" in nb_str)
    print("Notebook contains Supplementary cell?", "Supplementary Experiments" in nb_str)
except Exception as e:
    print(f"Notebook error: {e}")

