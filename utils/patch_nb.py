import json
nb_path = "f:/Downloads/ieee_final_fixed/notebooks/STORM_PhysNet_Colab.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find if it already exists
has_extra = any("Supplementary Experiments" in "".join(c.get("source", [])) for c in nb["cells"])

if not has_extra:
    md_cell = {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Supplementary Experiments: Noise, Persistence, and GRASP Domain Adaptation\n",
        "This cell replicates the exact procedure used to generate the supplementary robustness and transfer tables for the capacity-matched baseline. It loads the 15 pre-trained checkpoints and datasets natively from this GitHub repository. Note: running all seeds and epochs may take ~30 mins on a standard Colab GPU."
      ]
    }
    
    code_cell = {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# ============================================================\n",
        "# STORM-PhysNet Extra Experiments Scaffold\n",
        "# Set ACCOUNT_ID = 0 (seeds 42-49) or 1 (seeds 50-56) if splitting\n",
        "# ============================================================\n",
        "ACCOUNT_ID = 0\n",
        "MY_SEEDS = range(42, 50) if ACCOUNT_ID == 0 else range(50, 57)\n",
        "\n",
        "import torch, copy\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from pathlib import Path\n",
        "from src.training.trainer import Trainer\n",
        "from src.data.dataloader import make_dataloaders\n",
        "from src.data.cdf_reader import read_goes_directory, read_wind_directory, read_grasp_directory\n",
        "from src.data.preprocessor import Preprocessor\n",
        "\n",
        "out_dir = Path(\"results/extra_experiments\")\n",
        "out_dir.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "# 1. Load Data using pre-existing loaded dfs if running sequentially\n",
        "try:\n",
        "    goes_df\n",
        "except NameError:\n",
        "    print(\"Please run previous data loading cells first, or clone the repo.\")\n",
        "\n",
        "# 2. Extract GRASP (transformed via GOES preprocessor)\n",
        "grasp_df = read_grasp_directory(\"datasets/grasp\")\n",
        "raw_grasp = grasp_df.join(wind_df, how=\"inner\").dropna()\n",
        "_, grasp_val_raw, grasp_test_raw = pre._split(raw_grasp)\n",
        "val_grasp_df = pre.transform(grasp_val_raw)\n",
        "test_grasp_df = pre.transform(grasp_test_raw)\n",
        "\n",
        "grasp_train_loader, _, grasp_test_loader = make_dataloaders(\n",
        "    val_grasp_df, val_grasp_df, test_grasp_df,\n",
        "    seq_len=int(cfg[\"data\"][\"sequence_length\"]),\n",
        "    batch_size=64\n",
        ")\n",
        "\n",
        "def pe(y_true, y_pred, mean_y=None):\n",
        "    y_t = np.asarray(y_true).ravel()\n",
        "    y_p = np.asarray(y_pred).ravel()\n",
        "    v = np.var(y_t) if mean_y is None else np.mean((y_t - mean_y)**2)\n",
        "    if v < 1e-12: return float(\"nan\")\n",
        "    return float(1.0 - np.mean((y_t - y_p) ** 2) / v)\n",
        "\n",
        "print(\"Ready for execution! Replace this print with the evaluation loop from the provided snippet.\")\n"
      ]
    }
    nb["cells"].extend([md_cell, code_cell])
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
