import os
import re
import sys
import time
import zipfile
import shutil
from pathlib import Path

# Ensure local repo code is importable when running from a temp or notebook directory.
search_paths = [Path.cwd()]
if '__file__' in globals():
    search_paths.append(Path(__file__).resolve().parent)
if len(sys.argv) > 0:
    argv0 = Path(sys.argv[0])
    if argv0.exists():
        search_paths.append(argv0.resolve().parent)
search_paths += list(Path.cwd().parents)
search_paths += [
    Path('/content/storm_work'),
    Path('/content/drive/MyDrive/storm_physnet'),
    Path('/content/drive/MyDrive/storm_physnet/ieee_final_fixed'),
    Path.home() / 'storm_physnet',
    Path.home() / 'ieee_final_fixed',
]
repo_root = next((p for p in search_paths if (p / 'src').exists()), None)
if repo_root is None and '__file__' in globals():
    repo_root = next((p for p in Path(__file__).resolve().parent.parents if (p / 'src').exists()), None)
if repo_root is not None:
    sys.path.insert(0, str(repo_root))

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# `src` imports are deferred until repo code is available.
# They are imported inside `main()` after `load_config_and_data()` ensures `src/` is on sys.path.

# -------------------- USER SETTINGS --------------------
DRIVE_CODE_ZIP = "/content/drive/MyDrive/storm_physnet/ieee_final_fixed.zip"
DRIVE_DATA_ZIP = "/content/drive/MyDrive/storm_physnet/datasets.zip"
DRIVE_NB3_OUT  = "/content/drive/MyDrive/storm_physnet/nb3_outputs"
# -------------------------------------------------------

os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

HORIZON_NAMES = ['45min', '6h', '12h']
FLUX_THRESHOLD = 4.0


def find_code_root(root: Path) -> Path:
    if (root / 'src').exists() and (root / 'configs').exists():
        return root
    for child in root.iterdir():
        if child.is_dir() and (child / 'src').exists() and (child / 'configs').exists():
            return child
    raise FileNotFoundError(f'Could not find code root with src/ and configs/ under {root}')


def make_output_dirs(output_dir: Path):
    for sub in ['Figures', 'Tables', 'JSON', 'LaTeX']:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def build_model(label: str, n_sw: int, seq_len: int, config: dict, n_horizons: int = 3):
    if label == 'lstm':
        return StandardLSTM(n_sw_features=n_sw, seq_len=seq_len, n_horizons=n_horizons)
    if label == 'mlp':
        return StandardMLP(n_sw_features=n_sw, seq_len=seq_len, n_horizons=n_horizons)
    if label == 'cnn':
        return StandardCNN(n_sw_features=n_sw, seq_len=seq_len, n_horizons=n_horizons)
    if label == 'transformer':
        return VanillaTransformer(n_sw_features=n_sw, seq_len=seq_len, n_horizons=n_horizons)

    gate_type = 'bz'
    if 'cathode' in label:
        gate_type = 'cathode_anode'
    elif 'radio' in label:
        gate_type = 'radiotrophic'

    ablation = 'none'
    if 'no_delay' in label:
        ablation = 'no_delay'
    if 'no_physics' in label:
        ablation = 'no_physics'

    return STORMPhysNet(
        n_sw_features=n_sw,
        seq_len=seq_len,
        d_model=int(config['model']['d_model']),
        n_heads=int(config['model']['transformer']['n_heads']),
        n_transformer_layers=int(config['model']['transformer']['n_layers']),
        n_ssm_layers=int(config['model']['ssm']['n_layers']),
        d_state=int(config['model']['ssm']['d_state']),
        d_ff=int(config['model']['transformer']['d_ff']),
        hidden_dim=int(config['model']['heads']['hidden_dim']),
        n_horizons=n_horizons,
        dropout=float(config['model']['transformer']['dropout']),
        ablation=ablation,
        backbone='transformer',
        gate_type=gate_type,
        use_spectral_head=bool(config['model'].get('use_spectral_head', False)),
    )


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device):
    state = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if isinstance(state, dict) and any(str(k).startswith('member_0') for k in state.keys()):
        state = {k.replace('member_0.', ''): v for k, v in state.items() if str(k).startswith('member_0')}
    model.load_state_dict(state, strict=False)
    return model


def safe_corr(y_true, y_pred):
    if len(y_true) <= 1:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return float(0.0 if np.isnan(corr) else corr)


def prediction_efficiency_against_persistence(y_true, y_pred, y_persist):
    mse_pred = np.mean((y_true - y_pred) ** 2, axis=0)
    mse_base = np.mean((y_true - y_persist) ** 2, axis=0)
    return 1.0 - mse_pred / (mse_base + 1e-10)


def heidke_skill_score(y_true, y_pred, threshold=FLUX_THRESHOLD):
    obs_event = y_true >= threshold
    pred_event = y_pred >= threshold
    TP = np.sum(obs_event & pred_event)
    FP = np.sum(~obs_event & pred_event)
    FN = np.sum(obs_event & ~pred_event)
    TN = np.sum(~obs_event & ~pred_event)
    denom = ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) + 1e-10
    hss = float(2 * (TP * TN - FP * FN) / denom)
    pod = float(TP / max(TP + FN, 1))
    far = float(FP / max(FP + TP, 1))
    csi = float(TP / max(TP + FP + FN, 1))
    return hss, pod, far, csi


def compute_metrics_for_mask(y_true, y_pred, y_persist, mask, period):
    y_t = y_true[mask]
    y_p = y_pred[mask]
    y_b = y_persist[mask]
    if y_t.size == 0:
        return []

    rmse_vals = np.sqrt(np.mean((y_t - y_p) ** 2, axis=0))
    mae_vals = np.mean(np.abs(y_t - y_p), axis=0)
    bias_vals = np.mean(y_p - y_t, axis=0)
    corr_vals = np.array([safe_corr(y_t[:, i], y_p[:, i]) for i in range(y_p.shape[1])])
    pe_vals = prediction_efficiency_against_persistence(y_t, y_p, y_b)

    rows = []
    for h_idx, horizon in enumerate(HORIZON_NAMES):
        h_true = y_t[:, h_idx]
        h_pred = y_p[:, h_idx]
        hss, pod, far, csi = heidke_skill_score(h_true, h_pred)
        rows.append({
            'horizon': horizon,
            'period': period,
            'rmse': float(rmse_vals[h_idx]),
            'mae': float(mae_vals[h_idx]),
            'bias': float(bias_vals[h_idx]),
            'r2': float(r2_score(h_true, h_pred)) if len(h_true) > 1 else 0.0,
            'pe': float(pe_vals[h_idx]),
            'corr': float(corr_vals[h_idx]),
            'hss': hss,
            'pod': pod,
            'far': far,
            'csi': csi,
            'n_total': int(len(h_true)),
        })
    return rows


def eval_metrics(y_true, y_pred, y_persist, kp=None, dst=None):
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
        y_persist = y_persist[:, None]

    rows = []
    rows.extend(compute_metrics_for_mask(y_true, y_pred, y_persist, np.ones(len(y_true), dtype=bool), 'all'))
    if dst is not None:
        rows.extend(compute_metrics_for_mask(y_true, y_pred, y_persist, dst <= -50.0, 'storm (Dst<=-50)'))
        rows.extend(compute_metrics_for_mask(y_true, y_pred, y_persist, dst > -50.0, 'quiet (Dst>-50)'))
    if kp is not None:
        rows.extend(compute_metrics_for_mask(y_true, y_pred, y_persist, kp >= 5.0, 'storm (Kp>=5)'))
    return pd.DataFrame(rows)


def predict_model(model, loader, device, record_aux: bool = False, mc_dropout: bool = False, mc_passes: int = 10):
    model.to(device)
    if mc_dropout:
        model.train()
    else:
        model.eval()

    preds, trues, persists, dsts, kps, gates, taus = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x_sw = torch.nan_to_num(batch['x_sw'].to(device), nan=0.0)
            x_flux = torch.nan_to_num(batch['x_flux'].to(device), nan=0.0)
            y_persist = batch['y_persist'].to(device)
            if mc_dropout and mc_passes > 1:
                preds_mc = []
                for _ in range(mc_passes):
                    out = model(x_sw, x_flux, y_persist=y_persist)
                    preds_mc.append(out['flux_pred'].cpu().numpy() if isinstance(out, dict) else out.cpu().numpy())
                pred = np.mean(np.stack(preds_mc, axis=0), axis=0)
            else:
                out = model(x_sw, x_flux, y_persist=y_persist)
                pred = out['flux_pred'].cpu().numpy() if isinstance(out, dict) else out.cpu().numpy()

            preds.append(pred)
            trues.append(batch['y_flux'].numpy())
            persists.append(batch['y_persist'].numpy())
            if 'y_dst' in batch:
                dsts.append(batch['y_dst'].numpy().min(axis=1))
            if 'y_kp' in batch:
                kps.append(batch['y_kp'].numpy().max(axis=1))
            if record_aux and isinstance(out, dict):
                if 'gate_values' in out:
                    gates.append(out['gate_values'].cpu().numpy())
                if 'tau' in out:
                    taus.append(out['tau'].cpu().numpy())

    return (
        np.concatenate(trues, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(persists, axis=0),
        np.concatenate(dsts, axis=0) if dsts else None,
        np.concatenate(kps, axis=0) if kps else None,
        np.concatenate(gates, axis=0) if gates else None,
        np.concatenate(taus, axis=0) if taus else None,
    )


def predict_model_mc_dropout(model, loader, device, mc_passes: int = 10):
    model.to(device)
    model.train()
    all_preds, trues, persists, dsts, kps = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x_sw = torch.nan_to_num(batch['x_sw'].to(device), nan=0.0)
            x_flux = torch.nan_to_num(batch['x_flux'].to(device), nan=0.0)
            y_persist = batch['y_persist'].to(device)
            batch_preds = []
            for _ in range(mc_passes):
                out = model(x_sw, x_flux, y_persist=y_persist)
                batch_preds.append(out['flux_pred'].cpu().numpy() if isinstance(out, dict) else out.cpu().numpy())
            all_preds.append(np.stack(batch_preds, axis=0))
            trues.append(batch['y_flux'].numpy())
            persists.append(batch['y_persist'].numpy())
            if 'y_dst' in batch:
                dsts.append(batch['y_dst'].numpy().min(axis=1))
            if 'y_kp' in batch:
                kps.append(batch['y_kp'].numpy().max(axis=1))

    all_preds = np.concatenate(all_preds, axis=1)
    return (
        np.concatenate(trues, axis=0),
        np.mean(all_preds, axis=0),
        np.std(all_preds, axis=0),
        np.concatenate(persists, axis=0),
        np.concatenate(dsts, axis=0) if dsts else None,
        np.concatenate(kps, axis=0) if kps else None,
    )


def plot_residuals(horizon_index, y_true, y_pred, save_path):
    residuals = y_pred[:, horizon_index] - y_true[:, horizon_index]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_true[:, horizon_index], residuals, s=3, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('True flux')
    ax.set_ylabel('Residual (pred - true)')
    ax.set_title(f'Residual scatter for {HORIZON_NAMES[horizon_index]}')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_histogram(residuals, save_path, title=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=60, alpha=0.8)
    ax.set_title(title or 'Residual histogram')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Count')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def discover_checkpoints(root: Path):
    mapping = {}
    if not root.exists():
        return mapping
    for file_path in sorted(root.rglob('*.pt')) + sorted(root.rglob('*.zip')):
        if not file_path.is_file():
            continue
        stem = file_path.stem.replace('_best', '').replace('-best', '')
        label = stem.split('_seed')[0]
        seed = 0
        match = re.search(r'_seed[_]?(\d+)$', file_path.stem)
        if match:
            seed = int(match.group(1))
        mapping.setdefault(label, {})[seed] = file_path
    return mapping


def permutation_importance(model, loader, device, feature_count, horizon_index=1, n_repeats=2):
    y_true, y_pred, _, _, _, _, _ = predict_model(model, loader, device)
    baseline = np.sqrt(np.mean((y_true[:, horizon_index] - y_pred[:, horizon_index]) ** 2))
    rows = []
    for feat in range(feature_count):
        scores = []
        for _ in range(n_repeats):
            perturbed_preds = []
            for batch in loader:
                x_sw = batch['x_sw'].clone().numpy()
                np.random.shuffle(x_sw[:, :, feat])
                x_sw = torch.from_numpy(x_sw).to(device)
                x_flux = torch.nan_to_num(batch['x_flux'].to(device), nan=0.0)
                y_persist = batch['y_persist'].to(device)
                out = model(x_sw, x_flux, y_persist=y_persist)
                perturbed_preds.append(out['flux_pred'].cpu().numpy() if isinstance(out, dict) else out.cpu().numpy())
            perturbed_preds = np.concatenate(perturbed_preds, axis=0)
            scores.append(np.sqrt(np.mean((y_true[:, horizon_index] - perturbed_preds[:, horizon_index]) ** 2)))
        rows.append({
            'feature': feat,
            'rmse_mean': float(np.mean(scores)),
            'rmse_delta': float(np.mean(scores) - baseline),
        })
    return pd.DataFrame(rows).sort_values('rmse_delta', ascending=False)


def load_config_and_data(work_root: Path, drive_root: Path, code_work: Path):
    code_zip = drive_root / 'ieee_final_fixed.zip'
    data_zip = drive_root / 'datasets.zip'

    if not (code_work / 'src').exists() and code_zip.exists():
        if code_work.exists():
            shutil.rmtree(code_work)
        code_work.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(code_zip, 'r') as zf:
            zf.extractall(code_work)

    if (code_work / 'src').exists():
        repo_root = find_code_root(code_work)
    elif (Path.cwd() / 'src').exists():
        repo_root = find_code_root(Path.cwd())
    else:
        # fallback: search parent directories for repo root
        repo_root = next((p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / 'src').exists() and (p / 'configs').exists()), None)
        if repo_root is None:
            raise FileNotFoundError(f'Could not find repo root containing src/ and configs/')
    sys.path.insert(0, str(repo_root))

    data_root = work_root / 'datasets'
    if not data_root.exists() and data_zip.exists():
        temp_dir = work_root / 'data_tmp'
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(data_zip, 'r') as zf:
            zf.extractall(temp_dir)
        for name in ['goes', 'omni', 'grasp']:
            found = next((p for p in temp_dir.rglob(name) if p.is_dir()), None)
            if found:
                shutil.copytree(found, data_root / name, dirs_exist_ok=True)
        shutil.rmtree(temp_dir)

    config_path = repo_root / 'configs' / 'config.yaml'
    assert config_path.exists(), f'Config file missing: {config_path}'
    config = yaml.safe_load(open(config_path, 'r'))
    return repo_root, config, data_root


def main():
    on_colab = False
    drive = None
    try:
        from google.colab import drive as colab_drive
        drive = colab_drive
        on_colab = True
    except Exception:
        on_colab = False

    work_root = Path('/content/storm_work') if on_colab else Path.cwd()
    work_root.mkdir(parents=True, exist_ok=True)
    os.chdir(work_root)
    if on_colab:
        drive.mount('/content/drive', force_remount=False)

    # Install notebook dependencies early so CDF reading works in the current Python environment.
    os.system(f"{sys.executable} -m pip install -q cdflib 'numpy<2' pandas scikit-learn pyyaml tqdm matplotlib")

    drive_root = Path('/content/drive/MyDrive/storm_physnet') if on_colab else Path.cwd()
    code_work = work_root / 'code'

    repo_root, config, data_root = load_config_and_data(work_root, drive_root, code_work)

    # Now that repo_root is on sys.path, import local project code.
    from src.data.cdf_reader import read_goes_directory, read_wind_directory, read_grasp_directory
    from src.data.preprocessor import Preprocessor
    from src.data.dataloader import make_dataloaders
    from src.model.baselines import StandardLSTM, StandardMLP, StandardCNN, VanillaTransformer
    from src.model.storm_physnet import STORMPhysNet

    # Expose model classes to module scope so build_model() can access them.
    globals().update({
        'StandardLSTM': StandardLSTM,
        'StandardMLP': StandardMLP,
        'StandardCNN': StandardCNN,
        'VanillaTransformer': VanillaTransformer,
        'STORMPhysNet': STORMPhysNet,
    })

    goes_dir = data_root / 'goes'
    omni_dir = data_root / 'omni'
    grasp_dir = data_root / 'grasp'
    assert goes_dir.exists() and omni_dir.exists(), 'GOES and OMNI data must be available.'

    goes_df = read_goes_directory(str(goes_dir))
    wind_df = read_wind_directory(str(omni_dir))
    raw_df = goes_df.join(wind_df, how='inner')

    preproc = Preprocessor()
    train_df, val_df, test_df = preproc.fit_transform(raw_df)

    sequence_length = int(config['data']['sequence_length'])
    batch_size = int(config['data'].get('batch_size', 64))
    storm_weight = float(config['training'].get('storm_weight', 10.0))
    num_workers = int(config['data'].get('num_workers', 0))

    _, _, test_loader = make_dataloaders(
        train_df,
        val_df,
        test_df,
        seq_len=sequence_length,
        batch_size=batch_size,
        storm_weight=storm_weight,
        num_workers=num_workers,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_root = (Path.cwd() / 'checkpoints') if (Path.cwd() / 'checkpoints').exists() else drive_root / 'nb1_outputs' / 'checkpoints'
    checkpoint_map = discover_checkpoints(checkpoint_root)
    print('Found checkpoint labels:', checkpoint_map.keys())
    assert checkpoint_map, f'No checkpoints found under {checkpoint_root}'

    if on_colab:
        output_dir = Path(DRIVE_NB3_OUT) / '03_IEEE_Analysis_results'
    else:
        output_dir = work_root / '03_IEEE_Analysis_results'
    make_output_dirs(output_dir)

    all_metrics = []
    for label, seeds in checkpoint_map.items():
        for seed, ckpt_path in seeds.items():
            try:
                model = build_model(label, n_sw=test_loader.dataset.n_sw_features, seq_len=sequence_length, config=config)
                load_checkpoint(model, ckpt_path, device)
                y_true, y_pred, y_persist, y_dst, y_kp, _, _ = predict_model(model, test_loader, device)
                metrics_df = eval_metrics(y_true, y_pred, y_persist, kp=y_kp, dst=y_dst)
                metrics_df['label'] = label
                metrics_df['seed'] = seed
                metrics_df['checkpoint'] = ckpt_path.name
                all_metrics.append(metrics_df)
                print(f'Evaluated {label} seed={seed}')
            except Exception as exc:
                print(f'Failed {label} seed={seed}:', exc)

    if all_metrics:
        results_df = pd.concat(all_metrics, ignore_index=True)
        results_df.to_csv(output_dir / 'Tables' / 'benchmark_metrics.csv', index=False)
        (output_dir / 'JSON' / 'benchmark_metrics.json').write_text(results_df.to_json(orient='records', indent=2))
        print('Saved benchmark_metrics.csv')

    primary_labels = ['transformer', 'storm_physnet']
    for label in primary_labels:
        if label not in checkpoint_map:
            print('Skipping missing label:', label)
            continue
        values = []
        for seed, ckpt_path in checkpoint_map[label].items():
            model = build_model(label, n_sw=test_loader.dataset.n_sw_features, seq_len=sequence_length, config=config)
            load_checkpoint(model, ckpt_path, device)
            y_true, y_pred, y_persist, y_dst, y_kp, _, _ = predict_model(model, test_loader, device)
            metrics_df = eval_metrics(y_true, y_pred, y_persist, kp=y_kp, dst=y_dst)
            values.append([float(metrics_df.loc[(metrics_df['horizon'] == h) & (metrics_df['period'] == 'all'), 'pe'].iloc[0]) for h in HORIZON_NAMES])
        mean_pe = np.nanmean(np.array(values), axis=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(HORIZON_NAMES, mean_pe, marker='o', label=label)
        ax.set_xlabel('Horizon')
        ax.set_ylabel('Prediction efficiency')
        ax.set_title(f'Horizon analysis for {label}')
        ax.legend()
        fig_path = output_dir / 'Figures' / f'horizon_analysis_{label}.png'
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('Saved', fig_path)

    if grasp_dir.exists():
        grasp_df = read_grasp_directory(str(grasp_dir))
        if not grasp_df.empty:
            grasp_raw = grasp_df.join(wind_df, how='inner')
            if grasp_raw.empty:
                print('GRASP and OMNI data have no overlapping timestamps.')
            else:
                grasp_processed = preproc.transform(grasp_raw)
                n = len(grasp_processed)
                train_grasp = grasp_processed.iloc[: int(n * 0.8)]
                val_grasp = grasp_processed.iloc[int(n * 0.8) : int(n * 0.9)]
                test_grasp = grasp_processed.iloc[int(n * 0.9) :]
                _, _, grasp_loader = make_dataloaders(
                    train_grasp,
                    val_grasp,
                    test_grasp,
                    seq_len=sequence_length,
                    batch_size=batch_size,
                    storm_weight=1.0,
                    num_workers=0,
                )
                if 'storm_physnet' in checkpoint_map:
                    ckpt_path = checkpoint_map['storm_physnet'][min(checkpoint_map['storm_physnet'])]
                    goes_model = build_model('storm_physnet', n_sw=test_loader.dataset.n_sw_features, seq_len=sequence_length, config=config)
                    load_checkpoint(goes_model, ckpt_path, device)
                    y_true, y_pred, y_persist, y_dst, y_kp, _, _ = predict_model(goes_model, grasp_loader, device)
                    before_df = eval_metrics(y_true, y_pred, y_persist, kp=y_kp, dst=y_dst)
                    before_df['stage'] = 'GOES-only'
                    before_df.to_csv(output_dir / 'Tables' / 'grasp_zero_shot_metrics.csv', index=False)
                    print('Saved GRASP zero-shot metrics')
                    try:
                        from src.training.transfer_learning import GRASPTransferLearner
                        transfer = GRASPTransferLearner(config=config, device=device)
                        model_ft = transfer.fine_tune(goes_model, grasp_loader, grasp_loader, epochs=3, lr=1e-4)
                        torch.save(model_ft.state_dict(), output_dir / 'Tables' / 'storm_bz_grasp_finetuned.pt')
                        y_true, y_pred, y_persist, y_dst, y_kp, _, _ = predict_model(model_ft, grasp_loader, device)
                        after_df = eval_metrics(y_true, y_pred, y_persist, kp=y_kp, dst=y_dst)
                        after_df['stage'] = 'GRASP-finetuned'
                        pd.concat([before_df, after_df], ignore_index=True).to_csv(output_dir / 'Tables' / 'grasp_transfer_comparison.csv', index=False)
                        print('Saved GRASP transfer comparison metrics')
                    except Exception as exc:
                        print('Transfer learning module unavailable or failed:', exc)
                else:
                    print('No storm_bz checkpoint found for GRASP analysis.')
        else:
            print('GRASP directory exists but contains no data.')
    else:
        print('GRASP data not present; skipping transfer learning analysis.')

    ablation_labels = ['storm_physnet', 'storm_bz_no_delay', 'storm_bz_no_physics']
    ablation_results = []
    for label in ablation_labels:
        if label not in checkpoint_map:
            print('Missing ablation checkpoint for', label)
            continue
        ckpt_path = checkpoint_map[label][min(checkpoint_map[label])]
        model = build_model(label, n_sw=test_loader.dataset.n_sw_features, seq_len=sequence_length, config=config)
        load_checkpoint(model, ckpt_path, device)
        y_true, y_pred, y_persist, y_dst, y_kp, _, _ = predict_model(model, test_loader, device)
        df = eval_metrics(y_true, y_pred, y_persist, kp=y_kp, dst=y_dst)
        df['label'] = label
        ablation_results.append(df)
    if ablation_results:
        ablation_df = pd.concat(ablation_results, ignore_index=True)
        ablation_df.to_csv(output_dir / 'Tables' / 'ablation_metrics.csv', index=False)
        print('Saved ablation_metrics.csv')

    seed_summary = []
    for label in ['transformer', 'storm_physnet']:
        if label not in checkpoint_map:
            continue
        all_dfs = []
        for seed, ckpt_path in checkpoint_map[label].items():
            model = build_model(label, n_sw=test_loader.dataset.n_sw_features, seq_len=sequence_length, config=config)
            load_checkpoint(model, ckpt_path, device)
            y_true, y_pred, y_persist, y_dst, y_kp, _, _ = predict_model(model, test_loader, device)
            df = eval_metrics(y_true, y_pred, y_persist, kp=y_kp, dst=y_dst)
            all_dfs.append(df.assign(seed=seed))
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            summary = combined.groupby(['horizon', 'period'])[['pe', 'rmse']].agg(['mean', 'std']).reset_index()
            summary.columns = ['horizon', 'period', 'pe_mean', 'pe_std', 'rmse_mean', 'rmse_std']
            summary['label'] = label
            seed_summary.append(summary)
    if seed_summary:
        pd.concat(seed_summary, ignore_index=True).to_csv(output_dir / 'Tables' / 'seed_significance_summary.csv', index=False)
        print('Saved seed_significance_summary.csv')

    phys_labels = [l for l in ['storm_physnet', 'storm_bz_no_delay', 'storm_bz_no_physics'] if l in checkpoint_map]
    for label in phys_labels:
        ckpt_path = checkpoint_map[label][min(checkpoint_map[label])]
        model = build_model(label, n_sw=test_loader.dataset.n_sw_features, seq_len=sequence_length, config=config)
        load_checkpoint(model, ckpt_path, device)
        y_true, y_pred, y_persist, y_dst, y_kp, gates, taus = predict_model(model, test_loader, device, record_aux=True)
        if gates is not None:
            plt.figure(figsize=(8, 4))
            plt.plot(np.mean(gates, axis=0))
            plt.title(f'Average gate values for {label}')
            plt.xlabel('Gate index')
            plt.ylabel('Mean gate')
            path = output_dir / 'Figures' / f'{label}_gate_values.png'
            plt.savefig(path, dpi=200, bbox_inches='tight')
            plt.close()
            print('Saved gate analysis:', path)
        if taus is not None:
            plt.figure(figsize=(8, 4))
            plt.hist(taus.ravel(), bins=40)
            plt.title(f'Tau distribution for {label}')
            plt.xlabel('Tau')
            plt.ylabel('Count')
            path = output_dir / 'Figures' / f'{label}_tau_distribution.png'
            plt.savefig(path, dpi=200, bbox_inches='tight')
            plt.close()
            print('Saved tau distribution:', path)

    if 'storm_physnet' in checkpoint_map:
        model = build_model('storm_physnet', n_sw=test_loader.dataset.n_sw_features, seq_len=sequence_length, config=config)
        load_checkpoint(model, checkpoint_map['storm_physnet'][min(checkpoint_map['storm_physnet'])], device)
        imp_df = permutation_importance(model, test_loader, device, feature_count=test_loader.dataset.n_sw_features, horizon_index=1, n_repeats=2)
        imp_df.to_csv(output_dir / 'Tables' / 'storm_bz_permutation_importance.csv', index=False)
        print('Saved permutation importance')

    if 'storm_physnet' in checkpoint_map:
        model = build_model('storm_physnet', n_sw=test_loader.dataset.n_sw_features, seq_len=sequence_length, config=config)
        load_checkpoint(model, checkpoint_map['storm_physnet'][min(checkpoint_map['storm_physnet'])], device)
        y_true, y_pred, y_persist, y_dst, y_kp, _, _ = predict_model(model, test_loader, device)
        residuals = np.abs(y_pred[:, 1] - y_true[:, 1])
        top_idxs = np.argsort(residuals)[-5:][::-1]
        case_rows = []
        for idx in top_idxs:
            case_rows.append({
                'index': int(idx),
                'true_6h': float(y_true[idx, 1]),
                'pred_6h': float(y_pred[idx, 1]),
                'persist_6h': float(y_persist[idx, 1]),
                'abs_error': float(residuals[idx]),
                'dst': float(y_dst[idx]) if y_dst is not None else None,
                'kp': float(y_kp[idx]) if y_kp is not None else None,
            })
        pd.DataFrame(case_rows).to_csv(output_dir / 'Tables' / 'event_case_studies.csv', index=False)
        print('Saved event case studies')

    for label in ['storm_physnet', 'transformer']:
        if label not in checkpoint_map:
            continue
        model = build_model(label, n_sw=test_loader.dataset.n_sw_features, seq_len=sequence_length, config=config)
        load_checkpoint(model, checkpoint_map[label][min(checkpoint_map[label])], device)
        y_true, y_pred, y_persist, y_dst, y_kp, _, _ = predict_model(model, test_loader, device)
        residuals = y_pred[:, 1] - y_true[:, 1]
        plot_residuals(1, y_true, y_pred, output_dir / 'Figures' / f'{label}_residual_scatter_6h.png')
        plot_histogram(residuals, output_dir / 'Figures' / f'{label}_residual_hist_6h.png', title=f'{label} 6h residuals')
        print('Saved residual diagnostics for', label)

    if 'storm_physnet' in checkpoint_map:
        model = build_model('storm_physnet', n_sw=test_loader.dataset.n_sw_features, seq_len=sequence_length, config=config)
        load_checkpoint(model, checkpoint_map['storm_physnet'][min(checkpoint_map['storm_physnet'])], device)
        y_true, mean_pred, std_pred, _, _, _ = predict_model_mc_dropout(model, test_loader, device, mc_passes=10)
        unc_df = pd.DataFrame({'mean_pred_6h': mean_pred[:, 1], 'std_pred_6h': std_pred[:, 1]})
        unc_df.to_csv(output_dir / 'Tables' / 'storm_bz_mc_dropout_uncertainty.csv', index=False)
        print('Saved MC dropout uncertainty results')

    if 'storm_physnet' in checkpoint_map:
        model = build_model('storm_physnet', n_sw=test_loader.dataset.n_sw_features, seq_len=sequence_length, config=config)
        load_checkpoint(model, checkpoint_map['storm_physnet'][min(checkpoint_map['storm_physnet'])], device)
        model.to(device)
        model.eval()
        start = time.time()
        with torch.no_grad():
            for batch in test_loader:
                x_sw = torch.nan_to_num(batch['x_sw'].to(device), nan=0.0)
                x_flux = torch.nan_to_num(batch['x_flux'].to(device), nan=0.0)
                y_persist = batch['y_persist'].to(device)
                _ = model(x_sw, x_flux, y_persist=y_persist)
        elapsed = time.time() - start
        n_samples = len(test_loader.dataset)
        cost_df = pd.DataFrame([{'model': 'storm_physnet', 'n_samples': int(n_samples), 'elapsed_sec': float(elapsed), 'sec_per_sample': float(elapsed / max(n_samples, 1))}])
        cost_df.to_csv(output_dir / 'Tables' / 'compute_cost.csv', index=False)
        print('Saved compute cost')

    robust_rows = []
    for label in ['storm_physnet', 'transformer']:
        if label not in checkpoint_map:
            continue
        model = build_model(label, n_sw=test_loader.dataset.n_sw_features, seq_len=sequence_length, config=config)
        load_checkpoint(model, checkpoint_map[label][min(checkpoint_map[label])], device)
        y_true, y_pred, y_persist, y_dst, y_kp, _, _ = predict_model(model, test_loader, device)
        df = eval_metrics(y_true, y_pred, y_persist, kp=y_kp, dst=y_dst)
        for period in ['storm (Dst<=-50)', 'quiet (Dst>-50)']:
            row = df[df['period'] == period]
            if not row.empty:
                robust_rows.append({
                    'label': label,
                    'period': period,
                    'horizon': row['horizon'].tolist(),
                    'pe': row['pe'].tolist(),
                    'rmse': row['rmse'].tolist(),
                })
    if robust_rows:
        pd.DataFrame(robust_rows).to_csv(output_dir / 'Tables' / 'robustness_summary.csv', index=False)
        print('Saved robustness summary')

    discussion_rows = []
    if 'transformer' in checkpoint_map and 'storm_physnet' in checkpoint_map:
        for label in ['transformer', 'storm_physnet']:
            ckpt_path = checkpoint_map[label][min(checkpoint_map[label])]
            model = build_model(label, n_sw=test_loader.dataset.n_sw_features, seq_len=sequence_length, config=config)
            load_checkpoint(model, ckpt_path, device)
            y_true, y_pred, y_persist, y_dst, y_kp, _, _ = predict_model(model, test_loader, device)
            df = eval_metrics(y_true, y_pred, y_persist, kp=y_kp, dst=y_dst)
            all_pe = df[df['period'] == 'all']['pe'].mean()
            storm_pe = df[df['period'] == 'storm (Dst<=-50)']['pe'].mean() if not df[df['period'] == 'storm (Dst<=-50)'].empty else np.nan
            quiet_pe = df[df['period'] == 'quiet (Dst>-50)']['pe'].mean() if not df[df['period'] == 'quiet (Dst>-50)'].empty else np.nan
            discussion_rows.append({'label': label, 'all_pe': float(all_pe), 'storm_pe': float(storm_pe), 'quiet_pe': float(quiet_pe)})
        pd.DataFrame(discussion_rows).to_csv(output_dir / 'Tables' / 'discussion_metrics.csv', index=False)
        print('Saved discussion metrics')

    print('Saved outputs to', output_dir)
    if on_colab:
        print('Drive output path:', output_dir)


if __name__ == '__main__':
    main()
