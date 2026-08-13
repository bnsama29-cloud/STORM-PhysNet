"""
Synthetic data generator for GOES >2 MeV electron flux and Wind solar wind.
Produces 11 years of realistic space weather data including storm events,
using Burton-equation-driven Dst proxy for physically consistent storms.
"""

# DEV/TESTING ONLY — not used for paper GOES/OMNI/GRASP tables (use_real_data: true).

import numpy as np
import pandas as pd
from scipy import signal


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants and empirical parameters
# ─────────────────────────────────────────────────────────────────────────────
HOURS_PER_YEAR   = 8760
SOLAR_CYCLE_DAYS = 4015        # ~11-year solar cycle in days
Bz_THRESHOLD     = -0.5        # Burton Eq: Bz threshold (mV/m)
TAU_RING_CURRENT = 7.7         # hours, ring current decay timescale
Q_QUIET          = 11.0        # Burton Eq: quiet-time Dst contribution
A_BURTON         = 3.6e-5      # Burton injection coefficient
B_BURTON         = 0.20        # Burton decay rate coefficient


def _solar_wind_background(n_hours: int, rng: np.random.Generator) -> dict:
    """Generate background solar wind with realistic statistics."""
    t = np.arange(n_hours)

    # Solar cycle modulation (11-year sinusoid)
    solar_cycle = 0.5 * (1 + np.sin(2 * np.pi * t / (SOLAR_CYCLE_DAYS * 24) - np.pi / 2))

    # --- Solar wind speed (km/s) ---
    # Slow wind ~400 km/s, fast streams up to ~700 km/s
    vsw_base = 400 + 80 * solar_cycle
    vsw_noise = rng.normal(0, 30, n_hours)
    # Inject high-speed streams (CIRs) periodically ~27-day recurrence
    cir_times = np.where(rng.random(n_hours) < 0.002)[0]  # ~0.2% per hour
    vsw_hss = np.zeros(n_hours)
    for ct in cir_times:
        duration = rng.integers(24, 96)
        end = min(ct + duration, n_hours)
        vsw_hss[ct:end] += rng.uniform(150, 350) * np.exp(
            -np.arange(end - ct) / (duration / 3)
        )
    vsw = np.clip(vsw_base + vsw_noise + vsw_hss, 250, 900)

    # --- IMF Bz (nT) ---
    # Quiet: ~0 nT, storms: sustained negative
    bz_base = rng.normal(0, 3, n_hours)
    # Add low-frequency oscillations
    bz_base += 2 * np.sin(2 * np.pi * t / 120 + rng.uniform(0, 2 * np.pi))
    bz = bz_base.copy()

    # --- IMF Bt (nT) ---
    bt_base = 5 + 3 * solar_cycle + rng.exponential(2, n_hours)
    bt = np.clip(bt_base, 1, 40)

    # --- Proton density (cm^-3) ---
    density = np.clip(rng.lognormal(2.3, 0.5, n_hours), 1, 60)

    # --- Dynamic pressure (nPa) ---
    mp = 1.6726e-27
    pdyn = 0.5 * mp * (density * 1e6) * (vsw * 1e3) ** 2 * 1e9
    pdyn = np.clip(pdyn, 0.5, 30)

    return dict(vsw=vsw, bz=bz, bt=bt, density=density, pdyn=pdyn,
                solar_cycle=solar_cycle)


def _inject_storms(sw: dict, n_hours: int, rng: np.random.Generator,
                   storm_rate: float = 0.004) -> tuple:
    """
    Inject geomagnetic storms using a simplified Burton-equation Dst model.
    Returns updated solar wind and Dst time series.
    """
    bz    = sw["bz"].copy()
    vsw   = sw["vsw"].copy()
    pdyn  = sw["pdyn"].copy()
    dst   = np.zeros(n_hours)
    storm_mask = np.zeros(n_hours, dtype=bool)

    # Storm onset times
    storm_onsets = np.where(rng.random(n_hours) < storm_rate)[0]

    for onset in storm_onsets:
        # Storm duration: 12–72 hours
        duration = rng.integers(12, 73)
        end = min(onset + duration, n_hours)
        sl  = slice(onset, end)

        # Inject sustained southward Bz (−5 to −30 nT)
        bz_storm = -rng.uniform(5, 30)
        bz_profile = bz_storm * np.exp(-np.arange(end - onset) / (duration * 0.4))
        bz[sl] = np.minimum(bz[sl], bz_profile)  # take the more southward

        # Elevated solar wind speed
        vsw[sl] = np.clip(vsw[sl] + rng.uniform(100, 300), 300, 900)

        storm_mask[sl] = True

    # ── Burton equation for Dst ──────────────────────────────────────────────
    Ey = vsw * np.maximum(0, -bz) * 1e-3   # dawn-to-dusk electric field (mV/m)
    for t in range(1, n_hours):
        Q = Q_QUIET if Ey[t] <= Bz_THRESHOLD else A_BURTON * (Ey[t] - Bz_THRESHOLD)
        dDst = Q - B_BURTON * dst[t - 1]
        dst[t] = dst[t - 1] + dDst

    # Dst* corrected for solar wind dynamic pressure
    b0     = 15.8  # Chapman-Ferraro constant (nT * nPa^-0.5)
    dst    = dst - b0 * (np.sqrt(np.maximum(pdyn, 0.5)) - np.sqrt(2.0))
    dst    = np.clip(dst, -500, 50)

    sw["bz"]    = bz
    sw["vsw"]   = vsw
    return sw, dst, storm_mask


def _flux_from_solar_wind(sw: dict, dst: np.ndarray, n_hours: int,
                          rng: np.random.Generator,
                          longitude_offset: float = 0.0) -> np.ndarray:
    """
    Generate physically-motivated electron flux using solar wind and Dst.
    Based on empirical relationships:
      - Flux enhanced ~2–3 days after storm onset (acceleration time)
      - High-speed streams drive gradual flux enhancement
      - Strong storms can cause flux dropouts followed by recovery
    """
    log_flux = np.zeros(n_hours)

    # Baseline from solar wind speed (ULF wave acceleration proxy)
    log_flux += 0.004 * (sw["vsw"] - 400)

    # Dst-driven modulation (delayed ~48 hours)
    dst_delayed = np.roll(dst, 48)
    dst_delayed[:48] = 0
    # Negative Dst → eventual flux enhancement (with delay)
    log_flux += -0.005 * np.minimum(dst_delayed, 0)

    # Immediate dropout during main phase (Dst < -50)
    dropout = np.where(dst < -50, 0.01 * dst, 0.0)
    log_flux += dropout

    # Solar cycle modulation
    log_flux += 0.5 * sw["solar_cycle"]

    # Add realistic mean and noise
    log_flux += 3.5 + rng.normal(0, 0.15, n_hours)

    # Temporal smoothing (flux changes slowly in log space)
    b, a = signal.butter(3, 0.05, btype="low")
    log_flux = signal.filtfilt(b, a, log_flux)

    # Longitude offset (GRASP at Indian longitude sees slightly different flux)
    log_flux += longitude_offset + rng.normal(0, 0.05, n_hours)

    # Physical bounds: 10^-2 to 10^6 e/cm²/s/sr
    log_flux = np.clip(log_flux, -2.0, 6.0)
    return log_flux


def generate_synthetic_dataset(
    n_years: int = 11,
    seed: int = 42,
    storm_rate: float = 0.004,
    resample_freq: str = "1h",
    grasp_longitude_offset: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic GOES + Wind + GRASP dataset.

    Parameters
    ----------
    n_years : int
        Number of years to simulate.
    seed : int
        Random seed for reproducibility.
    storm_rate : float
        Probability per hour of a storm onset.
    resample_freq : str
        Time resolution of output.
    grasp_longitude_offset : float
        Log-flux offset for GRASP (Indian longitude).

    Returns
    -------
    goes_wind_df : pd.DataFrame
        GOES + Wind combined dataset (11 years).
    grasp_df : pd.DataFrame
        GRASP dataset (last 2 years).
    """
    rng     = np.random.default_rng(seed)
    n_hours = n_years * HOURS_PER_YEAR
    start   = pd.Timestamp("2010-01-01")
    index   = pd.date_range(start, periods=n_hours, freq="1h")

    # Generate solar wind
    sw = _solar_wind_background(n_hours, rng)

    # Inject storms and compute Dst
    sw, dst, storm_mask = _inject_storms(sw, n_hours, rng, storm_rate)

    # Generate electron flux
    log_flux_goes = _flux_from_solar_wind(sw, dst, n_hours, rng, 0.0)
    log_flux_grasp = _flux_from_solar_wind(sw, dst, n_hours, rng,
                                            grasp_longitude_offset)

    # Compute derived features
    bz_south_duration = np.zeros(n_hours)
    counter = 0
    for i in range(n_hours):
        if sw["bz"][i] < -3:
            counter += 1
        else:
            counter = 0
        bz_south_duration[i] = counter

    storm_onset_hours = np.zeros(n_hours)
    last_storm = -9999
    for i in range(n_hours):
        if storm_mask[i] and not (i > 0 and storm_mask[i - 1]):
            last_storm = i
        storm_onset_hours[i] = i - last_storm if last_storm >= 0 else 9999

    # Kp proxy (0–9 scale, correlated with storm activity)
    kp_proxy = np.clip(3 + (-dst / 30) + rng.normal(0, 0.5, n_hours), 0, 9)

    # Build main DataFrame (GOES + Wind)
    df = pd.DataFrame({
        "log_flux":           log_flux_goes,
        "vsw":                sw["vsw"],
        "bz":                 sw["bz"],
        "bt":                 sw["bt"],
        "density":            sw["density"],
        "pdyn":               sw["pdyn"],
        "dst":                dst,
        "kp":                 kp_proxy,
        "bz_south_duration":  bz_south_duration,
        "storm_onset_hours":  storm_onset_hours,
        "storm_flag":         storm_mask.astype(float),
    }, index=index)

    # GRASP dataset: last 2 years only
    grasp_start = n_hours - 2 * HOURS_PER_YEAR
    grasp_df = pd.DataFrame({
        "log_flux": log_flux_grasp[grasp_start:],
        "vsw":      sw["vsw"][grasp_start:],
        "bz":       sw["bz"][grasp_start:],
        "dst":      dst[grasp_start:],
        "kp":       kp_proxy[grasp_start:],
        "storm_flag": storm_mask[grasp_start:].astype(float),
    }, index=index[grasp_start:])

    print(f"[SyntheticGenerator] Generated {n_hours:,} hourly samples "
          f"({n_years} years)")
    print(f"  Storm periods: {storm_mask.sum():,} hours "
          f"({100*storm_mask.mean():.1f}%)")
    print(f"  log_flux range: [{log_flux_goes.min():.2f}, "
          f"{log_flux_goes.max():.2f}]")
    print(f"  Dst range: [{dst.min():.1f}, {dst.max():.1f}] nT")

    return df, grasp_df
