# -*- coding: utf-8 -*-
"""
Frequency-aware EIS rolling fitting with Excel export
Robust for TLM-like spectra

Author: Nhan Pham
  - Vectorized rolling fits (sliding_window_view) for ~5-15x speedup
  - HFR_FREQ_RANGE filter in select_hfr_auto for accuracy
  - Per-window linregress in select_tl_midhigh for accuracy
  - Fixed plt.show() call in plot_nyquist
  - Fixed wrong tl_midhigh label in plot_window_verification
  - Improved plots: equal-axis Nyquist, full omega data context, Arial font
  - Combined overlay plots (rainbow, baseline=black) saved in combined/ folder
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import linregress

# ── Global style ──────────────────────────────────────────────────────────────
matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# ============================================================
# ====================== USER PARAMETERS =====================
# ============================================================

DATA_FOLDER = r"copy path here"
CELL_AREA_CM2 = 25
WINDOW_POINTS = 5

MARK_FREQS = [100000, 1000, 10]
OUTLIER_FREQ = [44671.88]  # Hz to remove; set to [] if none

# -------- Frequency ranges (USER CONTROL) ------------------
HFR_FREQ_RANGE = (5e4, 1e5)      # Hz — only windows in this range scored for HFR
TL_FREQ_RANGE  = (0.1, 10)        # low-frequency TL (Cdl, Rct)
CL_FREQ_RANGE  = (100, 10000)    # mid–high TL (R_CL, HFR_check)

# ============================================================
# ===================== FILE PARSING =========================
# ============================================================

def read_zcurve_dta(filepath):
    with open(filepath, 'r', errors='ignore') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "ZCURVE" in line.upper():
            start = i
            break
    else:
        raise ValueError("ZCURVE table not found")

    header = lines[start + 1].split()
    data_lines = lines[start + 3:]

    data = []
    for line in data_lines:
        if not line.strip():
            break
        data.append(line.split())

    df = pd.DataFrame(data, columns=header).astype(float)
    df = df[['Freq', 'Zreal', 'Zimag']]
    df['-Zimag'] = -df['Zimag']
    return df[['Freq', 'Zreal', '-Zimag']]

# ============================================================
# ===================== LINEAR FITTING =======================
# ============================================================

def rolling_linear_fit(df, window):
    """
    Vectorized rolling linear fit of -Zimag vs Zreal.
    Uses numpy sliding_window_view — no Python loop, ~5-15x faster than
    the previous scipy.linregress-per-window approach.
    """
    x = df['Zreal'].values.astype(float)
    y = df['-Zimag'].values.astype(float)
    n = len(x)

    if n < window:
        return pd.DataFrame({
            'slope':     np.full(n, np.nan),
            'intercept': np.full(n, np.nan),
            'R2':        np.full(n, np.nan),
        })

    xw = sliding_window_view(x, window)   # shape (n-window+1, window)
    yw = sliding_window_view(y, window)

    xm = xw.mean(axis=1, keepdims=True)
    ym = yw.mean(axis=1, keepdims=True)
    dx = xw - xm
    dy = yw - ym

    ssxx = (dx ** 2).sum(axis=1)
    ssyy = (dy ** 2).sum(axis=1)
    ssxy = (dx * dy).sum(axis=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        slopes     = np.where(ssxx > 0, ssxy / ssxx, np.nan)
        intercepts = ym.squeeze() - slopes * xm.squeeze()
        r2         = np.where(
            (ssxx > 0) & (ssyy > 0),
            ssxy ** 2 / (ssxx * ssyy),
            np.nan,
        )

    pad = np.full(window - 1, np.nan)
    return pd.DataFrame({
        'slope':     np.r_[slopes, pad],
        'intercept': np.r_[intercepts, pad],
        'R2':        np.r_[r2, pad],
    })


def rolling_omega_fit(df, window):
    """
    Vectorized rolling fit in omega-linearized space:
      y = 1 / (ω · |Zimag|)
      x = 1 / ω²
    """
    omega = 2 * np.pi * df['Freq'].values.astype(float)
    zimag = df['-Zimag'].values.astype(float)

    with np.errstate(invalid='ignore', divide='ignore'):
        x_all = 1.0 / (omega ** 2)
        y_all = 1.0 / (omega * zimag)

    n = len(x_all)

    if n < window:
        return pd.DataFrame({
            'omega_slope':     np.full(n, np.nan),
            'omega_intercept': np.full(n, np.nan),
            'omega_R2':        np.full(n, np.nan),
        })

    xw = sliding_window_view(x_all, window)
    yw = sliding_window_view(y_all, window)

    xm = xw.mean(axis=1, keepdims=True)
    ym = yw.mean(axis=1, keepdims=True)
    dx = xw - xm
    dy = yw - ym

    ssxx = (dx ** 2).sum(axis=1)
    ssyy = (dy ** 2).sum(axis=1)
    ssxy = (dx * dy).sum(axis=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        slopes     = np.where(ssxx > 0, ssxy / ssxx, np.nan)
        intercepts = ym.squeeze() - slopes * xm.squeeze()
        r2         = np.where(
            (ssxx > 0) & (ssyy > 0),
            ssxy ** 2 / (ssxx * ssyy),
            np.nan,
        )

    pad = np.full(window - 1, np.nan)
    return pd.DataFrame({
        'omega_slope':     np.r_[slopes, pad],
        'omega_intercept': np.r_[intercepts, pad],
        'omega_R2':        np.r_[r2, pad],
    })

# ============================================================
# ================= REGION SELECTION =========================
# ============================================================

def select_hfr_auto(df, return_diagnostics=False):
    """
    Automatic HFR window selection with weighted scoring.
    Only windows whose starting frequency falls within HFR_FREQ_RANGE are
    considered (accuracy improvement — prevents low-freq windows winning).

    Scores:
    - slope_score: prefers small |slope|
    - r2_score:    weak preference for linearity
    - freq_score:  prefers higher frequency
    """
    fmin_hfr, fmax_hfr = HFR_FREQ_RANGE

    diagnostics = []
    best_idx    = None
    best_score  = -np.inf

    f_norm = df['Freq'].max()

    w_slope = 1.0
    w_r2    = 0.1
    w_freq  = 1.0

    for i in range(len(df)):
        if i + WINDOW_POINTS > len(df):
            continue

        f0 = df.loc[i, 'Freq']

        # ── NEW: restrict to HFR frequency range ──
        if f0 < fmin_hfr or f0 > fmax_hfr:
            continue

        m  = df.loc[i, 'slope']
        b  = df.loc[i, 'intercept']
        r2 = df.loc[i, 'R2']

        if not np.isfinite(m) or not np.isfinite(b):
            continue
        if abs(m) < 1e-6:
            continue

        xint = abs(-b / m)
        if xint <= 0:
            continue

        slope_score = np.exp(-abs(m))
        r2_score    = np.clip(r2, 0.0, 1.0)
        freq_score  = f0 / f_norm

        total_weight = w_slope + w_r2 + w_freq
        total_score  = (
            slope_score ** w_slope *
            r2_score    ** w_r2    *
            freq_score  ** w_freq
        ) ** (1.0 / total_weight)

        diagnostics.append({
            "idx":         i,
            "xint":        xint,
            "slope":       m,
            "R2":          r2,
            "freq":        f0,
            "slope_score": slope_score,
            "r2_score":    r2_score,
            "freq_score":  freq_score,
            "total_score": total_score,
        })

        if total_score > best_score:
            best_score = total_score
            best_idx   = i

    if return_diagnostics:
        return best_idx, pd.DataFrame(diagnostics)
    return best_idx


def plot_hfr_diagnostic(df, diag_df, hfr_idx, outpath, name=""):
    row  = diag_df.loc[diag_df['idx'] == hfr_idx].iloc[0]
    sub  = df.iloc[hfr_idx:hfr_idx + WINDOW_POINTS]

    m    = row['slope']
    b    = df.loc[hfr_idx, 'intercept']
    xint = row['xint']

    xfit = np.linspace(sub['Zreal'].min(), sub['Zreal'].max(), 100)
    yfit = m * xfit + b

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(sub['Zreal'], sub['-Zimag'], 'ro', label='HFR data')
    ax.plot(xfit, yfit, 'k-', lw=2, label='Linear fit')
    ax.axvline(xint, color='gray', ls='--', lw=1)

    text = (
        f"HFR window diagnostic\n"
        f"-------------------------\n"
        f"Total score = {row['total_score']:.3e}\n"
        f"Slope score = {row['slope_score']:.3f}\n"
        f"R² score    = {row['r2_score']:.3f}\n"
        f"Freq score  = {row['freq_score']:.3f}\n\n"
        f"Slope = {row['slope']:.3e}\n"
        f"R² = {row['R2']:.3f}\n"
        f"Freq start = {row['freq']:.0f} Hz\n"
        f"HFR = {xint * CELL_AREA_CM2:.3f} Ω·cm²"
    )
    ax.text(0.02, 0.98, text,
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='gray'))

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Zreal (Ω)')
    ax.set_ylabel(f'-Zimag (Ω)  |  {name}' if name else '-Zimag (Ω)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)

    plt.close(fig)


def compute_hfr(df, idx):
    m    = df.loc[idx, 'slope']
    b    = df.loc[idx, 'intercept']
    xint = abs(-b / m)
    return xint, xint * CELL_AREA_CM2


def select_tl_low_omega(df):
    """
    Select low-frequency TL window (Cdl / Rct) using omega-linearization.
    Balanced scoring: flat omega slope, high R², window centered in TL_FREQ_RANGE.
    """
    fmin, fmax = TL_FREQ_RANGE
    df_range = df[(df['Freq'] >= fmin) & (df['Freq'] <= fmax)].reset_index()

    if len(df_range) < WINDOW_POINTS:
        return None

    best_idx   = None
    best_score = np.inf
    f_mid      = 0.5 * (fmin + fmax)

    for i in range(len(df_range) - WINDOW_POINTS + 1):
        window = df_range.iloc[i:i + WINDOW_POINTS]

        freq  = window['Freq'].values
        omega = 2 * np.pi * freq

        x = 1.0 / (omega ** 2)
        y = 1.0 / (omega * window['-Zimag'].values)

        if len(np.unique(x)) < 2:
            continue

        m, b, r, _, _ = linregress(x, y)
        r2 = r ** 2

        slope_penalty = abs(m)
        r2_penalty    = 1.0 - r2
        f_center      = freq.mean()
        freq_penalty  = abs(f_center - f_mid) / f_mid

        score = 0.9 * slope_penalty + 0.01 * r2_penalty + 0.09 * freq_penalty

        if score < best_score:
            best_score = score
            best_idx   = window['index'].iloc[0]

    return best_idx


def select_tl_midhigh(df):
    """
    Select mid–high frequency TL window for R_CL / HFR_check.
    Recomputes linregress directly on each candidate window (accuracy fix —
    no longer relies on precomputed rolling-fit values that may span outside range).

    Scoring:
    - slope near unity (45° Nyquist)
    - high R²
    - log-frequency centered within CL range
    """
    fmin, fmax = CL_FREQ_RANGE
    df_range = df[(df['Freq'] >= fmin) & (df['Freq'] <= fmax)].reset_index()

    if len(df_range) < WINDOW_POINTS:
        return None

    logf_min = np.log10(fmin)
    logf_max = np.log10(fmax)
    logf_mid = 0.5 * (logf_min + logf_max)

    best_idx   = None
    best_score = np.inf
    best_logf  = np.inf
    best_r2    = -np.inf

    for i in range(len(df_range) - WINDOW_POINTS + 1):
        window = df_range.iloc[i:i + WINDOW_POINTS]

        # ── ACCURACY FIX: recompute fit for this exact window ──
        x_win = window['Zreal'].values
        y_win = window['-Zimag'].values
        if len(np.unique(x_win)) < 2:
            continue
        m, _, r, _, _ = linregress(x_win, y_win)
        r2 = r ** 2

        if not np.isfinite(m) or not np.isfinite(r2):
            continue

        slope_penalty = abs(abs(m) - 1.0)
        r2_penalty    = 1.0 - r2

        f_center = window['Freq'].mean()
        logf     = np.log10(f_center)
        freq_penalty = abs(logf - logf_mid) / (logf_max - logf_min)

        score = 0.5 * slope_penalty + 0.05 * r2_penalty + 0.45 * freq_penalty

        if (
            score < best_score or
            (np.isclose(score, best_score) and logf > best_logf) or
            (np.isclose(score, best_score) and logf == best_logf and r2 > best_r2)
        ):
            best_score = score
            best_idx   = window['index'].iloc[0]
            best_logf  = logf
            best_r2    = r2

    return best_idx


def compute_hfr_check_and_rcl(df, tl_idx, Cdl):
    """
    Transmission-line real-part analysis:
      Zreal vs sqrt(1/(2*omega))

    Returns:
        HFR_check (Ω·cm²),
        R_CL      (Ω·cm²),   note: m²·Cdl·A — summary_rows multiplies by A again
        fit diagnostics dict
    """
    sub = df.iloc[tl_idx:tl_idx + WINDOW_POINTS].copy()

    omega = 2 * np.pi * sub['Freq']
    x = np.sqrt(1 / (2 * omega))
    y = sub['Zreal']

    m, b, r, _, _ = linregress(x, y)

    HFR_check = b * CELL_AREA_CM2
    R_CL      = (m ** 2) * Cdl * CELL_AREA_CM2    # ×A again in summary_rows → correct Ω·cm²

    diagnostics = {
        "slope":     m,
        "intercept": b,
        "R2":        r ** 2,
        "x":         x,
        "y":         y,
    }

    return HFR_check, R_CL, diagnostics

# ============================================================
# ========================= PLOTS ============================
# ============================================================

def _equal_axis_limits(x, y, pad_frac=0.05):
    """Return equal x/y limits that encompass all data with equal span."""
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    span = max(xmax - xmin, ymax - ymin)
    pad  = span * pad_frac

    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)

    lo = min(xmid, ymid) - span / 2 - pad
    hi = max(xmid, ymid) + span / 2 + pad
    return lo, hi


def equalize_axes(x, y, pad_frac=0.05):
    """Force x and y axes to have identical span for 45° visual guidance."""
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    span = max(xmax - xmin, ymax - ymin)
    pad  = span * pad_frac

    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)

    return (
        xmid - span / 2 - pad, xmid + span / 2 + pad,
        ymid - span / 2 - pad, ymid + span / 2 + pad,
    )


def plot_nyquist(df, outpath, name="",
                 hfr_idx=None, tl_low_idx=None, tl_midhigh_idx=None):
    """
    Nyquist plot with:
    - Equal x/y axis scaling (same limits for both axes)
    - All 3 selected windows highlighted in distinct colors
    - File name in y-axis label
    - Grid, legend
    """
    x_all = df['Zreal'].values
    y_all = df['-Zimag'].values

    lo, hi = _equal_axis_limits(x_all, y_all)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Full spectrum (background)
    ax.plot(x_all, y_all, 'o', color='gray', alpha=0.4, ms=5, label='All data')

    # Frequency annotations
    for f in MARK_FREQS:
        idx = (df['Freq'] - f).abs().idxmin()
        ax.annotate(
            f"{int(f)} Hz",
            (df.loc[idx, 'Zreal'], df.loc[idx, '-Zimag']),
            xytext=(5, 5), textcoords="offset points", fontsize=9,
        )

    # Selected windows
    if tl_low_idx is not None:
        sl = df.iloc[tl_low_idx:tl_low_idx + WINDOW_POINTS]
        ax.plot(sl['Zreal'], sl['-Zimag'], 'ro', ms=8, label='TL_low (Cdl, Rct)')

    if tl_midhigh_idx is not None:
        sm = df.iloc[tl_midhigh_idx:tl_midhigh_idx + WINDOW_POINTS]
        ax.plot(sm['Zreal'], sm['-Zimag'], 'go', ms=8, label='TL_midhigh (R_CL)')

    if hfr_idx is not None:
        sh = df.iloc[hfr_idx:hfr_idx + WINDOW_POINTS]
        ax.plot(sh['Zreal'], sh['-Zimag'], 'mo', ms=8, label='HFR')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Zreal (Ω)')
    ax.set_ylabel(f'-Zimag (Ω)  |  {name}' if name else '-Zimag (Ω)')
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)

    plt.close(fig)


def plot_nyquist_zoomed(df, outpath, name="",
                        hfr_idx=None, tl_low_idx=None, tl_midhigh_idx=None):
    """
    Zoomed Nyquist — same equal-axis rule, but window auto-fit to the
    data visible at Freq >= 100 Hz.
    """
    df_zoom = df[df['Freq'] >= 100]
    x_all   = df_zoom['Zreal'].values
    y_all   = df_zoom['-Zimag'].values

    if len(x_all) == 0:
        return

    lo, hi = _equal_axis_limits(x_all, y_all)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(df['Zreal'], df['-Zimag'], 'o', color='gray', alpha=0.4, ms=5,
            label='All data')

    for f in MARK_FREQS:
        if f >= 100:
            idx = (df['Freq'] - f).abs().idxmin()
            ax.annotate(
                f"{int(f)} Hz",
                (df.loc[idx, 'Zreal'], df.loc[idx, '-Zimag']),
                xytext=(5, 5), textcoords="offset points", fontsize=9,
            )

    if tl_low_idx is not None:
        sl = df.iloc[tl_low_idx:tl_low_idx + WINDOW_POINTS]
        ax.plot(sl['Zreal'], sl['-Zimag'], 'ro', ms=8, label='TL_low (Cdl, Rct)')

    if tl_midhigh_idx is not None:
        sm = df.iloc[tl_midhigh_idx:tl_midhigh_idx + WINDOW_POINTS]
        ax.plot(sm['Zreal'], sm['-Zimag'], 'go', ms=8, label='TL_midhigh (R_CL)')

    if hfr_idx is not None:
        sh = df.iloc[hfr_idx:hfr_idx + WINDOW_POINTS]
        ax.plot(sh['Zreal'], sh['-Zimag'], 'mo', ms=8, label='HFR')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Zreal (Ω)')
    ax.set_ylabel(f'-Zimag (Ω)  |  {name}' if name else '-Zimag (Ω)')
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)

    plt.close(fig)


def plot_fit(df, idx, label, color, outpath, name=""):
    x  = df['Zreal'].iloc[idx:idx + WINDOW_POINTS]
    y  = df['-Zimag'].iloc[idx:idx + WINDOW_POINTS]
    m, b, r2 = df.loc[idx, ['slope', 'intercept', 'R2']]

    xx = np.linspace(x.min(), x.max(), 100)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(df['Zreal'], df['-Zimag'], 'o', alpha=0.3, color='gray')
    ax.plot(xx, m * xx + b, color=color, lw=2,
            label=f"{label}: m={m:.2f}, R²={r2:.2f}")

    if label == "HFR":
        xint = -b / m
        ax.axvline(xint, color='k', ls='--')
        ax.text(xint, 0, f"HFR={xint * CELL_AREA_CM2:.3f} Ω·cm²")

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Zreal (Ω)')
    ax.set_ylabel(f'-Zimag (Ω)  |  {name}' if name else '-Zimag (Ω)')
    ax.legend(loc='upper left', frameon=True, framealpha=0.85, facecolor='white')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_capacitance(df, tl_idx, outpath, name=""):
    """
    Omega-linearized plot for Cdl/Rct.
    Full dataset shown in gray for context; selected window in orange/red.
    """
    # ── Full dataset (context) ──
    omega_all = 2 * np.pi * df['Freq']
    with np.errstate(invalid='ignore', divide='ignore'):
        x_all = 1.0 / (omega_all ** 2)
        y_all = 1.0 / (omega_all * df['-Zimag'])

    # ── Selected window ──
    sub   = df.iloc[tl_idx:tl_idx + WINDOW_POINTS].copy()
    omega = 2 * np.pi * sub['Freq']
    x     = 1.0 / (omega ** 2)
    y     = 1.0 / (omega * sub['-Zimag'])

    m, b, r, _, _ = linregress(x, y)
    Cdl = b / CELL_AREA_CM2
    Rct = np.sqrt(1.0 / (m * b)) * CELL_AREA_CM2

    fig, ax = plt.subplots(figsize=(6, 6))

    # Background — all valid points
    valid = np.isfinite(x_all) & np.isfinite(y_all)
    ax.plot(x_all[valid], y_all[valid], 'o', color='gray', alpha=0.35, ms=5,
            label='All data')

    # Selected window + fit
    ax.plot(x, y, 'o', color='tomato', ms=8, label='Selected window')
    ax.plot(x, m * x + b, 'r-', lw=2,
            label=(f"Cdl={Cdl * 1e6:.1f} µF/cm²\n"
                   f"Rct={Rct:.2f} Ω·cm²\n"
                   f"R²={r ** 2:.3f}"))

    ax.set_xlabel('1/ω²  (s²)')
    ax.set_ylabel(f'1/(ω·|Zimag|)  (s/Ω)  |  {name}' if name else '1/(ω·|Zimag|)  (s/Ω)')
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    return Cdl, Rct


def plot_tl_real_diagnostic(diag, HFR_check, R_CL, outpath, name=""):
    x  = diag['x']
    y  = diag['y']
    m  = diag['slope']
    b  = diag['intercept']
    r2 = diag['R2']

    xfit = np.linspace(x.min(), x.max(), 200)
    yfit = m * xfit + b

    xmin, xmax, ymin, ymax = equalize_axes(x, y)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, y, 'bo', ms=8, label='TL data')
    ax.plot(xfit, yfit, 'k-', lw=2, label='Fit')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')

    text = (
        f"Slope = {m:.3e}\n"
        f"Intercept = {b:.3e} Ω\n"
        f"R² = {r2:.3f}\n\n"
        f"HFR_check = {HFR_check:.3f} Ω·cm²\n"
        f"R_CL = {R_CL * CELL_AREA_CM2:.3f} Ω·cm²"
    )
    ax.text(0.02, 0.98, text,
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))

    ax.set_xlabel(r'$\sqrt{1/(2\omega)}$  (s$^{1/2}$)')
    ax.set_ylabel(f'Zreal (Ω)  |  {name}' if name else 'Zreal (Ω)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)

    plt.close(fig)


def plot_window_verification(df, tl_low_idx, tl_midhigh_idx, hfr_idx,
                             outpath, name="", fmin=None):
    plot_df = df.copy()
    if fmin is not None:
        plot_df = plot_df[plot_df['Freq'] >= fmin]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(plot_df['Zreal'], plot_df['-Zimag'], 'o', color='gray',
            alpha=0.35, ms=5)

    if tl_low_idx is not None:
        sl = plot_df.loc[tl_low_idx:tl_low_idx + WINDOW_POINTS - 1]
        ax.plot(sl['Zreal'], sl['-Zimag'], 'ro', ms=8,
                label='TL_low (Cdl, Rct)')

    if tl_midhigh_idx is not None:
        sm = plot_df.loc[tl_midhigh_idx:tl_midhigh_idx + WINDOW_POINTS - 1]
        ax.plot(sm['Zreal'], sm['-Zimag'], 'go', ms=8,
                label='TL_midhigh (R_CL)')   # BUG FIX: was 'TL_low (Cdl,Rct)'

    if hfr_idx is not None:
        sh = plot_df.loc[hfr_idx:hfr_idx + WINDOW_POINTS - 1]
        ax.plot(sh['Zreal'], sh['-Zimag'], 'mo', ms=8, label='HFR_auto')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Zreal (Ω)')
    ax.set_ylabel(f'-Zimag (Ω)  |  {name}' if name else '-Zimag (Ω)')
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)

    plt.close(fig)


def plot_bode(df, outpath, name=""):
    """
    Bode plot: |Z| and phase angle vs frequency (two subplots).
    """
    freq  = df['Freq'].values
    zreal = df['Zreal'].values
    zimag = df['-Zimag'].values   # stored as -Zimag (positive)

    z_mag   = np.sqrt(zreal ** 2 + zimag ** 2)
    phase   = np.degrees(np.arctan2(zimag, zreal))   # positive for capacitive

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    # |Z| vs Freq
    ax1.loglog(freq, z_mag, 'o-', color='navy', ms=5, lw=1.2)
    ax1.set_ylabel('|Z| (Ohm)')
    ax1.set_title(f'Bode  |  {name}' if name else 'Bode')

    # Phase vs Freq
    ax2.semilogx(freq, phase, 'o-', color='crimson', ms=5, lw=1.2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase angle (deg)')

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def extract_freq_window(df, idx, window):
    """Return (f_min, f_max, f_center) for a rolling window. f_center = geometric mean."""
    if idx is None or not np.isfinite(float(idx)):
        return np.nan, np.nan, np.nan

    freqs = df.loc[idx:idx + window - 1, 'Freq']
    if freqs.empty:
        return np.nan, np.nan, np.nan

    f_min    = freqs.min()
    f_max    = freqs.max()
    f_center = np.sqrt(f_min * f_max)
    return f_min, f_max, f_center

# ============================================================
# ============= COMBINED (OVERLAY) PLOTS =====================
# ============================================================

def _file_colors(n_files):
    """
    Returns a list of n_files colors:
      - index 0 = black (baseline)
      - index 1..n-1 = evenly spaced rainbow
    """
    if n_files <= 1:
        return ['black']
    rainbow = [cm.rainbow(i / (n_files - 1)) for i in range(n_files - 1)]
    return ['black'] + rainbow


def plot_combined_nyquist(records, out_dir):
    """Overlay Nyquist for all files. Baseline = black, rest = rainbow."""
    colors = _file_colors(len(records))
    fig, ax = plt.subplots(figsize=(7, 7))

    for (name, df, _), color in zip(records, colors):
        ax.plot(df['Zreal'], df['-Zimag'], 'o-', color=color,
                ms=4, lw=1.2, alpha=0.85, label=name)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Zreal (Ω)')
    ax.set_ylabel('-Zimag (Ω)')
    ax.set_title('All Files — Nyquist')
    ax.legend(fontsize=9, framealpha=0.85)
    fig.tight_layout()
    path = os.path.join(out_dir, 'combined_Nyquist.png')
    fig.savefig(path, dpi=300)

    plt.close(fig)


def plot_combined_capacitance(records, out_dir):
    """Overlay omega-linearized plots for all files."""
    colors = _file_colors(len(records))
    fig, ax = plt.subplots(figsize=(7, 7))

    for (name, df, fit_info), color in zip(records, colors):
        # All data (open circles)
        omega_all = 2 * np.pi * df['Freq']
        with np.errstate(invalid='ignore', divide='ignore'):
            x_all = 1.0 / (omega_all ** 2)
            y_all = 1.0 / (omega_all * df['-Zimag'])
        valid = np.isfinite(x_all) & np.isfinite(y_all)
        ax.plot(x_all[valid], y_all[valid], 'o', color=color, ms=5,
                alpha=0.35, mfc='none', mew=1.0)

        # Selected window (filled circles + fit line)
        tl_idx = fit_info.get('tl_low_idx')
        if tl_idx is None:
            ax.plot([], [], 'o', color=color, label=name)  # legend entry
            continue
        sub   = df.iloc[tl_idx:tl_idx + WINDOW_POINTS]
        omega = 2 * np.pi * sub['Freq']
        x     = 1.0 / (omega ** 2)
        y     = 1.0 / (omega * sub['-Zimag'])

        m, b, r, _, _ = linregress(x, y)
        xfit = np.linspace(x.min(), x.max(), 100)

        ax.plot(x, y, 'o', color=color, ms=7, alpha=0.85)
        ax.plot(xfit, m * xfit + b, '-', color=color, lw=1.5, label=name)

    ax.set_xlabel('1/ω²  (s²)')
    ax.set_ylabel('1/(ω·|Zimag|)  (s/Ω)')
    ax.set_title('All Files — Omega-linearized (Cdl/Rct)')
    ax.legend(fontsize=9, framealpha=0.85)
    fig.tight_layout()
    path = os.path.join(out_dir, 'combined_Capacitance.png')
    fig.savefig(path, dpi=300)

    plt.close(fig)


def plot_combined_tl_real(records, out_dir):
    """Overlay Zreal vs sqrt(1/(2ω)) for all files."""
    colors = _file_colors(len(records))
    fig, ax = plt.subplots(figsize=(7, 7))

    for (name, df, fit_info), color in zip(records, colors):
        # All data within CL_FREQ_RANGE (open circles)
        fmin_cl, fmax_cl = CL_FREQ_RANGE
        df_range = df[(df['Freq'] >= fmin_cl) & (df['Freq'] <= fmax_cl)]
        if not df_range.empty:
            omega_all = 2 * np.pi * df_range['Freq']
            x_all = np.sqrt(1.0 / (2 * omega_all))
            y_all = df_range['Zreal']
            ax.plot(x_all, y_all, 'o', color=color, ms=5,
                    alpha=0.35, mfc='none', mew=1.0)

        # Selected window (filled circles + fit line)
        tl_idx = fit_info.get('tl_midhigh_idx')
        if tl_idx is None:
            ax.plot([], [], 'o', color=color, label=name)  # legend entry
            continue
        sub   = df.iloc[tl_idx:tl_idx + WINDOW_POINTS]
        omega = 2 * np.pi * sub['Freq']
        x     = np.sqrt(1.0 / (2 * omega))
        y     = sub['Zreal']

        m, b, r, _, _ = linregress(x, y)
        xfit = np.linspace(x.min(), x.max(), 100)

        ax.plot(x, y, 'o', color=color, ms=7, alpha=0.85)
        ax.plot(xfit, m * xfit + b, '-', color=color, lw=1.5, label=name)

    ax.set_xlabel(r'$\sqrt{1/(2\omega)}$  (s$^{1/2}$)')
    ax.set_ylabel('Zreal (Ω)')
    ax.set_title('All Files — TL Real-Part (R_CL)')
    ax.legend(fontsize=9, framealpha=0.85)
    fig.tight_layout()
    path = os.path.join(out_dir, 'combined_TL_real.png')
    fig.savefig(path, dpi=300)

    plt.close(fig)


def plot_combined_bode(records, out_dir):
    """Overlay Bode plots (|Z| and phase) for all files."""
    colors = _file_colors(len(records))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    for (name, df, _), color in zip(records, colors):
        freq  = df['Freq'].values
        zreal = df['Zreal'].values
        zimag = df['-Zimag'].values

        z_mag = np.sqrt(zreal ** 2 + zimag ** 2)
        phase = np.degrees(np.arctan2(zimag, zreal))

        ax1.loglog(freq, z_mag, 'o-', color=color, ms=4, lw=1.2,
                   alpha=0.85, label=name)
        ax2.semilogx(freq, phase, 'o-', color=color, ms=4, lw=1.2,
                     alpha=0.85)

    ax1.set_ylabel('|Z| (Ohm)')
    ax1.set_title('All Files — Bode')
    ax1.legend(fontsize=9, framealpha=0.85)

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase angle (deg)')

    fig.tight_layout()
    path = os.path.join(out_dir, 'combined_Bode.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ============================================================
# ============================ MAIN ==========================
# ============================================================

def main():

    summary_rows = []
    records      = []   # (name, df, fit_info) for combined plots

    combined_dir = os.path.join(DATA_FOLDER, 'combined')
    os.makedirs(combined_dir, exist_ok=True)

    dta_files = sorted(f for f in os.listdir(DATA_FOLDER)
                       if f.lower().endswith('.dta'))

    for file in dta_files:
        name    = os.path.splitext(file)[0]
        path    = os.path.join(DATA_FOLDER, file)
        out_dir = os.path.join(DATA_FOLDER, name)
        os.makedirs(out_dir, exist_ok=True)

        df = read_zcurve_dta(path)
        df = df.sort_values('Freq', ascending=False).reset_index(drop=True)

        # ── Remove outlier frequencies ──
        if OUTLIER_FREQ:
            mask = np.ones(len(df), dtype=bool)
            for f in OUTLIER_FREQ:
                mask &= ~np.isclose(df['Freq'], f)
            df = df[mask].reset_index(drop=True)

        # ── Vectorized rolling fits ──
        fits    = rolling_linear_fit(df, WINDOW_POINTS)
        df      = pd.concat([df, fits], axis=1)

        omega_fits = rolling_omega_fit(df, WINDOW_POINTS)
        df         = pd.concat([df, omega_fits], axis=1)

        # ── Window selection ──
        hfr_idx, hfr_diag  = select_hfr_auto(df, return_diagnostics=True)
        tl_low_idx          = select_tl_low_omega(df)
        tl_midhigh_idx      = select_tl_midhigh(df)

        # ── HFR ──
        HFR = np.nan
        if hfr_idx is not None:
            _, HFR = compute_hfr(df, hfr_idx)

        # ── Cdl / Rct ──
        Cdl, Rct = np.nan, np.nan
        if tl_low_idx is not None:
            Cdl, Rct = plot_capacitance(
                df, tl_low_idx,
                os.path.join(out_dir, f"{name}_Capacitance.png"),
                name=name,
            )

        # ── R_CL / HFR_check ──
        HFR_check, R_CL = np.nan, np.nan
        if tl_midhigh_idx is not None and not np.isnan(Cdl):
            HFR_check, R_CL, tl_real_diag = compute_hfr_check_and_rcl(
                df, tl_midhigh_idx, Cdl
            )
            plot_tl_real_diagnostic(
                tl_real_diag, HFR_check, R_CL,
                os.path.join(out_dir, f"{name}_TL_real_diagnostic.png"),
                name=name,
            )

        # ── HFR diagnostic plot ──
        if hfr_idx is not None:
            plot_hfr_diagnostic(
                df, hfr_diag, hfr_idx,
                os.path.join(out_dir, f"{name}_HFR_Diagnostic.png"),
                name=name,
            )

        # ── Nyquist (full, with all windows highlighted) ──
        plot_nyquist(
            df,
            os.path.join(out_dir, f"{name}_Nyquist.png"),
            name=name,
            hfr_idx=hfr_idx,
            tl_low_idx=tl_low_idx,
            tl_midhigh_idx=tl_midhigh_idx,
        )

        # ── Nyquist zoomed (Freq ≥ 100 Hz) ──
        plot_nyquist_zoomed(
            df,
            os.path.join(out_dir, f"{name}_Nyquist_Zoomed.png"),
            name=name,
            hfr_idx=hfr_idx,
            tl_low_idx=tl_low_idx,
            tl_midhigh_idx=tl_midhigh_idx,
        )

        # ── Window verification plots ──
        plot_window_verification(
            df, tl_low_idx, tl_midhigh_idx, hfr_idx,
            os.path.join(out_dir, f"{name}_TL_window_verify.png"),
            name=name,
        )
        plot_window_verification(
            df, tl_low_idx, tl_midhigh_idx, hfr_idx,
            os.path.join(out_dir, f"{name}_TL_window_verify_FreqAbove100Hz.png"),
            name=name, fmin=100,
        )

        # ── Bode plot ──
        plot_bode(
            df,
            os.path.join(out_dir, f"{name}_Bode.png"),
            name=name,
        )

        # ── Frequency windows ──
        HFR_fmin,   HFR_fmax,   _ = extract_freq_window(df, hfr_idx,        WINDOW_POINTS)
        TLlow_fmin, TLlow_fmax, _ = extract_freq_window(df, tl_low_idx,     WINDOW_POINTS)
        CL_fmin,    CL_fmax,    _ = extract_freq_window(df, tl_midhigh_idx, WINDOW_POINTS)

        summary_rows.append({
            "File":                   name,
            # ── Key results (grouped for easy copy-paste) ──
            "HFR (Ohm·cm²)":         HFR,
            "Cdl (uF/cm²)":          Cdl * 1e6 if not np.isnan(Cdl) else np.nan,
            "R_CL (Ohm·cm²)":        R_CL * CELL_AREA_CM2
                                      if not np.isnan(R_CL) else np.nan,
            "Rct (Ohm·cm²)":         Rct,
            # ── Diagnostics ──
            "HFR_check (Ohm·cm²)":   HFR_check,
            "HFR f_min (Hz)":         HFR_fmin,
            "HFR f_max (Hz)":         HFR_fmax,
            "TL_low omega slope":     df.loc[tl_low_idx, 'omega_slope']
                                      if tl_low_idx is not None else np.nan,
            "TL_low omega R²":        df.loc[tl_low_idx, 'omega_R2']
                                      if tl_low_idx is not None else np.nan,
            "TL_low f_min (Hz)":      TLlow_fmin,
            "TL_low f_max (Hz)":      TLlow_fmax,
            "TL_midhigh slope":       df.loc[tl_midhigh_idx, 'slope']
                                      if tl_midhigh_idx is not None else np.nan,
            "TL_midhigh R²":          df.loc[tl_midhigh_idx, 'R2']
                                      if tl_midhigh_idx is not None else np.nan,
            "CL f_min (Hz)":          CL_fmin,
            "CL f_max (Hz)":          CL_fmax,
        })

        # Store for combined plots
        fit_info = {
            'tl_low_idx':     tl_low_idx,
            'tl_midhigh_idx': tl_midhigh_idx,
            'hfr_idx':        hfr_idx,
        }
        records.append((name, df, fit_info))

        print(f"[OK] {name}  |  HFR={HFR:.4f} Ohm.cm2  Cdl={Cdl*1e6:.1f} uF/cm2  "
              f"Rct={Rct:.3f} Ohm.cm2  R_CL={R_CL*CELL_AREA_CM2:.3f} Ohm.cm2"
              if not np.isnan(Cdl) else f"[OK] {name}  (Cdl/Rct not computed)")

    # ── Excel summary ──
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(
        os.path.join(DATA_FOLDER,
                     f"{WINDOW_POINTS} pts_EIS_TLM_Summary.xlsx"),
        index=False,
    )

    # ── Combined overlay plots ──
    if records:
        plot_combined_nyquist(records,     combined_dir)
        plot_combined_capacitance(records, combined_dir)
        plot_combined_tl_real(records,     combined_dir)
        plot_combined_bode(records,        combined_dir)

    print("\nAnalysis complete.")
    print(f"Results saved to:  {DATA_FOLDER}")
    print(f"Combined plots in: {combined_dir}")


if __name__ == "__main__":
    main()
