"""Stages 1 & 3-4: unit labeling (noise / good / mua) and over-merge demotion.

Three independent labelers are run where available and the exported label is a
conservative ensemble:

  * threshold labeler (IBL-style quality-metric thresholds)  — always runs
  * Bombcell (Fabre et al.), via spikeinterface.curation      — if enabled
  * UnitRefine ML classifiers (noise/neural + SUA/MUA)        — if enabled and
    the models can be loaded (downloaded from Hugging Face on first use)

Ensemble policy (conservative):
  noise  <- every labeler that ran says noise (and never when a unit passed merging)
  good   <- passes thresholds AND bombcell-good (if ran) AND unitrefine-SUA (if ran)
  mua    <- everything else
Nothing is ever deleted: noise units remain in the export, labeled 'noise'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evidence import get_quality_metrics


def threshold_labels(metrics: pd.DataFrame, cfg: dict, uv_scaled: bool) -> pd.Series:
    th = cfg["labeling"]["thresholds"]
    ok = pd.Series(True, index=metrics.index)

    def apply(col, fn):
        nonlocal ok
        if col in metrics.columns:
            vals = metrics[col]
            ok &= vals.apply(lambda v: bool(fn(v)) if pd.notna(v) else False)

    apply("num_spikes", lambda v: v >= th["min_num_spikes"])
    apply("presence_ratio", lambda v: v >= th["min_presence_ratio"])
    # contamination: sliding RP (IBL) where defined, falling back per-unit to
    # rp_contamination (sliding RP is NaN for low-count units)
    max_contam = th["max_sliding_rp_contamination"]
    if "sliding_rp_violation" in metrics.columns and "rp_contamination" in metrics.columns:
        contam = metrics["sliding_rp_violation"].fillna(metrics["rp_contamination"])
        ok &= contam.apply(lambda v: bool(v <= max_contam) if pd.notna(v) else False)
    elif "rp_contamination" in metrics.columns:
        apply("rp_contamination", lambda v: v <= max_contam)
    # amplitude_cutoff is NaN for low-count units; missing evidence is not failure
    # (those units are already gated by min_num_spikes and contamination)
    if "amplitude_cutoff" in metrics.columns:
        ok &= metrics["amplitude_cutoff"].apply(
            lambda v: bool(v <= th["max_amplitude_cutoff"]) if pd.notna(v) else True)
    apply("snr", lambda v: v >= th["min_snr"])
    if uv_scaled and "amplitude_median" in metrics.columns:
        apply("amplitude_median", lambda v: abs(v) >= th["min_amplitude_median_uv"])

    return ok.map({True: "good", False: "mua"})


def bombcell_labels(analyzer, cfg: dict, verbose: bool = True):
    """Returns (labels Series in {'good','mua','noise','non-somatic*'}, error or None)."""
    try:
        from spikeinterface.curation import bombcell_label_units, bombcell_get_default_thresholds
        thresholds = bombcell_get_default_thresholds()
        df = bombcell_label_units(sorting_analyzer=analyzer, thresholds=thresholds)
        col = "bombcell_label" if "bombcell_label" in df.columns else df.columns[0]
        labels = df[col].astype(str).str.lower()
        labels.index = analyzer.unit_ids
        return labels, None
    except Exception as e:
        if verbose:
            print(f"[label] bombcell unavailable: {type(e).__name__}: {e}")
        return None, f"{type(e).__name__}: {e}"


def unitrefine_labels(analyzer, verbose: bool = True):
    """Returns (labels Series in {'sua','mua','noise'}, error or None)."""
    try:
        from spikeinterface.curation import unitrefine_label_units
        df = unitrefine_label_units(
            sorting_analyzer=analyzer,
            noise_neural_classifier="SpikeInterface/UnitRefine_noise_neural_classifier",
            sua_mua_classifier="SpikeInterface/UnitRefine_sua_mua_classifier",
        )
        if isinstance(df, pd.DataFrame):
            col = "unitrefine_label" if "unitrefine_label" in df.columns else df.columns[0]
            labels = df[col]
        else:
            labels = pd.Series(df)
        labels = labels.astype(str).str.lower()
        labels.index = analyzer.unit_ids
        return labels, None
    except Exception as e:
        if verbose:
            print(f"[label] unitrefine unavailable: {type(e).__name__}: {e}")
        return None, f"{type(e).__name__}: {e}"


def amplitude_bimodality_demotions(analyzer, cfg: dict, exclude_ids=()) -> dict:
    """Over-merge screen: units whose spike-amplitude distribution is strongly
    bimodal get demoted to mua (demote-don't-split). Returns {unit_id: info}.

    `exclude_ids`: units produced by an accepted drift merge — their amplitude
    distribution is bimodal by construction (slow drift), and they were already
    vetted by the merge gate (contamination + junction continuity)."""
    if not cfg["overmerge"]["enabled"]:
        return {}
    exclude_ids = set(exclude_ids)
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return {}
    amp_ext = analyzer.get_extension("spike_amplitudes")
    amps_by_unit = amp_ext.get_data(outputs="by_unit")[0]
    flagged = {}
    for u in analyzer.unit_ids:
        if u in exclude_ids:
            continue
        a = np.asarray(amps_by_unit[u], dtype=float).reshape(-1, 1)
        if a.size < 200:
            continue
        a = a[np.isfinite(a).ravel()]
        a = a.reshape(-1, 1)
        try:
            g1 = GaussianMixture(1, random_state=0).fit(a)
            g2 = GaussianMixture(2, random_state=0).fit(a)
        except Exception:
            continue
        # require a real BIC win AND well-separated, balanced components
        if g2.bic(a) >= g1.bic(a) - 10:
            continue
        mu = g2.means_.ravel()
        sd = np.sqrt(g2.covariances_.ravel())
        w = g2.weights_.ravel()
        ashman_d = abs(mu[0] - mu[1]) * np.sqrt(2.0) / np.sqrt(sd[0] ** 2 + sd[1] ** 2)
        if ashman_d > 3.0 and w.min() > 0.2:
            flagged[u] = dict(ashman_d=float(ashman_d), weights=w.tolist(), means=mu.tolist())
    return flagged


def ensemble_labels(analyzer, cfg: dict, uv_scaled: bool, verbose: bool = True,
                    merged_unit_ids=()):
    """Run all labelers and combine. Returns (final: Series, per_labeler: DataFrame, errors: dict)."""
    metrics = get_quality_metrics(analyzer)
    errors: dict[str, str] = {}

    th_lab = threshold_labels(metrics, cfg, uv_scaled)
    columns = {"threshold_label": th_lab}

    bc_lab = None
    if cfg["labeling"].get("use_bombcell", True):
        bc_lab, err = bombcell_labels(analyzer, cfg, verbose)
        if err:
            errors["bombcell"] = err
        else:
            columns["bombcell_label"] = bc_lab

    ur_lab = None
    if cfg["labeling"].get("use_unitrefine", True):
        ur_lab, err = unitrefine_labels(analyzer, verbose)
        if err:
            errors["unitrefine"] = err
        else:
            columns["unitrefine_label"] = ur_lab

    demotions = amplitude_bimodality_demotions(analyzer, cfg, exclude_ids=merged_unit_ids)

    # Ensemble policy:
    #   noise <- every labeler that ran (bombcell/unitrefine) agrees on noise
    #   good  <- passes thresholds AND unitrefine-SUA (if it ran) AND not flagged
    #            non-somatic by bombcell AND not amplitude-bimodal.
    # Bombcell's good/mua boundary is recorded but does NOT gate 'good' by default:
    # its default thresholds are tuned on mouse data and are often far too strict
    # elsewhere, while its noise and non-somatic detections are robust. Set
    # labeling.bombcell_gates_good: true to enforce it anyway.
    bombcell_gates_good = bool(cfg["labeling"].get("bombcell_gates_good", False))
    final = {}
    for u in analyzer.unit_ids:
        t = th_lab.loc[u]
        b = bc_lab.loc[u] if bc_lab is not None else None
        r = ur_lab.loc[u] if ur_lab is not None else None

        ran_noise_votes = [lab for lab in (b, r) if lab is not None]
        is_noise = len(ran_noise_votes) > 0 and all("noise" in lab for lab in ran_noise_votes)

        if is_noise:
            final[u] = "noise"
            continue
        is_good = t == "good"
        if b is not None:
            if cfg["labeling"].get("demote_non_somatic", True) and "soma" in b and "non" in b:
                is_good = False
            elif bombcell_gates_good and b != "good":
                is_good = False
        if r is not None and r not in ("sua", "good"):
            is_good = False
        if u in demotions:
            is_good = False
        final[u] = "good" if is_good else "mua"

    final = pd.Series(final, name="final_label")
    per_labeler = pd.DataFrame(columns)
    per_labeler["final_label"] = final
    if verbose:
        print("[label] final counts:", final.value_counts().to_dict())
        if demotions:
            print(f"[label] {len(demotions)} units demoted for amplitude bimodality: {sorted(demotions)}")
    return final, per_labeler, {"errors": errors, "bimodality_demotions": {int(k): v for k, v in demotions.items()}}
