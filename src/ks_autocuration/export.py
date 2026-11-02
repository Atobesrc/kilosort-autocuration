"""Stage 5: export the curated result as a Kilosort/phy-format folder.

Guarantees:
  * identical file set to the input Kilosort folder (per-spike extras copied/subset),
    so phy and any Kilosort reader opens the result unchanged;
  * untouched units keep their original cluster id; a merge keeps the id of its
    largest constituent; absorbed ids are labeled 'merged' in the tsvs (and keep
    zero spikes), so no downstream reader can double-count them;
  * spike_clusters.npy and spike_templates.npy are written identically (some
    pipelines read one, some the other);
  * templates.npy rows of merged units are replaced by the spike-count-weighted
    average of their constituents' templates (consistent whitened-space average);
  * full provenance: curation.json (SpikeInterface curation format), lineage.json,
    labels.tsv per labeler, and the exact config used.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# per-spike arrays that must stay row-aligned with spike_times.npy
PER_SPIKE_FILES = [
    "amplitudes.npy",
    "spike_positions.npy",
    "pc_features.npy",
    "spike_detection_templates.npy",
]
# files copied through verbatim if present
PASSTHROUGH_FILES = [
    "channel_map.npy", "channel_positions.npy", "channel_shanks.npy",
    "pc_feature_ind.npy", "templates_ind.npy", "similar_templates.npy",
    "whitening_mat.npy", "whitening_mat_dat.npy", "whitening_mat_inv.npy",
    "ops.npy", "params.py",
]


def build_lineage(cluster_ids: np.ndarray, spike_clusters: np.ndarray, merge_groups: list) -> dict:
    """old id -> final id (merge target = largest constituent by spike count)."""
    counts = {int(c): int(np.sum(spike_clusters == c)) for c in cluster_ids}
    mapping = {int(c): int(c) for c in cluster_ids}
    merges = []
    for group in merge_groups:
        group = [int(u) for u in group]
        target = max(group, key=lambda u: counts.get(u, 0))
        for u in group:
            mapping[u] = target
        merges.append({"constituents": sorted(group), "target": target,
                       "n_spikes": {str(u): counts.get(u, 0) for u in group}})
    return {"mapping": mapping, "merges": merges}


def censor_mask(spike_times: np.ndarray, spike_clusters: np.ndarray,
                censored_period_samples: int) -> np.ndarray:
    """True for spikes to keep: within each (final) cluster, drop later duplicates
    closer than the censored period (over-split pairs often double-count events)."""
    keep = np.ones(spike_times.size, dtype=bool)
    order = np.lexsort((spike_times, spike_clusters))
    st_o = spike_times[order]
    sc_o = spike_clusters[order]
    same_cluster = sc_o[1:] == sc_o[:-1]
    too_close = (st_o[1:] - st_o[:-1]) < censored_period_samples
    dup = np.zeros(st_o.size, dtype=bool)
    dup[1:] = same_cluster & too_close
    keep[order] = ~dup
    return keep


def merged_templates(templates: np.ndarray, lineage: dict) -> np.ndarray:
    """Replace each merge target's template row with the spike-count-weighted
    average of its constituents (all in Kilosort's own template space)."""
    out = templates.copy()
    for merge in lineage["merges"]:
        target = merge["target"]
        if target >= templates.shape[0]:
            continue
        weights, acc = 0.0, np.zeros_like(templates[0], dtype=np.float64)
        for u_str, n in merge["n_spikes"].items():
            u = int(u_str)
            if u < templates.shape[0] and n > 0:
                acc += templates[u].astype(np.float64) * n
                weights += n
        if weights > 0:
            out[target] = (acc / weights).astype(templates.dtype)
    return out


def write_label_tsv(path: Path, labels: dict[int, str], column: str = "KSLabel") -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"cluster_id\t{column}\n")
        for cid in sorted(labels):
            f.write(f"{cid}\t{labels[cid]}\n")


def export_curated(ks, merge_groups: list, final_labels: pd.Series,
                   out_folder: str | Path, cfg: dict,
                   per_labeler: pd.DataFrame | None = None,
                   extra_provenance: dict | None = None,
                   verbose: bool = True) -> Path:
    """Write the curated Kilosort-format folder. `ks` is a KilosortFolder (read-only)."""
    out = Path(out_folder)
    if out.resolve() == ks.folder.resolve():
        raise ValueError("Refusing to export onto the input Kilosort folder")
    out.mkdir(parents=True, exist_ok=True)

    lineage = build_lineage(ks.cluster_ids, ks.spike_clusters, merge_groups)
    mapping = lineage["mapping"]

    # 1) relabel spikes
    new_clusters = np.array([mapping.get(int(c), int(c)) for c in ks.spike_clusters],
                            dtype=ks.spike_clusters.dtype)

    # 2) censor duplicate spikes inside merged units
    if cfg["merge"].get("censor_duplicates", True) and lineage["merges"]:
        censored_samples = int(round(cfg["merge"]["gate"]["censored_period_ms"] / 1000.0
                                     * ks.sampling_frequency))
        merged_targets = {m["target"] for m in lineage["merges"]}
        in_merged = np.isin(new_clusters, list(merged_targets))
        keep = np.ones(new_clusters.size, dtype=bool)
        sub_keep = censor_mask(ks.spike_times[in_merged], new_clusters[in_merged], censored_samples)
        keep[np.flatnonzero(in_merged)[~sub_keep]] = False
    else:
        keep = np.ones(new_clusters.size, dtype=bool)
    n_censored = int((~keep).sum())

    np.save(out / "spike_times.npy", ks.spike_times[keep])
    np.save(out / "spike_clusters.npy", new_clusters[keep])
    np.save(out / "spike_templates.npy", new_clusters[keep])

    # 3) per-spike extras, subset to the same rows
    for name in PER_SPIKE_FILES:
        src = ks.folder / name
        if not src.exists() or not cfg["export"].get("copy_phy_extras", True):
            continue
        arr = np.load(src, mmap_mode="r")
        if arr.shape[0] == keep.size:
            np.save(out / name, np.asarray(arr[keep]))
        else:  # not spike-aligned after all; copy verbatim
            shutil.copy2(src, out / name)

    # kept_spikes.npy (KS4) refers to pre-curation rows; recompute if present
    if (ks.folder / "kept_spikes.npy").exists() and cfg["export"].get("copy_phy_extras", True):
        orig_kept = np.load(ks.folder / "kept_spikes.npy")
        if orig_kept.shape[0] == keep.size:
            np.save(out / "kept_spikes.npy", orig_kept[keep])

    # 4) templates with merged rows replaced
    np.save(out / "templates.npy", merged_templates(ks.templates, lineage))

    # 5) passthrough files
    for name in PASSTHROUGH_FILES:
        src = ks.folder / name
        if src.exists():
            shutil.copy2(src, out / name)

    # 6) labels: final labels for surviving units, 'merged' for absorbed ids
    absorbed = {int(u) for u, t in mapping.items() if u != t}
    labels_out: dict[int, str] = {}
    for cid in sorted({int(c) for c in ks.cluster_ids}):
        if cid in absorbed:
            labels_out[cid] = "merged"
        elif cid in final_labels.index:
            labels_out[cid] = str(final_labels.loc[cid])
        else:
            labels_out[cid] = ks.effective_labels().get(cid, "mua")
    write_label_tsv(out / "cluster_KSLabel.tsv", labels_out, "KSLabel")
    write_label_tsv(out / "cluster_group.tsv", labels_out, "group")
    # preserve the pre-curation labels for reference
    write_label_tsv(out / "cluster_KSLabel_original.tsv", ks.effective_labels(), "KSLabel")

    # original KS metric tsvs (ContamPct, Amplitude) still apply per original id
    for name in ("cluster_ContamPct.tsv", "cluster_Amplitude.tsv"):
        src = ks.folder / name
        if src.exists():
            shutil.copy2(src, out / name)

    # 7) provenance
    with open(out / "lineage.json", "w", encoding="utf-8") as f:
        json.dump({**lineage, "n_censored_spikes": n_censored}, f, indent=2)

    curation_json = {
        "format_version": "1",
        "unit_ids": [int(c) for c in ks.cluster_ids],
        "label_definitions": {
            "quality": {"label_options": ["good", "mua", "noise", "merged"], "exclusive": True}
        },
        "manual_labels": [
            {"unit_id": cid, "quality": [lab]} for cid, lab in labels_out.items()
        ],
        "merge_unit_groups": [sorted(m["constituents"]) for m in lineage["merges"]],
        "removed_units": [],
    }
    with open(out / "curation.json", "w", encoding="utf-8") as f:
        json.dump(curation_json, f, indent=2)

    if per_labeler is not None:
        per_labeler.to_csv(out / "cluster_labels_by_labeler.tsv", sep="\t", index_label="cluster_id")
    if extra_provenance:
        with open(out / "curation_provenance.json", "w", encoding="utf-8") as f:
            json.dump(extra_provenance, f, indent=2, default=str)

    if verbose:
        from collections import Counter
        print(f"[export] wrote {out}")
        print(f"[export] {len(lineage['merges'])} merges, {n_censored} duplicate spikes censored")
        print(f"[export] label counts: {dict(Counter(labels_out.values()))}")
    return out
