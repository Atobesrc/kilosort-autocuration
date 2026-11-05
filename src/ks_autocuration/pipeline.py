"""Pipeline orchestration: Kilosort folder in -> curated Kilosort folder out."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .config import load_config, dump_config
from .io import KilosortFolder, load_sorting, load_recording
from .evidence import build_analyzer
from .merging import collect_candidates, verify_and_resolve
from .labeling import ensemble_labels
from .export import export_curated
from .report import write_report


def run(ks_folder: str | Path,
        recording_path: str | Path | None = None,
        out_folder: str | Path | None = None,
        work_dir: str | Path | None = None,
        config_path: str | Path | None = None,
        verbose: bool = True) -> Path:
    """Run the full curation pipeline.

    Parameters
    ----------
    ks_folder : Kilosort output folder (read-only input).
    recording_path : raw binary override; default comes from params.py.
    out_folder : destination; default = sibling '<ks_folder>_curated'.
    work_dir : where the SortingAnalyzer cache lives; default = '<out_folder>/_work'.
    config_path : optional YAML overriding configs/default.yaml.
    """
    t0 = time.time()
    cfg = load_config(config_path)
    ks_folder = Path(ks_folder)
    out_folder = Path(out_folder) if out_folder else ks_folder.parent / (ks_folder.name + cfg["export"]["suffix"])
    work_dir = Path(work_dir) if work_dir else out_folder.parent / (out_folder.name + "_work")
    out_folder.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[pipeline] input : {ks_folder}")
        print(f"[pipeline] output: {out_folder}")

    ks = KilosortFolder(ks_folder)
    sorting = load_sorting(ks)
    recording = load_recording(ks, recording_path, cfg)
    if recording is None:
        raise RuntimeError(
            "Raw recording not found (params.py dat_path is stale and no --recording given). "
            "Template-only mode is not implemented yet; pass --recording explicitly."
        )
    if verbose:
        print(f"[pipeline] recording: {recording.get_num_channels()} ch, "
              f"{recording.get_total_duration():.0f} s")
        print(f"[pipeline] sorting  : {len(sorting.unit_ids)} units")

    # Stage 0: evidence
    analyzer = build_analyzer(sorting, recording, work_dir, cfg, verbose)

    # Stage 1+2: merge candidates + verification gate
    provenance: dict = {"config": cfg, "input": str(ks_folder)}
    if cfg["merge"].get("enabled", True):
        pairs_by_preset, failed_presets = collect_candidates(analyzer, cfg, verbose)
        merge_groups, decisions = verify_and_resolve(analyzer, pairs_by_preset, cfg, verbose)
        provenance["merge"] = {
            "failed_presets": failed_presets,
            "decisions": [d.to_dict() for d in decisions],
        }
    else:
        merge_groups, decisions = [], []

    # Apply merges to the analyzer so labeling sees the merged units
    if merge_groups:
        # 'take_first' keeps the first id in each group; order groups so the
        # largest constituent comes first, matching the exporter's id policy
        sizes = {u: len(analyzer.sorting.get_unit_spike_train(u)) for u in analyzer.unit_ids}
        ordered_groups = [sorted(g, key=lambda u: -sizes[u]) for g in merge_groups]
        analyzer_merged = analyzer.merge_units(
            merge_unit_groups=ordered_groups,
            censor_ms=cfg["merge"]["gate"]["censored_period_ms"],
            merging_mode="soft",
            new_id_strategy="take_first",
        )
    else:
        analyzer_merged = analyzer

    # Stage 3+4: labeling on the merged population (merge targets are exempt from
    # the bimodality screen: drift merges are amplitude-bimodal by construction)
    uv_scaled = bool(cfg["recording"].get("uv_per_bit"))
    merged_targets = [g[0] for g in ordered_groups] if merge_groups else []
    final_labels, per_labeler, label_prov = ensemble_labels(analyzer_merged, cfg, uv_scaled, verbose,
                                                            merged_unit_ids=merged_targets)
    provenance["labeling"] = label_prov

    # Stage 5: export + report
    out = export_curated(ks, merge_groups, final_labels, out_folder, cfg,
                         per_labeler=per_labeler, extra_provenance=provenance, verbose=verbose)
    dump_config(cfg, out / "curation_config.yaml")

    if cfg["report"].get("enabled", True):
        report_path = write_report(out, analyzer_merged, merge_groups, decisions,
                                   final_labels, per_labeler, ks, cfg, provenance)
        if verbose:
            print(f"[pipeline] report: {report_path}")

    summary = {
        "n_original_units": int(len(ks.cluster_ids)),
        "n_merges": len(merge_groups),
        "final_label_counts": {str(k): int(v) for k, v in final_labels.value_counts().items()},
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(out / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"[pipeline] done in {summary['runtime_s']} s")
    return out
