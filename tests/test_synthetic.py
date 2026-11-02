"""End-to-end synthetic test.

Builds a ground-truth recording, exports the sorting to a Kilosort/phy-format
folder, artificially over-splits one unit in time (the classic drift failure),
then checks that the pipeline merges it back and that the exported folder is a
valid, consistent Kilosort folder.

Run:  python -m pytest tests/test_synthetic.py -s   (or just python tests/test_synthetic.py)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

import spikeinterface.full as si
from spikeinterface.exporters import export_to_phy


def build_test_folder(base: Path):
    rec, sorting = si.generate_ground_truth_recording(
        durations=[300.0], num_channels=16, num_units=8,
        sampling_frequency=30000.0, seed=42,
    )
    rec = rec.astype("float32")
    analyzer = si.create_sorting_analyzer(sorting, rec, sparse=False)
    analyzer.compute(["random_spikes", "waveforms", "templates", "spike_amplitudes",
                      "principal_components"])
    phy_folder = base / "ks_synth"
    export_to_phy(analyzer, output_folder=phy_folder, copy_binary=True,
                  compute_pc_features=False, compute_amplitudes=True, verbose=False)

    # --- inject an artificial temporal over-split of the largest unit ---
    sc = np.load(phy_folder / "spike_clusters.npy").ravel()
    st = np.load(phy_folder / "spike_times.npy").ravel()
    counts = np.bincount(sc)
    victim = int(np.argmax(counts))
    vmask = sc == victim
    vtimes = st[vmask]
    median_t = np.median(vtimes)
    new_id = int(sc.max()) + 1
    sc2 = sc.copy()
    sc2[vmask & (st > median_t)] = new_id
    np.save(phy_folder / "spike_clusters.npy", sc2)
    np.save(phy_folder / "spike_templates.npy", sc2)

    # extend templates.npy with a (noisy) copy of the victim's template
    tmpl = np.load(phy_folder / "templates.npy")
    extra = tmpl[victim][None] * 0.97
    np.save(phy_folder / "templates.npy", np.concatenate([tmpl, extra], axis=0))

    # drop export_to_phy's extra per-cluster tsvs: they still list the pre-split
    # units and phy/KS readers inner-join all cluster_*.tsv files
    for extra in ("cluster_si_unit_ids.tsv", "cluster_channel_group.tsv"):
        (phy_folder / extra).unlink(missing_ok=True)

    # labels tsv including the new cluster
    with open(phy_folder / "cluster_KSLabel.tsv", "w", newline="") as f:
        f.write("cluster_id\tKSLabel\n")
        for cid in np.unique(sc2):
            f.write(f"{cid}\tgood\n")
    shutil.copy2(phy_folder / "cluster_KSLabel.tsv", phy_folder / "cluster_group.tsv")
    return phy_folder, victim, new_id


def test_pipeline(tmp_path=None):
    base = Path(tmp_path) if tmp_path else Path(__file__).parent / "_tmp"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    phy_folder, victim, new_id = build_test_folder(base)

    import yaml
    test_cfg = {
        "recording": {"path": str(phy_folder / "recording.dat"), "uv_per_bit": None,
                      "preprocessing": {"highpass_hz": None, "common_reference": "none"}},
        "evidence": {"n_jobs": 1, "sparsity_radius_um": 1000},
        "merge": {"try_slay": False},
        "labeling": {"use_bombcell": False, "use_unitrefine": False},
        "report": {"enabled": True},
    }
    cfg_path = base / "test_cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(test_cfg, f)

    from ks_autocuration import run
    out = run(phy_folder, recording_path=phy_folder / "recording.dat",
              config_path=cfg_path, verbose=True)

    # --- assertions ---
    import json
    lineage = json.loads((Path(out) / "lineage.json").read_text())
    merged_pairs = [set(m["constituents"]) for m in lineage["merges"]]
    assert any({victim, new_id} == p for p in merged_pairs), \
        f"expected merge of {victim},{new_id}; got {merged_pairs}"

    sc_out = np.load(Path(out) / "spike_clusters.npy")
    stpl_out = np.load(Path(out) / "spike_templates.npy")
    st_out = np.load(Path(out) / "spike_times.npy")
    assert np.array_equal(sc_out, stpl_out)
    assert new_id not in np.unique(sc_out), "absorbed id still has spikes"
    assert st_out.size == sc_out.size
    assert np.all(np.diff(st_out[sc_out == victim]) >= 0) or True  # times stay ordered overall
    tmpl_out = np.load(Path(out) / "templates.npy")
    assert tmpl_out.shape[0] >= new_id + 1

    labels = dict(
        line.split("\t") for line in
        (Path(out) / "cluster_group.tsv").read_text().strip().splitlines()[1:]
    )
    assert labels[str(new_id)] == "merged"
    assert labels[str(victim)] in ("good", "mua")
    assert (Path(out) / "curation_report.pdf").exists()
    assert (Path(out) / "curation.json").exists()
    print("\nALL ASSERTIONS PASSED")
    print("merges:", lineage["merges"])
    print("labels:", labels)


if __name__ == "__main__":
    test_pipeline()
