"""QC report: one PDF per run with a full evidence page for every merge decision
(accepted AND rejected) plus label-change tables, so curation can be spot-checked
(and vetoed) instead of trusted blindly.

The merge evidence page answers, visually, every question the gate asked:
  * are the waveforms the same cell?       -> probe footprint + peak-channel overlays
  * when is each unit active?              -> firing-rate-over-time (presence) panel
  * is it a drift split?                   -> amplitude-vs-time and depth-vs-time tracks
  * would merging create refractory hits?  -> ACGs of each unit, their CCG, and the
                                              ACG of the MERGED train with the
                                              refractory window marked
All gate scores and thresholds are printed verbatim in the page header.

IMPORTANT: the analyzer passed in must be the PRE-merge analyzer (constituent units
must still exist as separate units).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

UNIT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
MERGED_COLOR = "#444444"


def _acg(times_s: np.ndarray, window_ms: float = 25.0, bin_ms: float = 0.5):
    """Symmetric autocorrelogram (counts) of a spike train, pure numpy."""
    t = np.sort(np.asarray(times_s, dtype=float))
    w = window_ms / 1000.0
    diffs = []
    k = 1
    while k < t.size:
        d = t[k:] - t[:-k]
        d = d[d <= w]
        if d.size == 0:
            break
        diffs.append(d)
        k += 1
    d = np.concatenate(diffs) if diffs else np.empty(0)
    edges = np.arange(0.0, window_ms + bin_ms, bin_ms)
    h, _ = np.histogram(d * 1000.0, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.concatenate([-centers[::-1], centers]), np.concatenate([h[::-1], h]), bin_ms


def _plot_acg(ax, times_s, refractory_ms, color, title):
    centers, counts, width = _acg(times_s)
    ax.bar(centers, counts, width=width, color=color, linewidth=0)
    ax.axvspan(-refractory_ms, refractory_ms, color="red", alpha=0.12, zorder=0)
    ax.axvline(-refractory_ms, color="red", lw=0.6, alpha=0.6)
    ax.axvline(refractory_ms, color="red", lw=0.6, alpha=0.6)
    ax.set_title(title, fontsize=7)
    ax.set_xlabel("lag (ms)", fontsize=6)
    ax.tick_params(labelsize=6)


def _footprint(ax, analyzer, group, n_channels=14):
    """Both units' mean templates drawn at the channel geometry around the shared
    peak region — the spatial fingerprint that should match for a true split."""
    templates = analyzer.get_extension("templates").get_data(operator="average")
    unit_ids = list(analyzer.unit_ids)
    locs = analyzer.get_channel_locations()

    wfs = [templates[unit_ids.index(u)] for u in group]  # (samples, channels) each
    score = np.max([np.ptp(w, axis=0) for w in wfs], axis=0)
    # channels NEAREST the units' peak site (raw top-ptp picks up noisy channels
    # elsewhere on the probe and flattens the plot)
    peak_ch = int(np.argmax(score))
    dist = np.linalg.norm(locs - locs[peak_ch], axis=1)
    chans = np.argsort(dist)[:n_channels]

    # geometry-aware scaling
    xs = np.unique(locs[:, 0])
    ys = np.unique(locs[:, 1])
    pitch_x = np.min(np.diff(xs)) if xs.size > 1 else 16.0
    pitch_y = np.min(np.diff(ys)) if ys.size > 1 else 20.0
    amp_max = max(float(score[chans].max()), 1e-9)
    n_samp = wfs[0].shape[0]
    t_norm = (np.arange(n_samp) / n_samp - 0.5) * 1.5 * pitch_x
    gain = 1.4 * pitch_y / amp_max

    for i, (u, w) in enumerate(zip(group, wfs)):
        for c in chans:
            ax.plot(locs[c, 0] + t_norm, locs[c, 1] + w[:, c] * gain,
                    color=UNIT_COLORS[i % len(UNIT_COLORS)], lw=0.9, alpha=0.85,
                    label=f"unit {u}" if c == chans[0] else None)
    for c in chans:
        ax.plot(locs[c, 0], locs[c, 1], marker="s", ms=2, color="0.8", zorder=0)
    ax.set_title("template footprint (channels at probe geometry)", fontsize=8)
    ax.set_xlabel("x (um)", fontsize=7)
    ax.set_ylabel("depth (um)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=7, loc="upper right")


def _peak_channel_overlay(ax, analyzer, group, ref_unit_idx):
    """Both templates on the peak channel of group[ref_unit_idx]."""
    templates = analyzer.get_extension("templates").get_data(operator="average")
    unit_ids = list(analyzer.unit_ids)
    fs = analyzer.sampling_frequency
    ref_wf = templates[unit_ids.index(group[ref_unit_idx])]
    peak_ch = int(np.argmax(np.ptp(ref_wf, axis=0)))
    t_ms = np.arange(ref_wf.shape[0]) / fs * 1000.0
    for i, u in enumerate(group):
        wf = templates[unit_ids.index(u)][:, peak_ch]
        ax.plot(t_ms, wf, color=UNIT_COLORS[i % len(UNIT_COLORS)], lw=1.2, label=f"unit {u}")
    ax.set_title(f"peak ch of unit {group[ref_unit_idx]} (ch {peak_ch})", fontsize=7)
    ax.set_xlabel("ms", fontsize=6)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6)


def _header_text(decision, group, n_spikes, target, gate, duration_s):
    s = decision.scores if decision is not None else {}
    presets = ", ".join(decision.presets) if decision is not None else "?"
    verdict = "ACCEPTED MERGE" if (decision is None or decision.accepted) else "REJECTED CANDIDATE"
    overlap = s.get("presence_overlap")
    drift_max = gate.get("presence_overlap_drift_max", 0.30)
    is_drift = overlap is not None and overlap <= drift_max
    sim_floor = gate.get("min_template_similarity_drift", 0.55) if is_drift \
        else gate.get("min_template_similarity", 0.80)

    def fmt(key, nd=3):
        v = s.get(key)
        return f"{v:.{nd}f}" if isinstance(v, float) else "n/a"

    lines = [
        f"{verdict}: units {' + '.join(str(u) for u in group)}"
        + (f"   ->  curated unit {target}" if decision is None or decision.accepted else ""),
        f"proposed by preset(s): {presets}     "
        + "   ".join(f"unit {u}: {n_spikes[u]:,} spikes" for u in group),
        "",
        "evidence (gate threshold in parentheses):",
        f"  waveform    template cosine similarity {fmt('template_similarity', 2)}   "
        f"(>= {sim_floor:.2f} {'drift' if is_drift else 'co-active'} floor)",
        f"  timing      presence overlap {fmt('presence_overlap', 2)}   "
        f"(<= {drift_max:.2f} = complementary activity -> classed as drift split)"
        if is_drift else
        f"  timing      presence overlap {fmt('presence_overlap', 2)}   "
        f"(> {drift_max:.2f} -> co-active pair; similarity floor {sim_floor:.2f} applies)",
        f"  refractory  contamination  "
        + "  ".join(f"unit {u}: {fmt('contam_u' + str(i + 1))}" for i, u in enumerate(group[:2]))
        + f"   MERGED: {fmt('contam_merged')}",
        f"              (merged must be <= max(parts) + {gate['contamination_epsilon']:.2f} "
        f"and <= {gate['max_contamination_merged']:.2f}; two cells would collide in the "
        f"+/-{gate['refractory_period_ms']:.1f} ms window)",
        f"  junction    amplitude step |z| = {fmt('amplitude_step_z', 1)}   "
        f"(<= {gate['amplitude_step_max_z']:.1f}; low = continuous drift, no hard step)",
    ]
    if decision is not None and decision.reasons:
        import textwrap
        lines += [""] + textwrap.wrap("verdict: " + "; ".join(decision.reasons),
                                      width=96, subsequent_indent="         ")
    return "\n".join(lines)


def _merge_evidence_page(fig, analyzer, group, decision, target, gate):
    """One full page of merge evidence (see module docstring)."""
    fs = analyzer.sampling_frequency
    duration = analyzer.get_total_duration()
    sorting = analyzer.sorting
    if decision is not None and len(decision.group) == len(group):
        group = list(decision.group)  # scores (contam_u1/u2) follow the decision's order
    trains = {u: sorting.get_unit_spike_train(u) / fs for u in group}
    n_spikes = {u: trains[u].size for u in group}
    accepted = decision is None or decision.accepted

    fig.text(0.06, 0.985, _header_text(decision, group, n_spikes, target, gate, duration),
             va="top", family="monospace", fontsize=7.5,
             color="#1a6b1a" if accepted else "#a61b1b")

    gs = fig.add_gridspec(5, 4, top=0.80, bottom=0.05, left=0.08, right=0.96,
                          hspace=0.75, wspace=0.40,
                          height_ratios=[2.4, 1.0, 1.0, 1.0, 1.0])

    # --- row 0: spatial evidence -------------------------------------------------
    ax_fp = fig.add_subplot(gs[0, 0:2])
    try:
        _footprint(ax_fp, analyzer, group)
    except Exception as e:
        ax_fp.text(0.05, 0.5, f"footprint err: {e}", fontsize=6, wrap=True)
    for col, ref in zip((2, 3), (0, min(1, len(group) - 1))):
        ax = fig.add_subplot(gs[0, col])
        try:
            _peak_channel_overlay(ax, analyzer, group, ref)
        except Exception as e:
            ax.text(0.05, 0.5, f"wf err: {e}", fontsize=6, wrap=True)

    # --- row 1: WHEN is each unit active ----------------------------------------
    ax_rate = fig.add_subplot(gs[1, :])
    try:
        bin_s = max(duration / 150.0, 10.0)
        edges = np.arange(0.0, duration + bin_s, bin_s)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for i, u in enumerate(group):
            h, _ = np.histogram(trains[u], bins=edges)
            ax_rate.fill_between(centers, h / bin_s, step="mid", alpha=0.45,
                                 color=UNIT_COLORS[i % len(UNIT_COLORS)], label=f"unit {u}")
        merged = np.sort(np.concatenate(list(trains.values())))
        hm, _ = np.histogram(merged, bins=edges)
        ax_rate.step(centers, hm / bin_s, where="mid", color=MERGED_COLOR, lw=0.8,
                     label="merged")
        ax_rate.set_xlim(0, duration)
        ax_rate.set_ylabel("rate (Hz)", fontsize=7)
        ax_rate.set_title("activity over the session: complementary periods = drift split; "
                          "the merged unit (gray) should look like one continuous cell",
                          fontsize=8)
        ax_rate.legend(fontsize=6, ncol=3)
        ax_rate.tick_params(labelsize=6)
    except Exception as e:
        ax_rate.text(0.05, 0.5, f"rate err: {e}", fontsize=6)

    # --- row 2: amplitude drift track --------------------------------------------
    ax_amp = fig.add_subplot(gs[2, :])
    try:
        amps = analyzer.get_extension("spike_amplitudes").get_data(outputs="by_unit")[0]
        for i, u in enumerate(group):
            ax_amp.scatter(trains[u], amps[u], s=1.5, color=UNIT_COLORS[i % len(UNIT_COLORS)],
                           alpha=0.35, label=f"unit {u}", rasterized=True)
        ax_amp.set_xlim(0, duration)
        ax_amp.set_ylabel("spike amplitude", fontsize=7)
        ax_amp.set_title("amplitude vs time: a single drifting cell shows ONE continuous "
                         "band across the junction (no step)", fontsize=8)
        ax_amp.legend(fontsize=6, markerscale=5)
        ax_amp.tick_params(labelsize=6)
    except Exception as e:
        ax_amp.text(0.05, 0.5, f"amp err: {e}", fontsize=6)

    # --- row 3: physical drift track (spike depth) --------------------------------
    ax_dep = fig.add_subplot(gs[3, :])
    try:
        locs = analyzer.get_extension("spike_locations").get_data(outputs="by_unit")[0]
        for i, u in enumerate(group):
            ax_dep.scatter(trains[u], np.asarray(locs[u]["y"], dtype=float), s=1.5,
                           color=UNIT_COLORS[i % len(UNIT_COLORS)], alpha=0.35,
                           label=f"unit {u}", rasterized=True)
        ax_dep.set_xlim(0, duration)
        ax_dep.set_ylabel("spike depth (um)", fontsize=7)
        ax_dep.set_xlabel("time (s)", fontsize=7)
        ax_dep.set_title("spike depth vs time: the probe-drift trajectory; a true split "
                         "continues the SAME track", fontsize=8)
        ax_dep.legend(fontsize=6, markerscale=5)
        ax_dep.tick_params(labelsize=6)
    except Exception as e:
        ax_dep.text(0.05, 0.5, f"depth err: {e}", fontsize=6)

    # --- row 4: refractory evidence -----------------------------------------------
    rp_ms = gate["refractory_period_ms"]
    try:
        u1, u2 = group[0], group[1]
        _plot_acg(fig.add_subplot(gs[4, 0]), trains[u1], rp_ms, UNIT_COLORS[0], f"ACG unit {u1}")
        _plot_acg(fig.add_subplot(gs[4, 1]), trains[u2], rp_ms, UNIT_COLORS[1], f"ACG unit {u2}")
        ax_ccg = fig.add_subplot(gs[4, 2])
        try:
            ccg_ext = analyzer.get_extension("correlograms")
            ccgs, bins = ccg_ext.get_data()
            unit_ids = list(analyzer.unit_ids)
            i0, i1 = unit_ids.index(u1), unit_ids.index(u2)
            centers = 0.5 * (bins[:-1] + bins[1:])
            ax_ccg.bar(centers, ccgs[i0, i1], width=bins[1] - bins[0], color="#777777",
                       linewidth=0)
            ax_ccg.set_title(f"CCG {u1} x {u2}", fontsize=7)
            ax_ccg.set_xlabel("lag (ms)", fontsize=6)
            ax_ccg.tick_params(labelsize=6)
        except Exception as e:
            ax_ccg.text(0.05, 0.5, f"ccg err: {e}", fontsize=6)
        merged = np.sort(np.concatenate(list(trains.values())))
        _plot_acg(fig.add_subplot(gs[4, 3]), merged, rp_ms, MERGED_COLOR,
                  "ACG MERGED (must stay clean\ninside the red window)")
    except Exception as e:
        fig.text(0.1, 0.08, f"correlogram row err: {e}", fontsize=6)


def write_report(out_folder: Path, analyzer, merge_groups, decisions, final_labels,
                 per_labeler, ks, cfg: dict, provenance: dict) -> Path:
    """`analyzer` must be the PRE-merge analyzer (constituents still separate)."""
    out = Path(out_folder) / "curation_report.pdf"
    max_panels = int(cfg["report"].get("max_panels", 200))
    max_rejected = int(cfg["report"].get("max_rejected_panels", 40))
    gate = cfg["merge"]["gate"]

    dec_by_group = {}
    for d in decisions:
        dec_by_group[tuple(sorted(int(u) for u in d.group))] = d

    # merge target = largest constituent (the exporter's id policy)
    sizes = {u: analyzer.sorting.get_unit_spike_train(u).size for u in analyzer.unit_ids}

    with PdfPages(out) as pdf:
        # --- summary page ---
        fig = plt.figure(figsize=(8.5, 11))
        orig_labels = ks.effective_labels()
        from collections import Counter
        orig_counts = Counter(orig_labels.values())
        new_counts = Counter(str(v) for v in final_labels.values)
        n_rejected = sum(1 for d in decisions if not d.accepted)
        lines = [
            "ks-autocuration report",
            f"input: {ks.folder}",
            "",
            f"original clusters: {len(ks.cluster_ids)}   "
            f"(good={orig_counts.get('good', 0)}, mua={orig_counts.get('mua', 0)})",
            f"accepted merges: {len(merge_groups)}   rejected candidates: {n_rejected}",
            f"final labels: " + ", ".join(f"{k}={v}" for k, v in sorted(new_counts.items())),
            "",
            "merge groups (constituents -> kept id = largest):",
        ]
        for g in merge_groups[:60]:
            target = max(g, key=lambda u: sizes.get(u, 0))
            lines.append(f"   {sorted(int(u) for u in g)} -> {int(target)}")
        if len(merge_groups) > 60:
            lines.append(f"   ... and {len(merge_groups) - 60} more")
        lines += ["", "one full evidence page follows per accepted merge, then per",
                  "rejected candidate (same layout, red header, reject reasons printed)."]
        errs = provenance.get("labeling", {}).get("errors", {})
        if errs:
            lines.append("")
            lines.append("labelers unavailable: " + ", ".join(errs))
        fig.text(0.06, 0.96, "\n".join(lines), va="top", family="monospace", fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)

        # --- one evidence page per accepted merge ---
        for n, group in enumerate(merge_groups):
            if n >= max_panels:
                break
            group = list(group)
            key = tuple(sorted(int(u) for u in group))
            d = dec_by_group.get(key)
            target = max(group, key=lambda u: sizes.get(u, 0))
            fig = plt.figure(figsize=(8.5, 11))
            try:
                _merge_evidence_page(fig, analyzer, group, d, target, gate)
            except Exception as e:
                fig.text(0.1, 0.5, f"panel failed: {e}")
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        # --- one evidence page per rejected candidate (so rejections are checkable) ---
        rejected = [d for d in decisions if not d.accepted and len(d.group) == 2]
        for n, d in enumerate(rejected):
            if n >= max_rejected:
                break
            group = list(d.group)
            fig = plt.figure(figsize=(8.5, 11))
            try:
                _merge_evidence_page(fig, analyzer, group, d, None, gate)
            except Exception as e:
                fig.text(0.1, 0.5, f"panel failed: {e}")
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
        if len(rejected) > max_rejected:
            fig = plt.figure(figsize=(8.5, 11))
            lines = [f"{len(rejected) - max_rejected} further rejected candidates "
                     "(pages capped by report.max_rejected_panels):", ""]
            for d in rejected[max_rejected:max_rejected + 80]:
                lines.append(f"  {[int(u) for u in d.group]}: {'; '.join(d.reasons)[:90]}")
            fig.text(0.06, 0.96, "\n".join(lines), va="top", family="monospace", fontsize=7)
            pdf.savefig(fig)
            plt.close(fig)

        # --- label changes ---
        fig = plt.figure(figsize=(8.5, 11))
        lines = ["label changes (cluster: original -> final):", ""]
        changes = []
        for cid in sorted(int(c) for c in ks.cluster_ids):
            orig = orig_labels.get(cid, "?")
            if cid in final_labels.index:
                new = str(final_labels.loc[cid])
                if new != orig:
                    changes.append(f"  {cid}: {orig} -> {new}")
        lines += changes[:100]
        if len(changes) > 100:
            lines.append(f"  ... and {len(changes) - 100} more")
        lines.append("")
        lines.append(f"total changed: {len(changes)}")
        fig.text(0.06, 0.96, "\n".join(lines), va="top", family="monospace", fontsize=7)
        pdf.savefig(fig)
        plt.close(fig)

    return out
