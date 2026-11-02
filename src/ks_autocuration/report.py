"""QC report: one PDF per run with a decision panel for every merge and demotion,
so curation can be spot-checked (and vetoed) instead of trusted blindly."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _merge_panel(fig, analyzer, group, decision_scores, title):
    fs = analyzer.sorting.sampling_frequency
    duration = analyzer.get_total_duration()
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    ax_wf = fig.add_subplot(gs[0, 0])
    ax_ccg = fig.add_subplot(gs[0, 1])
    ax_acg = fig.add_subplot(gs[0, 2])
    ax_amp = fig.add_subplot(gs[1, :])

    templates_ext = analyzer.get_extension("templates")
    unit_ids = list(analyzer.unit_ids)

    # waveforms on the peak channel of the first unit
    try:
        dense = templates_ext.get_data(operator="average")
        idx0 = unit_ids.index(group[0])
        peak_ch = int(np.argmax(np.ptp(dense[idx0], axis=0)))
        for i, u in enumerate(group):
            wf = dense[unit_ids.index(u)][:, peak_ch]
            ax_wf.plot(wf, color=colors[i % 10], label=f"unit {u}")
        ax_wf.set_title("template @ shared peak ch", fontsize=8)
        ax_wf.legend(fontsize=6)
    except Exception as e:
        ax_wf.text(0.1, 0.5, f"wf err: {e}", fontsize=6)

    # correlograms
    try:
        ccg_ext = analyzer.get_extension("correlograms")
        ccgs, bins = ccg_ext.get_data()
        i0, i1 = unit_ids.index(group[0]), unit_ids.index(group[1])
        centers = 0.5 * (bins[:-1] + bins[1:])
        ax_ccg.bar(centers, ccgs[i0, i1], width=bins[1] - bins[0], color="gray")
        ax_ccg.set_title(f"CCG {group[0]}x{group[1]}", fontsize=8)
        ax_acg.bar(centers, ccgs[i0, i0], width=bins[1] - bins[0], color=colors[0], alpha=0.6)
        ax_acg.bar(centers, ccgs[i1, i1], width=bins[1] - bins[0], color=colors[1], alpha=0.6)
        ax_acg.set_title("ACGs", fontsize=8)
    except Exception as e:
        ax_ccg.text(0.1, 0.5, f"ccg err: {e}", fontsize=6)

    # amplitude vs time
    try:
        amps = analyzer.get_extension("spike_amplitudes").get_data(outputs="by_unit")[0]
        for i, u in enumerate(group):
            t = analyzer.sorting.get_unit_spike_train(u) / fs
            ax_amp.scatter(t, amps[u], s=1, color=colors[i % 10], alpha=0.4, label=f"unit {u}")
        ax_amp.set_xlim(0, duration)
        ax_amp.set_xlabel("time (s)", fontsize=8)
        ax_amp.set_ylabel("amplitude", fontsize=8)
        ax_amp.legend(fontsize=6, markerscale=4)
    except Exception as e:
        ax_amp.text(0.1, 0.5, f"amp err: {e}", fontsize=6)

    score_txt = "  ".join(f"{k}={v:.3f}" for k, v in decision_scores.items() if isinstance(v, float))
    fig.suptitle(f"{title}\n{score_txt}", fontsize=9)


def write_report(out_folder: Path, analyzer, merge_groups, decisions, final_labels,
                 per_labeler, ks, cfg: dict, provenance: dict) -> Path:
    out = Path(out_folder) / "curation_report.pdf"
    max_panels = int(cfg["report"].get("max_panels", 200))

    dec_by_group = {}
    for d in decisions:
        dec_by_group[tuple(sorted(int(u) for u in d.group))] = d

    with PdfPages(out) as pdf:
        # --- summary page ---
        fig = plt.figure(figsize=(8.5, 11))
        orig_labels = ks.effective_labels()
        from collections import Counter
        orig_counts = Counter(orig_labels.values())
        new_counts = Counter(str(v) for v in final_labels.values)
        lines = [
            "ks-autocuration report",
            f"input: {ks.folder}",
            "",
            f"original clusters: {len(ks.cluster_ids)}   "
            f"(good={orig_counts.get('good', 0)}, mua={orig_counts.get('mua', 0)})",
            f"accepted merges: {len(merge_groups)}",
            f"final labels: " + ", ".join(f"{k}={v}" for k, v in sorted(new_counts.items())),
            "",
            "merge groups (target <- constituents):",
        ]
        for g in merge_groups[:60]:
            lines.append(f"   {sorted(int(u) for u in g)}")
        if len(merge_groups) > 60:
            lines.append(f"   ... and {len(merge_groups) - 60} more")
        errs = provenance.get("labeling", {}).get("errors", {})
        if errs:
            lines.append("")
            lines.append("labelers unavailable: " + ", ".join(errs))
        fig.text(0.06, 0.96, "\n".join(lines), va="top", family="monospace", fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)

        # --- one panel per accepted merge ---
        for n, group in enumerate(merge_groups):
            if n >= max_panels:
                break
            key = tuple(sorted(int(u) for u in group))
            d = dec_by_group.get(key)
            scores = d.scores if d is not None else {}
            presets = ",".join(d.presets) if d is not None else "?"
            fig = plt.figure(figsize=(10, 6))
            try:
                _merge_panel(fig, analyzer, list(group), scores,
                             f"ACCEPTED merge {sorted(int(u) for u in group)}  [{presets}]")
            except Exception as e:
                fig.text(0.1, 0.5, f"panel failed: {e}")
            pdf.savefig(fig)
            plt.close(fig)

        # --- rejected candidates, compact table ---
        rejected = [d for d in decisions if not d.accepted]
        if rejected:
            fig = plt.figure(figsize=(8.5, 11))
            lines = ["rejected merge candidates (pair: reason):", ""]
            for d in rejected[:80]:
                lines.append(f"  {[int(u) for u in d.group]}: {'; '.join(d.reasons)[:90]}")
            if len(rejected) > 80:
                lines.append(f"  ... and {len(rejected) - 80} more")
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
