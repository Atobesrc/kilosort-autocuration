"""Stage 2: merge pass — fix over-splitting.

Candidates come from SpikeInterface's published merge presets (including the SLAy
preset when available); every candidate group then has to pass a conservative
verification gate before it is accepted:

  1. the merged spike train's refractory contamination must not exceed the worst
     constituent by more than `contamination_epsilon`, and must stay below an
     absolute ceiling;
  2. templates must be genuinely similar on their shared channels;
  3. the amplitude time course must be continuous across the junction between the
     two units' activity periods (a hard amplitude step means two different cells).

Every candidate — accepted or rejected — is logged with its scores for the QC report.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .evidence import rp_contamination


def _presence_intersection(t1: np.ndarray, t2: np.ndarray, duration_s: float,
                           n_bins: int = 50) -> float:
    """Histogram intersection of the two units' spike-mass distributions over the
    session (0 = perfectly complementary in time — the drift-split signature;
    1 = identical temporal profile). Rate-weighted, so sparse background activity
    barely moves it, unlike a binary presence vector."""
    h1, _ = np.histogram(t1, bins=n_bins, range=(0, duration_s))
    h2, _ = np.histogram(t2, bins=n_bins, range=(0, duration_s))
    s1, s2 = h1.sum(), h2.sum()
    if s1 == 0 or s2 == 0:
        return 1.0
    return float(np.minimum(h1 / s1, h2 / s2).sum())


@dataclass
class MergeDecision:
    group: list            # unit ids (original dtype preserved via list)
    accepted: bool
    reasons: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    presets: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["group"] = [int(u) for u in self.group]
        return d


def _union_find_groups(pairs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)
    groups = {}
    for x in parent:
        groups.setdefault(find(x), set()).add(x)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def collect_candidates(analyzer, cfg: dict, verbose: bool = True):
    """Union of merge candidates across the configured SI presets.

    Returns (pairs_by_preset, failed_presets)."""
    from spikeinterface.curation import compute_merge_unit_groups

    presets = list(cfg["merge"].get("presets", []))
    if cfg["merge"].get("try_slay", True):
        presets.append("slay")

    # Candidate generation is deliberately liberal (loosened template threshold):
    # precision comes from the verification gate, not from the preset chain.
    cand = cfg["merge"].get("candidates", {})
    steps_params = {"template_similarity":
                    {"template_diff_thresh": float(cand.get("template_diff_thresh", 0.4))}}

    pairs_by_preset: dict[str, list] = {}
    failed: dict[str, str] = {}
    for preset in presets:
        try:
            try:
                groups = compute_merge_unit_groups(analyzer, preset=preset, resolve_graph=True,
                                                   steps_params=steps_params)
            except (TypeError, KeyError, ValueError):
                # preset without a template_similarity step (or param renamed)
                groups = compute_merge_unit_groups(analyzer, preset=preset, resolve_graph=True)
            pairs = []
            for g in groups:
                g = list(g)
                pairs.extend((g[i], g[j]) for i in range(len(g)) for j in range(i + 1, len(g)))
            pairs_by_preset[preset] = pairs
            if verbose:
                print(f"[merge] preset '{preset}': {len(groups)} groups / {len(pairs)} pairs")
        except Exception as e:  # preset may need optional deps or fail on edge cases
            failed[preset] = f"{type(e).__name__}: {e}"
            if verbose:
                print(f"[merge] preset '{preset}' unavailable: {failed[preset]}")
    return pairs_by_preset, failed


def _template_similarity(analyzer, u1, u2) -> float:
    """Pairwise template similarity from the analyzer's template_similarity
    extension (1 = identical), which already handles sparsity correctly."""
    sim_matrix = analyzer.get_extension("template_similarity").get_data()
    unit_ids = list(analyzer.unit_ids)
    return float(sim_matrix[unit_ids.index(u1), unit_ids.index(u2)])


def _amplitude_step_z(analyzer, u1, u2) -> float:
    """|z| of the amplitude jump across the junction between the two units.

    We take each unit's spike amplitudes near the boundary between their activity
    (last chunk of the earlier unit vs first chunk of the later one) and compute a
    two-sample z of medians. Low |z| = continuous drift (same cell); high = step.
    """
    amp_ext = analyzer.get_extension("spike_amplitudes")
    amps = amp_ext.get_data(outputs="by_unit")[0]
    sorting = analyzer.sorting
    fs = sorting.sampling_frequency
    t1 = sorting.get_unit_spike_train(u1) / fs
    t2 = sorting.get_unit_spike_train(u2) / fs
    a1 = np.asarray(amps[u1], dtype=float)
    a2 = np.asarray(amps[u2], dtype=float)
    if t1.size < 20 or t2.size < 20:
        return 0.0  # too few spikes to call it a step; let contamination gate decide
    # order the two units in time by median spike time
    if np.median(t1) > np.median(t2):
        t1, t2, a1, a2 = t2, t1, a2, a1
    n = 50
    tail = a1[np.argsort(t1)][-n:]
    head = a2[np.argsort(t2)][:n]
    # robust step size in SD units: |median jump| / pooled MAD-based sigma
    mad = lambda x: np.median(np.abs(x - np.median(x))) * 1.4826
    pooled_sd = np.sqrt((mad(tail) ** 2 + mad(head) ** 2) / 2.0)
    if pooled_sd == 0:
        return 0.0
    return float(abs(np.median(head) - np.median(tail)) / pooled_sd)


def verify_and_resolve(analyzer, pairs_by_preset: dict, cfg: dict, verbose: bool = True):
    """Gate every candidate pair, then resolve accepted pairs into merge groups."""
    gate = cfg["merge"]["gate"]
    fs = analyzer.sorting.sampling_frequency
    duration_s = analyzer.get_total_duration()

    # map pair -> presets that proposed it
    proposals: dict[tuple, list] = {}
    for preset, pairs in pairs_by_preset.items():
        for a, b in pairs:
            key = tuple(sorted((a, b), key=lambda x: str(x)))
            proposals.setdefault(key, []).append(preset)

    decisions: list[MergeDecision] = []
    accepted_pairs = []
    trains = {u: analyzer.sorting.get_unit_spike_train(u) / fs for u in analyzer.unit_ids}

    for (u1, u2), presets in proposals.items():
        reasons, scores = [], {}
        c1 = rp_contamination(trains[u1], duration_s, gate["refractory_period_ms"], gate["censored_period_ms"])
        c2 = rp_contamination(trains[u2], duration_s, gate["refractory_period_ms"], gate["censored_period_ms"])
        merged = np.sort(np.concatenate([trains[u1], trains[u2]]))
        cm = rp_contamination(merged, duration_s, gate["refractory_period_ms"], gate["censored_period_ms"])
        scores.update(contam_u1=c1, contam_u2=c2, contam_merged=cm)

        sim = _template_similarity(analyzer, u1, u2)
        scores["template_similarity"] = sim

        step_z = _amplitude_step_z(analyzer, u1, u2)
        scores["amplitude_step_z"] = step_z

        # presence overlap: ~0 = complementary activity in time (drift split)
        overlap = _presence_intersection(trains[u1], trains[u2], duration_s)
        scores["presence_overlap"] = overlap
        is_drift_pair = overlap <= gate.get("presence_overlap_drift_max", 0.30)
        sim_floor = (gate.get("min_template_similarity_drift", 0.55) if is_drift_pair
                     else gate["min_template_similarity"])

        ok = True
        if cm > max(c1, c2) + gate["contamination_epsilon"]:
            ok = False
            reasons.append(f"merged contamination {cm:.3f} > max({c1:.3f},{c2:.3f}) + eps")
        if cm > gate["max_contamination_merged"]:
            ok = False
            reasons.append(f"merged contamination {cm:.3f} above absolute ceiling")
        if sim < sim_floor:
            ok = False
            reasons.append(f"template similarity {sim:.2f} below threshold "
                           f"({'drift' if is_drift_pair else 'co-active'} floor {sim_floor})")
        if step_z > gate["amplitude_step_max_z"]:
            ok = False
            reasons.append(f"amplitude step z {step_z:.1f} too large")
        if ok:
            reasons.append("passed all gates")
            accepted_pairs.append((u1, u2))
        decisions.append(MergeDecision(group=[u1, u2], accepted=ok, reasons=reasons,
                                       scores=scores, presets=sorted(set(presets))))

    merge_groups = _union_find_groups(accepted_pairs)

    # Re-verify whole groups larger than 2 (transitive merges must also be clean)
    final_groups = []
    for group in merge_groups:
        if len(group) == 2:
            final_groups.append(group)
            continue
        all_train = np.sort(np.concatenate([trains[u] for u in group]))
        cm = rp_contamination(all_train, duration_s, gate["refractory_period_ms"], gate["censored_period_ms"])
        worst = max(rp_contamination(trains[u], duration_s, gate["refractory_period_ms"],
                                     gate["censored_period_ms"]) for u in group)
        if cm <= worst + gate["contamination_epsilon"] and cm <= gate["max_contamination_merged"]:
            final_groups.append(group)
            decisions.append(MergeDecision(group=list(group), accepted=True,
                                           reasons=[f"group of {len(group)} passed re-verification"],
                                           scores={"contam_merged": cm}, presets=["group_recheck"]))
        else:
            # fall back to the single best pair within the group (highest similarity)
            best, best_sim = None, -1
            for d in decisions:
                if d.accepted and len(d.group) == 2 and set(d.group) <= set(group):
                    if d.scores.get("template_similarity", 0) > best_sim:
                        best, best_sim = d.group, d.scores["template_similarity"]
            if best is not None:
                final_groups.append(sorted(best))
            decisions.append(MergeDecision(group=list(group), accepted=False,
                                           reasons=[f"group contamination {cm:.3f} failed; kept best pair only"],
                                           scores={"contam_merged": cm}, presets=["group_recheck"]))

    if verbose:
        n_acc = len(final_groups)
        print(f"[merge] {len(proposals)} candidate pairs -> {n_acc} accepted merge groups")
    return final_groups, decisions
