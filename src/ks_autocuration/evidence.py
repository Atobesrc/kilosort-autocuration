"""Stage 0: build a SortingAnalyzer and compute all evidence used by later stages."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import spikeinterface.full as si


ANALYZER_SUBDIR = "sorting_analyzer"


def build_analyzer(sorting, recording, work_dir: Path, cfg: dict, verbose: bool = True):
    """Create (or reload) a SortingAnalyzer with every extension the pipeline needs.

    The analyzer is persisted inside the work dir so re-runs skip recomputation.
    """
    ev = cfg["evidence"]
    folder = Path(work_dir) / ANALYZER_SUBDIR
    job_kwargs = dict(n_jobs=ev.get("n_jobs", 4), chunk_duration=ev.get("chunk_duration", "1s"),
                      progress_bar=verbose)

    if folder.exists():
        analyzer = si.load_sorting_analyzer(folder)
    else:
        analyzer = si.create_sorting_analyzer(
            sorting,
            recording,
            format="binary_folder",
            folder=folder,
            sparse=True,
            method="radius",
            radius_um=float(ev.get("sparsity_radius_um", 75)),
            **job_kwargs,
        )

    wanted = {
        "random_spikes": dict(max_spikes_per_unit=int(ev.get("max_spikes_per_unit", 300))),
        "waveforms": dict(ms_before=float(ev.get("ms_before", 1.0)), ms_after=float(ev.get("ms_after", 2.0))),
        "templates": dict(operators=["average", "std"]),
        "noise_levels": dict(),
        "spike_amplitudes": dict(),
        "unit_locations": dict(method="monopolar_triangulation"),
        "correlograms": dict(window_ms=100.0, bin_ms=0.5),
        "template_similarity": dict(method="l1"),
        "principal_components": dict(n_components=5, mode="by_channel_local"),
        "spike_locations": dict(),
        "template_metrics": dict(include_multi_channel_metrics=True),
        "quality_metrics": dict(metric_names=list(ev.get("quality_metrics", []))) if ev.get("quality_metrics") else dict(),
    }
    todo = {name: params for name, params in wanted.items() if not analyzer.has_extension(name)}
    if todo:
        analyzer.compute(todo, **job_kwargs)
    return analyzer


def get_quality_metrics(analyzer):
    return analyzer.get_extension("quality_metrics").get_data()


def rp_contamination(spike_train_s: np.ndarray, duration_s: float,
                     refractory_ms: float = 1.0, censored_ms: float = 0.3) -> float:
    """Refractory-period contamination estimate for a single spike train.

    Standard estimator (Hill et al. 2011 / UltraMegaSort style): the rate of ISI
    violations inside the refractory period, converted to an estimated fraction of
    contaminating spikes. Returns a value clipped to [0, 1] (1 = unresolvable).
    """
    n = spike_train_s.size
    if n < 2 or duration_s <= 0:
        return 0.0
    isis = np.diff(np.sort(spike_train_s))
    t_ref = refractory_ms / 1000.0
    t_cens = censored_ms / 1000.0
    n_viol = int(np.sum((isis > t_cens) & (isis < t_ref)))
    violation_window = 2.0 * (t_ref - t_cens) * n
    if violation_window <= 0:
        return 0.0
    total_rate = n / duration_s
    if total_rate <= 0:
        return 0.0
    viol_rate = n_viol / violation_window
    contamination = viol_rate / total_rate
    return float(np.clip(contamination, 0.0, 1.0))


def presence_vector(spike_train_s: np.ndarray, duration_s: float, n_bins: int = 40) -> np.ndarray:
    """Fraction of each session time-bin in which the unit emitted spikes."""
    counts, _ = np.histogram(spike_train_s, bins=n_bins, range=(0, duration_s))
    return (counts > 0).astype(float)
