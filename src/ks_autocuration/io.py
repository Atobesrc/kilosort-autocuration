"""Loading Kilosort output folders and the raw recording they were sorted from.

Everything here is strictly read-only with respect to the input folders: the raw
recording is memory-mapped, and the Kilosort folder is only ever read. All outputs
go to the export folder chosen by the pipeline.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np

import spikeinterface.full as si


# Files that define a Kilosort/phy output folder (minimum set we require)
REQUIRED_KS_FILES = [
    "spike_times.npy",
    "spike_clusters.npy",
    "templates.npy",
    "channel_map.npy",
    "channel_positions.npy",
    "params.py",
]


class KilosortFolder:
    """Read-only view of a Kilosort output folder with the arrays we need."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        missing = [f for f in REQUIRED_KS_FILES if not (self.folder / f).exists()]
        if missing:
            raise FileNotFoundError(f"{self.folder} is not a Kilosort output folder (missing {missing})")

        self.params = self._read_params(self.folder / "params.py")
        self.spike_times = np.load(self.folder / "spike_times.npy").ravel()
        self.spike_clusters = np.load(self.folder / "spike_clusters.npy").ravel()
        st_path = self.folder / "spike_templates.npy"
        self.spike_templates = np.load(st_path).ravel() if st_path.exists() else self.spike_clusters.copy()
        self.templates = np.load(self.folder / "templates.npy")  # (n_templates, n_samples, n_channels)
        self.channel_map = np.load(self.folder / "channel_map.npy").ravel()
        self.channel_positions = np.load(self.folder / "channel_positions.npy")
        amp_path = self.folder / "amplitudes.npy"
        self.amplitudes = np.load(amp_path).ravel() if amp_path.exists() else None

        self.ks_labels = self._read_label_tsv("cluster_KSLabel.tsv")
        self.group_labels = self._read_label_tsv("cluster_group.tsv")

    @staticmethod
    def _read_params(path: Path) -> dict:
        """Parse params.py without executing it (it is plain assignments)."""
        params: dict = {}
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            m = re.match(r"^\s*(\w+)\s*=\s*(.+?)\s*$", line)
            if not m:
                continue
            key, raw = m.group(1), m.group(2)
            try:
                params[key] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                params[key] = raw
        return params

    def _read_label_tsv(self, name: str) -> dict[int, str]:
        path = self.folder / name
        labels: dict[int, str] = {}
        if not path.exists():
            return labels
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    labels[int(parts[0])] = parts[1].strip()
                except ValueError:
                    continue
        return labels

    @property
    def sampling_frequency(self) -> float:
        return float(self.params.get("sample_rate", 30000.0))

    @property
    def cluster_ids(self) -> np.ndarray:
        return np.unique(self.spike_clusters)

    def effective_labels(self) -> dict[int, str]:
        """KS labels overridden by cluster_group.tsv (phy convention)."""
        labels = dict(self.ks_labels)
        labels.update(self.group_labels)
        return labels

    def resolve_raw_path(self, override: str | Path | None = None) -> Path | None:
        """Locate the raw binary the sorting was computed from."""
        if override is not None:
            p = Path(override)
            return p if p.exists() else None
        dat = self.params.get("dat_path")
        if isinstance(dat, (list, tuple)):
            dat = dat[0] if dat else None
        if dat is None:
            return None
        p = Path(str(dat))
        if p.exists():
            return p
        # dat_path is often stale; try relative to the KS folder's parent (common layout:
        # continuous.dat next to the kilosort folder)
        candidate = self.folder.parent / p.name
        return candidate if candidate.exists() else None


def load_sorting(ks: KilosortFolder):
    """Kilosort folder -> SpikeInterface sorting (read-only)."""
    sorting = si.read_kilosort(ks.folder, keep_good_only=False)
    return sorting


def load_recording(ks: KilosortFolder, raw_path: str | Path | None, cfg: dict):
    """Build a preprocessed SpikeInterface recording for waveform extraction.

    Returns None when no raw file can be located (template-only mode).
    The probe geometry is reconstructed from channel_positions.npy/channel_map.npy,
    so this works for any Kilosort output regardless of acquisition system.
    """
    raw = ks.resolve_raw_path(raw_path)
    if raw is None:
        return None

    import probeinterface as pi
    import spikeinterface.preprocessing as spre

    n_chan = int(ks.params.get("n_channels_dat", ks.templates.shape[2]))
    dtype = str(ks.params.get("dtype", "int16"))
    rec = si.read_binary(
        raw,
        sampling_frequency=ks.sampling_frequency,
        dtype=dtype,
        num_channels=n_chan,
        file_offset=int(ks.params.get("offset", 0)),
    )

    uv_per_bit = cfg["recording"].get("uv_per_bit")
    if uv_per_bit:
        rec = spre.scale(rec, gain=float(uv_per_bit))
        # physical units from here on
        rec = spre.astype(rec, "float32")

    # Probe from Kilosort's own channel geometry
    probe = pi.Probe(ndim=2)
    positions = ks.channel_positions
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 6})
    probe.set_device_channel_indices(ks.channel_map.astype(int))
    rec.set_probe(probe, group_mode="by_probe")  # in-place in spikeinterface >= 0.105

    prep = cfg["recording"]["preprocessing"]
    if not bool(ks.params.get("hp_filtered", False)) and prep.get("highpass_hz"):
        rec = spre.highpass_filter(rec, freq_min=float(prep["highpass_hz"]))
    if prep.get("common_reference", "none") == "global_median":
        rec = spre.common_reference(rec, reference="global", operator="median")
    return rec
