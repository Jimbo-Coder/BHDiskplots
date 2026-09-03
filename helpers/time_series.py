"""Restart-safe operations shared by scalar and waveform readers."""
from __future__ import annotations

import numpy as np


def clean_time_series(data, tcol=0):
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    order = np.argsort(data[:, tcol], kind="mergesort")
    sorted_data = data[order]
    # Keep the last row for each repeated time, then return monotone time.
    rev_unique = np.unique(sorted_data[::-1, tcol], return_index=True)[1]
    keep = sorted_data.shape[0] - 1 - rev_unique
    return sorted_data[np.sort(keep)]


def merge_restart_time_series(segments, tcol=0):
    """Merge ordered restart segments, with each later segment authoritative."""
    merged = None
    for segment in segments:
        current = clean_time_series(segment, tcol=tcol)
        finite = np.isfinite(current[:, tcol])
        if not np.any(finite):
            continue
        current = current[finite]
        restart_time = float(current[0, tcol])
        if merged is None:
            merged = current
            continue
        # Overlap need not reproduce identical sample times. The later branch
        # replaces the old branch from its first finite time onward.
        merged = merged[merged[:, tcol] < restart_time]
        merged = np.concatenate((merged, current), axis=0)
    if merged is None:
        raise ValueError("No finite time samples were found")
    return clean_time_series(merged, tcol=tcol)
