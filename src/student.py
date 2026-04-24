from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class LoadedSpectrum:
    spectrum_id: str
    label: str
    x_cm1: np.ndarray
    y: np.ndarray
    meta: Dict


def load_spectrum_json(path: str | Path) -> LoadedSpectrum:
    """
    Load a JSON spectrum from `exercise/data/**.json`.

    Expected JSON:
      { "id": "...", "label": "...", "x_cm1": [...], "y": [...] }
    """
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))

    spectrum_id = obj["id"]
    label = obj.get("label", "")
    x = np.asarray(obj["x_cm1"], dtype=float)
    y = np.asarray(obj["y"], dtype=float)

    # TODO: implement validation (lengths, NaNs, monotonic axis).
    # Raise ValueError with a helpful message if invalid.

    return LoadedSpectrum(
        spectrum_id=spectrum_id,
        label=label,
        x_cm1=x,
        y=y,
        meta={"path": str(p)},
    )


def ensure_increasing_x(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (x_inc, y_inc) where x is strictly increasing.
    If x is decreasing, reverse x and y.
    """
    # TODO: implement.
    raise NotImplementedError


def resample_to_grid(x: np.ndarray, y: np.ndarray, grid_x: np.ndarray) -> np.ndarray:
    """
    Interpolate y(x) onto `grid_x`. Caller ensures both axes are increasing.
    """
    # TODO: implement via np.interp.
    raise NotImplementedError


def normalize_minmax(y: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    y = np.asarray(y, dtype=float)
    lo = float(np.min(y))
    hi = float(np.max(y))
    return (y - lo) / (hi - lo + 1e-12)


def smooth_savgol(y: np.ndarray, window_length: int = 21, polyorder: int = 3) -> np.ndarray:
    """Savitzky–Golay smoothing (peak-preserving)."""
    # TODO: implement (use scipy.signal.savgol_filter).
    raise NotImplementedError


def smooth_gaussian(y: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian smoothing."""
    # TODO: implement (use scipy.ndimage.gaussian_filter1d).
    raise NotImplementedError


def similarity_cosine(y1: np.ndarray, y2: np.ndarray) -> float:
    """Cosine similarity on min-max normalized vectors."""
    # TODO: implement.
    raise NotImplementedError


def similarity_pearson(y1: np.ndarray, y2: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    # TODO: implement.
    raise NotImplementedError


SimilarityMethod = Literal["cosine", "pearson"]


def rank_queries(
    queries: List[LoadedSpectrum],
    references: List[LoadedSpectrum],
    *,
    method: SimilarityMethod,
    top_k: int = 3,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Return mapping query_id -> list of (reference_label, score) sorted descending.

    Requirements:
    - interpolate query onto each reference x grid (or onto a shared grid you choose)
    - normalize appropriately for the metric
    - vectorize where feasible (avoid slow nested Python loops over points)
    """
    # TODO: implement.
    raise NotImplementedError

