from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Spectrum:
    id: str
    label: str
    x_cm1: np.ndarray
    y: np.ndarray

    def to_json(self) -> Dict:
        return {
            "id": self.id,
            "label": self.label,
            "x_cm1": [float(v) for v in self.x_cm1.tolist()],
            "y": [float(v) for v in self.y.tolist()],
        }


def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def make_reference_spectra(grid: np.ndarray, seed: int) -> List[Spectrum]:
    """
    Produce a small library with distinct peak patterns.
    This is synthetic but shaped like IR: multiple absorption-like peaks.
    """
    rng = np.random.default_rng(seed)

    # label -> (peaks: (mu, sigma, amp), baseline_slope, baseline_offset)
    templates: Dict[str, Tuple[List[Tuple[float, float, float]], float, float]] = {
        "acetone": ([(2970, 18, 1.2), (1715, 12, 1.6), (1360, 16, 0.7)], -1e-5, 0.10),
        "ethanol": ([(3350, 28, 1.4), (2970, 20, 0.8), (1050, 14, 1.1)], 1e-5, 0.08),
        "benzene": ([(3060, 14, 1.0), (1600, 10, 1.2), (1500, 10, 1.0), (700, 10, 0.8)], 0.0, 0.07),
        "water": ([(3400, 60, 1.8), (1650, 30, 0.9)], -0.5e-5, 0.06),
        "co2": ([(2350, 10, 2.0), (667, 12, 0.7)], 0.0, 0.05),
        "ammonia": ([(3330, 35, 1.2), (1627, 16, 1.1), (930, 18, 0.6)], 0.8e-5, 0.07),
        "methane": ([(3016, 10, 1.4), (1306, 12, 0.5)], 0.0, 0.05),
        "formaldehyde": ([(2780, 16, 0.7), (1740, 12, 1.4), (1160, 14, 0.9)], -0.3e-5, 0.08),
        "acetonitrile": ([(2940, 18, 0.7), (2250, 10, 1.8), (1450, 14, 0.5)], 0.0, 0.06),
        "isopropanol": ([(3340, 32, 1.2), (2970, 20, 0.9), (1120, 14, 1.0)], 0.2e-5, 0.08),
    }

    refs: List[Spectrum] = []
    for i, (label, (peaks, slope, offset)) in enumerate(templates.items(), start=1):
        y = np.zeros_like(grid, dtype=float)
        for mu, sig, amp in peaks:
            y += amp * gaussian(grid, mu, sig)
        baseline = offset + slope * (grid - grid.mean())
        y = baseline + y

        # Mild deterministic "instrument ripple" so similarity isn't trivial.
        ripple = 0.015 * np.sin(2 * math.pi * (grid / 850.0))
        y = y * (1.0 + ripple)

        # Very small noise on references (still mostly clean).
        y = y + rng.normal(0.0, 0.003, size=y.shape)

        # Normalize into a stable range, but not strictly [0, 1]
        y = (y - y.min()) / (y.max() - y.min() + 1e-12)
        y = 0.05 + 0.95 * y

        refs.append(Spectrum(id=f"ref_{i:03d}", label=label, x_cm1=grid.copy(), y=y))

    return refs


def perturb_query(ref: Spectrum, seed: int, query_id: str) -> Spectrum:
    """Add baseline drift, noise, and a small axis shift."""
    rng = np.random.default_rng(seed)
    x = ref.x_cm1.copy()
    y = ref.y.copy()

    # small wavenumber shift (simulates slight calibration offset)
    shift_cm1 = rng.normal(0.0, 2.0)
    x_shifted = x + shift_cm1

    # baseline drift (quadratic)
    t = (x - x.mean()) / (x.max() - x.min())
    drift = (0.12 * t**2) + rng.normal(0.0, 0.005)

    # extra noise
    noise = rng.normal(0.0, 0.02, size=y.shape)

    yq = y + drift + noise

    # occasional missing segment (drop and later expect candidate interpolation to handle)
    # We remove a band in the middle by setting to NaN, then linearly fill to keep JSON finite.
    start = int(0.45 * len(x))
    end = int(0.52 * len(x))
    yq[start:end] = np.nan
    # fill for JSON (simulate pre-filled but distorted segment)
    yq = np.interp(np.arange(len(yq)), np.where(~np.isnan(yq))[0], yq[~np.isnan(yq)])

    # rescale
    yq = (yq - yq.min()) / (yq.max() - yq.min() + 1e-12)

    return Spectrum(id=query_id, label=ref.label, x_cm1=x_shifted, y=yq)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = (a - a.min()) / (a.max() - a.min() + 1e-12)
    b = (b - b.min()) / (b.max() - b.min() + 1e-12)
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den


def build_expected_rankings(
    queries: List[Spectrum], references: List[Spectrum], top_k: int = 3
) -> Dict[str, Dict]:
    """
    Baseline expected ranking: cosine similarity after min-max normalization, on the native grid.
    (Queries have small shifts; this baseline is intentionally imperfect but deterministic.)
    """
    out: Dict[str, Dict] = {}
    for q in queries:
        scores = []
        for r in references:
            # align by interpolating q onto r's grid
            yq = np.interp(r.x_cm1, q.x_cm1, q.y)
            score = cosine_similarity(yq, r.y)
            scores.append((r.label, float(score), r.id))
        scores.sort(key=lambda t: t[1], reverse=True)
        out[q.id] = {
            "true_label": q.label,
            "top_k": [{"label": lab, "score": sc, "ref_id": rid} for (lab, sc, rid) in scores[:top_k]],
        }
    return out


def build_expected_metrics() -> Dict[str, Dict]:
    """
    Expected metrics use small fixed vectors; these are used by tests to validate smoothing/similarity.
    Keep these short to avoid massive JSON.
    """
    x = np.linspace(0, 2 * math.pi, 64)
    y = np.sin(x) + 0.2 * np.cos(9 * x)
    y_noisy = y + 0.05 * np.sin(31 * x)

    return {
        "vectors": {
            "x": [float(v) for v in x.tolist()],
            "y_clean": [float(v) for v in y.tolist()],
            "y_noisy": [float(v) for v in y_noisy.tolist()],
        },
        "tolerances": {
            "smooth_rmse_max": 0.12,
            "cosine_min": 0.92,
            "pearson_min": 0.90,
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_ref = root / "data" / "reference"
    data_q = root / "data" / "queries"
    expected_dir = root / "expected"

    data_ref.mkdir(parents=True, exist_ok=True)
    data_q.mkdir(parents=True, exist_ok=True)
    expected_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: keep x increasing so simple interpolation works (np.interp expects ascending x).
    grid = np.linspace(500.0, 4000.0, 700)  # typical IR region, ascending

    refs = make_reference_spectra(grid=grid, seed=1337)
    # choose 4 queries from distinct references
    q_refs = [refs[0], refs[2], refs[4], refs[7]]
    queries = [
        perturb_query(q_refs[0], seed=202, query_id="q_001"),
        perturb_query(q_refs[1], seed=303, query_id="q_002"),
        perturb_query(q_refs[2], seed=404, query_id="q_003"),
        perturb_query(q_refs[3], seed=505, query_id="q_004"),
    ]

    for s in refs:
        (data_ref / f"{s.id}_{s.label}.json").write_text(
            json.dumps(s.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    for s in queries:
        (data_q / f"{s.id}_{s.label}.json").write_text(
            json.dumps(s.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    expected_rankings = build_expected_rankings(queries=queries, references=refs, top_k=3)
    (expected_dir / "expected_rankings.json").write_text(
        json.dumps(expected_rankings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    expected_metrics = build_expected_metrics()
    (expected_dir / "expected_metrics.json").write_text(
        json.dumps(expected_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

