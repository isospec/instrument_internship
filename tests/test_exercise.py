from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from exercise.src import student


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DIR = ROOT / "expected"
DATA_DIR = ROOT / "data"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_smoothing_and_similarity_sanity() -> None:
    expected = json.loads((EXPECTED_DIR / "expected_metrics.json").read_text(encoding="utf-8"))
    vec = expected["vectors"]
    tol = expected["tolerances"]

    y_clean = np.asarray(vec["y_clean"], dtype=float)
    y_noisy = np.asarray(vec["y_noisy"], dtype=float)

    y_sg = student.smooth_savgol(y_noisy, window_length=15, polyorder=3)
    y_g = student.smooth_gaussian(y_noisy, sigma=1.5)

    # Both smoothers should reduce error vs the clean signal.
    assert rmse(y_sg, y_clean) <= float(tol["smooth_rmse_max"])
    assert rmse(y_g, y_clean) <= float(tol["smooth_rmse_max"])

    # Similarity against clean should be high after smoothing.
    assert student.similarity_cosine(y_sg, y_clean) >= float(tol["cosine_min"])
    assert student.similarity_pearson(y_sg, y_clean) >= float(tol["pearson_min"])


def test_rankings_match_expected_baseline() -> None:
    expected_rankings = json.loads((EXPECTED_DIR / "expected_rankings.json").read_text(encoding="utf-8"))

    refs = [student.load_spectrum_json(p) for p in sorted((DATA_DIR / "reference").glob("*.json"))]
    queries = [student.load_spectrum_json(p) for p in sorted((DATA_DIR / "queries").glob("*.json"))]

    got = student.rank_queries(queries, refs, method="cosine", top_k=3)

    for q in queries:
        assert q.spectrum_id in got
        got_labels = [lab for (lab, _score) in got[q.spectrum_id]]
        exp_labels = [d["label"] for d in expected_rankings[q.spectrum_id]["top_k"]]
        assert got_labels == exp_labels

