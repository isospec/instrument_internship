## IR Spectra Benchmarking Exercise (Interns)

### Context
You will work on a small, self-contained IR spectra task similar to what we do when benchmarking acquisition + preprocessing for gas-phase IR spectra.

You are given a **tiny reference library** of spectra and a few **query spectra** (noisy / baseline-drifted / slightly shifted). Your job is to:
- visualize the raw data
- apply smoothing filters
- implement spectral similarity and retrieve top matches

The exercise is designed to test: correctness, engineering judgment, and efficient implementation.

### Timebox
Target: **~4 hours** (excluding environment setup).

### Deliverables
- Complete the notebook: `exercise/notebook/ir_exercise.ipynb`
- Make `pytest` pass: `pytest exercise/tests`

### Provided data
All files are in `exercise/data/` as JSON:

```json
{
  "id": "ref_003",
  "label": "acetone",
  "x_cm1": [4000.0, 3997.5, "..."],
  "y": [0.0123, 0.0121, "..."]
}
```

- **`x_cm1`**: wavenumber axis in cm⁻¹ (monotonic, but direction may differ across files)
- **`y`**: intensity / yield (arbitrary units)

### What you must implement
Implement the functions marked **TODO** in the notebook (and/or in `exercise/src/student.py` if you choose to factor code out).

#### 1) Load, validate, and align spectra
- Load JSON into numpy arrays
- Validate:
  - `x_cm1` and `y` same length
  - no NaNs
  - monotonic axis (increasing or decreasing)
- Align spectra to a **common grid** (interpolate) so similarity compares like-for-like

#### 2) Visualize raw data
At minimum produce:
- **overlay plot**: one query spectrum with its top-3 reference candidates (raw)
- **before/after plot**: raw vs smoothed spectrum for one query

#### 3) Smoothing filters
Implement at least **two** smoothing methods, with parameters exposed:
- **Savitzky–Golay** (peak-preserving)
- **Gaussian** (or moving average)

You should briefly justify default parameter choices in the notebook.

#### 4) Similarity techniques
Implement at least **two** similarity approaches:
- cosine similarity (after normalization)
- Pearson correlation (or Euclidean distance)

Then implement **one robustness improvement** (pick one):
- compare **1st derivative** spectra (reduces baseline sensitivity)
- do **cross-correlation alignment** (handles small shifts)
- peak-based similarity (match prominent peaks)

#### 5) Ranking
For each query spectrum, return **top-k matches** from the reference library with scores.

### Expected outputs (for grading)
We provide deterministic expected outputs in `exercise/expected/`:
- `expected_rankings.json`: expected top-3 labels for the baseline method
- `expected_metrics.json`: sanity checks for smoothing/similarity functions (within tolerance)

Your implementation does not need to match our internal method exactly, but it must:
- pass the provided tests
- produce sensible plots
- be robust to small perturbations

### Constraints / allowed libraries
You may use the repo dependencies (already in `requirements.txt`), including:
- numpy, pandas, scipy, scikit-learn
- matplotlib and/or plotly

Do not use external services or LLM calls.

### Evaluation rubric (what we look for)
- **Correctness**: alignment, smoothing, and similarity behave as expected; tests pass.
- **Signal processing judgment**: smoothing removes noise without destroying peaks.
- **Similarity reasoning**: explains trade-offs (normalization, baseline drift, axis shift).
- **Efficiency**: avoids slow Python loops when vectorization is straightforward.
- **Clarity**: code is readable, modular, and the notebook tells a coherent story.

