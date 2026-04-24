## Running the intern exercise

### Setup
From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

### Work on the notebook
Open and run:
- `exercise/notebook/ir_exercise.ipynb`

Implement TODOs in either:
- `exercise/src/student.py` (recommended), or directly in the notebook

### Validate your solution
From the repo root:

```bash
pytest exercise/tests
```

### Notes
- The exercise data is under `exercise/data/` and expected outputs under `exercise/expected/`.\n
