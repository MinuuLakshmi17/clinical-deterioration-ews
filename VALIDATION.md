# Validation Record

This project is intended to be reproducible from source.

Validation commands:

```bash
python -m compileall -q app scripts tests
python scripts/train.py --output models
pytest -q
```

Optional production-style checks:

```bash
python -m pip check
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The included automated tests cover temporal feature generation, deterministic synthetic-data generation, application startup, patient listing, and end-to-end risk inference.

Important: synthetic data is used. Model metrics are demonstration metrics, not clinical validation.

## Independent review fixes (2026-10-07)

An independent review of the uploaded scaffold found and fixed four issues.
All fixes were re-verified: `pytest` (4 passed), retraining reproduces the
metrics below, and a live API smoke test returned 200 on `/health`,
`/model`, `/patients`, `/patients/{id}/risk`, and `/`.

1. **Train/serve skew in temporal features (serious).** `ModelService`
   called `prepare_matrix()` on the patient's latest row in isolation,
   so every 1h/3h delta was NaN → filled with 0 at inference time — the
   very features the model relies on most. Both `patients()` and
   `risk()` now engineer features over a trailing 4-observation window
   per patient and score the latest row. Verified: temporal drivers now
   return real non-zero values.
2. **Dead validation split.** `train.py` created a second
   `GroupShuffleSplit` whose indices were never used. Removed; the split
   is now a single 70/30 patient-grouped train/test split.
3. **Copy-paste env var.** `config.py` read `FERRITE_ENV`; renamed to
   `CLINICAL_EWS_ENV`.
4. **Relative model path in lifespan.** `train_and_save(Path("models"))`
   depended on the working directory; now uses the config-derived
   `MODEL_DIR`.

Retrained metrics after the fixes (70/30 patient-grouped split,
positive rate ~6%):

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6493 |
| PR-AUC | 0.1291 |
| Brier score | 0.0556 |
| Expected calibration error | 0.0241 |

Demonstration metrics on synthetic data, not clinical validation.
