# Freight Rate Prediction — Spotter ML Assessment

**Candidate:** Sunil Kumar
**Loom Walkthrough:** [Watch the video](https://www.loom.com/share/e80ae0fcd483400f8f9f821a65837c0a)

LightGBM model predicting rate per mile × distance. Out-of-time backtest (test months Jul–Oct 2025):
**MAE $47.10 average, $40.60 on August**, the month that behaves most like Nov–Dec.
Ridge: $58.14 · naive baseline: $175.24.

## Run
    python -m venv venv
    venv\Scripts\activate
    python -m pip install -r requirements.txt
    python src/eda.py        # data audit + figures
    python src/train.py      # backtest, ablations, final model
    python src/predict.py    # validation_predictions.csv, December file, runs score.py
    python -m pytest -q      # sanity tests

## Key decisions
- **Target:** rate per mile (rate is 0.91 correlated with distance); loss weighted by distance.
- **Cleaning:** 677 label errors (1.4%) outside $1.30–$4.00/mile dropped; negative weights are sign errors (abs + flag);
  missing values filled with training statistics + flags; coordinates instead of city names (8 cities only in validation).
- **Validation:** expanding-window monthly backtest. Random K-fold would leak future data.
- **quote_signal excluded:** equals the answer in 5 months, mirrors it in 4, noise in August — and Nov–Dec match August.
  With it: $28 on Sep but $110 on Aug.
- **market_index excluded:** swings 40% over the year while rates move ~10%; worse on average (mean $78.52).

| Experiment | Jul | Aug | Sep | Oct | Mean |
|---|---|---|---|---|---|
| **LightGBM (final)** | 64.43 | **40.60** | 43.92 | 39.47 | **47.10** |
| LightGBM + quote_signal | 44.73 | 110.38 | 28.25 | 33.67 | 54.26 |
| Ridge | 70.70 | 54.18 | 56.80 | 50.89 | 58.14 |
| LightGBM + market_index | 39.80 | 65.71 | 129.14 | 79.45 | 78.52 |
| Naive baseline | 156.74 | 193.42 | 177.84 | 172.98 | 175.24 |

**Expected score on raw labels:** about $91 on August, because ~1.4% of labels are entry errors no model can predict.

