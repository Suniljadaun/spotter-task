"""it will precit valication loads and the december chart inputs then run the provided scorer.
"""

import subprocess
import sys


import joblib
import numpy as np
import pandas as pd

from preprocess import DATA_DIR, ROOT, load_dataset, preprocess_data

MODEL_PATH = ROOT/ "models" /  "freight_model.pkl"
SUBMISSION_PATH = ROOT / "validation_predictions.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"


def predict_rates(bundle: dict,  df:pd.DataFrame) -> np.ndarray:
    """
    it will cleaen and featureise with saved reference , predcit the rate  per miles 
    and it will convert it to dollars."""

    features = preprocess_data(df, bundle["reference"])

    rates = bundle["model"].predict(features[bundle["features"]]) * features["distance"].to_numpy()

    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("Model produced non-finite or non-positive rates")
    return np.round(rates, 2)

def main()-> None:
    if not MODEL_PATH.is_file():
        sys.exit("No trained model found  . run python src/train.py")

    bundle = joblib.load(MODEL_PATH)


    val = load_dataset("validation")
    predictions = pd.DataFrame({"load_id": val["load_id"], "predicted_rate":predict_rates(bundle, val)})

    template = pd.read_csv(DATA_DIR/ "validation_predictions_template.csv")
    submission = template[["load_id"]].merge(predictions, on="load_id", how="left", validate="one_to_one")

    if submission["predicted_rate"].isna().any():
        raise ValueError("some template load_ids have no predicoitn")

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"wrote {SUBMISSION_PATH.name}: {len(submission):,}, rows")


    december = pd.read_csv(DECEMBER_PATH)
    december["predicted_rate"] = predict_rates(bundle, load_dataset("december_chart_inputs"))

    december.to_csv(DECEMBER_PATH, index=False)
    print(f"December range: ${december['predicted_rate'].min():.2f} - ${december['predicted_rate'].max():.2f}")



# here I'll run spotter's scorer 

    result = subprocess.run(
        [sys.executable, "score.py", "--predictions", SUBMISSION_PATH.name,
         "--december-predictions", "data/december_chart_inputs.csv"],
        cwd=ROOT, capture_output=True, text=True,
    )

    print(result.stdout.strip())
    if result.returncode != 0:
        sys.exit(f"score.py failed:\n{result.stderr.strip()}")


if __name__ == "__main__":
    main()
