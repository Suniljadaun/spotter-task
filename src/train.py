"""Out-of-time backtest, model comparison, feature ablations and final model fit."""

import json

import joblib
import lightgbm as lgb 
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

from preprocess import(FEATURE_COLS, MARKET_COLS, QUOTE_COLS, ROOT, build_reference,
                       filter_target_outliers, load_dataset, preprocess_data)

MODEL_PATH = ROOT/ "models" / "freight_model.pkl"
METRICS_PATH = ROOT /  "reports" / "metrics.json"


TEST_MONTHS = ["2025-07", "2025-08", "2025-09", "2025-10"]
VALIDATION_LIKE = "2025-08"  # QUOTE_SIGNAL IS NOISE HERE AS IN NOV AND DEC MONTHS 


LGB_PARAMS = dict(

    objective="huber", alpha=0.5,learning_rate=0.03, n_estimators=450,
    num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, random_state=42,  verbose=-1,

)


def make_lgbm():
    return lgb.LGBMRegressor(**LGB_PARAMS)


def make_ridge():
    numeric = [c for c in FEATURE_COLS if c != "equipment"]

    encode = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["equipment"]),
    ])
    return make_pipeline(encode, Ridge(alpha=1.0))


def fit(model, X, y, distance):
    """
    fit on rate per mile , it will be weighting each load by distance so loss tracks dollars.
    """

    if hasattr(model, "steps"):
        return model.fit(X,y, ridge__sample_weight=distance)
    return model.fit(X, y, sample_weight=distance)

def dollar_mae(model, X, distance, actual_rate):
    predicted_rate = model.predict(X) * distance
    return float(np.mean(np.abs(predicted_rate - actual_rate)))


def backtest(data):
    """ it will retur one row per (test month , experiment) with mae in dollars.
    """   
    experiments = {
        "Naive: equipment median": None,
        "Ridge": (make_ridge, FEATURE_COLS),
        "LightGBM": (make_lgbm, FEATURE_COLS),
        "LightGBM + quote_signal": (make_lgbm,  FEATURE_COLS + QUOTE_COLS),
        "LightGBM + market_index": (make_lgbm, FEATURE_COLS + MARKET_COLS),

    }


    rows = []
    for month in TEST_MONTHS:
        start = pd.Timestamp(month + "-01")
        train = data[data["date"] < start]
        test = data[data["date"].dt.to_period("M") == month]

        for name, spec in experiments.items():
            if spec is None:
                rpm = train.groupby("equipment", observed=True)["rate_per_mile"].median()
                pred = test["equipment"].map(rpm).astype(float) * test["distance"]
                mae = float(np.mean(np.abs(pred - test["posted_rate"])))
            else:
                make, cols = spec
                model = fit(make(), train[cols], train["rate_per_mile"], train["distance"])
                mae = dollar_mae(model, test[cols], test["distance"], test["posted_rate"])
            rows.append({"month": month, "experiment": name, "MAE": round(mae, 2)})
            print(f"{month}  {name:<26} MAE ${mae:8.2f}")
    return pd.DataFrame(rows)


def main() -> None:
    train_raw = filter_target_outliers(load_dataset("train_test"))
    ref = build_reference(train_raw, [load_dataset("validation")])
    data = preprocess_data(train_raw, ref)

    results = backtest(data)
    table = results.pivot(index="experiment", columns="month", values="MAE")
    table["mean"] = table.mean(axis=1).round(2)
    print("\n" + table.sort_values("mean").to_string())



    final = fit(make_lgbm(), data[FEATURE_COLS], data["rate_per_mile"], data["distance"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model":final, "reference": ref,  "features": FEATURE_COLS}, MODEL_PATH)


    importance = pd.Series(final.booster_.feature_importance("gain"), index=FEATURE_COLS)
    importance = (100 * importance / importance.sum()).round(1).sort_values(ascending=False)
    print("\nFeature importance (% of gain):\n" + importance.head(8).to_string())

    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps({
        "backtest": results.to_dict("records"),
        "importance_pct": importance.to_dict(),
    }, indent=2))
    print(f"\nSaved {MODEL_PATH.name} and {METRICS_PATH.name}")


if __name__ == "__main__":
    main()

    