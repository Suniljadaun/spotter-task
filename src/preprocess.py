"""
in this we will do data loading , cleaning and preprocessign and feature engineering for the fright reate model  . 
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

#valid rate per  mile range for labels . EDA shows two error clusters . 

MIN_RPM  = 1.30
MAX_RPM = 4.00

EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]

EARTH_RADIUS_MILES = 3958.8

FEATURE_COLS = [
    "distance", "weight", "equipment",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "delta_lat", "delta_lon", "circuity", "weight_per_mile",
    "dayofweek", "is_weekend", "weight_missing", "weight_was_negative",
]

QUOTE_COLS = ["quote_signal"]
MARKET_COLS  = ["market_index"]




def load_dataset(name: str) -> pd.DataFrame:
    """ it will  laod the data file from data/<name>.csv """

    path = DATA_DIR/f"{name}.csv"

    if not path.is_file():
        raise FileNotFoundError(f"expected {path}")
    return pd.read_csv(path, parse_dates=["date"])


def filter_target_outliers(df: pd.DataFrame)->pd.DataFrame:
    """ we will drop rows of data whose rate is outside of valid rnage"""

    rpm = df["posted_rate"] / df["distance"]

    keep = rpm.between(MIN_RPM, MAX_RPM)
    print(f"Dropped {(~keep).sum()} rows with rate per mile outside ${MIN_RPM:.2f}-${MAX_RPM:.2f}")

    return df.loc[keep].copy()


def build_reference(train: pd.DataFrame, feature_frames=()) -> dict:
    """
    Statics will be learned hrea and we ewill apply them to every datset
    our medians come from traingin data only . """

    frames = pd.concat([train, *feature_frames], ignore_index=True)

    coords = {}
    for side in ("pickup","delivery"):
        table = frames.groupby(side)[[f"{side}_lat", f"{side}_lon"]].first()
        table.columns = ["lat","lon"]

        coords.update(table.to_dict("index"))

    return {
        "weight_median" : train["weight"].abs().median(),
        "market_index_median": train["market_index"].median(),
        "quote_signal_median":train["quote_signal"].median(),
        "coords":coords,
    }


def clean_inputs(df:pd.DataFrame, ref:dict) ->pd.DataFrame:
    """ fix the sign errors asnd we wil fill misssing values here 
    at the end we will return the data frame ."""


    data  = df.copy()

    data["weight_missing"] = data["weight"].isna().astype(int)
    data["weight_was_negative"] = (data["weight"]<0).astype(int)
    data["weight"] = data["weight"].abs().fillna(ref["weight_median"])


    for col in ("market_index", "quote_signal"):
        if col not in data.columns:
            data[col] = np.nan

    same_day = data.groupby("date")["market_index"].transform("mean")
    data["market_index"] = data["market_index"].fillna(same_day).fillna(ref["market_index_median"])
    data["quote_signal"] = data["quote_signal"].fillna(ref["quote_signal_median"])

    return data


def haversine_miles(lat1, lon1 , lat2, lon2):

    """ it will give us the great circle (straight line) distance in miles . also it 
    will work on whole columns.  """


    lat1 , lon1, lat2, lon2 = map(np.radians, (lat1, lon1,   lat2, lon2))

    a = np.sin((lat2-lat1)/2) ** 2 + np.cos(lat1)*np.cos(lat2) * np.sin((lon2-lon1)/2)**2

    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def add_features(data: pd.DataFrame, ref:dict)->pd.DataFrame:

    """ we will add some model features here to a cleaned datafreame .. and return new datafreme. 
    """

    data = data.copy()

    coords = pd.DataFrame.from_dict(ref["coords"], orient='index')

    for side in ("pickup", "delivery"):
        for axis in("lat", "lon"):
            col = f"{side}_{axis}"

            lookup = data[side].map(coords[axis])

            data[col] = data[col].fillna(lookup) if col in data.columns else lookup

    if data[["pickup_lat", "delivery_lat"]].isna().any().any():
        raise ValueError(" some of the cities have no known coordinates")


    data["delta_lat"] = data["delivery_lat"] - data["pickup_lat"]
    data["delta_lon"]  = data["delivery_lon"]- data["pickup_lon"]

    straight = haversine_miles(data["pickup_lat"],  data["pickup_lon"], data["delivery_lat"], data["delivery_lon"])

    data["circuity"] = data["distance"] /  straight.clip(lower=1.0) 

    data["weight_per_mile"] = data["weight"] / data["distance"]

    data["dayofweek"] = data["date"].dt.dayofweek
    data["is_weekend"] =  (data["dayofweek"] >=5 ).astype(int)

    data["equipment"] = pd.Categorical(data["equipment"], categories=EQUIPMENT_TYPES)

    if "posted_rate" in data.columns:
        data["rate_per_mile"] = data["posted_rate"] / data["distance"]
    return data



def preprocess_data(df: pd.DataFrame, ref:dict) -> pd.DataFrame:
    """ its a full pipeline for any dataset . clean then add features."""

    return add_features(clean_inputs(df, ref), ref)



if __name__ == "__main__":
    train = filter_target_outliers(load_dataset("train_test"))                    
    ref = build_reference(train, [load_dataset("validation")])                    

    for name in ("train_test", "validation", "december_chart_inputs"):            
        frame = train if name == "train_test" else load_dataset(name)              
        out = preprocess_data(frame, ref)                                          
        missing = out[FEATURE_COLS].isna().sum().sum()                             
        print(f"{name:<22} rows={len(out):>6}  missing feature values={missing}")  
