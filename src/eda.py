"""Exploratory data analysis: the evidence behind every cleaning and modelling decision."""

import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
from preprocess import MAX_RPM, MIN_RPM, ROOT, load_dataset

FIG_DIR = ROOT / "reports" / "figures"


def section(title: str) -> None:
    print(f"\n{title}~")


def main()->None:
    train = load_dataset("train_test")
    val = load_dataset("validation")
    dec = load_dataset("december_chart_inputs")


    rpm = train["posted_rate"] / train["distance"]
    month = train["date`"].dt.month

    section("Shapes and date ranges")
    for name, df in [("train", train), ("validation", val), ("december", dec)]:
        print(f"{name:<11} {df.shape}  {df['date'].min().date()} -> {df['date'].max().date()}")


    section("Data quality")
    for name, df in [("train", train), ("validation", val)]:
        missing = df.isna().sum()
        print(f"{name}: missing {missing[missing > 0].to_dict()}, negative weights {(df['weight'] < 0).sum()}")


    seen = set(train["pickup"]) | set(train["delivery"])
    unseen = sorted((set(val["pickup"]) | set(val["delivery"])) - seen)
    print(f"Cities only in validation ({len(unseen)}): {unseen}")

    print(f"December inputs lack: {sorted(set(val.columns) - set(dec.columns) - {'load_id'})}")

    section("Target: rate per mile")
    bins = [0, 0.5, 1.0, MIN_RPM, 1.67, 3.2, MAX_RPM, 8, 20]
    print(pd.cut(rpm, bins).value_counts().sort_index().to_string())

    outside = ~rpm.between(MIN_RPM, MAX_RPM)
    print(f"Outside ${MIN_RPM:.2f}-${MAX_RPM:.2f}: {outside.sum()} rows ({outside.mean():.1%})")

    clean = ~outside
    section("What drives rate per mile (clean labels)")
    print(f"corr(rate, distance) = {train['posted_rate'].corr(train['distance']):.3f}")
    print(f"corr(rate per mile, distance) = {rpm[clean].corr(train.loc[clean, 'distance']):.3f}")

    print(rpm[clean].groupby(train.loc[clean, "equipment"]).mean().round(3).to_string())
    print("By day of week (0=Mon):", rpm[clean].groupby(train.loc[clean, "date"].dt.dayofweek).mean().round(3).tolist())


    section("quote_signal: correlation with rate per mile, by month")
    q_rpm = train.loc[clean, "quote_signal"].groupby(month[clean]).corr(rpm[clean])
    print(q_rpm.round(3).to_string())


    section("quote_signal: correlation with distance, by month (validation has no labels)")
    both = pd.concat([train, val])
    q_dist = both.groupby(both["date"].dt.month).apply(lambda g: g["quote_signal"].corr(g["distance"]))
    print(q_dist.round(3).to_string())



    section("market_index vs rate per mile, monthly means")
    print(pd.DataFrame({"market_index": train.groupby(month)["market_index"].mean(),
                        "rate_per_mile": rpm[clean].groupby(month[clean]).mean()}).round(3).to_string())


    save_figures(rpm, q_dist)


def save_figures(rpm: pd.Series, q_dist: pd.Series) ->None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(rpm, bins=150, range=(0,15))
    ax.set_yscale("log")
    ax.axvspan(MIN_RPM, MAX_RPM, alpha=0.1)

    ax.set(title="Rate per mile (log scale): two label-error clusters", xlabel="$ per mile", ylabel="Loads")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rate_per_mile.png", dpi=150)
    plt.close(fig)




    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(q_dist.index, q_dist.values)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set(title="corr(quote_signal, distance) by month; Nov-Dec match August",
           xlabel="Month", ylabel="Correlation", xticks=range(1, 13))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "quote_signal_by_month.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved figures to {FIG_DIR}")

if __name__ == "__main__":
    main()  

