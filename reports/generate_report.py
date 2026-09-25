"""
Generate the Spotter ML Assessment report as PDF.
Uses fpdf2 since LaTeX is not installed.
All text is ASCII/Latin-1 safe for Helvetica core font.
"""

from fpdf import FPDF
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
CHART_PATH = ROOT / "scorer_results" / "candidate_december.png"
RPM_PATH = REPORT_DIR / "figures" / "rate_per_mile.png"
QS_PATH = REPORT_DIR / "figures" / "quote_signal_by_month.png"
OUTPUT = REPORT_DIR / "assessment_report.pdf"


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(120)
            self.cell(0, 8, "Freight Rate Prediction - Spotter ML Assessment", align="R")
            self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def title_page(self):
        self.add_page()
        self.ln(60)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(6, 74, 86)
        self.cell(0, 15, "Freight Rate Prediction", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 18)
        self.set_text_color(80)
        self.cell(0, 12, "Spotter ML Assessment Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(20)
        self.set_font("Helvetica", "", 12)
        self.set_text_color(100)
        self.cell(0, 8, "Sunil Jadaun", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, "September 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    def section_title(self, title, level=1):
        self.ln(6)
        if level == 1:
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(6, 74, 86)
        else:
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(40, 80, 90)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        if level == 1:
            self.set_draw_color(6, 74, 86)
            self.set_line_width(0.6)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def body_text(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(30)
        self.multi_cell(0, 6.5, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30)
        self.multi_cell(0, 6.5, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(30)
        self.cell(6, 6.5, "-")
        self.multi_cell(0, 6.5, text)
        self.ln(1)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [(self.w - self.l_margin - self.r_margin) / len(headers)] * len(headers)

        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(6, 74, 86)
        self.set_text_color(255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()

        self.set_font("Helvetica", "", 10)
        self.set_text_color(30)
        for j, row in enumerate(rows):
            if j % 2 == 0:
                self.set_fill_color(240, 247, 248)
            else:
                self.set_fill_color(255)
            for i, cell in enumerate(row):
                align = "L" if i == 0 else "C"
                self.cell(col_widths[i], 7, str(cell), border=1, fill=True, align=align)
            self.ln()
        self.ln(4)


def build_report():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # --- Title page ---
    pdf.title_page()

    # --- 1. Executive summary ---
    pdf.add_page()
    pdf.section_title("1. Executive Summary")
    pdf.body_text(
        "This report describes a freight rate prediction model built for the Spotter ML Assessment. "
        "The task is to predict the posted_rate (in dollars) for 12,000 unseen loads in November 2025, "
        "plus a fixed-route December 2025 daily chart."
    )
    pdf.body_text(
        "The final model is a LightGBM regressor predicting rate-per-mile (then multiplied back by distance). "
        "An out-of-time expanding-window backtest over July-October 2025 yields a mean MAE of $47.10. "
        "On August -- the month whose data regime most closely matches the November-December validation period -- "
        "the MAE is $40.60, beating Ridge regression ($54.18) and a naive equipment-median baseline ($193.42)."
    )

    # --- 2. Data exploration ---
    pdf.section_title("2. Data Exploration & Quality")

    pdf.section_title("2.1 Dataset Overview", level=2)
    pdf.add_table(
        ["Dataset", "Rows", "Date Range", "Has Labels?"],
        [
            ["train_test.csv", "48,000", "Jan - Oct 2025", "Yes (posted_rate)"],
            ["validation.csv", "12,000", "Nov 2025", "No"],
            ["december_chart_inputs.csv", "31", "Dec 1-31, 2025", "No"],
        ],
        [55, 30, 50, 45],
    )
    pdf.body_text(
        "The training data spans January to October 2025 with 48,000 labeled freight loads. "
        "Each load contains pickup/delivery cities with coordinates, distance, equipment type, weight, "
        "date, market_index, quote_signal, and the target posted_rate."
    )

    pdf.section_title("2.2 Data Quality Issues Identified", level=2)

    pdf.bold_text("Label Errors (677 rows, 1.4%)")
    pdf.body_text(
        "Analysis of rate-per-mile (posted_rate / distance) reveals two distinct error clusters: "
        "one below $0.50/mile and another above $8.00/mile. These are clearly data-entry errors -- "
        "no freight rate realistically falls in those ranges. Rows outside the $1.30-$4.00/mile "
        "range were dropped (677 rows, 1.4% of training data)."
    )

    pdf.bold_text("Negative Weights (sign errors)")
    pdf.body_text(
        "Some loads have negative weight values. Since physical weight cannot be negative, these are "
        "treated as sign errors: absolute value is taken, and a binary flag (weight_was_negative) "
        "is created so the model can learn from this pattern."
    )

    pdf.bold_text("Missing Values")
    pdf.body_text(
        "Missing weight values are filled with the training-set median and flagged (weight_missing). "
        "Missing market_index values are filled with the same-day mean, then training median. "
        "Missing quote_signal is filled with training median."
    )

    pdf.bold_text("Unseen Cities in Validation")
    pdf.body_text(
        "8 cities appear only in the validation set (not in training). Rather than using city names "
        "as features (which would fail for unseen cities), the model uses raw latitude/longitude "
        "coordinates. A city-to-coordinate lookup table is built from both datasets so every city "
        "has coordinates at inference time."
    )

    # Rate-per-mile histogram
    if RPM_PATH.is_file():
        pdf.section_title("2.3 Rate-per-Mile Distribution", level=2)
        pdf.image(str(RPM_PATH), w=150)
        pdf.ln(3)
        pdf.body_text(
            "The histogram above (log scale) clearly shows the two outlier clusters outside the "
            "$1.30-$4.00/mile valid range. These 677 rows are removed before training."
        )

    # --- 3. Feature engineering ---
    pdf.add_page()
    pdf.section_title("3. Feature Engineering")

    pdf.bold_text("Target Decomposition")
    pdf.body_text(
        "Raw posted_rate is 0.91 correlated with distance -- the model would essentially just learn "
        "'longer trips cost more'. Instead, the target is rate_per_mile = posted_rate / distance. "
        "The model predicts rate-per-mile, then multiplies by distance to get dollar predictions. "
        "Training loss is weighted by distance so the objective directly minimises dollar MAE."
    )

    pdf.bold_text("Engineered Features (15 total)")
    features = [
        ("distance", "Straight from data"),
        ("weight", "Cleaned: abs() + fill missing"),
        ("equipment", "Categorical: Dry Van, Flatbed, Reefer"),
        ("pickup_lat/lon, delivery_lat/lon", "Coordinates (4 features)"),
        ("delta_lat, delta_lon", "Direction of travel"),
        ("circuity", "distance / haversine -- how indirect the route is"),
        ("weight_per_mile", "weight / distance -- load density"),
        ("dayofweek", "0=Monday to 6=Sunday"),
        ("is_weekend", "Binary flag"),
        ("weight_missing", "Binary flag for imputed weights"),
        ("weight_was_negative", "Binary flag for sign-corrected weights"),
    ]
    for feat, desc in features:
        pdf.bullet(f"{feat}: {desc}")

    pdf.ln(2)
    pdf.bold_text("Feature Importance (% of LightGBM split gain)")
    pdf.add_table(
        ["Feature", "Importance %"],
        [
            ["weight_per_mile", "50.6%"],
            ["equipment", "30.1%"],
            ["distance", "11.5%"],
            ["delivery_lat", "1.6%"],
            ["pickup_lat", "1.6%"],
            ["delivery_lon", "1.0%"],
            ["pickup_lon", "1.0%"],
            ["Other (5 features)", "2.6%"],
        ],
        [90, 90],
    )

    # --- 4. Excluded features ---
    pdf.section_title("4. Deliberately Excluded Features")

    pdf.section_title("4.1 quote_signal -- Data Leakage Risk", level=2)
    pdf.body_text(
        "quote_signal is highly suspicious: its correlation with rate-per-mile is near +1.0 for "
        "5 months (essentially the answer), but drops to near-zero in August. Its correlation with "
        "distance by month reveals the same pattern -- November and December match August's noise pattern. "
        "Including it gives $28 MAE on September (where it contains the answer) but $110 on August "
        "(where it is noise) -- exactly the months the validation data resembles."
    )

    if QS_PATH.is_file():
        pdf.image(str(QS_PATH), w=150)
        pdf.ln(3)
        pdf.body_text(
            "The chart shows corr(quote_signal, distance) by month. November and December match August's "
            "near-zero correlation -- confirming quote_signal is noise for the target period."
        )

    pdf.section_title("4.2 market_index -- Regime Shift", level=2)
    pdf.body_text(
        "market_index swings ~40% across the year while actual rate-per-mile moves only ~10%. "
        "Including it causes the model to hallucinate large rate changes (MAE $129 on September). "
        "Mean backtest MAE with market_index: $78.52 vs $47.10 without."
    )

    # --- 5. Validation strategy ---
    pdf.section_title("5. Validation Strategy")

    pdf.bold_text("Why not random K-fold?")
    pdf.body_text(
        "The data is time-series: freight rates evolve over months, and features like quote_signal "
        "change behaviour over time. Random K-fold would let the model see future data during training, "
        "producing misleadingly optimistic scores that don't reflect true forward prediction performance."
    )

    pdf.bold_text("Expanding-window monthly backtest")
    pdf.body_text(
        "Instead, four out-of-time backtests are run. For each test month (July, August, September, "
        "October 2025), the model is trained on all data before that month and evaluated on that month's "
        "data. This directly simulates the real task: predicting November loads using all data through October."
    )
    pdf.body_text(
        "August is identified as the most relevant proxy for November-December because quote_signal "
        "exhibits the same noise pattern in all three months. The final model's MAE on August ($40.60) "
        "is the most informative estimate of real-world performance."
    )

    # --- 6. Model selection ---
    pdf.add_page()
    pdf.section_title("6. Model Selection & Results")

    pdf.body_text(
        "Five experiments were compared across the four backtest months:"
    )

    pdf.add_table(
        ["Experiment", "Jul", "Aug", "Sep", "Oct", "Mean MAE"],
        [
            ["LightGBM (final)", "$64.43", "$40.60", "$43.92", "$39.47", "$47.10"],
            ["LightGBM + quote_signal", "$44.73", "$110.38", "$28.25", "$33.67", "$54.26"],
            ["Ridge Regression", "$70.70", "$54.18", "$56.80", "$50.89", "$58.14"],
            ["LightGBM + market_index", "$39.80", "$65.71", "$129.14", "$79.45", "$78.52"],
            ["Naive (equip. median)", "$156.74", "$193.42", "$177.84", "$172.98", "$175.24"],
        ],
        [55, 20, 22, 20, 20, 27],
    )

    pdf.bold_text("Why LightGBM?")
    pdf.bullet("Best mean MAE ($47.10) and best August MAE ($40.60).")
    pdf.bullet("Huber loss (alpha=0.5) for robustness to remaining outliers.")
    pdf.bullet("Native categorical support for equipment type.")
    pdf.bullet("Distance-weighted sample_weight so loss directly tracks dollar error.")
    pdf.bullet("Regularisation: 31 leaves, min 40 samples/leaf, subsampling 80%, L2 reg 1.0.")

    # --- 7. December chart ---
    pdf.section_title("7. December 2025 Predictions")

    pdf.body_text(
        "The model generates daily rate predictions for a fixed route: Lexington to Fort Wayne, "
        "360 miles, Dry Van, 32,000 lb. Only the date changes across the 31 rows."
    )

    if CHART_PATH.is_file():
        pdf.image(str(CHART_PATH), w=170)
        pdf.ln(4)

    pdf.body_text(
        "The predictions show a clear weekly periodicity (rates peak mid-week, dip on weekends) "
        "within a narrow $809-$833 range. This pattern is physically plausible: freight demand "
        "is higher during business days. The model learned day-of-week effects from the training data "
        "and applies them sensibly to unseen December dates."
    )

    # --- 8. Conclusion ---
    pdf.section_title("8. Conclusion")
    pdf.body_text(
        "The final pipeline consists of: (1) outlier removal ($1.30-$4.00/mile filter), "
        "(2) data cleaning (sign errors, missing values), (3) feature engineering (15 features), "
        "(4) LightGBM with Huber loss on rate-per-mile target."
    )
    pdf.body_text(
        "Key insight: quote_signal and market_index appear helpful on some months but are "
        "unreliable for the November-December target period. Deliberately excluding them yields "
        "the most robust model (MAE $47.10 mean, $40.60 on August)."
    )
    pdf.body_text(
        "All 12,000 validation predictions and the 31 December predictions pass the provided "
        "score.py validator. The December chart shows plausible weekly demand patterns."
    )

    # Save
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    print(f"Report saved to: {OUTPUT}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    build_report()
