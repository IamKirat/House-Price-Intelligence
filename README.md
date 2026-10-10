# House Price Intelligence: Analysis & Prediction

A Python-based data analytics project that explores Indian residential property listings, highlights location and property trends, checks data quality, and trains a baseline machine-learning model to estimate indicative listing prices.

## Project goals

- Turn semi-structured listing prices and area strings into analyzable numeric features.
- Summarize price distributions and compare locations.
- Audit missingness, duplicate rows, and out-of-range target values.
- Train and evaluate a reproducible Random Forest baseline.
- Provide an interactive Streamlit dashboard and indicative estimator.
- Translate analysis into cautious, evidence-based recommendations.

## Dataset

Expected filename: `house_prices.csv` (21 source columns; 187,531 rows in the supplied file).

**Dataset source URL:** The uploaded ZIP did not include the original download URL. Before submission, add the exact public source URL here. Do not invent or guess the source. If the dataset was downloaded from Kaggle or another repository, link to the exact dataset page and comply with its license/attribution rules.

The CSV is approximately 106 MB and exceeds GitHub's normal 100 MiB single-file limit. Keep it out of the Git repository unless using a suitable large-file solution. Provide the authorized source link in this README, or follow your instructor's approved dataset-sharing method. Place `house_prices.csv` next to the app locally, or upload it through the sidebar.

## Run locally

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Put `house_prices.csv` next to `house_price_project.py`.
5. Start the dashboard:

   ```bash
   streamlit run house_price_project.py
   ```

6. Open the local URL printed by Streamlit.

## Dashboard sections

- **Executive Overview:** listing count, median listed price, price distribution, location comparisons, and key observations.
- **Market Analysis:** comparisons by location and property attributes.
- **Data Quality:** missing values, duplicate rows, invalid/missing target values.
- **Price Estimator:** train/test evaluation and an interactive indicative estimate.

## Methodology

- Target: parsed value of `Amount(in rupees)` converted to INR (e.g. `42 Lac` → ₹4,200,000; `1.40 Cr` → ₹14,000,000).
- Features: parsed carpet/super area, BHK extracted from the title, bathroom/balcony/parking counts where available, floor information, location, status, transaction type, furnishing, facing, and ownership.
- Excluded from predictors: `Price (in rupees)`, which appears to be a price-per-unit field in sample records and could leak information about the target. It is displayed as a separate exploratory metric only.
- Numeric missing values are imputed with medians; categorical missing values use the most frequent category; unseen categories are handled at prediction time.
- A reproducible 80/20 train/test split is used. The dashboard allows training on a sample of up to 18,000 usable rows for practical runtime.
- Price ranges are filtered for analysis/modeling to reduce obvious parsing errors and extreme outliers; source data is not overwritten.

## Preliminary data audit

- Source rows: 187,531
- Source columns: 21
- Rows with parsed listed prices within ₹1 lakh–₹50 crore: 177,838
- Median parsed listed price within that range: ₹7,800,000
- Area coverage (at least one parsed carpet/super area): 57.1%
- Median absolute error on the fixed reproducible model sample: ₹1,351,881
- Root mean squared error: ₹5,617,560
- R²: 0.805

These metrics are a baseline on a random holdout sample, not a guarantee of performance in other locations or future markets. A listing price is not necessarily a completed transaction price. Results may be biased by missing data, location mix, and inconsistent listing quality.

## Files

- `house_price_project.py` — complete Streamlit app, data preparation, analysis, training and prediction flow.
- `requirements.txt` — Python dependencies.
- `README.md` — project overview and run instructions.
- `Project_Report.pdf` — report with method, findings, limitations and charts.
- `price_distribution.png`, `location_median_prices.png`, `missingness.png` — report figures.

## Limitations and responsible use

This project is for educational analytics. It is not a certified property valuation tool or investment advice. The dataset appears to contain listing data; no independent validation of actual sale prices is provided. Missingness and geographic coverage can skew comparisons. Use the estimate only as a rough model output and seek verified local data and professional advice for real decisions.

## Submission checklist

- [ ] Replace the dataset source placeholder above with the exact source URL.
- [ ] Verify the source license and attribution requirements.
- [ ] Run the app from a clean environment.
- [ ] Review the final submission form for exact filenames and report format.
- [ ] Upload code, requirements, README and report; submit the GitHub repository URL.
