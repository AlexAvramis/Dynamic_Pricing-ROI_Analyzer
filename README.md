# Dynamic Pricing & ROI Analyzer for Short-Term Rentals

An end-to-end data science project that:
- ingests public short-term rental data (Inside Airbnb),
- performs data cleaning and feature engineering,
- trains an XGBoost regression model to estimate nightly price,
- runs an ROI simulation to recommend profit-maximizing price ranges.

## Project Structure

```text
DynamicPricing_ROIAnalyzer/
  configs/default.yaml
  data/
    raw/
    processed/
  artifacts/
  app/
    streamlit_app.py
  scripts/
    run_pipeline.py
    generate_shap_report.py
  src/dpra/
    __init__.py
    data.py
    features.py
    model.py
    roi.py
    pipeline.py
    explain.py
  tests/
    test_features.py
    test_roi.py
    test_model.py
    test_config.py
  requirements.txt
  .gitignore
  README.md
```

## Data Source Options

### Option A: Inside Airbnb (recommended)
Use publicly available datasets from https://insideairbnb.com/get-the-data/

Download links are city/date-specific. Example URL pattern:

```text
https://data.insideairbnb.com/{country}/{city}/{city_slug}/{snapshot_date}/data/listings.csv.gz
https://data.insideairbnb.com/{country}/{city}/{city_slug}/{snapshot_date}/data/calendar.csv.gz
```

### Option B: Manual files
Place files directly into `data/raw/`:
- `listings.csv` or `listings.csv.gz`
- `calendar.csv` or `calendar.csv.gz`

## Setup

```bash
python -m venv .venv

# CMD
.venv\Scripts\activate

# PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Run End-to-End Pipeline

```bash
python scripts/run_pipeline.py --listings-url "<INSIDE_AIRBNB_LISTINGS_URL>" --calendar-url "<INSIDE_AIRBNB_CALENDAR_URL>"
```

Or, if files already exist in `data/raw/`:

```bash
python scripts/run_pipeline.py
```

Both scripts load defaults from `configs/default.yaml`.
You can point to a custom YAML with `--config`:

```bash
python scripts/run_pipeline.py --config configs/my_city.yaml
```

## Outputs

After running, check:
- `data/processed/training_table.csv`: cleaned and engineered modeling dataset.
- `artifacts/price_model.joblib`: trained XGBoost pipeline.
- `artifacts/metrics.json`: MAE, RMSE, and R².
- `artifacts/market_snapshot.csv`: neighborhood-level market metrics.
- `artifacts/roi_curve.csv`: ROI across candidate nightly prices.
- `artifacts/roi_best.json`: best estimated price + occupancy + ROI.

## Generate SHAP Explainability Artifacts

```bash
python scripts/generate_shap_report.py
```

This creates:
- `artifacts/shap_feature_importance.csv`
- `artifacts/shap_summary_top20.png`

## Launch Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Dashboard capabilities:
- Interactive listing feature inputs
- Real-time predicted nightly price
- ROI what-if simulation with adjustable assumptions
- SHAP feature-importance display

## Feature Engineering Highlights
- Multi-format price normalization (US `$1,338.00`, European `1.338,00`, plain `500`).
- Bathroom count extraction from free-text fields.
- Amenity count extraction from JSON-style arrays.
- Host profile features:
  - tenure in years (derived from `host_since`)
  - response rate and acceptance rate (percent parsing)
  - superhost, identity verified, profile picture flags
  - listing count metrics (total, entire homes, private rooms, shared rooms)
- Time features from calendar dates:
  - month (raw + cyclical sin/cos encoding)
  - weekend indicator
- Review score sub-dimensions (accuracy, cleanliness, checkin, communication, location, value).
- Availability windows (30/60/90/365 day).
- Estimated occupancy (last 365 days).
- Property and location categorical signals:
  - room type, property type
  - neighbourhood
  - latitude/longitude
- Log-transformed target (`TransformedTargetRegressor` with `log1p`/`expm1`) for better regression on skewed prices.
- Upper-tail outlier clipping (configurable quantile, default 99.5th percentile).

## Run Tests

```bash
pytest tests/ -v
```

## Next Portfolio Enhancements
1. Add geospatial clustering and walkability/POI features.
2. Add holiday/event calendar features for demand spikes.
3. Add a CI workflow for smoke tests and artifact checks.
