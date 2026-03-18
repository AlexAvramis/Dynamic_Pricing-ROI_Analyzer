from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dpra.roi import simulate_roi_curve


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TRAINING_TABLE = PROJECT_ROOT / "data/processed/training_table.csv"
MODEL_PATH = ARTIFACTS_DIR / "price_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
ROI_BEST_PATH = ARTIFACTS_DIR / "roi_best.json"
SHAP_CSV_PATH = ARTIFACTS_DIR / "shap_feature_importance.csv"
SHAP_PNG_PATH = ARTIFACTS_DIR / "shap_summary_top20.png"

R2_WARN_THRESHOLD = 0.40  # warn if model quality is below this


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


@st.cache_data
def load_training_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def model_feature_spec(model) -> tuple[list[str], list[str], list[str]]:
    preprocessor = model.named_steps["preprocessor"]
    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for name, _, cols in preprocessor.transformers_:
        if name == "num":
            numeric_features = list(cols)
        elif name == "cat":
            categorical_features = list(cols)

    return numeric_features, categorical_features, numeric_features + categorical_features


def build_input_row(
    data: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    values = {}

    for col in numeric_features:
        if col in data.columns and not data[col].dropna().empty:
            col_min = float(data[col].quantile(0.05))
            col_max = float(data[col].quantile(0.95))
            if col_min == col_max:
                col_min = float(data[col].min())
                col_max = float(data[col].max())
            default = float(data[col].median())
        else:
            col_min, col_max, default = 0.0, 10.0, 1.0

        step = 1.0 if max(abs(col_min), abs(col_max)) > 10 else 0.1
        values[col] = st.sidebar.slider(
            label=f"{col}",
            min_value=float(col_min),
            max_value=float(col_max) if col_max > col_min else float(col_min + step),
            value=float(default),
            step=float(step),
        )

    for col in categorical_features:
        if col in data.columns:
            options = sorted([str(v) for v in data[col].dropna().astype(str).unique().tolist()])
            options = options[:200] if len(options) > 200 else options
        else:
            options = ["unknown"]
        if not options:
            options = ["unknown"]
        values[col] = st.sidebar.selectbox(col, options=options, index=0)

    return pd.DataFrame([values])


def safe_load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Dynamic Pricing & ROI Analyzer", layout="wide")
    st.title("Dynamic Pricing & ROI Analyzer")
    st.caption("Interactive pricing recommendation and ROI simulation for short-term rentals")

    if not MODEL_PATH.exists():
        st.error(
            "Model artifact not found. Run the training pipeline first:\n\n"
            "```python scripts/run_pipeline.py```"
        )
        return

    model = load_model(MODEL_PATH)
    data = load_training_table(TRAINING_TABLE)
    metrics = safe_load_json(METRICS_PATH)
    roi_best = safe_load_json(ROI_BEST_PATH)

    r2 = metrics.get("r2", 0.0)
    train_rows = int(metrics.get("train_rows", 0))
    if r2 < R2_WARN_THRESHOLD or train_rows < 10000:
        st.warning(
            f"⚠️ Model quality is low (R²={r2:.3f}, trained on {train_rows:,} rows). "
            "Predictions and the ROI curve may not be reliable. "
            "Re-run the full pipeline to get accurate results:\n\n"
            "`python scripts/run_pipeline.py`"
        )

    st.info(
        "💱 Prices are in the **local currency of your data source** "
        "(e.g. DKK for Copenhagen, USD for New York)."
    )

    numeric_features, categorical_features, model_features = model_feature_spec(model)

    st.sidebar.header("Listing profile")
    input_row = build_input_row(data, numeric_features, categorical_features)
    input_row = input_row[model_features]

    predicted_price = float(model.predict(input_row)[0])

    st.subheader("Predicted optimal nightly price")
    st.metric(label="Model predicted nightly price (local currency)", value=f"{predicted_price:,.2f}")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("MAE", f"{metrics.get('mae', 0):.2f}")
    col_b.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
    col_c.metric("R²", f"{metrics.get('r2', 0):.3f}")
    col_d.metric("Train rows", f"{int(metrics.get('train_rows', 0)):,}")

    st.sidebar.header("ROI assumptions")
    if "available_flag" in data.columns and not data.empty:
        default_occupancy = float((1.0 - data["available_flag"].mean()))
    else:
        default_occupancy = float(roi_best.get("estimated_occupancy", 0.65) or 0.65)

    base_occupancy = st.sidebar.slider("base occupancy", 0.1, 0.98, float(default_occupancy), 0.01)
    investment = st.sidebar.number_input("investment", min_value=50000.0, value=350000.0, step=10000.0)
    fixed_costs_annual = st.sidebar.number_input(
        "fixed costs annual", min_value=1000.0, value=14000.0, step=1000.0
    )
    variable_cost_rate = st.sidebar.slider("variable cost rate", 0.05, 0.7, 0.22, 0.01)
    elasticity = st.sidebar.slider("price elasticity", 0.2, 3.0, 1.4, 0.05)

    curve, best = simulate_roi_curve(
        base_price=predicted_price,
        base_occupancy=base_occupancy,
        investment=investment,
        fixed_costs_annual=fixed_costs_annual,
        variable_cost_rate=variable_cost_rate,
        elasticity=elasticity,
    )

    st.subheader("ROI what-if curve")
    chart_data = curve.set_index("nightly_price")
    st.caption("Annual profit")
    st.line_chart(chart_data[["annual_profit"]])
    profit_col, occ_col = st.columns(2)
    with profit_col:
        st.caption("ROI")
        st.line_chart(chart_data[["roi"]])
    with occ_col:
        st.caption("Estimated occupancy")
        st.line_chart(chart_data[["estimated_occupancy"]])

    c1, c2, c3 = st.columns(3)
    c1.metric("Best nightly price (local currency)", f"{best['nightly_price']:,.2f}")
    c2.metric("Best occupancy", f"{best['estimated_occupancy'] * 100:.1f}%")
    c3.metric("Best annual ROI", f"{best['roi'] * 100:.2f}%")

    st.subheader("SHAP explainability")
    if SHAP_CSV_PATH.exists():
        shap_df = pd.read_csv(SHAP_CSV_PATH).head(20)
        st.dataframe(shap_df, use_container_width=True)
    else:
        st.info("Run scripts/generate_shap_report.py to create SHAP importance table.")

    if SHAP_PNG_PATH.exists():
        st.image(str(SHAP_PNG_PATH), caption="Top 20 SHAP features")


if __name__ == "__main__":
    main()
