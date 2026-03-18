from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def generate_shap_artifacts(
    model_path: Path,
    training_table_path: Path,
    output_dir: Path,
    target_col: str = "calendar_price",
    sample_size: int = 2000,
    random_state: int = 42,
) -> tuple[Path, Path]:
    model = joblib.load(model_path)

    if not training_table_path.exists():
        raise FileNotFoundError(f"Training table not found at {training_table_path}")

    data = pd.read_csv(training_table_path)
    if target_col in data.columns:
        data = data.drop(columns=[target_col])

    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]
    # If target transformation is used, explain the underlying tree model.
    if hasattr(regressor, "regressor_"):
        regressor = regressor.regressor_

    feature_order: list[str] = []
    for name, _, cols in preprocessor.transformers_:
        if name in {"num", "cat"}:
            feature_order.extend(list(cols))

    missing = [c for c in feature_order if c not in data.columns]
    if missing:
        raise ValueError(f"Training table missing features used by model: {missing[:10]}")

    X = data[feature_order].copy()
    if len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=random_state)

    X_processed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_processed)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "shap_feature_importance.csv"
    png_path = output_dir / "shap_summary_top20.png"

    importance.to_csv(csv_path, index=False)

    top = importance.head(20).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title("Top 20 Features by Mean Absolute SHAP")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    return csv_path, png_path
