from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


@dataclass
class TrainResult:
    model: Pipeline
    metrics: dict[str, float]
    feature_columns: list[str]


def train_price_model(
    data: pd.DataFrame,
    target_col: str,
    numeric_features: list[str],
    categorical_features: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    clip_target_upper_quantile: float | None = 0.995,
) -> TrainResult:
    feature_columns = numeric_features + categorical_features

    if not feature_columns:
        raise ValueError("No model features were detected in the training table.")

    usable = data.dropna(subset=[target_col]).copy()
    if clip_target_upper_quantile is not None:
        upper = usable[target_col].quantile(clip_target_upper_quantile)
        usable = usable[usable[target_col] <= upper]

    if usable.empty:
        raise ValueError(
            f"No rows available for model training after filtering target '{target_col}'."
        )

    X = usable[feature_columns]
    y = usable[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    base_regressor = XGBRegressor(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )

    regressor = TransformedTargetRegressor(
        regressor=base_regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    return TrainResult(model=model, metrics=metrics, feature_columns=feature_columns)


def save_training_outputs(
    artifacts_dir: Path,
    train_result: TrainResult,
    market_snapshot: pd.DataFrame,
    roi_curve: pd.DataFrame,
    roi_best: dict[str, Any],
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(train_result.model, artifacts_dir / "price_model.joblib")

    metrics_path = artifacts_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(train_result.metrics, indent=2),
        encoding="utf-8",
    )

    market_snapshot.to_csv(artifacts_dir / "market_snapshot.csv", index=False)
    roi_curve.to_csv(artifacts_dir / "roi_curve.csv", index=False)

    (artifacts_dir / "roi_best.json").write_text(
        json.dumps(roi_best, indent=2),
        encoding="utf-8",
    )
