from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .data import load_raw_data, maybe_download_sources
from .features import build_market_snapshot, build_training_table, clean_calendar, clean_listings
from .model import save_training_outputs, train_price_model
from .roi import simulate_roi_curve


def _available_features(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_candidates = [
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "minimum_nights",
        "maximum_nights",
        "minimum_nights_avg_ntm",
        "maximum_nights_avg_ntm",
        "availability_30",
        "availability_60",
        "availability_90",
        "availability_365",
        "estimated_occupancy_l365d",
        "number_of_reviews",
        "number_of_reviews_ltm",
        "number_of_reviews_l30d",
        "reviews_per_month",
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
        "latitude",
        "longitude",
        "host_listings_count",
        "host_total_listings_count",
        "calculated_host_listings_count",
        "calculated_host_listings_count_entire_homes",
        "calculated_host_listings_count_private_rooms",
        "calculated_host_listings_count_shared_rooms",
        "host_response_rate",
        "host_acceptance_rate",
        "host_tenure_years",
        "host_is_superhost",
        "host_has_profile_pic",
        "host_identity_verified",
        "instant_bookable",
        "has_availability",
        "amenities_count",
        "month",
        "month_sin",
        "month_cos",
        "is_weekend",
    ]
    categorical_candidates = [
        "room_type",
        "property_type",
        "neighbourhood_cleansed",
        "host_response_time",
    ]

    numeric_features = [c for c in numeric_candidates if c in table.columns]
    categorical_features = [c for c in categorical_candidates if c in table.columns]

    return numeric_features, categorical_features


def _representative_listing(table: pd.DataFrame) -> pd.DataFrame:
    # Use medians/modes to create a realistic listing profile for ROI simulation.
    row: dict[str, Any] = {}
    numeric_cols = table.select_dtypes(include=["number"]).columns.tolist()

    for col in numeric_cols:
        if col in {"calendar_price", "available_flag", "listing_id"}:
            continue
        non_null = table[col].dropna()
        row[col] = float(non_null.median()) if not non_null.empty else 0.0

    for col in ["room_type", "property_type", "neighbourhood_cleansed", "host_response_time"]:
        if col in table.columns:
            mode = table[col].mode(dropna=True)
            row[col] = mode.iloc[0] if not mode.empty else "unknown"

    # July weekend is often high-demand and easy to explain in a portfolio demo.
    row["month"] = 7
    month_angle = 2 * math.pi * row["month"] / 12.0
    row["month_sin"] = float(math.sin(month_angle))
    row["month_cos"] = float(math.cos(month_angle))
    row["is_weekend"] = 1

    return pd.DataFrame([row])


def run_pipeline(
    raw_dir: Path,
    processed_dir: Path,
    artifacts_dir: Path,
    listings_url: str | None,
    calendar_url: str | None,
    max_rows: int = 200000,
    test_size: float = 0.2,
    random_state: int = 42,
    clip_target_upper_quantile: float | None = 0.995,
    investment: float = 350000,
    fixed_costs_annual: float = 14000,
    variable_cost_rate: float = 0.22,
    elasticity: float = 1.4,
) -> dict[str, Any]:
    listings_path, calendar_path = maybe_download_sources(
        raw_dir=raw_dir,
        listings_url=listings_url,
        calendar_url=calendar_url,
    )

    listings_raw, calendar_raw = load_raw_data(listings_path, calendar_path)

    listings = clean_listings(listings_raw)
    calendar = clean_calendar(calendar_raw)
    training_table = build_training_table(
        listings=listings,
        calendar=calendar,
        max_rows=max_rows,
        random_state=random_state,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    training_table.to_csv(processed_dir / "training_table.csv", index=False)

    numeric_features, categorical_features = _available_features(training_table)
    train_result = train_price_model(
        data=training_table,
        target_col="calendar_price",
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        test_size=test_size,
        random_state=random_state,
        clip_target_upper_quantile=clip_target_upper_quantile,
    )

    market_snapshot = build_market_snapshot(training_table)

    listing_for_sim = _representative_listing(training_table)
    model_features = listing_for_sim.reindex(columns=train_result.feature_columns)
    simulation_default_features: list[str] = []

    for col in model_features.columns:
        series = model_features[col]
        if series.isna().all():
            simulation_default_features.append(col)
            if pd.api.types.is_numeric_dtype(training_table[col]) if col in training_table.columns else False:
                model_features[col] = 0.0
            else:
                model_features[col] = "unknown"
        else:
            if pd.api.types.is_numeric_dtype(series):
                model_features[col] = series.fillna(0.0)
            else:
                model_features[col] = series.fillna("unknown")

    base_price = float(train_result.model.predict(model_features)[0])

    if "available_flag" in training_table.columns:
        base_occupancy = float((1.0 - training_table["available_flag"].mean()))
    else:
        base_occupancy = 0.65

    roi_curve, roi_best = simulate_roi_curve(
        base_price=base_price,
        base_occupancy=base_occupancy,
        investment=investment,
        fixed_costs_annual=fixed_costs_annual,
        variable_cost_rate=variable_cost_rate,
        elasticity=elasticity,
    )

    save_training_outputs(
        artifacts_dir=artifacts_dir,
        train_result=train_result,
        market_snapshot=market_snapshot,
        roi_curve=roi_curve,
        roi_best=roi_best,
    )

    summary = {
        "metrics": train_result.metrics,
        "roi_best": roi_best,
        "rows_used": int(len(training_table)),
        "listings_used": int(training_table["listing_id"].nunique()),
        "feature_count": int(len(train_result.feature_columns)),
    }

    if simulation_default_features:
        summary["warnings"] = [
            "ROI simulation used default values for missing representative features. "
            f"count={len(simulation_default_features)}"
        ]
        summary["simulation_default_features"] = sorted(simulation_default_features)

    return summary
