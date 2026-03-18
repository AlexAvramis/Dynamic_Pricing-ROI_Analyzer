from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_price(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    # Keep numeric separators and sign, strip currency symbols and text.
    cleaned = cleaned.str.replace(r"[^0-9,\.\-]", "", regex=True)

    has_both = cleaned.str.contains(",", na=False) & cleaned.str.contains(".", regex=False, na=False)

    # European format: 1.234,56  (period = thousands, comma = decimal)
    # Detected when the LAST period appears BEFORE the last comma.
    euro_mask = has_both & (
        cleaned.str.rfind(".").where(has_both, other=-1)
        < cleaned.str.rfind(",").where(has_both, other=-1)
    )
    cleaned.loc[euro_mask] = (
        cleaned.loc[euro_mask].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    )

    # US / InsideAirbnb format: 1,234.56  (comma = thousands, period = decimal)
    # The last comma comes BEFORE the last period → comma is thousands → remove it.
    us_thousands_mask = has_both & ~euro_mask
    cleaned.loc[us_thousands_mask] = cleaned.loc[us_thousands_mask].str.replace(",", "", regex=False)

    comma_only_mask = cleaned.str.contains(",", na=False) & ~cleaned.str.contains(
        ".", regex=False, na=False
    )

    # Thousands-style commas only: 1,200 or 12,000,000 -> remove commas.
    comma_thousands_mask = comma_only_mask & cleaned.str.match(r"^-?\d{1,3}(,\d{3})+$", na=False)
    cleaned.loc[comma_thousands_mask] = cleaned.loc[comma_thousands_mask].str.replace(",", "", regex=False)

    # Decimal comma only (no period present): 123,45 -> 123.45
    comma_decimal_mask = comma_only_mask & ~comma_thousands_mask
    cleaned.loc[comma_decimal_mask] = cleaned.loc[comma_decimal_mask].str.replace(",", ".", regex=False)

    # Remaining commas are treated as thousands separators.
    cleaned = cleaned.str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_listing_id(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip()
    # CSV readers may coerce ids to float strings like "12345.0".
    return normalized.str.replace(r"\.0+$", "", regex=True)


def _extract_bathrooms(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _clean_percent(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce") / 100.0


def _to_flag(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"t": 1, "true": 1, "f": 0, "false": 0})
    return mapped.fillna(0).astype(int)


def clean_listings(listings: pd.DataFrame) -> pd.DataFrame:
    df = listings.copy()

    if "id" in df.columns and "listing_id" not in df.columns:
        df = df.rename(columns={"id": "listing_id"})

    if "bathrooms" not in df.columns and "bathrooms_text" in df.columns:
        df["bathrooms"] = _extract_bathrooms(df["bathrooms_text"])

    if "price" in df.columns:
        df["listing_price"] = _clean_price(df["price"])

    for col in ["host_is_superhost", "instant_bookable"]:
        if col in df.columns:
            df[col] = _to_flag(df[col])

    for col in ["host_has_profile_pic", "host_identity_verified", "has_availability"]:
        if col in df.columns:
            df[col] = _to_flag(df[col])

    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col] = _clean_percent(df[col])

    if "host_since" in df.columns:
        host_since = pd.to_datetime(df["host_since"], errors="coerce")
        reference_date = pd.Timestamp("today").normalize()
        df["host_tenure_years"] = (reference_date - host_since).dt.days / 365.25

    if "amenities" in df.columns:
        df["amenities_count"] = (
            df["amenities"]
            .fillna("")
            .astype(str)
            .str.strip("{}")
            .replace("", np.nan)
            .str.split(",")
            .str.len()
            .fillna(0)
            .astype(int)
        )

    return df


def clean_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    df = calendar.copy()

    price_candidates = ["price", "adjusted_price", "nightly_price"]
    present_price_cols = [c for c in price_candidates if c in df.columns]
    if present_price_cols:
        parsed = [_clean_price(df[col]) for col in present_price_cols]
        df["calendar_price"] = parsed[0]
        for series in parsed[1:]:
            df["calendar_price"] = df["calendar_price"].fillna(series)
    else:
        df["calendar_price"] = pd.Series(np.nan, index=df.index)

    if "available" in df.columns:
        # In Inside Airbnb data, available == f means booked.
        avail = df["available"].astype(str).str.lower().map({"t": 1, "true": 1, "f": 0, "false": 0})
        df["available_flag"] = avail.fillna(0).astype(int)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month

    return df


def build_training_table(
    listings: pd.DataFrame,
    calendar: pd.DataFrame,
    max_rows: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if "listing_id" not in listings.columns:
        raise ValueError("listings data must include 'listing_id' (or raw 'id').")

    required_calendar_cols = {"listing_id", "date", "calendar_price", "month", "available_flag"}
    missing_calendar = required_calendar_cols.difference(calendar.columns)
    if missing_calendar:
        raise ValueError(f"calendar data is missing required columns: {sorted(missing_calendar)}")

    candidate_columns = [
        "listing_id",
        "listing_price",
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
        "room_type",
        "property_type",
        "neighbourhood_cleansed",
        "host_response_time",
    ]

    available_columns = [c for c in candidate_columns if c in listings.columns]
    listings_small = listings[available_columns].copy()
    calendar = calendar.copy()

    listings_small["listing_id"] = _normalize_listing_id(listings_small["listing_id"])
    calendar["listing_id"] = _normalize_listing_id(calendar["listing_id"])

    merged = calendar.merge(listings_small, on="listing_id", how="inner")
    merged = merged.dropna(subset=["date"])

    # Some snapshots contain empty calendar price fields; fallback to listing-level price.
    if merged["calendar_price"].notna().sum() == 0 and "listing_price" in merged.columns:
        merged["calendar_price"] = merged["listing_price"]

    merged = merged.dropna(subset=["calendar_price"])

    if merged.empty:
        raise ValueError(
            "No training rows after joining listings and calendar. "
            "Check data source compatibility and price parsing. "
            f"Unique listing ids: listings={listings_small['listing_id'].nunique()}, "
            f"calendar={calendar['listing_id'].nunique()}; "
            f"non-null calendar_price rows={int(calendar['calendar_price'].notna().sum())}; "
            f"non-null listing_price rows={int(listings_small['listing_price'].notna().sum()) if 'listing_price' in listings_small.columns else 0}."
        )

    merged["is_weekend"] = (merged["date"].dt.dayofweek >= 5).astype(int)
    merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12.0)
    merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12.0)

    if max_rows and len(merged) > max_rows:
        merged = merged.sample(n=max_rows, random_state=random_state)

    return merged


def build_market_snapshot(training_table: pd.DataFrame) -> pd.DataFrame:
    snapshot = (
        training_table.groupby("neighbourhood_cleansed", dropna=False)
        .agg(
            market_price=("calendar_price", "median"),
            occupancy_rate=("available_flag", lambda s: 1.0 - float(s.mean())),
            listings_count=("listing_id", "nunique"),
        )
        .reset_index()
    )
    return snapshot
