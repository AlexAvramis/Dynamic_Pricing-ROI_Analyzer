"""Tests for price parsing and feature engineering helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dpra.features import (
    _clean_price,
    _extract_bathrooms,
    _clean_percent,
    _normalize_listing_id,
    _to_flag,
    clean_calendar,
    clean_listings,
    build_training_table,
)


# ---------------------------------------------------------------------------
# _clean_price
# ---------------------------------------------------------------------------
class TestCleanPrice:
    def test_us_format(self):
        s = pd.Series(["$1,338.00", "$50.00", "$999.00", "$1,200.00"])
        result = _clean_price(s)
        assert result.tolist() == [1338.0, 50.0, 999.0, 1200.0]

    def test_european_format(self):
        s = pd.Series(["1.338,00", "1.234,56"])
        result = _clean_price(s)
        assert result.tolist() == [1338.0, 1234.56]

    def test_plain_number(self):
        s = pd.Series(["500", "0", "99.99"])
        result = _clean_price(s)
        assert result.tolist() == [500.0, 0.0, 99.99]

    def test_currency_symbols_stripped(self):
        s = pd.Series(["kr 500", "€100", "£75.50"])
        result = _clean_price(s)
        assert result.tolist() == [500.0, 100.0, 75.5]

    def test_comma_thousands_only(self):
        """1,200 with no decimal point should be 1200, not 1.2."""
        s = pd.Series(["1,200", "12,000,000"])
        result = _clean_price(s)
        assert result.tolist() == [1200.0, 12000000.0]

    def test_comma_decimal_only(self):
        """123,45 (no period) with non-thousands pattern → decimal comma."""
        s = pd.Series(["123,45", "0,99"])
        result = _clean_price(s)
        assert result.tolist() == [123.45, 0.99]

    def test_nan_and_empty(self):
        s = pd.Series([np.nan, "", "N/A", "nan"])
        result = _clean_price(s)
        assert result.isna().all()

    def test_negative(self):
        s = pd.Series(["-50.00"])
        result = _clean_price(s)
        assert result.tolist() == [-50.0]


# ---------------------------------------------------------------------------
# _normalize_listing_id
# ---------------------------------------------------------------------------
class TestNormalizeListingId:
    def test_strips_float_suffix(self):
        s = pd.Series(["12345.0", "67890.00"])
        assert _normalize_listing_id(s).tolist() == ["12345", "67890"]

    def test_plain_int_unchanged(self):
        s = pd.Series(["12345", "67890"])
        assert _normalize_listing_id(s).tolist() == ["12345", "67890"]


# ---------------------------------------------------------------------------
# _extract_bathrooms
# ---------------------------------------------------------------------------
class TestExtractBathrooms:
    def test_from_text(self):
        s = pd.Series(["2 baths", "1.5 shared baths", "0 baths"])
        result = _extract_bathrooms(s)
        assert result.tolist() == [2.0, 1.5, 0.0]

    def test_missing(self):
        s = pd.Series(["", np.nan])
        result = _extract_bathrooms(s)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# _clean_percent
# ---------------------------------------------------------------------------
class TestCleanPercent:
    def test_with_symbol(self):
        s = pd.Series(["95%", "50%", "100%"])
        result = _clean_percent(s)
        assert result.tolist() == pytest.approx([0.95, 0.50, 1.0])

    def test_plain_number(self):
        s = pd.Series(["80", "0"])
        result = _clean_percent(s)
        assert result.tolist() == pytest.approx([0.80, 0.0])


# ---------------------------------------------------------------------------
# _to_flag
# ---------------------------------------------------------------------------
class TestToFlag:
    def test_mapping(self):
        s = pd.Series(["t", "f", "True", "False", "other", np.nan])
        result = _to_flag(s)
        assert result.tolist() == [1, 0, 1, 0, 0, 0]


# ---------------------------------------------------------------------------
# clean_listings
# ---------------------------------------------------------------------------
class TestCleanListings:
    def test_renames_id_to_listing_id(self):
        df = pd.DataFrame({"id": [1, 2], "price": ["$100.00", "$200.00"]})
        result = clean_listings(df)
        assert "listing_id" in result.columns
        assert "id" not in result.columns

    def test_creates_listing_price(self):
        df = pd.DataFrame({"listing_id": [1], "price": ["$1,500.00"]})
        result = clean_listings(df)
        assert result["listing_price"].iloc[0] == 1500.0

    def test_bathroom_extraction(self):
        df = pd.DataFrame({"listing_id": [1], "bathrooms_text": ["2 baths"]})
        result = clean_listings(df)
        assert result["bathrooms"].iloc[0] == 2.0

    def test_flag_columns(self):
        df = pd.DataFrame({
            "listing_id": [1, 2],
            "host_is_superhost": ["t", "f"],
            "instant_bookable": ["True", "False"],
        })
        result = clean_listings(df)
        assert result["host_is_superhost"].tolist() == [1, 0]
        assert result["instant_bookable"].tolist() == [1, 0]

    def test_amenities_count(self):
        df = pd.DataFrame({
            "listing_id": [1],
            "amenities": ['{"Wifi","Kitchen","Heating"}'],
        })
        result = clean_listings(df)
        assert result["amenities_count"].iloc[0] == 3


# ---------------------------------------------------------------------------
# clean_calendar
# ---------------------------------------------------------------------------
class TestCleanCalendar:
    def test_basic(self):
        df = pd.DataFrame({
            "listing_id": [1, 1],
            "date": ["2025-07-01", "2025-07-02"],
            "price": ["$100.00", "$120.00"],
            "available": ["t", "f"],
        })
        result = clean_calendar(df)
        assert result["calendar_price"].tolist() == [100.0, 120.0]
        assert result["available_flag"].tolist() == [1, 0]
        assert result["month"].tolist() == [7, 7]


# ---------------------------------------------------------------------------
# build_training_table
# ---------------------------------------------------------------------------
class TestBuildTrainingTable:
    def _make_data(self):
        listings = pd.DataFrame({
            "listing_id": ["1", "2"],
            "price": ["$100.00", "$200.00"],
            "accommodates": [2, 4],
            "room_type": ["Entire home/apt", "Private room"],
        })
        calendar = pd.DataFrame({
            "listing_id": ["1", "1", "2", "2"],
            "date": ["2025-07-01", "2025-07-02", "2025-07-01", "2025-07-02"],
            "price": ["$100.00", "$110.00", "$200.00", "$210.00"],
            "available": ["t", "f", "t", "t"],
        })
        return listings, calendar

    def test_merge_produces_rows(self):
        listings, calendar = self._make_data()
        listings = clean_listings(listings)
        calendar = clean_calendar(calendar)
        table = build_training_table(listings, calendar)
        assert len(table) == 4
        assert "calendar_price" in table.columns
        assert "is_weekend" in table.columns
        assert "month_sin" in table.columns

    def test_max_rows_caps_output(self):
        listings, calendar = self._make_data()
        listings = clean_listings(listings)
        calendar = clean_calendar(calendar)
        table = build_training_table(listings, calendar, max_rows=2)
        assert len(table) == 2

    def test_fallback_to_listing_price(self):
        """When all calendar prices are NaN, fallback to listing_price."""
        listings = clean_listings(pd.DataFrame({
            "listing_id": ["1"], "price": ["$500.00"], "accommodates": [2],
        }))
        calendar = clean_calendar(pd.DataFrame({
            "listing_id": ["1"],
            "date": ["2025-07-01"],
            "price": [np.nan],
            "available": ["t"],
        }))
        table = build_training_table(listings, calendar)
        assert table["calendar_price"].iloc[0] == 500.0
