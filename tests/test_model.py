"""Tests for model training pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dpra.model import TrainResult, train_price_model


def _make_training_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "accommodates": rng.randint(1, 6, n),
        "bedrooms": rng.randint(1, 4, n),
        "room_type": rng.choice(["Entire home/apt", "Private room"], n),
        "calendar_price": rng.uniform(50, 500, n),
    })


class TestTrainPriceModel:
    def test_returns_train_result(self):
        data = _make_training_data()
        result = train_price_model(
            data, "calendar_price",
            numeric_features=["accommodates", "bedrooms"],
            categorical_features=["room_type"],
        )
        assert isinstance(result, TrainResult)
        assert "mae" in result.metrics
        assert "rmse" in result.metrics
        assert "r2" in result.metrics
        assert isinstance(result.metrics["train_rows"], int)
        assert isinstance(result.metrics["test_rows"], int)
        assert result.feature_columns == ["accommodates", "bedrooms", "room_type"]

    def test_model_can_predict(self):
        data = _make_training_data()
        result = train_price_model(
            data, "calendar_price",
            numeric_features=["accommodates", "bedrooms"],
            categorical_features=["room_type"],
        )
        sample = pd.DataFrame({
            "accommodates": [3], "bedrooms": [2], "room_type": ["Private room"],
        })
        pred = result.model.predict(sample)
        assert len(pred) == 1
        assert pred[0] > 0

    def test_empty_features_raises(self):
        data = _make_training_data()
        with pytest.raises(ValueError, match="No model features"):
            train_price_model(data, "calendar_price", [], [])

    def test_clip_upper_quantile(self):
        data = _make_training_data(500)
        # Add a few extreme outliers.
        data.loc[0, "calendar_price"] = 100_000
        data.loc[1, "calendar_price"] = 200_000
        result = train_price_model(
            data, "calendar_price",
            numeric_features=["accommodates", "bedrooms"],
            categorical_features=["room_type"],
            clip_target_upper_quantile=0.99,
        )
        # The model should still train without outlier distortion.
        assert result.metrics["train_rows"] + result.metrics["test_rows"] < 500
