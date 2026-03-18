"""Tests for ROI simulation."""
from __future__ import annotations

import pytest

from src.dpra.roi import simulate_roi_curve


class TestSimulateRoiCurve:
    def test_returns_31_rows(self):
        curve, best = simulate_roi_curve(
            base_price=100, base_occupancy=0.7,
            investment=300000, fixed_costs_annual=10000,
        )
        assert len(curve) == 31
        assert set(curve.columns) == {
            "nightly_price", "estimated_occupancy",
            "annual_revenue", "annual_cost", "annual_profit", "roi",
        }

    def test_best_has_required_keys(self):
        _, best = simulate_roi_curve(
            base_price=100, base_occupancy=0.7,
            investment=300000, fixed_costs_annual=10000,
        )
        for key in ["nightly_price", "estimated_occupancy", "annual_revenue",
                     "annual_cost", "annual_profit", "roi"]:
            assert key in best
            assert isinstance(best[key], float)

    def test_occupancy_decreases_with_price(self):
        curve, _ = simulate_roi_curve(
            base_price=100, base_occupancy=0.7,
            investment=300000, fixed_costs_annual=10000,
            elasticity=1.4,
        )
        occupancies = curve["estimated_occupancy"].tolist()
        # With positive elasticity, occupancy should decrease as price increases.
        assert occupancies[0] > occupancies[-1]

    def test_zero_base_price_no_crash(self):
        curve, best = simulate_roi_curve(
            base_price=0.0, base_occupancy=0.7,
            investment=300000, fixed_costs_annual=10000,
        )
        assert len(curve) == 31
        assert best["nightly_price"] >= 0

    def test_negative_base_price_no_crash(self):
        curve, best = simulate_roi_curve(
            base_price=-50.0, base_occupancy=0.7,
            investment=300000, fixed_costs_annual=10000,
        )
        assert len(curve) == 31

    def test_price_range_centered_on_base(self):
        curve, _ = simulate_roi_curve(
            base_price=200, base_occupancy=0.6,
            investment=100000, fixed_costs_annual=5000,
        )
        prices = curve["nightly_price"]
        assert prices.min() == pytest.approx(200 * 0.7, rel=1e-6)
        assert prices.max() == pytest.approx(200 * 1.3, rel=1e-6)
