from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_roi_curve(
    base_price: float,
    base_occupancy: float,
    investment: float,
    fixed_costs_annual: float,
    variable_cost_rate: float = 0.22,
    elasticity: float = 1.4,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if base_price <= 0:
        base_price = 1.0

    prices = np.linspace(0.7 * base_price, 1.3 * base_price, num=31)

    # Occupancy decays as price increases, controlled by elasticity.
    occupancy = base_occupancy * np.exp(-elasticity * (prices / base_price - 1.0))
    occupancy = np.clip(occupancy, 0.10, 0.98)

    annual_revenue = prices * 365.0 * occupancy
    annual_cost = fixed_costs_annual + variable_cost_rate * annual_revenue
    annual_profit = annual_revenue - annual_cost
    roi = annual_profit / investment

    curve = pd.DataFrame(
        {
            "nightly_price": prices,
            "estimated_occupancy": occupancy,
            "annual_revenue": annual_revenue,
            "annual_cost": annual_cost,
            "annual_profit": annual_profit,
            "roi": roi,
        }
    )

    best_idx = int(curve["roi"].idxmax())
    best_row = curve.iloc[best_idx].to_dict()

    return curve, {k: float(v) for k, v in best_row.items()}
