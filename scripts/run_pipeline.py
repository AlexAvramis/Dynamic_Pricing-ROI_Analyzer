from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dpra.pipeline import run_pipeline

CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return {}


def parse_args() -> argparse.Namespace:
    cfg = _load_config()
    paths = cfg.get("paths", {})
    data_sources = cfg.get("data_sources", {})
    model = cfg.get("model", {})
    roi = cfg.get("roi", {})

    parser = argparse.ArgumentParser(
        description="Dynamic Pricing and ROI Analyzer for short-term rentals"
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / paths.get("raw_dir", "data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / paths.get("processed_dir", "data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / paths.get("artifacts_dir", "artifacts"))

    parser.add_argument("--listings-url", type=str, default=data_sources.get("listings_url") or None)
    parser.add_argument("--calendar-url", type=str, default=data_sources.get("calendar_url") or None)

    parser.add_argument("--max-rows", type=int, default=model.get("max_rows", 200000))
    parser.add_argument("--test-size", type=float, default=model.get("test_size", 0.2))
    parser.add_argument("--random-state", type=int, default=model.get("random_state", 42))
    parser.add_argument("--clip-target-upper-quantile", type=float, default=model.get("clip_target_upper_quantile", 0.995))

    parser.add_argument("--investment", type=float, default=roi.get("investment", 350000))
    parser.add_argument("--fixed-costs-annual", type=float, default=roi.get("fixed_costs_annual", 14000))
    parser.add_argument("--variable-cost-rate", type=float, default=roi.get("variable_cost_rate", 0.22))
    parser.add_argument("--elasticity", type=float, default=roi.get("elasticity", 1.4))

    args = parser.parse_args()

    # Support --config override: reload from the specified file.
    if args.config is not None:
        override = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        o_paths = override.get("paths", {})
        o_ds = override.get("data_sources", {})
        o_model = override.get("model", {})
        o_roi = override.get("roi", {})
        for attr, section, key in [
            ("raw_dir", o_paths, "raw_dir"), ("processed_dir", o_paths, "processed_dir"),
            ("artifacts_dir", o_paths, "artifacts_dir"),
        ]:
            if key in section:
                setattr(args, attr, PROJECT_ROOT / section[key])
        for attr, section, key in [
            ("listings_url", o_ds, "listings_url"), ("calendar_url", o_ds, "calendar_url"),
        ]:
            if key in section:
                setattr(args, attr, section[key] or None)
        for attr, section, key in [
            ("max_rows", o_model, "max_rows"), ("test_size", o_model, "test_size"),
            ("random_state", o_model, "random_state"),
            ("clip_target_upper_quantile", o_model, "clip_target_upper_quantile"),
        ]:
            if key in section:
                setattr(args, attr, section[key])
        for attr, section, key in [
            ("investment", o_roi, "investment"), ("fixed_costs_annual", o_roi, "fixed_costs_annual"),
            ("variable_cost_rate", o_roi, "variable_cost_rate"), ("elasticity", o_roi, "elasticity"),
        ]:
            if key in section:
                setattr(args, attr, section[key])

    return args


def main() -> None:
    args = parse_args()

    summary = run_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        listings_url=args.listings_url,
        calendar_url=args.calendar_url,
        max_rows=args.max_rows,
        test_size=args.test_size,
        random_state=args.random_state,
        clip_target_upper_quantile=args.clip_target_upper_quantile,
        investment=args.investment,
        fixed_costs_annual=args.fixed_costs_annual,
        variable_cost_rate=args.variable_cost_rate,
        elasticity=args.elasticity,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
