from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dpra.explain import generate_shap_artifacts

CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return {}


def parse_args() -> argparse.Namespace:
    cfg = _load_config()
    paths = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})
    shap_cfg = cfg.get("shap", {})

    artifacts = PROJECT_ROOT / paths.get("artifacts_dir", "artifacts")
    processed = PROJECT_ROOT / paths.get("processed_dir", "data/processed")

    parser = argparse.ArgumentParser(description="Generate SHAP explainability artifacts")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=artifacts / "price_model.joblib",
    )
    parser.add_argument(
        "--training-table-path",
        type=Path,
        default=processed / "training_table.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=artifacts)
    parser.add_argument("--sample-size", type=int, default=shap_cfg.get("sample_size", 2000))
    parser.add_argument("--random-state", type=int, default=model_cfg.get("random_state", 42))

    args = parser.parse_args()

    if args.config is not None:
        override = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        o_paths = override.get("paths", {})
        o_model = override.get("model", {})
        o_shap = override.get("shap", {})
        if "artifacts_dir" in o_paths:
            o_artifacts = PROJECT_ROOT / o_paths["artifacts_dir"]
            args.model_path = o_artifacts / "price_model.joblib"
            args.output_dir = o_artifacts
        if "processed_dir" in o_paths:
            args.training_table_path = PROJECT_ROOT / o_paths["processed_dir"] / "training_table.csv"
        if "sample_size" in o_shap:
            args.sample_size = o_shap["sample_size"]
        if "random_state" in o_model:
            args.random_state = o_model["random_state"]

    return args


def main() -> None:
    args = parse_args()
    csv_path, png_path = generate_shap_artifacts(
        model_path=args.model_path,
        training_table_path=args.training_table_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    print(f"Saved SHAP CSV: {csv_path}")
    print(f"Saved SHAP chart: {png_path}")


if __name__ == "__main__":
    main()
