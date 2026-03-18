"""Tests for YAML config loading in scripts."""
from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


class TestDefaultConfig:
    def test_config_exists(self):
        assert CONFIG_PATH.exists(), "configs/default.yaml must exist"

    def test_config_is_valid_yaml(self):
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        assert isinstance(cfg, dict)

    def test_required_sections_present(self):
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        for section in ["paths", "data_sources", "model", "roi"]:
            assert section in cfg, f"Missing config section: {section}"

    def test_paths_section(self):
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        paths = cfg["paths"]
        for key in ["raw_dir", "processed_dir", "artifacts_dir"]:
            assert key in paths

    def test_model_section(self):
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        model = cfg["model"]
        assert model["test_size"] == 0.2
        assert model["random_state"] == 42
        assert model["max_rows"] == 200000
        assert model["clip_target_upper_quantile"] == 0.995

    def test_roi_section(self):
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        roi = cfg["roi"]
        assert roi["investment"] == 350000
        assert roi["fixed_costs_annual"] == 14000
        assert roi["variable_cost_rate"] == 0.22
        assert roi["elasticity"] == 1.4

    def test_shap_section(self):
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        assert "shap" in cfg
        assert cfg["shap"]["sample_size"] == 2000
