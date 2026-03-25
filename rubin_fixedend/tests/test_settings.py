"""Tests für die Konfigurationsvalidierung."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from rubin.settings import load_config, AnalysisConfig, SUPPORTED_MODEL_NAMES


def _write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")
    return path


@pytest.fixture
def minimal_raw() -> dict:
    """Minimale gültige Konfiguration."""
    return {
        "data_files": {
            "x_file": "X.parquet",
            "t_file": "T.parquet",
            "y_file": "Y.parquet",
        },
        "models": {
            "models_to_train": ["SLearner"],
        },
    }


class TestLoadConfig:
    def test_minimal_config(self, tmp_path, minimal_raw):
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert isinstance(cfg, AnalysisConfig)
        assert cfg.models.models_to_train == ["SLearner"]
        assert cfg.constants.random_seed == 42

    def test_seed_alias(self, tmp_path, minimal_raw):
        minimal_raw["constants"] = {"SEED": 123}
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.constants.random_seed == 123
        assert cfg.constants.SEED == 123

    def test_unknown_model_rejected(self, tmp_path, minimal_raw):
        minimal_raw["models"]["models_to_train"] = ["UnknownModel"]
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception, match="unbekannte Modelle"):
            load_config(path)

    def test_extra_keys_rejected(self, tmp_path, minimal_raw):
        minimal_raw["unknown_key"] = "should_fail"
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception):
            load_config(path)

    def test_manual_champion_must_be_in_models(self, tmp_path, minimal_raw):
        minimal_raw["selection"] = {"manual_champion": "TLearner"}
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception, match="manual_champion"):
            load_config(path)

    def test_manual_champion_valid(self, tmp_path, minimal_raw):
        minimal_raw["selection"] = {"manual_champion": "SLearner"}
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.selection.manual_champion == "SLearner"

    def test_all_model_names_supported(self):
        expected = {"SLearner", "TLearner", "XLearner", "DRLearner",
                    "NonParamDML", "ParamDML", "CausalForestDML"}
        assert SUPPORTED_MODEL_NAMES == expected

    def test_source_config_path_set(self, tmp_path, minimal_raw):
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.source_config_path == str(path)

    def test_validate_on_holdout(self, tmp_path, minimal_raw):
        minimal_raw["data_processing"] = {"validate_on": "holdout", "test_size": 0.2}
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.data_processing.validate_on == "holdout"

    def test_validate_on_test_rejected(self, tmp_path, minimal_raw):
        minimal_raw["data_processing"] = {"validate_on": "test", "test_size": 0.2}
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception):
            load_config(path)

    # --- Feature-Selection Tests ---

    def test_feature_selection_defaults(self, tmp_path, minimal_raw):
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.feature_selection.enabled is False
        assert cfg.feature_selection.methods == ["lgbm_importance"]
        assert cfg.feature_selection.top_pct == 15.0
        assert cfg.feature_selection.max_features is None
        assert cfg.feature_selection.correlation_threshold == 0.9

    def test_feature_selection_multiple_methods(self, tmp_path, minimal_raw):
        minimal_raw["feature_selection"] = {
            "enabled": True,
            "methods": ["lgbm_importance", "causal_forest"],
            "top_pct": 20.0,
            "max_features": 50,
        }
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.feature_selection.methods == ["lgbm_importance", "causal_forest"]
        assert cfg.feature_selection.top_pct == 20.0
        assert cfg.feature_selection.max_features == 50

    def test_feature_selection_permutation_method(self, tmp_path, minimal_raw):
        minimal_raw["feature_selection"] = {
            "methods": ["lgbm_permutation"],
        }
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.feature_selection.methods == ["lgbm_permutation"]

    def test_feature_selection_old_method_field_rejected(self, tmp_path, minimal_raw):
        """Altes Singular-Feld 'method' wird durch extra=forbid abgewiesen."""
        minimal_raw["feature_selection"] = {
            "method": "lgbm_importance",
        }
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception):
            load_config(path)

    def test_feature_selection_old_importance_threshold_rejected(self, tmp_path, minimal_raw):
        """Altes Feld 'importance_threshold' wird durch extra=forbid abgewiesen."""
        minimal_raw["feature_selection"] = {
            "importance_threshold": 2.0,
        }
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception):
            load_config(path)

    # --- Surrogate-Tree Tests ---

    def test_surrogate_tree_default_disabled(self, tmp_path, minimal_raw):
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.surrogate_tree.enabled is False
        assert cfg.surrogate_tree.min_samples_leaf == 50
        assert cfg.surrogate_tree.num_leaves == 31
        assert cfg.surrogate_tree.max_depth is None

    def test_surrogate_tree_enabled(self, tmp_path, minimal_raw):
        minimal_raw["surrogate_tree"] = {
            "enabled": True,
            "min_samples_leaf": 100,
            "num_leaves": 15,
            "max_depth": 5,
        }
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.surrogate_tree.enabled is True
        assert cfg.surrogate_tree.min_samples_leaf == 100
        assert cfg.surrogate_tree.num_leaves == 15
        assert cfg.surrogate_tree.max_depth == 5

    # --- DataPrep Deduplicate Tests ---

    def test_data_prep_deduplicate_default(self, tmp_path, minimal_raw):
        minimal_raw["data_prep"] = {
            "data_path": ["data.csv"],
            "feature_path": "features.xlsx",
            "output_path": "out",
        }
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.data_prep.deduplicate is False
        assert cfg.data_prep.deduplicate_id_column is None

    def test_data_prep_deduplicate_enabled(self, tmp_path, minimal_raw):
        minimal_raw["data_prep"] = {
            "data_path": ["data.csv"],
            "feature_path": "features.xlsx",
            "output_path": "out",
            "deduplicate": True,
            "deduplicate_id_column": "PARTNER_ID",
        }
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.data_prep.deduplicate is True
        assert cfg.data_prep.deduplicate_id_column == "PARTNER_ID"

    # --- Multi-Treatment Tests ---

    def test_treatment_config_default_binary(self, tmp_path, minimal_raw):
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.treatment.type == "binary"
        assert cfg.treatment.reference_group == 0

    def test_treatment_config_multi(self, tmp_path, minimal_raw):
        minimal_raw["treatment"] = {"type": "multi", "reference_group": 0}
        # MT-kompatible Modelle verwenden
        minimal_raw["models"] = {"models_to_train": ["NonParamDML"]}
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.treatment.type == "multi"

    def test_mt_blocks_bt_only_models(self, tmp_path, minimal_raw):
        minimal_raw["treatment"] = {"type": "multi"}
        minimal_raw["models"] = {"models_to_train": ["SLearner"]}
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception, match="nicht kompatibel"):
            load_config(path)

    def test_mt_blocks_xlearner(self, tmp_path, minimal_raw):
        minimal_raw["treatment"] = {"type": "multi"}
        minimal_raw["models"] = {"models_to_train": ["XLearner"]}
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception, match="nicht kompatibel"):
            load_config(path)

    def test_mt_blocks_tlearner(self, tmp_path, minimal_raw):
        minimal_raw["treatment"] = {"type": "multi"}
        minimal_raw["models"] = {"models_to_train": ["TLearner"]}
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception, match="nicht kompatibel"):
            load_config(path)

    def test_mt_allows_dml(self, tmp_path, minimal_raw):
        minimal_raw["treatment"] = {"type": "multi"}
        minimal_raw["models"] = {"models_to_train": ["NonParamDML", "DRLearner", "CausalForestDML"]}
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.treatment.type == "multi"
        assert len(cfg.models.models_to_train) == 3
