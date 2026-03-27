"""Tests für Schema-Validierung."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from rubin.utils.schema_utils import Schema, validate_schema, save_schema, load_schema


@pytest.fixture
def sample_schema():
    return Schema(
        columns=["a", "b", "c"],
        dtypes={"a": "float64", "b": "int64", "c": "object"},
        categorical_columns=["c"],
    )


class TestSchema:
    def test_from_dataframe(self):
        df = pd.DataFrame({"x": [1.0], "y": ["a"]})
        schema = Schema.from_dataframe(df, categorical_columns=["y"])
        assert "x" in schema.columns
        assert "y" in schema.categorical_columns

    def test_save_and_load(self, tmp_path, sample_schema):
        path = str(tmp_path / "schema.json")
        save_schema(sample_schema, path)
        loaded = load_schema(path)
        assert loaded.columns == sample_schema.columns
        assert loaded.dtypes == sample_schema.dtypes

    def test_validate_ok(self, sample_schema):
        df = pd.DataFrame({"a": [1.0], "b": [1], "c": ["x"]})
        result = validate_schema(df, sample_schema)
        assert result.ok is True

    def test_validate_missing_column(self, sample_schema):
        df = pd.DataFrame({"a": [1.0], "c": ["x"]})
        result = validate_schema(df, sample_schema)
        assert result.ok is False
        assert "b" in result.missing_columns

    def test_validate_extra_columns_ok_non_strict(self, sample_schema):
        df = pd.DataFrame({"a": [1.0], "b": [1], "c": ["x"], "extra": [1]})
        result = validate_schema(df, sample_schema, strict=False)
        assert result.ok is True
        assert "extra" in result.extra_columns

    def test_validate_extra_columns_fail_strict(self, sample_schema):
        df = pd.DataFrame({"a": [1.0], "b": [1], "c": ["x"], "extra": [1]})
        result = validate_schema(df, sample_schema, strict=True)
        assert result.ok is False
