"""Tests für das Preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rubin.preprocessing import (
    FittedPreprocessor,
    fit_preprocessor,
    SimpleCSVPreprocessor,
    build_simple_preprocessor_from_dataframe,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, np.nan, 5.0],
        "num2": [10, 20, 30, 40, 50],
        "cat1": ["a", "b", "a", "c", "b"],
    })


class TestFitPreprocessor:
    def test_roundtrip(self, sample_df):
        preproc = fit_preprocessor(sample_df, categorical_columns=["cat1"], fill_na_method="median")
        result = preproc.transform(sample_df)
        assert list(result.columns) == list(sample_df.columns)
        assert result["num1"].isna().sum() == 0  # NaN wurde gefüllt
        assert result["cat1"].dtype.name == "category"

    def test_unknown_category_gets_minus_one(self, sample_df):
        preproc = fit_preprocessor(sample_df, categorical_columns=["cat1"])
        new_df = pd.DataFrame({
            "num1": [1.0],
            "num2": [10],
            "cat1": ["UNKNOWN"],
        })
        result = preproc.transform(new_df)
        assert int(result["cat1"].iloc[0]) == -1

    def test_missing_column_filled_with_nan(self, sample_df):
        preproc = fit_preprocessor(sample_df, categorical_columns=["cat1"])
        new_df = pd.DataFrame({"num1": [1.0], "cat1": ["a"]})
        result = preproc.transform(new_df)
        assert "num2" in result.columns

    def test_fill_na_mean(self, sample_df):
        preproc = fit_preprocessor(sample_df, categorical_columns=["cat1"], fill_na_method="mean")
        assert "num1" in preproc.fillna_values
        expected = sample_df["num1"].mean()
        assert abs(preproc.fillna_values["num1"] - expected) < 1e-6

    def test_fill_na_zero(self, sample_df):
        preproc = fit_preprocessor(sample_df, categorical_columns=["cat1"], fill_na_method="zero")
        assert preproc.fillna_values.get("num1") == 0.0
        assert preproc.fillna_values.get("num2") == 0.0

    def test_fill_na_mode(self, sample_df):
        preproc = fit_preprocessor(sample_df, categorical_columns=["cat1"], fill_na_method="mode")
        assert "num1" in preproc.fillna_values

    def test_fill_na_none(self, sample_df):
        preproc = fit_preprocessor(sample_df, categorical_columns=["cat1"], fill_na_method=None)
        assert preproc.fillna_values == {}


class TestSimpleCSVPreprocessor:
    def test_build_and_transform(self, sample_df):
        preproc = build_simple_preprocessor_from_dataframe(sample_df)
        assert isinstance(preproc, SimpleCSVPreprocessor)
        result = preproc.transform(sample_df)
        assert list(result.columns) == list(sample_df.columns)

    def test_schema_validation(self, sample_df):
        preproc = build_simple_preprocessor_from_dataframe(sample_df)
        result = preproc.validate(sample_df, strict=False)
        assert result.ok is True

    def test_schema_missing_column(self, sample_df):
        preproc = build_simple_preprocessor_from_dataframe(sample_df)
        new_df = sample_df.drop(columns=["num1"])
        result = preproc.validate(new_df, strict=False)
        assert "num1" in result.missing_columns
