"""Tests für Data-Utils."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rubin.utils.data_utils import reduce_mem_usage


class TestReduceMemUsage:
    def test_no_float16(self):
        """Stellt sicher, dass float16 nie verwendet wird."""
        df = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [1.0, 2.0, 3.0]})
        result = reduce_mem_usage(df)
        for col in result.columns:
            assert result[col].dtype != np.float16, (
                f"Spalte {col} wurde auf float16 gecastet – "
                f"float16 hat zu wenig Genauigkeit für ML-Anwendungen."
            )

    def test_integer_downcast(self):
        df = pd.DataFrame({"small": [1, 2, 3], "big": [100000, 200000, 300000]})
        result = reduce_mem_usage(df)
        assert result["small"].dtype == np.int8
        assert result["big"].dtype in (np.int32, np.int64)

    def test_float_at_least_float32(self):
        df = pd.DataFrame({"f": np.random.uniform(0, 1, 100).astype(np.float64)})
        result = reduce_mem_usage(df)
        assert result["f"].dtype == np.float32
