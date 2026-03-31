from __future__ import annotations

from typing import Dict
import json

import numpy as np
import pandas as pd


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Reduziert den Speicherbedarf eines DataFrames per best-effort Downcast.
    Kategorie-Spalten werden NICHT angetastet — ihr Dtype muss erhalten bleiben,
    damit die kategoriale Patchlogik (LightGBM/CatBoost) korrekt funktioniert."""
    df = df.copy()
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        # Kategorie-Spalten niemals downcasten — der category-Dtype ist
        # essenziell für patch_categorical_features und schema.json.
        if isinstance(col_type, pd.CategoricalDtype):
            continue
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                # float16 wird bewusst nicht verwendet: nur ~3 Dezimalstellen
                # Genauigkeit, was bei feinen Score-Unterschieden zu
                # Informationsverlust und Qualitätseinbußen führen kann.
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    df.attrs["memory_usage_mb_before"] = float(start_mem)
    df.attrs["memory_usage_mb_after"] = float(end_mem)
    return df


def load_dtypes_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
