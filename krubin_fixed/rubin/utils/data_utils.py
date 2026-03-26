from __future__ import annotations

from typing import Dict, Optional, Tuple
import json

import numpy as np
import pandas as pd


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Reduziert den Speicherbedarf eines DataFrames per best-effort Downcast."""
    df = df.copy()
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
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


def _random_train_test_split(
    X: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    S: Optional[np.ndarray],
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, shuffle=True)
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    T_arr = np.asarray(T)
    Y_arr = np.asarray(Y)
    T_train, T_test = T_arr[train_idx], T_arr[test_idx]
    Y_train, Y_test = Y_arr[train_idx], Y_arr[test_idx]

    if S is None:
        return X_train, X_test, T_train, T_test, Y_train, Y_test, None, None

    S_arr = np.asarray(S)
    return X_train, X_test, T_train, T_test, Y_train, Y_test, S_arr[train_idx], S_arr[test_idx]


def stratified_train_test_split(
    X: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    S: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Train/Test-Split mit Stratifikation über die Kombination aus Treatment und Outcome.

    Falls eine saubere Stratifikation wegen zu kleiner Gruppen nicht möglich ist,
    wird robust auf einen normalen Shuffle-Split zurückgefallen.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    strata = (pd.Series(T).astype(str) + "_" + pd.Series(Y).astype(str)).to_numpy()
    counts = pd.Series(strata).value_counts(dropna=False)
    if counts.empty or int(counts.min()) < 2:
        return _random_train_test_split(X, T, Y, S, test_size=test_size, random_state=random_state)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    try:
        train_idx, test_idx = next(splitter.split(X, strata))
    except ValueError:
        return _random_train_test_split(X, T, Y, S, test_size=test_size, random_state=random_state)

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    # Index zurücksetzen: position-basierte Konsistenz mit numpy T/Y Arrays
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    T_train, T_test = np.asarray(T)[train_idx], np.asarray(T)[test_idx]
    Y_train, Y_test = np.asarray(Y)[train_idx], np.asarray(Y)[test_idx]

    if S is None:
        return X_train, X_test, T_train, T_test, Y_train, Y_test, None, None

    S_arr = np.asarray(S)
    S_train, S_test = S_arr[train_idx], S_arr[test_idx]
    return X_train, X_test, T_train, T_test, Y_train, Y_test, S_train, S_test


def load_dtypes_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
