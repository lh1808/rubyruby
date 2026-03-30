"""Kategorische Feature-Unterstützung für EconML-Modelle.

Problem
-------
EconML konvertiert X intern zu numpy-Arrays (via sklearn check_array), wodurch
pandas category-Dtypes verloren gehen. LightGBM und CatBoost erhalten dann nur
float64-Werte und können keine kategorialen Splits mehr nutzen. Stattdessen
werden ordinale Splits auf nominalen Features angewendet — deutlich schwächere
Modellierung.

Lösung
------
Die ``partialmethod``-Technik patcht die ``.fit()``-Methoden von LGBMClassifier,
LGBMRegressor, CatBoostClassifier und CatBoostRegressor so, dass
``categorical_feature`` (LightGBM) bzw. ``cat_features`` (CatBoost) bei jedem
``.fit()``-Aufruf automatisch übergeben wird — auch wenn EconML intern nur
``model.fit(X_numpy, y)`` aufruft.

Nutzung
-------
>>> with patch_categorical_features(X):
...     model = NonParamDML(model_y=LGBMClassifier(), ...)
...     model.fit(Y, T, X=X)
...     cate = model.const_marginal_effect(X)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import partialmethod
from typing import List, Optional

import pandas as pd

_logger = logging.getLogger("rubin.categorical")


def _detect_cat_indices(X: pd.DataFrame) -> List[int]:
    """Ermittelt die Spaltenindizes aller kategorialen Features."""
    cat_cols = X.select_dtypes(include=["category", "object"]).columns
    return [i for i, col in enumerate(X.columns) if col in cat_cols]


@contextmanager
def patch_categorical_features(X: pd.DataFrame, base_learner_type: str = "lgbm"):
    """Context-Manager: Patcht LightGBM/CatBoost für kategorische Features.

    Erkennt automatisch category-/object-Spalten in X, berechnet deren
    Spaltenindizes und patcht die .fit()-Methoden der Base Learner so, dass
    ``categorical_feature`` (LightGBM) bzw. ``cat_features`` (CatBoost)
    bei jedem Aufruf automatisch übergeben wird.

    Nach Verlassen des Kontexts werden die originalen .fit()-Methoden
    wiederhergestellt (kein globaler State-Leak).

    Parameters
    ----------
    X : pd.DataFrame
        Feature-Matrix mit den finalen Spalten (nach Feature-Selektion).
    base_learner_type : str
        "lgbm" oder "catboost". Bestimmt, welche Klassen gepatcht werden.
    """
    cat_indices = _detect_cat_indices(X)

    if not cat_indices:
        _logger.debug("Keine kategorialen Spalten erkannt — kein Patching nötig.")
        yield cat_indices
        return

    cat_col_names = [X.columns[i] for i in cat_indices]
    _logger.info(
        "Kategorische Spalten erkannt: %d von %d Features (%s). "
        "Patche %s .fit()-Methoden für EconML-Kompatibilität.",
        len(cat_indices), len(X.columns),
        cat_col_names[:5] if len(cat_col_names) <= 5 else f"{cat_col_names[:3]}... (+{len(cat_col_names)-3})",
        base_learner_type.upper(),
    )

    originals = {}
    base_type = (base_learner_type or "lgbm").lower()

    try:
        if base_type == "lgbm":
            import lightgbm as lgbm

            # Originale sichern
            originals["LGBMClassifier.fit"] = lgbm.LGBMClassifier.fit
            originals["LGBMRegressor.fit"] = lgbm.LGBMRegressor.fit

            # Patchen: categorical_feature wird bei JEDEM .fit()-Aufruf übergeben,
            # auch wenn EconML intern nur model.fit(X_numpy, y) aufruft.
            lgbm.LGBMClassifier.fit = partialmethod(
                lgbm.LGBMClassifier.fit, categorical_feature=cat_indices
            )
            lgbm.LGBMRegressor.fit = partialmethod(
                lgbm.LGBMRegressor.fit, categorical_feature=cat_indices
            )

        elif base_type == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor

            # Originale sichern
            originals["CatBoostClassifier.fit"] = CatBoostClassifier.fit
            originals["CatBoostRegressor.fit"] = CatBoostRegressor.fit

            # CatBoost: cat_features statt categorical_feature
            CatBoostClassifier.fit = partialmethod(
                CatBoostClassifier.fit, cat_features=cat_indices
            )
            CatBoostRegressor.fit = partialmethod(
                CatBoostRegressor.fit, cat_features=cat_indices
            )

        yield cat_indices

    finally:
        # Originale wiederherstellen — IMMER, auch bei Exceptions
        if base_type == "lgbm" and "LGBMClassifier.fit" in originals:
            import lightgbm as lgbm
            lgbm.LGBMClassifier.fit = originals["LGBMClassifier.fit"]
            lgbm.LGBMRegressor.fit = originals["LGBMRegressor.fit"]
            _logger.debug("LightGBM .fit()-Methoden wiederhergestellt.")

        elif base_type == "catboost" and "CatBoostClassifier.fit" in originals:
            from catboost import CatBoostClassifier, CatBoostRegressor
            CatBoostClassifier.fit = originals["CatBoostClassifier.fit"]
            CatBoostRegressor.fit = originals["CatBoostRegressor.fit"]
            _logger.debug("CatBoost .fit()-Methoden wiederhergestellt.")
