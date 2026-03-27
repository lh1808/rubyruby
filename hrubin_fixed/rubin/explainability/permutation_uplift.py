from __future__ import annotations

"""Permutation-Importance für Uplift/CATE.
Warum Permutation-Importance?
----------------------------
Permutation-Importance ist ein sehr robuster, modellagnostischer Ansatz:
* keine speziellen Modellannahmen
* keine zusätzlichen Bibliotheken nötig
* gut als Fallback, wenn SHAP nicht verfügbar ist
Grundidee:
----------
Wir betrachten eine Vorhersagefunktion f(X)=CATE(X). Dann permutieren wir jeweils
ein Feature und messen, wie stark sich die Vorhersagen im Mittel ändern.
Wichtiger Hinweis:
------------------
Permutation-Importance misst *Einfluss auf die Vorhersage*, nicht notwendigerweise
kausale Bedeutung. Für die Modellinterpretation in Business-Kontexten ist es jedoch
oft ein sehr nützlicher Indikator."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def _predict_uplift(model: object, X: pd.DataFrame) -> np.ndarray:
    """Vorhersage der Uplift-Werte. Bei Multi-Treatment wird die L2-Norm
    über alle Treatment-Arme verwendet, um einen skalaren Einfluss zu messen."""
    if hasattr(model, "const_marginal_effect"):
        pred = np.asarray(model.const_marginal_effect(X))
    elif hasattr(model, "effect"):
        pred = np.asarray(model.effect(X))
    else:
        raise TypeError(
            "Das übergebene Modell unterstützt weder 'const_marginal_effect' noch 'effect'."
        )
    # BT: (n,1) -> (n,); MT: (n, K-1) bleibt erhalten
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    return pred


@dataclass
class PermutationImportanceResult:
    feature_names: list[str]
    importances: np.ndarray  # mean absolute delta
    std: np.ndarray

    def as_series(self) -> pd.Series:
        return pd.Series(self.importances, index=self.feature_names).sort_values(ascending=False)


def compute_permutation_importance_for_uplift(
    model: object,
    X: pd.DataFrame,
    n_repeats: int = 5,
    seed: int = 42,
    max_rows: Optional[int] = 20000,
) -> PermutationImportanceResult:
    """Berechnet Permutation-Importance auf den Uplift-Vorhersagen.
Parameter
---------
model:
Trainiertes kausales Modell.
X:
Feature-Matrix (typisch transformierte Features).
n_repeats:
Anzahl Permutationswiederholungen pro Feature.
seed:
Seed für deterministische Permutationen.
max_rows:
Optional: Stichprobe aus X für Performance. None = alles.
Rückgabe
--------
Ergebnisobjekt mit Importance-Mittelwert und Standardabweichung je Feature."""
    rng = np.random.default_rng(seed)
    X_work = X
    if max_rows is not None and len(X) > max_rows:
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X_work = X.iloc[idx].copy()
    else:
        X_work = X.copy()

    base = _predict_uplift(model, X_work)
    feature_names = list(X_work.columns)
    n_features = len(feature_names)

    all_scores = np.zeros((n_repeats, n_features), dtype=float)

    for j, col in enumerate(feature_names):
        col_values = X_work[col].to_numpy(copy=True)
        for r in range(n_repeats):
            permuted = col_values.copy()
            rng.shuffle(permuted)
            X_work[col] = permuted

            pred = _predict_uplift(model, X_work)
            delta = pred - base
            if delta.ndim == 2:
                # MT: L2-Norm über Treatment-Arme
                all_scores[r, j] = float(np.mean(np.linalg.norm(delta, axis=1)))
            else:
                all_scores[r, j] = float(np.mean(np.abs(delta)))

        # Originalwerte zurücksetzen
        X_work[col] = col_values

    importances = all_scores.mean(axis=0)
    std = all_scores.std(axis=0)
    return PermutationImportanceResult(feature_names=feature_names, importances=importances, std=std)
