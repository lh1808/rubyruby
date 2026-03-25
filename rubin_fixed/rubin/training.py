from __future__ import annotations

"""Training- und Cross-Prediction-Hilfsfunktionen.
Die Analyse-Pipeline benötigt eine einheitliche Routine, um:
- Modelle zu trainieren und
- aus dem Training heraus Cross-Predictions zu erzeugen.
Warum Cross-Predictions?
Bei Uplift-/Causal-Modellen ist eine robuste Evaluation wichtig. Wenn man die
Effekte auf denselben Daten bewertet, auf denen ein Modell trainiert wurde,
werden Kennzahlen (z. B. Qini/AUUC) oft zu optimistisch.
Die hier implementierte Methode nutzt eine K-fache Aufteilung und erzeugt für
jede Beobachtung eine Vorhersage, die aus einem Modell stammt, das diese
Beobachtung nicht gesehen hat.
Parallelisierung:
Die CV-Folds sind vollständig unabhängig voneinander. Dieses Modul nutzt
joblib mit Thread-basiertem Backend, um Folds parallel zu verarbeiten.
Da LightGBM und CatBoost den GIL während des C++-Trainings freigeben,
erzielt dies echte Parallelität ohne Serialisierungs-Overhead.
Hinweis zur Modell-API:
EconML-Modelle bieten typischerweise .effect(X) oder .const_marginal_effect(X).
Für eine möglichst breite Kompatibilität wird beides unterstützt."""

from typing import Any
import copy
import gc
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

_logger = logging.getLogger("rubin.training")


def _predict_effect(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Berechnet Effekte/CATE-Vorhersagen für ein Modell.
Unterstützte Varianten (in Prioritätsreihenfolge):
- model.const_marginal_effect(X)  -- bevorzugt (DML-Familie)
- model.effect(X)                 -- Fallback (Meta-Learner)
Rückgabe:
- Binary Treatment:  1D-Array (n,)
- Multi Treatment:   2D-Array (n, K-1)"""
    if hasattr(model, "const_marginal_effect"):
        pred = model.const_marginal_effect(X)
    elif hasattr(model, "effect"):
        pred = model.effect(X)
    else:
        raise AttributeError(
            "Das Modell unterstützt weder 'const_marginal_effect' noch 'effect'. "
            "Bitte eine kompatible EconML-Implementierung verwenden oder die Funktion erweitern."
        )

    pred = np.asarray(pred)
    # EconML gibt je nach Modell/Version verschiedene Shapes zurück:
    # BT: (n,), (n,1), (n,1,1) → immer auf (n,) reduzieren
    # MT: (n, K-1), (n, K-1, 1) → auf (n, K-1) reduzieren
    pred = pred.squeeze()
    # Falls squeeze() ein Skalar erzeugt (n=1), zurück zu 1D
    if pred.ndim == 0:
        pred = pred.reshape(1)
    return pred.astype(float)


def _n_treatment_arms(T: np.ndarray) -> int:
    """Anzahl der Treatment-Arme (inkl. Control)."""
    return len(np.unique(T))


def is_multi_treatment(T: np.ndarray) -> bool:
    """Prüft, ob mehr als 2 Treatment-Gruppen vorliegen."""
    return _n_treatment_arms(T) > 2


# ---------------------------------------------------------------------------
# Parallelisierungs-Helfer
# ---------------------------------------------------------------------------

def _auto_parallel_folds(n_splits: int, parallel_level: int = 2) -> int:
    """Bestimmt die Anzahl paralleler CV-Folds basierend auf dem Parallelisierungs-Level.

    Level 1 (Minimal):  1 — sequentiell
    Level 2 (Moderat):  1 — sequentiell (nur Base Learner parallel)
    Level 3 (Hoch):     auto — 2–4 parallele Folds (Heuristik nach Kernzahl)
    Level 4 (Maximum):  n_splits — alle Folds parallel (gecappt durch CPU-Zahl)

    Kann über die Umgebungsvariable RUBIN_PARALLEL_FOLDS überschrieben werden.
    Wert 1 = sequentiell. Wert 0 oder nicht gesetzt = Level-basiert.
    """
    env = os.environ.get("RUBIN_PARALLEL_FOLDS", "0")
    try:
        forced = int(env)
        if forced >= 1:
            return min(forced, n_splits)
    except ValueError:
        pass

    if parallel_level <= 2:
        return 1  # Level 1+2: Folds sequentiell

    n_cpus = os.cpu_count() or 1

    if parallel_level >= 4:
        # Level 4: Alle Folds parallel, aber nie mehr als CPU-Kerne
        return min(n_splits, n_cpus)

    # Level 3: Auto-Heuristik
    # Jeder Fold bekommt mindestens 4 Kerne für interne Parallelisierung
    max_workers = max(1, n_cpus // 4)
    return min(n_splits, max_workers, n_cpus)


def _fit_single_fold(
    model_template: Any,
    X: pd.DataFrame,
    Y: np.ndarray,
    T: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Trainiert einen einzelnen CV-Fold und gibt (va_idx, fold_predictions) zurück.

    Thread-safe: Jeder Aufruf arbeitet auf einer eigenen deepcopy des Modells.
    LightGBM/CatBoost geben den GIL während des C++-Trainings frei,
    sodass mehrere Threads echte Parallelität erzielen."""
    m = copy.deepcopy(model_template)
    m.fit(Y[tr_idx], T[tr_idx], X=X.iloc[tr_idx])
    fold_pred = _predict_effect(m, X.iloc[va_idx])
    del m
    # Kein gc.collect() hier — gc ist global und verursacht Thread-Contention.
    # Wird nach Abschluss aller Folds einmalig aufgerufen.
    return va_idx, fold_pred


def _run_folds_sequential(
    model: Any,
    X: pd.DataFrame,
    Y: np.ndarray,
    T: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    preds: np.ndarray,
    is_mt: bool,
) -> np.ndarray:
    """Sequentielle Fold-Verarbeitung (Fallback oder bei wenig Kernen)."""
    for tr_idx, va_idx in folds:
        try:
            m = copy.deepcopy(model)
        except Exception as e:
            raise RuntimeError(
                "Das Modell konnte nicht kopiert werden. "
                "Bitte sicherstellen, dass das Modell deepcopy-fähig ist, oder die Trainingsroutine "
                "so erweitern, dass pro Fold eine neue Instanz erzeugt wird."
            ) from e

        m.fit(Y[tr_idx], T[tr_idx], X=X.iloc[tr_idx])
        fold_pred = _predict_effect(m, X.iloc[va_idx])

        if is_mt:
            preds[va_idx, :] = fold_pred
        else:
            preds[va_idx] = fold_pred

        del m
        gc.collect()

    return preds


def train_and_crosspredict_bt_bo(
    model: Any,
    X: pd.DataFrame,
    Y: np.ndarray,
    T: np.ndarray,
    n_splits: int,
    model_name: str,
    random_state: int,
    return_train_predictions: bool = True,
    parallel_level: int = 2,
) -> pd.DataFrame:
    """Trainiert ein Modell und erzeugt Cross-Predictions für BT/BO und MT/BO.

BT/BO = Binary Treatment / Binary Outcome.
MT/BO = Multi Treatment / Binary Outcome.

Vorgehen:
- StratifiedKFold auf der Kombination aus Treatment und Outcome, damit die
  Grundgruppen (T x Y) pro Fold möglichst stabil bleiben.
- Die CV-Folds werden parallel verarbeitet (joblib, Thread-Backend).
  LightGBM und CatBoost geben den GIL während des C++-Trainings frei,
  sodass mehrere Folds echte Parallelität erzielen.

Ergebnis (BT):
  DataFrame mit Spalten: Y, T, Predictions_<model_name>, optional Train_<model_name>
Ergebnis (MT, K Treatment-Arme):
  DataFrame mit Spalten: Y, T, Predictions_<model_name>_T1, ..., Predictions_<model_name>_T{K-1},
  OptimalTreatment_<model_name>, optional Train_<model_name>_T1, ..."""
    if n_splits < 2:
        raise ValueError("n_splits muss >= 2 sein.")

    t_int = np.asarray(T).astype(int)
    y_int = np.asarray(Y).astype(int)
    strata = (pd.Series(t_int).astype(str) + "_" + pd.Series(y_int).astype(str)).to_numpy()
    strata_counts = pd.Series(strata).value_counts(dropna=False)

    effective_splits = int(n_splits)
    if not strata_counts.empty:
        effective_splits = min(effective_splits, int(strata_counts.min()))

    if effective_splits >= 2:
        cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        split_iter = cv.split(np.zeros(len(X)), strata)
    else:
        fallback_splits = min(int(n_splits), len(X))
        if fallback_splits < 2:
            raise ValueError("Für Cross-Predictions werden mindestens 2 Zeilen benötigt.")
        cv = KFold(n_splits=fallback_splits, shuffle=True, random_state=random_state)
        split_iter = cv.split(np.zeros(len(X)))

    K = _n_treatment_arms(T)
    is_mt = K > 2
    n_effects = K - 1

    if is_mt:
        preds = np.full(shape=(len(X), n_effects), fill_value=np.nan, dtype=float)
    else:
        preds = np.full(shape=(len(X),), fill_value=np.nan, dtype=float)

    # Alle Fold-Splits vorab materialisieren (nötig für parallele Verarbeitung)
    folds = [(np.asarray(tr, dtype=int), np.asarray(va, dtype=int)) for tr, va in split_iter]

    # Anzahl paralleler Folds bestimmen
    n_parallel = _auto_parallel_folds(len(folds), parallel_level=parallel_level)

    # CausalForestDML nutzt intern GRF (sklearn CausalForest), das joblib-Prozesse
    # für die Baum-Parallelisierung spawnt. In einem äußeren joblib-Thread führt
    # fork()-aus-Thread zu Deadlocks. Folds für CausalForestDML daher immer
    # sequentiell — jeder Fold bekommt alle Kerne für die interne Parallelisierung.
    if (model_name or "").lower() == "causalforestdml" and n_parallel >= 2:
        _logger.info(
            "%s: Erzwinge sequentielle Folds (GRF nutzt intern joblib-Prozesse, "
            "die in Threads zu Deadlocks führen).", model_name,
        )
        n_parallel = 1

    if n_parallel >= 2 and len(folds) >= 2:
        # ── Parallele Fold-Verarbeitung ──
        # Thread-Backend: LightGBM/CatBoost geben den GIL frei → echte Parallelität.
        # Shared Memory: kein Serialisierungs-Overhead für X, Y, T.
        try:
            from joblib import Parallel, delayed

            _logger.info(
                "%s: %d Folds parallel (n_jobs=%d, threads) auf %d Kernen.",
                model_name, len(folds), n_parallel, os.cpu_count() or 1,
            )

            results = Parallel(n_jobs=n_parallel, prefer="threads")(
                delayed(_fit_single_fold)(model, X, Y, T, tr_idx, va_idx)
                for tr_idx, va_idx in folds
            )

            for va_idx, fold_pred in results:
                if is_mt:
                    preds[va_idx, :] = fold_pred
                else:
                    preds[va_idx] = fold_pred

            del results
            gc.collect()

        except Exception as e:
            _logger.warning(
                "%s: Parallele Fold-Verarbeitung fehlgeschlagen (%s). "
                "Fallback auf sequentiell.", model_name, e,
            )
            preds = _run_folds_sequential(model, X, Y, T, folds, preds, is_mt)
    else:
        # ── Sequentielle Fold-Verarbeitung ──
        _logger.info("%s: %d Folds sequentiell.", model_name, len(folds))
        preds = _run_folds_sequential(model, X, Y, T, folds, preds, is_mt)

    # Ergebnis-DataFrame bauen
    out = pd.DataFrame({"Y": Y, "T": T})

    if is_mt:
        for k in range(n_effects):
            out[f"Predictions_{model_name}_T{k+1}"] = preds[:, k]
        # Optimale Treatment-Zuweisung: argmax über die K-1 Effekte,
        # aber nur wenn der beste Effekt > 0 ist. Sonst Control (0).
        best_effect = np.nanmax(preds, axis=1)
        best_arm = np.nanargmax(preds, axis=1) + 1  # 1-basiert
        out[f"OptimalTreatment_{model_name}"] = np.where(best_effect > 0, best_arm, 0)
    else:
        out[f"Predictions_{model_name}"] = preds

    if return_train_predictions:
        try:
            m_full = copy.deepcopy(model)
            m_full.fit(Y, T, X=X)
            train_pred = _predict_effect(m_full, X)
            del m_full
            gc.collect()
            if is_mt:
                for k in range(n_effects):
                    out[f"Train_{model_name}_T{k+1}"] = train_pred[:, k]
            else:
                out[f"Train_{model_name}"] = train_pred
        except Exception:
            if is_mt:
                for k in range(n_effects):
                    out[f"Train_{model_name}_T{k+1}"] = np.nan
            else:
                out[f"Train_{model_name}"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Surrogate-Einzelbaum
# ---------------------------------------------------------------------------

SURROGATE_MODEL_NAME = "SurrogateTree"


class SurrogateTreeWrapper:
    """Wrapper um einen Einzelbaum-Regressor für CATE-kompatible Schnittstelle.

    Der Surrogate-Einzelbaum lernt die CATE-Vorhersagen des Champion-Modells
    nach (Teacher-Learner-Prinzip). Intern wird ein einzelner Baum des
    konfigurierten Base-Learners (LightGBM/CatBoost mit n_estimators=1)
    verwendet. Damit er in der Production-Pipeline wie ein normales
    CATE-Modell gescoret werden kann, stellt dieser Wrapper die
    Methoden ``const_marginal_effect`` und ``effect`` bereit.

    Bei Binary Treatment wird ein einzelner Baum gespeichert (``tree``).
    Bei Multi-Treatment wird pro Treatment-Arm ein eigener Baum trainiert
    (``trees``-Dict), da LightGBM/CatBoost nur 1D-Targets unterstützen.
    """

    _is_surrogate = True

    def __init__(self, tree=None, trees: dict | None = None, champion_name: str = ""):
        self.tree = tree
        self.trees = trees or {}
        self.champion_name = champion_name

    def const_marginal_effect(self, X):
        if self.trees:
            # MT: pro Arm predicten und zu (n, K-1)-Matrix zusammensetzen
            arm_keys = sorted(self.trees.keys())
            preds = np.column_stack([self.trees[k].predict(X) for k in arm_keys])
            return preds
        return self.tree.predict(X)

    def effect(self, X):
        return self.const_marginal_effect(X)
