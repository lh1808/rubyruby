from __future__ import annotations

"""Feature-Selektion für kausale Modellierung.

Unterstützte Methoden
---------------------
- ``lgbm_importance``: LightGBM-Regressor auf Outcome (Y), Gain-Importance.
  Schnell, erfasst prädiktive Relevanz für das Outcome.
- ``lgbm_permutation``: LightGBM-Regressor auf Outcome (Y), Permutation-Importance.
  Robuster als Gain (kein Split-Bias), aber rechenintensiver.
- ``causal_forest``: EconML GRF CausalForest Feature-Importances.
  Erfasst kausale Relevanz (Heterogenität des Treatment-Effekts).
  Nutzt die direkte GRF-Implementierung ohne separates Nuisance-Fitting.

Bei mehreren Methoden werden die Top-X% aus jeder Methode berechnet und
per Union zusammengeführt. Dadurch werden Features behalten, die entweder
prädiktiv wichtig (Outcome) oder kausal relevant (CATE-Heterogenität) sind.
"""

from typing import Dict, Iterable, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd


_logger = logging.getLogger("rubin.feature_selection")


# ---------------------------------------------------------------------------
# Korrelationsfilter
# ---------------------------------------------------------------------------

def remove_highly_correlated_features(
    X: pd.DataFrame,
    correlation_threshold: float = 0.9,
    correlation_methods: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Entfernt stark korrelierte numerische Features.

    Für jede Methode (Standard: Pearson + Spearman) wird eine Korrelationsmatrix
    berechnet. Sobald ein Feature in *einer* Methode oberhalb des Schwellwerts
    liegt, wird es als redundant markiert."""
    methods = list(correlation_methods or ["pearson", "spearman"])
    threshold = float(correlation_threshold)

    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if len(numeric_cols) < 2:
        return X.copy(), []

    to_drop: List[str] = []
    for method in methods:
        corr_matrix = X[numeric_cols].corr(method=method).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop.extend([column for column in upper.columns if any(upper[column] > threshold)])

    to_drop = sorted(set(to_drop))
    return X.drop(columns=to_drop, errors="ignore"), to_drop


# ---------------------------------------------------------------------------
# Importance-Methoden
# ---------------------------------------------------------------------------

def _lgbm_gain_importance(
    X: pd.DataFrame, Y: np.ndarray, seed: int, n_jobs: int = -1,
) -> pd.Series:
    """LightGBM-Regressor auf Outcome trainieren, Gain-Importance extrahieren."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, num_leaves=31,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, random_state=seed, n_jobs=n_jobs, verbose=-1,
    )
    model.fit(X, Y)
    imp = model.feature_importances_
    return pd.Series(imp, index=X.columns, name="lgbm_gain").sort_values(ascending=False)


def _lgbm_permutation_importance(
    X: pd.DataFrame, Y: np.ndarray, seed: int,
    n_repeats: int = 5, n_jobs: int = -1,
) -> pd.Series:
    """LightGBM-Regressor auf Outcome trainieren, Permutation-Importance berechnen."""
    import lightgbm as lgb
    from sklearn.inspection import permutation_importance

    model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, num_leaves=31,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, random_state=seed, n_jobs=n_jobs, verbose=-1,
    )
    model.fit(X, Y)
    result = permutation_importance(
        model, X, Y, n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs,
    )
    imp = result.importances_mean
    return pd.Series(imp, index=X.columns, name="lgbm_permutation").sort_values(ascending=False)


def _causal_forest_importance(
    X: pd.DataFrame, T: np.ndarray, Y: np.ndarray, seed: int,
    n_jobs: int = -1, in_thread: bool = False,
) -> pd.Series:
    """GRF CausalForest trainieren und Feature-Importances extrahieren.

    Nutzt die GRF-Implementierung aus EconML (``econml.grf.CausalForest``),
    nicht das DML-Wrapper-Modell. Vorteil: kein separates Nuisance-Fitting,
    direkte Schätzung der Treatment-Effekt-Heterogenität über Honest Splitting.

    Parameters
    ----------
    n_jobs : int
        Kerne für die Baum-Parallelisierung. Wird nur im Hauptthread genutzt.
    in_thread : bool
        True wenn der Aufruf aus einem joblib-Thread kommt. Dann wird n_jobs=1
        erzwungen, da GRF intern joblib-Prozesse spawnt (fork-aus-Thread → Deadlock).

    Besonderheiten
    --------------
    - GRF kann keine fehlenden Werte verarbeiten.
    - X wird intern zu numpy float64 konvertiert.
    - Bei Multi-Treatment wird T automatisch binarisiert (Control vs. Any Treatment).
    - Bei großen Datensätzen (>100k Zeilen) wird automatisch stratifiziert subsampled,
      da Feature-Importance-Rankings sich ab dieser Größe kaum noch ändern.
    """
    from econml.grf import CausalForest

    # ── Datenkonversion ──
    feature_names = list(X.columns)
    try:
        X_np = np.asarray(X, dtype=np.float64)
    except (ValueError, TypeError):
        X_np = np.column_stack([
            pd.to_numeric(X.iloc[:, i], errors="coerce").to_numpy(dtype=np.float64)
            for i in range(X.shape[1])
        ])
        _logger.info("CausalForest: spaltenweise numpy-Konversion (gemischte Dtypes).")

    T_np = np.asarray(T, dtype=np.float64).ravel()
    Y_np = np.asarray(Y, dtype=np.float64).ravel()

    # ── NaN-Check ──
    if np.isnan(X_np).any():
        nan_cols = [feature_names[i] for i in range(X_np.shape[1]) if np.isnan(X_np[:, i]).any()]
        raise ValueError(
            f"CausalForest: X enthält NaN in {len(nan_cols)} Spalten nach Konversion: "
            f"{nan_cols[:5]}{'...' if len(nan_cols) > 5 else ''}. "
            "GRF kann keine fehlenden Werte verarbeiten."
        )

    # ── Multi-Treatment → Binarisierung (Control vs. Any Treatment) ──
    n_treatments = len(np.unique(T_np))
    if n_treatments > 2:
        _logger.info(
            "CausalForest FS: Multi-Treatment (%d Gruppen) → binarisiere zu "
            "Control(0) vs. AnyTreatment(1) für GRF Feature-Importances.",
            n_treatments,
        )
        T_np = (T_np > 0).astype(np.float64)

    # ── Subsampling bei großen Datensätzen ──
    # Feature-Importance-Rankings stabilisieren sich ab ~50-100k Zeilen.
    # Stratifiziert nach Treatment, damit die Balance erhalten bleibt.
    max_rows = 100_000
    n_orig = len(X_np)
    if n_orig > max_rows:
        rng = np.random.RandomState(seed)
        idx_list = []
        for t_val in np.unique(T_np):
            t_mask = np.where(T_np == t_val)[0]
            frac = len(t_mask) / n_orig
            n_keep = max(1, int(frac * max_rows))
            idx_list.append(rng.choice(t_mask, size=min(n_keep, len(t_mask)), replace=False))
        idx = np.sort(np.concatenate(idx_list))
        X_np = X_np[idx]
        T_np = T_np[idx]
        Y_np = Y_np[idx]
        _logger.info(
            "CausalForest FS: Subsampling %d → %d Zeilen (stratifiziert nach T).",
            n_orig, len(X_np),
        )

    # ── n_jobs-Entscheidung ──
    # In einem joblib-Thread: n_jobs=1 (GRF spawnt intern Prozesse → fork-Deadlock).
    # Im Hauptthread: n_jobs wie übergeben (typisch -1 = alle Kerne).
    effective_n_jobs = 1 if in_thread else n_jobs

    # ── Fit ──
    cf = CausalForest(
        n_estimators=100,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=effective_n_jobs,
    )

    _logger.info(
        "CausalForest FS: fit(%d×%d, T unique=%d, n_estimators=100, n_jobs=%d%s)...",
        X_np.shape[0], X_np.shape[1], len(np.unique(T_np)), effective_n_jobs,
        ", in_thread" if in_thread else "",
    )
    cf.fit(X_np, T_np, Y_np)

    # ── Feature-Importances extrahieren ──
    imp = None
    try:
        if hasattr(cf, "feature_importances_"):
            raw = cf.feature_importances_
            if raw is not None:
                imp = np.asarray(raw).ravel()
    except Exception as e:
        _logger.warning("CausalForest feature_importances_ fehlgeschlagen: %s", e)

    if imp is None:
        try:
            if hasattr(cf, "feature_importances"):
                raw_fn = cf.feature_importances
                raw = raw_fn() if callable(raw_fn) else raw_fn
                if raw is not None:
                    imp = np.asarray(raw).ravel()
        except Exception as e:
            _logger.warning("CausalForest feature_importances() fehlgeschlagen: %s", e)

    if imp is None:
        _logger.warning(
            "CausalForest: keine feature_importances verfügbar. "
            "Fallback auf Nullen. Prüfe EconML-Version (>=0.15 empfohlen)."
        )
        imp = np.zeros(len(feature_names))

    if len(imp) != len(feature_names):
        _logger.warning(
            "CausalForest: feature_importances Länge %d ≠ Features %d. Fallback auf Nullen.",
            len(imp), len(feature_names),
        )
        imp = np.zeros(len(feature_names))

    return pd.Series(imp, index=feature_names, name="causal_forest").sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Top-Prozent + Union
# ---------------------------------------------------------------------------

def _top_pct_features(importance: pd.Series, top_pct: float, n_total: int) -> List[str]:
    """Gibt die Top-X% Features zurück (mindestens 1)."""
    n_keep = max(1, int(np.ceil(top_pct / 100.0 * n_total)))
    return list(importance.sort_values(ascending=False).head(n_keep).index)


def compute_importances(
    methods: List[str],
    X: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    seed: int,
    n_jobs: int = -1,
    parallel_methods: bool = False,
) -> Dict[str, pd.Series]:
    """Berechnet Feature-Importances für die angegebenen Methoden.

    Returns
    -------
    Dict mit Methodennamen als Keys und Importance-Serien als Values.

    Parameters
    ----------
    parallel_methods : bool
        Bei True werden unabhängige Methoden parallel ausgeführt (Level 3/4).
        Jede Methode bekommt dann n_jobs=1 für ihren internen Fit, da die
        Parallelisierung auf Methoden-Ebene stattfindet.

    Hinweis
    -------
    Die Methode ``causal_forest`` (GRF) kann keine fehlenden Werte verarbeiten.
    Wenn X fehlende Werte enthält, wird diese Methode automatisch übersprungen
    und eine Warnung ausgegeben. LightGBM-basierte Methoden sind davon nicht
    betroffen, da LightGBM fehlende Werte nativ unterstützt.
    """
    results: Dict[str, pd.Series] = {}
    has_missing = X.isnull().any().any()

    effective_methods = []
    for method in methods:
        if method == "none":
            continue
        if method == "causal_forest" and has_missing:
            n_missing_cols = int(X.isnull().any().sum())
            _logger.warning(
                "Feature-Selektion: Methode 'causal_forest' (GRF) übersprungen – "
                "Daten enthalten fehlende Werte (%d Spalten betroffen). "
                "GRF kann keine fehlenden Werte verarbeiten. "
                "Die übrigen Methoden werden weiterhin berechnet.",
                n_missing_cols,
            )
            continue
        effective_methods.append(method)

    if not effective_methods:
        return results

    def _run_method(method: str, method_n_jobs: int, in_thread: bool = False) -> Tuple[str, Optional[pd.Series]]:
        try:
            if method == "lgbm_importance":
                return method, _lgbm_gain_importance(X, Y, seed, n_jobs=method_n_jobs)
            elif method == "lgbm_permutation":
                return method, _lgbm_permutation_importance(X, Y, seed, n_jobs=method_n_jobs)
            elif method == "causal_forest":
                _logger.info(
                    "CausalForest FS: X=%s (dtypes: %d numeric, %d category), "
                    "T=%s (unique=%d), Y=%s, n_jobs=%d, in_thread=%s",
                    X.shape,
                    X.select_dtypes(include=[np.number]).shape[1],
                    X.select_dtypes(include=["category"]).shape[1],
                    T.shape, len(np.unique(T)), Y.shape, method_n_jobs, in_thread,
                )
                return method, _causal_forest_importance(X, T, Y, seed, n_jobs=method_n_jobs, in_thread=in_thread)
            else:
                _logger.warning("Unbekannte Feature-Selection-Methode: '%s', überspringe.", method)
                return method, None
        except Exception:
            _logger.warning("Feature-Importance '%s' fehlgeschlagen.", method, exc_info=True)
            return method, None

    if parallel_methods and len(effective_methods) >= 2:
        # Methoden laufen sequentiell, aber jede bekommt ALLE Kerne.
        # Thread-Parallelisierung ist nicht möglich wenn CausalForest dabei ist
        # (GRF spawnt intern joblib-Prozesse → fork-aus-Thread → Deadlock),
        # und auch reine LightGBM-Kombinationen profitieren kaum, da OpenMP
        # bereits alle Kerne pro Fit nutzt.
        # Sequentiell mit n_jobs=-1 ist einfacher, sicherer und bei
        # Subsampling + voller Kernauslastung schnell genug.
        _logger.info(
            "Feature-Selektion: %d Methoden sequentiell (alle Kerne pro Methode).",
            len(effective_methods),
        )

    # Sequentiell — jede Methode im Hauptthread mit voller Kernauslastung
    import os
    all_cores = os.cpu_count() or 1
    effective_n_jobs = 1 if n_jobs == 1 else -1  # Level 1 → 1 Kern, sonst alle
    for method in effective_methods:
        _, imp = _run_method(method, effective_n_jobs, False)
        if imp is not None:
            results[method] = imp

    return results


def select_features_by_importance(
    X: pd.DataFrame,
    importances: Dict[str, pd.Series],
    top_pct: float,
    max_features: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    """Wählt Features per Top-Prozent-Union aus allen Importance-Methoden.

    Parameters
    ----------
    X : Feature-DataFrame
    importances : Dict von Methodenname -> Importance-Serie
    top_pct : Prozent der Features, die pro Methode behalten werden
    max_features : Absolute Obergrenze nach Union

    Returns
    -------
    X_filtered : Gefilterter DataFrame
    removed : Liste der entfernten Spaltennamen
    top_per_method : Dict mit den Top-Features pro Methode (für Logging)
    """
    if not importances:
        return X.copy(), [], {}

    n_total = X.shape[1]
    all_keep: set = set()
    top_per_method: Dict[str, List[str]] = {}

    for method_name, imp in importances.items():
        # Nur Features berücksichtigen, die in X vorhanden sind
        imp = imp.reindex(X.columns).dropna()
        if imp.empty:
            continue
        top_features = _top_pct_features(imp, top_pct, n_total)
        top_per_method[method_name] = top_features
        all_keep.update(top_features)
        _logger.info(
            "Feature-Selection '%s': Top-%.0f%% = %d / %d Features.",
            method_name, top_pct, len(top_features), n_total,
        )

    if not all_keep:
        return X.copy(), [], top_per_method

    # Absolute Obergrenze anwenden (nach kombinierter Importance sortieren)
    if max_features is not None and len(all_keep) > int(max_features):
        # Bei Union mehrerer Methoden: mittlere Rank-Position als Tiebreaker
        rank_sum = pd.Series(0.0, index=X.columns)
        for imp in importances.values():
            imp_reindexed = imp.reindex(X.columns).fillna(0.0)
            rank_sum += imp_reindexed.rank(ascending=False)
        ranked = rank_sum.loc[list(all_keep)].sort_values()
        all_keep = set(ranked.head(int(max_features)).index)

    keep_ordered = [c for c in X.columns if c in all_keep]
    removed = [c for c in X.columns if c not in all_keep]

    _logger.info(
        "Feature-Selection Union: %d / %d Features behalten, %d entfernt.",
        len(keep_ordered), n_total, len(removed),
    )

    return X[keep_ordered].copy(), removed, top_per_method


# ---------------------------------------------------------------------------
# Legacy-kompatible Hilfsfunktionen
# ---------------------------------------------------------------------------

def calculate_feature_importance(model, X: pd.DataFrame, T, Y) -> pd.Series:
    """Berechnet Feature-Importance über ein kausales Modell (Legacy-Schnittstelle)."""
    try:
        model.fit(Y, T, X=X)
    except TypeError:
        model.fit(Y, T, X)
    if hasattr(model, "feature_importances_"):
        imps = getattr(model, "feature_importances_")
        return pd.Series(imps, index=X.columns).sort_values(ascending=False)
    if hasattr(model, "model_final_") and hasattr(model.model_final_, "feature_importances_"):
        imps = model.model_final_.feature_importances_
        return pd.Series(imps, index=X.columns).sort_values(ascending=False)
    return pd.Series(dtype=float)


def remove_low_importance_features(
    X: pd.DataFrame,
    importance: pd.Series,
    importance_threshold_pct_of_max: float = 2.0,
    max_features: int | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Entfernt Features unter einem Schwellwert (Legacy-Schnittstelle)."""
    if importance.empty:
        return X, []
    max_imp = float(importance.max())
    keep = importance[importance >= (importance_threshold_pct_of_max / 100.0) * max_imp].index.tolist()
    if max_features is not None and len(keep) > int(max_features):
        keep = importance.loc[keep].sort_values(ascending=False).head(int(max_features)).index.tolist()
    removed = [c for c in X.columns if c not in keep]
    return X[keep].copy(), removed
