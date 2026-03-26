from __future__ import annotations

"""Orchestrierung der Analyse-Pipeline.
Die Analyse-Pipeline ist der zentrale Einstiegspunkt für Entwicklungs- und
Evaluationsläufe.
Aufgaben:
- Einlesen der vorbereiteten Input-Dateien (X/T/Y, optional S)
- optionale Feature-Filterung (Korrelation / Importance)
- optionales Base-Learner-Tuning mit Optuna
- Training der konfigurierten kausalen Learner
- Cross-Predictions (standardmäßig) für robuste Evaluation
- Uplift-Metriken (Qini, AUUC, Uplift@k, Policy Value)
- Logging nach MLflow
- optional: synchroner Bundle-Export (Production-Artefakte)
Wichtig:
Die Production-Pipeline arbeitet ausschließlich auf Bundles.
Der Bundle-Export ist deshalb bewusst am Ende des Analyselaufs angesiedelt."""

from dataclasses import dataclass
import copy
import gc
import tempfile
import time
from typing import Any, Dict, Optional
import json
import logging
import os

import numpy as np
import pandas as pd

from rubin.artifacts import ArtifactBundler
from rubin.feature_selection import (
    compute_importances,
    select_features_by_importance,
    remove_highly_correlated_features,
)
from rubin.model_management import ModelEntry, choose_champion, write_registry, float_metrics
from rubin.model_registry import ModelContext, ModelRegistry, default_registry
from rubin.preprocessing import build_simple_preprocessor_from_dataframe
from rubin.settings import AnalysisConfig
from rubin.training import _predict_effect, train_and_crosspredict_bt_bo, is_multi_treatment, SurrogateTreeWrapper, SURROGATE_MODEL_NAME, SURROGATE_CF_NAME
from rubin.utils.data_utils import stratified_train_test_split, reduce_mem_usage
from rubin.utils.io_utils import read_table
from rubin.utils.categorical_patch import patch_categorical_features
from rubin.tuning_optuna import BaseLearnerTuner, FinalModelTuner, _first_crossfit_train_indices, build_base_learner
from rubin.utils.uplift_metrics import auuc, policy_value, qini_coefficient, uplift_at_k, uplift_curve, mt_eval_summary
from rubin.evaluation.drtester_plots import (
    CustomDRTester,
    evaluate_cate_with_plots,
    fit_drtester_nuisance,
    generate_cate_distribution_plot,
    generate_sklift_plots,
    plot_custom_qini_curve,
    save_dataframe_as_png,
    policy_value_comparison_plots,
)
from rubin.reporting import ReportCollector, generate_html_report


@dataclass
class AnalysisResult:
    """Rückgabeobjekt der Analyse-Pipeline."""

    models: Dict[str, Any]
    predictions: Dict[str, pd.DataFrame]
    removed_features: Dict[str, Any]
    eval_summary: Dict[str, Dict[str, float]]


def _log_temp_artifact(mlflow, content_fn, filename: str) -> None:
    """Schreibt ein temporäres Artefakt, loggt es nach MLflow und räumt auf.

    Vermeidet Dateikonflikte bei parallelen Runs, indem ein temporäres
    Verzeichnis verwendet wird.

    Parameters
    ----------
    mlflow:
        Das MLflow-Modul.
    content_fn:
        Callable, das den vollständigen Dateipfad als Argument erhält und die
        Datei schreibt.
    filename:
        Gewünschter Dateiname im MLflow-Artefakt-Store.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        content_fn(path)
        mlflow.log_artifact(path)


class AnalysisPipeline:
    """Führt einen kompletten Analyselauf gemäß Konfiguration aus."""

    _logger = logging.getLogger("rubin.analysis")

    def __init__(self, cfg: AnalysisConfig, registry: Optional[ModelRegistry] = None) -> None:
        self.cfg = cfg
        self.registry = registry or default_registry()

    def _read_table(self, path: str, use_index: bool = True) -> pd.DataFrame:
        """Liest eine Tabelle aus CSV oder Parquet."""
        return read_table(path, use_index=use_index)

    def _load_inputs(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Lädt X, T, Y, optional S und optional eval_mask."""
        X = self._read_table(self.cfg.data_files.x_file)
        T_df = self._read_table(self.cfg.data_files.t_file)
        Y_df = self._read_table(self.cfg.data_files.y_file)
        T = T_df["T"].to_numpy()
        Y = Y_df["Y"].to_numpy()

        S: Optional[np.ndarray] = None
        S_df: Optional[pd.DataFrame] = None
        if self.cfg.data_files.s_file:
            try:
                col = self.cfg.historical_score.column
                S_df = self._read_table(self.cfg.data_files.s_file)
                S = S_df[col].to_numpy(dtype=float)
                n_nan = int(np.isnan(S).sum())
                if n_nan > 0:
                    self._logger.info("Historischer Score: %d NaN-Werte durch 0 ersetzt.", n_nan)
                    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
            except FileNotFoundError:
                S = None

        # ── Eval-Maske laden (Train Many, Evaluate One) ──
        # Wird VOR df_frac geladen, damit sie mit-subsampled wird.
        eval_mask: Optional[np.ndarray] = None
        if self.cfg.data_files.eval_mask_file:
            try:
                _raw_mask = np.load(self.cfg.data_files.eval_mask_file).astype(bool)
                if len(_raw_mask) == len(X):
                    eval_mask = _raw_mask
                else:
                    self._logger.warning(
                        "eval_mask Länge (%d) passt nicht zu X (%d) — Maske wird ignoriert.",
                        len(_raw_mask), len(X),
                    )
            except Exception:
                self._logger.warning("eval_mask konnte nicht geladen werden.", exc_info=True)

        if self.cfg.data_processing.df_frac:
            X = X.sample(frac=float(self.cfg.data_processing.df_frac), random_state=self.cfg.constants.random_seed)
            idx = X.index
            T = T_df["T"].loc[idx].to_numpy()
            Y = Y_df["Y"].loc[idx].to_numpy()
            if S is not None and S_df is not None:
                try:
                    col = self.cfg.historical_score.column
                    S = S_df[col].loc[idx].to_numpy(dtype=float)
                    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    S = None
            if eval_mask is not None:
                eval_mask = eval_mask[idx.to_numpy()]

        if self.cfg.data_files.dtypes_file:
            try:
                with open(self.cfg.data_files.dtypes_file, "r", encoding="utf-8") as f:
                    dtypes = json.load(f)
                for c, dt in dtypes.items():
                    if c in X.columns:
                        try:
                            X[c] = X[c].astype(dt)
                        except Exception:
                            pass
            except FileNotFoundError:
                pass

        # Memory-Reduktion: Datentypen downcasten (float64→float32, int64→int32 etc.)
        if getattr(self.cfg.data_processing, "reduce_memory", True):
            n_before = X.memory_usage(deep=True).sum()
            X = reduce_mem_usage(X)
            n_after = X.memory_usage(deep=True).sum()
            self._logger.info(
                "Memory-Reduktion: %.1f MB → %.1f MB (%.0f%% gespart).",
                n_before / 1e6, n_after / 1e6, (1 - n_after / max(n_before, 1)) * 100,
            )

        # Kategorische Spalten explizit als category-Dtype markieren.
        # Drei Quellen (in Prioritätsreihenfolge):
        #   1. cfg.data_processing.categorical_columns (explizit vom User)
        #   2. dtypes.json (von DataPrep → enthält "category" wenn korrekt erzeugt)
        #   3. Parquet-Metadaten (preserven category-Dtype automatisch)
        # Quelle 2+3 werden bereits oben bei dtypes-Anwendung / Parquet-Load abgedeckt.
        # Hier ergänzen wir Quelle 1 als Override: Falls der User explizite
        # categorical_columns angegeben hat, werden diese zu category konvertiert.
        cat_cols = getattr(self.cfg.data_processing, "categorical_columns", None)
        if cat_cols:
            n_cat = 0
            for c in cat_cols:
                if c in X.columns and not isinstance(X[c].dtype, pd.CategoricalDtype):
                    try:
                        X[c] = X[c].astype("category")
                        n_cat += 1
                    except Exception:
                        pass
            if n_cat > 0:
                self._logger.info("Kategorische Spalten aus Config angewendet: %d Spalten als category markiert.", n_cat)

        self._logger.info(
            "Daten geladen: X=%s, T=%s (unique=%s), Y=%s (unique=%s), S=%s",
            X.shape, T.shape, np.unique(T).tolist(), Y.shape, np.unique(Y).tolist(),
            S.shape if S is not None else "None",
        )

        # Defensiv: Index zurücksetzen, damit X.iloc[i] ↔ T[i] ↔ Y[i] ↔ S[i]
        # garantiert position-konsistent sind (nach sample/holdout wäre X.index nicht-konsekutiv).
        X = X.reset_index(drop=True)

        if eval_mask is not None:
            self._logger.info(
                "Eval-Maske: %d von %d Zeilen für Evaluation (Train Many, Evaluate One).",
                int(eval_mask.sum()), len(eval_mask),
            )

        return X, T, Y, S, eval_mask

    # ------------------------------------------------------------------
    # Submethoden für die einzelnen Pipeline-Schritte
    # ------------------------------------------------------------------

    def _run_feature_selection(self, cfg, X, T, Y, mlflow):
        """Feature-Selektion: Importance-basierte Filterung (Union) und Korrelationsfilter."""
        removed: Dict[str, Any] = {}
        fs = cfg.feature_selection

        # Importance-basierte Filterung
        effective_methods = [m for m in fs.methods if m != "none"]
        if fs.enabled and effective_methods:
            importances = compute_importances(
                methods=effective_methods, X=X, T=T, Y=Y,
                seed=cfg.constants.random_seed,
                n_jobs=1 if cfg.constants.parallel_level <= 1 else -1,
                parallel_methods=cfg.constants.parallel_level >= 3,
            )

            if importances:
                X, removed_imp, top_per_method = select_features_by_importance(
                    X, importances, top_pct=fs.top_pct,
                    max_features=fs.max_features,
                )
                removed["importance"] = removed_imp

                # Logging: Methoden, Anzahl Features pro Methode, Union-Ergebnis
                mlflow.log_param("feature_selection_methods", ",".join(effective_methods))
                mlflow.log_param("feature_selection_top_pct", fs.top_pct)
                mlflow.log_metric("features_after_importance", len(X.columns))
                mlflow.log_metric("features_removed_importance", len(removed_imp))
                for method_name, top_features in top_per_method.items():
                    mlflow.log_metric(f"features_top_{method_name}", len(top_features))

                # Importances als Artefakte speichern
                for method_name, imp in importances.items():
                    def _write_imp(p, _imp=imp, _method=method_name):
                        df = _imp.rename_axis("feature").reset_index()
                        df.columns = ["feature", _method]
                        df.to_csv(p, index=False)
                    _log_temp_artifact(mlflow, _write_imp, f"feature_importance_{method_name}.csv")

                def _write_removed(p):
                    with open(p, "w", encoding="utf-8") as fh:
                        json.dump(removed_imp, fh, ensure_ascii=False, indent=2)
                _log_temp_artifact(mlflow, _write_removed, "removed_features_imp.json")

        # Korrelationsfilter
        if fs.enabled and float(fs.correlation_threshold) > 0:
            X, removed_corr = remove_highly_correlated_features(
                X, correlation_threshold=fs.correlation_threshold,
                correlation_methods=["pearson", "spearman"],
            )
            removed["high_correlation"] = removed_corr

            def _write_corr(p):
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(removed_corr, fh, ensure_ascii=False, indent=2)

            _log_temp_artifact(mlflow, _write_corr, "removed_features_corr.json")
        else:
            removed["high_correlation"] = []

        return X, removed

    def _run_tuning(self, cfg, X, T, Y, mlflow):
        """Base-Learner-Tuning (Optuna) und optionales Final-Model-Tuning."""
        tuned_params_by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}

        if cfg.tuning.enabled:
            tuner = BaseLearnerTuner(cfg)
            self._logger.info(
                "Starte Tuning: X=%s, Y=%s (unique=%s), T=%s (unique=%s)",
                X.shape, Y.shape, np.unique(Y).tolist(), T.shape, np.unique(T).tolist(),
            )
            tuned_params_by_model = tuner.tune_all(cfg.models.models_to_train, X=X, Y=Y, T=T)

            # Modellgüte der Base-Learner-Tuning-Tasks loggen
            for task_key, score in tuner.best_scores.items():
                mlflow.log_metric(f"tuning_best__{task_key}", score)

            # Kurzform: gut sichtbare Top-Level-Metriken pro Modell+Rolle
            for model_name, roles_dict in tuned_params_by_model.items():
                for role in roles_dict:
                    if role == "default":
                        continue
                    # Task-Key rekonstruieren und Score zuordnen
                    try:
                        tk = tuner._task_key(model_name, role)
                        if tk in tuner.best_scores:
                            short_name = f"bl_score__{model_name}__{role}"
                            mlflow.log_metric(short_name, tuner.best_scores[tk])
                    except Exception:
                        pass

            # Beste Parameter auch als MLflow-Params loggen (für schnelle Übersicht)
            for model_name, roles_dict in tuned_params_by_model.items():
                for role, params in roles_dict.items():
                    if isinstance(params, dict):
                        for pk, pv in params.items():
                            try:
                                mlflow.log_param(f"hp__{model_name}__{role}__{pk}", pv)
                            except Exception:
                                pass  # MLflow Param-Limit (500 chars) oder Duplikate

            # Tuned Params + Scores als JSON-Artefakt (leicht inspizierbar)
            def _write_tuned_params(p):
                import json
                payload = {
                    "params": {k: {r: dict(v) for r, v in roles.items()} for k, roles in tuned_params_by_model.items()},
                    "scores": dict(tuner.best_scores),
                }
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2, ensure_ascii=False, default=float)
            _log_temp_artifact(mlflow, _write_tuned_params, "tuned_baselearner_params.json")

            # Für HTML-Report sammeln
            if hasattr(self, '_report'):
                self._report.tuning_scores.update(tuner.best_scores)
                # Tuning-Plan (Task-Sharing-Tabelle)
                try:
                    plan = tuner._build_plan(cfg.models.models_to_train)
                    plan_list = []
                    for task in sorted(plan.values(), key=tuner._task_priority):
                        role_label = {
                            "outcome": "Outcome (Classifier)",
                            "outcome_regression": "Outcome (Regressor)",
                            "propensity": "Propensity (T)",
                            "grouped_outcome": "Grouped Outcome (Classifier)",
                            "grouped_outcome_regression": "Grouped Outcome (Regressor)",
                            "pseudo_effect": "Pseudo-Effekt",
                        }.get(task.objective_family, task.objective_family)
                        sig_parts = [cfg.base_learner.type, task.estimator_task, task.sample_scope,
                                     "with_t" if task.uses_treatment_feature else "no_t", task.target_name]
                        plan_list.append({
                            "task_key": task.key,
                            "role": role_label,
                            "models": [m for m, _ in task.roles],
                            "signature": " | ".join(sig_parts),
                        })
                    self._report.add_tuning_plan(plan_list)
                except Exception:
                    pass
                # Best Hyperparameter (pro Tuning-Task)
                try:
                    for task in plan.values():
                        if task.roles:
                            mname, role = task.roles[0]
                            params = tuned_params_by_model.get(mname, {}).get(role, {})
                            if params:
                                self._report.add_best_params(task.key, dict(params))
                except Exception:
                    pass

        if getattr(cfg, "final_model_tuning", None) is not None and cfg.final_model_tuning.enabled:
            final_tuner = FinalModelTuner(cfg)
            # Nur die in final_model_tuning.models konfigurierten Modelle tunen.
            # None = alle FMT-fähigen Modelle (NonParamDML, DRLearner).
            fmt_models = cfg.final_model_tuning.models
            for mname in cfg.models.models_to_train:
                if fmt_models is not None and mname not in fmt_models:
                    continue
                current = dict(tuned_params_by_model.get(mname, {}) or {})
                add = final_tuner.tune_final_model(
                    model_name=mname, X=X, Y=Y, T=T,
                    base_type=cfg.base_learner.type,
                    base_fixed_params=dict(cfg.base_learner.fixed_params or {}),
                    fmt_fixed_params=dict(cfg.final_model_tuning.fixed_params or {}),
                    tuned_roles=current,
                    crosspred_splits=cfg.data_processing.cross_validation_splits,
                )
                if add:
                    current.update(add)
                    tuned_params_by_model[mname] = current

            # Modellgüte der Final-Model-Tuning-Tasks loggen
            for task_key, score in final_tuner.best_scores.items():
                mlflow.log_metric(f"tuning_best__{task_key}", score)
            if hasattr(self, '_report'):
                # FMT-Scores NICHT in tuning_scores mischen (die gehören zum BL-Tuning).
                # Stattdessen in fmt_info["best_scores"] für die FMT-Report-Sektion.
                # FMT-Plan für Report
                try:
                    fmt_n_trials = cfg.final_model_tuning.n_trials
                    fmt_single_fold = getattr(cfg.final_model_tuning, "single_fold", False)
                    fmt_cv = cfg.data_processing.cross_validation_splits
                    fmt_plan_list = []
                    for mname in cfg.models.models_to_train:
                        name_lower = (mname or "").lower()
                        if name_lower == "nonparamdml":
                            fmt_plan_list.append({
                                "model": mname, "method": "RScorer", "trials": fmt_n_trials,
                                "fits_per_trial": 1, "total_fits": fmt_n_trials,
                                "note": "RScorer wird einmal vorab gefittet (berechnet Residuen mit CV). Danach 1 Kandidaten-Fit pro Trial.",
                            })
                        elif name_lower == "drlearner":
                            fpt = 1 if fmt_single_fold else fmt_cv
                            fmt_plan_list.append({
                                "model": mname, "method": f"score() {'Single-Fold' if fmt_single_fold else 'CV'}",
                                "trials": fmt_n_trials, "fits_per_trial": fpt,
                                "total_fits": fmt_n_trials * fpt,
                                "note": f"{'Single-Fold aktiv: 1 Fit pro Trial statt ' + str(fmt_cv) + '.' if fmt_single_fold else str(fmt_cv) + ' CV-Folds pro Trial.'} Kein RScorer verfügbar (andere Nuisance-Struktur).",
                            })
                    if fmt_plan_list:
                        self._report.add_fmt_plan(fmt_plan_list)
                    # FMT-Info
                    self._report.add_fmt_info({
                        "n_trials": fmt_n_trials,
                        "single_fold": fmt_single_fold,
                        "cv": fmt_cv,
                        "models": [m for m in cfg.models.models_to_train if (m or "").lower() in {"nonparamdml", "drlearner"}],
                        "best_scores": {
                            k.replace("final__", ""): v
                            for k, v in final_tuner.best_scores.items()
                        },
                    })
                except Exception:
                    pass
                # FMT Best Params
                try:
                    for mname in cfg.models.models_to_train:
                        final_params = tuned_params_by_model.get(mname, {}).get("model_final", {})
                        if final_params:
                            self._report.add_fmt_best_params(mname, dict(final_params))
                except Exception:
                    pass

        return tuned_params_by_model

    def _run_training(self, cfg, X, T, Y, tuned_params_by_model, holdout_data, mlflow):
        """Training der kausalen Learner + Cross-Predictions bzw. Holdout-Scoring."""
        models: Dict[str, Any] = {}
        preds: Dict[str, pd.DataFrame] = {}
        has_missing = X.isnull().any().any()

        for name in cfg.models.models_to_train:
            # CausalForestDML (GRF) kann keine fehlenden Werte verarbeiten.
            # Bei fehlenden Werten wird das Modell übersprungen.
            if name == "CausalForestDML" and has_missing:
                n_missing_cols = int(X.isnull().any().sum())
                self._logger.warning(
                    "CausalForestDML übersprungen – Daten enthalten fehlende Werte "
                    "(%d Spalten betroffen). GRF-basierte Modelle können keine "
                    "fehlenden Werte verarbeiten. Alle anderen Modelle (mit "
                    "LightGBM/CatBoost als Base Learner) sind davon nicht betroffen.",
                    n_missing_cols,
                )
                mlflow.log_param(f"model_enabled__{name}", False)
                mlflow.log_param(f"model_skipped__{name}", "missing_values")
                continue

            mlflow.log_param(f"model_enabled__{name}", True)

            # parallel_jobs: Kerne pro Base-Learner-Fit.
            # Bei Level 3/4 laufen mehrere Folds parallel → pro Fold weniger Kerne,
            # um CPU-Übersubskription zu vermeiden.
            # CatBoost braucht mehr Threads pro Fit als LightGBM (Symmetric Tree).
            pl = cfg.constants.parallel_level
            is_catboost = (cfg.base_learner.type or "").lower() == "catboost"
            if pl <= 1:
                pj = 1
            elif pl <= 2:
                pj = -1  # alle Kerne, Folds sequentiell → kein Konflikt
            else:
                # Level 3/4: Folds parallel → Kerne aufteilen
                n_cpus = os.cpu_count() or 1
                n_cv = cfg.data_processing.cross_validation_splits
                if pl >= 4:
                    if is_catboost:
                        # CatBoost: weniger parallele Folds, mehr Threads pro Fit
                        n_fold_workers = min(n_cv, max(1, n_cpus // 4))
                    else:
                        n_fold_workers = min(n_cv, n_cpus)
                else:
                    n_fold_workers = min(n_cv, max(1, n_cpus // 4), n_cpus)
                pj = max(1, n_cpus // max(1, n_fold_workers))

            ctx = ModelContext(
                seed=cfg.constants.random_seed,
                base_learner_type=cfg.base_learner.type,
                base_fixed_params=dict(cfg.base_learner.fixed_params or {}),
                fmt_fixed_params=dict(cfg.final_model_tuning.fixed_params or {}),
                tuned_params=tuned_params_by_model.get(name, {}),
                parallel_jobs=pj,
            )

            if name.lower() == "causalforestdml":
                ctx.tuned_params = dict(ctx.tuned_params or {})
                forest_defaults = dict(getattr(cfg.causal_forest, "forest_fixed_params", {}) or {})
                if forest_defaults:
                    existing = dict(ctx.tuned_params.get("forest") or {})
                    ctx.tuned_params["forest"] = {**forest_defaults, **existing}

            model = self.registry.create(name, ctx)

            # Debug: Effektive Parameter für jede Rolle loggen
            if name.lower() in {"nonparamdml", "drlearner"}:
                final_params = dict(ctx.fmt_fixed_params or {})
                explicit_final = ctx.tuned_params.get("model_final")
                if explicit_final:
                    final_params.update(explicit_final)
                self._logger.info(
                    "%s model_final effektive Params: %s (explicit_tuned=%s, fmt_fixed=%s)",
                    name, {k: v for k, v in final_params.items() if k in (
                        "min_child_samples", "num_leaves", "max_depth", "n_estimators",
                        "min_data_in_leaf", "depth", "iterations", "min_child_weight",
                        "colsample_bytree", "rsm", "path_smooth", "model_size_reg",
                        "reg_lambda", "reg_alpha", "l2_leaf_reg",
                    )},
                    "ja" if explicit_final else "nein",
                    bool(ctx.fmt_fixed_params),
                )

            # Warnung: DML-Modelle mit model_final profitieren stark von FinalModelTuning
            if name.lower() in {"nonparamdml", "drlearner"} and not cfg.final_model_tuning.enabled:
                has_explicit_final = bool(ctx.tuned_params.get("model_final"))
                if not has_explicit_final:
                    self._logger.warning(
                        "%s: model_final hat keine getunten Parameter. "
                        "Das CATE-Effektmodell nutzt nur base_fixed_params. "
                        "Empfehlung: final_model_tuning.enabled=true aktivieren, "
                        "um model_final über R-Score zu optimieren.",
                        name,
                    )

            if name.lower() == "causalforestdml" and getattr(cfg.causal_forest, "use_econml_tune", False):
                try:
                    tr_idx = _first_crossfit_train_indices(len(X), T, n_splits=cfg.data_processing.cross_validation_splits, seed=cfg.constants.random_seed)
                    X_tune, Y_tune, T_tune = X.iloc[tr_idx], Y[tr_idx], T[tr_idx]
                    tune_max = getattr(cfg.causal_forest, "tune_max_rows", None)
                    if tune_max is None:
                        tune_max = cfg.tuning.max_tuning_rows if cfg.tuning.enabled else None
                    if tune_max is not None and len(X_tune) > int(tune_max):
                        rng = np.random.RandomState(cfg.constants.random_seed)
                        idx = rng.choice(np.arange(len(X_tune)), size=int(tune_max), replace=False)
                        X_tune, Y_tune, T_tune = X_tune.iloc[idx], Y_tune[idx], T_tune[idx]
                    model.tune(Y_tune, T_tune, X=X_tune, params=getattr(cfg.causal_forest, "econml_tune_params", "auto"))
                    mlflow.log_param("causal_forest__econml_tune", True)
                except Exception:
                    self._logger.warning("CausalForestDML EconML-Tune fehlgeschlagen.", exc_info=True)
                    mlflow.log_param("causal_forest__econml_tune", False)

            if holdout_data is None:
                df_pred = train_and_crosspredict_bt_bo(
                    model=model, X=X, Y=Y, T=T,
                    n_splits=cfg.data_processing.cross_validation_splits,
                    model_name=name, random_state=cfg.constants.random_seed,
                    parallel_level=cfg.constants.parallel_level,
                )
                # Im CV-Pfad enthält `model` das untrainierte Basisobjekt. Für den
                # Bundle-Export wird das Modell später separat gefittet.
                current_model = model
            else:
                X_test, T_test, Y_test, _ = holdout_data
                model.fit(Y, T, X=X)
                current_model = model
                test_pred = _predict_effect(model, X_test)
                df_pred = pd.DataFrame({"Y": Y_test, "T": T_test})
                if test_pred.ndim == 2 and test_pred.shape[1] > 1:
                    n_effects = test_pred.shape[1]
                    for k in range(n_effects):
                        df_pred[f"Predictions_{name}_T{k+1}"] = test_pred[:, k]
                        df_pred[f"Train_{name}_T{k+1}"] = np.nan
                    best_eff = np.nanmax(test_pred, axis=1)
                    best_arm = np.nanargmax(test_pred, axis=1) + 1
                    df_pred[f"OptimalTreatment_{name}"] = np.where(best_eff > 0, best_arm, 0)
                else:
                    df_pred[f"Predictions_{name}"] = test_pred
                    df_pred[f"Train_{name}"] = np.nan

            preds[name] = df_pred
            models[name] = current_model
            # Predictions mit voller Präzision speichern (float64, 10 signifikante Stellen)
            _log_temp_artifact(mlflow, lambda p, _df=df_pred: _df.to_csv(p, index=False, float_format="%.10g"), f"predictions_{name}.csv")

            # Diagnose: Warnung wenn CATEs kollabiert sind
            pred_cols = [c for c in df_pred.columns if c.startswith(f"Predictions_{name}")]
            for pc in pred_cols:
                vals = df_pred[pc].dropna()
                if len(vals) == 0:
                    continue
                n_unique = vals.nunique()
                val_range = float(vals.max() - vals.min())
                cv_folds = cfg.data_processing.cross_validation_splits

                if (vals == 0).all():
                    self._logger.warning(
                        "WARNUNG: %s hat ausschließlich CATE=0 Predictions! "
                        "Mögliche Ursachen: (1) Daten haben keinen Treatment-Effekt, "
                        "(2) Modell verwendet predict statt predict_proba (Meta-Learner), "
                        "(3) Extrem unbalancierte Klassen.", pc,
                    )
                elif n_unique <= cv_folds + 1 and val_range < abs(vals.mean()) * 0.1:
                    # CATE hat nur so viele Werte wie CV-Folds → model_final/CATE-Modell
                    # ist zu einem Intercept kollabiert (1 Wert pro Fold).
                    self._logger.warning(
                        "WARNUNG: %s hat nur %d distinkte Werte bei %d Folds (Range=%.2e, Mean=%.2e). "
                        "Das CATE-Modell ist wahrscheinlich zu einem Intercept kollabiert. "
                        "Empfehlungen: (1) final_model_tuning.enabled=true aktivieren, "
                        "(2) Prüfen ob base_fixed_params zu restriktiv sind "
                        "(min_child_samples, num_leaves, max_depth), "
                        "(3) Mehr Features oder Feature-Engineering.",
                        pc, n_unique, cv_folds, val_range, abs(vals.mean()),
                    )
                elif n_unique < 20 and len(vals) > 1000:
                    self._logger.warning(
                        "HINWEIS: %s hat nur %d distinkte Werte bei %d Samples. "
                        "Das Modell differenziert wenig zwischen Individuen.",
                        pc, n_unique, len(vals),
                    )
                else:
                    self._logger.info(
                        "%s: CATE min=%.6g, median=%.6g, max=%.6g, std=%.6g, "
                        "unique=%d/%d, non-zero=%d/%d",
                        pc, vals.min(), vals.median(), vals.max(), vals.std(),
                        n_unique, len(vals), (vals != 0).sum(), len(vals),
                    )

            gc.collect()  # Zwischen Modellen: deepcopy-Reste und Fold-Daten freigeben

        return models, preds

    def _run_evaluation(self, cfg, X, T, Y, S, holdout_data, preds, models, tuned_params_by_model, mlflow, eval_mask=None):
        """Uplift-Evaluation und Diagnose-Plots."""
        eval_summary: Dict[str, Dict[str, float]] = {}
        policy_values_dict: Dict[str, pd.DataFrame] = {}
        is_mt = is_multi_treatment(T)

        # Warnung: eval_mask wird bei holdout/external ignoriert
        if eval_mask is not None and holdout_data is not None:
            self._logger.warning(
                "eval_mask_file ist gesetzt, wird aber ignoriert, da validate_on='%s' "
                "einen eigenen Eval-Datensatz verwendet.",
                cfg.data_processing.validate_on,
            )

        # ── Bootstrap-Iterationen für DRTester Konfidenzintervalle ──
        n_bootstrap = 1000
        self._eval_n_bootstrap = n_bootstrap
        pl = cfg.constants.parallel_level
        import matplotlib.pyplot as plt

        # ── DRTester Nuisance EINMAL fitten (für alle Modelle gleich) ──
        # Beste Classifier-Params finden: DML model_y/model_t bevorzugt,
        # dann DRLearner model_propensity, dann base_learner.fixed_params
        _best_clf_params_y = dict(cfg.base_learner.fixed_params or {})
        _best_clf_params_t = dict(cfg.base_learner.fixed_params or {})
        for mname in ["NonParamDML", "ParamDML", "CausalForestDML", "DRLearner"]:
            roles = tuned_params_by_model.get(mname, {})
            if roles.get("model_y"):
                _best_clf_params_y = {**_best_clf_params_y, **roles["model_y"]}
                break
        for mname in ["NonParamDML", "ParamDML", "CausalForestDML", "DRLearner"]:
            roles = tuned_params_by_model.get(mname, {})
            if roles.get("model_t") or roles.get("model_propensity"):
                _best_clf_params_t = {**_best_clf_params_t, **(roles.get("model_t") or roles.get("model_propensity") or {})}
                break

        fitted_tester_bt = None  # Pre-fitted DRTester für BT
        fitted_tester_mt = {}    # Pre-fitted DRTester pro Arm für MT
        try:
            # DRTester-Nuisance-Modelle: Leichtere Varianten der getunten Modelle.
            # DRTester benötigt Nuisance-Predictions für DR-Outcomes. Die Qualität
            # muss gut sein, aber nicht perfekt — es geht um Diagnostik-Plots, nicht
            # um Champion-Selektion. Daher cappen wir n_estimators auf 100 (statt
            # potentiell 400+) und reduzieren den internen CV auf 3 Folds.
            # Speedup: ~6-7× für den Nuisance-Fit bei minimaler AUC-Einbuße (~0.5-1%).
            _drtester_n_estimators_cap = 100
            _drtester_cv = 3

            def _cap_estimators(params: dict) -> dict:
                """Begrenzt n_estimators/iterations für DRTester-Modelle."""
                p = dict(params)
                for key in ("n_estimators", "iterations"):
                    if key in p and int(p[key]) > _drtester_n_estimators_cap:
                        p[key] = _drtester_n_estimators_cap
                return p

            dr_params_y = _cap_estimators(_best_clf_params_y)
            dr_params_t = _cap_estimators(_best_clf_params_t)
            pj = 1 if cfg.constants.parallel_level <= 1 else -1
            model_reg = build_base_learner(cfg.base_learner.type, dr_params_y, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=pj)
            model_prop = build_base_learner(cfg.base_learner.type, dr_params_t, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=pj)

            X_val = holdout_data[0] if holdout_data is not None else X
            T_val = holdout_data[1] if holdout_data is not None else T
            Y_val = holdout_data[2] if holdout_data is not None else Y

            # Prüfe ob Train-Preds vorhanden (für irgendein Modell)
            has_train = any(
                any(c.startswith("Train_") and not np.all(np.isnan(dfp[c].to_numpy(dtype=float)))
                    for c in dfp.columns if c.startswith("Train_"))
                for dfp in preds.values()
            )

            if not is_mt:
                fitted_tester_bt = fit_drtester_nuisance(
                    model_regression=model_reg,
                    model_propensity=model_prop,
                    X_val=X_val, T_val=T_val, Y_val=Y_val,
                    X_train=X if has_train else None,
                    T_train=T if has_train else None,
                    Y_train=Y if has_train else None,
                    cv=_drtester_cv,
                )
                self._logger.info("DRTester Nuisance einmalig gefittet (BT, cv=%d, n_est≤%d). Wird für alle Modelle wiederverwendet.", _drtester_cv, _drtester_n_estimators_cap)
            else:
                # MT: Pro Arm einen binären DRTester fitten (Control vs. Arm k)
                K = len(np.unique(T_val))

                def _fit_arm(arm):
                    arm_mask_val = (T_val == 0) | (T_val == arm)
                    arm_X_val = X_val.loc[X_val.index[arm_mask_val]].copy() if isinstance(X_val, pd.DataFrame) else X_val[arm_mask_val]
                    arm_T_val = (T_val[arm_mask_val] == arm).astype(int)
                    arm_Y_val = Y_val[arm_mask_val]

                    arm_X_train, arm_T_train, arm_Y_train = None, None, None
                    if has_train:
                        arm_mask_tr = (T == 0) | (T == arm)
                        arm_X_train = X.iloc[np.where(arm_mask_tr)[0]].copy()
                        arm_T_train = (T[arm_mask_tr] == arm).astype(int)
                        arm_Y_train = Y[arm_mask_tr]

                    tester = fit_drtester_nuisance(
                        model_regression=build_base_learner(cfg.base_learner.type, dr_params_y, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=pj),
                        model_propensity=build_base_learner(cfg.base_learner.type, dr_params_t, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=pj),
                        X_val=arm_X_val, T_val=arm_T_val, Y_val=arm_Y_val,
                        X_train=arm_X_train, T_train=arm_T_train, Y_train=arm_Y_train,
                        cv=_drtester_cv,
                    )
                    return arm, tester

                if pl >= 3 and K > 2:
                    # Level 3/4: Arme parallel fitten
                    try:
                        from joblib import Parallel, delayed
                        import os
                        n_cpus = os.cpu_count() or 1
                        n_arm_workers = min(K - 1, n_cpus)
                        self._logger.info("DRTester MT: %d Arme parallel (n_jobs=%d).", K - 1, n_arm_workers)
                        results = Parallel(n_jobs=n_arm_workers, prefer="threads")(
                            delayed(_fit_arm)(arm) for arm in range(1, K)
                        )
                        for arm, tester in results:
                            fitted_tester_mt[arm] = tester
                    except Exception:
                        self._logger.warning("Parallele MT-Arm-Fits fehlgeschlagen, Fallback sequentiell.", exc_info=True)
                        for arm in range(1, K):
                            _, tester = _fit_arm(arm)
                            fitted_tester_mt[arm] = tester
                else:
                    for arm in range(1, K):
                        _, tester = _fit_arm(arm)
                        fitted_tester_mt[arm] = tester

                self._logger.info("DRTester Nuisance einmalig gefittet (MT, %d Arme). Wird für alle Modelle wiederverwendet.", K - 1)
        except Exception:
            self._logger.warning("DRTester Nuisance Pre-Fit fehlgeschlagen. Fallback auf Per-Modell-Fit.", exc_info=True)

        # ── Eval-Maske für DRTester: X_val/T_val/Y_val auch filtern ──
        if eval_mask is not None and holdout_data is None:
            X_val_eval = X.loc[eval_mask].reset_index(drop=True)
            T_val_eval = T[eval_mask]
            Y_val_eval = Y[eval_mask]
            self._logger.info("Eval-Maske aktiv: Metriken auf %d von %d Zeilen.", len(X_val_eval), len(X))
        else:
            X_val_eval = None  # Sentinel: benutze Standard-X_val

        for mname, dfp in preds.items():
            # ── Train Many, Evaluate One: Nur Eval-Zeilen für Metriken ──
            if eval_mask is not None and holdout_data is None:
                dfp = dfp.loc[eval_mask].reset_index(drop=True)
            y = dfp["Y"].to_numpy()
            t = dfp["T"].to_numpy()

            # Diagnose: CATE-Verteilung pro Modell loggen (vor Evaluation)
            pred_cols = [c for c in dfp.columns if c.startswith(f"Predictions_{mname}")]
            for pc in pred_cols:
                vals = dfp[pc].dropna()
                if len(vals) > 0:
                    self._logger.info(
                        "Evaluation %s: n=%d, min=%.6g, median=%.6g, max=%.6g, "
                        "std=%.6g, non-zero=%d/%d, unique=%.0f",
                        pc, len(vals), vals.min(), vals.median(), vals.max(),
                        vals.std(), (vals != 0).sum(), len(vals),
                        min(vals.nunique(), 999),
                    )

            # ── Phase 1: Schnelle Metriken (IMMER, für Champion-Selektion) ──
            if is_mt:
                mt_pred_cols = [c for c in dfp.columns if c.startswith(f"Predictions_{mname}_T")]
                if mt_pred_cols:
                    scores_2d = dfp[mt_pred_cols].to_numpy()
                    eval_summary[mname] = mt_eval_summary(y=y, t=t, scores_2d=scores_2d, propensity=None)
                    for key, val in eval_summary[mname].items():
                        if isinstance(val, (int, float)):
                            mlflow.log_metric(f"{key}__{mname}", float(val))
                        elif isinstance(val, dict):
                            for sub_key, sub_val in val.items():
                                mlflow.log_metric(f"{key}_{sub_key}__{mname}", float(sub_val))
            else:
                pred_col = f"Predictions_{mname}"
                if pred_col in dfp.columns:
                    s = dfp[pred_col].to_numpy()
                    curve = uplift_curve(y=y, t=t, score=s)
                    eval_summary[mname] = {
                        "qini": float(qini_coefficient(curve)),
                        "auuc": float(auuc(curve)),
                        "uplift_at_10pct": float(uplift_at_k(curve, k_fraction=0.10)),
                        "uplift_at_20pct": float(uplift_at_k(curve, k_fraction=0.20)),
                        "uplift_at_50pct": float(uplift_at_k(curve, k_fraction=0.50)),
                        "policy_value_treat_positive": float(policy_value(y=y, t=t, score=s, threshold=0.0)),
                    }
                    for key, val in eval_summary[mname].items():
                        short = {"uplift_at_10pct": "uplift10", "uplift_at_20pct": "uplift20", "uplift_at_50pct": "uplift50", "policy_value_treat_positive": "policy_value"}.get(key, key)
                        mlflow.log_metric(f"{short}__{mname}", val)

            # ── CATE-Verteilungs-Plots (IMMER, alle Modelle) ──
            # Schnell (~0.5s), reine Histogramme. Zeigen Training- und
            # Cross-Validated-Predictions nebeneinander.
            try:
                if is_mt:
                    mt_pred_cols = [c for c in dfp.columns if c.startswith(f"Predictions_{mname}_T")]
                    for k, pc in enumerate(mt_pred_cols):
                        arm = k + 1
                        val_arr = dfp[pc].to_numpy(dtype=float)
                        train_col = f"Train_{mname}_T{arm}"
                        train_arr = None
                        if train_col in dfp.columns and not np.all(np.isnan(dfp[train_col].to_numpy(dtype=float))):
                            train_arr = dfp[train_col].to_numpy(dtype=float)
                        fig_dist = generate_cate_distribution_plot(
                            cate_preds_val=val_arr, cate_preds_train=train_arr,
                            model_name=mname, arm_label=f"T{arm}",
                        )
                        if fig_dist is not None:
                            mlflow.log_figure(fig_dist, f"distribution__{mname}_T{arm}.png")
                            if hasattr(self, '_report'):
                                self._report.add_plot(f"{mname}_T{arm}", "cate_distribution", fig_dist)
                            plt.close(fig_dist)
                else:
                    pred_col = f"Predictions_{mname}"
                    if pred_col in dfp.columns:
                        val_arr = dfp[pred_col].to_numpy(dtype=float)
                        train_col = f"Train_{mname}"
                        train_arr = None
                        if train_col in dfp.columns and not np.all(np.isnan(dfp[train_col].to_numpy(dtype=float))):
                            train_arr = dfp[train_col].to_numpy(dtype=float)
                        fig_dist = generate_cate_distribution_plot(
                            cate_preds_val=val_arr, cate_preds_train=train_arr,
                            model_name=mname,
                        )
                        if fig_dist is not None:
                            mlflow.log_figure(fig_dist, f"distribution__{mname}.png")
                            if hasattr(self, '_report'):
                                self._report.add_plot(mname, "cate_distribution", fig_dist)
                            plt.close(fig_dist)
            except Exception:
                self._logger.warning("CATE-Verteilungsplot für %s fehlgeschlagen.", mname, exc_info=True)

        # ── Champion frühzeitig bestimmen (für Level-abhängige Plot-Steuerung) ──
        _early_champion = self._determine_champion(cfg, eval_summary, models) if eval_summary else None
        self._logger.info(
            "Metriken für %d Modelle berechnet. Vorläufiger Champion: %s. "
            "Diagnostik-Plots: %s",
            len(eval_summary), _early_champion or "–",
            "alle Modelle" if pl <= 2 else ("nur Champion" if pl >= 4 else "Champion + Challenger"),
        )

        # ── Phase 2: DRTester-Plots + sklift (Level-abhängig) ──
        # Level 1-2: Alle Modelle bekommen volle Diagnostik
        # Level 3:   Champion + bester Challenger
        # Level 4:   Nur Champion
        if pl >= 4:
            plot_models = {_early_champion} if _early_champion else set()
        elif pl >= 3:
            # Champion + 1 bester Challenger
            plot_models = {_early_champion} if _early_champion else set()
            sel_met = cfg.selection.metric
            other = [(m, eval_summary[m].get(sel_met, 0)) for m in eval_summary if m != _early_champion and isinstance(eval_summary[m].get(sel_met), (int, float))]
            if other:
                other.sort(key=lambda x: x[1], reverse=cfg.selection.higher_is_better)
                plot_models.add(other[0][0])
        else:
            plot_models = set(preds.keys())

        for mname, dfp in preds.items():
            if mname not in plot_models:
                continue
            y = dfp["Y"].to_numpy()
            t = dfp["T"].to_numpy()

            if is_mt:
                eval_summary, policy_values_dict = self._evaluate_mt(
                    cfg, X, T, Y, holdout_data, mname, dfp, y, t,
                    tuned_params_by_model, eval_summary, policy_values_dict, mlflow,
                    fitted_testers=fitted_tester_mt, eval_mask=eval_mask)
            else:
                eval_summary, policy_values_dict = self._evaluate_bt(
                    cfg, X, T, Y, holdout_data, mname, dfp, y, t,
                    tuned_params_by_model, eval_summary, policy_values_dict, mlflow,
                    fitted_tester=fitted_tester_bt, eval_mask=eval_mask)

        # ── Phase 3: scikit-uplift Plots (IMMER für alle Modelle) ──
        # Schnell (~2-5s pro Modell), reine Histogramm/Kurven-Plots ohne
        # Bootstrap-Berechnungen. Unabhängig von Phase 2 (DRTester), damit
        # alle Modelle Qini-Kurve, Uplift-by-Percentile und Treatment-Balance
        # im Report haben.
        eval_X = holdout_data[0] if holdout_data is not None else X
        eval_T = holdout_data[1] if holdout_data is not None else T
        eval_Y = holdout_data[2] if holdout_data is not None else Y
        for mname, dfp in preds.items():
            # ── Train Many, Evaluate One: auch sklift-Plots auf Eval-Subset ──
            if eval_mask is not None and holdout_data is None:
                dfp = dfp.loc[eval_mask].reset_index(drop=True)
                eval_T_sk = T[eval_mask]
                eval_Y_sk = Y[eval_mask]
            else:
                eval_T_sk = eval_T
                eval_Y_sk = eval_Y
            y = dfp["Y"].to_numpy()
            t = dfp["T"].to_numpy()
            if is_mt:
                mt_pred_cols = [c for c in dfp.columns if c.startswith(f"Predictions_{mname}_T")]
                if not mt_pred_cols:
                    continue
                K = len(np.unique(T))
                for k in range(1, K):
                    arm = k
                    arm_scores = dfp[mt_pred_cols[k-1]].to_numpy(dtype=float) if k-1 < len(mt_pred_cols) else None
                    if arm_scores is None:
                        continue
                    arm_mask = (eval_T_sk == 0) | (eval_T_sk == arm)
                    arm_cate = arm_scores[arm_mask]
                    arm_T_bin = (eval_T_sk[arm_mask] == arm).astype(int)
                    arm_Y = eval_Y_sk[arm_mask]
                    try:
                        sk_qini, sk_pct, sk_tb = generate_sklift_plots(arm_cate, arm_T_bin, arm_Y)
                        if sk_qini is not None:
                            mlflow.log_figure(sk_qini, f"sklift_qini__{mname}_T{arm}.png")
                        if sk_pct is not None:
                            mlflow.log_figure(sk_pct, f"sklift_percentile__{mname}_T{arm}.png")
                        if sk_tb is not None:
                            mlflow.log_figure(sk_tb, f"treatment_balance__{mname}_T{arm}.png")
                        if hasattr(self, '_report'):
                            label = f"{mname}_T{arm}"
                            for fig, key in [(sk_qini, "sklift_qini"), (sk_pct, "sklift_percentile"), (sk_tb, "treatment_balance")]:
                                if fig is not None:
                                    self._report.add_plot(label, key, fig)
                        for fig in [sk_qini, sk_pct, sk_tb]:
                            if fig is not None:
                                plt.close(fig)
                    except Exception:
                        self._logger.warning("sklift-Plots für %s T%d fehlgeschlagen.", mname, arm, exc_info=True)
            else:
                pred_col = f"Predictions_{mname}"
                if pred_col not in dfp.columns:
                    continue
                cate_vals = dfp[pred_col].to_numpy(dtype=float)
                try:
                    sk_qini, sk_pct, sk_tb = generate_sklift_plots(cate_vals, eval_T_sk, eval_Y_sk)
                    if sk_qini is not None:
                        mlflow.log_figure(sk_qini, f"sklift_qini__{mname}.png")
                    if sk_pct is not None:
                        mlflow.log_figure(sk_pct, f"sklift_percentile__{mname}.png")
                    if sk_tb is not None:
                        mlflow.log_figure(sk_tb, f"treatment_balance__{mname}.png")
                    if hasattr(self, '_report'):
                        for fig, key in [(sk_qini, "sklift_qini"), (sk_pct, "sklift_percentile"), (sk_tb, "treatment_balance")]:
                            if fig is not None:
                                self._report.add_plot(mname, key, fig)
                    for fig in [sk_qini, sk_pct, sk_tb]:
                        if fig is not None:
                            plt.close(fig)
                except Exception:
                    self._logger.warning("sklift-Plots für %s fehlgeschlagen.", mname, exc_info=True)

        # Historischer Score (nur BT)
        hist_score_eval = S
        if holdout_data is not None and holdout_data[3] is not None:
            hist_score_eval = holdout_data[3]
        elif eval_mask is not None and S is not None:
            hist_score_eval = S[eval_mask]

        if hist_score_eval is not None and is_mt:
            self._logger.info("Historischer Score (S) vorhanden, wird aber bei Multi-Treatment übersprungen.")

        if hist_score_eval is not None and not is_mt:
            eval_summary, policy_values_dict = self._evaluate_historical_score(
                cfg, X, T, Y, holdout_data, preds, hist_score_eval, eval_summary, policy_values_dict, mlflow,
                fitted_tester=fitted_tester_bt, eval_mask=eval_mask)

        if eval_summary:
            def _write_eval(p):
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(eval_summary, fh, ensure_ascii=False, indent=2)

            _log_temp_artifact(mlflow, _write_eval, "uplift_eval_summary.json")

        return eval_summary, policy_values_dict

    def _evaluate_bt(self, cfg, X, T, Y, holdout_data, mname, dfp, y, t, tuned_params_by_model, eval_summary, policy_values_dict, mlflow, fitted_tester=None, eval_mask=None):
        """Binary-Treatment: DRTester-Diagnostik-Plots für ein Modell.

        Metriken (Qini, AUUC, etc.) werden bereits in Phase 1 berechnet.
        Diese Methode erzeugt nur die teuren DRTester-Plots (Calibration,
        Qini/TOC mit Bootstrap-CIs) und scikit-uplift-Plots."""
        import matplotlib.pyplot as plt
        pred_col = f"Predictions_{mname}"
        if pred_col not in dfp.columns:
            return eval_summary, policy_values_dict

        # X/T/Y für DRTester-Plots: bei eval_mask auf Eval-Zeilen filtern
        use_mask = eval_mask is not None and holdout_data is None
        X_eval = X.loc[eval_mask].reset_index(drop=True) if use_mask else X
        T_eval = T[eval_mask] if use_mask else T
        Y_eval = Y[eval_mask] if use_mask else Y
        dfp_eval = dfp.loc[eval_mask].reset_index(drop=True) if use_mask else dfp

        # Wenn eval_mask aktiv ist, muss auch der fitted_tester auf das Eval-Subset
        # gefiltert werden, damit dr_val_ und cate_preds_val_ dimensional übereinstimmen.
        ft_eval = fitted_tester
        if use_mask and fitted_tester is not None:
            ft_eval = None  # Neuen Tester bauen lassen — fitted_tester.dr_val_ passt nicht
            # Stattdessen: gefilterten Tester manuell zusammenbauen
            try:
                ft_eval = CustomDRTester(
                    model_regression=getattr(fitted_tester, '_model_regression_ref', None),
                    model_propensity=getattr(fitted_tester, '_model_propensity_ref', None),
                    cate=None,
                    cate_preds_val=np.zeros(len(X_eval)),  # Placeholder, wird überschrieben
                )
                ft_eval.dr_val_ = fitted_tester.dr_val_[eval_mask] if hasattr(fitted_tester, 'dr_val_') and fitted_tester.dr_val_ is not None else None
                ft_eval.ate_val = ft_eval.dr_val_.mean(axis=0) if ft_eval.dr_val_ is not None else None
                ft_eval.Dval = fitted_tester.Dval[eval_mask] if hasattr(fitted_tester, 'Dval') and fitted_tester.Dval is not None else None
                ft_eval.treatments = fitted_tester.treatments
                ft_eval.n_treat = fitted_tester.n_treat
                ft_eval.fit_on_train = False
                if ft_eval.dr_val_ is None:
                    ft_eval = None  # Fallback: kein fitted_tester
            except Exception:
                ft_eval = None
                self._logger.debug("Eval-Mask: fitted_tester Filterung fehlgeschlagen, Fallback auf None.")

        try:
            train_col = f"Train_{mname}"
            cate_train = None
            if train_col in dfp_eval.columns and not np.all(np.isnan(dfp_eval[train_col].to_numpy(dtype=float))):
                cate_train = dfp_eval[train_col].to_numpy(dtype=float)

            bundle = evaluate_cate_with_plots(
                X_val=(holdout_data[0] if holdout_data is not None else X_eval),
                T_val=(holdout_data[1] if holdout_data is not None else T_eval),
                Y_val=(holdout_data[2] if holdout_data is not None else Y_eval),
                cate_preds_val=dfp_eval[pred_col].to_numpy(dtype=float),
                X_train=X if cate_train is not None else None,
                T_train=T if cate_train is not None else None,
                Y_train=Y if cate_train is not None else None,
                cate_preds_train=cate_train, n_groups=10,
                fitted_tester=ft_eval,
                n_bootstrap=getattr(self, '_eval_n_bootstrap', 1000),
            )
            policy_values_dict[mname] = bundle.policy_values
            _log_temp_artifact(mlflow, lambda p, _b=bundle: save_dataframe_as_png(_b.summary, p), f"summary__{mname}.png")
            _log_temp_artifact(mlflow, lambda p, _b=bundle: save_dataframe_as_png(_b.policy_values, p), f"policy_values__{mname}.png")
            if bundle.cal_plot is not None:
                mlflow.log_figure(bundle.cal_plot, f"cal_plot__{mname}.png")
            if bundle.qini_plot is not None:
                mlflow.log_figure(bundle.qini_plot, f"qini_plot__{mname}.png")
            if bundle.toc_plot is not None:
                mlflow.log_figure(bundle.toc_plot, f"toc_plot__{mname}.png")
            if bundle.sklift_qini is not None:
                mlflow.log_figure(bundle.sklift_qini, f"sklift_qini__{mname}.png")
            if bundle.sklift_percentile is not None:
                mlflow.log_figure(bundle.sklift_percentile, f"sklift_percentile__{mname}.png")
            if bundle.treatment_balance is not None:
                mlflow.log_figure(bundle.treatment_balance, f"treatment_balance__{mname}.png")

            # Plots für HTML-Report sammeln
            if hasattr(self, '_report'):
                for fig, key in [(bundle.cal_plot, "cal_plot"), (bundle.qini_plot, "qini_plot"),
                                 (bundle.toc_plot, "toc_plot"), (bundle.sklift_qini, "sklift_qini"),
                                 (bundle.sklift_percentile, "sklift_percentile"), (bundle.treatment_balance, "treatment_balance")]:
                    if fig is not None:
                        self._report.add_plot(mname, key, fig)

            # Figures freigeben (add_plot hat bereits base64 erzeugt)
            for fig in [bundle.cal_plot, bundle.qini_plot, bundle.toc_plot,
                        bundle.sklift_qini, bundle.sklift_percentile, bundle.treatment_balance]:
                if fig is not None:
                    plt.close(fig)
        except Exception:
            self._logger.warning("DRTester/SkLift-Plots für %s fehlgeschlagen.", mname, exc_info=True)

        return eval_summary, policy_values_dict

    def _evaluate_mt(self, cfg, X, T, Y, holdout_data, mname, dfp, y, t, tuned_params_by_model, eval_summary, policy_values_dict, mlflow, fitted_testers=None, eval_mask=None):
        """Multi-Treatment: DRTester-Diagnostik-Plots für ein Modell.

        Metriken (per-Arm Qini, AUUC, globaler Policy Value) werden bereits
        in Phase 1 berechnet. Diese Methode erzeugt nur die teuren
        DRTester-Plots pro Treatment-Arm und scikit-uplift-Plots."""
        import matplotlib.pyplot as plt

        # eval_mask: auf Eval-Subset filtern (wie bei _evaluate_bt)
        use_mask = eval_mask is not None and holdout_data is None
        if use_mask:
            dfp = dfp.loc[eval_mask].reset_index(drop=True)
            X = X.loc[eval_mask].reset_index(drop=True)
            T = T[eval_mask]
            Y = Y[eval_mask]
            y = dfp["Y"].to_numpy()
            t = dfp["T"].to_numpy()

        mt_pred_cols = [c for c in dfp.columns if c.startswith(f"Predictions_{mname}_T")]
        if not mt_pred_cols:
            return eval_summary, policy_values_dict

        scores_2d = dfp[mt_pred_cols].to_numpy()

        try:
            n_effects = scores_2d.shape[1]
            for k in range(n_effects):
                arm = k + 1
                arm_scores = scores_2d[:, k]
                try:
                    eval_X = (holdout_data[0] if holdout_data is not None else X)
                    eval_T = (holdout_data[1] if holdout_data is not None else T)
                    eval_Y = (holdout_data[2] if holdout_data is not None else Y)
                    arm_mask = (eval_T == 0) | (eval_T == arm)
                    arm_X = eval_X.loc[eval_X.index[arm_mask]].copy()
                    arm_T = (eval_T[arm_mask] == arm).astype(int)
                    arm_Y = eval_Y[arm_mask]
                    arm_cate = arm_scores[arm_mask]

                    train_col = f"Train_{mname}_T{arm}"
                    cate_train, arm_X_train, arm_T_train, arm_Y_train = None, None, None, None
                    if train_col in dfp.columns and not np.all(np.isnan(dfp[train_col].to_numpy(dtype=float))):
                        train_mask = (T == 0) | (T == arm)
                        cate_train = dfp[train_col].to_numpy(dtype=float)[train_mask]
                        arm_X_train = X.iloc[np.where(train_mask)[0]].copy()
                        arm_T_train = (T[train_mask] == arm).astype(int)
                        arm_Y_train = Y[train_mask]

                    # Pre-fitted DRTester für diesen Arm verwenden (wenn vorhanden)
                    arm_tester = (fitted_testers or {}).get(arm)
                    bundle = evaluate_cate_with_plots(
                        X_val=arm_X, T_val=arm_T, Y_val=arm_Y, cate_preds_val=arm_cate,
                        X_train=arm_X_train, T_train=arm_T_train, Y_train=arm_Y_train,
                        cate_preds_train=cate_train, n_groups=10,
                        fitted_tester=arm_tester,
                        n_bootstrap=getattr(self, '_eval_n_bootstrap', 1000),
                    )
                    policy_values_dict[f"{mname}_T{arm}"] = bundle.policy_values
                    _log_temp_artifact(mlflow, lambda p, _b=bundle: save_dataframe_as_png(_b.summary, p), f"summary__{mname}_T{arm}.png")
                    _log_temp_artifact(mlflow, lambda p, _b=bundle: save_dataframe_as_png(_b.policy_values, p), f"policy_values__{mname}_T{arm}.png")
                    if bundle.cal_plot is not None:
                        mlflow.log_figure(bundle.cal_plot, f"cal_plot__{mname}_T{arm}.png")
                    if bundle.qini_plot is not None:
                        mlflow.log_figure(bundle.qini_plot, f"qini_plot__{mname}_T{arm}.png")
                    if bundle.toc_plot is not None:
                        mlflow.log_figure(bundle.toc_plot, f"toc_plot__{mname}_T{arm}.png")
                    if bundle.sklift_qini is not None:
                        mlflow.log_figure(bundle.sklift_qini, f"sklift_qini__{mname}_T{arm}.png")
                    if bundle.sklift_percentile is not None:
                        mlflow.log_figure(bundle.sklift_percentile, f"sklift_percentile__{mname}_T{arm}.png")
                    if bundle.treatment_balance is not None:
                        mlflow.log_figure(bundle.treatment_balance, f"treatment_balance__{mname}_T{arm}.png")

                    # Plots für HTML-Report (MT per Arm)
                    if hasattr(self, '_report'):
                        label = f"{mname}_T{arm}"
                        for fig, key in [(bundle.cal_plot, "cal_plot"), (bundle.qini_plot, "qini_plot"),
                                         (bundle.toc_plot, "toc_plot"), (bundle.sklift_qini, "sklift_qini"),
                                         (bundle.sklift_percentile, "sklift_percentile"), (bundle.treatment_balance, "treatment_balance")]:
                            if fig is not None:
                                self._report.add_plot(label, key, fig)

                    # Figures freigeben
                    for fig in [bundle.cal_plot, bundle.qini_plot, bundle.toc_plot,
                                bundle.sklift_qini, bundle.sklift_percentile, bundle.treatment_balance]:
                        if fig is not None:
                            plt.close(fig)
                except Exception:
                    self._logger.warning("DRTester-Plots für %s T%d fehlgeschlagen.", mname, arm, exc_info=True)
        except Exception:
            self._logger.warning("DRTester-Plots für %s fehlgeschlagen.", mname, exc_info=True)

        return eval_summary, policy_values_dict

    def _evaluate_historical_score(self, cfg, X, T, Y, holdout_data, preds, hist_score_eval, eval_summary, policy_values_dict, mlflow, fitted_tester=None, eval_mask=None):
        """Vergleich der kausalen Modelle gegen einen historischen Score."""
        import matplotlib.pyplot as plt
        from rubin.utils.plot_theme import apply_rubin_theme
        apply_rubin_theme()

        hist_name = cfg.historical_score.name
        hist_score = np.asarray(hist_score_eval).astype(float)
        hist_score = np.nan_to_num(hist_score, nan=0.0, posinf=0.0, neginf=0.0)
        if not cfg.historical_score.higher_is_better:
            hist_score = -hist_score

        # Bei eval_mask: X/T/Y auf Eval-Subset filtern (hist_score ist bereits gefiltert)
        use_mask = eval_mask is not None and holdout_data is None
        eval_X = X.loc[eval_mask].reset_index(drop=True) if use_mask else X
        eval_y = holdout_data[2] if holdout_data is not None else (Y[eval_mask] if use_mask else Y)
        eval_t = holdout_data[1] if holdout_data is not None else (T[eval_mask] if use_mask else T)
        curve_h = uplift_curve(y=eval_y, t=eval_t, score=hist_score)
        eval_summary[hist_name] = {
            "qini": float(qini_coefficient(curve_h)),
            "auuc": float(auuc(curve_h)),
            "uplift_at_10pct": float(uplift_at_k(curve_h, k_fraction=0.10)),
            "uplift_at_20pct": float(uplift_at_k(curve_h, k_fraction=0.20)),
            "uplift_at_50pct": float(uplift_at_k(curve_h, k_fraction=0.50)),
            "policy_value_treat_positive": float(policy_value(y=eval_y, t=eval_t, score=hist_score, threshold=0.0)),
        }
        for key, val in eval_summary[hist_name].items():
            short = {"uplift_at_10pct": "uplift10", "uplift_at_20pct": "uplift20", "uplift_at_50pct": "uplift50", "policy_value_treat_positive": "policy_value"}.get(key, key)
            mlflow.log_metric(f"{short}__{hist_name}", val)

        # Distribution-Plot für historischen Score
        try:
            fig_dist = generate_cate_distribution_plot(
                cate_preds_val=hist_score, model_name=hist_name,
            )
            if fig_dist is not None:
                mlflow.log_figure(fig_dist, f"distribution__{hist_name}.png")
                if hasattr(self, '_report'):
                    self._report.add_plot(hist_name, "cate_distribution", fig_dist)
                plt.close(fig_dist)
        except Exception:
            self._logger.warning("Distribution-Plot für %s fehlgeschlagen.", hist_name, exc_info=True)

        try:
            # Bei eval_mask: fitted_tester filtern (wie in _evaluate_bt)
            ft_hist = fitted_tester
            if use_mask and fitted_tester is not None:
                try:
                    ft_hist = CustomDRTester(
                        model_regression=getattr(fitted_tester, '_model_regression_ref', None),
                        model_propensity=getattr(fitted_tester, '_model_propensity_ref', None),
                        cate=None, cate_preds_val=np.zeros(len(eval_X)),
                    )
                    ft_hist.dr_val_ = fitted_tester.dr_val_[eval_mask] if hasattr(fitted_tester, 'dr_val_') and fitted_tester.dr_val_ is not None else None
                    ft_hist.ate_val = ft_hist.dr_val_.mean(axis=0) if ft_hist.dr_val_ is not None else None
                    ft_hist.Dval = fitted_tester.Dval[eval_mask] if hasattr(fitted_tester, 'Dval') and fitted_tester.Dval is not None else None
                    ft_hist.treatments = fitted_tester.treatments
                    ft_hist.n_treat = fitted_tester.n_treat
                    ft_hist.fit_on_train = False
                    if ft_hist.dr_val_ is None:
                        ft_hist = None
                except Exception:
                    ft_hist = None

            if ft_hist is not None:
                bundle_h = evaluate_cate_with_plots(
                    fitted_tester=ft_hist,
                    X_val=eval_X,
                    T_val=eval_t,
                    Y_val=eval_y,
                    cate_preds_val=hist_score,
                    X_train=X, T_train=T, Y_train=Y,
                    cate_preds_train=hist_score if not use_mask else None, n_groups=10,
                    n_bootstrap=getattr(self, '_eval_n_bootstrap', 1000),
                )
            else:
                # Fallback: eigene Nuisance fitten
                params_y = dict(cfg.base_learner.fixed_params or {})
                params_t = dict(cfg.base_learner.fixed_params or {})
                model_reg = build_base_learner(cfg.base_learner.type, params_y, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=1 if cfg.constants.parallel_level <= 1 else -1)
                model_prop = build_base_learner(cfg.base_learner.type, params_t, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=1 if cfg.constants.parallel_level <= 1 else -1)
                bundle_h = evaluate_cate_with_plots(
                    model_regression=model_reg, model_propensity=model_prop,
                    X_val=eval_X,
                    T_val=eval_t,
                    Y_val=eval_y,
                    cate_preds_val=hist_score, X_train=X, T_train=T, Y_train=Y,
                    cate_preds_train=hist_score, n_groups=10,
                    n_bootstrap=getattr(self, '_eval_n_bootstrap', 1000),
                )
            policy_values_dict[hist_name] = bundle_h.policy_values
            _log_temp_artifact(mlflow, lambda p, _b=bundle_h: save_dataframe_as_png(_b.summary, p), f"summary__{hist_name}.png")
            _log_temp_artifact(mlflow, lambda p, _b=bundle_h: save_dataframe_as_png(_b.policy_values, p), f"policy_values__{hist_name}.png")
            if bundle_h.cal_plot is not None:
                mlflow.log_figure(bundle_h.cal_plot, f"cal_plot__{hist_name}.png")
            if bundle_h.qini_plot is not None:
                mlflow.log_figure(bundle_h.qini_plot, f"qini_plot__{hist_name}.png")
            if bundle_h.toc_plot is not None:
                mlflow.log_figure(bundle_h.toc_plot, f"toc_plot__{hist_name}.png")
            if bundle_h.sklift_qini is not None:
                mlflow.log_figure(bundle_h.sklift_qini, f"sklift_qini__{hist_name}.png")
            if bundle_h.sklift_percentile is not None:
                mlflow.log_figure(bundle_h.sklift_percentile, f"sklift_percentile__{hist_name}.png")
            if bundle_h.treatment_balance is not None:
                mlflow.log_figure(bundle_h.treatment_balance, f"treatment_balance__{hist_name}.png")

            # Plots für HTML-Report (historischer Score)
            if hasattr(self, '_report'):
                for fig, key in [(bundle_h.cal_plot, "cal_plot"), (bundle_h.qini_plot, "qini_plot"),
                                 (bundle_h.toc_plot, "toc_plot"), (bundle_h.sklift_qini, "sklift_qini"),
                                 (bundle_h.sklift_percentile, "sklift_percentile"), (bundle_h.treatment_balance, "treatment_balance")]:
                    if fig is not None:
                        self._report.add_plot(hist_name, key, fig)

            # Figures freigeben
            for fig in [bundle_h.cal_plot, bundle_h.qini_plot, bundle_h.toc_plot,
                        bundle_h.sklift_qini, bundle_h.sklift_percentile, bundle_h.treatment_balance]:
                if fig is not None:
                    plt.close(fig)

            for mname, dfp in preds.items():
                pred_col = f"Predictions_{mname}"
                if pred_col not in dfp.columns:
                    continue
                # Bei eval_mask: preds auf Eval-Subset filtern
                dfp_cmp = dfp.loc[eval_mask].reset_index(drop=True) if use_mask else dfp
                df_cmp = pd.DataFrame({
                    "Y": dfp_cmp["Y"].to_numpy(), "T": dfp_cmp["T"].to_numpy(),
                    mname: dfp_cmp[pred_col].to_numpy(dtype=float), hist_name: hist_score,
                })
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_custom_qini_curve(data=df_cmp, causal_score_label=mname, affinity_score_label=hist_name, ax=ax, relative_axes=True, x_vert=0.5, print_uplift=True)
                mlflow.log_figure(fig, f"custom_qini__{mname}_vs_{hist_name}.png")
                if hasattr(self, '_report'):
                    self._report.add_plot(mname, f"qini_vs_{hist_name}", fig)
                plt.close(fig)
        except Exception:
            self._logger.warning("DRTester/SkLift-Plots für historischen Score fehlgeschlagen.", exc_info=True)

        try:
            if hist_name in policy_values_dict and len(policy_values_dict) > 1:
                figs = policy_value_comparison_plots(policy_values_dict, comparison_model_name=hist_name)
                for model_name, fig in figs.items():
                    mlflow.log_figure(fig, f"policy_compare__{model_name}_vs_{hist_name}.png")
                    if hasattr(self, '_report'):
                        self._report.add_plot(model_name, f"policy_compare_vs_{hist_name}", fig)
                    plt.close(fig)
        except Exception:
            self._logger.warning("Policy-Value-Vergleich fehlgeschlagen.", exc_info=True)

        _log_temp_artifact(mlflow, lambda p: pd.DataFrame({"fraction": curve_h.fraction, "uplift": curve_h.uplift, "n_treat": curve_h.n_treat, "n_control": curve_h.n_control}).to_csv(p, index=False), f"uplift_curve__{hist_name}.csv")

        return eval_summary, policy_values_dict

    # ------------------------------------------------------------------
    # Surrogate-Einzelbaum (Teacher-Learner)
    # ------------------------------------------------------------------

    def _determine_champion(self, cfg, eval_summary, models):
        """Ermittelt den Champion-Modellnamen anhand der Konfiguration und eval_summary.

        Wird vor dem Bundle-Export aufgerufen, damit der Surrogate-Einzelbaum
        den Champion bereits kennt."""
        manual = (getattr(cfg.selection, "manual_champion", None) or "").strip() or None
        if manual and manual in models:
            return manual

        entries = [
            ModelEntry(name=name, artifact_path=f"models/{name}.pkl", metrics=float_metrics(metrics or {}))
            for name, metrics in eval_summary.items()
            if name not in (SURROGATE_MODEL_NAME, SURROGATE_CF_NAME)
        ]
        champion = choose_champion(entries, metric=cfg.selection.metric, higher_is_better=cfg.selection.higher_is_better)
        if champion is None and entries:
            champion = entries[0].name
        return champion

    def _build_surrogate_regressor(self, cfg):
        """Erzeugt einen Einzelbaum-Regressor aus der Surrogate- und Base-Learner-Konfiguration."""
        tree_cfg = cfg.surrogate_tree
        base_type = cfg.base_learner.type.lower()
        seed = cfg.constants.random_seed

        if base_type == "lgbm":
            params = {
                "n_estimators": 1,
                "num_leaves": tree_cfg.num_leaves,
                "min_child_samples": tree_cfg.min_samples_leaf,
                "max_depth": tree_cfg.max_depth if tree_cfg.max_depth is not None else -1,
                "learning_rate": 1.0,  # Einzelbaum: kein Shrinkage
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
            }
        elif base_type == "catboost":
            params = {
                "iterations": 1,
                "min_data_in_leaf": tree_cfg.min_samples_leaf,
                "depth": tree_cfg.max_depth if tree_cfg.max_depth is not None else 6,
                "learning_rate": 1.0,
                "bootstrap_type": "No",  # Kein Bootstrapping für Einzelbaum
                "rsm": 1.0,
                "l2_leaf_reg": 0.0,
            }
        else:
            raise ValueError(f"Unbekannter base_learner.type={base_type!r} für Surrogate-Tree.")

        return build_base_learner(base_type, params, seed=seed, task="regressor", parallel_jobs=1 if cfg.constants.parallel_level <= 1 else -1)

    @staticmethod
    def _log_surrogate_tree_info(tree_model, base_type: str):
        """Extrahiert Baumtiefe und Blattanzahl für Logging."""
        base_type = base_type.lower()
        depth, n_leaves = None, None
        try:
            if base_type == "lgbm":
                info = tree_model.booster_.dump_model()
                tree_info = info.get("tree_info", [{}])
                if tree_info:
                    n_leaves = tree_info[0].get("num_leaves")
                    depth = tree_info[0].get("max_depth")
            elif base_type == "catboost":
                all_params = tree_model.get_all_params()
                depth = all_params.get("depth")
                # CatBoost: symmetrischer Baum → 2^depth Blätter
                if depth is not None:
                    n_leaves = 2 ** int(depth)
        except Exception:
            pass
        return depth, n_leaves

    def _train_and_evaluate_surrogate(self, cfg, X, T, Y, teacher_name, preds, holdout_data, models, eval_summary, mlflow, surrogate_name=None, run_drtester=False, fitted_tester=None, eval_mask=None):
        """Trainiert den Surrogate-Einzelbaum und evaluiert ihn.

        Parameters
        ----------
        teacher_name : str
            Name des Modells, dessen CATE-Predictions als Regressionsziel dienen.
        surrogate_name : str, optional
            Name für den Surrogate im eval_summary/Report. Default: SURROGATE_MODEL_NAME.
        run_drtester : bool
            Wenn True, werden volle DRTester-Plots (BLP, Calibration, Qini, TOC, Policy Values) erzeugt.
        fitted_tester : CustomDRTester, optional
            Pre-fitted DRTester für DRTester-Plots (vermeidet redundantes Nuisance-Fitting).

        Returns
        -------
        surrogate_wrapper : SurrogateTreeWrapper
        surrogate_df : pd.DataFrame
        """
        from sklearn.model_selection import KFold

        base_type = cfg.base_learner.type
        seed = cfg.constants.random_seed
        n_splits = cfg.data_processing.cross_validation_splits
        sname = surrogate_name or SURROGATE_MODEL_NAME
        is_mt = is_multi_treatment(T)

        champion_df = preds[teacher_name]

        if holdout_data is None:
            # --- Cross-Modus ---
            if is_mt:
                mt_cols = sorted([c for c in champion_df.columns if c.startswith(f"Predictions_{teacher_name}_T")])
                target_2d = champion_df[mt_cols].to_numpy()
                n_effects = target_2d.shape[1]

                surrogate_preds = np.full_like(target_2d, np.nan, dtype=float)
                final_trees = {}
                for k in range(n_effects):
                    target_k = target_2d[:, k]
                    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    for tr_idx, va_idx in cv.split(X):
                        tree_k = self._build_surrogate_regressor(cfg)
                        tree_k.fit(X.iloc[tr_idx], target_k[tr_idx])
                        surrogate_preds[va_idx, k] = tree_k.predict(X.iloc[va_idx])

                    final_tree_k = self._build_surrogate_regressor(cfg)
                    final_tree_k.fit(X, target_k)
                    final_trees[k] = final_tree_k

                surrogate_df = pd.DataFrame({"Y": Y, "T": T})
                for k in range(n_effects):
                    surrogate_df[f"Predictions_{sname}_T{k+1}"] = surrogate_preds[:, k]
                    surrogate_df[f"Train_{sname}_T{k+1}"] = final_trees[k].predict(X)
                best_eff = np.nanmax(surrogate_preds, axis=1)
                best_arm = np.nanargmax(surrogate_preds, axis=1) + 1
                surrogate_df[f"OptimalTreatment_{sname}"] = np.where(best_eff > 0, best_arm, 0)

                final_tree = None  # Nicht genutzt bei MT
            else:
                target = champion_df[f"Predictions_{teacher_name}"].to_numpy()
                surrogate_preds = np.full_like(target, np.nan, dtype=float)
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                for tr_idx, va_idx in cv.split(X):
                    tree = self._build_surrogate_regressor(cfg)
                    tree.fit(X.iloc[tr_idx], target[tr_idx])
                    surrogate_preds[va_idx] = tree.predict(X.iloc[va_idx])

                final_tree = self._build_surrogate_regressor(cfg)
                final_tree.fit(X, target)
                final_trees = {}

                surrogate_df = pd.DataFrame({"Y": Y, "T": T})
                surrogate_df[f"Predictions_{sname}"] = surrogate_preds
                surrogate_df[f"Train_{sname}"] = final_tree.predict(X)

        else:
            # --- Holdout-Modus ---
            champion_model = models[teacher_name]
            train_target = np.asarray(_predict_effect(champion_model, X))

            X_test, T_test, Y_test, _ = holdout_data

            if is_mt and train_target.ndim == 2:
                n_effects = train_target.shape[1]
                final_trees = {}
                holdout_preds = np.full((len(X_test), n_effects), np.nan, dtype=float)
                for k in range(n_effects):
                    tree_k = self._build_surrogate_regressor(cfg)
                    tree_k.fit(X, train_target[:, k])
                    holdout_preds[:, k] = tree_k.predict(X_test)
                    final_trees[k] = tree_k
                final_tree = None

                surrogate_df = pd.DataFrame({"Y": Y_test, "T": T_test})
                for k in range(n_effects):
                    surrogate_df[f"Predictions_{sname}_T{k+1}"] = holdout_preds[:, k]
                    surrogate_df[f"Train_{sname}_T{k+1}"] = np.nan
                best_eff = np.nanmax(holdout_preds, axis=1)
                best_arm = np.nanargmax(holdout_preds, axis=1) + 1
                surrogate_df[f"OptimalTreatment_{sname}"] = np.where(best_eff > 0, best_arm, 0)
            else:
                final_tree = self._build_surrogate_regressor(cfg)
                final_tree.fit(X, train_target.reshape(-1))
                holdout_pred = final_tree.predict(X_test)
                final_trees = {}

                surrogate_df = pd.DataFrame({"Y": Y_test, "T": T_test})
                surrogate_df[f"Predictions_{sname}"] = holdout_pred.reshape(-1)
                surrogate_df[f"Train_{sname}"] = np.nan

        surrogate_wrapper = SurrogateTreeWrapper(
            tree=final_tree, trees=final_trees, champion_name=teacher_name,
        )

        # Evaluation mit denselben Metriken
        # Bei eval_mask: Metriken nur auf Eval-Subset (konsistent mit Phase 1)
        use_mask = eval_mask is not None and holdout_data is None
        surr_eval_df = surrogate_df.loc[eval_mask].reset_index(drop=True) if use_mask else surrogate_df
        try:
            y_s = surr_eval_df["Y"].to_numpy()
            t_s = surr_eval_df["T"].to_numpy()
            if is_mt:
                mt_cols_s = [c for c in surr_eval_df.columns if c.startswith(f"Predictions_{sname}_T")]
                scores_2d = surr_eval_df[mt_cols_s].to_numpy()
                eval_summary[sname] = mt_eval_summary(y=y_s, t=t_s, scores_2d=scores_2d)
            else:
                s = surr_eval_df[f"Predictions_{sname}"].to_numpy()
                curve = uplift_curve(y=y_s, t=t_s, score=s)
                eval_summary[sname] = {
                    "qini": float(qini_coefficient(curve)),
                    "auuc": float(auuc(curve)),
                    "uplift_at_10pct": float(uplift_at_k(curve, k_fraction=0.10)),
                    "uplift_at_20pct": float(uplift_at_k(curve, k_fraction=0.20)),
                    "uplift_at_50pct": float(uplift_at_k(curve, k_fraction=0.50)),
                    "policy_value_treat_positive": float(policy_value(y=y_s, t=t_s, score=s, threshold=0.0)),
                }
            for key, val in eval_summary[sname].items():
                if isinstance(val, (int, float)):
                    short = {"uplift_at_10pct": "uplift10", "uplift_at_20pct": "uplift20", "uplift_at_50pct": "uplift50", "policy_value_treat_positive": "policy_value"}.get(key, key)
                    mlflow.log_metric(f"{short}__{sname}", float(val))
                elif isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        mlflow.log_metric(f"{key}_{sub_key}__{sname}", float(sub_val))
            self._logger.info("Surrogate-Evaluation: %s", eval_summary[sname])
        except Exception:
            self._logger.warning("Surrogate-Evaluation fehlgeschlagen.", exc_info=True)

        # Surrogate CATE-Verteilungsplot
        try:
            import matplotlib.pyplot as plt
            if is_mt:
                mt_cols_s = [c for c in surrogate_df.columns if c.startswith(f"Predictions_{sname}_T")]
                for k, pc in enumerate(mt_cols_s):
                    arm = k + 1
                    fig_dist = generate_cate_distribution_plot(
                        cate_preds_val=surrogate_df[pc].to_numpy(dtype=float),
                        model_name=sname, arm_label=f"T{arm}",
                    )
                    if fig_dist is not None:
                        mlflow.log_figure(fig_dist, f"distribution__{sname}_T{arm}.png")
                        if hasattr(self, '_report'):
                            self._report.add_plot(f"{sname}_T{arm}", "cate_distribution", fig_dist)
                        plt.close(fig_dist)
            else:
                pred_col_s = f"Predictions_{sname}"
                if pred_col_s in surrogate_df.columns:
                    fig_dist = generate_cate_distribution_plot(
                        cate_preds_val=surrogate_df[pred_col_s].to_numpy(dtype=float),
                        model_name=sname,
                    )
                    if fig_dist is not None:
                        mlflow.log_figure(fig_dist, f"distribution__{sname}.png")
                        if hasattr(self, '_report'):
                            self._report.add_plot(sname, "cate_distribution", fig_dist)
                        plt.close(fig_dist)
        except Exception:
            self._logger.warning("Surrogate-CATE-Verteilungsplot fehlgeschlagen.", exc_info=True)

        # Surrogate sklift-Plots (Qini-Kurve, Uplift-by-Percentile, Treatment-Balance)
        try:
            import matplotlib.pyplot as plt
            y_s = surr_eval_df["Y"].to_numpy()
            t_s = surr_eval_df["T"].to_numpy()
            if is_mt:
                mt_cols_s = [c for c in surr_eval_df.columns if c.startswith(f"Predictions_{sname}_T")]
                K = len(np.unique(t_s))
                for k in range(1, K):
                    if k - 1 >= len(mt_cols_s):
                        continue
                    arm_scores = surr_eval_df[mt_cols_s[k - 1]].to_numpy(dtype=float)
                    arm_mask = (t_s == 0) | (t_s == k)
                    arm_cate = arm_scores[arm_mask]
                    arm_T_bin = (t_s[arm_mask] == k).astype(int)
                    arm_Y = y_s[arm_mask]
                    try:
                        sk_qini, sk_pct, sk_tb = generate_sklift_plots(arm_cate, arm_T_bin, arm_Y)
                        label = f"{sname}_T{k}"
                        for fig, key, fname in [
                            (sk_qini, "sklift_qini", f"sklift_qini__{sname}_T{k}.png"),
                            (sk_pct, "sklift_percentile", f"sklift_percentile__{sname}_T{k}.png"),
                            (sk_tb, "treatment_balance", f"treatment_balance__{sname}_T{k}.png"),
                        ]:
                            if fig is not None:
                                mlflow.log_figure(fig, fname)
                                if hasattr(self, '_report'):
                                    self._report.add_plot(label, key, fig)
                                plt.close(fig)
                    except Exception:
                        self._logger.warning("sklift-Plots für %s T%d fehlgeschlagen.", sname, k, exc_info=True)
            else:
                pred_col_s = f"Predictions_{sname}"
                if pred_col_s in surr_eval_df.columns:
                    cate_vals = surr_eval_df[pred_col_s].to_numpy(dtype=float)
                    try:
                        sk_qini, sk_pct, sk_tb = generate_sklift_plots(cate_vals, t_s, y_s)
                        for fig, key, fname in [
                            (sk_qini, "sklift_qini", f"sklift_qini__{sname}.png"),
                            (sk_pct, "sklift_percentile", f"sklift_percentile__{sname}.png"),
                            (sk_tb, "treatment_balance", f"treatment_balance__{sname}.png"),
                        ]:
                            if fig is not None:
                                mlflow.log_figure(fig, fname)
                                if hasattr(self, '_report'):
                                    self._report.add_plot(sname, key, fig)
                                plt.close(fig)
                    except Exception:
                        self._logger.warning("sklift-Plots für %s fehlgeschlagen.", sname, exc_info=True)
        except Exception:
            self._logger.warning("Surrogate-sklift-Plots fehlgeschlagen.", exc_info=True)

        # Baumtiefe und Blattanzahl loggen
        try:
            log_tree = final_tree if final_tree is not None else (final_trees.get(0) if final_trees else None)
            if log_tree is not None:
                depth, n_leaves = self._log_surrogate_tree_info(log_tree, base_type)
                if depth is not None:
                    mlflow.log_param("surrogate_tree_depth", int(depth))
                if n_leaves is not None:
                    mlflow.log_param("surrogate_tree_n_leaves", int(n_leaves))
            if is_mt:
                mlflow.log_param("surrogate_tree_n_arms", len(final_trees))
            mlflow.log_param("surrogate_tree_base_type", base_type)
            mlflow.log_param("surrogate_tree_teacher", teacher_name)
        except Exception:
            pass

        # ── DRTester-Diagnostik-Plots (nur wenn run_drtester=True) ──
        if run_drtester and not is_mt:
            try:
                import matplotlib.pyplot as plt
                pred_col_s = f"Predictions_{sname}"
                if pred_col_s in surrogate_df.columns:
                    cate_vals = surr_eval_df[pred_col_s].to_numpy(dtype=float)
                    y_s = surr_eval_df["Y"].to_numpy()
                    t_s = surr_eval_df["T"].to_numpy()
                    eval_X = holdout_data[0] if holdout_data is not None else (X.loc[eval_mask].reset_index(drop=True) if use_mask else X)

                    # fitted_tester filtern bei eval_mask (wie _evaluate_bt)
                    ft_surr = fitted_tester
                    if use_mask and fitted_tester is not None:
                        try:
                            ft_surr = CustomDRTester(
                                model_regression=getattr(fitted_tester, '_model_regression_ref', None),
                                model_propensity=getattr(fitted_tester, '_model_propensity_ref', None),
                                cate=None, cate_preds_val=np.zeros(len(eval_X)),
                            )
                            ft_surr.dr_val_ = fitted_tester.dr_val_[eval_mask] if hasattr(fitted_tester, 'dr_val_') and fitted_tester.dr_val_ is not None else None
                            ft_surr.ate_val = ft_surr.dr_val_.mean(axis=0) if ft_surr.dr_val_ is not None else None
                            ft_surr.Dval = fitted_tester.Dval[eval_mask] if hasattr(fitted_tester, 'Dval') and fitted_tester.Dval is not None else None
                            ft_surr.treatments = fitted_tester.treatments
                            ft_surr.n_treat = fitted_tester.n_treat
                            ft_surr.fit_on_train = False
                            if ft_surr.dr_val_ is None:
                                ft_surr = None
                        except Exception:
                            ft_surr = None

                    if ft_surr is not None:
                        bundle_s = evaluate_cate_with_plots(
                            fitted_tester=ft_surr,
                            X_val=eval_X, T_val=t_s, Y_val=y_s,
                            cate_preds_val=cate_vals,
                            X_train=X if holdout_data is not None else None,
                            T_train=T if holdout_data is not None else None,
                            Y_train=Y if holdout_data is not None else None,
                            cate_preds_train=None, n_groups=10,
                            n_bootstrap=getattr(self, '_eval_n_bootstrap', 1000),
                        )
                    else:
                        # Fallback: eigene Nuisance fitten (teurer, aber korrekt)
                        pj = 1 if cfg.constants.parallel_level <= 1 else -1
                        model_reg = build_base_learner(cfg.base_learner.type, {}, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=pj)
                        model_prop = build_base_learner(cfg.base_learner.type, {}, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=pj)
                        bundle_s = evaluate_cate_with_plots(
                            model_regression=model_reg, model_propensity=model_prop,
                            X_val=eval_X, T_val=t_s, Y_val=y_s,
                            cate_preds_val=cate_vals,
                            X_train=X, T_train=T, Y_train=Y,
                            cate_preds_train=None, n_groups=10,
                            n_bootstrap=getattr(self, '_eval_n_bootstrap', 1000),
                        )

                    if bundle_s is not None:
                        if bundle_s.summary is not None and len(bundle_s.summary) > 0:
                            _log_temp_artifact(mlflow, lambda p, _b=bundle_s: save_dataframe_as_png(_b.summary, p), f"summary__{sname}.png")
                        if bundle_s.policy_values is not None and len(bundle_s.policy_values) > 0:
                            _log_temp_artifact(mlflow, lambda p, _b=bundle_s: save_dataframe_as_png(_b.policy_values, p), f"policy_values__{sname}.png")
                        for fig, key, fname in [
                            (bundle_s.cal_plot, "cal_plot", f"cal_plot__{sname}.png"),
                            (bundle_s.qini_plot, "qini_plot", f"qini_plot__{sname}.png"),
                            (bundle_s.toc_plot, "toc_plot", f"toc_plot__{sname}.png"),
                        ]:
                            if fig is not None:
                                mlflow.log_figure(fig, fname)
                                if hasattr(self, '_report'):
                                    self._report.add_plot(sname, key, fig)
                                plt.close(fig)
                        self._logger.info("DRTester-Plots für %s erzeugt.", sname)
            except Exception:
                self._logger.warning("DRTester-Plots für %s fehlgeschlagen.", sname, exc_info=True)

        return surrogate_wrapper, surrogate_df

    def _run_bundle_export(self, cfg, models, eval_summary, X, T, Y, X_full, T_full, Y_full, selected_feature_columns, holdout_data, export_bundle, bundle_dir, bundle_id, mlflow):
        """Synchroner Bundle-Export am Ende des Analyselaufs."""
        bundle_cfg = getattr(cfg, "bundle", None)
        export_bundle_effective = bool(bundle_cfg.enabled) if bundle_cfg is not None else False
        if export_bundle is not None:
            export_bundle_effective = bool(export_bundle)
        if not export_bundle_effective:
            return

        bundle_base_dir = bundle_dir or (getattr(bundle_cfg, "base_dir", None) or "bundles")
        bundle_id_effective = bundle_id or getattr(bundle_cfg, "bundle_id", None)
        bundle_include_challengers = bool(getattr(bundle_cfg, "include_challengers", True))
        bundle_log_to_mlflow = bool(getattr(bundle_cfg, "log_to_mlflow", True))

        bundler = ArtifactBundler(base_dir=bundle_base_dir)
        paths = bundler.create_bundle_dir(bundle_id=bundle_id_effective)
        bundler.write_config(paths, config_path=cfg.source_config_path)
        preproc = build_simple_preprocessor_from_dataframe(X)
        bundler.write_preprocessor(paths, preproc)

        # Champion-Auswahl: SurrogateTree wird nicht als Kandidat berücksichtigt.
        _surr_names = {SURROGATE_MODEL_NAME, SURROGATE_CF_NAME}
        cate_model_names = [n for n in models.keys() if n not in _surr_names]
        entries = [
            ModelEntry(name=mname, artifact_path=f"models/{mname}.pkl", metrics=float_metrics(eval_summary.get(mname, {}) or {}))
            for mname in cate_model_names
        ]
        manual_champion = (getattr(cfg.selection, "manual_champion", None) or "").strip() or None
        selection_cfg = {"metric": cfg.selection.metric, "higher_is_better": cfg.selection.higher_is_better, "manual_champion": manual_champion}
        if manual_champion is not None:
            champion = manual_champion
        else:
            champion = choose_champion(entries, metric=cfg.selection.metric, higher_is_better=cfg.selection.higher_is_better)
            if champion is None and entries:
                available_metrics = sorted(set(k for e in entries for k in e.metrics.keys()))
                self._logger.warning("Champion-Auswahl: Metrik '%s' nicht gefunden. Verfügbar: %s. Fallback auf erstes Modell.", cfg.selection.metric, available_metrics)
                champion = entries[0].name

        models_to_export = {n: m for n, m in models.items() if n not in _surr_names}
        registry_entries = list(entries)
        if not bundle_include_challengers and champion is not None:
            models_to_export = {champion: models[champion]}
            registry_entries = [e for e in entries if e.name == champion]

        champion_refit_on_full_data = False
        champion_refit_rows = None
        champion_fitted_obj = None
        for mname, mobj in models_to_export.items():
            obj_to_write = mobj
            if mname == champion and bool(getattr(cfg.selection, "refit_champion_on_full_data", True)):
                try:
                    obj_to_write = copy.deepcopy(mobj)
                except Exception:
                    self._logger.warning("deepcopy des Champion-Modells fehlgeschlagen, verwende Original.", exc_info=True)
                    obj_to_write = mobj
                X_refit = X_full.loc[:, selected_feature_columns].copy()
                obj_to_write.fit(Y_full, T_full, X=X_refit)
                champion_refit_on_full_data = True
                champion_refit_rows = int(len(X_refit))
                champion_fitted_obj = obj_to_write
            else:
                if not holdout_data:
                    try:
                        obj_to_write = copy.deepcopy(mobj)
                    except Exception:
                        self._logger.warning("deepcopy des Challenger-Modells %s fehlgeschlagen.", mname, exc_info=True)
                        obj_to_write = mobj
                    X_fit = X.loc[:, selected_feature_columns].copy()
                    obj_to_write.fit(Y, T, X=X_fit)
                if mname == champion:
                    champion_fitted_obj = obj_to_write
            bundler.write_model(paths, mname, obj_to_write)

        # Surrogate-Einzelbaum für das Bundle:
        # Wird auf den (ggf. refitteten) Champion-Predictions trainiert.
        # Bei MT wird pro Treatment-Arm ein separater Baum trainiert.
        surrogate_exported = False
        cf_surrogate_exported = False
        if cfg.surrogate_tree.enabled and champion_fitted_obj is not None:
            try:
                if champion_refit_on_full_data:
                    X_surr = X_full.loc[:, selected_feature_columns].copy()
                else:
                    X_surr = X.loc[:, selected_feature_columns].copy()

                champion_preds_for_surr = _predict_effect(champion_fitted_obj, X_surr)
                champion_preds_for_surr = np.asarray(champion_preds_for_surr)

                if champion_preds_for_surr.ndim == 2 and champion_preds_for_surr.shape[1] > 1:
                    # MT: pro Arm einen Baum trainieren
                    n_effects = champion_preds_for_surr.shape[1]
                    bundle_trees = {}
                    for k in range(n_effects):
                        tree_k = self._build_surrogate_regressor(cfg)
                        tree_k.fit(X_surr, champion_preds_for_surr[:, k])
                        bundle_trees[k] = tree_k
                    bundle_surrogate = SurrogateTreeWrapper(trees=bundle_trees, champion_name=champion)
                    log_tree = bundle_trees.get(0)
                else:
                    bundle_tree = self._build_surrogate_regressor(cfg)
                    bundle_tree.fit(X_surr, champion_preds_for_surr.reshape(-1))
                    bundle_surrogate = SurrogateTreeWrapper(tree=bundle_tree, champion_name=champion)
                    log_tree = bundle_tree

                bundler.write_model(paths, SURROGATE_MODEL_NAME, bundle_surrogate)
                surrogate_exported = True

                # Surrogate in Registry aufnehmen (ohne Champion-Konkurrenz)
                surrogate_entry = ModelEntry(
                    name=SURROGATE_MODEL_NAME,
                    artifact_path=f"models/{SURROGATE_MODEL_NAME}.pkl",
                    metrics=float_metrics(eval_summary.get(SURROGATE_MODEL_NAME, {}) or {}),
                )
                registry_entries.append(surrogate_entry)

                depth, n_leaves = self._log_surrogate_tree_info(log_tree, cfg.base_learner.type) if log_tree else (None, None)
                self._logger.info(
                    "Surrogate-Einzelbaum exportiert (Typ=%s, Tiefe=%s, Blätter=%s, trainiert auf %d Zeilen).",
                    cfg.base_learner.type, depth, n_leaves, len(X_surr),
                )
            except Exception:
                self._logger.warning("Surrogate-Tree Bundle-Export fehlgeschlagen.", exc_info=True)

        # CausalForestDML-Surrogate: separat im Bundle exportieren falls vorhanden
        if SURROGATE_CF_NAME in models and SURROGATE_CF_NAME in eval_summary:
            try:
                bundler.write_model(paths, SURROGATE_CF_NAME, models[SURROGATE_CF_NAME])
                cf_surrogate_exported = True
                cf_surr_entry = ModelEntry(
                    name=SURROGATE_CF_NAME,
                    artifact_path=f"models/{SURROGATE_CF_NAME}.pkl",
                    metrics=float_metrics(eval_summary.get(SURROGATE_CF_NAME, {}) or {}),
                )
                registry_entries.append(cf_surr_entry)
            except Exception:
                self._logger.warning("SurrogateTree_CausalForestDML Bundle-Export fehlgeschlagen.", exc_info=True)

        all_surrogate_names = []
        if surrogate_exported:
            all_surrogate_names.append(SURROGATE_MODEL_NAME)
        if cf_surrogate_exported:
            all_surrogate_names.append(SURROGATE_CF_NAME)

        write_registry(paths.root, entries=registry_entries, champion=champion, selection=selection_cfg)

        bundler.write_metadata(paths, {
            "models": list(models_to_export.keys()) + all_surrogate_names,
            "champion": champion,
            "treatment_type": cfg.treatment.type,
            "n_treatment_arms": int(len(np.unique(T_full))),
            "reference_group": cfg.treatment.reference_group,
            "selection_metric": cfg.selection.metric, "selection_manual_champion": manual_champion,
            "created_from_run": True, "champion_refit_on_full_data": champion_refit_on_full_data,
            "champion_refit_rows": champion_refit_rows, "selected_feature_columns": selected_feature_columns,
            "bundle_include_challengers": bundle_include_challengers,
            "surrogate_tree_enabled": surrogate_exported,
        })
        if bundle_log_to_mlflow:
            mlflow.log_artifacts(str(paths.root), artifact_path=f"bundle_{paths.root.name}")

    def _run_optional_output(self, cfg, eval_summary, removed, preds):
        """Optionale lokale Ausgaben (zusätzlich zu MLflow)."""
        if not cfg.optional_output.output_dir:
            return
        out_dir = cfg.optional_output.output_dir
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "removed_features.json"), "w", encoding="utf-8") as f:
            json.dump(removed, f, ensure_ascii=False, indent=2)
        if cfg.optional_output.save_predictions:
            fmt = (cfg.optional_output.predictions_format or "parquet").lower()
            max_rows = cfg.optional_output.max_prediction_rows
            for model_name, df_pred in preds.items():
                df_out = df_pred if max_rows is None or len(df_pred) <= max_rows else df_pred.iloc[:max_rows].copy()
                if fmt == "parquet":
                    try:
                        df_out.to_parquet(os.path.join(out_dir, f"predictions_{model_name}.parquet"), index=False)
                    except Exception:
                        self._logger.warning("Parquet-Export fehlgeschlagen, Fallback auf CSV.", exc_info=True)
                        df_out.to_csv(os.path.join(out_dir, f"predictions_{model_name}.csv"), index=False, float_format="%.10g")
                else:
                    df_out.to_csv(os.path.join(out_dir, f"predictions_{model_name}.csv"), index=False, float_format="%.10g")

    # ------------------------------------------------------------------
    # Hauptmethode
    # ------------------------------------------------------------------

    def run(self, export_bundle: bool | None = None, bundle_dir: str | None = None, bundle_id: str | None = None) -> AnalysisResult:
        """Startet den Analyselauf."""
        cfg = self.cfg

        # ── Harmlose sklearn-Warnung unterdrücken ──
        # EconML übergibt X intern mal als DataFrame (mit Spaltennamen), mal als
        # numpy-Array (ohne Namen). sklearn warnt dann bei predict(), obwohl die
        # Feature-Reihenfolge garantiert ist. Die Warnung ist rein kosmetisch und
        # würde das Log mit hunderten identischen Zeilen fluten.
        import warnings
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module=r"sklearn\.utils\.validation",
        )
        # MLflow filesystem-backend Deprecation-Warnung unterdrücken.
        # Die Migration zu SQLite/DB-Backend ist eine Infrastrukturentscheidung
        # und kein Handlungsbedarf für den einzelnen Analyselauf.
        warnings.filterwarnings(
            "ignore",
            message=".*filesystem tracking backend.*is deprecated",
            category=FutureWarning,
            module=r"mlflow",
        )
        # EconML nutzt intern den alten sklearn-Parameter 'force_all_finite',
        # der in sklearn 1.6 zu 'ensure_all_finite' umbenannt wurde. Die Warnung
        # ist rein kosmetisch — EconML wird das in einer zukünftigen Version
        # aktualisieren. Bis dahin unterdrücken wir die hundertfache Wiederholung.
        warnings.filterwarnings(
            "ignore",
            message=".*force_all_finite.*was renamed to.*ensure_all_finite",
            category=FutureWarning,
            module=r"sklearn",
        )

        try:
            import mlflow
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("MLflow ist nicht installiert. Für Analyseläufe wird MLflow benötigt (pip install mlflow).") from e

        # ── Schritte zählen ──
        total = 5  # load, fs, tune, train, eval
        has_cf = "CausalForestDML" in (cfg.models.models_to_train or [])
        if cfg.surrogate_tree.enabled or has_cf:
            total += 1
        bundle_enabled = export_bundle if export_bundle is not None else cfg.bundle.enabled
        if bundle_enabled:
            total += 1
        total += 1  # report
        step = [0]
        step_times = {}
        _last_step_start = [time.perf_counter()]
        _last_step_label = [None]

        def _progress(label: str):
            now = time.perf_counter()
            if _last_step_label[0] is not None:
                step_times[_last_step_label[0]] = now - _last_step_start[0]
            _last_step_start[0] = now
            _last_step_label[0] = label
            step[0] += 1
            msg = f"[rubin] Step {step[0]}/{total}: {label}"
            print(msg, flush=True)
            self._logger.info(msg)

        mlflow.set_experiment(cfg.mlflow.experiment_name)
        with mlflow.start_run():
            mlflow.log_param("seed", cfg.constants.random_seed)
            mlflow.log_param("base_learner", cfg.base_learner.type)
            mlflow.log_param("parallel_level", cfg.constants.parallel_level)

            # ── Config-YAML nach MLflow loggen ──
            if cfg.source_config_path and os.path.isfile(cfg.source_config_path):
                try:
                    mlflow.log_artifact(cfg.source_config_path)
                except Exception:
                    self._logger.warning("Config-YAML konnte nicht nach MLflow geloggt werden.", exc_info=True)
            else:
                # Kein Datei-Pfad (z.B. UI-generiert) → Config als YAML-Text loggen
                try:
                    import yaml
                    cfg_dict = cfg.model_dump(mode="json", exclude_none=True)
                    _log_temp_artifact(mlflow, lambda p: open(p, "w", encoding="utf-8").write(yaml.dump(cfg_dict, allow_unicode=True, sort_keys=False)), "config.yml")
                except Exception:
                    self._logger.warning("Config konnte nicht als Artefakt geloggt werden.", exc_info=True)

            # Report-Collector initialisieren
            report = ReportCollector()
            report.add_config(cfg)
            self._report = report

            _progress("Daten laden & Preprocessing")
            X, T, Y, S, eval_mask = self._load_inputs()

            if eval_mask is not None:
                mlflow.log_param("eval_mask_mode", "train_many_evaluate_one")
                mlflow.log_metric("eval_mask_n_eval", int(eval_mask.sum()))
                mlflow.log_metric("eval_mask_n_total", len(eval_mask))

            # X_full/T_full/Y_full werden nur für Bundle-Refit benötigt.
            # Kopie erst erstellen wenn nötig, spart ~2 GB bei normalen Läufen.
            if bundle_enabled:
                X_full, T_full, Y_full = X.copy(), np.asarray(T).copy(), np.asarray(Y).copy()
            else:
                X_full, T_full, Y_full = X, T, Y  # Referenz, keine Kopie

            use_holdout = (
                str(getattr(cfg.data_processing, "validate_on", "cross")).lower() in {"holdout"}
                and float(getattr(cfg.data_processing, "test_size", 0.0) or 0.0) > 0
            )
            use_external = str(getattr(cfg.data_processing, "validate_on", "cross")).lower() == "external"
            holdout_data = None

            if use_external:
                # Externe Evaluationsdaten laden
                self._logger.info("Validierungsmodus: external – lade separate Eval-Daten.")
                X_eval = self._read_table(cfg.data_files.eval_x_file)
                T_eval = self._read_table(cfg.data_files.eval_t_file)["T"].to_numpy()
                Y_eval = self._read_table(cfg.data_files.eval_y_file)["Y"].to_numpy()
                S_eval = None
                if cfg.data_files.eval_s_file:
                    try:
                        col = cfg.historical_score.column
                        S_eval = self._read_table(cfg.data_files.eval_s_file)[col].to_numpy(dtype=float)
                        S_eval = np.nan_to_num(S_eval, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception:
                        S_eval = None
                holdout_data = (X_eval, T_eval, Y_eval, S_eval)
                mlflow.log_param("validation_mode", "external")
                mlflow.log_param("eval_n_rows", len(X_eval))
                self._logger.info("Eval-Daten: %d Zeilen, %d Features.", len(X_eval), X_eval.shape[1])
            elif use_holdout:
                X_train, X_test, T_train, T_test, Y_train, Y_test, S_train, S_test = stratified_train_test_split(
                    X=X, T=T, Y=Y, S=S, test_size=float(cfg.data_processing.test_size), random_state=cfg.constants.random_seed)
                X, T, Y, S = X_train, T_train, Y_train, S_train
                holdout_data = (X_test, T_test, Y_test, S_test)
                mlflow.log_param("validation_mode", "holdout")
                mlflow.log_param("holdout_test_size", float(cfg.data_processing.test_size))
            else:
                mlflow.log_param("validation_mode", "cross")

            report.add_data_stats(X, T, Y, S)
            if use_external and holdout_data is not None:
                report.add_eval_data_stats(*holdout_data)

            _progress("Feature-Selektion")
            # Kategorischer Patch auch für Feature-Selektion: LightGBM-Importance
            # braucht categorical_feature, sonst werden kategorische Features
            # systematisch unterbewertet und fliegen möglicherweise raus.
            with patch_categorical_features(X, base_learner_type=cfg.base_learner.type):
                X, removed = self._run_feature_selection(cfg, X, T, Y, mlflow)
            if removed.get("importance") or removed.get("high_correlation"):
                report.feature_selection_info["n_before"] = report.data_stats.get("n_features", 0)
                report.feature_selection_info["n_after"] = len(X.columns)

            # Feature-Alignment: holdout/external Eval-Daten auf selektierte Features reduzieren
            if holdout_data is not None:
                X_h, T_h, Y_h, S_h = holdout_data
                if list(X_h.columns) != list(X.columns):
                    X_h = X_h.reindex(columns=X.columns)
                    self._logger.info(
                        "Eval-Daten: Feature-Alignment auf %d selektierte Spalten angewendet.",
                        len(X.columns),
                    )
                    holdout_data = (X_h, T_h, Y_h, S_h)

            _progress("Base-Learner-Tuning")
            # Zweiter Patch mit post-FS Spaltenindizes (Spalten haben sich geändert).
            with patch_categorical_features(X, base_learner_type=cfg.base_learner.type) as cat_indices:
                if cat_indices:
                    mlflow.log_param("n_categorical_features", len(cat_indices))

                tuned_params_by_model = self._run_tuning(cfg, X, T, Y, mlflow)
                gc.collect()  # Tuner-Internals freigeben (Optuna Studies, Trial-Daten)

                _progress("Training & Cross-Predictions")
                models, preds = self._run_training(cfg, X, T, Y, tuned_params_by_model, holdout_data, mlflow)
                selected_feature_columns = list(X.columns)

                _progress("Evaluation & Metriken")
                eval_summary: Dict[str, Dict[str, float]] = {}
                try:
                    eval_summary, _ = self._run_evaluation(cfg, X, T, Y, S, holdout_data, preds, models, tuned_params_by_model, mlflow, eval_mask=eval_mask)
                except Exception:
                    self._logger.warning("Uplift-Evaluation fehlgeschlagen.", exc_info=True)

                # ── Surrogate-Einzelbäume ──
                # 1. CausalForestDML-Surrogate: IMMER wenn CausalForestDML trainiert wurde
                # 2. Champion-Surrogate: Wenn surrogate_tree.enabled UND Champion ≠ CausalForestDML
                if eval_summary:
                    _progress("Surrogate-Tree")
                    champion_name = self._determine_champion(cfg, eval_summary, models)

                    # --- 1. CausalForestDML-Surrogate ---
                    cf_name = "CausalForestDML"
                    if cf_name in preds:
                        try:
                            self._logger.info("Trainiere Surrogate auf %s (immer, unabhängig von Champion).", cf_name)
                            # Ranking bestimmen: DRTester-Plots nur wenn CF auf Rang 1 oder 2
                            metric = cfg.selection.metric
                            higher = cfg.selection.higher_is_better
                            ranked = sorted(
                                [(n, m.get(metric, 0)) for n, m in eval_summary.items()
                                 if n not in (SURROGATE_MODEL_NAME, SURROGATE_CF_NAME) and isinstance(m, dict)],
                                key=lambda x: x[1], reverse=higher,
                            )
                            cf_rank = next((i + 1 for i, (n, _) in enumerate(ranked) if n == cf_name), 99)
                            do_drtester = cf_rank <= 2
                            self._logger.info("CausalForestDML Rang: %d → DRTester-Plots: %s", cf_rank, "ja" if do_drtester else "nein")

                            cf_surr_wrapper, cf_surr_df = self._train_and_evaluate_surrogate(
                                cfg, X, T, Y, cf_name, preds, holdout_data,
                                models, eval_summary, mlflow,
                                surrogate_name=SURROGATE_CF_NAME,
                                run_drtester=do_drtester,
                                fitted_tester=fitted_tester_bt,
                                eval_mask=eval_mask,
                            )
                            preds[SURROGATE_CF_NAME] = cf_surr_df
                            models[SURROGATE_CF_NAME] = cf_surr_wrapper
                        except Exception:
                            self._logger.warning("SurrogateTree_CausalForestDML fehlgeschlagen.", exc_info=True)

                    # --- 2. Champion-Surrogate (klassisch) ---
                    if cfg.surrogate_tree.enabled and champion_name and champion_name != cf_name and champion_name in preds:
                        try:
                            self._logger.info("Trainiere Surrogate auf Champion %s.", champion_name)
                            surrogate_wrapper, surrogate_df = self._train_and_evaluate_surrogate(
                                cfg, X, T, Y, champion_name, preds, holdout_data,
                                models, eval_summary, mlflow,
                                surrogate_name=SURROGATE_MODEL_NAME,
                                eval_mask=eval_mask,
                            )
                            preds[SURROGATE_MODEL_NAME] = surrogate_df
                            models[SURROGATE_MODEL_NAME] = surrogate_wrapper
                        except Exception:
                            self._logger.warning("SurrogateTree (Champion) fehlgeschlagen.", exc_info=True)
                    elif cfg.surrogate_tree.enabled and champion_name == cf_name:
                        # Champion IST CausalForestDML → CF-Surrogate ist gleichzeitig der Champion-Surrogate
                        if SURROGATE_CF_NAME in models:
                            preds[SURROGATE_MODEL_NAME] = preds.get(SURROGATE_CF_NAME)
                            models[SURROGATE_MODEL_NAME] = models.get(SURROGATE_CF_NAME)
                            # eval_summary ebenfalls kopieren
                            if SURROGATE_CF_NAME in eval_summary:
                                eval_summary[SURROGATE_MODEL_NAME] = eval_summary[SURROGATE_CF_NAME]
                    elif cfg.surrogate_tree.enabled:
                        self._logger.warning("Surrogate-Tree: Kein Champion ermittelt, überspringe.")

                gc.collect()  # Nach Surrogate freigeben
                self._logger.info("RAM-Optimierung: gc.collect() nach Surrogate.")

                if bundle_enabled:
                    _progress("Bundle-Export")
                self._run_bundle_export(cfg, models, eval_summary, X, T, Y, X_full, T_full, Y_full, selected_feature_columns, holdout_data, export_bundle, bundle_dir, bundle_id, mlflow)
                self._run_optional_output(cfg, eval_summary, removed, preds)
            # ── Ende categorical patch — originale .fit()-Methoden wiederhergestellt ──

            # ── RAM-Optimierung: Nicht mehr benötigte Objekte freigeben ──
            # Nach Bundle-Export und Prediction-Output werden nur noch
            # eval_summary und report für den HTML-Report benötigt.
            _champ = self._determine_champion(cfg, eval_summary, models) if eval_summary else None
            _keep = {_champ, SURROGATE_MODEL_NAME, SURROGATE_CF_NAME}
            for mname in list(models.keys()):
                if mname not in _keep:
                    del models[mname]
            preds.clear()
            del X_full, T_full, Y_full
            gc.collect()
            self._logger.info("RAM-Optimierung: Modelle, Predictions und X_full freigegeben.")

            # ── .rubin_cache: Immer schreiben, damit der Server den Report finden kann ──
            cache_dir = os.path.join(os.getcwd(), ".rubin_cache")
            os.makedirs(cache_dir, exist_ok=True)

            # ── eval_summary.json → .rubin_cache + MLflow + output_dir ──
            try:
                cache_summary = os.path.join(cache_dir, "uplift_eval_summary.json")
                with open(cache_summary, "w", encoding="utf-8") as f:
                    json.dump(eval_summary, f, ensure_ascii=False, indent=2, default=float)
                _log_temp_artifact(mlflow, lambda p: open(p, "w", encoding="utf-8").write(
                    json.dumps(eval_summary, ensure_ascii=False, indent=2, default=float)
                ), "uplift_eval_summary.json")
                if cfg.optional_output.output_dir:
                    summary_path = os.path.join(cfg.optional_output.output_dir, "uplift_eval_summary.json")
                    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
                    import shutil
                    shutil.copy2(cache_summary, summary_path)
            except Exception:
                self._logger.warning("eval_summary.json konnte nicht gespeichert werden.", exc_info=True)

            # ── HTML-Report generieren ──
            _progress("HTML-Report")
            try:
                # Modell-Metriken + Champion
                for mname, metrics in eval_summary.items():
                    report.model_metrics[mname] = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                if eval_summary:
                    try:
                        report.champion_name = self._determine_champion(cfg, eval_summary, models) or ""
                    except Exception:
                        pass

                # Surrogate-Info
                if SURROGATE_MODEL_NAME in eval_summary:
                    report.surrogate_info = {
                        "champion": report.champion_name,
                        "metrics": {k: v for k, v in eval_summary[SURROGATE_MODEL_NAME].items() if isinstance(v, (int, float))},
                    }

                # Letzten Schritt abschließen + Timing (nach Setup, vor Generate)
                if _last_step_label[0] is not None:
                    step_times[_last_step_label[0]] = time.perf_counter() - _last_step_start[0]
                report.step_durations = step_times
                report.total_elapsed = sum(step_times.values())

                # Report generieren → .rubin_cache (immer) + MLflow + output_dir
                cache_report = os.path.join(cache_dir, "analysis_report.html")
                generate_html_report(report, cache_report)
                mlflow.log_artifact(cache_report)
                if cfg.optional_output.output_dir:
                    os.makedirs(cfg.optional_output.output_dir, exist_ok=True)
                    import shutil
                    shutil.copy2(cache_report, os.path.join(cfg.optional_output.output_dir, "analysis_report.html"))
                self._logger.info("HTML-Report geschrieben: %s", cache_report)
            except Exception:
                self._logger.warning("HTML-Report-Generierung fehlgeschlagen.", exc_info=True)

            print(f"[rubin] Step {total}/{total}: Fertig", flush=True)
            return AnalysisResult(models=models, predictions=preds, removed_features=removed, eval_summary=eval_summary)
