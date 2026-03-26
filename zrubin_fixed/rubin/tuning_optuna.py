from __future__ import annotations

"""Optuna-basiertes Hyperparameter-Tuning für Base Learner.
Kernidee:
- Es werden nicht "die kausalen Learner" direkt getunt, sondern die darunter
verwendeten Base Learner (z. B. Outcome-/Propensity-Modelle).
- Die verfügbare Datenmenge kann je kausalem Learner unterschiedlich sein
(S-Learner: alle Daten; T-/X-Learner: gruppenweise Daten). Diese Logik wird
beim Sampling der Tuning-Daten berücksichtigt, damit die gefundenen Hyperparameter
realistische Bedingungen widerspiegeln."""


from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

from rubin.settings import AnalysisConfig, SearchSpaceConfig, SearchSpaceParameterConfig


def _iter_stratified_or_kfold(labels: np.ndarray, n_splits: int, seed: int):
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1:
        labels_arr = labels_arr.reshape(-1)

    # Safety: Mindestens 2 Folds für Cross-Validation
    n_splits = max(2, int(n_splits))
    if len(labels_arr) < 2:
        import logging
        log = logging.getLogger("rubin.tuning")
        log.error(
            "Tuning-Split fehlgeschlagen: target hat nur %d Element(e). "
            "dtype=%s, unique=%s, shape=%s. "
            "Prüfe ob X/T/Y gleich viele Zeilen haben und ob df_frac aktiv ist.",
            len(labels_arr), labels_arr.dtype, np.unique(labels_arr).tolist(), labels_arr.shape,
        )
        raise ValueError(
            f"Für die Aufteilung werden mindestens 2 Beobachtungen benötigt "
            f"(erhalten: {len(labels_arr)}, unique={np.unique(labels_arr).tolist()}, "
            f"dtype={labels_arr.dtype})."
        )

    counts = pd.Series(labels_arr).value_counts(dropna=False)
    effective_splits = min(int(n_splits), len(labels_arr))
    if not counts.empty:
        effective_splits = min(effective_splits, int(counts.min()))

    if effective_splits >= 2:
        cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
        return cv.split(np.zeros(len(labels_arr)), labels_arr)

    fallback_splits = min(int(n_splits), len(labels_arr))
    if fallback_splits < 2:
        import logging
        log = logging.getLogger("rubin.tuning")
        log.error(
            "Tuning-Split Fallback fehlgeschlagen: n_splits=%d, len=%d, unique=%s, counts=%s",
            n_splits, len(labels_arr), np.unique(labels_arr).tolist(),
            dict(pd.Series(labels_arr).value_counts(dropna=False)),
        )
        raise ValueError(
            f"Für die Aufteilung werden mindestens 2 Folds benötigt "
            f"(n_splits={n_splits}, len={len(labels_arr)}, "
            f"unique={np.unique(labels_arr).tolist()})."
        )
    cv = KFold(n_splits=fallback_splits, shuffle=True, random_state=seed)
    return cv.split(np.zeros(len(labels_arr)))


def _first_crossfit_train_indices(n: int, t: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Ermittelt die Trainingsindizes des ersten Cross-Prediction-Folds.

    Die Aufteilung folgt derselben robusten Logik wie die übrigen Split-Helfer:
    Wenn eine Stratifikation wegen zu kleiner Klassen nicht zulässig ist,
    wird kontrolliert auf einen normalen KFold zurückgefallen.
    """
    t_int = np.asarray(t).astype(int)
    split_iter = _iter_stratified_or_kfold(t_int, n_splits=n_splits, seed=seed)
    for tr_idx, _ in split_iter:
        return np.asarray(tr_idx, dtype=int)
    return np.arange(n, dtype=int)


def _safe_import_optuna():
    try:
        import optuna  # type: ignore
        # Performance: Optuna-Logging auf WARNING reduzieren
        # (unterdrückt "Trial X finished with value..." Meldungen)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna
    except Exception as e:
        raise ImportError(
            "Optuna-Tuning ist aktiviert, aber 'optuna' konnte nicht importiert werden. "
            "Bitte installieren mit: pip install optuna"
        ) from e


def _build_lgbm_classifier(params: Dict[str, Any], seed: int, parallel_jobs: int = -1):
    import lightgbm as lgbm
    # Möglichst deterministisches Verhalten
    fixed = dict(
        random_state=seed,
        verbose=-1,  # Unterdrückt C++-Level stdout (KRITISCH für Pipe-Performance)
    )
    fixed.update(params)
    # n_jobs NACH params setzen, damit fixed_params es nicht überschreiben können.
    fixed["n_jobs"] = parallel_jobs
    # Objective wird NICHT hardcodiert, damit LGBMClassifier binary vs.
    # multiclass automatisch aus den Trainingsdaten ableiten kann.
    # Für BT-Propensity/Outcome → binary; für MT-Propensity → multiclass.
    fixed.pop("objective", None)
    return lgbm.LGBMClassifier(**fixed)


def _build_lgbm_regressor(params: Dict[str, Any], seed: int, parallel_jobs: int = -1):
    import lightgbm as lgbm
    fixed = dict(
        random_state=seed,
        verbose=-1,  # Unterdrückt C++-Level stdout (KRITISCH für Pipe-Performance)
    )
    fixed.update(params)
    # n_jobs NACH params setzen, damit fixed_params es nicht überschreiben können.
    fixed["n_jobs"] = parallel_jobs
    # Für Effektmodelle ist Regression zwingend sinnvoll. Falls in fixed_params ein
    # Klassifikations-Objective gesetzt wurde, überschreiben wir es bewusst.
    fixed["objective"] = "regression"
    return lgbm.LGBMRegressor(**fixed)


def _build_catboost_classifier(params: Dict[str, Any], seed: int, parallel_jobs: int = -1):
    from catboost import CatBoostClassifier  # type: ignore
    fixed = dict(
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    fixed.update(params)
    # thread_count NACH params setzen, damit fixed_params es nicht überschreiben können.
    fixed["thread_count"] = parallel_jobs
    # loss_function und eval_metric werden NICHT erzwungen, damit CatBoost
    # binary (Logloss) vs. multiclass (MultiClass) automatisch ableiten kann.
    # Bei BT wählt CatBoost Logloss, bei MT-Propensity MultiClass.
    return CatBoostClassifier(**fixed)


def _build_catboost_regressor(params: Dict[str, Any], seed: int, parallel_jobs: int = -1):
    from catboost import CatBoostRegressor  # type: ignore
    fixed = dict(
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    fixed.update(params)
    # thread_count NACH params setzen, damit fixed_params es nicht überschreiben können.
    fixed["thread_count"] = parallel_jobs
    # Analog zur LightGBM-Logik: für Effektmodelle wird eine Regression trainiert.
    fixed["loss_function"] = "RMSE"
    return CatBoostRegressor(**fixed)


def build_base_learner(base_type: str, params: Dict[str, Any], seed: int, task: str = "classifier", parallel_jobs: int = -1):
    """Die Funktion wird sowohl im Tuning als auch beim finalen Training verwendet, um sicherzustellen,
dass identische Defaults (z. B. Random Seed) gesetzt werden.
parallel_jobs: Anzahl Kerne für Base Learner. -1 = alle, 1 = single-core."""
    base_type = (base_type or "lgbm").lower()
    task = (task or "classifier").lower()
    if task not in {"classifier", "regressor"}:
        raise ValueError(f"Unbekannter task={task!r}. Erwartet: 'classifier' oder 'regressor'.")

    if base_type == "lgbm":
        return _build_lgbm_classifier(params, seed, parallel_jobs) if task == "classifier" else _build_lgbm_regressor(params, seed, parallel_jobs)
    if base_type == "catboost":
        return _build_catboost_classifier(params, seed, parallel_jobs) if task == "classifier" else _build_catboost_regressor(params, seed, parallel_jobs)
    raise ValueError(f"Unbekannter base_learner.type={base_type!r}. Erwartet: 'lgbm' oder 'catboost'.")


def _default_search_space(base_type: str) -> Dict[str, SearchSpaceParameterConfig]:
    base_type = (base_type or "").lower()
    if base_type == "lgbm":
        return {
            "n_estimators": SearchSpaceParameterConfig(type="int", low=200, high=600),
            "learning_rate": SearchSpaceParameterConfig(type="float", low=1e-2, high=1.5e-1, log=True),
            "num_leaves": SearchSpaceParameterConfig(type="int", low=15, high=127),
            "max_depth": SearchSpaceParameterConfig(type="int", low=3, high=8),
            "min_child_samples": SearchSpaceParameterConfig(type="int", low=10, high=200),
            "min_child_weight": SearchSpaceParameterConfig(type="float", low=1e-2, high=50.0, log=True),
            "subsample": SearchSpaceParameterConfig(type="float", low=0.5, high=1.0),
            "colsample_bytree": SearchSpaceParameterConfig(type="float", low=0.3, high=0.9),
            "min_split_gain": SearchSpaceParameterConfig(type="float", low=0.0, high=1.0),
            "reg_alpha": SearchSpaceParameterConfig(type="float", low=1e-6, high=20.0, log=True),
            "reg_lambda": SearchSpaceParameterConfig(type="float", low=1e-6, high=20.0, log=True),
            "path_smooth": SearchSpaceParameterConfig(type="float", low=0.0, high=10.0),
        }
    if base_type == "catboost":
        return {
            "iterations": SearchSpaceParameterConfig(type="int", low=200, high=600),
            "learning_rate": SearchSpaceParameterConfig(type="float", low=1e-2, high=1.5e-1, log=True),
            "depth": SearchSpaceParameterConfig(type="int", low=4, high=8),
            "l2_leaf_reg": SearchSpaceParameterConfig(type="float", low=1.0, high=30.0, log=True),
            "random_strength": SearchSpaceParameterConfig(type="float", low=1e-2, high=10.0, log=True),
            "subsample": SearchSpaceParameterConfig(type="float", low=0.5, high=1.0),
            "rsm": SearchSpaceParameterConfig(type="float", low=0.3, high=0.9),
            "min_data_in_leaf": SearchSpaceParameterConfig(type="int", low=10, high=200),
            "model_size_reg": SearchSpaceParameterConfig(type="float", low=0.0, high=10.0),
            "leaf_estimation_iterations": SearchSpaceParameterConfig(type="int", low=1, high=10),
        }
    raise ValueError(f"Unbekannter base_type {base_type}")


def _default_fmt_search_space(base_type: str) -> Dict[str, SearchSpaceParameterConfig]:
    """Default-Suchraum für Final-Model-Tuning (CATE-Regressor).

    BEWUSST konservativer als BL: weniger Bäume, flacher, stärkere
    Regularisierung. Verhindert Overfitting auf verrauschte Residuen."""
    base_type = (base_type or "").lower()
    if base_type == "lgbm":
        return {
            "n_estimators": SearchSpaceParameterConfig(type="int", low=50, high=300),
            "learning_rate": SearchSpaceParameterConfig(type="float", low=5e-3, high=8e-2, log=True),
            "num_leaves": SearchSpaceParameterConfig(type="int", low=4, high=31),
            "max_depth": SearchSpaceParameterConfig(type="int", low=2, high=5),
            "min_child_samples": SearchSpaceParameterConfig(type="int", low=100, high=1000),
            "min_child_weight": SearchSpaceParameterConfig(type="float", low=1.0, high=100.0, log=True),
            "subsample": SearchSpaceParameterConfig(type="float", low=0.4, high=0.8),
            "colsample_bytree": SearchSpaceParameterConfig(type="float", low=0.2, high=0.6),
            "max_bin": SearchSpaceParameterConfig(type="int", low=10, high=63),
            "min_split_gain": SearchSpaceParameterConfig(type="float", low=0.0, high=5.0),
            "reg_alpha": SearchSpaceParameterConfig(type="float", low=0.1, high=50.0, log=True),
            "reg_lambda": SearchSpaceParameterConfig(type="float", low=1.0, high=100.0, log=True),
            "path_smooth": SearchSpaceParameterConfig(type="float", low=0.0, high=50.0),
        }
    if base_type == "catboost":
        return {
            "iterations": SearchSpaceParameterConfig(type="int", low=50, high=300),
            "learning_rate": SearchSpaceParameterConfig(type="float", low=5e-3, high=8e-2, log=True),
            "depth": SearchSpaceParameterConfig(type="int", low=2, high=5),
            "l2_leaf_reg": SearchSpaceParameterConfig(type="float", low=5.0, high=100.0, log=True),
            "random_strength": SearchSpaceParameterConfig(type="float", low=1.0, high=20.0, log=True),
            "subsample": SearchSpaceParameterConfig(type="float", low=0.4, high=0.8),
            "rsm": SearchSpaceParameterConfig(type="float", low=0.2, high=0.6),
            "min_data_in_leaf": SearchSpaceParameterConfig(type="int", low=100, high=1000),
            "model_size_reg": SearchSpaceParameterConfig(type="float", low=0.1, high=50.0, log=True),
            "leaf_estimation_iterations": SearchSpaceParameterConfig(type="int", low=1, high=5),
        }
    raise ValueError(f"Unbekannter base_type {base_type}")


def _search_space_for(base_type: str, cfg_space: SearchSpaceConfig | None, is_fmt: bool = False) -> Dict[str, SearchSpaceParameterConfig]:
    base_type = (base_type or "").lower()
    custom = {}
    if cfg_space is not None:
        custom = dict(getattr(cfg_space, base_type, {}) or {})
    if custom:
        return custom
    return _default_fmt_search_space(base_type) if is_fmt else _default_search_space(base_type)


def _suggest_from_spec(trial, name: str, spec: SearchSpaceParameterConfig):
    if spec.type == "categorical":
        return trial.suggest_categorical(name, list(spec.choices or []))
    if spec.type == "int":
        step = None if spec.step is None else int(spec.step)
        return trial.suggest_int(name, int(spec.low), int(spec.high), step=step or 1, log=bool(spec.log))
    if spec.type == "float":
        kwargs = {"log": bool(spec.log)}
        if spec.step is not None and not spec.log:
            kwargs["step"] = float(spec.step)
        return trial.suggest_float(name, float(spec.low), float(spec.high), **kwargs)
    raise ValueError(f"Nicht unterstützter Parametertyp: {spec.type}")


def _apply_conditional_catboost_params(params: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(params)
    bootstrap_type = result.get("bootstrap_type")
    if bootstrap_type == "Bayesian":
        result.pop("subsample", None)
    elif bootstrap_type == "Bernoulli":
        result.pop("bagging_temperature", None)
    return result


def _suggest_params(trial, base_type: str, cfg_space: SearchSpaceConfig | None = None, is_fmt: bool = False) -> Dict[str, Any]:
    base_type = base_type.lower()
    specs = _search_space_for(base_type, cfg_space, is_fmt=is_fmt)
    params: Dict[str, Any] = {}

    # Für CatBoost muss bootstrap_type vor bagging_temperature/subsample
    # gesampelt werden, damit die Abhängigkeit korrekt aufgelöst wird –
    # auch wenn die Search-Space-Definition (z. B. aus YAML) keine
    # deterministische Reihenfolge garantiert.
    if base_type == "catboost" and "bootstrap_type" in specs:
        params["bootstrap_type"] = _suggest_from_spec(trial, "bootstrap_type", specs["bootstrap_type"])

    for name, spec in specs.items():
        if name == "bootstrap_type" and base_type == "catboost":
            continue  # bereits oben gesampelt
        if base_type == "catboost":
            bootstrap_type = params.get("bootstrap_type")
            if name == "bagging_temperature" and bootstrap_type not in {None, "Bayesian"}:
                continue
            if name == "subsample" and bootstrap_type not in {None, "Bernoulli"}:
                continue
        params[name] = _suggest_from_spec(trial, name, spec)

    if base_type == "catboost":
        params = _apply_conditional_catboost_params(params)
    return params




@dataclass
class TunedSet:
    role_params: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class TuningTask:
    key: str
    objective_family: str
    estimator_task: str
    uses_treatment_feature: bool
    sample_scope: str
    target_name: str
    roles: Tuple[Tuple[str, str], ...]


class BaseLearnerTuner:
    """Optimiert Base-Learner-Aufgaben und teilt Ergebnisse zwischen kompatiblen Rollen.

    Die Logik ist task-basiert:
    - Aus den angeforderten kausalen Modellen werden die tatsächlich benötigten
      internen Lernaufgaben abgeleitet.
    - Identische Aufgaben werden nur einmal getunt.
    - Die besten Parameter werden anschließend allen passenden Rollen zugeordnet.
    """

    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg
        self.seed = int(cfg.constants.random_seed)
        self.optuna = _safe_import_optuna() if cfg.tuning.enabled else None
        self.best_scores: Dict[str, float] = {}

    def _create_study(self, study_key: str):
        optuna = self.optuna
        if optuna is None:
            raise RuntimeError("Optuna ist nicht verfügbar. Bitte optuna installieren.")

        # WICHTIG: Jede Study bekommt einen eigenen Seed, abgeleitet aus dem
        # Basis-Seed + study_key. Ohne dies schlägt der TPE-Sampler für verschiedene
        # Tasks (z. B. model_y vs. model_t) exakt dieselben Hyperparameter vor,
        # was zu identischen Tuning-Ergebnissen führt.
        # hashlib statt hash() für Determinismus über Python-Sessions hinweg.
        import hashlib
        base_seed = int(self.cfg.tuning.optuna_seed)
        key_hash = int(hashlib.sha256(study_key.encode()).hexdigest(), 16) % (2**31)
        study_seed = (base_seed + key_hash) % (2**31)
        try:
            sampler = optuna.samplers.TPESampler(seed=study_seed)
        except Exception:
            sampler = None

        if self.cfg.tuning.storage_path:
            try:
                from pathlib import Path
                Path(self.cfg.tuning.storage_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            storage = f"sqlite:///{self.cfg.tuning.storage_path}"
            prefix = str(self.cfg.tuning.study_name_prefix).strip() or "baselearner"
            base_name = f"{prefix}__{study_key}"

            if bool(self.cfg.tuning.reuse_study_if_exists):
                return optuna.create_study(
                    study_name=base_name,
                    storage=storage,
                    direction="maximize",
                    load_if_exists=True,
                    sampler=sampler,
                )

            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            return optuna.create_study(
                study_name=f"{base_name}__{ts}",
                storage=storage,
                direction="maximize",
                load_if_exists=False,
                sampler=sampler,
            )

        return optuna.create_study(direction="maximize", sampler=sampler)

    def _role_signature(self, model_name: str, role: str) -> Tuple[str, str, bool, str, str]:
        name = (model_name or "").lower()
        role = (role or "").lower()

        # ── Meta-Learner: Outcome-Modelle sind Regressoren ──
        # EconML's SLearner/TLearner/XLearner rufen model.predict() auf.
        # Regressor.predict() = E[Y|X] ∈ [0,1]; Classifier.predict() = {0,1}.
        if name == "slearner" and role == "overall_model":
            return ("outcome_regression", "regressor", True, "all", "Y")

        if name == "tlearner" and role == "models":
            return ("grouped_outcome_regression", "regressor", False, "group_specific_shared_params", "Y")

        if name == "xlearner":
            if role == "models":
                return ("grouped_outcome_regression", "regressor", False, "group_specific_shared_params", "Y")
            if role == "cate_models":
                return ("pseudo_effect", "regressor", False, "group_specific_shared_params", "D")
            if role == "propensity_model":
                return ("propensity", "classifier", False, "all", "T")

        # ── DRLearner ──
        if name == "drlearner":
            if role == "model_propensity":
                return ("propensity", "classifier", False, "all", "T")
            if role == "model_regression":
                # DRLearner ruft model_regression.predict() auf (kein predict_proba).
                # Regressor liefert E[Y|X,T] → korrekte DR-Pseudo-Outcomes.
                return ("outcome_regression", "regressor", False, "all", "Y")

        # ── DML-Familie: Nuisance-Modelle sind Classifier ──
        # EconML wickelt predict_proba via discrete_outcome/discrete_treatment.
        if name in {"nonparamdml", "paramdml", "causalforestdml"}:
            if role == "model_y":
                return ("outcome", "classifier", False, "all", "Y")
            if role == "model_t":
                return ("propensity", "classifier", False, "all", "T")

        raise KeyError(f"Keine Tuning-Signatur für {model_name}.{role} definiert.")

    def _roles_for_model(self, model_name: str) -> Dict[str, str]:
        name = model_name.lower()
        if name == "slearner":
            return {"overall_model": "y"}
        if name == "tlearner":
            return {"models": "y"}
        if name == "xlearner":
            return {"models": "y", "cate_models": "d", "propensity_model": "t"}
        if name == "drlearner":
            return {"model_propensity": "t", "model_regression": "y"}
        if name in {"nonparamdml", "paramdml", "causalforestdml"}:
            return {"model_y": "y", "model_t": "t"}
        return {}

    def _task_key(self, model_name: str, role: str) -> str:
        objective_family, estimator_task, uses_treatment_feature, sample_scope, target_name = self._role_signature(model_name, role)
        parts = [
            self.cfg.base_learner.type.lower(),
            objective_family,
            estimator_task,
            sample_scope,
            "with_t" if uses_treatment_feature else "no_t",
            target_name.lower(),
        ]
        if self.cfg.tuning.per_role:
            parts.append(role.lower())
        if self.cfg.tuning.per_learner:
            parts.append(model_name.lower())
        return "__".join(parts)

    def _build_plan(self, model_names: List[str]) -> Dict[str, TuningTask]:
        plan: Dict[str, TuningTask] = {}
        collectors: Dict[str, List[Tuple[str, str]]] = {}

        for model_name in model_names:
            roles = self._roles_for_model(model_name)
            for role in roles:
                key = self._task_key(model_name, role)
                collectors.setdefault(key, []).append((model_name, role))
                if key not in plan:
                    objective_family, estimator_task, uses_treatment_feature, sample_scope, target_name = self._role_signature(model_name, role)
                    plan[key] = TuningTask(
                        key=key,
                        objective_family=objective_family,
                        estimator_task=estimator_task,
                        uses_treatment_feature=uses_treatment_feature,
                        sample_scope=sample_scope,
                        target_name=target_name,
                        roles=tuple(),
                    )

        for key, roles in collectors.items():
            task = plan[key]
            plan[key] = TuningTask(
                key=task.key,
                objective_family=task.objective_family,
                estimator_task=task.estimator_task,
                uses_treatment_feature=task.uses_treatment_feature,
                sample_scope=task.sample_scope,
                target_name=task.target_name,
                roles=tuple(sorted(roles)),
            )
        return plan

    def _task_priority(self, task: TuningTask) -> int:
        """Tuning-Reihenfolge: Outcome/Regression zuerst, dann Propensity, dann Pseudo-Effekt."""
        order = {
            "outcome": 0, "outcome_regression": 1,
            "grouped_outcome": 2, "grouped_outcome_regression": 3,
            "propensity": 4, "pseudo_effect": 5,
        }
        return order.get(task.objective_family, 99)

    def _downsample_indices(self, model_name: str, T: np.ndarray, n: int) -> np.ndarray:
        t = np.asarray(T).astype(int)
        n_t = int((t == 1).sum())
        n_c = int((t == 0).sum())
        min_group = max(1, min(n_t, n_c))
        name = model_name.lower()
        rng = np.random.RandomState(self.seed)

        def _sample(idx: np.ndarray, k: int) -> np.ndarray:
            k = min(int(k), len(idx))
            if k <= 0:
                return np.array([], dtype=int)
            return np.asarray(rng.choice(idx, size=k, replace=False), dtype=int)

        if name == "slearner":
            k = max(1000, int(n * self.cfg.learner_data_usage.s_learner_frac))
            return np.asarray(rng.choice(np.arange(n), size=min(n, k), replace=False), dtype=int)

        if name == "tlearner":
            k_group = max(1000, int(min_group * self.cfg.learner_data_usage.t_learner_group_frac))
            idx_t = np.where(t == 1)[0]
            idx_c = np.where(t == 0)[0]
            idx = np.concatenate([_sample(idx_t, k_group), _sample(idx_c, k_group)])
            rng.shuffle(idx)
            return idx.astype(int)

        if name == "xlearner":
            k_group = max(1000, int(min_group * self.cfg.learner_data_usage.x_learner_group_frac))
            idx_t = np.where(t == 1)[0]
            idx_c = np.where(t == 0)[0]
            idx = np.concatenate([_sample(idx_t, k_group), _sample(idx_c, k_group)])
            rng.shuffle(idx)
            return idx.astype(int)

        if name == "drlearner":
            k = max(1000, int(n * self.cfg.learner_data_usage.dr_learner_frac))
            return np.asarray(rng.choice(np.arange(n), size=min(n, k), replace=False), dtype=int)

        if name in {"nonparamdml", "paramdml", "causalforestdml"}:
            k = max(1000, int(n * self.cfg.learner_data_usage.dml_frac))
            return np.asarray(rng.choice(np.arange(n), size=min(n, k), replace=False), dtype=int)

        return np.arange(n, dtype=int)

    def _combined_indices_for_task(self, task: TuningTask, T: np.ndarray, n_rows: int) -> np.ndarray:
        idx_sets: List[np.ndarray] = []
        for model_name, _ in task.roles:
            idx_sets.append(self._downsample_indices(model_name, T=T, n=n_rows))
        if not idx_sets:
            return np.arange(n_rows, dtype=int)
        return np.unique(np.concatenate(idx_sets)).astype(int)

    def _prepare_task_frame(self, task: TuningTask, X: pd.DataFrame, T: np.ndarray, indices: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        X_sub = X.iloc[indices]  # View, keine Kopie — to_numpy() in _tune_task erstellt Kopie
        T_sub = np.asarray(T)[indices].astype(int)
        if task.uses_treatment_feature:
            X_sub = X_sub.copy()  # Kopie nur wenn wir mutieren müssen
            X_sub["__treatment__"] = T_sub
        return X_sub, indices.astype(int), T_sub

    def _score_classifier(self, y_true: np.ndarray, proba: np.ndarray, multiclass: bool = False) -> float:
        """Bewertet einen Classifier anhand der konfigurierten Tuning-Metrik.

        Unterstützte Metriken (tuning.metric):
        - log_loss (Default): Negierter Log-Loss (höher = besser, da direction=maximize).
          Bevorzugt für Nuisance-Modelle, da kalibrierte Wahrscheinlichkeiten
          direkt in die DML-Residualisierung einfließen.
        - roc_auc: ROC-AUC, bei Multiclass roc_auc_ovr (weighted)
        - accuracy: Accuracy auf Basis der Klasse mit höchster Wahrscheinlichkeit
        """
        metric = getattr(self.cfg.tuning, "metric", "log_loss")
        y_true = np.asarray(y_true).astype(int)
        if len(np.unique(y_true)) < 2:
            return 0.5 if metric == "roc_auc" else 0.0

        try:
            if metric == "log_loss":
                from sklearn.metrics import log_loss as _log_loss
                # Negiert, da Optuna direction=maximize (weniger Loss = besser)
                if proba.ndim == 1:
                    proba_2d = np.column_stack([1 - proba, proba])
                else:
                    proba_2d = proba
                return -float(_log_loss(y_true, proba_2d))

            if metric == "accuracy":
                from sklearn.metrics import accuracy_score
                if proba.ndim == 1:
                    y_pred = (proba >= 0.5).astype(int)
                else:
                    y_pred = np.argmax(proba, axis=1)
                return float(accuracy_score(y_true, y_pred))

            # Default: roc_auc
            if multiclass or len(np.unique(y_true)) > 2:
                return float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted"))
            return float(roc_auc_score(y_true, proba))
        except Exception:
            return 0.5 if metric == "roc_auc" else 0.0

    def _fit_model(self, params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, estimator_task: str):
        pl = self.cfg.constants.parallel_level
        if pl <= 1:
            pj = 1
        elif pl <= 2:
            pj = -1  # alle Kerne, Trials sequentiell
        else:
            # Level 3/4: Mehrere Trials parallel → Kerne aufteilen
            import os
            n_cpus = os.cpu_count() or 1
            n_trial_workers = self._tuning_n_jobs()
            pj = max(1, n_cpus // max(1, n_trial_workers))
        model = build_base_learner(self.cfg.base_learner.type, params, seed=self.seed, task=estimator_task, parallel_jobs=pj)
        model.fit(X_train, y_train)
        return model

    def _cv_splits(self, labels: np.ndarray, single_fold: bool = False):
        """Erzeugt CV-Splits. Bei single_fold=True wird nur der erste Fold verwendet."""
        splits = _iter_stratified_or_kfold(labels, n_splits=self.cfg.tuning.cv_splits, seed=self.seed)
        if single_fold:
            # Nur den ersten Split zurückgeben (schneller, etwas verrauschter)
            for tr, va in splits:
                yield tr, va
                return
        else:
            yield from splits

    def _objective_all_classification(self, params: Dict[str, Any], X_mat: np.ndarray, target: np.ndarray) -> float:
        scores: List[float] = []
        is_multiclass = len(np.unique(target)) > 2
        for tr, va in self._cv_splits(target.astype(int), single_fold=self.cfg.tuning.single_fold):
            model = self._fit_model(params, X_mat[tr], target[tr].astype(int), "classifier")
            if is_multiclass:
                proba = model.predict_proba(X_mat[va])
                scores.append(self._score_classifier(target[va], proba, multiclass=True))
            else:
                proba = model.predict_proba(X_mat[va])[:, 1]
                scores.append(self._score_classifier(target[va], proba))
        return float(np.mean(scores))

    def _objective_grouped_outcome(self, params: Dict[str, Any], X_mat: np.ndarray, Y: np.ndarray, T: np.ndarray) -> float:
        scores: List[float] = []
        K = len(np.unique(T))
        strat_labels = np.asarray(T).astype(int) * 10 + np.asarray(Y).astype(int)
        for tr, va in self._cv_splits(strat_labels, single_fold=self.cfg.tuning.single_fold):
            fold_scores: List[float] = []
            for group in range(K):
                tr_mask = T[tr] == group
                va_mask = T[va] == group
                if tr_mask.sum() < 2 or va_mask.sum() < 1:
                    continue
                y_train = Y[tr][tr_mask].astype(int)
                if len(np.unique(y_train)) < 2:
                    continue
                model = self._fit_model(params, X_mat[tr][tr_mask], y_train, "classifier")
                proba = model.predict_proba(X_mat[va][va_mask])[:, 1]
                fold_scores.append(self._score_classifier(Y[va][va_mask], proba))
            if fold_scores:
                scores.append(float(np.mean(fold_scores)))
        return float(np.mean(scores)) if scores else 0.5

    # ── Regression-Objectives (für Meta-Learner + DRLearner model_regression) ──

    def _score_regressor(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Neg. MSE (höher = besser, konsistent mit Optuna maximize)."""
        return -float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

    def _objective_all_regression(self, params: Dict[str, Any], X_mat: np.ndarray, target: np.ndarray) -> float:
        """Tuning-Objective für Regressoren auf dem Gesamt-Datensatz (SLearner, DRLearner model_regression)."""
        scores: List[float] = []
        for tr, va in self._cv_splits(target.astype(float), single_fold=self.cfg.tuning.single_fold):
            model = self._fit_model(params, X_mat[tr], target[tr].astype(float), "regressor")
            pred = model.predict(X_mat[va])
            scores.append(self._score_regressor(target[va], pred))
        return float(np.mean(scores)) if scores else -1.0

    def _objective_grouped_regression(self, params: Dict[str, Any], X_mat: np.ndarray, Y: np.ndarray, T: np.ndarray) -> float:
        """Tuning-Objective für Regressoren pro Treatment-Gruppe (TLearner, XLearner)."""
        scores: List[float] = []
        K = len(np.unique(T))
        strat_labels = np.asarray(T).astype(int) * 10 + np.clip(np.asarray(Y), 0, 1).astype(int)
        for tr, va in self._cv_splits(strat_labels, single_fold=self.cfg.tuning.single_fold):
            fold_scores: List[float] = []
            for group in range(K):
                tr_mask = T[tr] == group
                va_mask = T[va] == group
                if tr_mask.sum() < 2 or va_mask.sum() < 1:
                    continue
                y_train = Y[tr][tr_mask].astype(float)
                model = self._fit_model(params, X_mat[tr][tr_mask], y_train, "regressor")
                pred = model.predict(X_mat[va][va_mask])
                fold_scores.append(self._score_regressor(Y[va][va_mask], pred))
            if fold_scores:
                scores.append(float(np.mean(fold_scores)))
        return float(np.mean(scores)) if scores else -1.0

    def _build_xlearner_pseudo_outcomes(self, X_mat: np.ndarray, Y: np.ndarray, T: np.ndarray, nuisance_params: Dict[str, Any]) -> np.ndarray:
        mu0 = np.zeros(len(Y), dtype=float)
        mu1 = np.zeros(len(Y), dtype=float)
        filled = np.zeros(len(Y), dtype=bool)
        strat_labels = np.asarray(T).astype(int) * 10 + np.asarray(Y).astype(int)

        for tr, va in _iter_stratified_or_kfold(strat_labels, n_splits=self.cfg.tuning.cv_splits, seed=self.seed):
            control_train = tr[T[tr] == 0]
            treated_train = tr[T[tr] == 1]
            if len(control_train) < 2 or len(treated_train) < 2:
                continue

            control_y = Y[control_train].astype(int)
            treated_y = Y[treated_train].astype(int)
            if len(np.unique(control_y)) < 2 or len(np.unique(treated_y)) < 2:
                continue

            m0 = self._fit_model(nuisance_params, X_mat[control_train], control_y, "classifier")
            m1 = self._fit_model(nuisance_params, X_mat[treated_train], treated_y, "classifier")
            mu0[va] = m0.predict_proba(X_mat[va])[:, 1]
            mu1[va] = m1.predict_proba(X_mat[va])[:, 1]
            filled[va] = True

        if not filled.all():
            control_idx = np.where(T == 0)[0]
            treated_idx = np.where(T == 1)[0]
            control_y = Y[control_idx].astype(int)
            treated_y = Y[treated_idx].astype(int)
            if len(control_idx) >= 2 and len(np.unique(control_y)) >= 2:
                m0 = self._fit_model(nuisance_params, X_mat[control_idx], control_y, "classifier")
                mu0[~filled] = m0.predict_proba(X_mat[~filled])[:, 1]
            if len(treated_idx) >= 2 and len(np.unique(treated_y)) >= 2:
                m1 = self._fit_model(nuisance_params, X_mat[treated_idx], treated_y, "classifier")
                mu1[~filled] = m1.predict_proba(X_mat[~filled])[:, 1]

        return np.where(T == 1, Y - mu0, mu1 - Y).astype(float)

    def _objective_xlearner_cate(self, params: Dict[str, Any], X_mat: np.ndarray, Y: np.ndarray, T: np.ndarray, nuisance_params: Dict[str, Any]) -> float:
        pseudo = self._build_xlearner_pseudo_outcomes(X_mat, Y, T, nuisance_params=nuisance_params)
        scores: List[float] = []
        strat_labels = np.asarray(T).astype(int) * 10 + np.asarray(Y).astype(int)

        for tr, va in self._cv_splits(strat_labels, single_fold=self.cfg.tuning.single_fold):
            fold_scores: List[float] = []
            for group in (0, 1):
                tr_mask = T[tr] == group
                va_mask = T[va] == group
                if tr_mask.sum() < 2 or va_mask.sum() < 1:
                    continue
                model = self._fit_model(params, X_mat[tr][tr_mask], pseudo[tr][tr_mask], "regressor")
                pred = np.asarray(model.predict(X_mat[va][va_mask]), dtype=float)
                fold_scores.append(-float(np.mean((pseudo[va][va_mask] - pred) ** 2)))
            if fold_scores:
                scores.append(float(np.mean(fold_scores)))
        return float(np.mean(scores)) if scores else -1e12

    def _tune_task(self, task: TuningTask, X: pd.DataFrame, Y: np.ndarray, T: np.ndarray, shared_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        indices = self._combined_indices_for_task(task, T=T, n_rows=len(X))
        X_task, row_indices, T_task = self._prepare_task_frame(task, X=X, T=T, indices=indices)

        if task.target_name == "Y":
            target = np.asarray(Y)[row_indices]
        elif task.target_name == "T":
            target = np.asarray(T)[row_indices].astype(int)
        else:
            target = np.asarray(Y)[row_indices]

        max_rows = self.cfg.tuning.max_tuning_rows
        if max_rows is not None and len(X_task) > int(max_rows):
            rng = np.random.RandomState(self.seed)
            keep = np.asarray(rng.choice(np.arange(len(X_task)), size=int(max_rows), replace=False), dtype=int)
            X_task = X_task.iloc[keep]
            target = target[keep]
            row_indices = row_indices[keep]
            T_task = T_task[keep]
        X_mat = X_task.to_numpy()
        fixed_defaults = dict(self.cfg.base_learner.fixed_params or {})

        import logging
        _tlog = logging.getLogger("rubin.tuning")
        _tlog.info(
            "Tuning-Task '%s': X_input=%d rows, indices=%d, X_task=%s, target=%s (unique=%s), "
            "T_task unique=%s, cv_splits=%d, target_name=%s, objective=%s",
            task.key, len(X), len(indices), X_mat.shape, target.shape,
            np.unique(target).tolist(), np.unique(T_task).tolist(),
            self.cfg.tuning.cv_splits, task.target_name, task.objective_family,
        )

        study = self._create_study(task.key)

        def objective(trial):
            params = _suggest_params(trial, self.cfg.base_learner.type, self.cfg.tuning.search_space)
            params = {**fixed_defaults, **params}
            # Classifier-Objectives (Nuisance-Modelle: model_y, model_t, propensity)
            if task.objective_family in {"outcome", "propensity"}:
                return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int))
            if task.objective_family == "grouped_outcome":
                return self._objective_grouped_outcome(params, X_mat=X_mat, Y=target.astype(int), T=T_task.astype(int))
            # Regressor-Objectives (Meta-Learner outcome, DRLearner model_regression)
            if task.objective_family == "outcome_regression":
                return self._objective_all_regression(params, X_mat=X_mat, target=target.astype(float))
            if task.objective_family == "grouped_outcome_regression":
                return self._objective_grouped_regression(params, X_mat=X_mat, Y=target.astype(float), T=T_task.astype(int))
            # XLearner CATE-Modelle
            if task.objective_family == "pseudo_effect":
                nuisance = dict(shared_params.get("xlearner__models") or fixed_defaults)
                return self._objective_xlearner_cate(
                    params,
                    X_mat=X_mat,
                    Y=np.asarray(Y)[row_indices].astype(int),
                    T=T_task.astype(int),
                    nuisance_params=nuisance,
                )
            raise ValueError(f"Unbekannte objective_family={task.objective_family!r}")

        study.optimize(objective, n_trials=int(self.cfg.tuning.n_trials), timeout=self.cfg.tuning.timeout_seconds,
                       n_jobs=self._tuning_n_jobs())
        try:
            self.best_scores[task.key] = float(study.best_value)
        except Exception:
            pass
        return {**fixed_defaults, **dict(study.best_params)}

    def _tuning_n_jobs(self) -> int:
        """Anzahl paralleler Optuna-Trials basierend auf parallel_level und base_learner.

        Level 1-2: 1 (sequentiell, alle Kerne an den einzelnen Fit)
        Level 3:   moderate Parallelisierung
        Level 4:   max. parallele Trials

        CatBoost vs LightGBM: CatBoost's Symmetric-Tree-Algorithmus skaliert
        schlechter mit wenigen Threads pro Fit. Deshalb werden bei CatBoost
        WENIGER parallele Trials gestartet, dafür mit MEHR Threads pro Fit.
        Beispiel 16 Kerne Level 4: LightGBM 8 Trials × 2 Kerne = 16,
        CatBoost 4 Trials × 4 Kerne = 16 (gleiche CPU-Auslastung, aber
        CatBoost nutzt 4 Threads pro Fit effizienter als 2).
        """
        pl = self.cfg.constants.parallel_level
        if pl <= 2:
            return 1
        import os
        n_cpus = os.cpu_count() or 1
        is_catboost = (self.cfg.base_learner.type or "").lower() == "catboost"
        if pl >= 4:
            # Level 4: CatBoost braucht min. 4 Kerne/Fit, LightGBM reicht 2
            min_cores_per_fit = 4 if is_catboost else 2
            return max(1, n_cpus // min_cores_per_fit)
        # Level 3: moderate Parallelisierung
        return max(1, n_cpus // 4)

    def tune_all(self, model_names: List[str], X: pd.DataFrame, Y: np.ndarray, T: np.ndarray) -> Dict[str, Dict[str, Dict[str, Any]]]:
        if not self.cfg.tuning.enabled:
            return {}

        import logging
        _tlog = logging.getLogger("rubin.tuning")
        _tlog.info(
            "tune_all gestartet: models=%s, X=%s, Y=%s (unique=%s), T=%s (unique=%s), "
            "cv_splits=%d, n_trials=%d, parallel_trials=%d",
            model_names, X.shape, np.asarray(Y).shape, np.unique(Y).tolist(),
            np.asarray(T).shape, np.unique(T).tolist(),
            self.cfg.tuning.cv_splits, self.cfg.tuning.n_trials,
            self._tuning_n_jobs(),
        )

        if any((m or "").lower() == "causalforestdml" for m in model_names):
            # Hinweis: CausalForestDML-Waldparameter werden über EconML tune() bestimmt,
            # nicht über Optuna. Optuna optimiert nur die Nuisance-Modelle (model_y, model_t).
            import logging
            logging.getLogger("rubin.tuning").debug(
                "CausalForestDML erkannt: Wald-Parameter werden über EconML tune() bestimmt, "
                "nicht über Optuna. Optuna optimiert nur die Nuisance-Modelle."
            )

        plan = self._build_plan(model_names)
        tuned_by_task: Dict[str, Dict[str, Any]] = {}

        for task in sorted(plan.values(), key=self._task_priority):
            best = self._tune_task(task, X=X, Y=Y, T=T, shared_params=tuned_by_task)
            tuned_by_task[task.key] = best
            if task.objective_family == "grouped_outcome_regression" and any(m.lower() == "xlearner" for m, _ in task.roles):
                tuned_by_task.setdefault("xlearner__models", best)

        tuned_by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for task in plan.values():
            best = tuned_by_task.get(task.key, {})
            for model_name, role in task.roles:
                tuned_by_model.setdefault(model_name, {})
                tuned_by_model[model_name][role] = dict(best)

        for model_name, roles in tuned_by_model.items():
            if roles:
                # HINWEIS: "default" wird als Fallback für Rollen genutzt, die NICHT
                # explizit getunt wurden. Das betrifft v.a. model_final (CATE-Regression).
                # model_registry._base() verhindert, dass model_final diese Classifier-
                # Defaults erbt — model_final nutzt nur base_fixed_params oder explizit
                # getunte model_final-Params (via FinalModelTuning).
                roles.setdefault("default", next(iter(roles.values())))
        return tuned_by_model

    def tune(self, model_name: str, X: pd.DataFrame, Y: np.ndarray, T: np.ndarray) -> TunedSet:
        tuned = self.tune_all([model_name], X=X, Y=Y, T=T)
        return TunedSet(role_params=dict(tuned.get(model_name, {}) or {}))


class FinalModelTuner:
    """Tuning des Final-Modells über R-Loss/R-Score-Logik.
Unterstützte Fälle (aktuell):
- NonParamDML: Final-Modell wird über den EconML-RScorer bewertet.
- DRLearner: Final-Modell wird über die eingebaute score(...) Funktion bewertet.
Warum zwei Varianten?
- RScorer ist explizit als Modellselektions-Scorer für R-Loss konzipiert und
passt sehr gut zur DML-Familie.
- Für DRLearner ist die direkte score(...) Methode der natürlichere Weg, weil
die Nuisance-Struktur (Regression mit T) von RScorer abweicht. In der EconML-
Dokumentation wird score(...) genau für diese Art Out-of-Sample-Bewertung
empfohlen.
Locking-Regel:
Das Tuning wird auf der Trainingsmenge des ersten Cross-Prediction-Folds
durchgeführt. Danach werden die gefundenen Parameter als ctx.tuned_params
persistiert und in weiteren Folds wiederverwendet."""

    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg
        self.seed = int(cfg.constants.random_seed)
        self.optuna = _safe_import_optuna() if cfg.final_model_tuning.enabled else None
        self.best_scores: Dict[str, float] = {}

    def _tuning_n_jobs(self) -> int:
        """Anzahl paralleler Optuna-Trials basierend auf parallel_level und base_learner."""
        pl = self.cfg.constants.parallel_level
        if pl <= 2:
            return 1
        import os
        n_cpus = os.cpu_count() or 1
        is_catboost = (self.cfg.base_learner.type or "").lower() == "catboost"
        if pl >= 4:
            min_cores_per_fit = 4 if is_catboost else 2
            return max(1, n_cpus // min_cores_per_fit)
        return max(1, n_cpus // 4)

    def _parallel_jobs_per_fit(self) -> int:
        """Kerne pro Base-Learner-Fit, berücksichtigt parallele Trials."""
        pl = self.cfg.constants.parallel_level
        if pl <= 1:
            return 1
        if pl <= 2:
            return -1
        import os
        n_cpus = os.cpu_count() or 1
        n_trial_workers = self._tuning_n_jobs()
        return max(1, n_cpus // max(1, n_trial_workers))

    def _create_study(self, study_key: str):
        optuna = self.optuna
        if optuna is None:
            raise RuntimeError("Optuna ist nicht verfügbar. Bitte optuna installieren.")
        import hashlib
        base_seed = int(self.cfg.tuning.optuna_seed)
        key_hash = int(hashlib.sha256(study_key.encode()).hexdigest(), 16) % (2**31)
        study_seed = (base_seed + key_hash) % (2**31)
        try:
            sampler = optuna.samplers.TPESampler(seed=study_seed)
        except Exception:
            sampler = None
        return optuna.create_study(direction="maximize", sampler=sampler)

    def _build_classifier(self, base_type: str, fixed: Dict[str, Any], tuned: Dict[str, Any]) -> Any:
        params = dict(fixed or {})
        params.update(tuned or {})
        return build_base_learner(base_type, params, seed=self.seed, task="classifier", parallel_jobs=self._parallel_jobs_per_fit())

    def _build_regressor(self, base_type: str, fixed: Dict[str, Any], tuned: Dict[str, Any]) -> Any:
        params = dict(fixed or {})
        params.update(tuned or {})
        return build_base_learner(base_type, params, seed=self.seed, task="regressor", parallel_jobs=self._parallel_jobs_per_fit())

    def tune_final_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        Y: np.ndarray,
        T: np.ndarray,
        base_type: str,
        base_fixed_params: Dict[str, Any],
        tuned_roles: Dict[str, Dict[str, Any]],
        crosspred_splits: int,
        fmt_fixed_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Ermittelt getunte Parameter für die Rolle 'model_final'.

        base_fixed_params: Für Nuisance-Modelle (model_y, model_t, model_propensity).
        fmt_fixed_params: Für model_final (final_model_tuning.fixed_params). Falls None → {}."""
        if fmt_fixed_params is None:
            fmt_fixed_params = {}

        if not self.cfg.final_model_tuning.enabled:
            return {}

        name = (model_name or "").lower()
        if name not in {"nonparamdml", "drlearner"}:
            return {}

        # Locking: Falls model_final bereits vorhanden ist, nichts tun.
        if (tuned_roles.get("model_final") or {}) != {}:
            return {}

        optuna = self.optuna
        if optuna is None:
            return {}

        tr_idx = _first_crossfit_train_indices(len(X), T, n_splits=int(crosspred_splits), seed=self.seed)
        X_tune = X.iloc[tr_idx]
        Y_tune = Y[tr_idx]
        T_tune = T[tr_idx]

        max_rows = self.cfg.final_model_tuning.max_tuning_rows
        if max_rows is not None and len(X_tune) > int(max_rows):
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(np.arange(len(X_tune)), size=int(max_rows), replace=False)
            X_tune = X_tune.iloc[idx]
            Y_tune = Y_tune[idx]
            T_tune = T_tune[idx]

        role = "model_final"
        base_type = (base_type or "lgbm").lower()

        if name == "nonparamdml":
            from econml.dml import NonParamDML
            from econml.score import RScorer

            # RScorer nutzt eine R^2-ähnliche Kennzahl für das R-Loss.
            scorer = RScorer(
                model_y=self._build_classifier(base_type, base_fixed_params, tuned_roles.get("model_y", {})),
                model_t=self._build_classifier(base_type, base_fixed_params, tuned_roles.get("model_t", {})),
                discrete_treatment=True,
                discrete_outcome=True,
                cv=int(self.cfg.final_model_tuning.cv_splits),
                random_state=self.seed,
            )
            scorer.fit(Y_tune, T_tune, X=X_tune)

            study = self._create_study(f"final__{model_name}__{base_type}")
            stability_penalty = float(getattr(self.cfg.final_model_tuning, "stability_penalty", 0.0))

            def objective(trial):
                cand_params = _suggest_params(trial, base_type, self.cfg.final_model_tuning.search_space, is_fmt=True)
                # LGBM: subsample_freq=1 erzwingt Bagging in jedem Boosting-Schritt (analog causaluka)
                if base_type == "lgbm":
                    cand_params.setdefault("subsample_freq", 1)
                est = NonParamDML(
                    model_y=self._build_classifier(base_type, base_fixed_params, tuned_roles.get("model_y", {})),
                    model_t=self._build_classifier(base_type, base_fixed_params, tuned_roles.get("model_t", {})),
                    model_final=self._build_regressor(base_type, fmt_fixed_params, cand_params),
                    discrete_treatment=True,
                    discrete_outcome=True,
                    random_state=self.seed,
                )
                est.fit(Y_tune, T_tune, X=X_tune)
                r_score = float(scorer.score(est))
                trial.set_user_attr("r_score_raw", r_score)

                if stability_penalty > 0:
                    cate = est.effect(X_tune).ravel()
                    cv = float(np.std(cate)) / (abs(float(np.median(cate))) + 1e-8)
                    penalty = float(np.log1p(cv))
                    trial.set_user_attr("stability_cv", cv)
                    trial.set_user_attr("stability_penalty_value", stability_penalty * penalty)
                    return r_score - stability_penalty * penalty
                return r_score

            study.optimize(objective, n_trials=int(self.cfg.final_model_tuning.n_trials),
                           timeout=self.cfg.final_model_tuning.timeout_seconds,
                           n_jobs=self._tuning_n_jobs())
            try:
                raw = study.best_trial.user_attrs.get("r_score_raw", study.best_value)
                self.best_scores[f"final__{model_name}"] = raw
                self.best_scores[f"final__{model_name}__penalized"] = float(study.best_value)
            except Exception:
                pass
            return {role: dict(study.best_trial.params)}

        if name == "drlearner":
            from econml.dr import DRLearner

            study = self._create_study(f"final__{model_name}__{base_type}")
            stability_penalty = float(getattr(self.cfg.final_model_tuning, "stability_penalty", 0.0))

            def objective(trial):
                cand_params = _suggest_params(trial, base_type, self.cfg.final_model_tuning.search_space, is_fmt=True)
                # LGBM: subsample_freq=1 erzwingt Bagging in jedem Boosting-Schritt (analog causaluka)
                if base_type == "lgbm":
                    cand_params.setdefault("subsample_freq", 1)
                fold_scores_raw = []
                fold_scores_penalized = []
                split_iter = _iter_stratified_or_kfold(
                    labels=T_tune.astype(int),
                    n_splits=int(self.cfg.final_model_tuning.cv_splits),
                    seed=self.seed,
                )
                for tr, va in split_iter:
                    est = DRLearner(
                        model_propensity=self._build_classifier(base_type, base_fixed_params, tuned_roles.get("model_propensity", {})),
                        model_regression=self._build_regressor(base_type, base_fixed_params, tuned_roles.get("model_regression", {})),
                        model_final=self._build_regressor(base_type, fmt_fixed_params, cand_params),
                        cv=int(self.cfg.final_model_tuning.cv_splits),
                        random_state=self.seed,
                    )
                    est.fit(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr])
                    r2 = float(est.score(Y_tune[va], T_tune[va], X=X_tune.iloc[va]))
                    fold_scores_raw.append(r2)

                    penalized = r2
                    if stability_penalty > 0:
                        cate = est.effect(X_tune.iloc[va]).ravel()
                        cv = float(np.std(cate)) / (abs(float(np.median(cate))) + 1e-8)
                        penalized = r2 - stability_penalty * float(np.log1p(cv))
                    fold_scores_penalized.append(penalized)

                    if self.cfg.final_model_tuning.single_fold:
                        break
                trial.set_user_attr("r_score_raw", float(np.mean(fold_scores_raw)))
                return float(np.mean(fold_scores_penalized))

            study.optimize(objective, n_trials=int(self.cfg.final_model_tuning.n_trials),
                           timeout=self.cfg.final_model_tuning.timeout_seconds,
                           n_jobs=self._tuning_n_jobs())
            try:
                raw = study.best_trial.user_attrs.get("r_score_raw", study.best_value)
                self.best_scores[f"final__{model_name}"] = raw
                self.best_scores[f"final__{model_name}__penalized"] = float(study.best_value)
            except Exception:
                pass
            return {role: dict(study.best_trial.params)}

        return {}
