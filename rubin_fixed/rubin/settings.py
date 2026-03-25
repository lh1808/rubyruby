from __future__ import annotations

"""Konfigurationsmodell und YAML-Loader."""

from typing import Any, Dict, List, Optional, Literal, Union
from pathlib import Path
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Basis: strikte Modelle
# ---------------------------------------------------------------------------

_STRICT = ConfigDict(extra="forbid")


SUPPORTED_MODEL_NAMES = {
    "SLearner",
    "TLearner",
    "XLearner",
    "DRLearner",
    "NonParamDML",
    "ParamDML",
    "CausalForestDML",
}

# Modelle, die Multi-Treatment *nicht* unterstützen.
_BT_ONLY_MODELS = {"SLearner", "TLearner", "XLearner"}


class TreatmentConfig(BaseModel):
    model_config = _STRICT

    type: Literal["binary", "multi"] = "binary"
    # Welche Gruppe als Control/Baseline dient. Aktuell wird nur reference_group=0
    # unterstützt. Eine Erweiterung auf andere Baseline-Gruppen ist vorbereitet,
    # erfordert aber Anpassungen in den Metrik-Funktionen.
    reference_group: int = 0



class MLflowConfig(BaseModel):
    model_config = _STRICT

    experiment_name: str = "rubin"


class DataFilesConfig(BaseModel):
    model_config = _STRICT

    x_file: str
    t_file: str
    y_file: str
    dtypes_file: Optional[str] = None
    s_file: Optional[str] = None

    # Externe Evaluationsdaten (optional, für validate_on="external").
    # Wenn gesetzt, wird auf diesen Daten evaluiert statt auf einem Split
    # der Trainingsdaten. Die Trainingsdaten (x/t/y_file) werden vollständig
    # zum Training verwendet. Mehrere Trainingsdateien können vorab per
    # DataPrep zusammengeführt werden.
    eval_x_file: Optional[str] = None
    eval_t_file: Optional[str] = None
    eval_y_file: Optional[str] = None
    eval_s_file: Optional[str] = None


class HistoricalScoreConfig(BaseModel):
    model_config = _STRICT

    """Konfiguration für einen historischen Score (Vergleichsbasis)."""

    name: str = "historical_score"
    column: str = "S"
    higher_is_better: bool = True


class DataPrepConfig(BaseModel):
    model_config = _STRICT

    # I/O
    # Hinweis: Es werden häufig mehrere Rohdateien (CSV/SAS) verarbeitet.
    # Daher ist `data_path` eine Liste. Für einfache Fälle genügt eine Datei.
    data_path: List[str]

    # Optionaler separater Evaluationsdatensatz. Wenn gesetzt, wird der
    # Preprocessor auf data_path (Train) gefittet und nur transformiert auf
    # eval_data_path angewendet. Verhindert Data-Leakage bei externer Validierung.
    # Ausgabe: X_eval.parquet, T_eval.parquet, Y_eval.parquet im output_path.
    eval_data_path: Optional[List[str]] = None

    # Feature-/Info-Dateien sind projektspezifisch. In der Praxis liegen sie oft
    # als Excel-Dateien vor. Die Pipeline wirft in `DataPrepPipeline` eine klare
    # Fehlermeldung, wenn zwingende Pfade fehlen.
    feature_path: Optional[str] = None
    info_path: Optional[str] = None
    output_path: str

    # CSV/SAS
    delimiter: str = ","
    chunksize: Optional[int] = None
    sas_encoding: str = "utf-8"

    # Zielvariablen
    target: str = "Y"
    treatment: str = "T"
    score_name: Optional[str] = "S"
    score_as_feature: bool = False

    # Replacement/Mapping
    target_replacement: Optional[Dict[str, Any]] = None
    treatment_replacement: Optional[Dict[str, Any]] = None

    # Mehrdatei-Logik
    # - "merge": Dateien werden untereinander gehängt.
    # - "treatment_only": es werden nur Treatment-Zeilen aus allen Dateien genutzt;
    #   aus der Control-Datei (Index: control_file_index) wird zusätzlich eine
    #   „Control-Kopie“ erzeugt.
    multiple_files_option: Literal["merge", "treatment_only"] = "merge"
    control_file_index: int = 0

    # Sonstiges
    binary_target: bool = True
    fill_na_method: Optional[Literal["zero", "median", "mean", "mode"]] = None

    # Deduplizierung: Wenn Kunden mehrfach im Datensatz vorkommen, kann hier auf
    # einen Eintrag pro Kunde reduziert werden. Die Spalte (z. B. PartnerID) muss
    # im Rohdatensatz vorhanden sein, wird aber NICHT als Feature übernommen.
    # Die Deduplizierung geschieht direkt nach dem Einlesen, bevor auf Features
    # reduziert wird.
    deduplicate: bool = False
    deduplicate_id_column: Optional[str] = None

    # MLflow (DataPrep kann optional loggen)
    log_to_mlflow: bool = False
    mlflow_experiment_name: Optional[str] = None


class DataProcessingConfig(BaseModel):
    model_config = _STRICT

    # Memory-Reduktion: Datentypen downcasten (float64→float32, int64→int32 etc.).
    # Wird in der Analyse-Pipeline nach dem Laden der Daten angewendet.
    # In der DataPrep-Pipeline wird reduce_mem_usage() immer aufgerufen.
    reduce_memory: bool = True

    # Optionales Downsampling in der Analyse.
    # Wird als Anteil (0..1] interpretiert.
    df_frac: Optional[float] = None

    # Holdout-Anteil für den Validierungsmodus "holdout".
    test_size: float = 0.0
    # Validierungsmodus:
    # - "cross": Cross-Predictions (Out-of-Fold) auf dem gleichen Datensatz
    # - "holdout": Stratifizierter Split des gleichen Datensatzes
    # - "external": Training auf data_files, Evaluation auf separaten eval_*-Dateien
    validate_on: Literal["cross", "holdout", "external"] = "cross"

    # Anzahl der Splits für Cross-Predictions (Out-of-Fold-Vorhersagen).
    cross_validation_splits: int = 5


class FeatureSelectionConfig(BaseModel):
    model_config = _STRICT

    enabled: bool = False

    # Methoden zur Importance-Berechnung. Mehrere können kombiniert werden;
    # die Ergebnisse werden per Union zusammengeführt.
    # - "lgbm_importance": LightGBM auf Outcome (Y), Gain-Importance
    # - "lgbm_permutation": LightGBM auf Outcome (Y), Permutation-Importance (robuster, langsamer)
    # - "causal_forest": CausalForestDML Feature-Importances (kausale Relevanz)
    # - "none": keine Importance-Filterung
    methods: List[Literal["lgbm_importance", "lgbm_permutation", "causal_forest", "none"]] = Field(
        default_factory=lambda: ["lgbm_importance"]
    )

    # Prozent der Features, die pro Methode behalten werden.
    # Bei Union: aus jeder Methode werden die Top-X% behalten, dann vereinigt.
    # Beispiel: 15.0 bei 100 Features → 15 Features pro Methode, Union aller.
    top_pct: float = 15.0

    # Absolute Obergrenze für Features (nach Union).
    # None = keine Begrenzung.
    max_features: Optional[int] = None

    # Korrelation: ab welchem Betrag (|corr|) eine Spalte als redundant gilt.
    correlation_threshold: float = 0.9


class ModelsConfig(BaseModel):
    model_config = _STRICT

    models_to_train: List[str] = Field(default_factory=list)


class BaseLearnerConfig(BaseModel):
    model_config = _STRICT

    type: Literal["lgbm", "catboost"] = "lgbm"
    fixed_params: Dict[str, Any] = Field(default_factory=dict)


class CausalForestConfig(BaseModel):
    model_config = _STRICT

    forest_fixed_params: Dict[str, Any] = Field(default_factory=dict)
    use_econml_tune: bool = False
    econml_tune_params: Any = "auto"
    tune_max_rows: Optional[int] = None


class SearchSpaceParameterConfig(BaseModel):
    model_config = _STRICT

    type: Literal["int", "float", "categorical"]
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    step: Optional[float] = None
    choices: Optional[List[Any]] = None

    @model_validator(mode="after")
    def validate_definition(self) -> "SearchSpaceParameterConfig":
        if self.type == "categorical":
            if not self.choices:
                raise ValueError("Für type='categorical' muss choices gesetzt sein.")
            return self

        if self.low is None or self.high is None:
            raise ValueError("Für numerische Parameter müssen low und high gesetzt sein.")
        if self.high < self.low:
            raise ValueError("high muss größer oder gleich low sein.")
        if self.step is not None and self.step <= 0:
            raise ValueError("step muss > 0 sein.")
        return self


class SearchSpaceConfig(BaseModel):
    model_config = _STRICT

    lgbm: Dict[str, SearchSpaceParameterConfig] = Field(default_factory=dict)
    catboost: Dict[str, SearchSpaceParameterConfig] = Field(default_factory=dict)


class OptunaTuningConfig(BaseModel):
    model_config = _STRICT

    enabled: bool = False
    n_trials: int = 30
    timeout_seconds: Optional[int] = None
    cv_splits: int = 3
    single_fold: bool = False
    metric: str = "log_loss"
    per_learner: bool = False
    per_role: bool = False
    max_tuning_rows: Optional[int] = None
    storage_path: Optional[str] = None
    study_name_prefix: str = "baselearner"
    reuse_study_if_exists: bool = True
    optuna_seed: int = 42
    search_space: SearchSpaceConfig = Field(default_factory=SearchSpaceConfig)


class FinalModelTuningConfig(BaseModel):
    model_config = _STRICT

    enabled: bool = False
    n_trials: int = 30
    timeout_seconds: Optional[int] = None
    cv_splits: int = 3
    max_tuning_rows: Optional[int] = None
    method: Literal["rscorer"] = "rscorer"
    models: Optional[List[str]] = None
    single_fold: bool = False
    stability_penalty: float = Field(0.3, ge=0.0, le=2.0)
    fixed_params: Dict[str, Any] = Field(default_factory=dict)
    search_space: SearchSpaceConfig = Field(default_factory=SearchSpaceConfig)


class LearnerDataUsageConfig(BaseModel):
    model_config = _STRICT

    s_learner_frac: float = 1.0
    t_learner_group_frac: float = 1.0
    x_learner_group_frac: float = 1.0
    dml_frac: float = 1.0
    dr_learner_frac: float = 1.0


class ShapConfig(BaseModel):
    model_config = _STRICT

    # Deprecated: Diese Felder werden aktuell nicht ausgewertet. Explainability
    # wird über den separaten Runner `run_explain.py` gesteuert.
    # Die Felder bleiben für Rückwärtskompatibilität mit bestehenden Configs erhalten.
    calculate_shap_values: bool = False
    shap_calculation_models: List[str] = Field(default_factory=list)

    # Aktiv genutzt von run_explain.py (als Voreinstellungen aus config_snapshot.yml):
    n_shap_values: int = 10_000
    top_n_features: int = 20
    num_bins: int = 10


class SegmentAnalysisConfig(BaseModel):
    model_config = _STRICT

    enabled: bool = True
    quantiles: int = 10
    top_n_features: int = 8
    max_bins: int = 6
    max_categories: int = 15


class OptionalOutputConfig(BaseModel):
    model_config = _STRICT

    output_dir: Optional[str] = None
    save_predictions: bool = False
    predictions_format: Literal["csv", "parquet"] = "csv"

    # Schutz gegen extrem große Artefakte (z. B. Millionen Zeilen).
    max_prediction_rows: Optional[int] = None




class BundleConfig(BaseModel):
    model_config = _STRICT

    enabled: bool = False
    base_dir: str = "bundles"
    bundle_id: Optional[str] = None
    include_challengers: bool = True
    log_to_mlflow: bool = True


class SelectionConfig(BaseModel):
    model_config = _STRICT

    metric: str = "qini"
    higher_is_better: bool = True
    refit_champion_on_full_data: bool = True
    manual_champion: Optional[str] = None


class SurrogateTreeConfig(BaseModel):
    model_config = _STRICT

    # Aktiviert den Surrogate-Einzelbaum, der das Champion-Modell nachlernt.
    # Der Baum wird mit den gleichen Features trainiert und lernt die
    # CATE-Vorhersagen des Champions als Regressionsziel.
    # Es wird ein einzelner Baum des konfigurierten Base-Learners (lgbm/catboost)
    # verwendet (n_estimators=1), was dank leaf-wise Growth (LightGBM) bzw.
    # symmetrischem Splitting (CatBoost) bessere Bäume als CART liefert.
    enabled: bool = False

    # Mindestanzahl Samples pro Blatt. Stellt sicher, dass jedes Blatt
    # statistisch belastbar ist.
    # Wird auf min_child_samples (LightGBM) bzw. min_data_in_leaf (CatBoost)
    # gemappt.
    min_samples_leaf: int = 50

    # Maximale Anzahl Blätter (nur LightGBM, leaf-wise Growth).
    # Steuert die Baumkomplexität direkt. Bei CatBoost wird stattdessen
    # max_depth verwendet.
    num_leaves: int = 31

    # Maximale Baumtiefe. None = keine Begrenzung bei LightGBM (-1),
    # bei CatBoost wird 6 als Default verwendet.
    max_depth: Optional[int] = None


class ConstantsConfig(BaseModel):
    # In älteren Konfigurationen war der Schlüssel oft "SEED".
    # Wir unterstützen beides. Intern wird konsequent `random_seed` verwendet.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    random_seed: int = Field(42, alias="SEED")

    # Parallelisierungs-Level:
    #   1 = Minimal:  Base Learner nutzen 1 Kern, Folds sequentiell.
    #                 Minimaler RAM-Verbrauch, sicher auf jeder Maschine.
    #   2 = Moderat:  Base Learner nutzen alle Kerne, Folds sequentiell.
    #                 Standard — guter Kompromiss aus Speed und RAM.
    #   3 = Hoch:     Base Learner nutzen alle Kerne, Folds auto-parallel
    #                 (2–4 gleichzeitig, abhängig von Kernzahl).
    #   4 = Maximum:  Base Learner nutzen alle Kerne, alle Folds parallel.
    #                 Höchster RAM-Verbrauch, aber schnellste Laufzeit.
    parallel_level: int = Field(2, ge=1, le=4)

    @property
    def SEED(self) -> int:  # pragma: no cover
        """Kompatibilitäts-Property (z. B. für ältere Codepfade)."""
        return int(self.random_seed)


class AnalysisConfig(BaseModel):
    model_config = _STRICT

    source_config_path: Optional[str] = None

    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    constants: ConstantsConfig = Field(default_factory=ConstantsConfig)
    data_files: DataFilesConfig
    historical_score: HistoricalScoreConfig = Field(default_factory=HistoricalScoreConfig)
    data_prep: Optional[DataPrepConfig] = None

    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    treatment: TreatmentConfig = Field(default_factory=TreatmentConfig)
    feature_selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    base_learner: BaseLearnerConfig = Field(default_factory=BaseLearnerConfig)
    causal_forest: CausalForestConfig = Field(default_factory=CausalForestConfig)
    tuning: OptunaTuningConfig = Field(default_factory=OptunaTuningConfig)
    final_model_tuning: FinalModelTuningConfig = Field(default_factory=FinalModelTuningConfig)
    learner_data_usage: LearnerDataUsageConfig = Field(default_factory=LearnerDataUsageConfig)
    shap_values: ShapConfig = Field(default_factory=ShapConfig)
    segment_analysis: SegmentAnalysisConfig = Field(default_factory=SegmentAnalysisConfig)
    optional_output: OptionalOutputConfig = Field(default_factory=OptionalOutputConfig)
    bundle: BundleConfig = Field(default_factory=BundleConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    surrogate_tree: SurrogateTreeConfig = Field(default_factory=SurrogateTreeConfig)



    @model_validator(mode="after")
    def validate_models_and_manual_champion(self) -> "AnalysisConfig":
        available = list(self.models.models_to_train or [])
        invalid = [name for name in available if name not in SUPPORTED_MODEL_NAMES]
        if invalid:
            raise ValueError(
                "models.models_to_train enthält unbekannte Modelle. "
                f"Erhalten: {invalid}. Erlaubt: {sorted(SUPPORTED_MODEL_NAMES)}"
            )

        # Multi-Treatment: Modelle blockieren, die MT nicht unterstützen.
        if self.treatment.type == "multi":
            if self.treatment.reference_group != 0:
                raise ValueError(
                    f"treatment.reference_group={self.treatment.reference_group}: "
                    f"Aktuell wird nur reference_group=0 unterstützt."
                )
            bt_only = [name for name in available if name in _BT_ONLY_MODELS]
            if bt_only:
                raise ValueError(
                    f"treatment.type='multi' ist nicht kompatibel mit: {bt_only}. "
                    f"Diese Modelle unterstützen nur Binary Treatment. "
                    f"Bitte entfernen oder treatment.type='binary' setzen."
                )
            # Hinweis: Bei MT gibt es keine einfache 'qini'-Metrik mehr.
            # Stattdessen: 'policy_value', 'qini_T1', 'qini_T2',
            # 'policy_value_treat_positive_T1', etc.
            _bt_only_metrics = {"qini", "auuc", "uplift_at_10pct", "uplift_at_20pct",
                                "uplift_at_50pct", "policy_value_treat_positive"}
            if self.selection.metric in _bt_only_metrics:
                raise ValueError(
                    f"treatment.type='multi' mit selection.metric='{self.selection.metric}': "
                    f"Diese Metrik existiert bei Multi-Treatment nicht und würde zum "
                    f"Fallback auf das erste Modell führen. "
                    f"Empfohlen: 'policy_value', 'policy_value_treat_positive_T1', "
                    f"'qini_T1', 'qini_T2', etc."
                )

        manual = (self.selection.manual_champion or "").strip()
        if not manual:
            self.selection.manual_champion = None
            return self
        if manual not in available:
            raise ValueError(
                "selection.manual_champion muss in models.models_to_train enthalten sein. "
                f"Erhalten: {manual!r}. Verfügbar: {available}"
            )
        self.selection.manual_champion = manual
        return self

    @model_validator(mode="after")
    def validate_external_eval_files(self) -> "AnalysisConfig":
        if str(self.data_processing.validate_on).lower() == "external":
            missing = []
            if not self.data_files.eval_x_file:
                missing.append("eval_x_file")
            if not self.data_files.eval_t_file:
                missing.append("eval_t_file")
            if not self.data_files.eval_y_file:
                missing.append("eval_y_file")
            if missing:
                raise ValueError(
                    f"validate_on='external' erfordert Evaluationsdaten. "
                    f"Fehlend in data_files: {', '.join(missing)}. "
                    f"Bitte eval_x_file, eval_t_file und eval_y_file angeben."
                )
        return self


def load_config(path: Union[str, Path]) -> AnalysisConfig:
    """Lädt und validiert eine YAML-Konfiguration.
- Unbekannte Schlüssel werden abgewiesen (extra="forbid").
- Fehlende Pflichtfelder führen zu klaren Fehlermeldungen."""
    path_str = str(path)
    with open(path_str, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Die Konfigurationsdatei muss ein YAML-Objekt (Mapping) sein.")

    raw = dict(raw)
    raw["source_config_path"] = path_str
    return AnalysisConfig.model_validate(raw)
