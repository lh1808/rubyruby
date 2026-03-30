# Entwicklerhandbuch – rubin

Dieses Dokument richtet sich an Entwicklerinnen und Entwickler, die **rubin** erweitern oder bestehende Teile umbauen.
Der Fokus liegt auf einer konsistenten Struktur, klaren Zuständigkeiten und stabilen Produktionsartefakten.

## Entwicklungsumgebung einrichten

[Pixi](https://pixi.sh) ist das empfohlene Tool für das Environment-Management.
Es verwaltet Python, conda-forge- und PyPI-Pakete einheitlich und erzeugt ein
reproduzierbares Lockfile (`pixi.lock`).

```bash
# Pixi installieren (einmalig)
curl -fsSL https://pixi.sh/install.sh | bash

# Dev-Environment aufbauen (Tests + Linting)
cd rubin_repo
pixi install -e dev

# Tests ausführen
pixi run test

# Tests mit Coverage
pixi run test-cov

# Linting (Ruff)
pixi run lint

# Auto-Fix für Lint-Fehler
pixi run lint-fix

# Alle verfügbaren Tasks anzeigen
pixi task list
```

**Alternativ (ohne pixi):** `pip install -e ".[dev,shap]"` in einer virtuellen Umgebung.

### Environments

| Environment | Inhalt | Typischer Einsatz |
|---|---|---|
| `default` | Core-Pipeline + SHAP | Training, Evaluation, Reporting |
| `app` | default + Flask | Web-UI starten (`pixi run app`) |
| `dev` | default + pytest + ruff | Entwicklung und CI |

## Grundprinzipien

1. **Analyse ≠ Produktion**  
   Analyse darf experimentieren; Produktion muss stabil sein.

2. **Bundles sind der Vertrag**  
   Produktion arbeitet ausschließlich mit exportierten Artefakten (Bundle-Verzeichnis).

3. **Registries statt verstreuter Sonderlogik**  
   Neue Learner und Modellvarianten werden über `ModelRegistry` angebunden, nicht über zusätzliche `if/else`-Blöcke in den Runnern.

## Projektstruktur

```text
rubin/
  pipelines/
    analysis_pipeline.py
    production_pipeline.py
    data_prep_pipeline.py
  evaluation/
    drtester_plots.py
  explainability/
    shap_uplift.py
    permutation_uplift.py
    segment_analysis.py
    reporting.py
  reporting/
    html_report.py            ← HTML-Report-Generator (analysis_report.html)
  model_registry.py
  model_management.py
  tuning_optuna.py
  training.py
  preprocessing.py
  feature_selection.py
  artifacts.py
  settings.py
  utils/
    categorical_patch.py       ← EconML-Kompatibilität: kategorische Features
    data_utils.py
    io_utils.py
    schema_utils.py
    uplift_metrics.py
configs/
docs/
run_analysis.py
run_production.py
run_dataprep.py
run_explain.py
run_promote.py
```

## 1) Neuen kausalen Learner ergänzen

### Ort
`rubin/model_registry.py`

### Vorgehen
1. Implementiere eine Factory-Funktion, die eine Modellinstanz erzeugt.
2. Nutze `ModelContext`, um Base-Learner-Typ, Fixparameter und getunte Rollenparameter zu beziehen.
3. Registriere das Modell im `ModelRegistry`.
4. Ergänze in `rubin/settings.py` den Modellnamen in `SUPPORTED_MODEL_NAMES`, damit Konfigurationen strikt validiert bleiben.
5. Prüfe, ob für das Modell task-basiertes Tuning benötigt wird. Falls ja, ergänze die Rollensignaturen in `rubin/tuning_optuna.py`.
6. **Multi-Treatment-Kompatibilität:** Falls das neue Modell Multi-Treatment nicht unterstützt, ergänze den Modellnamen in `_BT_ONLY_MODELS` in `rubin/settings.py`. `_predict_effect()` erwartet, dass kompatible Modelle bei MT ein 2D-Array (n, K-1) zurückgeben. Bei BT genügt (n,) oder (n, 1).

Beispiel-Skizze:

```python
from rubin.model_registry import ModelRegistry, ModelContext
from rubin.tuning_optuna import build_base_learner

def make_my_learner(ctx: ModelContext):
    base = build_base_learner(
        ctx.base_learner_type,
        {**ctx.base_fixed_params, **ctx.params_for("overall_model")},
        seed=ctx.seed,
        task="regressor",  # Meta-Learner Outcome-Modelle sind Regressoren
    )
    return MyLearner(model=base)

registry = ModelRegistry()
registry.register("MyLearner", make_my_learner)
```

Zusätzlich in der YAML:

```yaml
models:
  models_to_train:
    - MyLearner
```

## 2) Neuen Base-Learner ergänzen

### Orte
- `rubin/tuning_optuna.py`
- `rubin/model_registry.py` nutzt denselben Builder indirekt über `build_base_learner(...)`

### Schritte
- Builder-Zweig für Klassifikation und Regression ergänzen
- sinnvolle Default-Search-Spaces hinterlegen
- Search-Space-Dokumentation in `docs/tuning_optuna.md` anpassen
- Beispiel-Konfigurationen aktualisieren

## 3) Neue Metriken ergänzen

Metriken liegen in `rubin/utils/uplift_metrics.py`.

Konventionen:
- Funktionen sind möglichst side-effect-frei
- Inputs und Rückgaben bleiben numerisch und einfach serialisierbar
- neue Metriken sollten in der Analyse-Pipeline sowohl nach MLflow als auch in die JSON-Zusammenfassung geschrieben werden

## 4) Production Pipeline erweitern

Erweiterungen wie zusätzliche Ausgabeformate, Batching oder parallele Verarbeitung gehören nach
`rubin/pipelines/production_pipeline.py` und bei Bedarf in `run_production.py`.

Wichtig:
- keine Trainingslogik in Production
- keine Feature-Selektion in Production
- keine impliziten Schemaänderungen zur Laufzeit

## 5) Task-basiertes Optuna-Tuning

Das Base-Learner-Tuning arbeitet task-basiert. Das bedeutet:

1. aus `models_to_train` wird ein interner Trainingsplan erzeugt
2. identische Base-Learner-Aufgaben werden dedupliziert
3. die besten Parameter werden anschließend allen passenden Rollen zugeordnet

Eine Tuning-Task wird über folgende Merkmale beschrieben:
- Base-Learner-Familie
- Objective-Familie
- Estimator-Task (`classifier` oder `regressor`)
- Sample-Scope
- Nutzung des Treatment-Features
- Zieltyp

Beispiele für gemeinsam nutzbare Aufgaben:
- `TLearner / models` und `XLearner / models` (Outcome-Regression pro Gruppe)
- `model_y` wird über NonParamDML, ParamDML und CausalForestDML geteilt (Outcome-Classifier)
- `model_t` (DML) und `model_propensity` (DRLearner) teilen die Propensity-Aufgabe
- `DRLearner / model_regression` und `SLearner / overall_model` nutzen eigene Regression-Tasks

**Hinweis:** `model_y` und `model_t` werden **nicht** zusammengelegt (verschiedene Zieltypen und Objective-Familien). Jede Study erhält einen eigenen, deterministisch abgeleiteten Seed.

Nicht zusammengelegt werden Aufgaben, die auf anderer Datengrundlage, mit anderem Zieltyp oder unterschiedlichem Estimator-Task (Classifier vs. Regressor) trainiert werden.

## 6) Final-Modelle und R-Score-Tuning

Einige Learner besitzen ein Final-Modell, das den CATE aus Residuen oder Pseudo-Outcomes lernt.
In **rubin** läuft dieses Tuning getrennt vom Base-Learner-Tuning über `final_model_tuning`.

Praktische Konsequenzen:
- `model_final` ist typischerweise ein Regressor
- Parameter für `model_final` werden getrennt von `model_y`, `model_t` oder `model_propensity` behandelt
- das Tuning wird bewusst nur einmal auf dem ersten Cross-Prediction-Train-Fold durchgeführt und danach wiederverwendet
- `stability_penalty` (0.0–2.0) ergänzt den R-Score um einen Stabilitäts-Term: `score = R_score − λ · log(1 + CV)`, wobei CV = std(CATE) / |median(CATE)|. Das bestraft Parameterkombinationen, die zu extremer CATE-Streuung führen — ein häufiges Overfitting-Problem bei kleinen Treatment-Effekten. Die Berechnung findet auf den Tuning-Daten statt und verursacht keinen zusätzlichen Fit.

## 7) Modell-Management (Champion/Challenger)

Siehe `rubin/model_management.py`:
- beim Analyselauf entsteht ein Registry-Manifest
- ein Champion wird automatisch oder manuell gewählt
- optional wird der Champion vor dem Bundle-Export auf allen im Run verfügbaren Daten refittet
- `run_promote.py` kann den Champion im Manifest nachträglich umstellen

## 8) Explainability erweitern

Explainability ist bewusst als separater Batch-Workflow umgesetzt. Einstiegspunkt ist `run_explain.py`, der auf einem Bundle arbeitet.

Kernmodule unter `rubin/explainability/`:
- `shap_uplift.py`
- `permutation_uplift.py`
- `segment_analysis.py`
- `reporting.py`

Neue Explainability-Bausteine sollten:
1. eine klar testbare Funktion liefern
2. keine Trainingslogik benötigen
3. Artefakte als CSV/PNG oder andere einfache Dateiformate erzeugen

## 9) Kategorische Features und EconML

EconML konvertiert X intern zu numpy (`sklearn.check_array`), wodurch pandas `category`-Dtypes verloren gehen. rubin löst das über einen Context-Manager in `rubin/utils/categorical_patch.py`:

```python
# Patch 1: Feature-Selektion (vor Feature-Reduktion)
with patch_categorical_features(X_all, base_learner_type="lgbm"):
    compute_importances(...)  # LightGBM nutzt native kat. Splits
X_reduced = select_features(...)

# Patch 2: Tuning + Training (nach Feature-Reduktion, neue Spaltenindizes)
with patch_categorical_features(X_reduced, base_learner_type="lgbm") as cat_indices:
    tuner.tune_all(...)
    model.fit(Y, T, X=X_reduced)
# Originale .fit()-Methoden wiederhergestellt
```

Zwei separate Patches sind nötig, weil sich die Spaltenindizes nach Feature-Selektion ändern. Der Patch deckt LightGBM (`categorical_feature`) und CatBoost (`cat_features`) ab.

Bei neuen Base Learnern muss `categorical_patch.py` erweitert werden, falls der Learner einen eigenen Parameter für kategorische Features hat.

## 10) model_final Parameter-Isolation

Die getunten Nuisance-Parameter (aus Optuna für `model_y`/`model_t`) werden **nicht** an `model_final` vererbt. Dies wird in `model_registry._base()` über `cate_model_roles` sichergestellt:

- `model_final` nutzt nur `base_fixed_params` + explizite FMT-Params
- Kein Fallback auf den `"default"`-Key in `tuned_params`

Hintergrund: Classifier-optimierte Parameter wie `min_split_gain=0.95` lassen bei CATE-Regression keinen einzigen Split zu — der Baum kollabiert zu einem Intercept (konstante Vorhersage). Wenn neue Rollen mit ähnlicher Problematik hinzukommen, müssen sie in `cate_model_roles` ergänzt werden.

## Best Practices

- Konfiguration ist die Quelle der Wahrheit
- neue Felder immer in `settings.py` modellieren und validieren
- Runner schlank halten; Geschäftslogik gehört in Module unter `rubin/`
- Bundles rückwärtskompatibel erweitern statt implizit umzubauen
- bei neuen Modellrollen immer prüfen, ob sie in `tuning_optuna.py` und in der Dokumentation ergänzt werden müssen

## 11) Evaluation: 3-Phasen-Architektur

Die Evaluation ist in drei Phasen aufgeteilt, um Geschwindigkeit und Qualität zu balancieren:

**Phase 1 (alle Modelle, immer):** Schnelle NumPy-Metriken (Qini, AUUC, Uplift@k, Policy Value) + CATE-Verteilungs-Histogramme. <1s pro Modell. Bestimmt den Champion.

**Phase 2 (Level-abhängig):** DRTester-Diagnostik (Calibration, Qini/TOC mit Bootstrap-CIs). Level 1–2 alle, Level 3 Champion+Challenger, Level 4 nur Champion. Nuisance-Modelle mit gecappten Parametern (n_estimators≤100, cv=3).

**Phase 3 (alle Modelle, immer):** scikit-uplift-Plots (Qini-Kurve, Uplift-by-Percentile, Treatment-Balance).

Beim Hinzufügen neuer Plots: Schnelle Plots (<1s) in Phase 1, teure Plots in Phase 2 (Level-abhängig), moderate Plots in Phase 3 (immer). `plt.close(fig)` nach dem MLflow-Logging nicht vergessen.

## 12) CATE-Verteilungs-Plots

`generate_cate_distribution_plot()` in `drtester_plots.py` erzeugt Side-by-Side-Histogramme (Training + Cross-Validated). Wird in Phase 1 für alle Modelle aufgerufen — BT: 1 Plot/Modell, MT: 1 Plot/Arm. Surrogate bekommt einen eigenen Plot in `_train_and_evaluate_surrogate()`. Gespeichert als `distribution__<Model>.png` in MLflow und im HTML-Report unter `cate_distribution`.

## 13) CatBoost-aware Parallelisierung

CatBoost's Symmetric-Tree skaliert schlechter mit wenigen Threads als LightGBMs Leaf-wise Growth. Deshalb passen `_tuning_n_jobs()` (beide Tuner) und die Training-Fold-Parallelisierung die Worker-Zahl an den Base-Learner-Typ an:

- LightGBM Level 4: `n_cpus // 2` Workers × 2 Threads/Worker
- CatBoost Level 4: `n_cpus // 4` Workers × 4 Threads/Worker

`n_jobs`/`thread_count` werden in den Builder-Funktionen NACH `params.update()` gesetzt, damit `fixed_params` die Parallelisierung nicht überschreiben können.

## 14) Feature-Selektion: Parallelisierung

Die Importance-Methoden in `compute_importances()` laufen immer sequentiell — jede Methode bekommt alle CPU-Kerne (`n_jobs=-1`). Thread-Parallelisierung auf Methoden-Ebene ist nicht möglich, da CausalForest intern joblib-Prozesse spawnt (fork-aus-Thread → Deadlock). Sequentiell mit voller Kernauslastung + Subsampling (>100k) ist in der Praxis schnell genug.

Bei Multi-Treatment binarisiert `_causal_forest_importance()` T automatisch zu Control(0) vs. AnyTreatment(1), da GRF binäres Treatment erwartet. Bei Level 1 wird `n_jobs=1` übergeben (minimaler RAM).

## 15) sklearn/scikit-uplift Kompatibilitäts-Shim

`drtester_plots.py` registriert einen Shim für `sklearn.utils.check_matplotlib_support`, das in sklearn 1.6+ entfernt wurde. scikit-uplift <0.6 importiert es noch. Der Shim wird **vor** dem sklift-Import registriert. Bei einer neuen scikit-uplift-Version (≥0.6) kann der Shim entfernt werden.

## 16) Warning-Filter

In `analysis_pipeline.run()` werden drei harmlose Warnungen unterdrückt:

- `"X does not have valid feature names"` (sklearn): EconML mischt intern DataFrame- und numpy-Aufrufe
- `"filesystem tracking backend is deprecated"` (mlflow): Infrastruktur-Empfehlung, kein Fehler
- `"force_all_finite was renamed to ensure_all_finite"` (sklearn): EconML nutzt noch den alten Parameter-Namen. Rein kosmetisch, wird in einer zukünftigen EconML-Version behoben.

Gleiche Filter auch in `production_pipeline.__init__()`.

## 17) CausalForestDML und Parallelisierung

CausalForestDML nutzt intern einen GRF (Generalized Random Forest), der joblib-Prozesse für die Baum-Parallelisierung spawnt. Das führt zu Deadlocks wenn der GRF in einem joblib-Thread läuft (fork-aus-Thread-Problem). Drei Stellen sind geschützt:

- **Feature-Selektion** (`_causal_forest_importance`): Alle Kerne im Hauptthread + Subsampling auf 100k Zeilen bei großen Datensätzen
- **Training CV-Folds** (`train_and_crosspredict_bt_bo`): CausalForestDML erzwingt sequentielle Folds, auch bei Level 3/4. GRF bekommt alle Kerne für seine interne Parallelisierung.
- **BL-Tuning, EconML tune(), Bundle-Refit, Production**: Laufen im Hauptthread → kein Problem
