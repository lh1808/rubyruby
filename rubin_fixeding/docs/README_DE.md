# Dokumentation (Deutsch)

Diese Dokumente beschreiben Nutzung, Konfiguration und Erweiterung des Frameworks. Die zentrale Konfiguration wird über `config.yml` gesteuert und beim Laden mit **Pydantic** strikt validiert.

- Architekturüberblick: `docs/architektur.md`
- Globale Konfigurationsreferenz: `docs/konfiguration.md`
- Tuning (Optuna, R-Score/R-Loss, Locking): `docs/tuning_optuna.md`
- Evaluation (DRTester, scikit-uplift Plots): `docs/evaluation.md`
- Bundles/Production: `docs/bundles.md`
- Explainability: `docs/explainability.md`
- Entwicklerleitfaden: `docs/developer_guide.md`
- Domino App-Deployment: `docs/domino_deployment.md`


## Beispiel-Konfigurationen (`configs/`)

Im Ordner `configs/` liegen mehrere vorkonfigurierte Beispiele für verschiedene Szenarien:

- `config_reference_all_options.yml`: vollständige Referenz mit **allen** Feldern (Nachschlagewerk)
- `config_quickstart.yml`: minimaler Einstieg – ein Modell, kein Tuning, kein Bundle
- `config_exploration.yml`: schnelle Iteration mit 10% Downsampling und wenigen Trials
- `config_lgbm_standard.yml`: LightGBM mit moderatem Tuning (30 Trials) + Bundle-Export
- `config_lgbm_intensiv.yml`: LightGBM mit gründlichem Tuning (80 Trials), Final-Model-Tuning, persistente Studies
- `config_catboost_standard.yml`: CatBoost mit moderatem Tuning (30 Trials) + Bundle-Export
- `config_catboost_intensiv.yml`: CatBoost mit gründlichem Tuning (80 Trials), Final-Model-Tuning, persistente Studies
- `config_dml_focus.yml`: Fokus auf DML-Familie (NonParamDML, DRLearner, CausalForestDML mit EconML-Tune)
- `config_holdout_production.yml`: Holdout-Validierung (20%) als letzter Schritt vor Production
- `config_external_eval.yml`: Externe Validierung – Training und Evaluation auf getrennten Datensätzen (kein Leakage)
- `config_explainability.yml`: Feature-Selektion + erweiterte SHAP/Segment-Einstellungen
- `config_benchmark.yml`: Vergleich neuer Scores gegen einen historischen Score (S)
- `config_full_example.yml`: End-to-End mit DataPrep-Sektion (Pfade anpassen)
- `config_multi_treatment.yml`: Multi-Treatment-Szenario (T ∈ {0, 1, …, K-1}) mit DML-Modellen
- `config_binary_treatment.yml`: Binary-Treatment-Referenz mit allen BT-Modellen, FMT und Explainability
- `config_speed.yml`: Speed-Tuning mit Single-Fold überall – für große Datensätze

Aufrufbeispiel:

```bash
pixi run analyze -- --config configs/config_lgbm_standard.yml --export-bundle
# oder: python run_analysis.py --config configs/config_lgbm_standard.yml --export-bundle
```


Diese Codebasis trennt **Analyse** und **Production** sauber voneinander:

- **Analyse-Pipeline**: Trainieren, Feature-Selektion (LGBM + GRF Union), Tuning, Evaluieren, HTML-Report generieren.
- **Production-Pipeline**: Stabiles Scoring auf neuen Daten (inkl. Surrogate-Einzelbaum-Option).

Die wichtigsten Einstiege:

- `run_analysis.py` – startet die Analyse-Pipeline (erzeugt `analysis_report.html`)
- `run_production.py` – startet die Production-Pipeline (Scoring)  
- `run_explain.py` – erzeugt Explainability-Artefakte auf Bundle-Basis  
- `run_promote.py` – setzt manuell einen anderen Champion im Bundle  
- `config.yml` – zentrale Konfigurationsdatei


## Schnellstart

### Environment aufsetzen (empfohlen: pixi)

[Pixi](https://pixi.sh) verwaltet alle Dependencies (Python, conda-forge, PyPI) automatisch
und erzeugt ein reproduzierbares Lockfile. Installation: `curl -fsSL https://pixi.sh/install.sh | bash`

```bash
cd rubin_repo
pixi install                # Environment aufbauen (einmalig)
pixi run analyze-quick      # Smoke-Test (postinstall läuft automatisch)
pixi run app                # Web-UI starten
pixi run test               # Tests ausführen
```

Alle verfügbaren Tasks: `pixi task list`

> **Hinweis:** `pixi install` erzeugt automatisch ein `pixi.lock`, das exakte Paketversionen
> fixiert. Diese Datei sollte ins Repository eingecheckt werden, damit alle Teammitglieder
> identische Environments erhalten. Die Dateien `requirements.txt` und
> `app/requirements_app.txt` dienen als pip-Fallback und werden über
> `pixi run sync-requirements` aus `pyproject.toml` generiert.

**Alternativ (ohne pixi):** Python 3.10+ mit `pip install -r requirements.txt`.

### Analyse (ohne Bundle-Export)
```bash
pixi run analyze -- --config config.yml
# oder: python run_analysis.py --config config.yml
```

### Analyse mit synchronem Bundle-Export
```bash
pixi run analyze -- --config config.yml --export-bundle --bundle-dir bundles
# oder: python run_analysis.py --config config.yml --export-bundle --bundle-dir bundles
```

### Production Scoring mit Bundle
```bash
pixi run score -- --bundle bundles/<bundle_id> --x new_X.parquet --out scores.csv
# oder: python run_production.py --bundle bundles/<bundle_id> --x new_X.parquet --out scores.csv
# Standard: Champion aus model_registry.json
# Optional: --model-name NonParamDML oder --use-all-models
# Surrogate: --use-surrogate (interpretierbarer Einzelbaum)
```

### Optional: DataPrep aus derselben Konfiguration

Wenn du Rohdaten zuerst in `X.parquet`, `T.parquet`, `Y.parquet` (und optional `S.parquet`) überführen möchtest,
kannst du die DataPrepPipeline über die Sektion `data_prep` in der zentralen `config.yml` steuern.

```bash
pixi run dataprep -- --config config.yml
# oder: python run_dataprep.py --config config.yml
```

Hinweis: Eine separate `data_config.yml` ist in diesem Repository nicht erforderlich.
Alle relevanten Parameter der Datenaufbereitung sind in der Sektion `data_prep` der
zentralen Konfiguration abbildbar.


### Optional: Externe Validierung (separater Eval-Datensatz)

Für eine leakage-freie Evaluation auf einem separaten Datensatz:

1. In `data_prep`: `eval_data_path` setzen → Der Preprocessor wird nur auf den Train-Daten gefittet und auf die Eval-Daten nur transformierend angewendet.
2. In `data_processing`: `validate_on: "external"` setzen.
3. In `data_files`: `eval_x_file`, `eval_t_file`, `eval_y_file` auf die erzeugten Eval-Dateien zeigen lassen.

Die Analyse-Pipeline trainiert auf den vollen Trainingsdaten und evaluiert ausschließlich auf dem externen Datensatz. Tuning und Cross-Predictions laufen weiterhin intern auf den Trainingsdaten.


### Optional: Vergleich mit einem historischen Score

Wenn bereits ein historisches Scoring existiert, kann rubin die gleichen Uplift-Auswertungen
zusätzlich auch für diesen Score berechnen. Dazu wird in der Konfiguration eine Datei über
`data_files.s_file` angegeben (CSV oder Parquet mit Score-Spalte, Standard: `S`).

Wichtig ist die Richtung des Scores:

- `historical_score.higher_is_better: true` bedeutet: große Werte sind "gut" (Top-Scores zuerst behandeln).
- `historical_score.higher_is_better: false` bedeutet: kleine Werte sind "gut" (rubin invertiert intern für die Sortierung).

Die Kennzahlen werden in MLflow unter dem konfigurierten Namen `historical_score.name` geloggt


## Globale Konfiguration

Die gesamte Steuerung erfolgt zentral über eine YAML-Datei (typischerweise `config.yml`).
Alle relevanten Stellschrauben sind dort gebündelt, damit Läufe reproduzierbar und vergleichbar bleiben.

Priorität der Einstellungen:
- Kommandozeilenparameter (z. B. `--export-bundle`) überschreiben die YAML.
- Die YAML überschreibt interne Voreinstellungen.

Eine vollständige Referenz aller Felder (inkl. Begründungen und Empfehlungen) steht in `konfiguration.md`.

### Hinweise zu Tuning und "Locking"

- Base-Learner-Tuning (Optuna) wird vor dem Training ausgeführt und die gefundenen Parameter werden für den gesamten Lauf verwendet.
- Final-Model-Tuning über R-Score/R-Loss (optional) wird **nur einmal** auf der Trainingsmenge des ersten Cross-Prediction-Folds durchgeführt.
  Danach sind die Parameter fest und werden in allen weiteren Folds wiederverwendet.
- Für `CausalForestDML` gilt das Gleiche für das interne EconML-`tune(...)`: Tuning nur in der ersten Iteration, danach fest.

Details stehen in `tuning_optuna.md`.

### Parallelisierung

Über `constants.parallel_level` (1–4) lässt sich steuern, wie aggressiv parallelisiert wird:

- **Level 1 (Minimal):** 1 Kern pro Fit, Folds sequentiell, Trials sequentiell — sicher auf jeder Maschine.
- **Level 2 (Moderat, Default):** Alle Kerne pro Fit, Folds sequentiell, Trials sequentiell — guter Kompromiss.
- **Level 3 (Hoch):** Kerne aufgeteilt, 2–4 Folds parallel, 2–4 Trials parallel — schneller, mehr RAM.
- **Level 4 (Maximum):** Kerne aufgeteilt, alle Folds parallel, max. Trials parallel — schnellste Laufzeit, höchster RAM.

Die CV-Fold-Parallelisierung nutzt joblib mit Thread-Backend. Da LightGBM und CatBoost den GIL während des C++-Trainings freigeben, wird echte Parallelität erzielt. Bei Level 3–4 werden die CPU-Kerne proportional auf die parallelen Folds und Tuning-Trials aufgeteilt (keine Übersubskription). In der Web-UI ist der Level über einen 4-Button-Selektor in der Experiment-Sektion konfigurierbar.

**Trade-offs:** Level 3–4 beschleunigen das Tuning durch parallele Optuna-Trials (3–7× schneller), reduzieren aber die TPE-Explorationsqualität leicht (~90–95% der sequentiellen Performance bei 30+ Trials). Beim Training sind die Ergebnisse mathematisch identisch — nur der RAM-Verbrauch steigt proportional (2–5× Baseline). Level 3 bietet den besten Gesamtkompromiss. Level 4 nur bei ausreichend RAM und wenn Speed kritisch ist. Details in `konfiguration.md` und `tuning_optuna.md`.

### Kategorische Features

EconML konvertiert X intern zu numpy, wodurch pandas `category`-Dtypes verloren gehen. rubin patcht die `.fit()`-Methoden von LightGBM/CatBoost automatisch, sodass kategoriale Spaltenindizes bei jedem internen Aufruf mitgegeben werden. Dadurch nutzen die Base Learner native kategoriale Splits, auch wenn EconML die Daten als numpy übergibt.

### Internes Cross-Fitting

Alle DML-Modelle und DRLearner verwenden intern `cv=5` für die Nuisance-Residualisierung (EconML-Default wäre `cv=2`). Dadurch sieht jedes Nuisance-Modell 80% statt nur 50% der Daten, was stabilere Residuals und bessere CATE-Schätzungen liefert.

