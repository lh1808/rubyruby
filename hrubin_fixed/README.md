# rubin – Analyse- und Production-Pipelines für Causal ML

Dieses Repository enthält eine modular aufgebaute Codebasis für kausale Modellierung
mit klarer Trennung zwischen **Analyse** und **Production**.

Unterstützt werden:
- **Binary Treatment / Binary Outcome** (T ∈ {0, 1}, Y ∈ {0, 1})
- **Multi-Treatment / Binary Outcome** (T ∈ {0, 1, …, K-1}, Y ∈ {0, 1})

## Struktur

- `rubin/` – Kernlogik (Pipelines, Registry, Tuning, Feature-Selektion, Reporting)
- `app/` – Web-UI (React-Frontend + Flask-Backend)
- `run_dataprep.py` – Startpunkt Datenaufbereitung (optional)
- `run_analysis.py` – Startpunkt Analyse (erzeugt automatisch `analysis_report.html`)
- `run_production.py` – Startpunkt Production Scoring
- `run_explain.py` – Explainability auf Bundle-Basis
- `run_promote.py` – Champion im Bundle manuell umstellen
- `docs/` – ausführliche deutsche Dokumentation

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

### DataPrep (optional, wenn X/T/Y noch nicht vorliegen)
Voraussetzung: In der verwendeten YAML-Konfiguration existiert eine Sektion `data_prep`.

```bash
pixi run dataprep -- --config configs/config_full_example.yml
# oder: python run_dataprep.py --config configs/config_full_example.yml
```

Hinweis: Eine separate `data_config.yml` ist in diesem Repository nicht erforderlich.
Alle relevanten Parameter für die Datenaufbereitung werden zentral über `config.yml`
gesteuert.

### Analyse
```bash
pixi run analyze -- --config config.yml
# oder: python run_analysis.py --config config.yml
```

### Analyse mit synchronem Bundle-Export
```bash
pixi run analyze -- --config config.yml --export-bundle --bundle-dir bundles
# oder: python run_analysis.py --config config.yml --export-bundle --bundle-dir bundles
```

### Production Scoring
```bash
pixi run score -- --bundle bundles/<bundle_id> --x new_X.parquet --out scores.csv
# oder: python run_production.py --bundle bundles/<bundle_id> --x new_X.parquet --out scores.csv
# Standard: Champion aus model_registry.json
# Optional: --model-name NonParamDML oder --use-all-models
# Surrogate: --use-surrogate (interpretierbarer Einzelbaum)
```

## Web-UI

Interaktive Web-Oberfläche zur Konfiguration und Steuerung der Analyse-Pipeline:

```bash
# Mit pixi (empfohlen)
pixi run app

# Ohne pixi
pip install flask pandas pyarrow pyyaml
python -m app.server
```

Die UI führt durch alle Schritte: Datenvorbereitung (optional), Treatment-Typ und Vorlage, Modellauswahl, Tuning-Konfiguration, Parallelisierungs-Level, Config-Preview und Analyse-Start. Während der Analyse zeigt ein Live-Log-Fenster die Pipeline-Logs in Echtzeit. Erzeugt die gleiche YAML-Konfiguration, die auch direkt an `run_analysis.py` übergeben werden kann.

Architektur: React-Frontend (`app/frontend/index.html`) mit Flask-Backend (`app/server.py`). Die React-App wird als statische HTML vom Server ausgeliefert und kommuniziert über REST-API-Endpoints für Upload, Analyse-Start, Fortschritts-Tracking und Downloads.

### Build-Schritt (einmalig)

Die React-App nutzt React 18 und Babel. In Firmennetzen ohne CDN-Zugang müssen die Libraries einmalig eingebettet werden:

```bash
python scripts/build_app_html.py   # Braucht Internetzugang
```

Danach ist `app/frontend/index.html` selbständig (~3 MB) und funktioniert offline.

### Domino App-Deployment

Für das Deployment auf Domino Data Lab:

```bash
# Domino findet app.sh automatisch beim App-Deployment
# Keine manuelle Konfiguration nötig.
```

`app.sh` im Projektstamm konfiguriert Port und startet den Flask-Server. Details: `docs/domino_deployment.md`.

## Dokumentation

Siehe `docs/README_DE.md` für Einstieg und Details. Wichtige Themen sind:
- Konfiguration (`docs/konfiguration.md`)
- Optuna-Tuning (`docs/tuning_optuna.md`)
- Bundles (`docs/bundles.md`)
- Domino Deployment (`docs/domino_deployment.md`)
- Evaluation (`docs/evaluation.md`)
- Explainability (`docs/explainability.md`)
- Architektur und Entwicklerhinweise (`docs/architektur.md`, `docs/developer_guide.md`)

Abgedeckt sind unter anderem:
- Schema-Validierung in Production (inkl. `schema_report.json`)
- persistente Optuna-Studies (SQLite)
- Uplift-Metriken (Qini, AUUC, Uplift@k, Policy Value)
- Explainability (SHAP/Permutation) und Segment-Analysen auf Bundle-Basis
- NaN-Toleranz: Alle Modelle außer CausalForestDML können mit fehlenden Werten umgehen (via LightGBM/CatBoost). CausalForestDML und die Feature-Selektionsmethode `causal_forest` (GRF) werden bei NaN automatisch übersprungen.
- Validierungsmodi: Cross-Validation (K-Fold), Holdout (stratifizierter Split) und External (separater Eval-Datensatz, leakage-frei)
- Parallelisierung: Konfigurierbar über `constants.parallel_level` (1–4). Level 2 (Default) parallelisiert Base Learner, Level 3–4 parallelisieren zusätzlich CV-Folds via joblib. Kerne werden proportional aufgeteilt (keine Übersubskription)
- Kategorische Features: Automatisches Patching für EconML-Kompatibilität — LightGBM/CatBoost nutzen native kategoriale Splits, auch wenn EconML X intern zu numpy konvertiert
- Internes Cross-Fitting: Alle DML-Modelle und DRLearner verwenden `cv=5` (statt EconML-Default `cv=2`) für stabilere Nuisance-Residuals

## Explainability

Für nachvollziehbare Modellinterpretationen steht ein separater Startskript zur Verfügung.
Er arbeitet auf einem Bundle (Preprocessor + Modelle + Registry) und erzeugt Feature-
Importances sowie Segment-Reports.

```bash
pixi run explain -- --bundle bundles/<bundle_id> --x new_X.parquet --out-dir explain
# oder: python run_explain.py --bundle bundles/<bundle_id> --x new_X.parquet --out-dir explain
```

Hinweis zu Abhängigkeiten:
* Für die Methode `--method shap` wird das Paket `shap` benötigt.
* Ohne SHAP kann `--method permutation` genutzt werden.


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
- `config_holdout_production.yml`: Holdout-Validierung (20%) – in der Web-UI auch als Add-on verfügbar
- `config_external_eval.yml`: Externe Validierung (separater Eval-Datensatz, kein Data-Leakage)
- `config_explainability.yml`: Feature-Selektion + erweiterte SHAP/Segment-Einstellungen
- `config_benchmark.yml`: Vergleich neuer Scores gegen einen historischen Score (S)
- `config_full_example.yml`: End-to-End mit DataPrep-Sektion (Pfade anpassen)
- `config_binary_treatment.yml`: Binary Treatment (T ∈ {0, 1}) mit allen 7 Modellen inkl. CausalForestDML tune()
- `config_multi_treatment.yml`: Multi-Treatment-Szenario (T ∈ {0, 1, …, K-1}) mit DML-Modellen
- `config_speed.yml`: Speed-Tuning mit Single-Fold – für große Datensätze (>500k Zeilen)

Aufrufbeispiel:

```bash
pixi run analyze -- --config configs/config_lgbm_standard.yml --export-bundle
# oder: python run_analysis.py --config configs/config_lgbm_standard.yml --export-bundle
```


## Multi-Treatment

Neben dem klassischen Binary-Treatment-Szenario (T ∈ {0, 1}) unterstützt rubin auch
**Multi-Treatment** (T ∈ {0, 1, …, K-1}), z. B. für die Auswahl zwischen mehreren
Kampagnentypen.

Aktivierung in der Konfiguration:

```yaml
treatment:
  type: multi
  reference_group: 0

models:
  models_to_train:
    - NonParamDML
    - DRLearner
    - CausalForestDML

selection:
  metric: policy_value
```

**Wichtig:**
- SLearner, TLearner und XLearner sind bei Multi-Treatment nicht verfügbar.
- Statt eines skalaren CATE gibt es K-1 Effekt-Spalten (je Treatment-Arm vs. Control).
- Die Evaluation enthält pro-Arm-Qini/AUUC/Uplift-Werte, pro-Arm-Policy-Values
  (`policy_value_treat_positive_T{k}`) sowie einen globalen IPW-basierten `policy_value`.
- Für die Champion-Auswahl wird `metric: policy_value` empfohlen (alternativ:
  `policy_value_treat_positive_T1`, `qini_T1`, etc.).

Siehe `configs/config_multi_treatment.yml` für ein vollständiges Beispiel und
`docs/architektur.md` für Details zur Implementierung.


## Champion/Challenger (Model Registry)

Beim Bundle-Export wird automatisch ein `model_registry.json` erzeugt. Darin wird ein
**Champion-Modell** festgelegt (Standard für Produktion) und es werden alle trainierten
Modelle als **Challenger** dokumentiert.

Champion umschalten:

```bash
pixi run promote -- --bundle bundles/<bundle_id> --model <ModelName>
# oder: python run_promote.py --bundle bundles/<bundle_id> --model <ModelName>
```