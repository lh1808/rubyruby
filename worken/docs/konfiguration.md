# Globale Konfiguration (config.yml)

In **rubin** wird das Verhalten der Analyse‑ und Production‑Pipelines über eine zentrale YAML‑Datei gesteuert
(`config.yml`). Ziel ist eine Konfiguration, die

- reproduzierbar (gleicher Run → gleiche Einstellungen),
- nachvollziehbar (jede Stellschraube ist dokumentiert),

bleibt.

## Grundprinzip: Was wird wo konfiguriert?

- **In `config.yml`** stehen alle fachlich/technisch sinnvollen Pipeline‑Einstellungen (Datenpfade, Feature‑Filter,
  Modellliste, Base‑Learner, Tuning, Champion‑Auswahl, Explainability‑Voreinstellungen, optionale lokale Outputs).
- **Auf der Kommandozeile** werden nur *Run‑Parameter* gesetzt, die typischerweise von Job zu Job variieren
  (Pfad zur Konfigdatei sowie optional gezielte Overrides wie Bundle‑Export oder Bundle‑Zielordner).

### Priorität (wichtig in der Praxis)

1. **Kommandozeile** (z. B. `--config`, `--bundle-dir`)  
2. **`config.yml`** (alle inhaltlichen Einstellungen, inkl. Bundle-Block)  
3. **Voreinstellungen in `settings.py`** (wenn ein Feld in der YAML fehlt)

## Minimales Beispiel

Hinweis zur Validierung

Die YAML wird beim Laden **strikt validiert**. Unbekannte Schlüssel (z. B. Tippfehler wie `baselearner` statt `base_learner`) führen zu einer klaren Fehlermeldung.
Das ist bewusst so gewählt, um stille Fehlkonfigurationen früh zu vermeiden.


```yaml
data_files:
  x_file: "data/X.parquet"
  t_file: "data/T.parquet"
  y_file: "data/Y.parquet"

mlflow:
  experiment_name: "rubin"

constants:
  SEED: 42

models:
  models_to_train: ["SLearner", "TLearner"]

base_learner:
  type: "lgbm"
  fixed_params: {}

tuning:
  enabled: false
  search_space: {}

selection:
  metric: "qini"
  higher_is_better: true
  refit_champion_on_full_data: true
  manual_champion: null
```

---

# Referenz: Alle Konfigurationsbereiche

## `data_files` – Eingabedateien

```yaml
data_files:
  x_file: "data/X.parquet"
  t_file: "data/T.parquet"
  y_file: "data/Y.parquet"
  # Optional: historischer Score (Vergleichsbasis). Wenn gesetzt, werden die gleichen
  # Uplift-Auswertungen zusätzlich auch für diesen Score gerechnet.
  s_file: null
  # Optional: Referenz der Ziel-Datentypen (z. B. aus DataPrep als dtypes.json).
  dtypes_file: null

  # Optional: Externe Evaluationsdaten (für validate_on: "external").
  # Wenn gesetzt, wird auf diesen Daten evaluiert, während die obigen Daten
  # vollständig zum Training verwendet werden.
  eval_x_file: null
  eval_t_file: null
  eval_y_file: null
  eval_s_file: null   # optional: historischer Score im Eval-Datensatz

  # Optional: Boolean-Maske für "Train Many, Evaluate One".
  # Wird von DataPrep bei eval_file_index erzeugt.
  eval_mask_file: null  # z. B. "data/processed/eval_mask.npy"

# Optional: Einstellungen zum historischen Score (gilt nur, wenn data_files.s_file gesetzt ist)
historical_score:
  # Name, unter dem der historische Score in Ausgaben/Artefakten geführt wird.
  name: "historical_score"
  # Spaltenname im s_file (Standard: "S")
  column: "S"
  # True: große Werte sind "gut" (Top-Scores zuerst behandeln)
  # False: kleine Werte sind "gut" (intern wird der Score invertiert)
  higher_is_better: true
```

- `x_file`: Feature‑Tabelle (Spalten = Merkmale, Zeilen = Beobachtungen)
- `t_file`: Treatment‑Vektor (0/1 bei Binary Treatment)
- `y_file`: Outcome‑Vektor (0/1 bei Binary Outcome)
- `s_file` (optional): historischer Score als Vergleichsbasis (CSV oder Parquet mit Score-Spalte, Standard "S")
- `eval_x_file`, `eval_t_file`, `eval_y_file` (optional): Separater Evaluationsdatensatz für `validate_on: "external"`. Wenn gesetzt, wird auf diesen Daten evaluiert, während `x_file`/`t_file`/`y_file` vollständig zum Training verwendet werden. Kein Data-Leakage, da das Preprocessing in der DataPrep getrennt auf den Trainingsdaten gefittet und nur transformierend auf die Eval-Daten angewendet wird.
- `eval_s_file` (optional): Historischer Score im Eval-Datensatz (für Benchmark-Vergleich auf externen Daten)
- `eval_mask_file` (optional): Boolean-Maske (.npy) für „Train Many, Evaluate One". Alle Zeilen werden für Training und Cross-Prediction genutzt, aber Uplift-Metriken und DRTester-Plots nur auf den markierten Zeilen berechnet. Wird automatisch von DataPrep erzeugt, wenn `eval_file_index` gesetzt ist.

### Historischer Score (Vergleichsbasis)

Wenn `data_files.s_file` gesetzt ist, werden Uplift-Kennzahlen (Qini, AUUC, Uplift@k, Policy Value)
Modelle quantifizieren möchte.

Wichtig ist dabei die Interpretation der Score-Richtung:

- `higher_is_better: true` bedeutet: große Score-Werte sind "gut" (die Top-Scores sind die zuerst zu behandelnden Fälle).
- `higher_is_better: false` bedeutet: kleine Score-Werte sind "gut". In diesem Fall invertiert rubin den Score intern,
  damit die Sortierung in den Uplift-Metriken korrekt ist.

**Warum konfigurierbar?**  
Dateipfade unterscheiden sich zwischen lokalen Läufen, Batch‑Jobs und CI‑Umgebungen. Die Pipeline soll dafür
nicht angepasst werden müssen.

---

## `data_prep` – Datenaufbereitung (optional)

Die DataPrepPipeline ist optional. Sie wird genutzt, wenn die Rohdaten erst in die drei
Standarddateien `X.parquet`, `T.parquet`, `Y.parquet` (und optional `S.parquet`) überführt werden sollen.

Wichtig: Die Analyse-Pipeline benötigt weiterhin `data_files`. Typischer Workflow ist daher:

1. DataPrep ausführen (schreibt `X.parquet`, `T.parquet`, `Y.parquet`, `preprocessor.pkl`, … in `data_prep.output_path`)
2. `data_files.*` auf diese Ausgabedateien zeigen lassen (entweder im selben `config.yml` oder in einem
   zweiten, identischen Analyse-Config-File)
3. Analyse-Pipeline starten

Bei **externer Validierung** (`validate_on: "external"`) zusätzlich:

1. `eval_data_path` in `data_prep` setzen → DataPrep fittet Preprocessor auf Train-Daten und transformiert Eval-Daten getrennt
2. Ausgabe: zusätzlich `X_eval.parquet`, `T_eval.parquet`, `Y_eval.parquet` im gleichen Output-Verzeichnis
3. `data_files.eval_x_file` etc. auf diese Dateien setzen

Beispiel:

```yaml
data_prep:
  data_path: ["/pfad/zur/input.sas7bdat"]
  delimiter: ","
  chunksize: 300000
  sas_encoding: "ISO-8859-1"

  feature_path: "/pfad/zum/Feature_Dictionary.xlsx"  # Optional: Wenn nicht gesetzt, werden alle Spalten (außer Target/Treatment) als Features verwendet
  info_path: null

  target: "TA_HR_ABSCHLUSS_CNT"              # Einzelne Spalte oder Liste: ["SPALTE_A", "SPALTE_B"] → werden aufsummiert
  treatment: "KONTROLLGRUPPE_FLG"
  target_replacement: {0: 0, 1: 1}
  treatment_replacement: {"J": 0, "N": 1}

  score_name: "HIST_SCORE_WERT"
  score_as_feature: true

  multiple_files_option: "treatment_only"  # "merge" | "treatment_only"
  control_file_index: 0
  balance_treatments: false               # Treatment-Raten pro Datei angleichen (Downsampling)
  eval_file_index: null                   # Index der Datei für "Train Many, Evaluate One" (0-basiert)

  binary_target: true
  fill_na_method: "median"  # "median" | "mean" | "zero" | "mode" | "max" | null

  deduplicate: true                   # Kunden auf 1 Eintrag pro ID reduzieren
  deduplicate_id_column: "PARTNER_ID" # Spalte mit der Kunden-ID

  # Optional: Separater Eval-Datensatz. Der Preprocessor wird nur auf den
  # Train-Daten (data_path) gefittet und auf die Eval-Daten nur transformierend angewendet.
  eval_data_path: null                # z. B. ["/pfad/zur/eval_data.csv"]

  output_path: "data/prep_output"

  log_to_mlflow: true
  mlflow_experiment_name: "data_prep_experiment"
  mlflow_run_name: null  # null = automatisch (z. B. "Datenaufbereitung – roter-falke")
```

**Warum konfigurierbar?**
- DataPrep enthält viele "globale" Stellschrauben (Pfadlisten, Replacement-Maps, Encoding, Output-Pfade),
  die zwischen Use Cases variieren.
- In der Praxis ist es wichtig, dass diese Parameter *nicht* als Code-"Globals" gepflegt werden,

**Deduplizierung:** Wenn `deduplicate: true`, wird der Datensatz direkt nach dem Einlesen auf einen Eintrag pro `deduplicate_id_column` reduziert (erster Eintrag wird behalten). Dies geschieht *vor* der Feature-Reduktion über das Feature-Dictionary, da die ID-Spalte typischerweise kein Feature ist. Anzahl entfernter Duplikate wird geloggt.

**MLflow-Logging:** Bei `log_to_mlflow: true` wird ein eigener MLflow-Run mit zufällig generiertem Namen erzeugt (z. B. „Datenaufbereitung – roter-falke"). Experiment-Name und Run-Name werden als `.mlflow_experiment` und `.mlflow_run_name` im Output-Verzeichnis persistiert. Die Web-UI übernimmt den Experiment-Namen automatisch auf die Konfigurationsseite.

---

## `mlflow` – Experiment‑Tracking

```yaml
mlflow:
  experiment_name: "rubin"
```

- `experiment_name`: Name des MLflow‑Experiments

**Hinweis:**  
MLflow wird in der Analyse genutzt (Training/Evaluation). Die Production‑Pipeline ist bewusst unabhängig vom
Tracking und arbeitet ausschließlich mit Bundles.

---

## `constants` – Reproduzierbarkeit & Parallelisierung

```yaml
constants:
  SEED: 42
  parallel_level: 2  # 1–4
```

- `SEED`: globaler Seed (Sampling, CV‑Splits, Zufallsoperationen)
- `parallel_level`: steuert, wie aggressiv parallelisiert wird (Default: 2)

| Level | Name | Tuning-Trials | Base Learner | CV-Folds | Evaluation | RAM-Bedarf |
|-------|------|--------------|-------------|----------|------------|------------|
| 1 | Minimal | 1 sequentiell | 1 Kern | sequentiell | alle Modelle DRTester | ~1× |
| 2 | Moderat | 1 sequentiell | alle Kerne | sequentiell | alle Modelle DRTester | ~1× |
| 3 | Hoch | 2–4 parallel | Kerne/Workers | 2–4 parallel | Champion + Challenger DRTester | ~2–3× |
| 4 | Maximum | max. parallel | Kerne/Workers | alle parallel | nur Champion DRTester | ~3–5× |

Bei Level 3–4 werden die CPU-Kerne proportional aufgeteilt — sowohl für parallele Optuna-Trials im Tuning als auch für parallele CV-Folds im Training. Keine Übersubskription (z. B. 16 Kerne, Level 4: 8 parallele Trials × 2 Kerne = 16 aktiv). Die Feature-Selektion läuft immer sequentiell, nutzt aber ab Level 2 alle Kerne pro Methode (`n_jobs=-1`). Bei großen Datensätzen (>100k Zeilen) subsampelt CausalForest automatisch stratifiziert.

**Evaluation:** Schnelle Metriken (Qini, AUUC, Policy Value) und CATE-Verteilungs-Histogramme werden immer für alle Modelle berechnet — unabhängig vom Level. Die DRTester-Diagnostik-Plots (Calibration, Qini/TOC mit Bootstrap-CIs) werden bei Level 3 nur für Champion + besten Challenger erzeugt, bei Level 4 nur für den Champion. scikit-uplift-Plots (Qini-Kurve, Uplift-by-Percentile, Treatment-Balance) laufen ebenfalls immer für alle Modelle. Die DRTester-Nuisance-Modelle nutzen leichtere Varianten (n_estimators≤100, cv=3) für schnelleres Fitting.

**Trade-offs bei Level 3–4:**

*Tuning — Speed vs. Hyperparameter-Qualität:*  
Optuna (TPE) lernt aus vorherigen Trial-Ergebnissen, welche Hyperparameter-Regionen vielversprechend sind. Bei parallelen Trials fehlen diese Ergebnisse teilweise. Beispiel mit 30 Trials auf 16 Kernen: Level 2 hat 30 informierte Runden, Level 3 hat ~8 Runden (je 4 Trials), Level 4 hat ~4 Runden (je 8 Trials). Optuna kompensiert mit der „Constant Liar"-Strategie, aber die Exploration ist bei Level 4 weniger gezielt. Bei 30+ Trials ist der Qualitätsverlust gering (~90–95% der sequentiellen Performance). Bei wenigen Trials (<15) ist Level 4 nicht empfehlenswert.

*Training — Speed vs. RAM:*  
Die CV-Fold-Ergebnisse sind mathematisch identisch, egal ob sequentiell oder parallel. Der Trade-off ist rein RAM: Level 3 hält 2–4 EconML-Modelle gleichzeitig im Speicher (~2–3× Baseline), Level 4 hält alle Folds (~3–5× Baseline). Bei großen Datensätzen (>500k Zeilen) kann Level 4 den Kernel killen.

*CatBoost vs. LightGBM bei Level 3–4:*  
CatBoost's Symmetric-Tree-Algorithmus skaliert schlechter mit wenigen Threads pro Fit als LightGBM. Deshalb werden bei CatBoost automatisch weniger parallele Workers gestartet, dafür mit mehr Threads pro Worker. Beispiel 16 Kerne, Level 4: LightGBM 8 Trials × 2 Kerne, CatBoost 4 Trials × 4 Kerne. Die gesamte CPU-Auslastung bleibt gleich, aber CatBoost nutzt die Threads effizienter.

*CausalForestDML bei Level 3–4:*  
CausalForestDML läuft bei den CV-Folds immer sequentiell, auch bei Level 3–4. Der interne GRF (Generalized Random Forest) nutzt joblib-Prozesse für die Baum-Parallelisierung — in Threads würde das zu Deadlocks führen. Stattdessen bekommt jeder sequentielle Fold alle CPU-Kerne. Alle anderen Modelle (NonParamDML, DRLearner, Meta-Learner) profitieren normal von der Fold-Parallelisierung.

**Empfehlung:** Level 3 bietet den besten Gesamtkompromiss. Level 4 nur bei ausreichend RAM und wenn Speed kritisch ist. Level 2 für maximale Tuning-Qualität und minimalen RAM.

**Warum?**  
Ohne festen Seed werden Ergebnisse (insb. bei Optuna und Subsampling) schwer vergleichbar.
Level 2 ist der sichere Default.

---

## `data_processing` – Datenumfang & Validierungsmodus

```yaml
data_processing:
  reduce_memory: true
  df_frac: null
  validate_on: "cross"   # "cross" | "external"
  cross_validation_splits: 5
```

- `reduce_memory`: Datentypen automatisch downcasten (float64 → float32, int64 → int16/int32 etc.). Spart ca. 40–60% Arbeitsspeicher bei minimalem Präzisionsverlust. Wird sowohl in der DataPrep- als auch in der Analyse-Pipeline angewendet.
- `df_frac` (optional): Anteil der Daten für schnelle Tests (z. B. `0.1`)
- `validate_on`:
  - `"cross"`: Cross‑Predictions (robust, Standard). Kombinierbar mit einer Eval-Maske (`data_files.eval_mask_file`) für „Train Many, Evaluate One".
  - `"external"`: Training auf `data_files` (x/t/y_file), Evaluation auf separatem Datensatz (`eval_x/t/y_file`). Erfordert, dass die eval-Dateien in `data_files` angegeben sind. Kein Data-Leakage — der Preprocessor wird nur auf den Trainingsdaten gefittet.
- `cross_validation_splits`: Anzahl der Splits für Cross-Predictions (Out-of-Fold). **Zentrale Fold-Anzahl** – wird auch für das Base-Learner-Tuning (`tuning.cv_splits`), das Final-Model-Tuning und EconML-CausalForest-Tune verwendet. Einheitliche Folds gewährleisten konsistente Trainingsset-Größen über alle Schritte. Standard: 5. Wird auch bei `validate_on: "external"` benötigt, da Cross-Predictions für Tuning und interne Evaluation auf den Trainingsdaten laufen.

**Warum?**  
Für Entwicklung/Iteration wird häufig mit Teilmengen gearbeitet, während finale Runs auf dem vollen Datensatz
laufen sollen. Der Validierungsmodus ist zudem entscheidend für die Stabilität der Uplift‑Kennzahlen.

---


## `treatment` – Treatment-Typ (Binary vs. Multi)

```yaml
treatment:
  type: binary        # "binary" | "multi"
  reference_group: 0  # Baseline/Control-Gruppe
```

- `type`: Steuert, ob die Pipeline für binäres Treatment (T in {0,1}) oder Multi-Treatment (T in {0,1,...,K-1}) laeuft.
  Bei `"multi"` werden SLearner, TLearner und XLearner automatisch blockiert, da diese nur Binary Treatment unterstützen.
- `reference_group`: Welche Treatment-Gruppe als Control/Baseline dient (typisch 0).

**Wichtig:** Bei `type: "multi"` ändert sich die Struktur der Predictions und Evaluationsmetriken:
- Statt einer CATE-Spalte gibt es K-1 Spalten (eine pro Treatment-Arm vs. Control).
- Statt eines skalaren Qini-Koeffizienten gibt es pro-Arm-Metriken plus einen globalen Policy-Value.
- Die Champion-Auswahl sollte bei MT auf `metric: policy_value` umgestellt werden.

---


## `bundle` – Bundle-Export für Production

```yaml
bundle:
  enabled: false
  base_dir: "bundles"
  bundle_id: null
  include_challengers: true
  log_to_mlflow: true
```

- `enabled`: Export am Ende von `run_analysis.py` aktivieren/deaktivieren
- `base_dir`: Zielordner, unter dem das Bundle-Verzeichnis angelegt wird
- `bundle_id`: optionaler fixer Name des Bundle-Verzeichnisses; `null` erzeugt einen Zeitstempel-Namen
- `include_challengers`: `true` exportiert alle trainierten Modelle, `false` nur den Champion
- `log_to_mlflow`: zusätzliches Logging des erzeugten Bundle-Verzeichnisses als MLflow-Artefakt

**CLI-Overrides:**
- `--export-bundle` erzwingt Export
- `--no-export-bundle` deaktiviert Export
- `--bundle-dir` überschreibt `base_dir`
- `--bundle-id` überschreibt `bundle_id`

## `feature_selection` – optionale Feature‑Filter

```yaml
feature_selection:
  enabled: true
  methods: [lgbm_importance, causal_forest]   # Union der Top-Features
  top_pct: 15.0                                # Top-X% pro Methode
  max_features: null                           # Absolute Obergrenze nach Union
  correlation_threshold: 0.9
```

- `enabled`: Schaltet Feature‑Selektion an/aus.
- `methods`: Liste der Importance-Methoden. Mehrere können kombiniert werden – die Ergebnisse werden per Union zusammengeführt.
  - `"lgbm_importance"`: LightGBM-Regressor auf Outcome (Y), Gain-Importance. Schnell, erfasst prädiktive Relevanz.
  - `"lgbm_permutation"`: LightGBM-Regressor auf Outcome (Y), Permutation-Importance. Robuster als Gain (kein Split-Bias), aber rechenintensiver.
  - `"causal_forest"`: EconML GRF CausalForest Feature-Importances. Direkte GRF-Implementierung ohne Nuisance-Fitting; erfasst kausale Relevanz (welche Features die Heterogenität des Treatment-Effekts treiben). **Kann keine fehlenden Werte verarbeiten** – wird bei NaN in den Daten automatisch übersprungen.
  - `"none"`: Keine Importance-Filterung.
- `top_pct`: Prozent der Features, die pro Methode behalten werden. Bei Union: aus jeder Methode werden die Top-X% behalten, dann vereinigt. Beispiel: 15.0 bei 100 Features → je 15 Features pro Methode, Union kann bis zu 30 enthalten.
- `max_features` (optional): Absolute Obergrenze nach der Union. Bei Überschreitung wird nach mittlerer Rank-Position über alle Methoden sortiert.
- `correlation_threshold`: Entfernt stark korrelierte numerische Features (Pearson + Spearman). Wert von 0.9 bedeutet: ab |corr| > 0.9 wird eine der beiden Spalten entfernt.

**Warum Union?**
Die prädiktive Relevanz (Outcome-Importance) und die kausale Relevanz (CATE-Heterogenität) überlappen oft nur teilweise. Ein Feature kann stark prädiktiv für das Outcome sein, aber keinen heterogenen Treatment-Effekt haben (und umgekehrt). Durch die Union werden beide Perspektiven berücksichtigt.

**Kategorische Features:** Die LightGBM-Importance-Berechnung nutzt automatisch native kategoriale Splits (via `categorical_feature`). Dadurch werden kategorische Features korrekt bewertet und nicht systematisch unterbewertet.

**Parallelisierung:** Die Importance-Methoden laufen immer sequentiell, aber jede Methode bekommt alle CPU-Kerne (`n_jobs=-1`). GRF nutzt bei großen Datensätzen (>100k Zeilen) automatisch stratifiziertes Subsampling für schnelle Berechnung. Bei Level 1 wird nur ein Kern pro Fit verwendet (`n_jobs=1`).

---

## `models` – welche kausalen Learner trainiert werden

```yaml
models:
  models_to_train:
    - "SLearner"
    - "TLearner"
    - "XLearner"
    - "DRLearner"
    - "NonParamDML"        # DML-Variante (nichtlinear, frei wählbares Final-Modell)
    - "ParamDML"           # DML-Variante (linear, nutzt EconMLs LinearDML)
    - "CausalForestDML"    # DML-Residualisierung (model_y/model_t) + Causal Forest als letzte Stufe
```

Nur diese Modellnamen sind gültig. Allgemeine Kürzel wie `"DML"` sind nicht erlaubt, damit Konfiguration und Registry eindeutig bleiben.

**Hinweis zu `ParamDML`:**  
`ParamDML` nutzt intern EconMLs `LinearDML`. Das bedeutet, das Final-Modell nimmt eine **lineare CATE-Struktur** an
(CATE(X) = X · β). Für nichtlineare CATE-Schätzung eignet sich `NonParamDML` besser, da dort das Final-Modell
frei wählbar ist (z. B. LightGBM-Regressor).

**Hinweis zu Binary Treatment / Binary Outcome:**  
Alle DML-Modelle (`NonParamDML`, `ParamDML`, `CausalForestDML`) sowie `DRLearner` werden in rubin mit
`discrete_treatment=True` und `discrete_outcome=True` erstellt. Das stellt sicher, dass EconML intern
die korrekte Cross-Fitting-Logik für binäre Variablen verwendet (Klassifikatoren für die Nuisance-Modelle
`model_y`, `model_t` und `model_propensity`). Die Meta-Learner (`SLearner`, `TLearner`, `XLearner`) sowie
`DRLearner.model_regression` verwenden hingegen **Regressoren** als Outcome-Modelle, da EconML intern
`model.predict()` aufruft – ein Classifier gibt dort nur 0/1 (Klassen-Labels) zurück, ein Regressor
liefert E[Y|X] ∈ [0,1] (kontinuierliche Wahrscheinlichkeit), was für die CATE-Berechnung benötigt wird.

**Hinweis zu fehlenden Werten:**  
Alle Modelle außer `CausalForestDML` können mit fehlenden Werten in den Features umgehen, da sie
LightGBM oder CatBoost als Base Learner nutzen. `CausalForestDML` basiert intern auf einem
GRF (Generalized Random Forest), der keine NaN-Werte unterstützt. Enthält der Datensatz fehlende
Werte, wird `CausalForestDML` automatisch übersprungen und ein entsprechender Hinweis geloggt.


---

## `base_learner` – Basismodell (LightGBM oder CatBoost)

```yaml
base_learner:
  type: "lgbm"          # "lgbm" oder "catboost"
  fixed_params: {}
```

- `type`: Auswahl des Base Learners
- `fixed_params`: Parameter, die direkt für alle Nuisance-Modelle (Outcome, Propensity) gesetzt werden. Relevant wenn `tuning.enabled: false` – dann werden diese statt der getunedn Parameter verwendet. Wenn Tuning aktiv ist, werden `fixed_params` ignoriert und die Optuna-Ergebnisse genutzt.

**Praxisempfehlungen:**
- LightGBM: schnell, sehr gut für viele numerische Features
- CatBoost: robust, oft stark bei kategorischen Features

**Wichtig:** Beim Wechsel des Base Learners ändern sich die verfügbaren Parameter-Namen (z.B. `n_estimators` bei LightGBM vs. `iterations` bei CatBoost). Die `fixed_params` und `final_model_tuning.fixed_params` sollten dann ebenfalls angepasst werden.

**Kategorische Features:** EconML konvertiert X intern zu numpy-Arrays, wodurch pandas `category`-Dtypes verloren gehen. rubin löst dieses Problem automatisch: Vor dem Training werden die `fit()`-Methoden von LightGBM/CatBoost so gepatcht, dass `categorical_feature` (LightGBM) bzw. `cat_features` (CatBoost) bei jedem internen Aufruf mitgegeben wird. So nutzen die Base Learner native kategoriale Splits, selbst wenn EconML die Daten als numpy übergibt.

**Internes Cross-Fitting:** Alle DML-Modelle (NonParamDML, ParamDML, CausalForestDML) und DRLearner verwenden intern `cv=5` für die Nuisance-Residualisierung. Das ist höher als der EconML-Default (`cv=2`) und liefert stabilere Residuals, weil jedes Nuisance-Modell auf 80% statt nur 50% der Daten trainiert wird.

---

## `causal_forest` – Parameter für `CausalForestDML`

```yaml
causal_forest:
  forest_fixed_params: {}
  use_econml_tune: false
  econml_tune_params: "auto"
  tune_max_rows: null
```

`CausalForestDML` kombiniert **DML‑Residualisierung** mit einem **Causal Forest** als letzter Stufe.
Damit gibt es zwei Ebenen, die man konfigurieren kann:

1) **Nuisance‑Modelle** (`model_y`, `model_t`) – das sind Base Learner wie bei anderen DML‑Verfahren.
   Diese werden über `base_learner` (und ggf. `tuning`) gesteuert.

2) **Wald‑Parameter** der finalen Forest‑Stufe – diese werden über `causal_forest` gesteuert.

Felder:

- `forest_fixed_params`: Feste Parameter, die immer an die Forest‑Stufe übergeben werden
  (z. B. `honest`, `n_jobs`, `min_samples_leaf`).
- `use_econml_tune`: Wenn `true`, wird vor dem finalen Training einmal die EconML‑Methode
  `tune(...)` aufgerufen. Diese wählt zentrale Wald‑Parameter anhand eines Out‑of‑Sample‑R‑Scores
  und setzt sie am Estimator. Danach folgt ein reguläres `fit(...)`.
- `econml_tune_params`: Parameter‑Grid für `tune(...)`. Standard ist `"auto"`.
- `tune_max_rows`: Optionales Limit für die Anzahl Zeilen, die in `tune(...)` verwendet werden.
  Das ist ein Laufzeit‑Regler für sehr große Daten.

Wichtig:

- **Optuna** optimiert in rubin weiterhin die **Base Learner** (also `model_y`/`model_t`) – auch beim
  `CausalForestDML`.
- Die **Wald‑Parameter** werden (falls gewünscht) über **EconML `tune(...)`** bestimmt, nicht über Optuna.

---

## `tuning` – Optuna‑Tuning der Base Learner

```yaml
tuning:
  enabled: true
  n_trials: 50
  timeout_seconds: null
  cv_splits: 5
  single_fold: false
  metric: "log_loss"
  per_learner: false
  per_role: false
  max_tuning_rows: 200000
  optuna_seed: 42
  storage_path: null
  study_name_prefix: "baselearner"
  reuse_study_if_exists: true
```

**Kernidee:**  
Nicht die kausalen Learner selbst werden getunt, sondern die Base Learner, die intern verwendet werden
(Outcome‑Modelle, Propensity‑Modelle usw.). Optional werden getrennte Parameter‑Sets optimiert:

- Standardfall: identische Tuning-Aufgaben werden task-basiert zusammengefasst
- `per_learner=true`: separates Set je kausalem Verfahren
- `per_role=true`: separates Set je Rolle innerhalb eines Verfahrens (z. B. `model_y` vs. `model_t`)

**CV-Folds (`cv_splits`):**  
Sollte identisch mit `data_processing.cross_validation_splits` sein (Empfehlung: denselben Wert verwenden). Unterschiedliche Fold-Zahlen zwischen Validierung und Tuning führen zu unterschiedlichen Trainingsset-Größen, was die Übertragbarkeit der Tuning-Ergebnisse verschlechtert.

**Single-Fold-Tuning (`single_fold`):**  
Bei `single_fold: true` wird jeder Optuna-Trial nur auf **einem** zufällig gewählten Fold evaluiert statt auf allen K Folds. Das reduziert die Modell-Fits pro Trial von K auf 1 – bei 5 Folds also 5× schneller. Optuna (TPE) ist robust gegenüber verrauschteren Metriken, daher ist der Tradeoff für explorative Analysen oder große Datensätze sinnvoll.

**Persistenz (`storage_path`)**  
Wenn `storage_path` gesetzt ist, wird die Optuna‑Study in SQLite persistiert (Fortsetzen/Analyse möglich).

Die Task-Signatur berücksichtigt nicht nur den Learner-Typ, sondern auch die tatsächliche interne Lernaufgabe, u. a.:

- Base-Learner-Familie
- Objective-Familie
- Estimator-Task (Klassifikation/Regression)
- Datengrundlage bzw. Sample-Scope
- Nutzung des Treatment-Features
- Zieltyp

Dadurch werden nur wirklich gleiche Base-Learner-Aufgaben zusammengelegt.

**Hinweis zu `CausalForestDML`:**
`CausalForestDML` nutzt *zwei* Nuisance‑Modelle (`model_y`, `model_t`), die normale Base Learner sind.
Zusätzlich besitzt es eine Forest‑Stufe mit eigenen Wald‑Parametern.

- Das Optuna‑Tuning (`tuning`) betrifft weiterhin `model_y`/`model_t`.
- Die Wald‑Parameter können optional über die EconML‑Methode `tune(...)` bestimmt werden
  (siehe Abschnitt `causal_forest`).

---

## `final_model_tuning` – Optuna‑Tuning des Final‑Modells (R‑Loss / R‑Score)

```yaml
final_model_tuning:
  enabled: false
  n_trials: 30
  cv_splits: 5
  models: null
  single_fold: false
  stability_penalty: 0.3
  max_tuning_rows: 200000
  method: "rscorer"
  fixed_params: {}
```

Wofür ist das?

- Relevant für Modelle, die ein frei wählbares Final‑Modell besitzen (z. B. **NonParamDML**, **DRLearner**).
- Bewertet wird die Final‑Stage‑Güte über eine **Residual‑auf‑Residual** Logik.
  Das entspricht dem R‑Loss/R‑Score‑Gedanken in EconML.

Wichtige Regeln in der Pipeline:

- Das Tuning findet **nur einmal pro Run** statt.
- Um eine saubere Trennung zur Cross‑Prediction zu gewährleisten, wird auf der **Trainingsmenge des ersten
  Cross‑Prediction‑Folds** getunt.
- Die gefundenen Hyperparameter werden anschließend für alle weiteren Folds **wiederverwendet** ("Locking").

Parameter:

- `enabled`: Schaltet das Final‑Model‑Tuning an/aus.
- `n_trials`: Anzahl Optuna‑Trials.
- `cv_splits`: Interne CV‑Splits innerhalb des Tuning‑Datensatzes. Empfehlung: gleicher Wert wie `data_processing.cross_validation_splits`.
- `models`: Liste der Modelle, die per FMT optimiert werden sollen (z.B. `[NonParamDML]`). Bei `null` werden alle FMT-fähigen Modelle getuned, bei einer expliziten Liste nur die genannten. Nicht getunte Modelle verwenden die `fixed_params`.
- `single_fold`: Bei `true` wird DRLearner nur auf 1 Fold pro Trial evaluiert statt K. Reduziert die Fits von K×Trials auf 1×Trials. NonParamDML profitiert nicht, da RScorer ohnehin nur 1 Fit pro Trial benötigt.
- `stability_penalty`: Bestraft instabile CATE-Schätzungen mit hoher Streuung relativ zum Median-Effekt. 0.0 = reiner R-Score. **0.3 = Default** (moderate Regularisierung). 0.5 = stärker. 1.0+ = starke Stabilität (CATEs werden konservativ, ähnlich CausalForestDML). Formel: `score = R_score − λ · log(1 + CV)`, wobei CV = std(CATE) / |median(CATE)|.
- `max_tuning_rows`: Laufzeitregler für sehr große Datensätze.
- `method`: aktuell ausschließlich `"rscorer"`.
- `fixed_params`: Feste Hyperparameter für das Final-Modell (`model_final`). Werden verwendet, wenn FMT deaktiviert ist oder wenn ein Modell nicht in `models` steht.

**Wichtig:** Das Final-Modell (`model_final`) erhält niemals die getunten Nuisance-Parameter (z. B. aus dem Base-Learner-Tuning für `model_y`/`model_t`). Ohne FMT verwendet `model_final` ausschließlich LightGBM/CatBoost-Standardwerte (bzw. `base_learner.fixed_params`). Damit wird verhindert, dass Classifier-optimierte Parameter (hoher `min_split_gain`, viele `min_child_samples`) den CATE-Baum zu einem Intercept kollabieren lassen.

---

## `learner_data_usage` – Datenmengen‑Heuristiken fürs Tuning

```yaml
learner_data_usage:
  s_learner_frac: 1.0
  t_learner_group_frac: 1.0
  x_learner_group_frac: 1.0
  dml_frac: 1.0
  dr_learner_frac: 1.0
```

Diese Werte steuern, wie groß die Tuning‑Stichprobe im Verhältnis zur „effektiven“ Trainingsmenge je Verfahren ist.

Beispiele:
- S‑Learner: ein Modell auf allen Daten → `s_learner_frac` nahe 1.0
- T‑Learner: zwei Modelle auf Teilmengen → `t_learner_group_frac` steuert den Anteil pro Gruppe  
  (praktisch relevant für Regularisierung/Komplexität)

---

## `selection` – Champion‑Auswahl (Model Registry)

```yaml
selection:
  metric: "qini"
  higher_is_better: true
  refit_champion_on_full_data: true
  manual_champion: null
```

Beim Bundle‑Export wird eine Registry geschrieben, die alle Modelle inkl. Kennzahlen enthält und einen
**Champion** festlegt. In der Produktion wird standardmäßig der Champion verwendet.

- `metric`: Kennzahl für die automatische Champion-Auswahl.
  Bei Binary Treatment: `qini`, `auuc`, `uplift_at_10pct`, `uplift_at_20pct` oder `policy_value_treat_positive`.
  Bei Multi-Treatment: `policy_value` (empfohlen), `policy_value_treat_positive_T1`, oder arm-spezifisch `qini_T1`, `qini_T2`, etc.
- `higher_is_better`: Richtung der Kennzahl
- `refit_champion_on_full_data`: refittet das ausgewählte Champion-Modell vor dem Export auf allen im Run verfügbaren Daten
- `manual_champion`: optionaler Override; falls gesetzt, wird dieses Modell unabhängig von der Kennzahl Champion

**Warum?**  
Das erleichtert den Übergang von „Analyse mit vielen Kandidaten“ zu „stabilem Produktionsmodell“.

---

## `shap_values` – Explainability‑Voreinstellungen

```yaml
shap_values:
  calculate_shap_values: true
  shap_calculation_models: [NonParamDML]
  n_shap_values: 5000
  top_n_features: 20
  num_bins: 10
```

Diese Einstellungen steuern die Explainability-Berechnung in der Analyse-Pipeline. Bei `calculate_shap_values: true` wird nach dem Bundle-Export automatisch ein Explainability-Schritt ausgeführt (dreistufiger Fallback: EconML SHAP-Plots → generische SHAP → Permutation-Importance). Alle Artefakte werden als MLflow-Artefakte im selben Run geloggt. Zusätzlich weiterhin als CLI-Runner verfügbar: `run_explain.py` für Ad-hoc-Analysen auf Bundle-Basis.

- `calculate_shap_values`: Schaltet den Explainability-Schritt in der Analyse-Pipeline an/aus.
- `shap_calculation_models`: Liste der Modelle, für die Importance berechnet wird. Leer = nur Champion.
- `n_shap_values`: maximale Stichprobe für SHAP/Permutation (Performance)
- `top_n_features`: wie viele Features im Report/Plot ausgegeben werden
- `num_bins`: Standard‑Segmentierung (z. B. 10 = Dezile) für Dependency-Plots

**Hinweis:**  
Explainability ist bewusst als separater Schritt umgesetzt (kein Pflichtbestandteil eines Trainingslaufs).

---

## `optional_output` – lokale Ausgabe (zusätzlich zu MLflow)

```yaml
optional_output:
  output_dir: null
  save_predictions: false
  predictions_format: "parquet"
  max_prediction_rows: null
```

- `output_dir`: wenn gesetzt, werden ausgewählte Artefakte lokal geschrieben
- `save_predictions`: speichert Cross‑Predictions pro Modell (kann groß werden)
- `predictions_format`: `"parquet"` oder `"csv"`
- `max_prediction_rows`: optionales Limit, um I/O zu begrenzen

---

## `surrogate_tree` – Surrogate-Einzelbaum (Teacher-Learner)

```yaml
surrogate_tree:
  enabled: false
  min_samples_leaf: 50
  num_leaves: 31
  max_depth: null
```

Aktiviert einen Einzelbaum des konfigurierten Base-Learners (LightGBM/CatBoost mit `n_estimators=1`), der die CATE-Vorhersagen eines Teacher-Modells nachlernt (Teacher-Learner-Prinzip). rubin trainiert bis zu zwei Surrogates pro Analyselauf:

| Surrogate | Teacher | Wann trainiert | DRTester-Plots |
|---|---|---|---|
| `SurrogateTree_CausalForestDML` | CausalForestDML | Immer, wenn CausalForestDML in den Modellen ist | Wenn CausalForestDML auf Rang 1 oder 2 steht |
| `SurrogateTree` | Champion | Wenn `enabled: true` und Champion ≠ CausalForestDML | Nein (Basismetriken + sklift-Plots) |

Wenn CausalForestDML selbst Champion ist, wird der CF-Surrogate automatisch als `SurrogateTree` referenziert — kein redundantes zweites Training.

- `enabled`: Aktiviert den Champion-Surrogate-Einzelbaum.
- `min_samples_leaf`: Mindestanzahl an Beobachtungen pro Blatt. Wird auf `min_child_samples` (LightGBM) bzw. `min_data_in_leaf` (CatBoost) gemappt.
- `num_leaves`: Maximale Anzahl Blätter (nur LightGBM). Steuert die Baumkomplexität direkt über leaf-wise Growth.
- `max_depth`: Maximale Baumtiefe. `null` bedeutet keine Begrenzung bei LightGBM (`-1`), bei CatBoost wird `6` als Default verwendet.

**Ablauf in der Analyse-Pipeline:** Der CausalForestDML-Surrogate wird automatisch trainiert, wenn CausalForestDML unter den Modellen ist — unabhängig von `enabled`. Der Champion-Surrogate wird nach der Evaluation des Champions trainiert, wenn `enabled: true`. Beide werden mit denselben Uplift-Metriken evaluiert. Im Bundle-Export werden beide Surrogates (falls vorhanden) mit eigenen Registry-Einträgen exportiert.

**Production-Scoring:** Im Bundle ist der Surrogate als `SurrogateTree` verfügbar. In der Production-Pipeline kann er über `score_surrogate(X)` oder `score(X, model_names=["SurrogateTree"])` angesprochen werden. Über die CLI: `pixi run score -- --bundle ... --x ... --use-surrogate` (oder: `python run_production.py --bundle ... --x ... --use-surrogate`).

---

# Wo wird die Konfiguration „global“ wirksam?

- **Analyse‑Pipeline** (`run_analysis.py`): nutzt *alle* oben beschriebenen Bereiche.
- **Bundle‑Export**: legt `config_snapshot.yml` ab (später für Production/Explainability nutzbar).
- **Production‑Pipeline** (`run_production.py`): liest primär Artefakte aus dem Bundle (Preprocessor/Modelle/Registry).
- **Explainability** (`run_explain.py`): nutzt CLI‑Parameter, übernimmt aber Voreinstellungen aus `config_snapshot.yml`, falls vorhanden.

Damit ist die Pipeline global über `config.yml` steuerbar, während Run‑spezifische Aspekte über CLI‑Parameter
gesetzt werden können.


## `shap_values`

```yaml
shap_values:
  calculate_shap_values: true
  shap_calculation_models: []
  n_shap_values: 10000
  top_n_features: 20
  num_bins: 10
```

- `calculate_shap_values`: Aktiviert die Feature-Importance-Berechnung (SHAP oder Permutation).
- `shap_calculation_models`: Modelle für Importance. Leer = nur Champion, explizit z.B. `[NonParamDML, DRLearner]`.
- `num_bins` steuert die Binning-Tiefe für CATE-Profil- und SHAP-PDP-Plots.

## `segment_analysis`

```yaml
segment_analysis:
  enabled: true
  quantiles: 10
  top_n_features: 8
  max_bins: 6
  max_categories: 15
```

Diese Sektion steuert die Segmentauswertung in `run_explain.py`.

- `enabled`: Wenn `false`, werden keine Segment-Reports erzeugt (nur Importance-Artefakte).
- `quantiles`: Anzahl Quantile für den Score-basierten Segment-Report.
- `top_n_features`: Anzahl Features, für die featurebezogene Segmente ausgewertet werden.
- `max_bins`: Maximale Bin-Anzahl für numerische Features in der Segmenttabelle.
- `max_categories`: Maximale Kategorie-Anzahl für kategoriale Features (seltene Kategorien werden zusammengefasst).

Neben dem globalen Score-Report werden auch featurebezogene Segmenttabellen geschrieben, damit sichtbar wird, welche Kundensegmente besonders stark oder schwach reagieren.