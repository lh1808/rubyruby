# Evaluation

In der Analyse-Pipeline werden drei Evaluationsphasen unterschieden:

### Phase 1: Schnelle Metriken + CATE-Verteilung (alle Modelle)

**Sortierungsmetriken** (Qini, AUUC, Uplift@k, Policy Value) werden für **alle** Modelle berechnet — reines NumPy, <1s pro Modell. Grundlage für Champion-Selektion.

**CATE-Verteilungs-Plots** werden ebenfalls für alle Modelle erzeugt (~0.5s pro Modell). Zeigen Training-Predictions und Cross-Validated-Predictions nebeneinander als Histogramm. Dienen der visuellen Plausibilitätsprüfung: stark konzentrierte Verteilungen nahe Null deuten auf wenig Heterogenität hin, breite Verteilungen auf differenzierte Effektvorhersagen. Bei MT wird ein Plot pro Treatment-Arm erzeugt.

### Phase 2: DRTester-Diagnostik (Level-abhängig)

**Diagnose-Plots** (EconML DRTester) werden Level-abhängig erstellt: Level 1–2 alle Modelle, Level 3 Champion + Challenger, Level 4 nur Champion. Die DRTester-Nuisance-Modelle nutzen leichtere Varianten (n_estimators≤100, cv=3) für ~6-7× schnelleres Fitting bei minimaler Qualitätseinbuße. Bei Multi-Treatment werden die Nuisance-Fits pro Arm bei Level 3–4 parallel ausgeführt.

### Phase 3: scikit-uplift-Plots (alle Modelle)

Qini-Kurve, Uplift-by-Percentile, Treatment-Balance — immer für **alle** Modelle, da schnell (~2-5s).

## DRTester (EconML)

Der `DRTester` aus EconML liefert u. a.:

- **BLP-Summary** — Best-Linear-Predictor-Test
- **Calibration Plot** — Kalibrierungs-Check
- **Qini Plot** mit Bootstrap-Konfidenzintervallen
- **TOC Plot** (Targeting Operating Characteristic)
- **Policy Values** mit Konfidenzintervallen

DRTester-Plots werden erzeugt, wenn Train-Daten (`Train_*`-Spalten) vorhanden sind. Im Cross-Modus erzeugt `train_and_crosspredict` sowohl Out-of-Fold-Predictions (`Predictions_*`) als auch In-Sample-Predictions (`Train_*`) auf dem gesamten Datensatz. Damit steht `has_train=True` und die Nuisance-Modelle werden mit `X_train=X` gefittet — alle DRTester-Tests (BLP, Calibration, Qini, TOC) laufen.

In rubin werden dafür die Cross-Predictions genutzt. Die `Train_*`-Vorhersage ist **keine** Performance-Schätzung, sondern dient für konsistente DRTester-Diagnosen.

Die Nuisance-Modelle werden **einmal** gefittet und für alle kausalen Modelle wiederverwendet (siehe „DRTester-Nuisance" weiter unten).

## scikit-uplift (sklift)

Die Pipeline erzeugt auch Plots aus scikit-uplift:

- Qini Curve (`plot_qini_curve`)
- Uplift-by-Percentile (`plot_uplift_by_percentile`)
- Treatment Balance (`plot_treatment_balance_curve`)

### Extraktions-Patterns (versionsspezifisch)

Die drei sklift-Funktionen haben **unterschiedliche Signaturen und Rückgabetypen**. Die installierte Version (≥0.5) unterstützt den `ax=`-Parameter **nur** bei `plot_qini_curve`, nicht bei den anderen beiden. Das führte zu schwer zu diagnostizierenden Fehlern der Form:

```
plot_uplift_by_percentile() got an unexpected keyword argument 'ax'
plot_treatment_balance_curve() got an unexpected keyword argument 'ax'
```

Die korrekte Handhabung je Funktion:

| Funktion | `ax=` Support | Extraktions-Pattern | Rückgabetyp |
|---|---|---|---|
| `plot_qini_curve` | Ja | `fig, ax = plt.subplots(); plot_qini_curve(..., ax=ax); sk_qini = fig` | Axes |
| `plot_uplift_by_percentile` | **Nein** | `result = plot_uplift_by_percentile(...); sk_pct = result[0].get_figure()` | Tuple (BarContainer, …) |
| `plot_treatment_balance_curve` | **Nein** | `result = plot_treatment_balance_curve(...); sk_tb = result.get_figure()` | Axes |

Dieses Pattern wird konsistent in `generate_sklift_plots()` verwendet. Die Funktion ist der einzige Ort, an dem sklift-Plots erzeugt werden — alle Aufrufstellen (Phase 3, Surrogate, Historischer Score) rufen `generate_sklift_plots()` auf.

### Warum nicht einheitlich `ax=` für alle?

Es wäre naheliegend, alle drei Funktionen mit vorab erstelltem `fig, ax = plt.subplots()` und `ax=ax` aufzurufen. Aber:

1. Die installierten sklift-Versionen (getestet: 0.5.x) haben `ax=` nur bei `plot_qini_curve` in der Signatur. Bei den anderen führt es zu einem `TypeError`.
2. Verschiedene sklift-Versionen können sich darin unterscheiden. Deshalb prüft der Code nicht die Version, sondern nutzt **das Pattern, das für die installierte Version funktioniert**.
3. Ein einheitliches `ax=`-Pattern würde bei sklift-Updates wieder brechen, wenn sich die Signatur ändert.

### Figure-Memory-Management

Jeder Plot, der erfolgreich erzeugt wird, durchläuft diesen Lebenszyklus:

1. Erzeugung in `generate_sklift_plots()`
2. `recolor_figure()` für das rubin-Farbschema
3. `self._report.add_plot()` konvertiert zu Base64 für den HTML-Report
4. `mlflow.log_figure()` loggt als PNG-Artefakt
5. `plt.close(fig)` gibt den Speicher frei

Wenn `plot_qini_curve` fehlschlägt, wird die vorab erstellte Figure explizit geschlossen (`plt.close(fig_qini)` im `except`-Block), um Memory-Leaks zu vermeiden. Bei `plot_uplift_by_percentile` und `plot_treatment_balance_curve` wird keine vorab-Figure erstellt, daher entsteht bei Fehlern kein Leak.

Ohne diese explizite Freigabe entsteht bei >20 Figures eine Matplotlib-Warnung: `More than 20 figures have been opened. Consider using matplotlib.pyplot.close()`. Bei 7 Modellen × 3 Plots × 2 Phasen (Eval + Surrogate) können leicht >40 offene Figures entstehen.

### Native Fallbacks

Wenn sklift nicht verfügbar oder der Aufruf fehlschlägt, greifen native Implementierungen:

- `_native_uplift_by_percentile()` — NumPy-basiertes Dezil-Balkendiagramm
- `_native_treatment_balance()` — Sliding-Window Treatment-Balance-Kurve

Diese erzeugen visuell ähnliche Plots ohne sklift-Abhängigkeit. Die Qini-Kurve hat keinen nativen Fallback, da sie über `compute_qini_curve()` für den Custom-Qini-Plot separat implementiert ist.

Nach der Erzeugung wird `recolor_figure()` auf alle Plots angewendet (sklift und native), um das rubin-Farbschema zu übertragen.

Wenn ein historischer Score `S` vorliegt, werden die Plots auch für diesen
Score direkt gegen den historischen Score gegenüberstellt.

## Abhängigkeiten

Für die vollständige Plot-Ausgabe werden benötigt:

- `econml` (bereits Kernabhängigkeit)
- `scikit-uplift` (Import-Pfad `sklift`)
- `scipy` (für Interpolation im Custom-Qini-Plot)

Alle Pakete werden automatisch über `pixi install` installiert (empfohlen). Alternativ sind sie auch in `requirements.txt` enthalten.


## Policy-Value-Vergleich gegen einen Referenzscore (z. B. historischer Score)

Wenn ein historischer Score (`data_files.s_file`) vorliegt, wird dieser in der Analyse zusätzlich
wie ein weiteres „Modell“ evaluiert. Neben Qini/AUUC werden auch Policy Values (inkl.
Konfidenzintervall) über den `DRTester` erzeugt. Die Ergebnisse werden als Vergleichsplots
direkt gegen den historischen Score gestellt:

- `policy_compare__<modell>_vs_<historical_name>.png`

Inhalt des Plots:
- Policy Values des Modells (mit Konfidenzband)
- Policy Values des Referenzscores (mit Konfidenzband)
- Differenzkurve (Modell minus Referenz)

Hinweis: Die Vergleichsplots werden nur dann erzeugt, wenn die DRTester-Auswertung sowohl für
die kausalen Modelle als auch für den historischen Score erfolgreich berechnet wurde.

## Multi-Treatment-Evaluation

Bei Multi-Treatment (T ∈ {0, 1, …, K-1}) wird die Evaluation automatisch angepasst:

**Ansatz A – Pro Treatment-Arm:**
Für jeden Treatment-Arm k wird separat eine Uplift-Kurve berechnet, indem nur die Beobachtungen
mit T ∈ {0, k} betrachtet werden. Daraus ergeben sich pro-Arm-Kennzahlen:
`qini_T1`, `qini_T2`, `auuc_T1`, `auuc_T2`, etc.
Zusätzlich wird ein `policy_value_treat_positive_T{k}` berechnet – das ist das Analogon zu
`policy_value_treat_positive` bei Binary Treatment: Unter allen Beobachtungen mit positivem
geschätztem Effekt (CATE_k > 0) wird die Differenz der Outcome-Raten zwischen Arm k und
Control berechnet.

**Ansatz B – Policy Value (IPW):**
Die optimale Zuweisungspolicy π*(X) = argmax_k τ_k(X) wird über einen IPW-Schätzer bewertet.
Der resultierende `policy_value` misst den erwarteten inkrementellen Nutzen der optimalen
Zuweisungsentscheidung gegenüber der Baseline (Control).

Die Propensity-Gewichte werden standardmäßig als empirische Verteilung (d. h. Anteil pro
Gruppe) geschätzt – korrekt bei Randomisierung. Für observationale Daten kann eine geschätzte
Propensity als Parameter übergeben werden.

**Treatment-Verteilung:**
Zusätzlich wird dokumentiert, welchem Anteil der Population jedes Treatment zugewiesen wird
(`best_treatment_distribution`). Dies hilft bei der fachlichen Einordnung der Ergebnisse.

**DRTester-Plots:**
Bei MT werden DRTester-Plots pro Treatment-Arm erzeugt. Dafür werden die Daten auf
T ∈ {0, k} gefiltert, sodass der DRTester binäre Treatment-Daten sieht.

**Historischer Score-Vergleich:**
Der Vergleich gegen einen historischen Score ist nur bei Binary Treatment verfügbar, da ein
einzelner Score keine Multi-Treatment-Zuweisung abbilden kann.

**Beispiel-Ausgabe (`uplift_eval_summary.json` bei MT mit K=3):**

```json
{
  "NonParamDML": {
    "qini_T1": 0.0312,
    "auuc_T1": 0.0187,
    "uplift10_T1": 0.042,
    "uplift20_T1": 0.035,
    "uplift50_T1": 0.021,
    "policy_value_treat_positive_T1": 0.0089,
    "qini_T2": 0.0098,
    "auuc_T2": 0.0064,
    "uplift10_T2": 0.015,
    "uplift20_T2": 0.011,
    "uplift50_T2": 0.007,
    "policy_value_treat_positive_T2": 0.0043,
    "policy_value": 0.0245,
    "best_treatment_distribution": {
      "T0": 0.35,
      "T1": 0.48,
      "T2": 0.17
    }
  }
}
```

## Train Many, Evaluate One

Bei mehreren Eingabedateien kann über `data_prep.eval_file_index` eine einzelne Datei als Evaluationsgrundlage festgelegt werden. Alle Dateien werden für Training und Cross-Prediction genutzt, aber Uplift-Metriken (Qini, AUUC, Policy Value) und DRTester-Plots nur auf den Zeilen der gewählten Datei berechnet.

Die DataPrep-Pipeline erzeugt dafür eine Boolean-Maske (`eval_mask.npy`), die in der Analyse-Pipeline über `data_files.eval_mask_file` geladen wird. Die Maske überlebt Subsampling (`df_frac`) und wird positionskonsistent angewendet.

## Treatment-Balance bei mehreren Dateien

Wenn Trainingsdaten aus mehreren Dateien zusammengeführt werden, kann die Treatment-Rate pro Datei unterschiedlich sein. Eine Differenz von mehr als 5 Prozentpunkten wird als Warnung geloggt, da sie Uplift-Metriken verzerren kann — bestimmte Cross-Validation-Folds enthalten dann systematisch mehr oder weniger Treatment-Beobachtungen.

Mit `data_prep.balance_treatments: true` wird die überrepräsentierte Gruppe pro Datei per Random-Downsampling auf die niedrigste Treatment-Rate angeglichen. Es werden nur so viele Zeilen entfernt wie nötig.

## NaN/Inf-Behandlung und DRTester-Resilienz

Alle DR-Outcomes und CATE-Predictions werden vor der OLS-Regression (in EconML's `evaluate_blp()`) auf NaN/Inf geprüft und per Percentil-Clipping sanitisiert. Historische Scores (`S`) werden beim Laden auf NaN/Inf geprüft und durch 0 ersetzt. Die `_sanitize_dr()`-Methode erzwingt `float64`-Kopie, ersetzt NaN/Inf durch 0, clippt auf das 0.5/99.5-Perzentil und validiert abschließend nochmals.

**Sanitisierungs-Kette:** CATE-Predictions werden am Eingang von `evaluate_cate_with_plots()` per `nan_to_num` gesäubert, dann in `evaluate_all()` per `_sanitize_dr()` clippt, und als letzte Verteidigungslinie nochmals in der überschriebenen `evaluate_blp()` vor dem OLS-Aufruf.

### DRTester-Nuisance: Einmal fitten, für alle Modelle wiederverwenden

Der DRTester benötigt Nuisance-Modelle (Outcome + Propensity), um Doubly-Robust-Outcomes zu berechnen. Diese sind für alle kausalen Modelle **identisch** — nur die CATE-Predictions unterscheiden sich. Die Pipeline nutzt daher ein Pre-Fit-Pattern:

1. `fit_drtester_nuisance()` wird **einmal** aufgerufen (BT) bzw. einmal pro Treatment-Arm (MT)
2. Der gefittete Tester wird in `fitted_tester_bt` / `fitted_tester_mt[arm]` gespeichert
3. `evaluate_cate_with_plots(fitted_tester=...)` kopiert DR-Outcomes (`dr_val_`, `dr_train_`, `ate_val`, `Dval`) und tauscht nur die CATE-Predictions aus

Speedup: Bei 7 Modellen spart das 6× das teure Nuisance-CV-Fitting (~30-120s je nach Datengröße). Die Nuisance-Modelle nutzen leichtere Varianten: `n_estimators≤100` (statt potentiell 400+) und `cv=3` (statt 5), was nochmals ~6-7× schneller ist bei minimaler AUC-Einbuße (~0.5-1%).

### Datenfluss: Gespeicherte Predictions → Evaluation

Die Evaluation nutzt **ausschließlich bereits berechnete Predictions** — es werden keine Modelle erneut gefittet oder Predictions erneut berechnet. Der Datenfluss:

```
Step 4: Training & Cross-Predictions
│
│  preds["SLearner"]  = DataFrame(Y, T, Predictions_SLearner, Train_SLearner)
│  preds["TLearner"]  = DataFrame(Y, T, Predictions_TLearner, Train_TLearner)
│  preds["DRLearner"] = DataFrame(Y, T, Predictions_DRLearner, Train_DRLearner)
│  ...
│
▼
Step 5: Evaluation & Metriken
│
├── Phase 1: Schnelle Metriken ────────── preds[model]["Predictions_*"].to_numpy()
│   (Qini, AUUC, Uplift@k, Policy Value)   → reines NumPy, <1s pro Modell
│
├── Nuisance Pre-Fit (EINMAL) ─────────── fit_drtester_nuisance(X_val, T_val, Y_val,
│   fitted_tester_bt                        X_train, T_train, Y_train)
│                                           → gespeichert, für alle Modelle wiederverwendet
│
├── Phase 2: DRTester-Plots ───────────── evaluate_cate_with_plots(
│   (Level-abhängig)                        fitted_tester=fitted_tester_bt,
│                                           cate_preds_val=preds[model]["Predictions_*"],
│                                           cate_preds_train=preds[model]["Train_*"],
│                                           ...)
│                                           → BLP, Cal, Qini, TOC + Policy Values
│
├── Phase 3: sklift-Plots ─────────────── generate_sklift_plots(
│   (alle Modelle)                          preds[model]["Predictions_*"],
│                                           T, Y)
│                                           → Qini-Kurve, Uplift-by-Percentile, Balance
│
├── Historischer Score ────────────────── S-Datei (einmal geladen)
│                                         → evaluate_cate_with_plots(cate_preds_val=S)
│                                         → policy_value_comparison_plots()
│                                         → plot_custom_qini_curve()
│
▼
Step 6: Surrogate-Tree
│
│  surrogate_df["Predictions_SurrogateTree_*"] = tree.predict(X)
│  surrogate_df["Train_SurrogateTree_*"]       = tree.predict(X)
│
├── Surrogate-Metriken ────────────────── uplift_curve(score=surrogate_df["Predictions_*"])
├── Surrogate-DRTester ────────────────── evaluate_cate_with_plots(fitted_tester=...,
│                                           cate_preds_val=surrogate_df["Predictions_*"])
├── Surrogate-sklift ──────────────────── generate_sklift_plots(surrogate_preds, T, Y)
```

Wichtige Garantie: Weder `_evaluate_bt`, `_evaluate_mt`, `_evaluate_historical_score` noch die Surrogate-Evaluation rufen `.fit()` oder `.predict()` auf kausalen Modellen auf. Sie nutzen ausschließlich die in Step 4 bzw. Step 6 gespeicherten Spalten.

### Train-Daten und `has_train`-Prüfung

DRTester benötigt Train-Daten (`Xtrain`, `Dtrain`, `ytrain`) für Calibration, Qini-CI und TOC-CI. Die Pipeline prüft, ob in irgendeinem Modell `Train_*`-Spalten mit nicht-NaN-Werten vorhanden sind:

```python
has_train = any(
    any(c.startswith("Train_") and not np.all(np.isnan(dfp[c].to_numpy(dtype=float)))
        for c in dfp.columns if c.startswith("Train_"))
    for dfp in preds.values()
)
```

Wenn `has_train=True`: `X_train=X, T_train=T, Y_train=Y` → `fit_on_train=True` → alle DRTester-Tests laufen.
Wenn `has_train=False`: `X_train=None` → `fit_on_train=False` → nur BLP läuft, cal/qini/toc werden übersprungen.

Im Standardfall (Cross-Validation) erzeugt `train_and_crosspredict` sowohl Out-of-Fold-Predictions (`Predictions_*`) als auch In-Sample-Predictions (`Train_*`). Damit ist `has_train=True` und alle Plots werden erzeugt.

| Szenario | `has_train` | Nuisance X_train | DRTester-Tests |
|---|---|---|---|
| **Cross-Validation** | `True` (Train_* vorhanden) | X (voller Datensatz) | Alle (BLP, Cal, Qini, TOC) |
| **External Eval** | `True` (Train_* vorhanden) | X (Train-Datensatz) | Alle |
| **Train Many Eval One** | `True` (Train_* vorhanden) | X (voller Datensatz) | Alle |

### evaluate_cate_with_plots — Architektur

Die Funktion hat **einen** try/except um den gesamten DRTester-Block. Innerhalb werden alle Aufrufe **direkt** durchgeführt — ohne individuelle try/except-Blöcke:

```python
# evaluate_cate_with_plots:
try:
    res = tester.evaluate_all(X_val, X_train, ...)
    summary = res.summary()
    cal_plot = res.plot_cal(1).get_figure()
    recolor_figure(cal_plot)                    # ← rubin-Farbschema
    qini_plot = res.plot_qini(1).get_figure()
    recolor_figure(qini_plot)
    toc_plot = res.plot_toc(1).get_figure()
    recolor_figure(toc_plot)
    policy_values = res.get_policy_values(1)
except Exception:
    # Gesamter DRTester-Block fehlgeschlagen → leere Defaults
    summary = pd.DataFrame()
    policy_values = pd.DataFrame()
```

Dieses Pattern ist **identisch mit causaluka** und funktioniert zuverlässig. Wenn `evaluate_all` erfolgreich läuft, laufen auch `summary()`, `plot_cal()`, `plot_qini()`, `plot_toc()` und `get_policy_values()` — weil die Sub-Results (blp, cal, qini, toc) alle vorhanden sind.

### `summary()`-Override (Sicherheitsnetz)

Falls einzelne Sub-Results `None` sind (was im Normalbetrieb nicht vorkommt, aber bei sehr extremen Daten passieren könnte), überschreibt `CustomEvaluationResults.summary()` die EconML-Methode und baut eine partielle Tabelle:

- BLP vorhanden → BLP-Koeffizienten + p-Werte
- Calibration vorhanden → CAL-Koeffizienten + p-Werte
- Qini/TOC → haben keine tabellarische Summary, nur Plots

### numpy-Kompatibilität bei `plot_uplift_by_percentile`

Ab numpy ≥1.24 erzeugt `plot_uplift_by_percentile` intern den Fehler `setting an array element with a sequence. The requested array has an inhomogeneous shape`. Dies ist ein bekanntes Kompatibilitätsproblem in sklift. Der native Fallback (`_native_uplift_by_percentile`) greift automatisch und erzeugt einen visuell gleichwertigen Dezil-Barplot.

## HTML-Report: Plot-Vergrößerung

Alle Diagnose-Plots im HTML-Report sind klickbar. Ein Klick öffnet eine Lightbox-Vergrößerung mit dem Plot-Titel. Schließen per Klick auf den Hintergrund, ×-Button oder Escape-Taste.
