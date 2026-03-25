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

- **Calibration Plot**
- **Qini Plot**
- **TOC Plot**
- eine tabellarische **Summary** der Tests

In rubin werden dafür die Cross-Predictions genutzt. Optional wird zusätzlich
eine In-Sample-Vorhersage (`Train_<model>`) berechnet und an den Tester
übergeben. Diese `Train_*`-Vorhersage ist **keine** Performance-Schätzung,
sondern dient nur für konsistente Diagnose-Auswertungen.

## scikit-uplift (sklift)

Die Pipeline erzeugt auch Plots aus scikit-uplift:

- Qini Curve (`plot_qini_curve`)
- Uplift-by-Percentile (`plot_uplift_by_percentile`)
- Treatment Balance (`plot_treatment_balance_curve`)

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
