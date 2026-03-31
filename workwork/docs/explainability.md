# Explainability

Explainability ist in die **Analyse-Pipeline integriert**: Bei `shap_values.calculate_shap_values: true` werden SHAP-Werte und Importance-Plots automatisch für den Champion berechnet und als MLflow-Artefakte im selben Run geloggt. Der dreistufige Fallback lautet: ① EconML SHAP-Plot-Satz → ② generische SHAP-Werte → ③ Permutation-Importance. Die Plots werden außerdem in den HTML-Report eingebettet.

Zusätzlich kann Explainability **als separater CLI-Runner** auf Bundle-Basis ausgeführt werden – z. B. für nachträgliche Ad-hoc-Analysen auf neuen Daten oder mit einem anderen Modell als dem Champion.

## Integrierte Explainability (Analyse-Pipeline)

In der `config.yml`:

```yaml
shap_values:
  calculate_shap_values: true
  n_shap_values: 10000       # Max. Stichprobe
  top_n_features: 20         # Anzahl Features in Plots
  num_bins: 10               # Binning für PDP-Plots

```

Die Explainability wird nach dem Bundle-Export und vor dem HTML-Report ausgeführt. Artefakte in MLflow:
- SHAP-Plots (Summary, Average, PDP, Scatter) als PNG
- Importance-Barplot als PNG + CSV

**Out-of-Sample:** SHAP-Werte werden immer auf Daten berechnet, die das Modell nie gesehen hat:
- **CV-Modus:** Nutzt das letzte CV-Fold-Modell (bereits auf K-1 Folds trainiert, kein zusätzliches Training nötig) und sampelt aus dessen Out-of-Fold-Indices.
- **External-Modus:** Nutzt das auf Trainingsdaten gefittete Modell und die Eval-Daten.
- **Fallback:** Stratifizierter 80/20-Split, Fit auf 80%, Explain auf 20%.

## Separater CLI-Runner (`run_explain.py`)

```bash
pixi run explain -- --bundle bundles/<bundle_id> --x new_X.parquet --out-dir explain
# oder: python run_explain.py --bundle bundles/<bundle_id> --x new_X.parquet --out-dir explain
```

Optional kann ein bestimmtes Modell gewählt werden:

```bash
pixi run explain -- --bundle bundles/<bundle_id> --x new_X.parquet --model-name XLearner
# oder: python run_explain.py --bundle bundles/<bundle_id> --x new_X.parquet --model-name XLearner
```

## Erzeugte Artefakte

### SHAP
Wenn das Modell eine EconML-kompatible `shap_values`-Schnittstelle bereitstellt, werden die SHAP-Plots erzeugt:

- `SHAP_summary_plots_<MODEL>.png`
- `SHAP_average_plots_<MODEL>.png`
- `SHAP_pdp_plots_<MODEL>.png`
- `SHAP_scatter_plots_<MODEL>.png`
- `shap_importance_<MODEL>.csv`
- `shap_importance_<MODEL>.png`

Falls das nicht möglich ist, fällt der Runner auf modellagnostische SHAP-Werte zurück und schreibt zusätzlich `shap_values_<MODEL>.csv`.

### Permutation-Importance
Alternativ:

```bash
pixi run explain -- --bundle bundles/<bundle_id> --x new_X.parquet --method permutation
# oder: python run_explain.py --bundle bundles/<bundle_id> --x new_X.parquet --method permutation
```

Dann werden `permutation_importance_<MODEL>.csv` und `permutation_importance_<MODEL>.png` erzeugt.


## Konfiguration
Wenn im Bundle eine `config_snapshot.yml` vorhanden ist, übernimmt `run_explain.py` daraus Voreinstellungen:

- `shap_values.n_shap_values`
- `shap_values.top_n_features`
- `shap_values.num_bins`

CLI-Parameter überschreiben diese Voreinstellungen.

## Multi-Treatment-Besonderheiten

Bei Multi-Treatment-Modellen erzeugt `_predict_effect()` ein 2D-Array mit K-1 Effektschätzungen
(eine pro Treatment-Arm vs. Control). Für die Explainability wird daraus ein skalarer Wert
abgeleitet, damit SHAP und Permutation-Importance auf einer einzigen Zielgröße arbeiten:

- **SHAP:** Verwendet `max(τ_1(X), …, τ_{K-1}(X))` als skalaren Output. Das zeigt, welche
  Features den maximalen erwarteten Treatment-Effekt beeinflussen – unabhängig davon, welcher
  Arm der beste ist.
- **Permutation-Importance:** Verwendet die L2-Norm über alle Treatment-Arme
  (`||Δτ(X)||₂`), um den Gesamteinfluss eines Features auf die Effektschätzung zu messen.
