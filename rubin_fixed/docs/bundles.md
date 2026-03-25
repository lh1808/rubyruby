# Bundles

Der Bundle-Export wird über den Block `bundle` in der Konfiguration gesteuert.

```yaml
bundle:
  enabled: true
  base_dir: "bundles"
  bundle_id: null
  include_challengers: true
  log_to_mlflow: true
```

`run_analysis.py` kann diese Werte bei Bedarf per CLI überschreiben.

# Bundles (Analyse → Production)

## Was ist ein Bundle?
Ein Bundle ist ein Ordner, der **alle** Artefakte enthält, die für ein reproduzierbares Scoring benötigt werden:

- Konfigurations-Snapshot
- Preprocessing-Artefakte (Feature-Reihenfolge, Dtypes, ggf. Encoder)
- trainierte Modelle (Pickle-Dateien)

## Warum synchroner Export?
Der Export passiert **im gleichen Prozess** wie die Analyse. Dadurch gilt:
- das Bundle passt garantiert zur Analyse-Ausführung,
- keine Race Conditions,

## Typischer Workflow

1) Analyse laufen lassen und Bundle exportieren:
```bash
pixi run analyze -- --config config.yml --export-bundle --bundle-dir bundles
# oder: python run_analysis.py --config config.yml --export-bundle --bundle-dir bundles
```

```bash
pixi run score -- --bundle bundles/<bundle_id> --x new_X.parquet --out scores.csv
# oder: python run_production.py --bundle bundles/<bundle_id> --x new_X.parquet --out scores.csv
# Standard: Champion aus model_registry.json
# Optional: --model-name XLearner oder --use-all-models
# Surrogate: --use-surrogate (interpretierbarer Einzelbaum)
```

## Best Practices
- Bundle-IDs sollten eindeutig sein (Timestamp + optional Run-ID).
- Für Production zusätzlich Schema-Checks einführen (Spalten, Dtypes, Missing Rate).


## Zusätzliche Bundle-Inhalte (Schema & Metadaten)

### schema.json
Neben dem `preprocessor.pkl` enthält ein Bundle optional eine Datei `schema.json`.
Sie beschreibt erwartete Spalten und Datentypen des Feature-Matrix-Inputs.

**Nutzen**
- Production kann sofort melden, wenn sich das Input-Layout geändert hat.
- Typische Fehler (fehlende Spalten, falsche Typen) werden früh erkannt.

### metadata.json (automatisch angereichert)
Beim Schreiben der Metadaten werden automatisch ergänzt:
- `created_at_utc`
- `

Damit ist die Herkunft des Bundles schnell nachvollziehbar.


## Registry-Manifest und Promotion

### model_registry.json

Beim Bundle-Export schreibt rubin zusätzlich ein Registry-Manifest `model_registry.json`.
Dieses enthält:

- `models`: Liste aller Modelle im Bundle (Name, Artefaktpfad, Metriken)
- `champion`: Name des Standardmodells für Produktion
- `selection`: Regel, mit der der Champion initial gewählt wurde (z. B. Metrik)

### Champion wechseln (Promotion)

Wenn nach fachlichem Review ein anderes Modell produktiv gehen soll, kann der Champion
per CLI umgestellt werden:

```bash
pixi run promote -- --bundle bundles/<bundle_id> --model <ModelName>
# oder: python run_promote.py --bundle bundles/<bundle_id> --model <ModelName>
```

Alternativ kann in Production über `--model-name` ein anderes Modell erzwungen werden.

**Warum das bewusst so umgesetzt ist:**  
Produktionsentscheidungen sollen getrennt von Trainingsläufen getroffen werden können.
Das Manifest ist dabei der „Vertrag“ zwischen Analyse und Produktion.

### Trainingsstand der exportierten Modelle

Beim Bundle-Export werden alle Modelle so gespeichert, dass sie direkt in Production
einsetzbar sind:

- **Champion:** Wird (sofern `selection.refit_champion_on_full_data: true`) vor dem Export
  auf allen im Run verfügbaren Daten (inkl. Holdout, falls vorhanden) refittet.
- **Challenger:** Im Cross-Validation-Modus werden Challenger-Modelle vor dem Export auf
  den Trainingsdaten gefittet. Im Holdout-Modus sind sie bereits durch den initialen
  `fit()`-Aufruf trainiert. Damit ist sichergestellt, dass auch Challenger im Bundle
  für Production-Scoring nutzbar sind.
- **Surrogate-Einzelbaum** (optional, wenn `surrogate_tree.enabled: true`): Ein einzelner Baum
  des konfigurierten Base-Learners (`n_estimators=1`), der die CATE-Vorhersagen des
  (ggf. refitteten) Champions nachlernt. Wird als `SurrogateTree.pkl` im Bundle abgelegt
  und ist über `score_surrogate(X)` oder `--use-surrogate` in Production nutzbar.
  Bei Multi-Treatment wird pro Arm ein eigener Baum trainiert.

## Hinweis zu Parquet und Datentypen

Die DataPrepPipeline schreibt die standardisierten Eingabedaten als Parquet. Parquet speichert Datentypen bereits mit.
In der Praxis kommen Produktionsdaten jedoch oft aus anderen Exportwegen (und damit mit abweichenden Typen). `dtypes.json`
dient hier als robuste Referenz, um Typen konsistent auszurichten.


## Multi-Treatment-Scoring in Production

Bei Multi-Treatment (T ∈ {0, 1, …, K-1}) unterscheidet sich das Score-Ergebnis der
ProductionPipeline vom Binary-Treatment-Fall:

### Output-Spalten (Binary Treatment)

| Spalte | Beschreibung |
|---|---|
| `cate_{ModelName}` | Geschätzter CATE (ein Wert pro Beobachtung) |

### Output-Spalten (Multi-Treatment)

| Spalte | Beschreibung |
|---|---|
| `cate_{ModelName}_T1`, …, `cate_{ModelName}_T{K-1}` | Geschätzter CATE pro Treatment-Arm vs. Control |
| `optimal_treatment_{ModelName}` | Optimales Treatment (0 = Control, 1..K-1 = Treatment-Arm). Wird nur zugewiesen, wenn der beste Effekt > 0 ist. |
| `treatment_confidence_{ModelName}` | Differenz zwischen bestem und zweitbestem Treatment-Effekt. Höhere Werte = klarere Empfehlung. |

### Beispiel-Aufruf

```bash
pixi run score -- --bundle bundles/<bundle_id> --x new_X.parquet --out scores.csv
# oder: python run_production.py --bundle bundles/<bundle_id> --x new_X.parquet --out scores.csv
```

Die Production-Pipeline erkennt automatisch anhand der Modell-Ausgabe, ob es sich um
ein BT- oder MT-Modell handelt. Es ist keine separate Konfiguration nötig.