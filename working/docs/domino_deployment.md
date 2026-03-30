# Domino App-Deployment

Anleitung zum Deployment der rubin Web-UI als Domino App.

## Voraussetzungen

- Domino-Projekt mit dem rubin-Repository
- Python 3.10+ Compute Environment
- Flask in der Environment (oder wird automatisch installiert)
- **Wichtig:** CDN-Libraries müssen vor dem Deployment inlined sein (siehe Build-Schritt)

## Build-Schritt (einmalig, vor dem Deployment)

Die React-App lädt normalerweise React, ReactDOM und Babel von einem CDN.
In Firmennetzen ist das blockiert. Daher müssen die Libraries **einmalig** auf einem
Rechner mit Internetzugang eingebettet werden:

```bash
python scripts/build_app_html.py
git add app/frontend/index.html
git commit -m "React-App mit inlined Libraries"
git push
```

Danach braucht die App keinen Internetzugang mehr.

## Schnellstart

1. **Repository in Domino importieren** – Git-Repo verknüpfen oder Upload
2. **App-Deployment erstellen** – Domino → Publish → App
3. **Fertig** – Domino findet `app.sh` automatisch und startet die App

## Wie es funktioniert

### Architektur

```
Browser                         Domino (Flask-Server)
┌─────────────────────┐        ┌─────────────────────┐
│ React-App           │        │ app/server.py        │
│ (app/frontend/      │        │                      │
│  index.html)        │  HTTP  │ GET  /  → React-App  │
│                     │◄──────►│ POST /api/upload     │
│ Alle Config-        │        │ POST /api/run-*      │
│ Einstellungen       │        │ GET  /api/progress   │
│ im Browser          │        │ GET  /api/download/* │
│                     │        │ GET  /api/results    │
└─────────────────────┘        └─────────────────────┘
```

Die gesamte UI (Sidebar, 8 Seiten, Config-Builder) ist eine React-App, die als
statische HTML-Datei vom Flask-Server ausgeliefert wird. Flask stellt zusätzlich
REST-API-Endpoints bereit für Datei-Upload, Spalten-Erkennung, Analyse-Start,
Fortschritts-Tracking und Ergebnis-Downloads.

### Start-Reihenfolge

Domino sucht beim Start eines App-Deployments nach `app.sh` im Projektstamm:

1. Prüft, ob **pixi** verfügbar ist → `pixi run -e app app` (empfohlen)
2. **Fallback ohne pixi:** Installiert Dependencies via pip und startet den Flask-Server direkt

### Option A: pixi-basiertes Deployment (empfohlen)

Wenn pixi im Compute Environment installiert ist, übernimmt es die komplette
Dependency-Verwaltung:

```dockerfile
# Im Domino Compute Environment (einmalig):
RUN curl -fsSL https://pixi.sh/install.sh | bash
```

Beim App-Start erkennt `app.sh` pixi automatisch und delegiert an `pixi run -e app app`.
Alle Dependencies werden aus dem Lockfile installiert.

### Option B: pip-basiertes Deployment

Falls pixi nicht verfügbar ist:

## Compute Environment

Die Domino-Environment muss folgende Pakete enthalten:

**Minimum (nur UI + Config):**

```
flask>=3.0
pandas>=2.0
pyarrow>=12
pyyaml>=6.0
```

**Volle Pipeline (UI + Analyse starten):**

```
flask>=3.0
pandas>=2.0
pyarrow>=12
pyyaml>=6.0
scikit-learn
mlflow
optuna
lightgbm
catboost
econml
matplotlib
scipy
scikit-uplift
shap
```

**Pre-built Environment (empfohlen):**

```dockerfile
RUN pip install flask>=3.0 pandas>=2.0 pyarrow>=12 pyyaml>=6.0
# Für volle Pipeline:
RUN pip install -r requirements.txt
```

**On-the-fly:**
`app.sh` installiert fehlende Pakete automatisch beim Start.

## App-Dateien

```
app/
├── __init__.py              # Python-Package-Marker
├── app.py                   # Launcher (ruft server.py auf)
├── server.py                # Flask-Server (448 Zeilen, alle API-Endpoints)
├── requirements_app.txt     # pip-Fallback Dependencies
├── rubin_ui_src.jsx         # JSX-Quelldatei (wird nicht direkt geladen)
└── frontend/
    └── index.html           # React-App (nach Build ~3 MB mit inlined Libraries)
```

## API-Endpoints

| Endpoint | Methode | Beschreibung |
|---|---|---|
| `/` | GET | React-App (index.html) |
| `/api/upload` | POST | Datei-Upload (multipart) |
| `/api/detect-columns` | POST | Spalten aus Datei erkennen (pandas) |
| `/api/save-config` | POST | YAML auf Disk speichern |
| `/api/import-config` | POST | YAML importieren |
| `/api/run-analysis` | POST | Analyse-Pipeline starten (Background-Thread) |
| `/api/run-dataprep` | POST | Datenvorbereitung starten (Background-Thread) |
| `/api/progress` | GET | Fortschritts-Status (Polling) |
| `/api/reset` | POST | Status zurücksetzen |
| `/api/results` | GET | Verfügbare Ergebnis-Dateien auflisten |
| `/api/download/<path>` | GET | Datei herunterladen |
| `/api/report` | GET | HTML-Report + Metriken laden |

## Dateipfade in Domino

Domino mountet Projektdateien unter `/mnt/code/`. Datenpfade in der Config sollten
relativ oder auf Domino-Datasets verweisen:

| Domino-Pfad | Verwendung |
|---|---|
| `/mnt/code/` | Projektdateien (Code, Configs) |
| `/mnt/data/` | Domino Datasets (persistente Daten) |
| `/mnt/artifacts/` | Run-Artefakte (MLflow, Bundles) |
| `/mnt/imported/` | Importierte Daten aus anderen Projekten |

### Empfohlene Pfade in config.yml

```yaml
data_files:
  x_file: /mnt/data/rubin/X.parquet
  t_file: /mnt/data/rubin/T.parquet
  y_file: /mnt/data/rubin/Y.parquet
  s_file: null

optional_output:
  output_dir: /mnt/artifacts/rubin_output

bundle:
  enabled: true
  base_dir: /mnt/artifacts/bundles
```

## Lokale Entwicklung

```bash
# Mit pixi (empfohlen):
pixi run app

# Ohne pixi:
pip install flask pandas pyarrow pyyaml
python -m app.server

# Oder direkt über app.sh:
bash app.sh
```

Der Server startet auf Port 8501 (lokal) oder `$DOMINO_APP_PORT` (Domino).

## Umgebungsvariablen

| Variable | Default | Beschreibung |
|---|---|---|
| `DOMINO_APP_PORT` | – | Port für die App (von Domino gesetzt) |
| `PORT` | `8501` | Fallback-Port (lokal) |
| `HOST` | `0.0.0.0` | Bind-Adresse |
| `DOMINO_PROJECT_NAME` | (auto) | Von Domino gesetzt, nicht von rubin genutzt |

## Troubleshooting

### App startet nicht

1. **Port-Konflikt:** Prüfe ob `DOMINO_APP_PORT` korrekt gesetzt ist
2. **Fehlende Pakete:** Prüfe die Compute Environment oder `app/requirements_app.txt`
3. **Keine Logs?** Starte manuell: `python -m app.server` und prüfe stdout

### Weißer Bildschirm

1. **CDN-Libraries nicht inlined:** `python scripts/build_app_html.py` ausführen (einmalig)
2. **JavaScript-Fehler:** Browser F12 → Console prüfen
3. **index.html zu klein:** Sollte nach Build >1 MB sein (mit inlined React/Babel)

### Config-Pfade funktionieren nicht

Domino-Dateisystem unterscheidet sich von lokaler Entwicklung:
- Projektdateien: `/mnt/code/` statt `.`
- Datenpfade müssen absolut sein oder relativ zum Projektverzeichnis
- Uploads landen unter `data/uploads/` relativ zum Projekt-Root

### Analyse bleibt hängen

- Timeout im Domino-UI erhöhen (Standard oft 30 Min, Analyse kann länger dauern)
- Fortschritt prüfen: Browser → F12 → Network → `/api/progress` aufrufen
- Server-Log in Domino prüfen: `[rubin] Step 3/8: Training & Cross-Predictions`
