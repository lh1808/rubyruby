# Web-App bauen & offline bereitstellen

Anleitung zum Build der rubin Web-App für Umgebungen ohne Internetzugang.

## Überblick

Die rubin Web-App besteht aus zwei Teilen:

| Datei | Rolle |
|---|---|
| `app/rubin_ui_src.jsx` | React-Quellcode (ES-Module-Syntax, für Entwicklung) |
| `app/frontend/index.html` | Gebaute App (Babel-Standalone-kompatibel, wird vom Server ausgeliefert) |
| `app/server.py` | Flask-Backend (API-Endpoints, statische Dateien) |

`index.html` enthält den **gesamten** App-Code inline — kein Bundler nötig.
Babel Standalone kompiliert das JSX im Browser zur Laufzeit.

Dafür müssen **drei JavaScript-Libraries** in der HTML-Datei eingebettet sein:

- **React 18.2.0** (~10 KB) — UI-Framework
- **ReactDOM 18.2.0** (~130 KB) — DOM-Rendering
- **Babel Standalone 7.23.9** (~1.8 MB) — JSX → JavaScript Kompilierung im Browser

## Schnell-Anleitung

### Option A: CDN verfügbar (z.B. lokaler Rechner mit Internet)

```bash
pixi run build-app
# oder: python scripts/build_app_html.py
```

Das Script lädt die Libraries automatisch herunter und bettet sie in `index.html` ein.

### Option B: Kein Internet (Firmenumgebung, Domino)

1. Auf einem Rechner **mit Internet** die 3 Dateien herunterladen:

   ```
   https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js
   https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js
   https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.9/babel.min.js
   ```

2. Die Dateien auf den Server kopieren (USB, SCP, etc.) und ablegen als:

   ```
   app/frontend/lib/react.min.js        (ca. 10 KB)
   app/frontend/lib/react-dom.min.js    (ca. 130 KB)
   app/frontend/lib/babel.min.js        (ca. 1.8 MB)
   ```

3. Build-Script ausführen — erkennt die lokalen Dateien automatisch:

   ```bash
   pixi run build-app
   ```

4. Server starten:

   ```bash
   pixi run -e app app
   ```

### Option C: Bereits inlined, aber kaputt (erneut bauen)

```bash
python scripts/build_app_html.py --force
```

Die `--force`-Option ersetzt auch bereits inlinete Libraries neu — z.B. wenn
ein früherer Build fehlerhaften Inhalt eingebettet hat.

## Prüfen ob der Build geklappt hat

```bash
# Dateigröße prüfen — mit inlineten Libraries ca. 2+ MB
ls -lh app/frontend/index.html

# CDN-Tags sollten NICHT mehr vorhanden sein
grep "cdnjs.cloudflare" app/frontend/index.html
# (keine Ausgabe = OK)

# Inlined-Marker sollten vorhanden sein
grep "inlined" app/frontend/index.html
# Erwartete Ausgabe:
#   <!-- react (inlined) -->
#   <!-- react-dom (inlined) -->
#   <!-- babel (inlined) -->
```

## JSX-Quellcode → index.html: Transformationsregeln

Beim Einbetten von `rubin_ui_src.jsx` in `index.html` müssen diese
Transformationen angewendet werden, da **Babel Standalone** im Browser
keine ES-Module-Syntax verarbeiten kann:

### 1. Imports → globale Variablen

```jsx
// rubin_ui_src.jsx (ES-Module — für Bundler/Entwicklung):
import { useState, useEffect, useRef } from "react";

// index.html (Babel Standalone — für Browser):
const { useState, useEffect, useRef } = React;
```

**Warum:** Babel Standalone läuft im Browser und kennt kein Modul-System.
React ist als globale Variable `React` über das `<script>`-Tag verfügbar.

### 2. Export entfernen

```jsx
// rubin_ui_src.jsx:
export default function App() { ... }

// index.html:
function App() { ... }
```

**Warum:** Es gibt keinen Modul-Loader — die Funktion wird direkt
von `ReactDOM.createRoot(...)` aufgerufen.

### 3. React Hooks: Reihenfolge einhalten

Alle `useState()`, `useRef()` und `useEffect()`-Aufrufe **müssen am Anfang**
jeder Komponente stehen, in **immer gleicher Reihenfolge**. React prüft das.

```jsx
// RICHTIG — alle Hooks zuerst:
const MyComponent = () => {
  const [value, setValue] = useState(null);   // ← Hooks
  const [loading, setLoading] = useState(false);
  const ref = useRef(null);

  useEffect(() => { ... }, []);               // ← Effects nach Hooks

  const doSomething = async () => { ... };    // ← Logik nach Effects
  return <div>...</div>;
};

// FALSCH — useState nach useEffect:
const MyComponent = () => {
  const [value, setValue] = useState(null);
  useEffect(() => { ... }, []);
  const [loading, setLoading] = useState(false);  // ← CRASH!
  // ...
};
```

### 4. Script-Tag-Typ

```html
<!-- index.html: Babel kompiliert den JSX-Block zur Laufzeit -->
<script type="text/babel">
  const { useState, useEffect, useRef } = React;
  // ... gesamter App-Code ...
  ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
</script>
```

### 5. Kein Top-Level `await`

Babel Standalone unterstützt kein Top-Level `await`. Alle `async/await`-Aufrufe
müssen innerhalb von Funktionen stehen:

```jsx
// RICHTIG:
const detectColumns = async () => {
  const res = await fetch("/api/detect-columns", { ... });
};

// FALSCH:
const data = await fetch("/api/something");  // ← Top-Level await = Crash
```

## Zusammenfassung der Build-Kette

```
rubin_ui_src.jsx              (Quellcode, ES-Module-Syntax)
        │
        ├── import → const     (Transformation)
        ├── export → entfernen (Transformation)
        │
        ▼
index.html                    (Babel-Standalone-kompatibel)
        │
        ├── build_app_html.py  (Libraries inlinen)
        │
        ▼
index.html (final)            (Komplett offline-fähig, ~2+ MB)
        │
        ▼
server.py                     (Flask liefert index.html aus)
```

## Fehlerbehebung

| Problem | Ursache | Lösung |
|---|---|---|
| Leere Seite, Console: "React is not defined" | Libraries nicht inlined | `pixi run build-app` |
| Leere Seite, Console: "Unexpected token import" | `import`-Statement in index.html | JSX → `const { ... } = React;` |
| Leere Seite, Console: "Invalid hook call" | useState/useRef nach useEffect | Hooks-Reihenfolge korrigieren |
| `pixi run build-app` schlägt fehl | CDN blockiert, keine lokalen Dateien | Dateien manuell in `app/frontend/lib/` ablegen |
| build-app sagt "Bereits inlined" | Erster Build war fehlerhaft | `python scripts/build_app_html.py --force` |
| "Name or service not known" beim Start | `HOST`-Variable durch Conda/pixi überschrieben | server.py nutzt `RUBIN_HOST` statt `HOST` |
| "No module named app.server" | `app` nicht in `pyproject.toml` packages | `include = ["rubin", "rubin.*", "app"]` |

## Produktions-Prinzipien

Die rubin Web-App ist eine **reine Produktions-Anwendung** — es gibt keine
Demo-Daten, keine Simulationen und keine Offline-Fallbacks im UI-Code.

### Kein Demo-Modus

Alle Interaktionen erfordern ein laufendes Backend (`app/server.py`):

| Aktion | Backend-Endpoint | Ohne Backend |
|---|---|---|
| Spalten erkennen | `POST /api/detect-columns` | Fehlermeldung: "Backend nicht erreichbar" |
| Datenvorbereitung | `POST /api/run-dataprep` + Polling | Fehlermeldung: "Backend nicht erreichbar" |
| Analyse starten | `POST /api/run-analysis` + Polling | Fehlermeldung: "Backend nicht erreichbar" |
| Datei hochladen | `POST /api/upload` | Upload schlägt fehl |
| Report anzeigen | `GET /api/report` → iframe | Hinweis: "Report wird geladen" + Download-Link |
| Ergebnisse | `GET /api/results` | Hinweis: "Ergebnisse neu laden" |

### Keine Demo-Daten

- Keine hardcodierten Beispiel-Spalten, Datentypen oder Treatment-Werte
- Keine `DemoReport`-Komponente — der Report wird als echter HTML-Report
  vom Backend per iframe geladen
- Keine simulierten Fortschrittsbalken — Echtzeit-Status via `GET /api/progress`
- Ergebnis-Dateien kommen ausschließlich vom Backend (`GET /api/results`)

### Fehlerbehandlung statt Fallback

Wenn das Backend nicht erreichbar ist, zeigt die App klare Fehlermeldungen:

- **Rote Info-Box** mit Fehlerbeschreibung
- **"Erneut versuchen"**-Button
- **Keine** automatische Simulation oder Demo-Daten als Ausweich

### Dokumentations-Vorschau (rubin_overview.html)

Die Confluence-Dokumentation (`rubin_overview.html`) enthält eine **separate**,
vereinfachte JSX-Version mit eingebetteten Screenshots. Diese nutzt bewusst
Client-seitige Simulation und Demo-Daten — damit die Dokumentation auch
offline in Confluence korrekt angezeigt wird. Diese Doku-Version ist vollständig
getrennt vom Produktions-Code in `app/`.
