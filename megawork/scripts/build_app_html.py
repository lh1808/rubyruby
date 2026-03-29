#!/usr/bin/env python3
"""Bettet React, ReactDOM und Babel direkt in app/frontend/index.html ein.

Danach funktioniert die rubin Web-App komplett ohne Internetzugang.

Ablauf:
    1. Prueft app/frontend/lib/ auf lokale Kopien der Libraries
    2. Falls nicht lokal vorhanden: versucht 3 CDN-Quellen (cloudflare, unpkg, jsdelivr)
    3. Ersetzt die <script src="..."> Tags durch inline <script>...</script>

Ausfuehrung:
    python scripts/build_app_html.py          # normal
    python scripts/build_app_html.py --force  # auch bereits inlinete ersetzen

Manueller Fallback (kein Internet vorhanden):
    1. Auf einem Rechner MIT Internet diese 3 Dateien herunterladen:

       https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js
         -> speichern als: app/frontend/lib/react.min.js

       https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js
         -> speichern als: app/frontend/lib/react-dom.min.js

       https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.9/babel.min.js
         -> speichern als: app/frontend/lib/babel.min.js

    2. Die 3 Dateien auf den Server kopieren (USB, SCP, etc.)
    3. Ausfuehren: python scripts/build_app_html.py
"""

from __future__ import annotations

import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "app" / "frontend" / "index.html"
LIB_DIR = ROOT / "app" / "frontend" / "lib"

# ── Library-Definitionen ──
# cdn_url:    Original-URL (steht im HTML als <script src="...">)
# local_file: Dateiname fuer lokale Kopie in app/frontend/lib/
# alt_urls:   Alternative CDN-Quellen
LIBS = [
    {
        "name": "react",
        "cdn_url": "https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js",
        "local_file": "react.min.js",
        "alt_urls": [
            "https://unpkg.com/react@18.2.0/umd/react.production.min.js",
            "https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js",
        ],
        "min_size": 5_000,  # React production.min.js ist ~10 KB
    },
    {
        "name": "react-dom",
        "cdn_url": "https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js",
        "local_file": "react-dom.min.js",
        "alt_urls": [
            "https://unpkg.com/react-dom@18.2.0/umd/react-dom.production.min.js",
            "https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js",
        ],
        "min_size": 100_000,  # ReactDOM production.min.js ist ~130 KB
    },
    {
        "name": "babel",
        "cdn_url": "https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.9/babel.min.js",
        "local_file": "babel.min.js",
        "alt_urls": [
            "https://unpkg.com/@babel/standalone@7.23.9/babel.min.js",
            "https://cdn.jsdelivr.net/npm/@babel/standalone@7.23.9/babel.min.js",
        ],
        "min_size": 500_000,  # Babel standalone ist ~1.8 MB
    },
]


def _try_download(url: str, timeout: int = 20) -> str | None:
    """Versucht eine URL herunterzuladen."""
    try:
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            handler = urllib.request.ProxyHandler({"https": proxy, "http": proxy})
            opener = urllib.request.build_opener(handler)
        else:
            opener = urllib.request.build_opener()
        req = urllib.request.Request(url, headers={"User-Agent": "rubin-build/1.0"})
        with opener.open(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        return None


def _read_local(path: Path, min_size: int) -> str | None:
    """Liest eine lokale Datei. Gibt None zurueck wenn zu klein/fehlerhaft."""
    if not path.exists():
        return None
    try:
        data = path.read_text(encoding="utf-8")
        if len(data) >= min_size:
            return data
        print(f"    WARNUNG: {path.name} ist nur {len(data):,} bytes (erwartet >= {min_size:,})")
        return None
    except Exception as e:
        print(f"    WARNUNG: {path.name} nicht lesbar: {e}")
        return None


def fetch_lib(lib: dict) -> str | None:
    """Laedt Library-Code: erst lokal, dann CDN-URLs."""
    name = lib["name"]
    min_size = lib["min_size"]
    local = LIB_DIR / lib["local_file"]

    # 1. Lokale Datei
    data = _read_local(local, min_size)
    if data:
        print(f"  {name}: Lokale Datei verwendet ({len(data):,} bytes)")
        return data

    # 2. Alle CDN-URLs durchprobieren
    all_urls = [lib["cdn_url"]] + lib.get("alt_urls", [])
    for url in all_urls:
        domain = url.split("/")[2]
        print(f"  {name}: Versuche {domain} ...", end=" ", flush=True)
        data = _try_download(url)
        if data and len(data) >= min_size:
            print(f"OK ({len(data):,} bytes)")
            # Lokal speichern fuer naechstes Mal
            LIB_DIR.mkdir(parents=True, exist_ok=True)
            local.write_text(data, encoding="utf-8")
            print(f"    Gespeichert: {local}")
            return data
        elif data:
            print(f"WARNUNG: Antwort zu klein ({len(data)} bytes)")
        else:
            print("FEHLER")

    return None


def _build_cdn_tag(url: str) -> str:
    return f'<script src="{url}"></script>'


def _build_inline_tag(name: str, code: str) -> str:
    return f"<!-- {name} (inlined) -->\n<script>{code}</script>"


def _build_inline_pattern(name: str) -> str:
    """Regex-Pattern das einen bereits inlineten Block matcht."""
    return rf"<!-- {re.escape(name)} \(inlined\) -->\n<script>.*?</script>"


def main():
    force = "--force" in sys.argv

    print("[build_app_html] CDN-Libraries inlinen ...")
    if force:
        print("  (--force: Bereits inlinete Libraries werden ersetzt)")
    print()

    if not INDEX.exists():
        sys.exit(f"FEHLER: index.html nicht gefunden: {INDEX}")

    html = INDEX.read_text(encoding="utf-8")
    print(f"  Gelesen: {INDEX} ({len(html):,} bytes)")
    print()

    errors = []
    for lib in LIBS:
        name = lib["name"]
        cdn_tag = _build_cdn_tag(lib["cdn_url"])
        local_tag = f'<script src="lib/{lib["local_file"]}"></script>'
        inline_pattern = _build_inline_pattern(name)

        # Status pruefen: CDN-Tag, lokaler Tag oder bereits inlined?
        has_cdn_tag = cdn_tag in html
        has_local_tag = local_tag in html
        has_inline = re.search(inline_pattern, html, re.DOTALL) is not None

        if has_inline and not force:
            print(f"  {name}: Bereits inlined. (--force zum Ersetzen)")
            continue

        if not has_cdn_tag and not has_local_tag and not has_inline:
            print(f"  WARNUNG: Weder CDN-Tag noch lokaler Tag noch Inline-Block fuer {name} gefunden!")
            errors.append(name)
            continue

        # Library laden
        code = fetch_lib(lib)
        if not code:
            print(f"  FEHLER: {name} nicht verfuegbar!")
            print(f"    Lokale Datei erwartet: {LIB_DIR / lib['local_file']}")
            print(f"    Download: {lib['cdn_url']}")
            errors.append(name)
            continue

        # Ersetzen: CDN-Tag, lokalen Tag oder bestehenden Inline-Block
        new_tag = _build_inline_tag(name, code)
        if has_cdn_tag:
            html = html.replace(cdn_tag, new_tag)
        elif has_local_tag:
            html = html.replace(local_tag, new_tag)
        elif has_inline:
            html = re.sub(inline_pattern, new_tag, html, count=1, flags=re.DOTALL)
        print(f"  -> {name} inlined.\n")

    if errors:
        print()
        print(f"  FEHLER: {len(errors)} Library(s) fehlen: {', '.join(errors)}")
        print()
        print("  Manueller Fallback:")
        print(f"    mkdir -p {LIB_DIR}")
        for lib in LIBS:
            if lib["name"] in errors:
                print(f"    # {lib['name']}:")
                print(f"    #   Download: {lib['cdn_url']}")
                print(f"    #   Speichern: {LIB_DIR / lib['local_file']}")
        print()
        print("  Dann erneut: python scripts/build_app_html.py")
        sys.exit(1)

    INDEX.write_text(html, encoding="utf-8")
    size_kb = len(html) / 1024
    print()
    print(f"  Geschrieben: {INDEX} ({size_kb:.0f} KB)")
    print("[build_app_html] Fertig.")


if __name__ == "__main__":
    main()
