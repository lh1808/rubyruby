#!/usr/bin/env python3
"""Synchronisiert requirements.txt und app/requirements_app.txt aus pyproject.toml.

Wird über `pixi run sync-requirements` aufgerufen.
Vermeidet Divergenz zwischen pixi-Konfiguration und pip-Fallback-Dateien.
"""

from __future__ import annotations

import sys
from pathlib import Path

# tomllib ab Python 3.11, vorher tomli als Fallback
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        sys.exit(
            "Weder tomllib (Python ≥3.11) noch tomli gefunden.\n"
            "  → pip install tomli   oder Python ≥3.11 verwenden."
        )

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

HEADER_REQ = """\
# ────────────────────────────────────────────────────
# AUTO-GENERATED – nicht manuell bearbeiten!
# Quelle: pyproject.toml  →  pixi run sync-requirements
# ────────────────────────────────────────────────────
"""

HEADER_APP = """\
# ────────────────────────────────────────────────────
# AUTO-GENERATED – nicht manuell bearbeiten!
# Quelle: pyproject.toml  →  pixi run sync-requirements
# ────────────────────────────────────────────────────
# Nur fuer pip-Fallback in app.sh (Web-UI).
# ────────────────────────────────────────────────────
"""


def main() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]

    core_deps: list[str] = project.get("dependencies", [])
    optional: dict[str, list[str]] = project.get("optional-dependencies", {})
    shap_deps: list[str] = optional.get("shap", [])
    app_deps: list[str] = optional.get("app", [])

    # ── requirements.txt (core + shap) ──
    req_lines = [HEADER_REQ]
    req_lines.append("# Core")
    for dep in core_deps:
        req_lines.append(dep)
    if shap_deps:
        req_lines.append("")
        req_lines.append("# Explainability (optional)")
        for dep in shap_deps:
            req_lines.append(dep)
    req_lines.append("")

    req_path = ROOT / "requirements.txt"
    req_path.write_text("\n".join(req_lines), encoding="utf-8")
    print(f"  ✓ {req_path.relative_to(ROOT)}")

    # ── app/requirements_app.txt (Flask-Backend + React-Frontend) ──
    app_lines = [HEADER_APP]
    app_lines.append("# Die React-App (rubin_ui.html) laeuft rein client-side.")
    app_lines.append("# Flask dient als Backend-Server, React als Frontend.")
    app_lines.append("")
    for dep in app_deps:
        app_lines.append(dep)
    app_lines.append("")

    app_path = ROOT / "app" / "requirements_app.txt"
    app_path.write_text("\n".join(app_lines), encoding="utf-8")
    print(f"  ✓ {app_path.relative_to(ROOT)}")


if __name__ == "__main__":
    print("[sync-requirements] Generiere aus pyproject.toml ...")
    main()
    print("[sync-requirements] Fertig.")
