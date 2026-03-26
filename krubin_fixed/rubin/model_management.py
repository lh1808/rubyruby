from __future__ import annotations

"""Modell-Management (Champion/Challenger) auf Bundle-Ebene.
Ziel
----
In einem Analyselauf werden oft mehrere kausale Learner trainiert (z. B. S-, T-, X-Learner).
Für Produktion wird jedoch typischerweise **ein** Modell benötigt.
Dieses Modul implementiert deshalb ein leichtgewichtiges Registry-Manifest, das im Bundle
gespeichert wird:
- Liste aller trainierten Modelle (Challenger)
- relevante Metriken je Modell
- Auswahl eines **Champion-Modells** anhand einer konfigurierbaren Regel
Warum das sinnvoll ist
----------------------
- Produktion bleibt simpel: Standard = Champion.
- Entscheidungen sind nachvollziehbar: Manifest dokumentiert Alternativen und Kennzahlen.
- Flexibel: Ein Champion lässt sich nach fachlicher Prüfung „promoten“."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List
import json


REGISTRY_FILENAME = "model_registry.json"


@dataclass(frozen=True)
class ModelEntry:
    """Ein Eintrag im Registry-Manifest."""

    name: str
    artifact_path: str
    metrics: Dict[str, float]


def float_metrics(d: dict) -> dict:
    """Filtert ein Dictionary auf Einträge mit numerischen Werten.

    Bei Multi-Treatment enthält eval_summary z. B. ``best_treatment_distribution``
    als dict. ``ModelEntry.metrics`` erwartet aber float-Werte, daher wird hier
    nach int/float gefiltert.
    """
    return {k: v for k, v in d.items() if isinstance(v, (int, float))}


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def choose_champion(
    entries: List[ModelEntry],
    metric: str,
    higher_is_better: bool = True,
) -> Optional[str]:
    """Wählt den Champion anhand einer Metrik.
Hinweise:
- Wenn die Metrik bei einem Modell fehlt, wird dieses Modell ignoriert.
- Bei Gleichstand wird das erste Modell nach Sortierung gewählt (deterministisch).
Warum:
Eine klare, deterministische Regel verhindert „zufällige“ Champion-Wechsel."""
    scored = []
    for e in entries:
        val = _safe_float(e.metrics.get(metric))
        if val is None:
            continue
        scored.append((val, e.name))
    if not scored:
        return None
    scored.sort(reverse=higher_is_better)
    return scored[0][1]


def write_registry(
    bundle_dir: Path,
    entries: List[ModelEntry],
    champion: Optional[str],
    selection: Dict[str, Any],
) -> Path:
    """Schreibt das Registry-Manifest ins Bundle."""
    payload = {
        "champion": champion,
        "selection": selection,
        "models": [
            {"name": e.name, "artifact_path": e.artifact_path, "metrics": e.metrics}
            for e in entries
        ],
    }
    path = Path(bundle_dir) / REGISTRY_FILENAME
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_registry(bundle_dir: Path) -> Dict[str, Any]:
    """Liest das Registry-Manifest aus einem Bundle."""
    path = Path(bundle_dir) / REGISTRY_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Registry-Manifest nicht gefunden: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def promote_champion(bundle_dir: Path, model_name: str) -> Path:
    """Setzt ein bestimmtes Modell als Champion.
Diese Funktion ist bewusst *side-effect-light*:
- Es wird nur das Manifest geändert.
Typischer Use Case:
Nach fachlicher Prüfung möchte man einen Challenger als Champion einsetzen."""
    manifest = read_registry(bundle_dir)
    names = {m["name"] for m in manifest.get("models", [])}
    if model_name not in names:
        raise ValueError(
            f"Modell '{model_name}' ist nicht im Registry-Manifest enthalten. Verfügbar: {sorted(names)}"
        )
    manifest["champion"] = model_name
    path = Path(bundle_dir) / REGISTRY_FILENAME
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
