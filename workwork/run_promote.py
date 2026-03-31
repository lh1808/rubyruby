#!/usr/bin/env python
from __future__ import annotations

"""Champion-Modell in einem Bundle umschalten.
Dieses Skript ändert ausschließlich das Registry-Manifest (`model_registry.json`) im Bundle.
Typischer Workflow:
1) Analyse-Run erzeugt Bundle mit automatischem Champion (z. B. nach Qini).
2) Fachlicher Review / Abnahme entscheidet ggf. anders.
3) `run_promote.py` setzt einen anderen Challenger als Champion.
Beispiel:
python run_promote.py --bundle bundles/2026-03-05_... --model TLearner"""

import argparse
import logging
from pathlib import Path

from rubin.model_management import promote_champion, read_registry

_logger = logging.getLogger("rubin.promote")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="Pfad zum Bundle-Verzeichnis")
    parser.add_argument("--model", required=True, help="Name des Modells, das Champion werden soll")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle)
    path = promote_champion(bundle_dir, args.model)
    manifest = read_registry(bundle_dir)

    _logger.info("Champion gesetzt: %s", manifest.get("champion"))
    _logger.info("Aktualisiertes Manifest: %s", path)


if __name__ == "__main__":
    main()
