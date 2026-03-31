from __future__ import annotations

"""Einstiegspunkt für die Production-Pipeline (Scoring)."""

import argparse
import json
import logging

from rubin.pipelines.production_pipeline import ProductionPipeline
from rubin.utils.io_utils import read_table

_logger = logging.getLogger("rubin.production")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Pfad zum exportierten Bundle-Verzeichnis")
    ap.add_argument("--x", required=True, help="CSV- oder Parquet-Datei mit denselben Feature-Spalten wie im Training")
    ap.add_argument("--out", default="production_scores.csv")
    ap.add_argument("--model-name", default=None, help="Optional: genau ein Modell aus dem Bundle scoren")
    ap.add_argument("--use-all-models", action="store_true", help="Scort alle Modelle im Bundle statt nur den Champion")
    ap.add_argument("--use-surrogate", action="store_true", help="Scort mit dem Surrogate-Einzelbaum statt dem Champion")
    args = ap.parse_args()

    X = read_table(args.x)
    pipe = ProductionPipeline(args.bundle)
    model_names = None
    if args.use_surrogate:
        outputs = pipe.score_surrogate(X)
    elif args.use_all_models:
        model_names = sorted(pipe.models.keys())
        outputs = pipe.score(X, model_names=model_names)
    elif args.model_name:
        model_names = [args.model_name]
        outputs = pipe.score(X, model_names=model_names)
    else:
        outputs = pipe.score(X)
    outputs.cate.to_csv(args.out)
    _logger.info("Geschrieben: %s", args.out)

    # Optional: Schema-Report (hilfreich für Monitoring/Debugging)
    if getattr(pipe, "last_schema_report", None) is not None:
        rep_path = args.out + ".schema_report.json"
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(pipe.last_schema_report, f, ensure_ascii=False, indent=2)
        _logger.info("Geschrieben: %s", rep_path)


if __name__ == "__main__":
    main()