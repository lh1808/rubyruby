from __future__ import annotations

"""Runner für Explainability in rubin.
Der Runner arbeitet auf Basis eines Bundles und erzeugt Explainability-Artefakte
für ein ausgewähltes Modell. Neben SHAP- bzw. Permutation-Importances werden
Segmentanalysen geschrieben, damit neben globalen Wichtigkeiten auch sichtbar
wird, welche Kundensegmente besonders stark oder schwach reagieren."""

import argparse
import logging
from pathlib import Path
import json

import numpy as np

from rubin.pipelines.production_pipeline import ProductionPipeline
from rubin.utils.io_utils import read_table
from rubin.settings import load_config
from rubin.model_management import read_registry
from rubin.training import _predict_effect
from rubin.explainability import (
    compute_shap_for_uplift,
    compute_permutation_importance_for_uplift,
    build_shap_plots,
    build_segment_report,
    build_feature_segment_report,
    shap_available,
)
from rubin.explainability.reporting import ensure_dir, save_importance_barplot, save_shap_summary_csv, save_json


def _choose_model_name(bundle_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    manifest = read_registry(bundle_dir)
    champion = manifest.get("champion")
    if not champion:
        raise RuntimeError(
            "Im Bundle ist kein Champion definiert. Bitte '--model-name' angeben oder ein Champion-Modell promoten."
        )
    return str(champion)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Pfad zum Bundle-Verzeichnis")
    ap.add_argument("--x", required=True, help="CSV- oder Parquet-Datei mit Roh-Features")
    ap.add_argument("--out-dir", default="explain", help="Ausgabeordner für Explainability-Artefakte")
    ap.add_argument("--model-name", default=None, help="Optional: explizites Modell (sonst Champion)")
    ap.add_argument("--method", choices=["shap", "permutation"], default="shap")
    ap.add_argument("--sample-size", type=int, default=None, help="Maximale Anzahl Zeilen für Explainability")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bundle_dir = Path(args.bundle)
    out_dir = ensure_dir(args.out_dir)

    cfg = None
    snapshot = bundle_dir / "config_snapshot.yml"
    if snapshot.exists():
        try:
            cfg = load_config(str(snapshot))
        except Exception:
            cfg = None

    if args.sample_size is None and cfg is not None:
        args.sample_size = int(cfg.shap_values.n_shap_values)
    top_n_features = int(cfg.shap_values.top_n_features) if cfg is not None else 20
    num_bins = int(cfg.shap_values.num_bins) if cfg is not None else 10
    segment_quantiles = int(cfg.segment_analysis.quantiles) if cfg is not None else 10
    segment_top_features = int(cfg.segment_analysis.top_n_features) if cfg is not None else min(8, top_n_features)
    segment_max_bins = int(cfg.segment_analysis.max_bins) if cfg is not None else 6
    segment_max_categories = int(cfg.segment_analysis.max_categories) if cfg is not None else 15
    segment_enabled = bool(cfg.segment_analysis.enabled) if cfg is not None else True

    X_raw = read_table(args.x)

    pipe = ProductionPipeline(str(bundle_dir))
    model_name = _choose_model_name(bundle_dir, args.model_name)
    if model_name not in pipe.models:
        raise ValueError(
            f"Modell '{model_name}' ist im Bundle nicht vorhanden. Verfügbar: {sorted(pipe.models.keys())}"
        )
    model = pipe.models[model_name]

    X = pipe.preprocessor.transform(X_raw)

    rng = np.random.default_rng(args.seed)
    if args.sample_size is not None and len(X) > args.sample_size:
        idx = rng.choice(len(X), size=args.sample_size, replace=False)
        X = X.iloc[idx]

    raw_uplift = np.asarray(_predict_effect(model, X))
    # Bei Multi-Treatment: 2D (n, K-1) → verwende max-Effekt als skalaren Score
    # für Segment-Reports und Explainability.
    if raw_uplift.ndim == 2 and raw_uplift.shape[1] > 1:
        uplift = np.max(raw_uplift, axis=1)
    else:
        uplift = raw_uplift.reshape(-1)

    index = {
        "model": model_name,
        "out_dir": str(out_dir),
    }

    if segment_enabled:
        seg = build_segment_report(uplift=uplift, quantiles=segment_quantiles)
        seg_path = out_dir / f"segment_report_{model_name}.csv"
        seg.to_csv(str(seg_path))
        index["segment_report"] = seg_path.name

    if args.method == "shap":
        if not shap_available():
            raise RuntimeError(
                "Methode 'shap' gewählt, aber das Paket 'shap' ist nicht installiert. "
                "Bitte installieren oder '--method permutation' verwenden."
            )

        # Zunächst den EconML-kompatiblen SHAP-Plot-Satz versuchen. Falls das
        # Modell keine passende SHAP-Schnittstelle bereitstellt, auf den generischen
        # SHAP-Workflow zurückfallen.
        try:
            shap_result = build_shap_plots(
                model=model,
                X=X,
                data=X,
                cate=uplift,
                top_n=top_n_features,
                num_bins=num_bins,
            )
            shap_result.summary_plots.savefig(out_dir / f"SHAP_summary_plots_{model_name}.png", dpi=160, bbox_inches="tight")
            shap_result.average_plots.savefig(out_dir / f"SHAP_average_plots_{model_name}.png", dpi=160, bbox_inches="tight")
            shap_result.pdp_plots.savefig(out_dir / f"SHAP_pdp_plots_{model_name}.png", dpi=160, bbox_inches="tight")
            shap_result.scatter_plots.savefig(out_dir / f"SHAP_scatter_plots_{model_name}.png", dpi=160, bbox_inches="tight")
            shap_result.importance.head(top_n_features).to_csv(out_dir / f"shap_importance_{model_name}.csv")
            save_importance_barplot(
                shap_result.importance,
                str(out_dir / f"shap_importance_{model_name}.png"),
                top_n=top_n_features,
                title=f"SHAP-Importance (Uplift) – {model_name}",
            )
            top_features = list(shap_result.importance.head(segment_top_features).index)
            index.update({
                "summary_plots": f"SHAP_summary_plots_{model_name}.png",
                "average_plots": f"SHAP_average_plots_{model_name}.png",
                "pdp_plots": f"SHAP_pdp_plots_{model_name}.png",
                "scatter_plots": f"SHAP_scatter_plots_{model_name}.png",
                "importance": f"shap_importance_{model_name}.csv",
            })
        except Exception:
            res = compute_shap_for_uplift(model=model, X=X, seed=args.seed)
            imp = res.mean_abs_importance()
            imp.head(top_n_features).to_csv(out_dir / f"shap_importance_{model_name}.csv")
            save_importance_barplot(
                imp,
                str(out_dir / f"shap_importance_{model_name}.png"),
                top_n=top_n_features,
                title=f"SHAP-Importance (Uplift) – {model_name}",
            )
            save_shap_summary_csv(res.shap_values, res.feature_names, str(out_dir / f"shap_values_{model_name}.csv"))
            save_json(
                {"model": model_name, "method": "shap", "expected_value": res.expected_value},
                str(out_dir / f"explain_metadata_{model_name}.json"),
            )
            top_features = list(imp.head(segment_top_features).index)
            index.update({
                "importance": f"shap_importance_{model_name}.csv",
                "shap_values": f"shap_values_{model_name}.csv",
            })
    else:
        res = compute_permutation_importance_for_uplift(model=model, X=X, seed=args.seed)
        imp = res.as_series()
        imp.to_csv(out_dir / f"permutation_importance_{model_name}.csv")
        save_importance_barplot(
            imp,
            str(out_dir / f"permutation_importance_{model_name}.png"),
            title=f"Permutation-Importance (Uplift) – {model_name}",
        )
        save_json(
            {"model": model_name, "method": "permutation", "n_repeats": 5},
            str(out_dir / f"explain_metadata_{model_name}.json"),
        )
        top_features = list(imp.head(segment_top_features).index)
        index.update({"importance": f"permutation_importance_{model_name}.csv"})

    if segment_enabled:
        feature_seg = build_feature_segment_report(
            X=X,
            uplift=uplift,
            top_features=top_features,
            max_features=segment_top_features,
            max_bins=segment_max_bins,
            max_categories=segment_max_categories,
        )
        feature_seg_path = out_dir / f"feature_segment_report_{model_name}.csv"
        feature_seg.to_csv(str(feature_seg_path))
        index["feature_segment_report"] = feature_seg_path.name

    (out_dir / "INDEX.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.getLogger("rubin.explain").info("Explainability-Artefakte geschrieben nach: %s", out_dir)


if __name__ == "__main__":
    main()
