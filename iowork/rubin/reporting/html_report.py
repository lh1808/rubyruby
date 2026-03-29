from __future__ import annotations

"""HTML-Report-Generator für rubin-Analyseläufe.

Erzeugt einen selbstständigen HTML-Report mit eingebetteten Plots (base64)
und rubinrotem Design (abgestimmt auf die Web-UI). Der Report wird
am Ende des Analyselaufs generiert und enthält alle relevanten Informationen
zu Input, Training und Output.
"""

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from html import escape

import numpy as np
import pandas as pd

_logger = logging.getLogger("rubin.reporting")


# ---------------------------------------------------------------------------
# Plot-Konvertierung
# ---------------------------------------------------------------------------

def fig_to_base64(fig) -> str:
    """Konvertiert eine Matplotlib-Figure in einen base64-PNG-String."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass
    return encoded


def df_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Konvertiert ein DataFrame in eine HTML-Tabelle."""
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_html(
        classes="dt", index=False, border=0,
        float_format=lambda x: f"{x:.6f}" if abs(x) < 1 else f"{x:.4f}",
    )


# ---------------------------------------------------------------------------
# Report-Collector
# ---------------------------------------------------------------------------

@dataclass
class ReportCollector:
    """Sammelt Daten während des Analyselaufs für den HTML-Report."""

    config_summary: Dict[str, Any] = field(default_factory=dict)
    config_raw: Dict[str, Any] = field(default_factory=dict)
    data_stats: Dict[str, Any] = field(default_factory=dict)
    eval_data_stats: Dict[str, Any] = field(default_factory=dict)  # Stats des externen Eval-Datensatzes
    feature_selection_info: Dict[str, Any] = field(default_factory=dict)
    tuning_scores: Dict[str, float] = field(default_factory=dict)
    fmt_info: Dict[str, Any] = field(default_factory=dict)
    model_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model_plots: Dict[str, Dict[str, str]] = field(default_factory=dict)
    champion_name: str = ""
    champion_info: Dict[str, Any] = field(default_factory=dict)
    surrogate_info: Dict[str, Any] = field(default_factory=dict)
    explainability_plots: Dict[str, str] = field(default_factory=dict)
    explainability_info: Dict[str, Any] = field(default_factory=dict)
    step_durations: Dict[str, float] = field(default_factory=dict)
    total_elapsed: float = 0.0
    dataprep_info: Dict[str, Any] = field(default_factory=dict)
    run_name: str = ""

    # Tuning plan (task-sharing details)
    tuning_plan: List[Dict[str, Any]] = field(default_factory=list)
    # FMT plan (per-model method details)
    fmt_plan: List[Dict[str, Any]] = field(default_factory=list)
    # Best hyperparameters found
    best_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fmt_best_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_config(self, cfg) -> None:
        """Extrahiert Config-Zusammenfassung."""
        self.config_summary = {
            "experiment_name": cfg.mlflow.experiment_name,
            "seed": cfg.constants.random_seed,
            "parallel_level": cfg.constants.parallel_level,
            "base_learner": cfg.base_learner.type,
            "validate_on": cfg.data_processing.validate_on,
            "cv_splits": cfg.data_processing.cross_validation_splits,
            "treatment_type": cfg.treatment.type,
            "models": list(cfg.models.models_to_train or []),
            "tuning_enabled": cfg.tuning.enabled,
            "tuning_trials": cfg.tuning.n_trials if cfg.tuning.enabled else 0,
            "tuning_metric": getattr(cfg.tuning, "metric", "log_loss"),
            "tuning_single_fold": getattr(cfg.tuning, "single_fold", False),
            "final_tuning_enabled": cfg.final_model_tuning.enabled,
            "final_tuning_trials": cfg.final_model_tuning.n_trials if cfg.final_model_tuning.enabled else 0,
            "final_tuning_models": getattr(cfg.final_model_tuning, "models", None),
            "final_tuning_single_fold": getattr(cfg.final_model_tuning, "single_fold", False),
            "feature_selection_enabled": cfg.feature_selection.enabled,
            "feature_selection_methods": list(cfg.feature_selection.methods) if cfg.feature_selection.enabled else [],
            "feature_selection_top_pct": cfg.feature_selection.top_pct,
            "surrogate_enabled": cfg.surrogate_tree.enabled,
            "selection_metric": cfg.selection.metric,
            "higher_is_better": cfg.selection.higher_is_better,
            "bundle_enabled": cfg.bundle.enabled,
            "causal_forest_tune": getattr(cfg.causal_forest, "use_econml_tune", False),
            "reduce_memory": getattr(cfg.data_processing, "reduce_memory", True),
            "fill_na_method": getattr(cfg.data_processing, "fill_na_method", None),
        }

    def add_dataprep_info(self, x_dir) -> None:
        """Liest DataPrep-Konfiguration aus dem Datenverzeichnis."""
        from pathlib import Path
        p = Path(x_dir)
        try:
            import yaml
            dp_cfg_path = p / "dataprep_config.yml"
            if dp_cfg_path.is_file():
                raw = yaml.safe_load(dp_cfg_path.read_text(encoding="utf-8"))
                dp = raw.get("data_prep", raw) if isinstance(raw, dict) else {}
                self.dataprep_info = {
                    "data_files": dp.get("data_path", []),
                    "eval_files": dp.get("eval_data_path", []),
                    "target_column": dp.get("target_column", "Y"),
                    "treatment_column": dp.get("treatment_column", "T"),
                    "score_column": dp.get("score_column_name", ""),
                    "fill_na_method": dp.get("fill_na_method", None),
                    "fill_na_columns": dp.get("fill_na_columns", []),
                    "binary_target": dp.get("binary_target", False),
                    "deduplicate": dp.get("deduplicate", False),
                    "deduplicate_id_column": dp.get("deduplicate_id_column", None),
                    "balance_treatments": dp.get("balance_treatments", False),
                    "score_as_feature": dp.get("score_as_feature", False),
                    "feature_path": dp.get("feature_path", ""),
                    "multiple_files_option": dp.get("multiple_files_option", "merge"),
                    "delimiter": dp.get("delimiter", ","),
                    "output_path": dp.get("output_path", ""),
                    "log_to_mlflow": dp.get("log_to_mlflow", False),
                    "mlflow_experiment_name": dp.get("mlflow_experiment_name", ""),
                    "treatment_replacement": dp.get("treatment_replacement", {}),
                }
        except Exception:
            pass

    def add_data_stats(self, X, T, Y, S=None) -> None:
        """Berechnet Datenstatistiken."""
        t_arr = np.asarray(T).ravel()
        y_arr = np.asarray(Y).ravel()
        groups = sorted(np.unique(t_arr).tolist())
        treatment_dist = {f"T={int(g)}": int(np.sum(t_arr == g)) for g in groups}
        outcome_rates = {f"T={int(g)}": float(np.mean(y_arr[t_arr == g])) for g in groups}
        ate = float(np.mean(y_arr[t_arr == 1]) - np.mean(y_arr[t_arr == 0])) if len(groups) == 2 else None
        n_numeric = sum(1 for c in X.columns if pd.api.types.is_numeric_dtype(X[c]))
        self.data_stats = {
            "n_rows": len(X), "n_features": X.shape[1],
            "n_numeric": n_numeric, "n_categorical": X.shape[1] - n_numeric,
            "n_treatment_groups": len(groups),
            "treatment_distribution": treatment_dist, "outcome_rates": outcome_rates,
            "ate_diff_in_means": ate, "outcome_rate_overall": float(np.mean(y_arr)),
            "has_historical_score": S is not None,
        }

    def add_plot(self, model_name: str, plot_name: str, fig) -> None:
        try:
            self.model_plots.setdefault(model_name, {})[plot_name] = fig_to_base64(fig)
        except Exception:
            _logger.debug("Plot %s für %s konnte nicht konvertiert werden.", plot_name, model_name)

    def add_eval_data_stats(self, X_eval, T_eval, Y_eval, S_eval=None) -> None:
        """Berechnet Statistiken des externen Eval-Datensatzes."""
        t_arr = np.asarray(T_eval).ravel()
        y_arr = np.asarray(Y_eval).ravel()
        groups = sorted(np.unique(t_arr).tolist())
        treatment_dist = {f"T={int(g)}": int(np.sum(t_arr == g)) for g in groups}
        outcome_rates = {f"T={int(g)}": float(np.mean(y_arr[t_arr == g])) for g in groups}
        ate = float(np.mean(y_arr[t_arr == 1]) - np.mean(y_arr[t_arr == 0])) if len(groups) == 2 else None
        n_features = X_eval.shape[1] if hasattr(X_eval, "shape") else 0
        self.eval_data_stats = {
            "n_rows": len(X_eval), "n_features": n_features,
            "n_treatment_groups": len(groups),
            "treatment_distribution": treatment_dist, "outcome_rates": outcome_rates,
            "ate_diff_in_means": ate, "outcome_rate_overall": float(np.mean(y_arr)),
            "has_historical_score": S_eval is not None,
        }

    def add_explainability_plot(self, name: str, fig) -> None:
        try:
            self.explainability_plots[name] = fig_to_base64(fig)
        except Exception:
            _logger.debug("Explainability-Plot %s konnte nicht konvertiert werden.", name)

    def add_step_duration(self, step: str, seconds: float) -> None:
        self.step_durations[step] = seconds

    def add_fmt_info(self, info: Dict[str, Any]) -> None:
        self.fmt_info = info

    def add_tuning_plan(self, plan: List[Dict[str, Any]]) -> None:
        """Speichert den Tuning-Plan (Task-Sharing-Details).
        Jeder Eintrag: {task_key, role, models, signature, objective}.
        """
        self.tuning_plan = plan

    def add_fmt_plan(self, plan: List[Dict[str, Any]]) -> None:
        """Speichert den FMT-Plan (pro-Modell Methode).
        Jeder Eintrag: {model, method, studies, trials, fits_per_trial, total_fits, note}.
        """
        self.fmt_plan = plan

    def add_best_params(self, task_key: str, params: Dict[str, Any]) -> None:
        """Speichert die besten Hyperparameter für einen Tuning-Task."""
        self.best_params[task_key] = params

    def add_fmt_best_params(self, model: str, params: Dict[str, Any]) -> None:
        """Speichert die besten Hyperparameter für ein FMT-Modell."""
        self.fmt_best_params[model] = params


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

_PLOT_EXPLANATIONS = {
    "cate_distribution": ("CATE-Verteilung", "Histogramm der vorhergesagten Treatment-Effekte (Training + Cross-Validated). Zeigt, ob das Modell heterogene Effekte erkennt oder zum Intercept kollabiert."),
    "cal_plot": ("Calibration Plot", "Prüft, ob die vorhergesagten Treatment-Effekte kalibriert sind. Punkte nahe der Diagonale = gute Kalibrierung."),
    "qini_plot": ("Qini Plot (DRTester)", "Kumulierter Uplift bei absteigender Sortierung nach CATE. Je weiter über Random, desto besser."),
    "toc_plot": ("TOC Plot", "Durchschnittlicher Treatment-Effekt für die Top-k% der am höchsten bewerteten Kunden."),
    "sklift_qini": ("Qini-Kurve (scikit-uplift)", "Alternative Qini-Darstellung. Abstand zur Random-Linie = Qini-Koeffizient."),
    "sklift_percentile": ("Uplift by Percentile", "Durchschnittlicher Uplift pro Score-Dezil. Positive Balken links, null/negativ rechts = gute Sortierung."),
    "treatment_balance": ("Treatment Balance", "Treatment/Control-Verteilung über Score-Dezile. Starke Abweichungen → Propensity-Probleme."),
}
_PLOT_PREFIX_EXPLANATIONS = {
    "qini_vs_": ("Qini-Vergleich", "Gegenüberstellung der Qini-Kurven: Modell vs. Referenzscore. Abstand = Mehrwert des kausalen Modells."),
    "policy_compare_vs_": ("Policy-Value-Vergleich", "Inkrementeller Nutzen (Policy Value) des Modells vs. Referenzscore für verschiedene Treat-Anteile."),
}
_METRIC_EXPLANATIONS = {
    "qini": "Fläche unter der Uplift-Kurve ggü. Random – je höher, desto besser.",
    "auuc": "Area Under Uplift Curve – normierte Version des Qini.",
    "uplift_at_10pct": "Uplift bei Top-10% Targeting.",
    "uplift_at_20pct": "Uplift bei Top-20% Targeting.",
    "uplift_at_50pct": "Uplift bei Top-50% Targeting.",
    "policy_value_treat_positive": "Nur Kunden mit positivem CATE behandeln: erwarteter inkrementeller Nutzen.",
    "policy_value": "Globaler Policy Value (IPW): Optimale Zuweisung über alle Arme vs. Control.",
}
_METRIC_PREFIX_EXPLANATIONS = {
    "qini_T": "Qini für Treatment-Arm {arm} vs. Control.",
    "auuc_T": "AUUC für Treatment-Arm {arm} vs. Control.",
    "uplift10_T": "Uplift bei Top-10% für Arm {arm}.",
    "uplift20_T": "Uplift bei Top-20% für Arm {arm}.",
    "uplift50_T": "Uplift bei Top-50% für Arm {arm}.",
    "policy_value_treat_positive_T": "Nutzen bei Arm {arm}, wenn CATE > 0.",
}


def _get_metric_explanation(met: str) -> str:
    expl = _METRIC_EXPLANATIONS.get(met, "")
    if expl:
        return expl
    for prefix, template in _METRIC_PREFIX_EXPLANATIONS.items():
        if met.startswith(prefix):
            return template.format(arm=met[len(prefix):])
    return ""


def _fmt_dur(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}:{s % 60:04.1f}"


# ---------------------------------------------------------------------------
# HTML-Generator
# ---------------------------------------------------------------------------

def generate_html_report(collector: ReportCollector, output_path: str) -> str:
    """Generiert den HTML-Report und speichert ihn unter output_path."""
    sections_html = []
    nav_items = []
    sec_n = 0
    cs = collector.config_summary
    ds = collector.data_stats

    def _sec(sid: str, title: str, content: str):
        nonlocal sec_n
        sec_n += 1
        nav_items.append((sid, title, sec_n))
        sections_html.append(
            f'<section id="{sid}" class="rs">'
            f'<div class="rs-head"><span class="rs-num">{sec_n}</span><h2>{escape(title)}</h2></div>'
            f'{content}</section>'
        )

    # ── Executive Summary ──
    champ = collector.champion_name or "–"
    sel_met = cs.get("selection_metric", "qini")
    champ_score = collector.model_metrics.get(champ, {}).get(sel_met)
    n_models = len(collector.model_metrics) if collector.model_metrics else len(cs.get("models", []))
    is_external = cs.get("validate_on") == "external"
    eds = collector.eval_data_stats
    parts = [f'Champion: <strong>{escape(champ)}</strong>']
    if champ_score is not None:
        parts.append(f'{escape(sel_met)}: <strong>{champ_score:.4f}</strong>')
    parts.append(f'{n_models} Modelle verglichen')
    if is_external and eds.get("n_rows"):
        parts.append(f'Eval: {eds["n_rows"]:,} Beob.')
    if ds.get("ate_diff_in_means") is not None:
        parts.append(f'ATE: {ds["ate_diff_in_means"]:.4f}')
    if collector.total_elapsed > 0:
        parts.append(f'Laufzeit: {_fmt_dur(collector.total_elapsed)}')
    summary_html = '<div class="summary-bar"><div class="summary-items">' + ''.join(f'<div class="si">{p}</div>' for p in parts) + '</div></div>'

    # Champion callout
    if champ != "–" and champ_score is not None:
        higher = cs.get("higher_is_better", True)
        arrow = "▲" if higher else "▼"
        # Find runner-up
        runner_up = None
        runner_score = None
        for mn, mx in collector.model_metrics.items():
            if mn == champ:
                continue
            s = mx.get(sel_met)
            if s is not None and (runner_score is None or (higher and s > runner_score) or (not higher and s < runner_score)):
                runner_up = mn
                runner_score = s
        diff_html = ""
        if runner_up and runner_score is not None:
            diff = abs(champ_score - runner_score)
            diff_html = f'<div class="champ-diff">+{diff:.4f} vs. {escape(runner_up)} (Zweitplatziert)</div>'
        summary_html += (
            f'<div class="champ-callout">'
            f'<div class="champ-callout-inner">'
            f'<div class="champ-callout-label">Champion</div>'
            f'<div class="champ-callout-name">{escape(champ)}</div>'
            f'<div class="champ-callout-score">{escape(sel_met)} {arrow} {champ_score:.4f}</div>'
            f'{diff_html}'
            f'</div></div>'
        )

    # ── 1. Übersicht ──
    exp_name = cs.get("experiment_name", "–")
    run_name = collector.run_name or ""
    h = ''
    # Experiment-Name prominent als große Kachel
    h += '<div style="margin:14px 28px;padding:18px 24px;background:linear-gradient(135deg,var(--ruby-d),var(--ruby));border-radius:var(--r);color:#fff">'
    h += f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;opacity:.6;margin-bottom:4px">Experiment</div>'
    h += f'<div style="font-size:24px;font-weight:700;letter-spacing:.3px">{escape(exp_name)}</div>'
    if run_name:
        h += f'<div style="font-size:13px;opacity:.65;margin-top:4px">Run: {escape(run_name)}</div>'
    h += '</div>'
    h += '<div class="cg">'
    for lbl, val in [
        ("Base Learner", cs.get("base_learner", "–")),
        ("Validierung", f'external (separater Datensatz)' if cs.get("validate_on") == "external" else f'{cs.get("validate_on","cross")} ({cs.get("cv_splits",5)} Folds)'),
        ("Treatment", cs.get("treatment_type", "binary")),
        ("Features", f'{ds.get("n_features","–")} ({ds.get("n_numeric","?")}N / {ds.get("n_categorical","?")}K)'),
        ("Selektionsmetrik", cs.get("selection_metric", "–")),
        ("Seed", cs.get("seed", "–")),
        ("Parallel", f'Level {cs.get("parallel_level", 2)}'),
    ]:
        v_str = str(val)
        v_cls = "cd-v sm" if len(v_str) > 12 else "cd-v"
        h += f'<div class="cd"><div class="cd-l">{escape(str(lbl))}</div><div class="{v_cls}">{escape(v_str)}</div></div>'
    # Beobachtungen-Kachel: bei external Train+Eval zusammen
    if is_external and eds.get("n_rows"):
        h += (
            '<div class="cd"><div class="cd-l">Beobachtungen</div><div class="cd-v sm">'
            f'<div style="line-height:1.6">Train: <strong>{ds.get("n_rows",0):,}</strong></div>'
            f'<div style="line-height:1.6">Eval: <strong>{eds.get("n_rows",0):,}</strong></div>'
            '</div></div>'
        )
    else:
        n = ds.get("n_rows", 0)
        h += f'<div class="cd"><div class="cd-l">Beobachtungen</div><div class="cd-v">{n:,}</div></div>' if isinstance(n, int) else ''
    h += '</div>'
    if cs.get("models"):
        h += '<h3>Trainierte Modelle</h3><div class="tags">'
        for m in cs["models"]:
            cls = "tag champ" if m == champ else "tag"
            h += f'<span class="{cls}">{escape(m)}{"&thinsp;★" if m == champ else ""}</span>'
        h += '</div>'
    steps = []
    if cs.get("feature_selection_enabled"):
        steps.append("Feature-Selektion")
    if cs.get("tuning_enabled"):
        sf = " (SF)" if cs.get("tuning_single_fold") else ""
        steps.append(f'Tuning ({cs.get("tuning_trials",0)}T{sf})')
    if cs.get("final_tuning_enabled"):
        fm = cs.get("final_tuning_models")
        steps.append(f'FMT ({", ".join(fm) if fm else "alle"})')
    if cs.get("causal_forest_tune"):
        steps.append("CF tune()")
    if cs.get("surrogate_enabled"):
        steps.append("Surrogate")
    if cs.get("bundle_enabled"):
        steps.append("Bundle")
    if steps:
        h += '<h3>Pipeline</h3><div class="tags">' + ''.join(f'<span class="tag step">{escape(s)}</span>' for s in steps) + '</div>'
    if is_external:
        h += '<div style="margin-top:14px;padding:12px 16px;background:#fffbeb;border:1px solid #e8d49c;border-radius:8px;font-size:13px;line-height:1.5;color:#7a5a00"><strong>Externe Validierung:</strong> Tuning und Feature-Selektion liefen intern auf den Trainingsdaten (CV). Alle Modelle wurden auf 100% der Trainingsdaten nachtrainiert und auf dem separaten Eval-Datensatz evaluiert. Metriken und Diagnose-Plots basieren ausschließlich auf den Eval-Daten.</div>'
    _sec("overview", "Übersicht", h)

    # ── 2. Datengrundlage ──
    h = '<p class="expl">Statistische Grunddaten. Der ATE (Diff. in Means) ist ein erster, unbereinigter Hinweis auf den Gesamteffekt.</p>'
    if is_external:
        h += '<h3>Trainingsdaten</h3>'
    h += '<div class="cg">'
    if ds.get("ate_diff_in_means") is not None:
        h += f'<div class="cd hl"><div class="cd-l">ATE (Diff. in Means)</div><div class="cd-v">{ds["ate_diff_in_means"]:.6f}</div></div>'
    h += f'<div class="cd"><div class="cd-l">Outcome-Rate</div><div class="cd-v">{ds.get("outcome_rate_overall",0):.4f}</div></div>'
    if ds.get("has_historical_score"):
        h += '<div class="cd"><div class="cd-l">Hist. Score</div><div class="cd-v">vorhanden</div></div>'
    h += '</div>'
    if ds.get("treatment_distribution"):
        total_n = ds.get("n_rows", 1)
        h += '<h3>Treatment-Verteilung' + (' (Train)' if is_external else '') + '</h3><div class="tbl-scroll"><table class="dt"><thead><tr><th>Gruppe</th><th>Anzahl</th><th>Anteil</th><th>Outcome</th></tr></thead><tbody>'
        for grp, n in ds["treatment_distribution"].items():
            rate = ds.get("outcome_rates", {}).get(grp, 0)
            h += f'<tr><td>{escape(grp)}</td><td>{n:,}</td><td>{n/total_n*100:.1f}%</td><td>{rate:.4f}</td></tr>'
        h += '</tbody></table></div>'
    # Eval-Daten-Statistiken (nur bei external)
    if is_external and eds:
        h += '<h3>Evaluationsdaten (separater Datensatz)</h3><div class="cg">'
        if eds.get("ate_diff_in_means") is not None:
            h += f'<div class="cd hl"><div class="cd-l">ATE (Diff. in Means)</div><div class="cd-v">{eds["ate_diff_in_means"]:.6f}</div></div>'
        h += f'<div class="cd"><div class="cd-l">Beobachtungen</div><div class="cd-v">{eds.get("n_rows",0):,}</div></div>'
        h += f'<div class="cd"><div class="cd-l">Outcome-Rate</div><div class="cd-v">{eds.get("outcome_rate_overall",0):.4f}</div></div>'
        if eds.get("has_historical_score"):
            h += '<div class="cd"><div class="cd-l">Hist. Score</div><div class="cd-v">vorhanden</div></div>'
        h += '</div>'
        if eds.get("treatment_distribution"):
            eval_n = eds.get("n_rows", 1)
            h += '<h3>Treatment-Verteilung (Eval)</h3><div class="tbl-scroll"><table class="dt"><thead><tr><th>Gruppe</th><th>Anzahl</th><th>Anteil</th><th>Outcome</th></tr></thead><tbody>'
            for grp, n in eds["treatment_distribution"].items():
                rate = eds.get("outcome_rates", {}).get(grp, 0)
                h += f'<tr><td>{escape(grp)}</td><td>{n:,}</td><td>{n/eval_n*100:.1f}%</td><td>{rate:.4f}</td></tr>'
            h += '</tbody></table></div>'
        # Train vs. Eval Vergleich
        if ds.get("outcome_rate_overall") and eds.get("outcome_rate_overall"):
            diff_rate = abs(ds["outcome_rate_overall"] - eds["outcome_rate_overall"])
            diff_ate = None
            if ds.get("ate_diff_in_means") is not None and eds.get("ate_diff_in_means") is not None:
                diff_ate = abs(ds["ate_diff_in_means"] - eds["ate_diff_in_means"])
            if diff_rate > 0.05 or (diff_ate is not None and diff_ate > 0.02):
                h += '<div style="margin-top:10px;padding:10px 14px;background:#fff3cd;border:1px solid #ffc107;border-radius:8px;font-size:12.5px;color:#856404;line-height:1.5"><strong>Hinweis:</strong> Die Outcome-Rate oder der ATE unterscheiden sich deutlich zwischen Train und Eval. Dies kann auf einen Covariate Shift oder zeitliche Veränderungen hindeuten. Die Eval-Metriken sind dennoch valide – sie spiegeln die Modellperformance auf den tatsächlichen Zieldaten wider.</div>'
    _sec("data", "Datengrundlage", h)

    # ── 2b. Datenaufbereitung (wenn DataPrep-Info vorhanden) ──
    dpi = collector.dataprep_info
    if dpi:
        h = '<p class="expl">Einstellungen der Datenaufbereitung, die vor der Analyse durchgeführt wurde.</p>'
        h += '<div class="cg">'
        # Quelldateien
        data_files = dpi.get("data_files", [])
        if isinstance(data_files, str):
            data_files = [data_files]
        if data_files:
            files_str = ", ".join(str(f).rsplit("/", 1)[-1] for f in data_files[:5])
            if len(data_files) > 5:
                files_str += f" (+{len(data_files)-5})"
            h += f'<div class="cd"><div class="cd-l">Quelldateien</div><div class="cd-v sm">{escape(files_str)}</div></div>'
        h += f'<div class="cd"><div class="cd-l">Zielspalte</div><div class="cd-v">{escape(str(dpi.get("target_column", "Y")))}</div></div>'
        h += f'<div class="cd"><div class="cd-l">Treatment-Spalte</div><div class="cd-v">{escape(str(dpi.get("treatment_column", "T")))}</div></div>'
        if dpi.get("score_column"):
            h += f'<div class="cd"><div class="cd-l">Score-Spalte</div><div class="cd-v">{escape(str(dpi["score_column"]))}</div></div>'
        if dpi.get("feature_path"):
            h += f'<div class="cd"><div class="cd-l">Feature-Dictionary</div><div class="cd-v sm">{escape(str(dpi["feature_path"]).rsplit("/", 1)[-1])}</div></div>'
        h += '</div>'
        # Verarbeitungsschritte
        processing_steps = []
        fill_na = dpi.get("fill_na_method")
        if fill_na and fill_na != "none":
            na_cols = dpi.get("fill_na_columns", [])
            detail = f" ({len(na_cols)} Spalten)" if na_cols else ""
            processing_steps.append(f'NaN-Behandlung: {fill_na}{detail}')
        if dpi.get("binary_target"):
            processing_steps.append("Binäres Target")
        if dpi.get("deduplicate"):
            col = dpi.get("deduplicate_id_column")
            processing_steps.append(f'Deduplizierung' + (f' ({col})' if col else ''))
        if dpi.get("balance_treatments"):
            processing_steps.append("Treatment-Balance (Downsampling)")
        if dpi.get("score_as_feature"):
            processing_steps.append("Score als Feature")
        if dpi.get("treatment_replacement"):
            tr = dpi["treatment_replacement"]
            mapping = ", ".join(f'{k}→{v}' for k, v in tr.items()) if isinstance(tr, dict) else str(tr)
            processing_steps.append(f'Treatment-Mapping: {mapping}')
        multi = dpi.get("multiple_files_option", "merge")
        if multi and multi != "merge" and len(data_files) > 1:
            processing_steps.append(f'Mehrere Dateien: {multi}')
        eval_files = dpi.get("eval_files", [])
        if eval_files:
            processing_steps.append(f'{len(eval_files)} Eval-Datei(en)')
        if dpi.get("log_to_mlflow"):
            processing_steps.append("MLflow-Logging")
        if processing_steps:
            h += '<h3>Verarbeitungsschritte</h3><div class="tags">'
            for step in processing_steps:
                h += f'<span class="tag step">{escape(step)}</span>'
            h += '</div>'
        elif not dpi.get("data_files"):
            pass  # No data prep info at all
        else:
            h += '<h3>Verarbeitungsschritte</h3><div style="margin:8px 28px;font-size:13px;color:var(--text-l)">Standard-Verarbeitung (keine besonderen Schritte konfiguriert)</div>'

        # NaN-Handling aus der Analyse-Config (falls DataPrep keins hatte)
        analysis_fill = cs.get("fill_na_method")
        if analysis_fill and analysis_fill != "none" and not fill_na:
            h += f'<div style="margin:8px 28px;padding:10px 14px;background:#fffbeb;border:1px solid #e8d49c;border-radius:8px;font-size:12.5px;color:#7a5a00;line-height:1.5"><strong>Analyse-Pipeline:</strong> NaN-Behandlung via {escape(str(analysis_fill))}</div>'

        _sec("dataprep", "Datenaufbereitung", h)

    # ── 3. Feature-Selektion ──
    if cs.get("feature_selection_enabled"):
        fs = collector.feature_selection_info
        h = '<p class="expl">Union-Methode: Top-X% aus jeder Importance-Methode werden vereinigt.</p><div class="cg">'
        methods_html = "".join(f'<span class="cd-pill">{escape(m)}</span>' for m in cs.get("feature_selection_methods", []))
        h += f'<div class="cd"><div class="cd-l">Methoden</div><div class="cd-v pills">{methods_html}</div></div>'
        h += f'<div class="cd"><div class="cd-l">Top-%</div><div class="cd-v">{cs.get("feature_selection_top_pct","–")}%</div></div>'
        if fs.get("n_before") and fs.get("n_after"):
            red = (1 - fs["n_after"] / fs["n_before"]) * 100 if fs["n_before"] > 0 else 0
            h += f'<div class="cd"><div class="cd-l">Features</div><div class="cd-v">{fs["n_before"]} → {fs["n_after"]} (−{red:.0f}%)</div></div>'
        h += '</div>'
        _sec("feature_sel", "Feature-Selektion", h)

    # ── 4. Base-Learner-Tuning ──
    if cs.get("tuning_enabled") and collector.tuning_scores:
        n_studies = len(collector.tuning_scores)
        n_trials = cs.get("tuning_trials", 30)
        sf = cs.get("tuning_single_fold", False)
        cv = 1 if sf else cs.get("cv_splits", 5)
        total_fits = n_studies * n_trials * cv

        h = '<p class="expl">Optuna optimiert die Nuisance-Modelle (Outcome, Propensity). Die Best-Scores messen die Hilfsmodell-Güte auf dem internen Validierungsset – sie messen nicht den kausalen Effekt selbst, sondern wie gut das Outcome bzw. die Treatment-Zuordnung vorhergesagt wird.</p>'

        # Summary cards (like App TuningPlanPreview)
        h += '<div class="cg">'
        h += f'<div class="cd"><div class="cd-l">Optuna-Studies</div><div class="cd-v">{n_studies}</div></div>'
        h += f'<div class="cd"><div class="cd-l">Trials / Study</div><div class="cd-v">{n_trials}</div></div>'
        h += f'<div class="cd"><div class="cd-l">CV-Folds / Trial</div><div class="cd-v">{cv}{"&ensp;(SF)" if sf else ""}</div></div>'
        h += f'<div class="cd hl"><div class="cd-l">Modell-Fits gesamt</div><div class="cd-v">{total_fits:,}</div></div>'
        h += '</div>'

        # Calculation formula
        sf_note = " *(Single-Fold aktiv)*" if sf else ""
        h += f'<p style="font-size:13px;color:var(--text-l);margin:4px 28px 14px;font-family:var(--mono)">{n_studies} Studies × {n_trials} Trials × {cv} CV = <strong style="color:var(--ruby)">{total_fits:,} Fits</strong>{sf_note}</p>'

        # Config info
        h += '<div class="cg">'
        h += f'<div class="cd"><div class="cd-l">Metrik</div><div class="cd-v">{escape(str(cs.get("tuning_metric","log_loss")))}</div></div>'
        h += f'<div class="cd"><div class="cd-l">Base Learner</div><div class="cd-v">{escape(str(cs.get("base_learner","lgbm")))}</div></div>'
        if sf:
            h += '<div class="cd"><div class="cd-l">Single-Fold</div><div class="cd-v">aktiv (1 Fold/Trial)</div></div>'
        h += '</div>'

        # Task-sharing plan (if provided)
        if collector.tuning_plan:
            h += '<h3>Tuning-Tasks (Task-Sharing)</h3>'
            h += '<p class="expl">Tasks mit gleicher Signatur werden geteilt – gleiche Nuisance-Aufgaben werden nur einmal optimiert.'
            total_roles = sum(len(t.get("models", [])) for t in collector.tuning_plan)
            if total_roles > n_studies:
                h += f' <strong>{total_roles - n_studies} Tuning-Läufe eingespart.</strong>'
            h += '</p>'
            h += '<div class="tbl-scroll"><table class="dt"><thead><tr><th>Task</th><th>Rolle</th><th>Geteilt von</th><th>Signatur</th></tr></thead><tbody>'
            for t in collector.tuning_plan:
                models_str = ", ".join(t.get("models", []))
                h += f'<tr><td><code>{escape(t.get("task_key",""))}</code></td>'
                h += f'<td>{escape(t.get("role",""))}</td>'
                h += f'<td>{escape(models_str)}</td>'
                h += f'<td><code style="font-size:11px">{escape(t.get("signature",""))}</code></td></tr>'
            h += '</tbody></table></div>'

        # Best scores table
        h += '<h3>Best Scores pro Task</h3><div class="tbl-scroll"><table class="dt"><thead><tr><th>Task</th><th>Best Score</th></tr></thead><tbody>'
        for task, score in sorted(collector.tuning_scores.items()):
            h += f'<tr><td><code>{escape(task)}</code></td><td>{score:.6f}</td></tr>'
        h += '</tbody></table></div>'

        # Best hyperparameters (if provided)
        if collector.best_params:
            h += '<details class="detail-box"><summary>Gefundene Hyperparameter</summary><div class="detail-content">'
            for task_key, params in sorted(collector.best_params.items()):
                h += f'<h4 style="margin-top:10px"><code>{escape(task_key)}</code></h4>'
                h += '<div class="tbl-scroll" style="margin:4px 0 10px"><table class="dt"><thead><tr><th>Parameter</th><th>Wert</th></tr></thead><tbody>'
                for pk, pv in sorted(params.items()):
                    h += f'<tr><td><code>{escape(pk)}</code></td><td>{escape(str(pv))}</td></tr>'
                h += '</tbody></table></div>'
            h += '</div></details>'

        _sec("tuning", "Base-Learner-Tuning", h)

    # ── 5. Final-Model-Tuning ──
    if cs.get("final_tuning_enabled"):
        fmt = collector.fmt_info
        fmt_trials = cs.get("final_tuning_trials", 20)
        fmt_models = cs.get("final_tuning_models")
        fmt_sf = cs.get("final_tuning_single_fold", False)
        fmt_cv = cs.get("cv_splits", 5)

        h = '<p class="expl">Optimiert das CATE-Effektmodell (model_final) über eine Residual-basierte Zielfunktion (R-Score / R-Loss). Das Tuning läuft einmalig auf dem 1. CV-Fold; die Parameter werden für alle weiteren Folds übernommen (Locking).</p>'

        # Summary cards
        h += '<div class="cg">'
        h += f'<div class="cd"><div class="cd-l">Trials / Modell</div><div class="cd-v">{fmt_trials}</div></div>'
        if fmt_models:
            fm_pills = "".join(f'<span class="cd-pill">{escape(m)}</span>' for m in fmt_models)
            h += f'<div class="cd"><div class="cd-l">Modelle</div><div class="cd-v pills">{fm_pills}</div></div>'
        else:
            h += '<div class="cd"><div class="cd-l">Modelle</div><div class="cd-v sm">alle FMT-fähigen</div></div>'
        h += f'<div class="cd"><div class="cd-l">Methode</div><div class="cd-v">R-Score</div></div>'
        if fmt_sf:
            h += '<div class="cd"><div class="cd-l">DR Single-Fold</div><div class="cd-v">aktiv</div></div>'
        h += '</div>'

        # FMT plan table (like App FinalTuningPlanPreview)
        if collector.fmt_plan:
            total_fmt_fits = sum(r.get("total_fits", 0) for r in collector.fmt_plan)
            h += '<h3>Tuning-Plan pro Modell</h3>'
            h += '<div class="tbl-scroll"><table class="dt"><thead><tr><th>Modell</th><th>Methode</th><th>Trials</th><th>Fits/Trial</th><th>Total Fits</th></tr></thead><tbody>'
            for r in collector.fmt_plan:
                method_badge = ""
                method = r.get("method", "")
                if "RScorer" in method:
                    method_badge = '<span style="background:var(--green);color:#fff;padding:1px 6px;border-radius:4px;font-size:10px;margin-left:4px">effizient</span>'
                h += f'<tr><td><strong>{escape(r.get("model",""))}</strong></td>'
                h += f'<td>{escape(method)}{method_badge}</td>'
                h += f'<td>{r.get("trials",0)}</td>'
                h += f'<td>{r.get("fits_per_trial",0)}</td>'
                h += f'<td><strong>{r.get("total_fits",0):,}</strong></td></tr>'
            h += '</tbody></table></div>'
            # Explanation per model
            for r in collector.fmt_plan:
                if r.get("note"):
                    h += f'<p style="font-size:12.5px;color:var(--text-l);margin:2px 28px 8px;line-height:1.5"><strong>{escape(r.get("model",""))}</strong>: {escape(r.get("note",""))}</p>'
            h += f'<p style="font-size:13px;margin:8px 28px;font-family:var(--mono);color:var(--text-l)">Gesamt: <strong style="color:var(--gold)">{total_fmt_fits:,} Fits</strong></p>'
        elif not collector.fmt_plan:
            # Fallback: show basic info without plan
            h += '<p class="expl"><strong>NonParamDML</strong> nutzt RScorer (1 Fit/Trial – Residuen werden vorab berechnet). <strong>DRLearner</strong> nutzt score() + CV ({cv} Fits/Trial). Single-Fold reduziert DRLearner auf 1 Fit/Trial.</p>'.format(cv=fmt_cv)

        # Best R-Scores
        if fmt.get("best_scores"):
            # Separate raw R-Scores from penalized scores
            raw_scores = {k: v for k, v in fmt["best_scores"].items() if "__penalized" not in k}
            pen_scores = {k.replace("__penalized", ""): v for k, v in fmt["best_scores"].items() if "__penalized" in k}
            h += '<h3>Best R-Scores</h3><div class="tbl-scroll"><table class="dt"><thead><tr><th>Modell</th><th>R-Score</th>'
            if pen_scores:
                h += '<th>Penalisiert (λ·log(1+CV))</th>'
            h += '</tr></thead><tbody>'
            for model, score in raw_scores.items():
                h += f'<tr><td><strong>{escape(model)}</strong></td><td>{score:.6f}</td>'
                if pen_scores:
                    pen = pen_scores.get(model)
                    h += f'<td>{pen:.6f}</td>' if pen is not None else '<td>–</td>'
                h += '</tr>'
            h += '</tbody></table></div>'

        # Best hyperparameters
        if collector.fmt_best_params:
            h += '<details class="detail-box"><summary>Gefundene Hyperparameter (Final-Modell)</summary><div class="detail-content">'
            for model, params in sorted(collector.fmt_best_params.items()):
                h += f'<h4 style="margin-top:10px"><strong>{escape(model)}</strong></h4>'
                h += '<div class="tbl-scroll" style="margin:4px 0 10px"><table class="dt"><thead><tr><th>Parameter</th><th>Wert</th></tr></thead><tbody>'
                for pk, pv in sorted(params.items()):
                    h += f'<tr><td><code>{escape(pk)}</code></td><td>{escape(str(pv))}</td></tr>'
                h += '</tbody></table></div>'
            h += '</div></details>'

        _sec("fmt", "Final-Model-Tuning", h)

    # ── 6. Modellvergleich ──
    if collector.model_metrics:
        higher = cs.get("higher_is_better", True)
        all_mets = sorted({k for m in collector.model_metrics.values() for k in m if isinstance(m.get(k), (int, float))})

        # Compute best value per metric
        best_per_met = {}
        for met in all_mets:
            vals = [(mn, mx.get(met)) for mn, mx in collector.model_metrics.items() if isinstance(mx.get(met), float)]
            if vals:
                best_per_met[met] = max(vals, key=lambda x: x[1] if higher else -x[1])[1]

        h = '<p class="expl">' + ('Uplift-Metriken basierend auf dem <strong>externen Evaluationsdatensatz</strong>. Modelle wurden auf den Trainingsdaten trainiert und hier auf ungesehenen Daten evaluiert.' if is_external else 'Uplift-Metriken basierend auf Cross-Predictions (Out-of-Fold).') + ' Champion ist hervorgehoben. <span class="best-marker-legend">Bester Wert</span> je Metrik ist markiert.</p>'

        # Table with best-value highlighting
        h += '<div class="tbl-scroll"><table class="dt"><thead><tr><th>Modell</th>'
        for met in all_mets:
            arrow = " ▲" if higher else " ▼"
            is_sel = met == sel_met
            cls = ' class="sel-met"' if is_sel else ""
            h += f'<th{cls}>{escape(met)}{arrow}</th>'
        h += '</tr></thead><tbody>'
        order = sorted(collector.model_metrics.keys(), key=lambda m: (m != champ, m))
        for mn in order:
            mx = collector.model_metrics[mn]
            ic = mn == champ
            h += f'<tr{" class=\"champ-row\"" if ic else ""}><td><strong>{escape(mn)}</strong>{" <span class=\"badge-c\">Champion</span>" if ic else ""}</td>'
            for met in all_mets:
                v = mx.get(met)
                if isinstance(v, float):
                    is_best = best_per_met.get(met) is not None and abs(v - best_per_met[met]) < 1e-9
                    cls = ' class="best-val"' if is_best else ""
                    h += f'<td{cls}>{v:.6f}</td>'
                else:
                    h += '<td class="na">–</td>'
            h += '</tr>'
        h += '</tbody></table></div></div>'

        # Ranking mini-bars for selection metric
        sel_vals = [(mn, collector.model_metrics[mn].get(sel_met, 0)) for mn in order if isinstance(collector.model_metrics[mn].get(sel_met), float)]
        if sel_vals:
            max_val = max(abs(v) for _, v in sel_vals) or 1
            h += f'<h3>Ranking ({escape(sel_met)})</h3>'
            h += '<div class="rank-bars">'
            for i, (mn, v) in enumerate(sorted(sel_vals, key=lambda x: -x[1] if higher else x[1])):
                pct = abs(v) / max_val * 100
                ic = mn == champ
                bar_cls = "rank-bar champ" if ic else "rank-bar"
                medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
                h += (
                    f'<div class="{bar_cls}">'
                    f'<span class="rank-pos">{medal}</span>'
                    f'<span class="rank-name">{escape(mn)}</span>'
                    f'<div class="rank-track"><div class="rank-fill" style="width:{pct:.1f}%"></div></div>'
                    f'<span class="rank-val">{v:.4f}</span>'
                    f'</div>'
                )
            h += '</div>'

        h += '<details class="detail-box"><summary>Metriken erklärt</summary><div class="detail-content"><dl>'
        for met in all_mets:
            ex = _get_metric_explanation(met)
            if ex:
                h += f'<dt><code>{escape(met)}</code></dt><dd>{escape(ex)}</dd>'
        h += '</dl></div></details>'
        _sec("comparison", "Modellvergleich", h)

    # ── 7. Modell-Details ──
    for mn, plots in collector.model_plots.items():
        if not plots:
            continue
        ic = mn == champ
        eval_note = " Plots basieren auf dem externen Eval-Datensatz." if is_external else ""
        h = f'<p class="expl">Diagnose-Plots für <strong>{escape(mn)}</strong>{"&ensp;<span class=\"badge-c\">Champion</span>" if ic else ""}.{eval_note}</p><div class="plot-grid">'
        for pk, b64 in plots.items():
            title, expl = _PLOT_EXPLANATIONS.get(pk, (None, None))
            if title is None:
                for pfx, (pt, pe) in _PLOT_PREFIX_EXPLANATIONS.items():
                    if pk.startswith(pfx):
                        title, expl = pt, pe
                        break
                else:
                    title, expl = pk.replace("_", " ").title(), ""
            h += f'<div class="plot-card"><h4>{escape(title)}</h4>'
            if expl:
                h += f'<p class="plot-expl">{escape(expl)}</p>'
            h += f'<img src="data:image/png;base64,{b64}" alt="{escape(title)}" loading="lazy"></div>'
        h += '</div>'
        _sec(f"model_{mn}", f"Modell: {mn}", h)

    # ── 8. Surrogate ──
    if collector.surrogate_info:
        si = collector.surrogate_info
        h = '<p class="expl">Surrogate-Einzelbaum: interpretierbare Nachbildung der Champion-CATEs (Teacher-Learner).</p><div class="cg">'
        h += f'<div class="cd"><div class="cd-l">Teacher</div><div class="cd-v">{escape(str(si.get("champion","–")))}</div></div>'
        if si.get("depth"):
            h += f'<div class="cd"><div class="cd-l">Tiefe</div><div class="cd-v">{si["depth"]}</div></div>'
        if si.get("n_leaves"):
            h += f'<div class="cd"><div class="cd-l">Blätter</div><div class="cd-v">{si["n_leaves"]}</div></div>'
        h += '</div>'
        if si.get("metrics"):
            cm = collector.model_metrics.get(champ, {})
            h += '<h3>Surrogate vs. Champion</h3><div class="tbl-scroll"><table class="dt"><thead><tr><th>Metrik</th><th>Champion</th><th>Surrogate</th><th>Retention</th></tr></thead><tbody>'
            for k, v in si["metrics"].items():
                cv = cm.get(k)
                ret_pct = v / cv * 100 if isinstance(cv, float) and cv != 0 else 0
                ret_color = "var(--green)" if ret_pct >= 80 else "var(--gold)" if ret_pct >= 60 else "var(--ruby)"
                ret_bar = f'<div class="bar-bg" style="width:80px"><div class="bar-fill" style="width:{min(ret_pct,100)}%;background:{ret_color}"></div></div><span class="bar-pct">{ret_pct:.0f}%</span>' if ret_pct > 0 else "–"
                h += f'<tr><td><code>{escape(k)}</code></td><td>{f"{cv:.6f}" if isinstance(cv, float) else "–"}</td><td>{v:.6f}</td><td>{ret_bar}</td></tr>'
            h += '</tbody></table></div>'
        _sec("surrogate", "Surrogate-Einzelbaum", h)

    # ── 9. Explainability ──
    if collector.explainability_plots:
        h = '<p class="expl">Feature-Importances auf Uplift-Ebene: welche Features die Heterogenität des Treatment-Effekts treiben.</p><div class="plot-grid">'
        for pn, b64 in collector.explainability_plots.items():
            h += f'<div class="plot-card"><h4>{escape(pn)}</h4><img src="data:image/png;base64,{b64}" alt="{escape(pn)}" loading="lazy"></div>'
        h += '</div>'
        _sec("explainability", "Explainability", h)

    # ── 10. Laufzeiten ──
    if collector.step_durations:
        total = collector.total_elapsed or sum(collector.step_durations.values())
        max_dur = max(collector.step_durations.values()) if collector.step_durations else 0
        h = '<div class="tbl-scroll"><table class="dt"><thead><tr><th>Schritt</th><th>Dauer</th><th>Anteil</th></tr></thead><tbody>'
        for step, dur in collector.step_durations.items():
            pct = dur / total * 100 if total > 0 else 0
            is_max = abs(dur - max_dur) < 0.01
            row_cls = ' class="champ-row"' if is_max else ""
            bold_s, bold_e = ("<strong>", "</strong>") if is_max else ("", "")
            h += f'<tr{row_cls}><td>{bold_s}{escape(step)}{bold_e}</td><td>{bold_s}{_fmt_dur(dur)}{bold_e}</td><td><div class="bar-bg"><div class="bar-fill" style="width:{min(pct,100)}%"></div></div><span class="bar-pct">{pct:.1f}%</span></td></tr>'
        h += '</tbody></table></div>'
        if total > 0:
            h += f'<div style="margin-top:10px;font-size:14px;font-weight:600;color:var(--ruby-d)">Gesamt: {_fmt_dur(total)}</div>'
        _sec("timing", "Laufzeiten", h)

    # ── Assemble ──
    nav_html = "\n".join(f'<a href="#{sid}" data-section="{sid}"><span class="nav-num">{n}</span>{escape(t)}</a>' for sid, t, n in nav_items)
    body_html = summary_html + "\n" + "\n".join(sections_html)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = _HTML_TEMPLATE.replace("{{NAV}}", nav_html).replace("{{BODY}}", body_html).replace("{{TIMESTAMP}}", ts).replace("{{CHAMPION}}", escape(champ)).replace("{{EXPERIMENT}}", escape(cs.get("experiment_name", "rubin")))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    _logger.info("HTML-Report geschrieben: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>rubin – Analyse-Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
@font-face{font-family:'DM Sans';src:local('DM Sans'),local('DMSans')}
@font-face{font-family:'JetBrains Mono';src:local('JetBrains Mono'),local('JetBrainsMono')}
:root{--ruby:#9B111E;--ruby-l:#C4343F;--ruby-d:#6B0D15;--ruby-pale:#FDF2F3;--ruby-bg:#faf7f8;--gold:#D4A853;--green:#1a7f37;--text:#24292f;--text-l:#57606a;--text-f:#999;--border:#e0d6d7;--card:#fff;--code-bg:#f8f0f0;--sh:0 1px 4px rgba(155,17,30,.06),0 4px 20px rgba(155,17,30,.04);--sh-lg:0 4px 24px rgba(107,13,21,.1),0 1px 6px rgba(0,0,0,.04);--r:10px;--font:'DM Sans',system-ui,-apple-system,sans-serif;--mono:'JetBrains Mono',ui-monospace,monospace}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{font-family:var(--font);font-size:15px;line-height:1.7;color:var(--text);background:var(--ruby-bg)}
.topbar{position:fixed;top:0;left:0;right:0;z-index:100;background:linear-gradient(135deg,var(--ruby-d) 0%,var(--ruby) 55%,var(--ruby-l) 100%);height:60px;display:flex;align-items:center;padding:0 32px;box-shadow:0 2px 16px rgba(107,13,21,.3)}
.topbar-logo{margin-right:14px;flex-shrink:0}
.topbar h1{color:#fff;font-size:22px;font-weight:700;letter-spacing:.5px}
.topbar .sub{color:rgba(255,255,255,.55);font-size:13px;margin-left:16px;font-weight:400}
.topbar .sub strong{color:rgba(255,255,255,.85);font-weight:600}
.layout{display:flex;margin-top:60px;min-height:calc(100vh - 60px)}
.sidebar{width:260px;min-width:260px;background:var(--card);border-right:1px solid var(--border);position:sticky;top:60px;height:calc(100vh - 60px);overflow-y:auto;padding:20px 0}
.sidebar a{display:flex;align-items:center;gap:8px;padding:8px 20px;color:var(--text-l);text-decoration:none;font-size:13.5px;border-left:3px solid transparent;transition:all .15s}
.sidebar a:hover{background:var(--ruby-pale);color:var(--ruby)}
.sidebar a.active{border-left-color:var(--ruby);color:var(--ruby-d);font-weight:600;background:var(--ruby-pale)}
.nav-num{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:6px;background:var(--ruby-pale);color:var(--ruby);font-size:11px;font-weight:700;flex-shrink:0}
.sidebar a.active .nav-num{background:var(--ruby);color:#fff}
.sidebar .nav-lbl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:var(--text-f);padding:16px 20px 6px}
.main{flex:1;max-width:960px;padding:32px 48px 80px}
.summary-bar{background:linear-gradient(135deg,var(--ruby-d),var(--ruby));border-radius:var(--r);padding:20px 28px;margin-bottom:24px;box-shadow:var(--sh-lg);position:relative;overflow:hidden}
.summary-bar::after{content:'';position:absolute;right:-20px;top:-20px;width:140px;height:140px;background:rgba(255,255,255,0.04);clip-path:polygon(50% 0%,85% 25%,85% 75%,50% 100%,15% 75%,15% 25%);animation:gem-spin 80s linear infinite}
@keyframes gem-spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
.summary-items{display:flex;flex-wrap:wrap;gap:8px 24px}
.si{color:rgba(255,255,255,.7);font-size:13.5px}
.si strong{color:#fff;font-weight:700}
.rs{margin-bottom:20px;background:var(--card);border-radius:var(--r);padding:0;box-shadow:var(--sh);overflow:hidden}
.rs-head{display:flex;align-items:center;gap:12px;padding:18px 28px 14px;border-bottom:1px solid var(--border)}
.rs-num{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:8px;background:var(--ruby);color:#fff;font-size:13px;font-weight:700;flex-shrink:0}
.rs h2{font-size:19px;color:var(--ruby-d);font-weight:700;margin:0}
.rs h3{font-size:15px;color:var(--text);margin:18px 28px 8px;font-weight:600}
.rs h4{font-size:13.5px;color:var(--ruby);margin:10px 0 4px;font-weight:600}
.expl{background:var(--ruby-pale);border-left:3px solid var(--ruby-l);padding:10px 14px;border-radius:0 8px 8px 0;font-size:13.5px;color:var(--text-l);line-height:1.5;margin:14px 28px}
.expl strong{color:var(--ruby-d)}
.cg{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px;margin:14px 28px}
.cd{background:var(--ruby-pale);border-radius:8px;padding:14px 16px;text-align:center;border:1px solid transparent;transition:border-color .15s}
.cd:hover{border-color:var(--ruby-l)}
.cd.hl{background:var(--ruby);border-color:var(--ruby)}
.cd.hl .cd-l{color:rgba(255,255,255,.7)}
.cd.hl .cd-v{color:#fff}
.cd-l{font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:var(--text-l);margin-bottom:3px}
.cd-v{font-size:18px;font-weight:700;color:var(--ruby-d);overflow-wrap:break-word;word-break:break-word;line-height:1.3}
.cd-v.sm{font-size:14px}
.cd-v.pills{font-size:0;display:flex;flex-wrap:wrap;justify-content:center;gap:4px}
.cd-pill{display:inline-block;font-size:11.5px;font-weight:600;color:var(--ruby-d);background:var(--card);border:1px solid var(--border);padding:2px 8px;border-radius:6px;white-space:nowrap}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin:8px 28px 14px}
.tag{background:var(--ruby);color:#fff;padding:4px 14px;border-radius:14px;font-size:12.5px;font-weight:500}
.tag.champ{background:var(--gold)}
.tag.step{background:var(--ruby-pale);color:var(--ruby-d);border:1px solid var(--border)}
.dt{width:calc(100% - 56px);border-collapse:collapse;font-size:13.5px;margin:8px 28px 14px;border-radius:8px;overflow:hidden}
.dt thead th{background:var(--ruby-d);color:#fff;padding:10px 14px;text-align:left;font-size:12px;font-weight:600;letter-spacing:.3px}
.dt thead th:first-child{border-radius:8px 0 0 0}.dt thead th:last-child{border-radius:0 8px 0 0}
.dt td{padding:8px 14px;border-bottom:1px solid var(--border)}
.dt tbody tr:last-child td{border-bottom:none}
.dt tbody tr:nth-child(even) td{background:var(--ruby-pale)}
.dt code{font-size:12px;font-family:var(--mono);background:var(--code-bg);padding:1px 6px;border-radius:4px;word-break:break-all}
.dt .na{color:var(--text-f)}
.champ-row td{background:var(--ruby-pale) !important;font-weight:600}
.badge-c{background:var(--ruby);color:#fff;font-size:10px;padding:2px 8px;border-radius:8px;margin-left:6px;font-weight:700}
.champ-callout{background:var(--card);border-radius:var(--r);padding:0;margin-bottom:20px;box-shadow:var(--sh);overflow:hidden;border-left:4px solid var(--gold)}
.champ-callout-inner{padding:18px 24px;display:flex;align-items:center;gap:20px;flex-wrap:wrap}
.champ-callout-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--gold);background:rgba(212,168,83,0.12);padding:3px 10px;border-radius:6px}
.champ-callout-name{font-size:22px;font-weight:700;color:var(--ruby-d)}
.champ-callout-score{font-size:15px;font-weight:600;color:var(--text);font-family:var(--mono)}
.champ-diff{font-size:12.5px;color:var(--green);font-weight:600;margin-left:auto}
.best-val{background:rgba(26,127,55,0.08)!important;font-weight:700;color:var(--green)!important}
.best-marker-legend{display:inline-block;background:rgba(26,127,55,0.08);color:var(--green);padding:1px 6px;border-radius:4px;font-size:12px;font-weight:600}
.sel-met{background:var(--ruby)!important;position:relative}
.rank-bars{margin:10px 28px 14px;display:flex;flex-direction:column;gap:6px}
.rank-bar{display:flex;align-items:center;gap:10px;font-size:13px}
.rank-bar.champ{font-weight:700}
.rank-pos{width:28px;text-align:center;font-size:15px;flex-shrink:0}
.rank-name{width:140px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rank-track{flex:1;height:10px;background:var(--ruby-pale);border-radius:5px;overflow:hidden}
.rank-fill{height:100%;background:linear-gradient(90deg,var(--ruby),var(--ruby-l));border-radius:5px}
.rank-bar.champ .rank-fill{background:linear-gradient(90deg,var(--gold),#e8c36a)}
.rank-val{font-family:var(--mono);font-size:12.5px;color:var(--text-l);width:65px;text-align:right;flex-shrink:0}
.tbl-scroll{overflow-x:auto;margin:8px 28px 14px;-webkit-overflow-scrolling:touch}
.tbl-scroll .dt{width:100%;margin:0;overflow:visible}
.bar-bg{display:inline-block;width:120px;height:8px;background:var(--ruby-pale);border-radius:4px;vertical-align:middle;margin-right:8px}
.bar-fill{height:100%;background:linear-gradient(90deg,var(--ruby),var(--ruby-l));border-radius:4px}
.bar-pct{font-size:12px;color:var(--text-l);font-family:var(--mono)}
.plot-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:14px;margin:14px 28px}
.plot-card{background:var(--ruby-bg);border:1px solid var(--border);border-radius:var(--r);padding:16px;transition:box-shadow .2s}
.plot-card:hover{box-shadow:var(--sh-lg)}
.plot-card img{max-width:100%;height:auto;border-radius:6px;margin-top:6px;display:block;cursor:pointer;transition:opacity .15s}
.plot-card img:hover{opacity:.85}
.lightbox{display:none;position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.82);backdrop-filter:blur(6px);align-items:center;justify-content:center;cursor:zoom-out;padding:24px}
.lightbox.show{display:flex}
.lightbox img{max-width:92vw;max-height:88vh;border-radius:12px;box-shadow:0 8px 48px rgba(0,0,0,.5);object-fit:contain}
.lightbox .lb-title{position:fixed;top:16px;left:50%;transform:translateX(-50%);color:rgba(255,255,255,.85);font-size:14px;font-weight:600;background:rgba(0,0,0,.5);padding:6px 18px;border-radius:20px;pointer-events:none}
.lightbox .lb-close{position:fixed;top:16px;right:20px;color:#fff;font-size:28px;cursor:pointer;background:rgba(255,255,255,.12);width:40px;height:40px;border-radius:20px;display:flex;align-items:center;justify-content:center;border:none;font-family:inherit;transition:background .15s}
.lightbox .lb-close:hover{background:rgba(255,255,255,.25)}
.plot-expl{font-size:12.5px;color:var(--text-l);font-style:italic;line-height:1.4}
.detail-box{margin:12px 28px;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.detail-box summary{padding:10px 16px;cursor:pointer;font-size:13.5px;font-weight:600;color:var(--ruby-d);background:var(--ruby-pale);user-select:none}
.detail-box summary:hover{color:var(--ruby)}
.detail-content{padding:14px 16px;font-size:13.5px;line-height:1.6}
.detail-content dl{display:grid;grid-template-columns:auto 1fr;gap:4px 12px}
.detail-content dt{font-weight:600}.detail-content dd{color:var(--text-l);margin:0}
.footer{text-align:center;color:var(--text-f);font-size:12px;margin-top:40px;padding:20px}
.back-top{position:fixed;bottom:24px;right:24px;width:40px;height:40px;border-radius:20px;background:var(--ruby);color:#fff;border:none;cursor:pointer;font-size:18px;box-shadow:0 2px 12px rgba(107,13,21,.3);opacity:0;transition:opacity .2s;z-index:50}
.back-top.vis{opacity:1}
@media(max-width:900px){.sidebar{display:none}.main{padding:20px 16px}.topbar{padding:0 16px}.topbar .sub{display:none}.plot-grid{grid-template-columns:1fr}.cg{margin:14px 16px}.tags{margin:8px 16px}.expl{margin:14px 16px}.dt{width:calc(100% - 32px);margin:8px 16px}.rs h3{margin:18px 16px 8px}.detail-box{margin:12px 16px}.plot-grid{margin:14px 16px}.tbl-scroll{margin:8px 16px}}
@media print{.topbar,.sidebar,.back-top{display:none!important}.layout{display:block;margin-top:0}.main{max-width:100%;padding:0}.rs{break-inside:avoid;box-shadow:none;border:1px solid #ddd;margin-bottom:12px}.summary-bar{background:var(--ruby-pale)!important;color:var(--text)!important;-webkit-print-color-adjust:exact;print-color-adjust:exact}.si{color:var(--text)!important}.si strong{color:var(--ruby-d)!important}.plot-card{break-inside:avoid}.plot-card img{max-height:300px;object-fit:contain}body{background:#fff}@page{margin:1.5cm}}
</style>
</head>
<body>
<div class="topbar"><svg class="topbar-logo" xmlns="http://www.w3.org/2000/svg" viewBox="50 -5 120 125" width="34" height="34"><polygon points="110,0 148,16 162,56 148,96 110,112 72,96 58,56 72,16" fill="rgba(255,255,255,0.08)" stroke="rgba(255,255,255,0.7)" stroke-width="2"/><circle cx="88" cy="56" r="5" fill="rgba(255,255,255,0.65)"/><line x1="93" y1="51" x2="136" y2="26" stroke="rgba(255,255,255,0.85)" stroke-width="2.2"/><polygon points="139,22 134,29 144,29" fill="rgba(255,255,255,0.85)"/><line x1="93" y1="61" x2="136" y2="86" stroke="rgba(255,255,255,0.4)" stroke-width="1.6" stroke-dasharray="5 3"/><polygon points="139,90 134,83 144,83" fill="rgba(255,255,255,0.4)"/></svg><h1>rubin</h1><span class="sub"><strong style="font-size:15px;letter-spacing:.3px">{{EXPERIMENT}}</strong> &bull; Champion: <strong>{{CHAMPION}}</strong> &bull; {{TIMESTAMP}}</span></div>
<div class="layout">
<nav class="sidebar"><div class="nav-lbl">Report</div>
{{NAV}}
</nav>
<main class="main">
{{BODY}}
<div class="footer"><svg xmlns="http://www.w3.org/2000/svg" viewBox="50 -5 120 125" width="20" height="20" style="vertical-align:middle;margin-right:6px;opacity:0.5"><polygon points="110,0 148,16 162,56 148,96 110,112 72,96 58,56 72,16" fill="rgba(155,17,30,0.08)" stroke="#9B111E" stroke-width="2"/><circle cx="88" cy="56" r="5" fill="#9B111E"/><line x1="93" y1="51" x2="136" y2="26" stroke="#9B111E" stroke-width="2.2"/><polygon points="139,22 134,29 144,29" fill="#9B111E"/><line x1="93" y1="61" x2="136" y2="86" stroke="#c4343f" stroke-width="1.6" stroke-dasharray="5 3"/><polygon points="139,90 134,83 144,83" fill="#c4343f"/></svg>rubin – Causal ML Framework &bull; Report generiert am {{TIMESTAMP}}</div>
</main>
</div>
<button class="back-top" id="bt" onclick="window.scrollTo({top:0,behavior:'smooth'})">↑</button>
<div class="lightbox" id="lb" onclick="this.classList.remove('show')"><button class="lb-close" onclick="document.getElementById('lb').classList.remove('show')">&times;</button><div class="lb-title" id="lb-title"></div><img id="lb-img" src="" alt=""></div>
<script>
const lnk=document.querySelectorAll('.sidebar a'),secs=[];
lnk.forEach(a=>{const el=document.getElementById(a.dataset.section);if(el)secs.push({el,a})});
function upd(){let c=secs[0];for(const s of secs){if(s.el.getBoundingClientRect().top<=120)c=s}lnk.forEach(a=>a.classList.remove('active'));if(c)c.a.classList.add('active');document.getElementById('bt').classList.toggle('vis',window.scrollY>400)}
window.addEventListener('scroll',upd,{passive:true});upd();
document.querySelectorAll('.plot-card img').forEach(img=>{img.addEventListener('click',e=>{e.stopPropagation();const lb=document.getElementById('lb');document.getElementById('lb-img').src=img.src;const t=img.closest('.plot-card')?.querySelector('h4');document.getElementById('lb-title').textContent=t?t.textContent:'';lb.classList.add('show')})});
document.addEventListener('keydown',e=>{if(e.key==='Escape')document.getElementById('lb').classList.remove('show')});
</script>
</body>
</html>"""
