from __future__ import annotations

"""Evaluation mit EconML-DRTester und scikit-uplift Plots.

Hintergrund
-----------
EconML bringt mit dem `DRTester` eine Validierungsroutine mit, die u. a.
- BLP-Tests,
- Calibration,
- Qini-/TOC-basierte Uplift-Tests
bereitstellt.

rubin erzeugt CATEs als Cross-Predictions (Out-of-Fold), damit
Evaluationskennzahlen nicht optimistisch verzerrt sind. Der Standard-`DRTester`
erwartet jedoch ein CATE-Modell, aus dem er Vorhersagen selbst erzeugt.
Da die CATE-Vorhersagen in rubin bereits vorliegen, kapselt dieses Modul
eine angepasste DRTester-Variante (`CustomDRTester`), die vorberechnete
CATE-Werte direkt akzeptiert.

Zusätzlich werden Plots aus scikit-uplift (sklift) erzeugt, die für die
visuelle Beurteilung der Sortierung genutzt werden."""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")  # Headless-Backend für Batch-/Server-Umgebungen
import matplotlib.pyplot as plt
from pandas.plotting import table

from rubin.utils.plot_theme import (
    apply_rubin_theme, RUBIN_COLORS, RUBIN_PALETTE,
    COLOR_MODEL, COLOR_MODEL_FILL, COLOR_REFERENCE, COLOR_REFERENCE_FILL,
    COLOR_BASELINE, COLOR_DIFFERENCE, COLOR_HIGHLIGHT_BOX,
    recolor_figure,
)
apply_rubin_theme()

from econml.validate import EvaluationResults
from econml.validate.drtester import DRTester
from econml.validate.utils import calculate_dr_outcomes

from scipy.interpolate import interp1d

# ── sklearn/scikit-uplift Kompatibilitäts-Shim ──
# scikit-uplift <0.6 importiert `check_matplotlib_support` aus `sklearn.utils`,
# das in sklearn 1.6+ entfernt wurde. Wir registrieren einen minimalen Ersatz,
# damit der Import nicht fehlschlägt. Die Funktion prüft nur, ob matplotlib
# installiert ist — das ist in unserer Umgebung immer der Fall.
try:
    from sklearn.utils import check_matplotlib_support  # noqa: F401
except ImportError:
    import sklearn.utils as _sklearn_utils

    def _check_matplotlib_support(caller_name: str) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise ImportError(
                f"{caller_name} requires matplotlib. "
                "Install it with: pip install matplotlib"
            )

    _sklearn_utils.check_matplotlib_support = _check_matplotlib_support

try:
    # scikit-uplift heißt im Import-Pfad sklift
    from sklift.viz import plot_qini_curve, plot_uplift_by_percentile, plot_treatment_balance_curve

    _SKLIFT_AVAILABLE = True
except Exception as _sklift_err:
    _SKLIFT_AVAILABLE = False
    _logger.warning(
        "scikit-uplift Import fehlgeschlagen: %s – "
        "Qini-Curve, Uplift-by-Percentile und Treatment-Balance-Plots "
        "werden übersprungen. "
        "Installation: pip install 'scikit-uplift>=0.5' "
        "(alle Dependencies sind bereits über conda installiert).",
        _sklift_err,
    )


def save_dataframe_as_png(df: pd.DataFrame, filename: str) -> str:
    """Speichert ein DataFrame als PNG-Tabelle.
In MLflow sind Tabellen als Bild oft schneller zu sichten als als CSV.
Die CSV wird in rubin an anderer Stelle ohnehin zusätzlich geloggt."""

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    tbl = table(ax, df, loc="center", cellLoc="center", colWidths=[0.1] * len(df.columns))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.2)
    # Rubin-Theme: Header-Zeile einfärben
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(RUBIN_COLORS["ruby"])
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor(RUBIN_COLORS["ruby_dark"])
        else:
            cell.set_facecolor(RUBIN_COLORS["ruby_pale"] if row % 2 == 0 else RUBIN_COLORS["white"])
            cell.set_edgecolor(RUBIN_COLORS["grid"])
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return filename


class CustomEvaluationResults(EvaluationResults):
    """Erweiterung der EconML-Ergebnisse um Policy-Value-Extraktion.
Die EconML-Qini-Auswertung enthält Kurvenpunkte und Unsicherheiten.
Für den operativen Vergleich ist es praktisch, Policy Values für feste
Treat-Anteile als Tabelle auszugeben."""

    def get_policy_values(
        self,
        tmt: int,
        treated_percentages: List[float] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        conf_interval_type: str = "normal",  # "normal" | "ucb2" | "ucb1"
    ) -> pd.DataFrame:
        # Kurvendaten für das gewählte Treatment
        qini_curve = self.qini.curves[self.qini.treatments[tmt]]

        out: list[dict[str, float]] = []
        for p in treated_percentages:
            # Die Qini-Kurve arbeitet auf einer "Percentage treated" Skala.
            closest_idx = (qini_curve["Percentage treated"] - p).abs().idxmin()
            policy_value = float(qini_curve.loc[closest_idx, "value"])
            err = float(qini_curve.loc[closest_idx, "err"])
            ucb2 = float(qini_curve.loc[closest_idx, "uniform_critical_value"])
            ucb1 = float(qini_curve.loc[closest_idx, "uniform_one_side_critical_value"])

            if conf_interval_type == "normal":
                lo, hi = policy_value - 1.96 * err, policy_value + 1.96 * err
            elif conf_interval_type == "ucb2":
                lo, hi = policy_value - ucb2 * err, policy_value + ucb2 * err
            elif conf_interval_type == "ucb1":
                lo, hi = policy_value - ucb1 * err, policy_value
            else:
                raise ValueError(
                    f"Ungültiger conf_interval_type: {conf_interval_type}. Erlaubt: 'normal', 'ucb2', 'ucb1'."
                )

            out.append(
                {
                    "treated_percentage": float(p),
                    "policy_value": round(policy_value, 6),
                    "lower_bound": round(float(lo), 6),
                    "upper_bound": round(float(hi), 6),
                }
            )

        return pd.DataFrame(out)


class CustomDRTester(DRTester):
    """DRTester, der vorberechnete CATE-Vorhersagen akzeptiert.
Motivation
----------
In rubin werden CATEs in der Analyse typischerweise als Cross-Predictions
erzeugt. Damit DRTester die gleichen Werte nutzen kann, werden die
Vorhersagen an den Tester übergeben.
Hinweis
-------
`cate_preds_val` und `cate_preds_train` müssen zur jeweiligen X/T/Y-Menge
passen (gleiche Reihenfolge/Länge)."""

    def __init__(
        self,
        *,
        model_regression: Any,
        model_propensity: Any,
        cate: Any = None,
        cv: Union[int, List] = 5,
        cate_preds_val: Optional[np.ndarray] = None,
        cate_preds_train: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            model_regression=model_regression,
            model_propensity=model_propensity,
            cate=cate,
            cv=cv,
        )

        # EconML erwartet intern shape (n, 1)
        self.cate_preds_val_ = None
        self.cate_preds_train_ = None

        if cate_preds_val is not None:
            self.cate_preds_val_ = np.asarray(cate_preds_val).reshape(-1, 1)
        if cate_preds_train is not None:
            self.cate_preds_train_ = np.asarray(cate_preds_train).reshape(-1, 1)

        if self.cate_preds_val_ is None and cate is None:
            raise ValueError("Entweder cate oder cate_preds_val muss angegeben werden.")

    def fit_nuisance(
        self,
        Xval: np.ndarray,
        Dval: np.ndarray,
        yval: np.ndarray,
        Xtrain: Optional[np.ndarray] = None,
        Dtrain: Optional[np.ndarray] = None,
        ytrain: Optional[np.ndarray] = None,
    ):
        """Erzeugt Nuisance-Preds und DR-Outcomes über Cross-Fitting.

        Überschreibt die EconML-Methode, um sicherzustellen, dass Train-/Val-Daten
        konsistent verarbeitet werden, wenn beide Mengen übergeben werden."""

        self.Dval = Dval
        self.treatments = np.sort(np.unique(Dval))
        self.n_treat = len(self.treatments) - 1

        self.fit_on_train = (Xtrain is not None) and (Dtrain is not None) and (ytrain is not None)

        if self.fit_on_train:
            reg_preds_train, prop_preds_train = self.fit_nuisance_cv(Xtrain, Dtrain, ytrain)
            self.dr_train_ = calculate_dr_outcomes(Dtrain, ytrain, reg_preds_train, prop_preds_train)

            reg_preds_val, prop_preds_val = self.fit_nuisance_cv(Xval, Dval, yval)
            self.dr_val_ = calculate_dr_outcomes(Dval, yval, reg_preds_val, prop_preds_val)
        else:
            reg_preds_val, prop_preds_val = self.fit_nuisance_cv(Xval, Dval, yval)
            self.dr_val_ = calculate_dr_outcomes(Dval, yval, reg_preds_val, prop_preds_val)

        self.ate_val = self.dr_val_.mean(axis=0)
        return self

    def get_cate_preds(self, Xval: np.ndarray, Xtrain: Optional[np.ndarray] = None) -> None:
        # Falls Val-Preds übergeben wurden, nichts weiter tun.
        if self.cate_preds_val_ is None:
            if self.cate is None:
                raise ValueError("CATE-Modell ist nicht gesetzt, und keine cate_preds_val angegeben.")
            base = self.treatments[0]
            vals = [self.cate.effect(X=Xval, T0=base, T1=t) for t in self.treatments[1:]]
            self.cate_preds_val_ = np.stack(vals).T

        if Xtrain is not None:
            if self.cate_preds_train_ is None:
                if self.cate is None:
                    raise ValueError("CATE-Modell ist nicht gesetzt, und keine cate_preds_train angegeben.")
                base = self.treatments[0]
                trains = [self.cate.effect(X=Xtrain, T0=base, T1=t) for t in self.treatments[1:]]
                self.cate_preds_train_ = np.stack(trains).T

    def evaluate_all(
        self,
        Xval: Optional[np.ndarray] = None,
        Xtrain: Optional[np.ndarray] = None,
        n_groups: int = 10,
        n_bootstrap: int = 1000,
    ) -> CustomEvaluationResults:
        """Führt alle DRTester-Tests aus und liefert CustomEvaluationResults.

        n_bootstrap steuert die Anzahl der Bootstrap-Iterationen für Qini/TOC
        Konfidenzintervalle. Weniger Iterationen → schneller, breitere CIs."""

        if (not hasattr(self, "cate_preds_val_")) or (self.cate_preds_val_ is None):
            if Xval is None:
                raise ValueError("CATE-Preds sind nicht gesetzt. Xval muss angegeben werden.")
            self.get_cate_preds(Xval, Xtrain)

        blp_res = self.evaluate_blp()
        cal_res = self.evaluate_cal(n_groups=n_groups)
        qini_res = self.evaluate_uplift(metric="qini", n_bootstrap=n_bootstrap)
        toc_res = self.evaluate_uplift(metric="toc", n_bootstrap=n_bootstrap)

        self.res = CustomEvaluationResults(blp_res=blp_res, cal_res=cal_res, qini_res=qini_res, toc_res=toc_res)
        return self.res


@dataclass
class DrTesterPlotBundle:
    summary: pd.DataFrame
    cal_plot: plt.Figure
    qini_plot: plt.Figure
    toc_plot: plt.Figure
    policy_values: pd.DataFrame
    sklift_qini: Optional[plt.Figure]
    sklift_percentile: Optional[plt.Figure]
    treatment_balance: Optional[plt.Figure]


def fit_drtester_nuisance(
    *,
    model_regression: Any,
    model_propensity: Any,
    X_val: pd.DataFrame,
    T_val: np.ndarray,
    Y_val: np.ndarray,
    X_train: Optional[pd.DataFrame] = None,
    T_train: Optional[np.ndarray] = None,
    Y_train: Optional[np.ndarray] = None,
    cv: int = 5,
) -> CustomDRTester:
    """Erstellt und fittet einen DRTester NUR für die Nuisance-Modelle (Outcome + Propensity).

    Die Nuisance-Ergebnisse (DR-Outcomes, ATE) sind für alle kausalen Modelle gleich
    und müssen nur einmal berechnet werden. Der zurückgegebene Tester kann dann
    mit evaluate_cate_with_plots(fitted_tester=...) wiederverwendet werden.

    cv: Anzahl Cross-Fitting-Folds für Nuisance-Predictions (Default: 5).
        Weniger Folds = schneller, leicht ungenauere DR-Outcomes."""

    # Dummy-CATE-Preds (werden bei der Nuisance-Fitphase nicht benötigt,
    # aber der Konstruktor verlangt entweder cate oder cate_preds_val)
    dummy_cate = np.zeros(len(X_val))

    tester = CustomDRTester(
        model_regression=model_regression,
        model_propensity=model_propensity,
        cate=None,
        cate_preds_val=dummy_cate,
        cv=cv,
    )
    # Referenzen speichern für spätere Nuisance-Wiederverwendung
    tester._model_regression_ref = model_regression
    tester._model_propensity_ref = model_propensity

    if X_train is not None and T_train is not None and Y_train is not None:
        tester.fit_nuisance(
            Xval=X_val.values,
            Dval=np.asarray(T_val).ravel(),
            yval=np.asarray(Y_val).ravel(),
            Xtrain=X_train.values,
            Dtrain=np.asarray(T_train).ravel(),
            ytrain=np.asarray(Y_train).ravel(),
        )
    else:
        tester.fit_nuisance(
            Xval=X_val.values,
            Dval=np.asarray(T_val).ravel(),
            yval=np.asarray(Y_val).ravel(),
        )

    return tester


def generate_cate_distribution_plot(
    cate_preds_val: np.ndarray,
    cate_preds_train: Optional[np.ndarray] = None,
    model_name: str = "",
    arm_label: str = "",
) -> Optional[Any]:
    """Erzeugt ein Histogramm der CATE-Predictions (Training + Cross-Validated).

    Zeigt die Verteilung der vorhergesagten Treatment-Effekte. Dient der
    visuellen Plausibilitätsprüfung: stark konzentrierte Verteilungen nahe
    Null deuten auf wenig Heterogenität hin, breite Verteilungen auf
    differenzierte Effektvorhersagen.

    Returns
    -------
    matplotlib.figure.Figure oder None bei Fehler."""
    from rubin.utils.plot_theme import apply_rubin_theme, RUBIN_COLORS
    apply_rubin_theme()

    val = np.asarray(cate_preds_val, dtype=float).ravel()
    val = val[~np.isnan(val)]
    if len(val) == 0:
        return None

    has_train = cate_preds_train is not None
    if has_train:
        train = np.asarray(cate_preds_train, dtype=float).ravel()
        train = train[~np.isnan(train)]
        has_train = len(train) > 0

    suffix = f" {arm_label}" if arm_label else ""
    n_cols = 2 if has_train else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]

    # Gemeinsamer Bin-Bereich für Vergleichbarkeit
    all_vals = np.concatenate([val, train]) if has_train else val
    q_lo, q_hi = np.percentile(all_vals, [0.5, 99.5])
    bins = np.linspace(q_lo, q_hi, 80)

    color = RUBIN_COLORS["ruby"]
    edge_color = RUBIN_COLORS["ruby_dark"]

    if has_train:
        ax_train = axes[0]
        ax_train.hist(train, bins=bins, color=color, edgecolor=edge_color,
                      linewidth=0.3, alpha=0.85)
        ax_train.set_title(f"Training Predictions{suffix}", fontsize=12, fontweight="bold")
        ax_train.set_xlabel(f"Train_{model_name}{suffix}", fontsize=10)
        ax_train.set_ylabel("Count", fontsize=10)
        ax_train.axvline(0, color=RUBIN_COLORS["slate"], linewidth=1, linestyle="--", alpha=0.6)

    ax_val = axes[-1]
    ax_val.hist(val, bins=bins, color=color, edgecolor=edge_color,
                linewidth=0.3, alpha=0.85)
    ax_val.set_title(f"Cross-Validated Predictions{suffix}", fontsize=12, fontweight="bold")
    ax_val.set_xlabel(f"Predictions_{model_name}{suffix}", fontsize=10)
    ax_val.set_ylabel("Count", fontsize=10)
    ax_val.axvline(0, color=RUBIN_COLORS["slate"], linewidth=1, linestyle="--", alpha=0.6)

    fig.suptitle(f"CATE Distribution — {model_name}{suffix}", fontsize=14,
                 fontweight="bold", color=RUBIN_COLORS["ruby_dark"], y=1.02)
    fig.tight_layout()
    return fig


def _native_uplift_by_percentile(
    y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray,
    n_bins: int = 10,
) -> Optional[plt.Figure]:
    """Uplift-by-Percentile Barplot — numpy-safe Ersatz für sklift.

    Repliziert ``sklift.viz.plot_uplift_by_percentile(kind='bar')`` exakt:
    sortiert nach pred. Uplift absteigend, teilt in n_bins, berechnet
    pro Bin den beobachteten Uplift E[Y|T=1] − E[Y|T=0].

    Erzeugt den Plot mit Default-Styling. ``recolor_figure()`` wird
    vom Aufrufer angewandt (wie bei DRTester-Plots)."""
    try:
        order = np.argsort(uplift)[::-1]
        y_s = y_true[order]
        t_s = treatment[order]
        n = len(y_s)
        bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)

        bin_uplifts = []
        bin_labels = []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            y_bin, t_bin = y_s[lo:hi], t_s[lo:hi]
            treat_mask = t_bin == 1
            ctrl_mask = t_bin == 0
            if treat_mask.sum() > 0 and ctrl_mask.sum() > 0:
                u = float(y_bin[treat_mask].mean() - y_bin[ctrl_mask].mean())
            else:
                u = 0.0
            bin_uplifts.append(u)
            bin_labels.append(f"{i + 1}")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2ca02c" if u >= 0 else "#d62728" for u in bin_uplifts]
        ax.bar(range(n_bins), bin_uplifts, color=colors, edgecolor="black",
               linewidth=0.5, alpha=0.8)
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels(bin_labels)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Percentile Bucket (1 = höchster pred. Uplift)")
        ax.set_ylabel("Observed Uplift")
        ax.set_title("Uplift by Percentile")
        fig.tight_layout()
        return fig
    except Exception as e:
        _logger.warning("Native Uplift-by-Percentile fehlgeschlagen: %s", e)
        return None


def _native_treatment_balance(
    uplift: np.ndarray, treatment: np.ndarray,
    winsize: float = 0.1,
) -> Optional[plt.Figure]:
    """Treatment-Balance-Kurve — numpy-safe Ersatz für sklift.

    Repliziert ``sklift.viz.plot_treatment_balance_curve(winsize=0.1)`` exakt:
    sortiert nach pred. Uplift absteigend, berechnet Treatment-Rate in einem
    Sliding Window der Breite ``winsize * n``.

    Erzeugt den Plot mit Default-Styling. ``recolor_figure()`` wird
    vom Aufrufer angewandt (wie bei DRTester-Plots)."""
    try:
        order = np.argsort(uplift)[::-1]
        t_s = treatment[order].astype(float)
        n = len(t_s)
        window = max(1, int(winsize * n))

        # Sliding-Window Treatment-Rate (wie sklift)
        cumsum = np.cumsum(np.insert(t_s, 0, 0))
        window_sums = cumsum[window:] - cumsum[:-window]
        treatment_rates = window_sums / window

        x_axis = np.arange(len(treatment_rates)) / n
        overall_rate = float(treatment.mean())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_axis, treatment_rates, linewidth=1.5, label="Treatment Rate (sliding window)")
        ax.axhline(overall_rate, color="grey", linewidth=1.0,
                   linestyle="--", label=f"Overall Rate ({overall_rate:.3f})")
        ax.set_xlabel("Fraction (sorted by pred. uplift, descending)")
        ax.set_ylabel("Treatment Rate")
        ax.set_title("Treatment Balance Curve")
        ax.legend()
        fig.tight_layout()
        return fig
    except Exception as e:
        _logger.warning("Native Treatment-Balance fehlgeschlagen: %s", e)
        return None


def generate_sklift_plots(
    cate_preds_val: np.ndarray,
    T_val: np.ndarray,
    Y_val: np.ndarray,
) -> tuple:
    """Erzeugt Uplift-Plots: Qini-Kurve, Uplift-by-Percentile, Treatment-Balance.

    Versucht zuerst sklift. Bei numpy-Inkompatibilität (>=1.24) wird automatisch
    auf eigene Implementierungen zurückgefallen. Alle Plots werden mit
    ``recolor_figure()`` ins rubin-Farbschema überführt.

    Gibt (sklift_qini, sklift_percentile, treatment_balance) als Matplotlib-Figures zurück.
    Fehlende Plots sind None."""
    sk_qini, sk_pct, sk_tb = None, None, None

    uplift = np.asarray(cate_preds_val).ravel()
    y_val_arr = np.asarray(Y_val).ravel()
    t_val_arr = np.asarray(T_val).ravel()

    # ── Qini-Kurve ──
    if _SKLIFT_AVAILABLE:
        try:
            result = plot_qini_curve(
                y_true=y_val_arr, uplift=uplift, treatment=t_val_arr, perfect=False,
            )
            sk_qini = result.figure_
        except Exception as e:
            _logger.warning("sklift Qini-Curve fehlgeschlagen: %s", e)

    # ── Uplift-by-Percentile: sklift → Fallback native ──
    if _SKLIFT_AVAILABLE:
        try:
            result = plot_uplift_by_percentile(
                y_true=y_val_arr, uplift=uplift, treatment=t_val_arr, kind="bar",
            )
            sk_pct = result[0].get_figure()
        except Exception:
            pass  # Fallback unten
    if sk_pct is None:
        sk_pct = _native_uplift_by_percentile(y_val_arr, uplift, t_val_arr)

    # ── Treatment-Balance: sklift → Fallback native ──
    if _SKLIFT_AVAILABLE:
        try:
            result = plot_treatment_balance_curve(
                uplift=uplift, treatment=t_val_arr, winsize=0.1,
            )
            sk_tb = result.get_figure()
        except Exception:
            pass  # Fallback unten
    if sk_tb is None:
        sk_tb = _native_treatment_balance(uplift, t_val_arr)

    # ── rubin-Farbschema auf alle Plots anwenden ──
    for fig in [sk_qini, sk_pct, sk_tb]:
        if fig is not None:
            recolor_figure(fig)

    return sk_qini, sk_pct, sk_tb


def evaluate_cate_with_plots(
    *,
    model_regression: Any = None,
    model_propensity: Any = None,
    X_val: pd.DataFrame,
    T_val: np.ndarray,
    Y_val: np.ndarray,
    cate_preds_val: np.ndarray,
    X_train: Optional[pd.DataFrame] = None,
    T_train: Optional[np.ndarray] = None,
    Y_train: Optional[np.ndarray] = None,
    cate_preds_train: Optional[np.ndarray] = None,
    n_groups: int = 10,
    fitted_tester: Optional[CustomDRTester] = None,
    n_bootstrap: int = 1000,
) -> DrTesterPlotBundle:
    """Hauptfunktion: DRTester-Auswertung + sklift-Plots.

    Wenn fitted_tester übergeben wird, werden die vorberechneten Nuisance-Ergebnisse
    (DR-Outcomes) wiederverwendet — das spart das teure Nuisance-CV-Fitting.
    Nur die CATE-Predictions werden ausgetauscht."""

    if fitted_tester is not None:
        # Nuisance bereits berechnet → frischen Tester mit gleichen DR-Outcomes erstellen.
        # KEIN copy.copy (EconML-interner State ist nicht copy-sicher).
        # Stattdessen: neuer Tester, Nuisance-State manuell injizieren.
        tester = CustomDRTester(
            model_regression=getattr(fitted_tester, '_model_regression_ref', None),
            model_propensity=getattr(fitted_tester, '_model_propensity_ref', None),
            cate=None,
            cate_preds_val=cate_preds_val,
            cate_preds_train=cate_preds_train,
        )
        # DR-Outcomes + Nuisance-State aus dem Pre-Fit übernehmen (das teure Ergebnis)
        tester.dr_val_ = fitted_tester.dr_val_
        tester.ate_val = fitted_tester.ate_val
        tester.Dval = fitted_tester.Dval
        tester.treatments = fitted_tester.treatments
        tester.n_treat = fitted_tester.n_treat
        tester.fit_on_train = fitted_tester.fit_on_train
        if hasattr(fitted_tester, 'dr_train_'):
            tester.dr_train_ = fitted_tester.dr_train_
        if hasattr(fitted_tester, 'cate_preds_train_') and fitted_tester.cate_preds_train_ is not None and cate_preds_train is not None:
            tester.cate_preds_train_ = np.asarray(cate_preds_train).reshape(-1, 1)
        _logger.debug(
            "DRTester Nuisance wiederverwendet: dr_val=%s, cate_preds_val=%s (min=%.4g, max=%.4g)",
            tester.dr_val_.shape, tester.cate_preds_val_.shape,
            float(np.nanmin(cate_preds_val)), float(np.nanmax(cate_preds_val)),
        )
    else:
        # Fallback: Nuisance komplett neu fitten (wenn kein Pre-Fit vorhanden)
        if model_regression is None or model_propensity is None:
            raise ValueError("Entweder fitted_tester oder model_regression + model_propensity muss angegeben werden.")
        tester = CustomDRTester(
            model_regression=model_regression,
            model_propensity=model_propensity,
            cate=None,
            cate_preds_val=cate_preds_val,
            cate_preds_train=cate_preds_train,
        )

        if X_train is not None and T_train is not None and Y_train is not None:
            tester.fit_nuisance(
                Xval=X_val.values,
                Dval=np.asarray(T_val).ravel(),
                yval=np.asarray(Y_val).ravel(),
                Xtrain=X_train.values,
                Dtrain=np.asarray(T_train).ravel(),
                ytrain=np.asarray(Y_train).ravel(),
            )
        else:
            tester.fit_nuisance(
                Xval=X_val.values,
                Dval=np.asarray(T_val).ravel(),
                yval=np.asarray(Y_val).ravel(),
            )

    # ── DRTester-Plots (Calibration, Qini, TOC) ──
    summary, cal_plot, qini_plot, toc_plot, policy_values = None, None, None, None, None
    try:
        res = tester.evaluate_all(X_val.values, X_train.values if X_train is not None else None, n_groups=n_groups, n_bootstrap=n_bootstrap)
        summary = res.summary()
        cal_plot = res.plot_cal(1).get_figure()
        recolor_figure(cal_plot)
        qini_plot = res.plot_qini(1).get_figure()
        recolor_figure(qini_plot)
        toc_plot = res.plot_toc(1).get_figure()
        recolor_figure(toc_plot)
        policy_values = res.get_policy_values(1)
    except Exception as e:
        _logger.warning("DRTester evaluate_all fehlgeschlagen: %s", e, exc_info=True)
        if summary is None:
            summary = pd.DataFrame()
        if policy_values is None:
            policy_values = pd.DataFrame()

    # ── scikit-uplift Plots (Qini-Curve, Uplift by Percentile, Treatment Balance) ──
    sk_qini = None
    sk_pct = None
    sk_tb = None
    uplift = np.asarray(cate_preds_val).ravel()
    y_val_arr = np.asarray(Y_val).ravel()
    t_val_arr = np.asarray(T_val).ravel()

    # Qini-Kurve
    if _SKLIFT_AVAILABLE:
        try:
            result = plot_qini_curve(
                y_true=y_val_arr, uplift=uplift, treatment=t_val_arr, perfect=False,
            )
            sk_qini = result.figure_
        except Exception as e:
            _logger.warning("sklift Qini-Curve fehlgeschlagen: %s", e)

    # Uplift-by-Percentile: sklift → Fallback native
    if _SKLIFT_AVAILABLE:
        try:
            result = plot_uplift_by_percentile(
                y_true=y_val_arr, uplift=uplift, treatment=t_val_arr, kind="bar",
            )
            sk_pct = result[0].get_figure()
        except Exception:
            pass
    if sk_pct is None:
        sk_pct = _native_uplift_by_percentile(y_val_arr, uplift, t_val_arr)

    # Treatment-Balance: sklift → Fallback native
    if _SKLIFT_AVAILABLE:
        try:
            result = plot_treatment_balance_curve(
                uplift=uplift, treatment=t_val_arr, winsize=0.1,
            )
            sk_tb = result.get_figure()
        except Exception:
            pass
    if sk_tb is None:
        sk_tb = _native_treatment_balance(uplift, t_val_arr)

    # rubin-Farbschema auf alle Plots anwenden
    for fig in [sk_qini, sk_pct, sk_tb]:
        if fig is not None:
            recolor_figure(fig)

    return DrTesterPlotBundle(
        summary=summary,
        cal_plot=cal_plot,
        qini_plot=qini_plot,
        toc_plot=toc_plot,
        policy_values=policy_values,
        sklift_qini=sk_qini,
        sklift_percentile=sk_pct,
        treatment_balance=sk_tb,
    )


def _interpolate_curve(x_curve: np.ndarray, y_curve: np.ndarray, x_final: np.ndarray) -> np.ndarray:
    f = interp1d(x_curve, y_curve, kind="linear")
    return f(x_final)


def compute_qini_curve(outcomes: np.ndarray, score: np.ndarray, treatment: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrahiert Qini-Kurvenpunkte aus sklift-Plot.
sklift stellt die Kurve über eine Plot-Funktion bereit. Um eine konsistente
Weiterverarbeitung zu ermöglichen (Vergleich zweier Kurven), werden die
Linienpunkte aus dem Matplotlib-Axes gelesen und anschließend auf ein
gemeinsames x-Raster interpoliert."""

    if not _SKLIFT_AVAILABLE:
        raise RuntimeError("scikit-uplift ist nicht installiert.")

    fig, axis = plt.subplots()
    plot_qini_curve(outcomes, score, treatment, perfect=False, ax=axis)
    x_score, y_score = axis.lines[0].get_data()
    x_random, y_random = axis.lines[1].get_data()
    plt.close(fig)

    x = np.arange(len(outcomes))
    y_score_i = _interpolate_curve(np.asarray(x_score), np.asarray(y_score), x)
    y_rand_i = _interpolate_curve(np.asarray(x_random), np.asarray(y_random), x)
    return y_score_i, x, y_rand_i


def plot_custom_qini_curve(
    *,
    data: pd.DataFrame,
    causal_score_label: str,
    affinity_score_label: Optional[str] = None,
    ax: Optional[Any] = None,
    relative_axes: bool = True,
    x_vert: float = 0.5,
    print_uplift: bool = True,
):
    """Custom-Qini-Plot zur direkten Gegenüberstellung zweier Scores.

    Stellt den kausalen Score gegen einen optionalen Referenzscore (z. B.
    historischer Affinity-Score) auf derselben Qini-Kurve dar. Ermöglicht
    einen schnellen visuellen Vergleich der Sortierqualität."""

    y_causal, x, y_random = compute_qini_curve(data["Y"].to_numpy(), data[causal_score_label].to_numpy(), data["T"].to_numpy())
    y_affinity = None
    if affinity_score_label is not None:
        y_affinity, _, _ = compute_qini_curve(data["Y"].to_numpy(), data[affinity_score_label].to_numpy(), data["T"].to_numpy())

    if relative_axes:
        x = x / np.max(x)
        inc_total = np.max(y_random)
        y_causal = y_causal / inc_total
        y_random = y_random / inc_total
        if y_affinity is not None:
            y_affinity = y_affinity / inc_total

    if ax is None:
        fig, ax = plt.subplots()

    x_vertical_line = np.quantile(x, x_vert)
    y_vertical_line = y_random.max()
    ax.vlines(x_vertical_line, 0, y_vertical_line, color=RUBIN_COLORS["slate_light"], linestyle="--")
    ax.plot(x, y_random, color=COLOR_BASELINE, label="Random", linewidth=1.5, linestyle=":")

    ax.plot(x, y_causal, color=COLOR_MODEL, label=causal_score_label)
    ax.fill_between(x, y_causal, y_random, color=COLOR_MODEL, alpha=0.08)
    ax.hlines(np.quantile(y_causal, x_vert), 0, x_vertical_line, linestyle="--", color=COLOR_MODEL, alpha=0.7)

    if y_affinity is not None and affinity_score_label is not None:
        ax.plot(x, y_affinity, color=COLOR_REFERENCE, label=affinity_score_label)
        ax.fill_between(x, y_affinity, y_random, color=COLOR_REFERENCE, alpha=0.08)
        ax.hlines(np.quantile(y_affinity, x_vert), 0, x_vertical_line, linestyle="--", color=COLOR_REFERENCE, alpha=0.7)

    if relative_axes:
        ax.set(xticks=np.linspace(0, 1, 5), yticks=np.linspace(0, 1, 5))
        ax.set_xlabel("Anteil Experimentalmenge")
        ax.set_ylabel("Anteil inkrementelles Ergebnis")
    else:
        ax.set_xlabel("Experimentalmenge")
        ax.set_ylabel("Inkrementelles Ergebnis")

    ax.set_title(f"Qini-Vergleich: {causal_score_label}")
    ax.legend()

    if print_uplift and y_affinity is not None:
        uplift_abs = round((np.quantile(y_causal, x_vert) - np.quantile(y_affinity, x_vert)) / y_vertical_line * 100, 1)
        uplift_rel = round((np.quantile(y_causal, x_vert) - np.quantile(y_affinity, x_vert)) / np.quantile(y_affinity, x_vert) * 100, 1)
        textstr = f"abs. uplift = {uplift_abs}%\nrel. uplift = {uplift_rel}%"
        props = dict(boxstyle="round,pad=0.5", facecolor=COLOR_HIGHLIGHT_BOX, edgecolor=RUBIN_COLORS["ruby_light"], alpha=0.85)
        ax.text(0.95, 0.1, textstr, transform=ax.transAxes, fontsize=12, va="bottom", ha="right", bbox=props)

    return ax

def policy_value_comparison_plots(
    policy_values_dict: dict[str, pd.DataFrame],
    comparison_model_name: str,
) -> dict[str, plt.Figure]:
    """Erzeugt Vergleichsplots der Policy Values gegen ein Referenzmodell.

    Neben der Qini-Kurve ist der erwartete inkrementelle Nutzen (Policy Value)
    für feste Treat-Anteile (z. B. 10%, 20%, …) eine zentrale Entscheidungsgrundlage.
    Diese Funktion erstellt für jedes Modell (außer dem Referenzmodell) einen Plot
    mit drei Kurven:

    1) Policy Values des Modells inkl. Konfidenzintervall
    2) Policy Values des Referenzmodells inkl. Konfidenzintervall
    3) Differenzkurve (Modell − Referenz)

    Erwartetes DataFrame-Format je Modell:
    - treated_percentage
    - policy_value
    - lower_bound
    - upper_bound"""

    if comparison_model_name not in policy_values_dict:
        raise KeyError(
            f"Referenzmodell '{comparison_model_name}' nicht in policy_values_dict vorhanden."
        )

    plots: dict[str, plt.Figure] = {}
    ref = policy_values_dict[comparison_model_name]

    for model_name, df in policy_values_dict.items():
        if model_name == comparison_model_name:
            continue

        treated = df["treated_percentage"]
        model_values = df["policy_value"]
        model_lower = df["lower_bound"]
        model_upper = df["upper_bound"]

        ref_values = ref["policy_value"]
        ref_lower = ref["lower_bound"]
        ref_upper = ref["upper_bound"]

        difference = model_values.to_numpy() - ref_values.to_numpy()

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(treated, model_values, label=f"{model_name} Policy Value", marker="o", color=COLOR_MODEL)
        ax.fill_between(treated, model_lower, model_upper, alpha=0.12, color=COLOR_MODEL, label=f"{model_name} Konfidenzintervall")

        ax.plot(treated, ref_values, label=f"{comparison_model_name} Policy Value", linestyle="--", marker="o", color=COLOR_REFERENCE)
        ax.fill_between(treated, ref_lower, ref_upper, alpha=0.12, color=COLOR_REFERENCE, label=f"{comparison_model_name} Konfidenzintervall")

        ax.plot(treated, difference, label="Differenz (Modell - Referenz)", linestyle="-.", marker="x", color=COLOR_DIFFERENCE)
        ax.axhline(0, color=COLOR_BASELINE, linewidth=0.8, linestyle="--", label="Keine Differenz")

        ax.set_title(f"Policy Value Vergleich: {model_name} vs. {comparison_model_name}")
        ax.set_xlabel("Treated Percentage")
        ax.set_ylabel("Policy Value")
        ax.legend()
        ax.grid(True)

        plots[model_name] = fig

    return plots
