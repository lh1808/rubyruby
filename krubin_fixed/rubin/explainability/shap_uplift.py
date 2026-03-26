from __future__ import annotations

"""SHAP-basierte Erklärungen für Uplift/CATE.
Dieses Modul bündelt zwei Ebenen:
1. eine robuste, modellagnostische Berechnung von SHAP-Werten für die
Uplift-Funktion ``f(X) = CATE(X)``;
2. einen vollständigen Plot-Satz (Beeswarm, Mean Impact, Max Impact,
CATE-Profil, SHAP-PDP, SHAP-Scatter).
Diese Plots setzen voraus, dass das zugrunde liegende Modell
``shap_values(X=...)`` unterstützt und ein EconML-kompatibles SHAP-Ergebnis
liefert. Falls das nicht der Fall ist, kann weiterhin die generische
SHAP-Berechnung verwendet werden."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rubin.utils.plot_theme import apply_rubin_theme, RUBIN_COLORS, COLOR_MODEL
apply_rubin_theme()


def shap_available() -> bool:
    """Prüft, ob das Paket ``shap`` importierbar ist."""
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class ShapUpliftResult:
    """Ergebnisobjekt für modellagnostische SHAP-Erklärungen."""

    feature_names: list[str]
    shap_values: np.ndarray
    base_values: np.ndarray
    expected_value: float

    def mean_abs_importance(self) -> pd.Series:
        imp = np.mean(np.abs(self.shap_values), axis=0)
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)


@dataclass
class ShapPlotBundle:
    """Vollständiger SHAP-Plot-Satz für Uplift-Auswertung."""

    summary_plots: plt.Figure
    average_plots: plt.Figure
    pdp_plots: plt.Figure
    scatter_plots: plt.Figure
    importance: pd.Series


def _make_uplift_predict_fn(model: object) -> Callable[[pd.DataFrame], np.ndarray]:
    """Erzeugt eine Predict-Funktion für SHAP.
    Bei BT: Rückgabe (n,); bei MT: Rückgabe (n,) als Norm über Arme,
    damit SHAP einen skalaren Output hat."""
    if hasattr(model, "const_marginal_effect"):
        def _pred(x: pd.DataFrame) -> np.ndarray:
            y = np.asarray(model.const_marginal_effect(x))
            if y.ndim == 2 and y.shape[1] == 1:
                return y.reshape(-1)
            if y.ndim == 2 and y.shape[1] > 1:
                # MT: max-Effekt als skalarer Output für SHAP
                return np.max(y, axis=1)
            return y.reshape(-1)
        return _pred

    if hasattr(model, "effect"):
        def _pred(x: pd.DataFrame) -> np.ndarray:
            y = np.asarray(model.effect(x))
            if y.ndim == 2 and y.shape[1] == 1:
                return y.reshape(-1)
            if y.ndim == 2 and y.shape[1] > 1:
                return np.max(y, axis=1)
            return y.reshape(-1)
        return _pred

    raise TypeError(
        "Das übergebene Modell unterstützt weder 'const_marginal_effect' noch 'effect'. "
        "Für Explainability wird eine dieser Methoden benötigt."
    )


def compute_shap_for_uplift(
    model: object,
    X: pd.DataFrame,
    background: Optional[pd.DataFrame] = None,
    max_background_rows: int = 200,
    seed: int = 42,
    feature_names: Optional[Sequence[str]] = None,
) -> ShapUpliftResult:
    """Berechnet modellagnostische SHAP-Werte für die Uplift-Funktion."""
    if not shap_available():
        raise ImportError(
            "SHAP ist nicht installiert. Bitte installiere das Paket 'shap' oder verwende "
            "Permutation-Importance."
        )

    import shap

    rng = np.random.default_rng(seed)
    feature_names = list(feature_names) if feature_names is not None else list(X.columns)

    if background is None:
        if len(X) <= max_background_rows:
            background = X
        else:
            idx = rng.choice(len(X), size=max_background_rows, replace=False)
            background = X.iloc[idx]

    predict_fn = _make_uplift_predict_fn(model)
    explainer = shap.Explainer(predict_fn, background)
    explanation = explainer(X)

    shap_vals = np.asarray(explanation.values)
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(-1, 1)

    base_vals = np.asarray(explanation.base_values)
    if base_vals.ndim == 0:
        base_vals = np.full(shape=(len(X),), fill_value=float(base_vals))

    expected_value = float(np.mean(base_vals))

    return ShapUpliftResult(
        feature_names=feature_names,
        shap_values=shap_vals,
        base_values=base_vals,
        expected_value=expected_value,
    )


def _extract_primary_explanation(raw_shap_values: object):
    """Extrahiert die primäre SHAP-Explanation aus EconML-Strukturen.
Erwartet typischerweise die Struktur ``shap_values["Y0"]["T0_1"]``.
Falls die Struktur nicht vorliegt, wird versucht, das Objekt direkt als
``shap.Explanation`` zu verwenden."""
    try:
        return raw_shap_values["Y0"]["T0_1"]
    except Exception:
        return raw_shap_values



def _is_categorical(series: pd.Series) -> bool:
    return bool(isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series))



def _plot_binned_mean(ax: plt.Axes, feature_values: pd.Series, values: np.ndarray, title: str, y_label: str, num_bins: int) -> None:
    """Plottet segmentierte Mittelwerte für numerische oder kategoriale Features."""
    df = pd.DataFrame({"feature_value": feature_values, "value": np.asarray(values).reshape(-1)})

    if _is_categorical(df["feature_value"]):
        agg = (
            df.groupby("feature_value", dropna=False)["value"]
            .mean()
            .sort_values(ascending=False)
        )
        ax.bar(agg.index.astype(str), agg.values, color=COLOR_MODEL, edgecolor=RUBIN_COLORS["ruby_dark"], linewidth=0.5)
        ax.tick_params(axis="x", rotation=45)
    else:
        series = pd.to_numeric(df["feature_value"], errors="coerce")
        if series.notna().sum() <= 1:
            ax.text(0.5, 0.5, "Zu wenige gültige Werte", ha="center", va="center", color=RUBIN_COLORS["slate"])
            ax.set_title(title)
            ax.set_ylabel(y_label)
            return
        n_bins = max(2, min(int(num_bins), int(series.nunique())))
        bins = np.linspace(series.min(), series.max(), num=n_bins + 1)
        df = df.loc[series.notna()].copy()
        df["feature_value"] = series.loc[series.notna()].to_numpy()
        df["bin"] = pd.cut(df["feature_value"], bins=bins, include_lowest=True, duplicates="drop")
        agg = df.groupby("bin", observed=False)["value"].mean().reset_index()
        centers = agg["bin"].apply(lambda x: float(x.left + (x.right - x.left) / 2.0))
        ax.bar(centers.astype(str), agg["value"].to_numpy(), color=COLOR_MODEL, edgecolor=RUBIN_COLORS["ruby_dark"], linewidth=0.5)
        ax.tick_params(axis="x", rotation=45)

    ax.set_title(title)
    ax.set_xlabel(str(feature_values.name))
    ax.set_ylabel(y_label)



def build_shap_plots(
    model: object,
    X: pd.DataFrame,
    data: pd.DataFrame,
    cate: Sequence[float],
    top_n: int,
    num_bins: int = 10,
) -> ShapPlotBundle:
    """Erzeugt den vollständigen SHAP-Plot-Satz.
Voraussetzungen
---------------
* Das Modell stellt ``shap_values(X=...)`` bereit.
* Das Ergebnis enthält eine nutzbare ``shap.Explanation``.
Die Funktion erzeugt folgende Plots:
* Beeswarm
* Mean Impact (Bar Summary)
* Max Impact (Beeswarm nach maximaler absoluter Wirkung)
* CATE-Profil je Top-Feature
* SHAP-PDP je Top-Feature
* SHAP-Scatter je Top-Feature"""
    if not shap_available():
        raise ImportError("SHAP ist nicht installiert.")
    if not hasattr(model, "shap_values"):
        raise TypeError(
            "Das Modell stellt keine Methode 'shap_values' bereit. "
            "Für diesen Plot-Satz wird ein EconML-kompatibles Modell benötigt."
        )

    import shap

    raw_shap_values = model.shap_values(X=X)
    explanation = _extract_primary_explanation(raw_shap_values)

    if not hasattr(explanation, "values"):
        raise TypeError(
            "Das Ergebnis von 'model.shap_values' hat nicht die erwartete Struktur. "
            "Die SHAP-Plots können dafür nicht erzeugt werden."
        )

    mean_abs = explanation.abs.mean(axis=0)
    mean_abs_values = np.asarray(getattr(mean_abs, "values", mean_abs)).reshape(-1)
    feature_names = list(X.columns)
    sorted_idx = np.argsort(mean_abs_values)[::-1]
    top_idx = sorted_idx[: min(top_n, len(feature_names))]
    importance = pd.Series(mean_abs_values, index=feature_names).sort_values(ascending=False)

    fig1 = plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1)
    shap.plots.beeswarm(
        explanation,
        max_display=max(21, min(len(feature_names), top_n + 1)),
        color_bar=False,
        plot_size=None,
        show=False,
    )
    plt.title("Beeswarm")

    plt.subplot(3, 2, 3)
    shap.summary_plot(
        explanation,
        plot_type="bar",
        max_display=min(20, len(feature_names)),
        color_bar=False,
        plot_size=None,
        show=False,
    )
    plt.title("Mean Impact")

    plt.subplot(3, 2, 4)
    shap.plots.beeswarm(
        explanation.abs,
        max_display=max(21, min(len(feature_names), top_n + 1)),
        order=shap.Explanation.abs.max(0),
        color_bar=False,
        plot_size=None,
        show=False,
    )
    plt.title("Max Impact")
    plt.tight_layout()

    cate_values = np.asarray(cate).reshape(-1)
    fig2 = plt.figure(figsize=(15, max(5, 4 * len(top_idx))))
    for i, feature_idx in enumerate(top_idx, start=1):
        feature_name = feature_names[int(feature_idx)]
        ax = fig2.add_subplot(len(top_idx), 1, i)
        feature_values = data[feature_name] if feature_name in data.columns else X[feature_name]
        _plot_binned_mean(
            ax=ax,
            feature_values=feature_values,
            values=cate_values,
            title=f"CATE-Profil für {feature_name}",
            y_label="Durchschnittlicher CATE",
            num_bins=num_bins,
        )
    fig2.tight_layout()

    fig3 = plt.figure(figsize=(15, max(5, 4 * len(top_idx))))
    for i, feature_idx in enumerate(top_idx, start=1):
        feature_name = feature_names[int(feature_idx)]
        ax = fig3.add_subplot(len(top_idx), 1, i)
        shap_values_for_feature = np.asarray(explanation[:, int(feature_idx)].values).reshape(-1)
        _plot_binned_mean(
            ax=ax,
            feature_values=X[feature_name],
            values=shap_values_for_feature,
            title=f"SHAP-PDP für {feature_name}",
            y_label="Durchschnittlicher SHAP-Wert",
            num_bins=num_bins,
        )
    fig3.tight_layout()

    fig4, axes = plt.subplots(len(top_idx), 1, figsize=(15, max(5, 4 * len(top_idx))))
    if len(top_idx) == 1:
        axes = [axes]
    for ax, feature_idx in zip(axes, top_idx):
        feature_name = feature_names[int(feature_idx)]
        shap.plots.scatter(explanation[:, feature_name], ax=ax, show=False)
        ax.set_title(f"SHAP-Scatter für {feature_name}")
    fig4.tight_layout()

    return ShapPlotBundle(
        summary_plots=fig1,
        average_plots=fig2,
        pdp_plots=fig3,
        scatter_plots=fig4,
        importance=importance,
    )
