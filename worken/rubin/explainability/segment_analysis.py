from __future__ import annotations

"""Segmentbasierte Analyse für Uplift-Scores.
Neben globalen Kennzahlen ist in der Praxis oft relevant, welche Kundensegmente
besonders stark oder schwach auf eine Maßnahme reagieren. Dieses Modul stellt
zwei Ebenen bereit:
1. einen klassischen Score-Report nach Quantilen/Dezilen;
2. eine Segmentanalyse entlang einzelner Merkmale, damit sichtbar wird, welche
Teilpopulationen hohe bzw. niedrige erwartete Effekte aufweisen."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SegmentReport:
    """Tabellarisches Segment-Reporting nach Score-Quantilen."""

    table: pd.DataFrame

    def to_csv(self, path: str) -> None:
        self.table.to_csv(path, index=False)


@dataclass
class FeatureSegmentReport:
    """Segmentanalyse je Merkmal."""

    table: pd.DataFrame

    def to_csv(self, path: str) -> None:
        self.table.to_csv(path, index=False)


def build_segment_report(
    uplift: np.ndarray,
    y: Optional[np.ndarray] = None,
    t: Optional[np.ndarray] = None,
    quantiles: int = 10,
) -> SegmentReport:
    """Erzeugt eine Segmenttabelle nach Score-Quantilen."""
    uplift = np.asarray(uplift).reshape(-1)
    df = pd.DataFrame({"uplift": uplift})

    # Robuste Segmentbildung:
    # - bei konstanten Scores genau ein Segment bilden
    # - bei vielen Ties über den Rang arbeiten, damit qcut nicht nur NaN liefert
    n_unique = pd.Series(uplift).nunique(dropna=True)
    if n_unique <= 1:
        df["segment"] = 0
    else:
        n_bins = max(2, min(int(quantiles), len(df), int(n_unique)))
        ranked = pd.Series(uplift).rank(method="first")
        df["segment"] = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop").astype(int)

    agg = {
        "n": ("uplift", "size"),
        "uplift_mean": ("uplift", "mean"),
        "uplift_p10": ("uplift", lambda s: np.quantile(s, 0.10)),
        "uplift_p50": ("uplift", lambda s: np.quantile(s, 0.50)),
        "uplift_p90": ("uplift", lambda s: np.quantile(s, 0.90)),
    }

    if y is not None:
        df["y"] = np.asarray(y).reshape(-1)
        agg.update({"y_mean": ("y", "mean")})
    if t is not None:
        df["t"] = np.asarray(t).reshape(-1)
        agg.update({"t_rate": ("t", "mean")})

    out = (
        df.groupby("segment")
        .agg(**agg)
        .reset_index()
        .sort_values("segment", ascending=False)
    )

    out["cum_n"] = out["n"].cumsum()
    out["cum_uplift_mean"] = (out["uplift_mean"] * out["n"]).cumsum() / out["cum_n"]

    return SegmentReport(table=out)


def _bin_series(series: pd.Series, max_bins: int) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        valid = pd.to_numeric(series, errors="coerce")
        n_unique = valid.nunique(dropna=True)
        if n_unique <= 1:
            return pd.Series(["konstant"] * len(series), index=series.index, name=series.name)
        bins = max(2, min(int(max_bins), int(n_unique)))
        return pd.qcut(valid, q=bins, duplicates="drop").astype(str)
    return series.astype(str).fillna("<NA>")


def build_feature_segment_report(
    X: pd.DataFrame,
    uplift: np.ndarray,
    y: Optional[np.ndarray] = None,
    t: Optional[np.ndarray] = None,
    top_features: Optional[list[str]] = None,
    max_features: int = 8,
    max_bins: int = 6,
    max_categories: int = 15,
) -> FeatureSegmentReport:
    """Erstellt eine Segmentanalyse je Merkmal.
Für numerische Merkmale werden Quantil-Bins gebildet. Für kategoriale
Merkmale werden die beobachteten Kategorien verwendet. Zusätzlich werden,
sofern ``y`` und ``t`` vorliegen, deskriptive Erfolgsraten in Treatment und
Kontrollgruppe sowie deren Differenz ausgegeben."""
    uplift = np.asarray(uplift).reshape(-1)
    df = X.copy()
    df["__uplift__"] = uplift
    if y is not None:
        df["__y__"] = np.asarray(y).reshape(-1)
    if t is not None:
        df["__t__"] = np.asarray(t).reshape(-1)

    if top_features is None:
        top_features = list(X.columns[:max_features])
    else:
        top_features = list(top_features[:max_features])

    rows: list[dict[str, object]] = []
    for feature in top_features:
        if feature not in X.columns:
            continue
        groups = _bin_series(X[feature], max_bins=max_bins)
        tmp = pd.DataFrame({
            "feature": feature,
            "segment": groups.astype(str),
            "uplift": uplift,
        }, index=X.index)
        if y is not None:
            tmp["y"] = np.asarray(y).reshape(-1)
        if t is not None:
            tmp["t"] = np.asarray(t).reshape(-1)

        # Sehr breite kategoriale Merkmale begrenzen, damit die Tabelle lesbar bleibt.
        counts = tmp["segment"].value_counts(dropna=False)
        keep = set(counts.head(max_categories).index.astype(str))
        tmp.loc[~tmp["segment"].isin(keep), "segment"] = "__andere__"

        for seg_name, grp in tmp.groupby("segment", dropna=False):
            row: dict[str, object] = {
                "feature": feature,
                "segment": str(seg_name),
                "n": int(len(grp)),
                "uplift_mean": float(grp["uplift"].mean()),
                "uplift_p10": float(grp["uplift"].quantile(0.10)),
                "uplift_p50": float(grp["uplift"].quantile(0.50)),
                "uplift_p90": float(grp["uplift"].quantile(0.90)),
            }
            if y is not None and t is not None:
                treated = grp.loc[grp["t"] == 1, "y"]
                control = grp.loc[grp["t"] == 0, "y"]
                row["treated_rate"] = float(treated.mean()) if len(treated) else np.nan
                row["control_rate"] = float(control.mean()) if len(control) else np.nan
                row["observed_diff"] = (
                    float(treated.mean() - control.mean())
                    if len(treated) and len(control)
                    else np.nan
                )
                row["t_rate"] = float(grp["t"].mean())
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["feature", "uplift_mean"], ascending=[True, False]).reset_index(drop=True)
    return FeatureSegmentReport(table=out)
