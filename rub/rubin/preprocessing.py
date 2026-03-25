from __future__ import annotations

"""Production-fähiges Preprocessing.
Das Preprocessing ist so gestaltet, dass es:
- im Analyse-Lauf "fit" auf Trainingsdaten gemacht werden kann,
- in Production ausschließlich "transform" nutzt (keine Leaks),
- Artefakte serialisieren kann (z. B. Feature-Liste, Column Order, Dtypes)."""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import pandas as pd


@dataclass
class FittedPreprocessor:
    feature_columns: List[str]
    categorical_columns: List[str]
    encoding_maps: Dict[str, Dict[str, int]]
    fillna_values: Dict[str, float]
    dtypes_after: Dict[str, str]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Sicherstellen, dass alle erwarteten Spalten existieren
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_columns]

        # Kategoriale Kodierung mit Behandlung unbekannter Ausprägungen
        for col in self.categorical_columns:
            if col in X.columns:
                mp = self.encoding_maps.get(col, {})
                # .map(dict) ist deutlich schneller als .map(lambda v: dict.get(str(v), -1))
                mapped = X[col].astype(str).map(mp)
                X[col] = mapped.fillna(-1).astype("int32").astype("category")

        # Fehlende numerische Werte auffüllen
        for col, val in self.fillna_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)

        # Datentypen setzen
        for col, dtype_name in self.dtypes_after.items():
            if col in X.columns:
                try:
                    X[col] = X[col].astype(dtype_name)
                except Exception:
                    # Bei fehlgeschlagenem Cast (z. B. neue Kategorien) ursprüngliche Werte beibehalten
                    pass

        return X


def fit_preprocessor(
    X: pd.DataFrame,
    categorical_columns: List[str],
    fill_na_method: str | None = "median",
) -> FittedPreprocessor:
    """Erzeugt ein fitted Preprocessing-Artefakt aus Trainingsdaten.
Speichert u. a.:
- die finale Feature-Reihenfolge
- Encoding-Mappings für kategoriale Spalten
- Imputation/Fill-Strategie für Missing Values
Dieses Artefakt ist zentral für reproduzierbares Scoring in Production."""
    X = X.copy()
    encoding_maps: Dict[str, Dict[str, int]] = {}
    for col in categorical_columns:
        X[col] = X[col].astype("object")
        uniques = pd.Series(X[col].astype(str).fillna("nan")).unique().tolist()
        mp = {u: i for i, u in enumerate(uniques)}
        encoding_maps[col] = mp
        X[col] = X[col].astype(str).map(mp).astype("int32").astype("category")

    num_cols = X.select_dtypes(exclude=["category", "object"]).columns.tolist()
    if fill_na_method == "median":
        fill_vals = X[num_cols].median(numeric_only=True).to_dict()
    elif fill_na_method == "mean":
        fill_vals = X[num_cols].mean(numeric_only=True).to_dict()
    elif fill_na_method == "zero":
        fill_vals = {c: 0.0 for c in num_cols}
    elif fill_na_method == "mode":
        modes = X[num_cols].mode()
        fill_vals = {c: float(modes[c].iloc[0]) for c in num_cols if not modes[c].empty} if not modes.empty else {}
    else:
        fill_vals = {}

    for c, v in fill_vals.items():
        X[c] = X[c].fillna(v)

    dtypes_after = X.dtypes.apply(lambda d: d.name).to_dict()
    return FittedPreprocessor(
        feature_columns=X.columns.tolist(),
        categorical_columns=categorical_columns,
        encoding_maps=encoding_maps,
        fillna_values={k: float(v) for k, v in fill_vals.items() if v is not None},
        dtypes_after=dtypes_after,
    )


@dataclass
class SimpleCSVPreprocessor:
    """Ein sehr leichtgewichtiges Preprocessing für CSV-basierte Workflows.
Dieses Preprocessing deckt zwei typische Anforderungen ab:
1) Spalten-Alignment: fehlende Spalten werden ergänzt, zusätzliche Spalten werden ignoriert.
2) Typ-Konvertierung: Dtypes werden soweit möglich wiederhergestellt (z. B. int/float/category).
Ergänzend wird ein "Schema" mitgeliefert, das in Production geprüft werden kann."""

    feature_columns: List[str]
    dtypes: Dict[str, str]
    categorical_columns: List[str] = field(default_factory=list)

    def infer_schema(self) -> "Schema":
        from rubin.utils.schema_utils import Schema
        return Schema(columns=list(self.feature_columns), dtypes=dict(self.dtypes), categorical_columns=list(self.categorical_columns))

    def validate(self, X: pd.DataFrame, strict: bool = False) -> "SchemaValidationResult":
        """Prüft X gegen das erwartete Schema.
strict=False:
- zusätzliche Spalten sind erlaubt (werden später ohnehin verworfen)
strict=True:
- zusätzliche Spalten führen zu ok=False"""
        from rubin.utils.schema_utils import validate_schema
        return validate_schema(X, self.infer_schema(), strict=strict)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1) Fehlende Spalten ergänzen
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = np.nan

        # 2) Reproduzierbare Spaltenreihenfolge
        X = X[self.feature_columns]

        # 3) Dtypes wiederherstellen (best-effort)
        for col, dt in self.dtypes.items():
            if col in X.columns:
                try:
                    X[col] = X[col].astype(dt)
                except Exception:
                    # In Production lieber robust bleiben und notfalls das Original behalten.
                    pass
        return X


def build_simple_preprocessor_from_dataframe(X: pd.DataFrame) -> SimpleCSVPreprocessor:
    """Erzeugt ein leichtgewichtiges Preprocessing-Artefakt aus einer Feature-Matrix.
Zusätzlich zu den Dtypes werden kategoriale Spalten explizit markiert. Dadurch
bleibt das abgeleitete Schema im Bundle vollständiger und in der Dokumentation
nachvollziehbar."""
    dtypes = X.dtypes.apply(lambda d: d.name).to_dict()
    categorical_columns = X.select_dtypes(include=["category", "object"]).columns.tolist()
    return SimpleCSVPreprocessor(
        feature_columns=X.columns.tolist(),
        dtypes=dtypes,
        categorical_columns=categorical_columns,
    )
