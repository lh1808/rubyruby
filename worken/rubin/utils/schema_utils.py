from __future__ import annotations

"""Schema-Validierung für stabile Production-Scorings.
Dieses Modul nutzt Pydantic, um das Schema-Format selbst strikt zu validieren
(z. B. wenn ein Schema-JSON beschädigt ist oder ein Key falsch heißt). Die eigentliche
Prüfung von DataFrames gegen dieses Schema ist weiterhin bewusst „leichtgewichtig“:
Spalten, Datentypen und optionale Kennzeichnung von kategorischen Spalten.
Für inhaltliche Checks (Wertebereiche, Drift) sind spezialisierte Werkzeuge sinnvoll."""

from typing import Dict, List, Optional, Tuple
import json
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


_STRICT = ConfigDict(extra="forbid")


class Schema(BaseModel):
    model_config = _STRICT

    columns: List[str]
    dtypes: Dict[str, str]
    categorical_columns: List[str] = Field(default_factory=list)

    @staticmethod
    def from_dataframe(df: pd.DataFrame, categorical_columns: Optional[List[str]] = None) -> "Schema":
        cats = categorical_columns or []
        return Schema(
            columns=list(df.columns),
            dtypes={c: str(df[c].dtype) for c in df.columns},
            categorical_columns=list(cats),
        )

    def to_dict(self) -> dict:
        return self.model_dump()


class SchemaValidationResult(BaseModel):
    model_config = _STRICT

    ok: bool
    missing_columns: List[str] = Field(default_factory=list)
    extra_columns: List[str] = Field(default_factory=list)
    dtype_mismatches: Dict[str, Tuple[str, str]] = Field(default_factory=dict)  # col -> (expected, actual)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "missing_columns": self.missing_columns,
            "extra_columns": self.extra_columns,
            "dtype_mismatches": {k: {"expected": v[0], "actual": v[1]} for k, v in self.dtype_mismatches.items()},
        }


def save_schema(schema: Schema, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema.to_dict(), f, ensure_ascii=False, indent=2)


def load_schema(path: str) -> Schema:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return Schema.model_validate(raw)


def validate_schema(df: pd.DataFrame, expected: Schema, strict: bool = False) -> SchemaValidationResult:
    missing = [c for c in expected.columns if c not in df.columns]
    extra = [c for c in df.columns if c not in expected.columns]

    mismatches: Dict[str, Tuple[str, str]] = {}
    for c in expected.columns:
        if c in df.columns:
            exp = expected.dtypes.get(c)
            act = str(df[c].dtype)
            if exp is not None and exp != act:
                mismatches[c] = (exp, act)

    ok = (len(missing) == 0) and (len(mismatches) == 0) and ((not strict) or (len(extra) == 0))
    return SchemaValidationResult(ok=ok, missing_columns=missing, extra_columns=extra, dtype_mismatches=mismatches)
