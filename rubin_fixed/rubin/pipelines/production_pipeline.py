from __future__ import annotations

"""Orchestrierung der Production-Pipeline (Scoring).
Die Production-Pipeline:
- lädt ein Bundle,
- wendet das gespeicherte Preprocessing an,
- ruft die gespeicherten Modelle auf,
- schreibt konsistente Scores/Outputs.
Dadurch kann Scoring in einem separaten Job laufen (z. B. Batch),
ohne dass Analyse-spezifische Nebenwirkungen auftreten."""


from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, Optional

import pandas as pd
import numpy as np
import pickle

from rubin.training import _predict_effect, SURROGATE_MODEL_NAME
from rubin.utils.data_utils import load_dtypes_json
from rubin.model_management import read_registry


@dataclass
class ProductionOutputs:
    cate: pd.DataFrame
    metadata: Dict[str, Any]


class ProductionPipeline:
    """Designentscheidung:
- Production ist bewusst „schlank“ gehalten: kein Training, kein Tuning, keine Feature-Auswahl.
- Die Qualität/Kompatibilität wird über Bundle-Artefakte abgesichert (Preprocessor + Schema)."""
    def __init__(self, bundle_path: str) -> None:
        self.last_schema_report = None  # wird beim Scoring gesetzt
        self.bundle_root = Path(bundle_path)
        self.models_dir = self.bundle_root / "models"

        # Harmlose sklearn-Warnung unterdrücken (EconML mischt DataFrame/numpy intern)
        import warnings
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module=r"sklearn\.utils\.validation",
        )
        # EconML nutzt intern den alten sklearn-Parameter 'force_all_finite',
        # der in sklearn 1.6 zu 'ensure_all_finite' umbenannt wurde.
        warnings.filterwarnings(
            "ignore",
            message=".*force_all_finite.*was renamed to.*ensure_all_finite",
            category=FutureWarning,
            module=r"sklearn",
        )

        with open(self.bundle_root / "preprocessor.pkl", "rb") as f:
            self.preprocessor = pickle.load(f)

        self.models: Dict[str, Any] = {}
        for pkl in sorted(self.models_dir.glob("*.pkl")):
            with open(pkl, "rb") as f:
                self.models[pkl.stem] = pickle.load(f)

        self.champion_model_name: Optional[str] = None
        try:
            manifest = read_registry(self.bundle_root)
            champion = manifest.get("champion")
            if isinstance(champion, str) and champion in self.models:
                self.champion_model_name = champion
        except Exception:
            self.champion_model_name = None

        self.bundle_dtypes = None
        dtypes_path = self.bundle_root / "dtypes.json"
        if dtypes_path.exists():
            try:
                self.bundle_dtypes = load_dtypes_json(str(dtypes_path))
            except Exception:
                self.bundle_dtypes = None

    @property
    def has_surrogate(self) -> bool:
        """Prüft, ob ein Surrogate-Einzelbaum im Bundle vorhanden ist."""
        return SURROGATE_MODEL_NAME in self.models

    def score_surrogate(self, X_raw: pd.DataFrame) -> ProductionOutputs:
        """Scoring ausschließlich mit dem Surrogate-Einzelbaum.

        Convenience-Methode für Kunden, die über den interpretierbaren
        Einzelbaum statt über das vollständige CATE-Modell gescoret werden."""
        if not self.has_surrogate:
            raise ValueError(
                f"Das Bundle enthält keinen Surrogate-Einzelbaum ('{SURROGATE_MODEL_NAME}'). "
                f"Verfügbare Modelle: {sorted(self.models.keys())}"
            )
        return self.score(X_raw, model_names=[SURROGATE_MODEL_NAME])

    def score(self, X_raw: pd.DataFrame, model_names: Optional[Iterable[str]] = None) -> ProductionOutputs:
        # Schema-Check vor dem Transform:
        # - fehlende Spalten werden später ergänzt, sollten aber gemeldet werden
        # - zusätzliche Spalten sind i. d. R. unkritisch (werden verworfen), können aber auf Datenänderungen hinweisen
        # Optionale Best-Effort-Typangleichung auf Basis der im Bundle gespeicherten Referenz.
        # Das erleichtert stabile Scorings, wenn Rohdaten aus anderen Exportwegen kommen.
        X_input = X_raw.copy()
        if isinstance(self.bundle_dtypes, dict):
            for col, dt in self.bundle_dtypes.items():
                if col in X_input.columns:
                    try:
                        X_input[col] = X_input[col].astype(dt)
                    except Exception:
                        pass

        try:
            if hasattr(self.preprocessor, "validate"):
                res = self.preprocessor.validate(X_input, strict=False)
                self.last_schema_report = res.to_dict()
        except Exception:
            self.last_schema_report = None

        X = self.preprocessor.transform(X_input)

        selected_names = list(model_names) if model_names is not None else None
        if selected_names is None:
            if self.champion_model_name is not None:
                selected_names = [self.champion_model_name]
            else:
                selected_names = sorted(self.models.keys())

        missing = [name for name in selected_names if name not in self.models]
        if missing:
            raise ValueError(f"Unbekannte Modellnamen im Bundle: {missing}. Verfügbar: {sorted(self.models)}")

        out = pd.DataFrame(index=X.index)
        for name in selected_names:
            model = self.models[name]
            cate = _predict_effect(model, X)
            cate = np.asarray(cate)

            if cate.ndim == 2 and cate.shape[1] > 1:
                # Multi-Treatment: K-1 CATE-Spalten + optimale Zuweisung
                n_effects = cate.shape[1]
                for k in range(n_effects):
                    out[f"cate_{name}_T{k+1}"] = cate[:, k]
                # Optimale Zuweisung
                best_effect = np.max(cate, axis=1)
                best_arm = np.argmax(cate, axis=1) + 1
                out[f"optimal_treatment_{name}"] = np.where(best_effect > 0, best_arm, 0)
                # Confidence: Differenz zwischen bestem und zweitbestem Effekt
                if n_effects > 1:
                    sorted_cate = np.sort(cate, axis=1)[:, ::-1]
                    out[f"treatment_confidence_{name}"] = sorted_cate[:, 0] - sorted_cate[:, 1]
                else:
                    out[f"treatment_confidence_{name}"] = np.abs(cate[:, 0])
            else:
                out[f"cate_{name}"] = cate.reshape(-1)

        bundle_id = self.bundle_root.name
        out["bundle_id"] = bundle_id
        out["model_name"] = selected_names[0] if len(selected_names) == 1 else "multiple"

        return ProductionOutputs(
            cate=out,
            metadata={
                "bundle_id": bundle_id,
                "models_used": list(selected_names),
                "champion_model": self.champion_model_name,
            },
        )
