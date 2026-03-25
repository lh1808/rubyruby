from __future__ import annotations

"""Artefakt- und Bundle-Handling (Export/Import).
Ein Bundle enthält alles, was für ein reproduzierbares Scoring benötigt wird:
- Konfiguration (Snapshot)
- Preprocessing-Artefakte (z. B. Feature-Liste, Dtype-Alignment)
- trainierte Modelle (Pickles)
Damit wird eine klare Trennung zwischen Analyse (Entwicklung) und Production
(Scoring) ermöglicht."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import pickle
import shutil
import datetime


@dataclass(frozen=True)
class BundlePaths:
    root: Path
    config: Path
    preprocessor: Path
    models_dir: Path
    metadata: Path


class ArtifactBundler:
    """Sammelt und exportiert Artefakte für die Production.
Ein Bundle ist bewusst ein *Verzeichnis* (statt z. B. einer einzelnen Pickle-Datei), damit:
- Inhalte leicht inspiziert werden können,
- Metadaten (JSON) und Dokumentation (Snapshot der Konfiguration) sauber daneben liegen."""
    def __init__(self, base_dir: str = "bundles") -> None:
        self.base_dir = Path(base_dir)

    def create_bundle_dir(self, bundle_id: Optional[str] = None) -> BundlePaths:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bundle_id = bundle_id or f"bundle_{ts}"
        root = self.base_dir / bundle_id
        root.mkdir(parents=True, exist_ok=False)
        models_dir = root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return BundlePaths(
            root=root,
            config=root / "config_snapshot.yml",
            preprocessor=root / "preprocessor.pkl",
            models_dir=models_dir,
            metadata=root / "metadata.json",
        )

    def write_config(self, paths: BundlePaths, config_path: str) -> None:
        shutil.copy2(config_path, paths.config)

    def write_preprocessor(self, paths: BundlePaths, preprocessor: Any) -> None:
        """Serialisiert das Preprocessing und (wenn möglich) das zugehörige Schema.
Warum Schema mitloggen?
- Production-Inputs ändern sich oft schleichend. Ein gespeichertes Schema erlaubt,
beim Scoring sofort auf fehlende/anders typisierte Features hinzuweisen.
- So werden Fehler früh sichtbar, statt stillschweigend die Modellqualität zu verschlechtern."""
        with open(paths.preprocessor, "wb") as f:
            pickle.dump(preprocessor, f)

        # optional: Schema und Dtype-Referenz als JSON (best-effort)
        try:
            from rubin.utils.schema_utils import save_schema
            if hasattr(preprocessor, "infer_schema"):
                schema = preprocessor.infer_schema()
                save_schema(schema, str(paths.root / "schema.json"))

            # Dtype-Referenz separat ablegen, damit Production/Debugging auch ohne
            # Laden des Pickles nachvollziehen kann, welche Zieltypen erwartet werden.
            dtypes = None
            if hasattr(preprocessor, "dtypes"):
                dtypes = getattr(preprocessor, "dtypes")
            elif hasattr(preprocessor, "dtypes_after"):
                dtypes = getattr(preprocessor, "dtypes_after")
            if isinstance(dtypes, dict) and dtypes:
                with open(paths.root / "dtypes.json", "w", encoding="utf-8") as f:
                    json.dump(dtypes, f, indent=2, ensure_ascii=False)
        except Exception:
            # Schema/Dtypes sind hilfreich, aber nicht kritisch für ein lauffähiges Bundle.
            pass

    def write_model(self, paths: BundlePaths, model_name: str, model_obj: Any) -> Path:
        out = paths.models_dir / f"{model_name}.pkl"
        with open(out, "wb") as f:
            pickle.dump(model_obj, f)
        return out

    def write_metadata(self, paths: BundlePaths, metadata: Dict[str, Any]) -> None:
        """Schreibt Metadaten für das Bundle.
Zusätzlich zum übergebenen Dict werden sinnvolle Basisinformationen ergänzt:
- created_at (UTC)
Warum?
- Im Fehlerfall (oder bei Audits) ist schnell nachvollziehbar, wann das Bundle erzeugt wurde."""
        from datetime import datetime, timezone
        import platform

        enriched = dict(metadata)

        enriched.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
        enriched.setdefault("python_version", platform.python_version())

        with open(paths.metadata, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2, ensure_ascii=False)

