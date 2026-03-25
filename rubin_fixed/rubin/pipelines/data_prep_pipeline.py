from __future__ import annotations

"""DataPrepPipeline.
Dieses Modul implementiert die Datenaufbereitung als eigenständige, objektorientierte
Pipeline. Alle Parameter werden über die zentrale Projekt-Konfiguration (`config.yml`)
gesteuert (Sektion `data_prep`).
Ziel der DataPrepPipeline ist es, aus Rohdaten reproduzierbar die drei Kernobjekte
für kausale Verfahren zu erzeugen:
- **X**: Feature-Matrix
- **T**: Treatment (diskret: 0/1 bei Binary Treatment; 0, 1, …, K-1 bei Multi-Treatment)
- **Y**: Outcome (0/1)
Zusätzlich wird ein Preprocessing-Artefakt erzeugt, das in der Analyse- und später
in der Produktionspipeline wiederverwendet werden kann."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import json
import logging
import pickle

import numpy as np
import pandas as pd
from rubin.preprocessing import FittedPreprocessor, fit_preprocessor
from rubin.settings import AnalysisConfig, DataPrepConfig, load_config
from rubin.utils.data_utils import reduce_mem_usage


@dataclass
class DataPrepOutputs:
    """Outputs der Datenaufbereitung."""

    X: pd.DataFrame
    T: np.ndarray
    Y: np.ndarray
    S: Optional[np.ndarray]
    preprocessor: FittedPreprocessor


class DataPrepPipeline:
    """Objektorientierte Datenaufbereitung."""

    _logger = logging.getLogger("rubin.dataprep")

    def __init__(self, cfg: AnalysisConfig, dataprep_cfg: DataPrepConfig) -> None:
        self.cfg = cfg
        self.dp = dataprep_cfg

        # Sanity-Checks früh, damit Fehler nicht erst nach teuren I/O-Schritten auffallen.
        if not self.dp.data_path:
            raise ValueError("data_prep.data_path ist leer. Bitte mindestens eine Eingabedatei konfigurieren.")
        if not self.dp.target or not self.dp.treatment:
            raise ValueError("data_prep.target und data_prep.treatment müssen gesetzt sein.")

    @classmethod
    def from_config_path(cls, config_path: str) -> "DataPrepPipeline":
        """Erstellt die Pipeline aus der zentralen Projekt-Konfiguration."""

        cfg = load_config(config_path)
        if cfg.data_prep is None:
            raise ValueError(
                "In der Konfiguration fehlt die Sektion 'data_prep'. "
                "Für DataPrep-Läufe muss diese Sektion vorhanden sein."
            )
        return cls(cfg, cfg.data_prep)

    def _read_files(self, file_paths: Optional[List[str]] = None, merge_only: bool = False) -> pd.DataFrame:
        dp = self.dp
        paths = file_paths if file_paths is not None else dp.data_path
        df_list: List[pd.DataFrame] = []

        for file_path in paths:
            p = str(file_path)
            p_lower = p.lower()
            if p_lower.endswith(".csv"):
                obj = pd.read_csv(
                    file_path,
                    delimiter=dp.delimiter,
                    low_memory=False,
                    chunksize=dp.chunksize,
                )
            elif p_lower.endswith((".parquet", ".pq")):
                # Parquet liefert immer direkt ein DataFrame; eine Chunk-Verarbeitung ist hier
                # bewusst nicht vorgesehen.
                try:
                    obj = pd.read_parquet(file_path)
                except ImportError as e:
                    raise ImportError(
                        "Für Parquet-Dateien wird 'pyarrow' oder 'fastparquet' benötigt. "
                        "Bitte die Abhängigkeiten aus requirements.txt installieren."
                    ) from e
            elif p_lower.endswith(".sas7bdat"):
                obj = pd.read_sas(file_path, chunksize=dp.chunksize, encoding=dp.sas_encoding)
            else:
                raise ValueError(f"Nicht unterstützter Dateityp in data_prep.data_path: {file_path}")

            # pandas liefert bei gesetztem chunksize einen Iterator, sonst direkt ein DataFrame.
            if isinstance(obj, pd.DataFrame):
                iterator = iter([obj])
            else:
                iterator = iter(obj)

            chunk_list: List[pd.DataFrame] = []
            try:
                first = next(iterator)
            except StopIteration as e:
                raise ValueError(f"Die Eingabedatei '{file_path}' enthält keine Daten.") from e

            # Spaltennamen vereinheitlichen, damit Feature-Dictionary, CSV und SAS identisch behandelt werden.
            first.columns = [str(c).upper() for c in first.columns]
            target_col = str(dp.target).upper()
            treat_col = str(dp.treatment).upper()

            required_columns = [target_col, treat_col]
            missing_required = [c for c in required_columns if c not in first.columns]
            if missing_required:
                raise ValueError(
                    f"Pflichtspalten fehlen in '{file_path}': {missing_required}. "
                    f"Verfügbare Spalten (erste Datei/erster Chunk): {list(first.columns)[:20]}"
                )

            # Replacement-Mappings (z. B. 'J'/'N' -> 0/1)
            if dp.treatment_replacement:
                first[treat_col] = first[treat_col].replace(dp.treatment_replacement)
            if dp.target_replacement:
                first[target_col] = first[target_col].replace(dp.target_replacement)

            dtypes = first.dtypes.to_dict()
            chunk_list.append(first)

            for chunk in iterator:
                chunk.columns = [str(c).upper() for c in chunk.columns]
                if dp.treatment_replacement:
                    chunk[treat_col] = chunk[treat_col].replace(dp.treatment_replacement)
                if dp.target_replacement:
                    chunk[target_col] = chunk[target_col].replace(dp.target_replacement)
                for col, dt in dtypes.items():
                    if col in chunk.columns:
                        try:
                            chunk[col] = chunk[col].astype(dt)
                        except Exception:
                            # Typ-Downcasts sind best-effort; im Zweifel bleibt der Chunk wie er ist.
                            pass
                chunk_list.append(chunk)

            df_list.append(pd.concat(chunk_list, ignore_index=True))

        if len(df_list) == 1:
            return df_list[0]

        # Eval-Daten werden immer zusammengeführt (kein treatment_only).
        opt = "merge" if merge_only else dp.multiple_files_option
        if opt == "merge":
            return pd.concat(df_list, ignore_index=True)

        if opt == "treatment_only":
            treat_col = str(dp.treatment).upper()
            treatments = [df[df[treat_col] == 1] for df in df_list]
            ctrl_like = treatments[dp.control_file_index].copy()
            ctrl_like[treat_col] = 0
            del treatments[dp.control_file_index]
            return pd.concat([ctrl_like] + treatments, ignore_index=True)

        raise ValueError(f"Unbekannte data_prep.multiple_files_option={opt}")

    def run(self) -> DataPrepOutputs:
        dp = self.dp
        out_dir = Path(dp.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        mlflow_run = None
        if dp.log_to_mlflow:
            try:
                import mlflow
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "MLflow ist nicht installiert. Für DataPrep-Läufe mit Logging wird MLflow benötigt (pip install mlflow)."
                ) from e

            exp_name = dp.mlflow_experiment_name or self.cfg.mlflow.experiment_name
            mlflow.set_experiment(exp_name)
            mlflow.start_run()
            mlflow_run = mlflow.active_run()
            # Snapshot der zentralen Konfiguration ist in DataPrep-Läufen sehr hilfreich,
            # weil Datenaufbereitung und Analyse damit reproduzierbar bleiben.
            mlflow.log_artifact(self.cfg.source_config_path)

        try:
            return self._run_inner(dp, out_dir)
        finally:
            if dp.log_to_mlflow and mlflow_run is not None:
                import mlflow
                mlflow.end_run()

    def _run_inner(self, dp, out_dir) -> DataPrepOutputs:
        """Innere Logik der Datenaufbereitung (von run() aufgerufen)."""
        has_eval = dp.eval_data_path and len(dp.eval_data_path) > 0
        total = 7 if has_eval else 6
        step = [0]
        def _progress(label):
            step[0] += 1
            print(f"[rubin] Step {step[0]}/{total}: {label}", flush=True)

        _progress("Dateien einlesen")
        df = self._read_files()

        _progress("Deduplizierung")
        # Deduplizierung: auf einen Eintrag pro Kunde reduzieren.
        # Geschieht VOR der Feature-Reduktion, da die ID-Spalte typischerweise
        # kein Feature ist und nach der Deduplizierung entfernt wird.
        if dp.deduplicate:
            if not dp.deduplicate_id_column:
                raise ValueError(
                    "data_prep.deduplicate ist aktiviert, aber data_prep.deduplicate_id_column ist nicht gesetzt. "
                    "Bitte den Spaltennamen angeben, der die Kunden-ID enthält (z. B. 'PARTNER_ID')."
                )
            id_col = str(dp.deduplicate_id_column).upper()
            if id_col not in df.columns:
                raise ValueError(
                    f"Deduplizierungsspalte '{id_col}' ist nicht im Datensatz vorhanden. "
                    f"Verfügbare Spalten: {list(df.columns)[:20]}"
                )
            n_before = len(df)
            df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
            n_after = len(df)
            n_removed = n_before - n_after
            if n_removed > 0:
                import logging
                logging.getLogger("rubin.dataprep").info(
                    "Deduplizierung auf '%s': %d → %d Zeilen (%d Duplikate entfernt).",
                    id_col, n_before, n_after, n_removed,
                )
            if dp.log_to_mlflow:
                import mlflow
                mlflow.log_param("deduplicate_column", id_col)
                mlflow.log_metric("deduplicate_rows_before", n_before)
                mlflow.log_metric("deduplicate_rows_after", n_after)
                mlflow.log_metric("deduplicate_rows_removed", n_removed)

        _progress("Feature-Extraktion")

        score_col = str(dp.score_name).upper() if dp.score_name else None
        target_col = str(dp.target).upper()
        treat_col = str(dp.treatment).upper()

        if dp.feature_path:
            # Feature-Dictionary vorhanden → verwende ROLE/NAME/LEVEL
            feature_dictionary = pd.read_excel(dp.feature_path)
            required_feature_dict_columns = {"ROLE", "NAME", "LEVEL"}
            missing_feature_dict_columns = sorted(required_feature_dict_columns - set(feature_dictionary.columns))
            if missing_feature_dict_columns:
                raise ValueError(
                    "Im Feature-Dictionary fehlen Pflichtspalten: "
                    f"{missing_feature_dict_columns}. Erwartet werden mindestens ROLE, NAME und LEVEL."
                )
            if dp.log_to_mlflow:
                import mlflow
                mlflow.log_param("feature_path", dp.feature_path)

            input_list = feature_dictionary.loc[feature_dictionary["ROLE"].astype(str).str.upper() == "INPUT", "NAME"].astype(str).str.upper().tolist()
            available_features = [f for f in input_list if f in df.columns]
            X = df[available_features].copy()

            nominal_list = feature_dictionary.loc[feature_dictionary["LEVEL"].astype(str).str.upper() == "NOMINAL", "NAME"].astype(str).str.upper().tolist()
            categorical_columns = [c for c in nominal_list if c in X.columns]
            self._logger.info("Feature-Dictionary: %d INPUT-Features, %d NOMINAL.", len(available_features), len(categorical_columns))
        else:
            # Kein Feature-Dictionary → alle Spalten außer Target/Treatment/Score als Features
            exclude_cols = {target_col, treat_col}
            if score_col:
                exclude_cols.add(score_col)
            # Auch typische ID-/Index-Spalten ausschließen
            available_features = [c for c in df.columns if c not in exclude_cols]
            X = df[available_features].copy()
            categorical_columns = []
            self._logger.info("Kein Feature-Dictionary: %d Features (alle außer Target/Treatment).", len(available_features))

        if dp.score_as_feature and score_col and score_col in df.columns:
            X[score_col] = df[score_col]

        # object- und category-Spalten als kategorisch behandeln
        for col in X.columns:
            if X[col].dtype in ("object", "category") and col not in categorical_columns:
                categorical_columns.append(col)

        # Y, T, optional S
        Y = df[target_col].to_numpy().ravel()
        T = df[treat_col].to_numpy().ravel()
        S = None
        if score_col and score_col in df.columns:
            S = df[score_col].to_numpy().ravel()

        if dp.binary_target:
            Y = (Y > 0).astype(int)
            if dp.log_to_mlflow:
                import mlflow

                mlflow.log_param("binary_target_conversion", "Target converted to binary (0/1)")

        _progress("Preprocessing (Encoding, NaN)")
        preproc = fit_preprocessor(X, categorical_columns, fill_na_method=dp.fill_na_method)
        Xp = preproc.transform(X)

        # Artefakte persistieren
        with open(out_dir / "encoding.obj", "wb") as f:
            pickle.dump(preproc.encoding_maps, f)
        with open(out_dir / "missing_values.json", "w", encoding="utf-8") as f:
            json.dump(preproc.fillna_values, f, indent=2, ensure_ascii=False)

        _progress("Memory-Reduktion")
        Xp = reduce_mem_usage(Xp)
        _progress("Artefakte speichern")
        # Nach der Speicherreduktion muss auch das serialisierte Preprocessing den finalen
        # Zielzustand kennen. Sonst würden spätere Transforms wieder auf die alten Typen casten.
        preproc.dtypes_after = Xp.dtypes.apply(lambda x: x.name).to_dict()
        try:
            Xp.to_parquet(out_dir / "X.parquet")
        except ImportError as e:
            raise ImportError(
                "X.parquet konnte nicht geschrieben werden, weil kein Parquet-Engine verfügbar ist. "
                "Bitte 'pyarrow' oder 'fastparquet' installieren."
            ) from e

        dtypes_dict = dict(preproc.dtypes_after)
        with open(out_dir / "dtypes.json", "w", encoding="utf-8") as f:
            json.dump(dtypes_dict, f, indent=2, ensure_ascii=False)

        try:
            pd.DataFrame({"T": T}, index=Xp.index).to_parquet(out_dir / "T.parquet")
            pd.DataFrame({"Y": Y}, index=Xp.index).to_parquet(out_dir / "Y.parquet")
        except ImportError as e:
            raise ImportError(
                "T.parquet/Y.parquet konnten nicht geschrieben werden, weil kein Parquet-Engine verfügbar ist. "
                "Bitte 'pyarrow' oder 'fastparquet' installieren."
            ) from e
        if S is not None:
            try:
                pd.DataFrame({"S": S}, index=Xp.index).to_parquet(out_dir / "S.parquet")
            except ImportError as e:
                raise ImportError(
                    "S.parquet konnte nicht geschrieben werden, weil kein Parquet-Engine verfügbar ist. "
                    "Bitte 'pyarrow' oder 'fastparquet' installieren."
                ) from e

        with open(out_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump(preproc, f)

        # ── Eval-Daten: fit-on-train, transform-on-eval ──
        if has_eval:
            _progress("Eval-Daten transformieren")
            import logging as _log
            _logger = _log.getLogger("rubin.dataprep")
            try:
                df_eval = self._read_files(file_paths=dp.eval_data_path, merge_only=True)
                if dp.deduplicate and dp.deduplicate_id_column:
                    id_col = str(dp.deduplicate_id_column).upper()
                    if id_col in df_eval.columns:
                        df_eval = df_eval.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)

                # Gleiche Feature-Extraktion wie Train (gleiche feature_dictionary)
                X_eval = df_eval[[c for c in available_features if c in df_eval.columns]].copy()
                if dp.score_as_feature and score_col and score_col in df_eval.columns:
                    X_eval[score_col] = df_eval[score_col]
                # Fehlende Spalten auffüllen (preproc.transform kümmert sich)
                Y_eval = df_eval[target_col].to_numpy().ravel()
                T_eval = df_eval[treat_col].to_numpy().ravel()
                S_eval = None
                if score_col and score_col in df_eval.columns:
                    S_eval = df_eval[score_col].to_numpy().ravel()

                if dp.binary_target:
                    Y_eval = (Y_eval > 0).astype(int)

                # Transform mit dem auf Train gefitteten Preprocessor
                Xp_eval = preproc.transform(X_eval)
                Xp_eval = reduce_mem_usage(Xp_eval)

                # Eval-Artefakte speichern
                Xp_eval.to_parquet(out_dir / "X_eval.parquet")
                pd.DataFrame({"T": T_eval}, index=Xp_eval.index).to_parquet(out_dir / "T_eval.parquet")
                pd.DataFrame({"Y": Y_eval}, index=Xp_eval.index).to_parquet(out_dir / "Y_eval.parquet")
                if S_eval is not None:
                    pd.DataFrame({"S": S_eval}, index=Xp_eval.index).to_parquet(out_dir / "S_eval.parquet")

                _logger.info(
                    "Eval-Daten transformiert: %d Zeilen, %d Features (Train-Preprocessor angewendet).",
                    len(Xp_eval), Xp_eval.shape[1],
                )
            except Exception as e:
                _logger.error("Eval-Daten-Transformation fehlgeschlagen: %s", e, exc_info=True)
                raise

        # Schema zusätzlich separat ablegen, damit Datenänderungen schon vor dem Laden des Pickles
        # sichtbar gemacht werden können.
        try:
            from rubin.utils.schema_utils import save_schema, Schema
            save_schema(Schema.from_dataframe(Xp, categorical_columns=categorical_columns), str(out_dir / "schema.json"))
        except Exception:
            pass

        # Optional: MLflow-Logging der Artefakte
        if dp.log_to_mlflow:
            import mlflow

            artifact_files = [
                "encoding.obj",
                "missing_values.json",
                "X.parquet",
                "dtypes.json",
                "T.parquet",
                "Y.parquet",
                "preprocessor.pkl",
                "schema.json",
            ]
            if S is not None:
                artifact_files.append("S.parquet")
            if has_eval:
                artifact_files.extend(["X_eval.parquet", "T_eval.parquet", "Y_eval.parquet"])
                if (out_dir / "S_eval.parquet").exists():
                    artifact_files.append("S_eval.parquet")

            for fn in artifact_files:
                path = out_dir / fn
                if path.exists():
                    mlflow.log_artifact(str(path))

        print(f"[rubin] Step {total}/{total}: Fertig", flush=True)
        return DataPrepOutputs(X=Xp, T=T, Y=Y, S=S, preprocessor=preproc)
