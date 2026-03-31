from __future__ import annotations

"""Kleine I/O-Helfer für tabellarische Dateien."""

import pandas as pd


def read_table(path: str, *, use_index: bool = True) -> pd.DataFrame:
    """Liest CSV oder Parquet anhand der Dateiendung.

    Bei CSV-Dateien wird automatisch erkannt ob die erste Spalte ein Index ist
    (leerer Header oder 'Unnamed: 0'). Falls ja, wird sie als Index verwendet.
    Falls nein, werden alle Spalten als Daten behandelt.
    """
    p = str(path)
    if p.lower().endswith((".parquet", ".pq")):
        try:
            df = pd.read_parquet(p)
        except ImportError as e:
            raise ImportError(
                "Für Parquet-Dateien wird 'pyarrow' oder 'fastparquet' benötigt. "
                "Bitte die Abhängigkeiten aus requirements.txt installieren."
            ) from e
        return df if use_index else df.reset_index(drop=True)

    # CSV: erst ohne index_col lesen, dann prüfen ob Spalte 0 ein Index ist
    df = pd.read_csv(p, low_memory=False)

    if use_index and len(df.columns) > 0:
        first_col = str(df.columns[0])
        # Typische Index-Spalte: leerer Name oder "Unnamed: 0"
        if first_col == "" or first_col.startswith("Unnamed:"):
            df = df.set_index(df.columns[0])
            df.index.name = None

    return df
