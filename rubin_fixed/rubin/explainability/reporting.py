from __future__ import annotations

"""Reporting-Helfer für Explainability.
Dieses Modul kümmert sich bewusst nur um:
* Speichern von Tabellen (CSV/JSON)
* einfache, robuste Visualisierungen (matplotlib)
Warum nicht "zu fancy"?
----------------------
Explainability ist häufig ein Begleitprozess in einer Analyse. Die wichtigsten
Artefakte sollen daher in möglichst vielen Umgebungen funktionieren (auch ohne
Notebook und ohne GUI). Deshalb werden einfache Plot-Typen genutzt."""

from pathlib import Path
from typing import Optional

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rubin.utils.plot_theme import apply_rubin_theme, COLOR_MODEL, RUBIN_COLORS
apply_rubin_theme()


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict, path: str) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_importance_barplot(
    importance: pd.Series,
    out_path: str,
    top_n: int = 30,
    title: Optional[str] = None,
) -> None:
    """Speichert ein horizontales Balkendiagramm für Feature-Importances."""
    imp = importance.head(top_n)[::-1]  # für horizontale Plots: klein -> groß
    fig = plt.figure(figsize=(10, max(4, 0.25 * len(imp) + 2)))
    ax = fig.add_subplot(111)
    ax.barh(imp.index.astype(str), imp.values, color=COLOR_MODEL, edgecolor=RUBIN_COLORS["ruby_dark"], linewidth=0.5)
    ax.set_xlabel("Wichtigkeit")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_shap_summary_csv(
    shap_values: np.ndarray,
    feature_names: list[str],
    out_path: str,
) -> None:
    """Speichert SHAP-Werte als CSV (wide).
Hinweis:
SHAP-Werte können groß werden. Für sehr große Datensätze empfiehlt es sich,
nur eine Stichprobe zu erklären."""
    df = pd.DataFrame(shap_values, columns=feature_names)
    df.to_csv(out_path, index=False)
