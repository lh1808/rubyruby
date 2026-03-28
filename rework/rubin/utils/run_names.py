"""Zufällige, menschenlesbare Run-Namen für MLflow.

Beispiele: 'Datenaufbereitung – roter-falke', 'Analyse – stiller-wolf'
"""
from __future__ import annotations

import random as _random

_ADJECTIVES = [
    "roter", "blauer", "grüner", "goldener", "stiller",
    "schneller", "leiser", "heller", "dunkler", "kühler",
    "weiser", "scharfer", "wilder", "sanfter", "klarer",
    "hoher", "tiefer", "breiter", "feiner", "starker",
]

_NOUNS = [
    "falke", "wolf", "adler", "fuchs", "bär",
    "hirsch", "luchs", "rabe", "otter", "dachs",
    "sperber", "marder", "habicht", "uhu", "biber",
    "kranich", "eisvogel", "steinbock", "panther", "löwe",
]


def generate_run_name(prefix: str, seed: int | None = None) -> str:
    """Erzeugt einen Run-Namen im Format '<prefix> – <adjektiv>-<nomen>'.

    Parameters
    ----------
    prefix : str
        Vorsatz, z. B. 'Datenaufbereitung' oder 'Analyse'.
    seed : int, optional
        Für Reproduzierbarkeit. Wenn None, wird der aktuelle Zustand genutzt.
    """
    rng = _random.Random(seed) if seed is not None else _random.Random()
    adj = rng.choice(_ADJECTIVES)
    noun = rng.choice(_NOUNS)
    return f"{prefix} – {adj}-{noun}"
