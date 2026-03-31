from __future__ import annotations

"""Uplift-Metriken für Binary Treatment / Binary Outcome und Multi Treatment / Binary Outcome.
Diese Metriken beantworten eine andere Frage als klassische ML-Metriken:
- Klassifikation (ROC-AUC, LogLoss) bewertet: "Kann ich Y vorhersagen?"
- Uplift/Causal bewertet: "Kann ich entscheiden, *wen* ich behandeln sollte, um inkrementellen Nutzen zu erzeugen?"
Die zentrale Idee:
- Sortiere Beobachtungen nach geschätztem Uplift (CATE) absteigend.
- Betrachte kumulativ, wie stark sich Outcome in Treatment vs Control unterscheidet.
- Daraus lassen sich Kurven und Kennzahlen wie Qini-Koeffizient und AUUC ableiten.
Voraussetzungen:
- binary outcome y in {0,1}
- binary treatment t in {0,1} ODER multi treatment t in {0, 1, ..., K-1}
Hinweis:
- Es gibt mehrere Definitionen in der Literatur. Die hier implementierten Varianten sind pragmatisch und
in Uplift-Modelling gängig."""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np

# Kompatibilität: np.trapz wurde in numpy 2.0 zu np.trapezoid umbenannt.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


@dataclass(frozen=True)
class UpliftCurve:
    """Kumulierte Größen entlang einer Sortierung nach score."""
    fraction: np.ndarray           # Anteil der Population (0..1)
    n_treat: np.ndarray            # kumulierte Anzahl T=1 (bzw. T=k bei MT)
    n_control: np.ndarray          # kumulierte Anzahl T=0
    y_treat: np.ndarray            # kumulierte Summe Y unter T=1 (bzw. T=k)
    y_control: np.ndarray          # kumulierte Summe Y unter T=0
    uplift: np.ndarray             # kumulierter inkrementeller Outcome (Treatment-Control, skaliert)


def _check_binary(a: np.ndarray, name: str) -> None:
    u = np.unique(a)
    if not set(u).issubset({0, 1}):
        raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")


def _check_discrete(a: np.ndarray, name: str) -> None:
    """Prüft, ob ein Array diskrete nicht-negative ganzzahlige Werte enthält."""
    u = np.unique(a)
    if not np.all(u == u.astype(int)):
        raise ValueError(f"{name} muss ganzzahlig sein, gefunden: {u}")
    if np.any(u < 0):
        raise ValueError(f"{name} darf keine negativen Werte enthalten, gefunden: {u}")


def uplift_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> UpliftCurve:
    """Berechnet die Uplift-Kurve für Binary Treatment / Binary Outcome.
    Für Multi-Treatment siehe uplift_curve_mt_per_arm."""
    y = np.asarray(y).astype(int)
    t = np.asarray(t).astype(int)
    score = np.asarray(score).astype(float)
    _check_binary(y, "y")
    _check_binary(t, "t")

    n = len(y)
    order = np.argsort(-score)  # absteigend
    y = y[order]
    t = t[order]

    treat = (t == 1).astype(int)
    control = (t == 0).astype(int)

    n_t = np.cumsum(treat)
    n_c = np.cumsum(control)
    y_t = np.cumsum(y * treat)
    y_c = np.cumsum(y * control)

    # inkrementeller Outcome: (y_t / n_t) - (y_c / n_c), auf Gesamtpopulation skaliert
    rate_t = y_t / np.maximum(n_t, 1)
    rate_c = y_c / np.maximum(n_c, 1)
    inc_rate = rate_t - rate_c
    uplift = inc_rate * (np.arange(1, n + 1) / n)

    frac = (np.arange(1, n + 1) / n)

    return UpliftCurve(
        fraction=frac,
        n_treat=n_t,
        n_control=n_c,
        y_treat=y_t,
        y_control=y_c,
        uplift=uplift,
    )


def auuc(curve: UpliftCurve) -> float:
    """Fläche unter der Uplift-Kurve (trapezförmige Numerik)."""
    x = curve.fraction
    y = curve.uplift
    return float(_trapz(y, x))


def qini_coefficient(curve: UpliftCurve) -> float:
    """Qini-Koeffizient als Flächenmass über einer Basislinie.
Basislinie:
- Eine "zufällige" Behandlung hätte (unter idealer Randomisierung) eine annähernd lineare Kurve von 0 bis
zum finalen inkrementellen Outcome.
Wir messen daher die Fläche zwischen Uplift-Kurve und dieser linearen Referenz."""
    x = curve.fraction
    y = curve.uplift

    final = y[-1] if len(y) else 0.0
    baseline = x * final
    return float(_trapz(y - baseline, x))


def uplift_at_k(curve: UpliftCurve, k_fraction: float = 0.1) -> float:
    """Inkrementeller Outcome, wenn man die Top-k% nach score behandelt."""
    k_fraction = float(k_fraction)
    if k_fraction <= 0:
        return 0.0
    idx = int(np.clip(np.ceil(k_fraction * len(curve.fraction)) - 1, 0, len(curve.fraction) - 1))
    return float(curve.uplift[idx])


def policy_value(y: np.ndarray, t: np.ndarray, score: np.ndarray, threshold: float = 0.0) -> float:
    """Einfache Policy-Value-Kennzahl für Binary Treatment.
Policy:
- Behandle genau dann, wenn score >= threshold.
Wert:
- geschätzter inkrementeller Outcome dieser Policy."""
    y = np.asarray(y).astype(int)
    t = np.asarray(t).astype(int)
    score = np.asarray(score).astype(float)
    _check_binary(y, "y")
    _check_binary(t, "t")

    mask = score >= float(threshold)
    if mask.sum() == 0:
        return 0.0

    y_m = y[mask]
    t_m = t[mask]

    n_t = (t_m == 1).sum()
    n_c = (t_m == 0).sum()
    if n_t == 0 or n_c == 0:
        return 0.0

    return float(y_m[t_m == 1].mean() - y_m[t_m == 0].mean())


# ---------------------------------------------------------------------------
# Multi-Treatment Metriken
# ---------------------------------------------------------------------------

def uplift_curve_mt_per_arm(
    y: np.ndarray,
    t: np.ndarray,
    scores: np.ndarray,
    treatment_arm: int,
) -> UpliftCurve:
    """Uplift-Kurve für einen einzelnen Treatment-Arm vs. Control (T=0).
    Filtert auf Beobachtungen mit T in {0, treatment_arm} und berechnet
    die Standard-Uplift-Kurve auf diesem Subset.

    Parameters
    ----------
    y : Outcome-Vektor (binär)
    t : Treatment-Vektor (diskret, 0 = Control)
    scores : CATE-Schätzungen für diesen Treatment-Arm, Shape (n,)
    treatment_arm : welcher Arm (z. B. 1, 2, ...)
    """
    y = np.asarray(y).astype(int)
    t = np.asarray(t).astype(int)
    scores = np.asarray(scores).astype(float)
    _check_binary(y, "y")
    _check_discrete(t, "t")

    mask = (t == 0) | (t == treatment_arm)
    y_sub = y[mask]
    t_sub = (t[mask] == treatment_arm).astype(int)
    s_sub = scores[mask]

    n = len(y_sub)
    if n == 0:
        return UpliftCurve(
            fraction=np.array([]),
            n_treat=np.array([]),
            n_control=np.array([]),
            y_treat=np.array([]),
            y_control=np.array([]),
            uplift=np.array([]),
        )

    order = np.argsort(-s_sub)
    y_sub = y_sub[order]
    t_sub = t_sub[order]

    treat = t_sub.astype(int)
    control = (1 - t_sub).astype(int)

    n_t = np.cumsum(treat)
    n_c = np.cumsum(control)
    y_t = np.cumsum(y_sub * treat)
    y_c = np.cumsum(y_sub * control)

    rate_t = y_t / np.maximum(n_t, 1)
    rate_c = y_c / np.maximum(n_c, 1)
    inc_rate = rate_t - rate_c
    frac = np.arange(1, n + 1) / n
    uplift_vals = inc_rate * frac

    return UpliftCurve(
        fraction=frac,
        n_treat=n_t,
        n_control=n_c,
        y_treat=y_t,
        y_control=y_c,
        uplift=uplift_vals,
    )


def policy_value_mt(
    y: np.ndarray,
    t: np.ndarray,
    scores_2d: np.ndarray,
    propensity: Optional[np.ndarray] = None,
) -> float:
    """Policy-Value-Schätzer für Multi-Treatment via IPW.

    Die optimale Policy ist pi*(X) = argmax_k tau_k(X) (k=1..K-1).
    Wenn max_k tau_k(X) <= 0, wird "nicht behandeln" (T=0) empfohlen.

    Parameters
    ----------
    y : Outcome (binär)
    t : Beobachtete Treatment-Zuweisung (0..K-1)
    scores_2d : CATE-Schätzungen, Shape (n, K-1)
    propensity : Optional (n, K) Array mit P(T=k|X). Falls None, wird
                 die empirische Verteilung genutzt (Randomisierung angenommen).

    Returns
    -------
    Geschätzter Policy-Value (V_IPW(pi*) - Baseline-Rate)

    Hinweis
    -------
    Die Default-Propensity (empirische Verteilung) ist nur unter der Annahme
    einer Randomisierung (z. B. aus einem A/B-Test) unverzerrt. Bei
    observationalen Daten sollte eine geschätzte Propensity (z. B. aus einem
    Klassifikator) über den Parameter ``propensity`` übergeben werden, um
    Confounding-Bias zu reduzieren.
    """
    y = np.asarray(y).astype(float)
    t = np.asarray(t).astype(int)
    scores_2d = np.asarray(scores_2d).astype(float)
    _check_discrete(t, "t")

    n = len(y)
    K = int(t.max()) + 1  # Anzahl Treatment-Gruppen inkl. Control
    n_effects = K - 1

    if scores_2d.ndim == 1:
        scores_2d = scores_2d.reshape(-1, 1)

    # Optimale Zuweisung: argmax über K-1 Arme, aber nur wenn > 0
    best_effect = np.max(scores_2d, axis=1)
    best_arm = np.argmax(scores_2d, axis=1) + 1  # 1-basiert
    # Wenn der beste Effekt <= 0 ist, weise Control (0) zu
    optimal_treatment = np.where(best_effect > 0, best_arm, 0)

    # IPW-Schätzer
    if propensity is None:
        # Empirische Propensity (unter Randomisierungsannahme)
        propensity = np.zeros((n, K), dtype=float)
        for k in range(K):
            propensity[:, k] = max((t == k).sum(), 1) / n

    # V_IPW(pi) = (1/n) * sum_i Y_i * 1[T_i == pi(X_i)] / P(T_i | X_i)
    match = (t == optimal_treatment).astype(float)
    prop_obs = propensity[np.arange(n), t]
    prop_obs = np.maximum(prop_obs, 1e-10)

    policy_val = np.mean(y * match / prop_obs)
    # Baseline: Rate unter Control
    control_mask = t == 0
    baseline = y[control_mask].mean() if control_mask.sum() > 0 else 0.0

    return float(policy_val - baseline)


def policy_value_per_arm(
    y: np.ndarray,
    t: np.ndarray,
    scores: np.ndarray,
    treatment_arm: int,
    threshold: float = 0.0,
) -> float:
    """Policy-Value-Kennzahl für einen einzelnen Treatment-Arm vs. Control.

    Analogon zur Binary-Treatment-Funktion ``policy_value()``, aber für
    Multi-Treatment: Es wird auf Beobachtungen mit T ∈ {0, treatment_arm}
    gefiltert und dann die Differenz der Outcome-Raten zwischen Treatment
    und Control berechnet – nur für Beobachtungen, deren CATE-Schätzung
    den Schwellenwert übersteigt.

    Parameters
    ----------
    y : Outcome (binär)
    t : Treatment-Vektor (diskret, 0 = Control)
    scores : CATE-Schätzungen für diesen Arm, Shape (n,)
    treatment_arm : welcher Arm (z. B. 1, 2, …)
    threshold : Nur Beobachtungen mit score >= threshold einschließen.

    Returns
    -------
    Geschätzter inkrementeller Outcome (Treat-Rate − Control-Rate)
    unter den Beobachtungen mit score >= threshold.
    """
    y = np.asarray(y).astype(int)
    t = np.asarray(t).astype(int)
    scores = np.asarray(scores).astype(float)
    _check_binary(y, "y")
    _check_discrete(t, "t")

    # Filtere auf Control + gewählten Arm
    arm_mask = (t == 0) | (t == treatment_arm)
    y_sub = y[arm_mask]
    t_sub = (t[arm_mask] == treatment_arm).astype(int)
    s_sub = scores[arm_mask]

    # Threshold-Filter
    above = s_sub >= float(threshold)
    if above.sum() == 0:
        return 0.0

    y_m = y_sub[above]
    t_m = t_sub[above]

    n_t = (t_m == 1).sum()
    n_c = (t_m == 0).sum()
    if n_t == 0 or n_c == 0:
        return 0.0

    return float(y_m[t_m == 1].mean() - y_m[t_m == 0].mean())


def optimal_treatment_assignment(scores_2d: np.ndarray) -> np.ndarray:
    """Bestimmt für jede Beobachtung das optimale Treatment.

    Parameters
    ----------
    scores_2d : CATE-Schätzungen, Shape (n, K-1)

    Returns
    -------
    Array mit optimalen Treatment-Zuweisungen (0 = Control, 1..K-1 = Treatment)
    """
    scores_2d = np.asarray(scores_2d).astype(float)
    if scores_2d.ndim == 1:
        scores_2d = scores_2d.reshape(-1, 1)

    best_effect = np.max(scores_2d, axis=1)
    best_arm = np.argmax(scores_2d, axis=1) + 1
    return np.where(best_effect > 0, best_arm, 0)


def mt_eval_summary(
    y: np.ndarray,
    t: np.ndarray,
    scores_2d: np.ndarray,
    propensity: Optional[np.ndarray] = None,
) -> dict:
    """Erzeugt ein Evaluations-Summary für Multi-Treatment.

    Enthält:
    - Pro Treatment-Arm: qini_T{k}, auuc_T{k}, uplift10_T{k}, uplift20_T{k},
      uplift50_T{k}, policy_value_treat_positive_T{k}
    - Global: policy_value, best_treatment_distribution

    Parameters
    ----------
    y : Outcome (binär)
    t : Treatment (diskret, 0..K-1)
    scores_2d : CATE-Schätzungen, Shape (n, K-1)
    propensity : Optional (n, K) Array mit P(T=k|X). Falls None, wird
                 die empirische Verteilung genutzt (Randomisierung angenommen).
    """
    y = np.asarray(y).astype(int)
    t = np.asarray(t).astype(int)
    scores_2d = np.asarray(scores_2d).astype(float)

    n_effects = scores_2d.shape[1] if scores_2d.ndim == 2 else 1
    if scores_2d.ndim == 1:
        scores_2d = scores_2d.reshape(-1, 1)

    result: dict = {}

    # Per-Arm Metriken (Ansatz A)
    # Key-Namen folgen der BT-Konvention (uplift10/uplift20/uplift50 statt uplift_at_10pct/
    # uplift_at_20pct/uplift_at_50pct), damit MLflow-Metriken konsistent benannt sind.
    for k in range(n_effects):
        arm = k + 1
        try:
            curve = uplift_curve_mt_per_arm(y, t, scores_2d[:, k], treatment_arm=arm)
            if len(curve.fraction) > 0:
                result[f"qini_T{arm}"] = float(qini_coefficient(curve))
                result[f"auuc_T{arm}"] = float(auuc(curve))
                result[f"uplift10_T{arm}"] = float(uplift_at_k(curve, 0.10))
                result[f"uplift20_T{arm}"] = float(uplift_at_k(curve, 0.20))
                result[f"uplift50_T{arm}"] = float(uplift_at_k(curve, 0.50))
            else:
                result[f"qini_T{arm}"] = 0.0
                result[f"auuc_T{arm}"] = 0.0
                result[f"uplift10_T{arm}"] = 0.0
                result[f"uplift20_T{arm}"] = 0.0
                result[f"uplift50_T{arm}"] = 0.0
        except Exception:
            result[f"qini_T{arm}"] = 0.0
            result[f"auuc_T{arm}"] = 0.0
            result[f"uplift10_T{arm}"] = 0.0
            result[f"uplift20_T{arm}"] = 0.0
            result[f"uplift50_T{arm}"] = 0.0

        # Per-Arm Policy Value (Ansatz A.2): Behandle Arm k, wenn CATE_k > 0
        try:
            result[f"policy_value_treat_positive_T{arm}"] = float(
                policy_value_per_arm(y, t, scores_2d[:, k], treatment_arm=arm, threshold=0.0)
            )
        except Exception:
            result[f"policy_value_treat_positive_T{arm}"] = 0.0

    # Globaler Policy Value (Ansatz B): Optimale Zuweisung über alle Arme via IPW
    try:
        result["policy_value"] = policy_value_mt(y, t, scores_2d, propensity=propensity)
    except Exception:
        result["policy_value"] = 0.0

    # Treatment-Verteilung der optimalen Zuweisung
    optimal = optimal_treatment_assignment(scores_2d)
    K = int(t.max()) + 1
    dist = {}
    for k in range(K):
        dist[f"T{k}"] = float((optimal == k).sum() / len(optimal))
    result["best_treatment_distribution"] = dist

    return result
