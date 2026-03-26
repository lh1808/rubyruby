"""Erklärbarkeit (Explainability) für kausale Modelle.
Dieses Paket bündelt Hilfsfunktionen, um Uplift-/CATE-Modelle nachvollziehbarer
zu machen. Es unterstützt sowohl modellagnostische Erklärungen als auch einen
vollständigen SHAP-Plot-Satz."""

from .shap_uplift import (
    compute_shap_for_uplift,
    shap_available,
    build_shap_plots,
)
from .permutation_uplift import compute_permutation_importance_for_uplift
from .segment_analysis import build_segment_report, build_feature_segment_report

__all__ = [
    "compute_shap_for_uplift",
    "shap_available",
    "build_shap_plots",
    "compute_permutation_importance_for_uplift",
    "build_segment_report",
    "build_feature_segment_report",
]
