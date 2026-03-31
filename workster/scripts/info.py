"""Zeigt Framework-Informationen."""
from rubin.settings import SUPPORTED_MODEL_NAMES

sep = ", "
names = sep.join(sorted(SUPPORTED_MODEL_NAMES))
print("rubin – Causal ML Framework")
print(f"  Modelle:       {len(SUPPORTED_MODEL_NAMES)} ({names})")
print("  Base Learner:  LightGBM, CatBoost")
print("  Pipelines:     Analysis, Production, DataPrep, Explain")
