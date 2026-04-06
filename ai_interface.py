"""
ai_interface.py — AI Model Interface
======================================
Wraps aimodel.py so the rest of the project
never imports aimodel.py directly.

This is the ONLY file that touches aimodel.py.
If aimodel.py changes, only this file needs updating.

Functions:
  get_condition(vitals_dict) → condition string
  get_full_prediction(vitals_dict) → full dict with all probabilities
"""

import sys
import os
import logging

log = logging.getLogger(__name__)

# Add models/ folder to Python path so aimodel.py can be imported
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, MODELS_DIR)

try:
    from aimodel import predict_emergency, models_exist, train_all_models # type: ignore
    _AIMODEL_LOADED = True
except ImportError as e:
    log.warning(f"Could not import aimodel.py: {e}")
    _AIMODEL_LOADED = False


# ─────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────

def ensure_models_ready():
    """
    Train models if .pkl files don't exist yet.
    Called once at startup by main.py.
    """
    if not _AIMODEL_LOADED:
        raise RuntimeError("aimodel.py not found in models/ folder.")

    if not models_exist():
        log.info("Models not trained yet — training now (first time only)...")
        train_all_models()
    else:
        log.info("Models already trained. Loading from disk.")


def get_full_prediction(vitals: dict) -> dict | None:
    """
    Run the AI prediction on patient vitals.

    Parameters
    ----------
    vitals : dict with keys:
        age, sex, heart_rate, blood_pressure,
        spo2, body_temperature, glucose

    Returns
    -------
    dict:
        {
            heart_disease_probability: float,
            stroke_probability: float,
            respiratory_problem_probability: float,
            most_probable_condition: str
        }
    or None if validation fails.
    """
    if not _AIMODEL_LOADED:
        raise RuntimeError("aimodel.py is not loaded.")

    result = predict_emergency(
        age              = vitals["age"],
        sex              = vitals["sex"],
        heart_rate       = vitals["heart_rate"],
        blood_pressure   = vitals["blood_pressure"],
        spo2             = vitals["spo2"],
        body_temperature = vitals["body_temperature"],
        glucose          = vitals["glucose"],
    )
    return result


def get_condition(vitals: dict) -> str | None:
    """
    Convenience function — returns only the condition string.
    Use get_full_prediction() if you need all probabilities.
    """
    result = get_full_prediction(vitals)
    if result is None:
        return None
    return result["most_probable_condition"]