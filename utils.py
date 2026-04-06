"""
utils.py — Shared Utility Functions v3.0
==========================================
• get_current_traffic_summary() re-exported from offline_router
  so callers don't need to import it directly.
• build_final_output() now includes traffic_summary field.
"""

import math
import socket
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DISTANCE
# ─────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─────────────────────────────────────────────
# CONNECTIVITY DETECTION
# ─────────────────────────────────────────────

def detect_connectivity(host="8.8.8.8", port=53, timeout=2.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except (socket.error, OSError):
        return False


# ─────────────────────────────────────────────
# CONDITION → DEPARTMENT MAPPING
# ─────────────────────────────────────────────

CONDITION_TO_DEPT = {
    "heart emergency":                             "has_cardiology",
    "stroke risk":                                 "has_neurology",
    "respiratory emergency":                       "has_pulmonology",
    "moderate risk - possible heart issue":        "has_cardiology",
    "moderate risk - possible stroke":             "has_neurology",
    "moderate risk - possible respiratory issue":  "has_pulmonology",
    "low risk":      None,
    "moderate risk": None,
}


def get_required_dept(condition: str) -> str | None:
    cond_lower = condition.strip().lower()
    for key, col in CONDITION_TO_DEPT.items():
        if key in cond_lower:
            return col
    return None


# ─────────────────────────────────────────────
# HOSPITAL SCORING
# ─────────────────────────────────────────────

def score_hospital(hospital: dict, distance_km: float, dept_col: str | None) -> float:
    dist_score = distance_km * 0.60
    icu_score  = (1.0 / (hospital.get("icu_beds_available",  1) + 1)) * 0.20
    gen_score  = (1.0 / (hospital.get("general_beds_available", 1) + 1)) * 0.10
    dept_bonus = -0.10 if (dept_col and hospital.get(dept_col, 0) == 1) else 0.0
    return dist_score + icu_score + gen_score + dept_bonus


# ─────────────────────────────────────────────
# TRAVEL TIME ESTIMATE
# ─────────────────────────────────────────────

def estimate_travel_time(distance_km: float, avg_speed_kmh: float = 40.0) -> float:
    return round((distance_km / avg_speed_kmh) * 60, 1)


# ─────────────────────────────────────────────
# JSON OUTPUT BUILDER
# ─────────────────────────────────────────────

def build_final_output(condition: str, recommended_hospitals: list,
                       traffic_summary: dict = None) -> dict:
    out = {
        "condition":             condition,
        "recommended_hospitals": recommended_hospitals,
        "total_hospitals_found": len(recommended_hospitals),
    }
    if traffic_summary:
        out["traffic_summary"] = traffic_summary
    return out


def print_output(output: dict):
    print("\n" + "="*60)
    print("  MEDIALERT — AMBULANCE ROUTING RESULT")
    print("="*60)
    print(f"\n  Predicted Condition : {output['condition']}")
    print(f"  Hospitals Found     : {output['total_hospitals_found']}")
    print("\n  Recommended Hospitals:")
    print("  " + "-"*56)

    for h in output["recommended_hospitals"]:
        govt_tag  = "[GOVT]" if h.get("type") == "GOVT" else "[PVT] "
        dept_tag  = " ✓ Dept Match" if h.get("department_match") else ""
        icu_tag   = f"  ICU: {h['icu_beds_available']} free" if h['icu_beds_available']>0 else "  ICU: Full"
        jam_tag   = f"  ⚠ Jam {h.get('jam_confidence',0)*100:.0f}%" if h.get("jam_detected") else ""
        print(f"\n  {h['rank']}. {govt_tag} {h['name']}")
        print(f"     Distance   : {h['distance_km']} km")
        print(f"     Travel ETA : {h['travel_time_min']} min{jam_tag}")
        print(f"     Beds       :{icu_tag}  |  General: {h['general_beds_available']} free")
        print(f"     Routing    : {h.get('routing_method','straight-line')}{dept_tag}")

    print("\n" + "="*60)


def save_output_json(output: dict, path: str = "output.json"):
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Output saved to {path}")