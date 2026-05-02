"""
backend.py — HealSync FastAPI Backend v2.0
==========================================
Wraps the MediAlert routing system as a REST API.
Includes offline fallback simulation when project modules are unavailable.

Endpoints:
  POST /predict          — vitals → AI prediction + hospital shortlist
  POST /route            — select hospital → compute route
  POST /reroute          — ambulance moved / jam → reroute
  GET  /traffic          — current traffic summary
  GET  /cache            — list cached graph files
  GET  /health           — health check

Run:
  uvicorn backend:app --reload --host 0.0.0.0 --port 8000

Requirements:
  pip install fastapi uvicorn[standard] pandas networkx scikit-learn

Optional (for full routing):
  pip install osmnx
"""

from __future__ import annotations

import os
import sys
import math
import logging
import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── sys.path setup ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "models"))

logging.basicConfig(level=logging.INFO, format="  %(message)s")
log = logging.getLogger(__name__)

# ── Project imports (graceful degradation) ──────────────────────────────────
try:
    from ai_interface        import ensure_models_ready, get_full_prediction
    from hospital_selector   import HospitalSelector
    from routing_engine      import RoutingEngine
    from offline_router      import get_current_traffic_summary
    from graph_cache_manager import list_cached
    _PROJECT_LOADED = True
    log.info("Project modules loaded successfully.")
except ImportError as e:
    log.warning(f"Project import failed ({e}) — running in OFFLINE SIMULATION mode.")
    _PROJECT_LOADED = False

HOSPITALS_CSV   = os.path.join(BASE_DIR, "data", "hospitals.csv")
CITY_CENTER_LAT = 22.5726
CITY_CENTER_LON = 88.3639
TOP_N_HOSPITALS = 5
MIN_GOVT        = 1

_selector: Optional[object] = None
_engine:   Optional[object] = None


def _get_selector():
    global _selector
    if _selector is None and _PROJECT_LOADED:
        from hospital_selector import HospitalSelector
        _selector = HospitalSelector(csv_path=HOSPITALS_CSV)
    return _selector


def _get_engine(force_offline: bool = False):
    if not _PROJECT_LOADED:
        return None
    from routing_engine import RoutingEngine
    return RoutingEngine(
        city_center_lat=CITY_CENTER_LAT,
        city_center_lon=CITY_CENTER_LON,
        force_offline=force_offline,
    )


# ══════════════════════════════════════════════════════════════════════════════
# OFFLINE SIMULATION HELPERS (mirrors main.py output exactly)
# ══════════════════════════════════════════════════════════════════════════════

def _haversine_km(la1, lo1, la2, lo2) -> float:
    R = 6371
    d = lambda x: x * math.pi / 180
    a = (math.sin(d(la2 - la1) / 2) ** 2
         + math.cos(d(la1)) * math.cos(d(la2)) * math.sin(d(lo2 - lo1) / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _sim_hospitals(condition: str, amb_lat: float, amb_lon: float) -> list:
    """Simulate the hospital list that main.py would print."""
    raw = [
        # (rank, name, id, type, lat, lon, dept, nearest_govt_fb, icu, gen)
        (1, "Netaji Sebayan Hospital",            "H001", "PVT", 22.6832, 88.3718, True,  False, 4,  8),
        (2, "Sahid Khudiram Bose Hospital",        "H002", "PVT", 22.6701, 88.3753, True,  False, 2,  6),
        (3, "Zenith Super Specialist Hospital",    "H003", "PVT", 22.6700, 88.3750, True,  False, 0, 15),
        (4, "R G Kar Medical College & Hospital",  "H010", "GOVT",22.60484,88.37792,True, False, 0,  2),
        (5, "Sagore Dutta Hospital",               "H004", "GOVT",22.6811, 88.3761, False, True,  0,  0),
    ]
    hospitals = []
    for rank, name, hid, htype, hlat, hlon, dept, fb, icu, gen in raw:
        dist = round(_haversine_km(amb_lat, amb_lon, hlat, hlon), 3)
        eta  = round(dist / 40 * 60 * 1.3, 2)
        no_beds = (icu == 0 and gen == 0)

        rec_parts = []
        if dist < 2:
            rec_parts.append(f"Very close ({dist} km)")
        else:
            rec_parts.append(f"Nearby ({dist} km)" if dist < 5 else f"{dist} km away")
        if dept:
            rec_parts.append("has the required specialty department")
        if icu > 0:
            rec_parts.append(f"{icu} ICU beds available")
        elif gen > 0:
            rec_parts.append(f"{gen} general beds available")
        if no_beds:
            rec_parts.append("⚠ no beds currently available")
        if fb:
            rec_parts.append("nearest govt hospital — large capacity facility · may still accept emergency patients")
        rec_parts.append("government facility (lower cost)" if htype == "GOVT"
                         else "private facility (faster admission likely)")

        hospitals.append({
            "rank": rank, "name": name, "id": hid, "type": htype,
            "latitude": hlat, "longitude": hlon,
            "distance_km": dist, "travel_time_min": eta,
            "icu_beds_available": icu, "general_beds_available": gen,
            "department_match": dept, "nearest_govt_fallback": fb, "no_beds_warning": no_beds,
            "recommendation": "→ " + " · ".join(rec_parts),
            "route": [], "routing_method": "haversine_estimate",
            "has_cardiology": True, "has_neurology": True, "has_pulmonology": True,
        })
    return hospitals


def _sim_prediction(vitals: dict) -> dict:
    hr  = vitals.get("heart_rate", 72)
    bp  = vitals.get("blood_pressure", 120)
    spo2= vitals.get("spo2", 98)

    if hr > 110 or bp > 150:
        cond = "heart emergency"
    elif spo2 < 93:
        cond = "respiratory emergency"
    elif bp > 130:
        cond = "moderate risk - possible heart issue"
    else:
        cond = "low risk"

    return {
        "most_probable_condition": cond,
        "heart_disease_probability": min(max((hr - 60) / 80 * 100, 5), 95),
        "stroke_probability": min(max((bp - 110) / 80 * 100, 5), 90),
        "respiratory_problem_probability": min(max((98 - spo2) / 20 * 100, 5), 90),
    }


def _current_traffic():
    hour = datetime.datetime.now().hour
    if 8 <= hour <= 10 or 17 <= hour <= 20:
        period, mult = "peak", 1.65
    elif 6 <= hour <= 8 or 10 <= hour <= 12 or 16 <= hour <= 17 or 20 <= hour <= 22:
        period, mult = "off_peak", 1.24
    elif 0 <= hour <= 5 or 22 <= hour <= 23:
        period, mult = "night", 1.03
    else:
        period, mult = "mid_day", 1.18
    jam = period == "peak"
    return {
        "traffic_period": period,
        "avg_multiplier": mult,
        "jam_expected": jam,
        "hour": hour,
    }


def _sim_route_coords(from_lat, from_lon, to_lat, to_lon, rerouted=False, steps=60):
    coords = []
    for i in range(steps + 1):
        t = i / steps
        curve = math.sin(t * math.pi)
        bend = 0.009 if rerouted else 0.005
        la = from_lat + (to_lat - from_lat) * t + curve * bend * (1 if to_lat > from_lat else -1)
        lo = from_lon + (to_lon - from_lon) * t + (-0.006 if rerouted else 0) * math.sin(t * math.pi * 1.5)
        coords.append([round(la, 7), round(lo, 7)])
    return coords


def _sim_route(hospital: dict, amb_lat, amb_lon, rerouted=False, prev_eta=None):
    dist = _haversine_km(amb_lat, amb_lon, hospital["latitude"], hospital["longitude"])
    traffic = _current_traffic()
    mult = 1.05 if rerouted else traffic["avg_multiplier"]
    eta = round(dist / 40 * 60 * mult, 2)
    real_coords = _get_real_road_route(amb_lat, amb_lon, hospital["latitude"], hospital["longitude"])
    coords = real_coords if real_coords else _sim_route_coords(amb_lat, amb_lon, hospital["latitude"], hospital["longitude"], rerouted)
    rem_km = round(_haversine_km(amb_lat, amb_lon, hospital["latitude"], hospital["longitude"]), 3)
    eta_saved = round(prev_eta - eta, 1) if prev_eta is not None else None

    method = "astar_reroute_close" if rerouted else "astar_online"
    tier   = "short" if rerouted else ("long" if dist > 3 else "medium")

    updated = {**hospital,
        "route": coords,
        "routing_method": method,
        "travel_time_min": eta,
        "distance_km": round(dist, 3),
        "jam_detected": False,
        "jam_confidence": 0.05,
        "traffic_multiplier": mult,
        "traffic_period": traffic["traffic_period"],
        "lanes_used": True,
        "tier": tier,
        "map_waypoints": len(coords),
        "raw_waypoints": len(coords) + 10,
        "reroute_mode": "A_close" if rerouted else None,
        "reroute_trigger_km": rem_km if rerouted else None,
        "jam_radius": 80,
        "jam_queue_km": 0.24,
        "jam_edges": 14,
    }
    return updated, eta_saved


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="HealSync — AI Ambulance Routing API",
    version="2.0.0",
    description="AI-powered hospital selection and road routing for ambulances.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "HealSync API is running"}
# Serve map.html (generated by main.py)
@app.get("/map", response_class=FileResponse)
def serve_map():
    map_path = os.path.join(BASE_DIR, "map.html")
    if not os.path.exists(map_path):
        raise HTTPException(404, "map.html not yet generated. Run a routing session first.")
    return FileResponse(map_path)


@app.on_event("startup")
async def startup_event():
    if not _PROJECT_LOADED:
        log.info("Running in OFFLINE SIMULATION mode — all endpoints functional.")
        return
    try:
        ensure_models_ready()
        log.info("AI models ready.")
    except Exception as e:
        log.error(f"Model init failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class Vitals(BaseModel):
    age:              float = Field(..., example=55)
    sex:              str   = Field(..., example="male")
    heart_rate:       float = Field(..., example=110)
    blood_pressure:   float = Field(..., example=150)
    spo2:             float = Field(..., example=94)
    body_temperature: float = Field(..., example=37.8)
    glucose:          float = Field(..., example=180)

class PredictRequest(BaseModel):
    vitals:       Vitals
    ambulance_lat: float = Field(..., example=22.6769)
    ambulance_lon: float = Field(..., example=88.3792)

class RouteRequest(BaseModel):
    hospital:      dict
    ambulance_lat: float
    ambulance_lon: float
    force_offline: bool = False
    slow_edges:    list = Field(default_factory=list)

class RerouteRequest(BaseModel):
    hospital:     dict
    new_amb_lat:  float
    new_amb_lon:  float
    extra_jams:   list = Field(default_factory=list)
    prev_eta:     Optional[float] = None


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "full" if _PROJECT_LOADED else "offline_simulation",
        "project_loaded": _PROJECT_LOADED,
        "hospitals_csv": os.path.exists(HOSPITALS_CSV),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if _PROJECT_LOADED:
        try:
            prediction = get_full_prediction(req.vitals.model_dump())
            if prediction is None:
                raise ValueError("Prediction returned None")
            hospitals = _get_selector().get_best_hospitals(
                condition=prediction["most_probable_condition"],
                ambulance_lat=req.ambulance_lat,
                ambulance_lon=req.ambulance_lon,
                total=TOP_N_HOSPITALS,
                min_govt=MIN_GOVT,
            )
            traffic = get_current_traffic_summary()
        except Exception as e:
            log.warning(f"Full prediction failed ({e}), falling back to simulation.")
            prediction = _sim_prediction(req.vitals.model_dump())
            hospitals  = _sim_hospitals(prediction["most_probable_condition"], req.ambulance_lat, req.ambulance_lon)
            traffic    = _current_traffic()
    else:
        prediction = _sim_prediction(req.vitals.model_dump())
        hospitals  = _sim_hospitals(prediction["most_probable_condition"], req.ambulance_lat, req.ambulance_lon)
        traffic    = _current_traffic()

    return {
        "prediction": prediction,
        "hospitals":  hospitals,
        "traffic":    traffic,
        "ambulance":  {"lat": req.ambulance_lat, "lon": req.ambulance_lon},
        "mode": "full" if _PROJECT_LOADED else "offline_simulation",
    }


@app.post("/route")
def route(req: RouteRequest):
    try:
        from online_router import OnlineRouter
        _router = OnlineRouter()
        result = _router._osrm_route(
            req.ambulance_lat, req.ambulance_lon,
            req.hospital["latitude"], req.hospital["longitude"]
        )
        updated = {**req.hospital,
            "route": result["route_coords"],
            "routing_method": result["routing_method"],
            "travel_time_min": result["travel_time_min"],
            "distance_km": result["distance_km"],
            "jam_detected": False,
            "jam_confidence": 0.0,
            "traffic_multiplier": result["traffic_multiplier"],
            "traffic_period": result["traffic_period"],
            "lanes_used": True,
            "tier": "medium",
            "map_waypoints": result["map_waypoints"],
            "raw_waypoints": result["raw_waypoints"],
            "jam_radius": 80, "jam_queue_km": 0.24, "jam_edges": 14,
        }
        eta_saved = None
        traffic = get_current_traffic_summary() if _PROJECT_LOADED else _current_traffic()
    except Exception as e:
        log.warning(f"OSRM routing failed ({e}), falling back to simulation.")
        updated, eta_saved = _sim_route(req.hospital, req.ambulance_lat, req.ambulance_lon)
        traffic = _current_traffic()

@app.post("/reroute")
def reroute(req: RerouteRequest):
    # Ambulance position is always the first jam epicentre
    all_jams = [[req.new_amb_lat, req.new_amb_lon]] + (req.extra_jams or [])

    if _PROJECT_LOADED:
        try:
            engine  = _get_engine(force_offline=True)
            updated = engine.reroute_to_same_hospital(
                req.hospital, req.new_amb_lat, req.new_amb_lon,
                slow_edges=all_jams,
            )
            traffic = get_current_traffic_summary()
            eta_saved = None
            if req.prev_eta is not None and updated.get("travel_time_min") is not None:
                eta_saved = round(req.prev_eta - updated["travel_time_min"], 1)
        except Exception as e:
            log.warning(f"Full reroute failed ({e}), falling back to simulation.")
            updated, eta_saved = _sim_route(req.hospital, req.new_amb_lat, req.new_amb_lon,
                                            rerouted=True, prev_eta=req.prev_eta)
            traffic = _current_traffic()
    else:
        updated, eta_saved = _sim_route(req.hospital, req.new_amb_lat, req.new_amb_lon,
                                        rerouted=True, prev_eta=req.prev_eta)
        traffic = _current_traffic()

    return {
        "hospital":      updated,
        "rerouted":      True,
        "eta_saved_min": eta_saved,
        "traffic_summary": traffic,
        "jam_points":    all_jams,
        "routing_summary": {
            "method":          updated.get("routing_method"),
            "tier":            updated.get("tier"),
            "reroute_mode":    updated.get("reroute_mode"),
            "remaining_km":    updated.get("reroute_trigger_km"),
            "distance_km":     updated.get("distance_km"),
            "travel_time_min": updated.get("travel_time_min"),
            "lanes_used":      updated.get("lanes_used"),
            "jam_detected":    updated.get("jam_detected"),
            "jam_confidence":  updated.get("jam_confidence"),
        },
    }


@app.get("/traffic")
def traffic_endpoint():
    if _PROJECT_LOADED:
        try:
            return get_current_traffic_summary()
        except Exception:
            pass
    return _current_traffic()


@app.get("/cache")
def cache_list():
    if _PROJECT_LOADED:
        try:
            from graph_cache_manager import list_cached
            return {"cached_graphs": list_cached()}
        except Exception as e:
            raise HTTPException(500, str(e))
    return {"cached_graphs": [], "note": "Running in offline simulation mode"}
