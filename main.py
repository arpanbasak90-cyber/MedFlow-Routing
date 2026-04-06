"""
main.py — MediAlert: AI-Powered Ambulance Routing System v5.2
==============================================================
v5.2 changes
------------
• Ambulance current position is ALWAYS the primary jam epicentre.
• Progressive ETA display: shows ETA improvement vs previous route.
• Jam confidence display updated — now reflects the REROUTED path
  (alley-heavy escape → lower confidence, which is correct).
• _print_route_result() shows ETA delta (↓ X min saved) on reroutes.
• JAM_POPUP_THRESHOLD lowered to 0.35 (was 0.4) to catch more jams.
• On reroute confirmation, shows summary: old ETA → new ETA → saved.
"""

import os
import sys
import json
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "models"))

from ai_interface        import ensure_models_ready, get_full_prediction
from hospital_selector   import HospitalSelector
from routing_engine      import RoutingEngine
from offline_router      import get_current_traffic_summary
from graph_cache_manager import list_cached

logging.basicConfig(level=logging.INFO, format="  %(message)s")
log = logging.getLogger(__name__)

HOSPITALS_CSV       = os.path.join(BASE_DIR, "data", "hospitals.csv")
OUTPUT_JSON         = os.path.join(BASE_DIR, "output.json")
MAP_HTML            = os.path.join(BASE_DIR, "map.html")
CITY_CENTER_LAT     = 22.5726
CITY_CENTER_LON     = 88.3639
TOP_N_HOSPITALS     = 4
MIN_GOVT            = 1
JAM_POPUP_THRESHOLD = 0.35   # v5.2: lowered from 0.4

# Threshold (km) below which two points are treated as the same location
SAME_POSITION_KM = 0.05


# ─────────────────────────────────────────────
# STEP 1–3: PREDICT + SHORTLIST
# ─────────────────────────────────────────────

def predict_and_shortlist(vitals, ambulance_lat, ambulance_lon):
    _banner("MediAlert — AI-Powered Ambulance Routing")

    _step(1, 3, "Checking AI models…")
    ensure_models_ready()

    _step(2, 3, "Running AI prediction…")
    prediction = get_full_prediction(vitals)
    if prediction is None:
        print("\n  ✗ Prediction failed — check patient vitals.")
        return None, []

    condition = prediction["most_probable_condition"]
    print(f"\n  Predicted Condition  : {condition}")
    print(f"  Heart Risk           : {prediction['heart_disease_probability']:.1f}%")
    print(f"  Stroke Risk          : {prediction['stroke_probability']:.1f}%")
    print(f"  Respiratory Risk     : {prediction['respiratory_problem_probability']:.1f}%")

    _step(3, 3, f"Ranking hospitals for: {condition}")
    selector  = HospitalSelector(csv_path=HOSPITALS_CSV)
    hospitals = selector.get_best_hospitals(
        condition=condition,
        ambulance_lat=ambulance_lat,
        ambulance_lon=ambulance_lon,
        total=TOP_N_HOSPITALS,
        min_govt=MIN_GOVT,
    )
    return prediction, hospitals


# ─────────────────────────────────────────────
# DISPLAY & SELECTION
# ─────────────────────────────────────────────

def display_hospital_shortlist(hospitals):
    print(f"\n  {len(hospitals)} hospitals shortlisted (estimated by distance):")
    print("  " + "─"*56)
    for h in hospitals:
        govt_tag = " ★ NEAREST GOVT" if h.get("nearest_govt_fallback") else ""
        dept_tag = " ✓ dept"         if h.get("department_match")       else "  n/a"
        icu_tag  = f"ICU {h['icu_beds_available']} free"
        no_bed   = " ⚠ NO BEDS"      if h.get("no_beds_warning")        else ""
        print(f"\n  {h['rank']}. [{h['type']}] {h['name']}{govt_tag}")
        print(f"     ~{h['distance_km']} km  |  {icu_tag}  |  {dept_tag}{no_bed}")
        rec = h.get("recommendation", "")
        if rec: print(f"     → {rec}")
    print()


def select_hospital(hospitals):
    while True:
        try:
            rank  = int(input("  Select hospital (enter rank number): ").strip())
            match = next((h for h in hospitals if h["rank"] == rank), None)
            if match: return match
            print(f"  [!] Rank {rank} not in list. "
                  f"Choose from {[h['rank'] for h in hospitals]}")
        except ValueError:
            print("  [!] Enter a number.")


# ─────────────────────────────────────────────
# ROUTE — only for selected hospital
# ─────────────────────────────────────────────

def compute_route_for_selection(selected_hospital, ambulance_lat, ambulance_lon,
                                slow_edges=None, force_offline=False):
    print(f"\n  Computing route → {selected_hospital['name']}…")
    engine  = RoutingEngine(city_center_lat=CITY_CENTER_LAT,
                            city_center_lon=CITY_CENTER_LON,
                            force_offline=force_offline)
    updated = engine.route_to_selected_hospital(
        ambulance_lat, ambulance_lon, selected_hospital,
        slow_edges=slow_edges or [],
    )
    _print_route_result(updated)
    return updated


# ─────────────────────────────────────────────
# REROUTING
# ─────────────────────────────────────────────

def reroute_to_same_hospital(selected_hospital, new_amb_lat, new_amb_lon,
                              extra_jams=None, prev_eta=None):
    """
    Reroute to the same locked hospital.

    Ambulance's current position is ALWAYS prepended to slow_edges as the
    primary jam epicentre — the router estimates jam radius from road class
    at that point and replans via side streets to bypass the queue.

    extra_jams: additional [lat, lon] points for jams further ahead.
    prev_eta:   ETA from the previous route, used to show savings.
    """
    print(f"\n  Rerouting → {selected_hospital['name']} (destination locked)…")
    print(f"  Jam epicentre: ambulance position "
          f"({new_amb_lat:.4f}, {new_amb_lon:.4f})")

    all_jams = [[new_amb_lat, new_amb_lon]] + (extra_jams or [])

    if extra_jams:
        print(f"  Additional jam points: {len(extra_jams)}")

    engine  = RoutingEngine(city_center_lat=CITY_CENTER_LAT,
                            city_center_lon=CITY_CENTER_LON,
                            force_offline=True)
    updated = engine.reroute_to_same_hospital(
        selected_hospital, new_amb_lat, new_amb_lon,
        slow_edges=all_jams,
    )
    _print_route_result(updated, rerouted=True, prev_eta=prev_eta)
    return updated


# ─────────────────────────────────────────────
# OUTPUT HELPERS
# ─────────────────────────────────────────────

def _print_route_result(h, rerouted=False, prev_eta=None):
    tag = "REROUTED ROUTE" if rerouted else "ROUTE RESULT"
    print(f"\n  {'─'*56}")
    print(f"  {tag}: {h['name']}")
    print(f"  {'─'*56}")
    print(f"  Distance    : {h['distance_km']} km")

    # Progressive ETA display with savings indicator
    eta = h['travel_time_min']
    if rerouted and prev_eta is not None and prev_eta > eta:
        saved = round(prev_eta - eta, 1)
        print(f"  ETA         : {eta} min  ↓ {saved} min saved vs previous route")
    elif rerouted and prev_eta is not None and prev_eta <= eta:
        diff = round(eta - prev_eta, 1)
        print(f"  ETA         : {eta} min  (↑ {diff} min — detour adds time)")
    else:
        print(f"  ETA         : {eta} min")

    print(f"  Method      : {h.get('routing_method', '?')}")
    print(f"  Tier        : {h.get('tier', '?')}")

    if h.get("reroute_mode"):
        mode_label = {
            "A_close": "MODE A — alley shortcut direct (< 1.5 km to hospital)",
            "B_far":   "MODE B — alley escape → rejoin main road forward",
        }.get(h["reroute_mode"], h["reroute_mode"])
        print(f"  Reroute     : {mode_label}")
        if h.get("remaining_km") is not None:
            print(f"  Remaining   : {h['remaining_km']} km at reroute trigger")

    print(f"  Waypoints   : {h.get('map_waypoints', '?')} "
          f"(from {h.get('raw_waypoints', '?')} raw)")
    print(f"  Lanes used  : {'Yes (alleys/shortcuts)' if h.get('lanes_used') else 'No'}")

    conf = h.get("jam_confidence", 0)
    mult = h.get("traffic_multiplier", 1.0)
    period = h.get("traffic_period", "off_peak")

    if rerouted and conf < JAM_POPUP_THRESHOLD:
        print(f"  Traffic     : ✓ Clear on rerouted path ({conf*100:.0f}% jam chance)")
    elif conf >= JAM_POPUP_THRESHOLD:
        print(f"  Traffic     : ⚠ JAM {conf*100:.0f}% confidence  ×{mult:.1f}")
    else:
        print(f"  Traffic     : ✓ {period.replace('_',' ').title()} "
              f"({conf*100:.0f}% jam chance)")

    for jp in h.get("jam_points", []):
        jam_len = jp.get("jam_length_km",
                         round(jp["radius_m"] * 3.0 / 1000, 2))
        print(f"  Jam point   : ({jp['lat']:.4f},{jp['lon']:.4f})")
        print(f"  Jam radius  : {jp['radius_m']} m  "
              f"(est. queue ~{jam_len} km)")
        print(f"  Edges slowed: {jp['edges_slowed']}")

    rec = h.get("recommendation", "")
    if rec: print(f"  Why chosen  : {rec}")
    print()

    _save_json(h, rerouted)
    _save_map(h)

    if conf >= JAM_POPUP_THRESHOLD and not rerouted:
        print("  " + "!"*54)
        print(f"  ⚠  JAM DETECTED — {conf*100:.0f}% confidence  ×{mult:.1f}")
        print("  Consider rerouting via alternate roads.")
        print("  " + "!"*54)


def _save_json(hospital, rerouted=False):
    data = {
        "hospital":        hospital,
        "rerouted":        rerouted,
        "traffic_summary": get_current_traffic_summary(),
        "jam_points":      hospital.get("jam_points", []),
        "cache_info":      {"cached_graphs": list_cached()},
        "routing_summary": {
            "method":           hospital.get("routing_method", "unknown"),
            "tier":             hospital.get("tier", "unknown"),
            "reroute_mode":     hospital.get("reroute_mode"),
            "remaining_km":     hospital.get("remaining_km"),
            "distance_km":      hospital.get("distance_km", 0),
            "travel_time_min":  hospital.get("travel_time_min", 0),
            "lanes_used":       hospital.get("lanes_used", False),
            "road_quality_avg": hospital.get("road_quality_avg", 0),
            "jam_detected":     hospital.get("jam_detected", False),
            "jam_confidence":   hospital.get("jam_confidence", 0),
            "traffic_period":   hospital.get("traffic_period", "unknown"),
        },
    }
    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  output.json → {OUTPUT_JSON}")
    except Exception as e:
        print(f"  [!] Could not save output.json: {e}")


def _save_map(hospital, amb_lat=None, amb_lon=None):
    try:
        import folium
    except ImportError:
        print("  [!] folium not installed — pip install folium"); return
    try:
        lat = amb_lat or hospital.get("snapped_origin", [hospital["latitude"]])[0]
        lon = amb_lon or hospital.get("snapped_origin", [None, hospital["longitude"]])[1]
        m   = folium.Map(location=[lat, lon], zoom_start=14)
        route = hospital.get("route", [])

        if route:
            folium.PolyLine(route, color="red", weight=5, opacity=0.85,
                            tooltip=f"{hospital['name']} — {hospital['travel_time_min']} min"
                            ).add_to(m)
            for i, pt in enumerate(route[1:-1], 1):
                folium.CircleMarker(location=pt, radius=3, color="red",
                                    fill=True, fill_opacity=0.7,
                                    tooltip=f"wp {i}").add_to(m)

        for jp in hospital.get("jam_points", []):
            jam_len = jp.get("jam_length_km",
                             round(jp["radius_m"] * 3.0 / 1000, 2))
            folium.Circle(
                location=[jp["lat"], jp["lon"]],
                radius=jp["radius_m"],
                color="orange",
                fill=True,
                fill_opacity=0.25,
                tooltip=(f"Jam radius: {jp['radius_m']} m | "
                         f"est. queue: {jam_len} km | "
                         f"{jp['edges_slowed']} edges slowed"),
            ).add_to(m)
            folium.Marker(
                location=[jp["lat"], jp["lon"]],
                tooltip=f"Jam epicentre ({jp['lat']:.4f},{jp['lon']:.4f})",
                popup=folium.Popup(
                    f"<b>Jam epicentre</b><br>"
                    f"Radius: {jp['radius_m']} m<br>"
                    f"Est. queue: {jam_len} km<br>"
                    f"Edges slowed: {jp['edges_slowed']}",
                    max_width=200,
                ),
                icon=folium.Icon(color="orange", icon="warning-sign"),
            ).add_to(m)

        label = hospital["name"]
        if hospital.get("nearest_govt_fallback"): label += " ★ NEAREST GOVT"
        if hospital.get("no_beds_warning"):       label += " ⚠ NO BEDS"

        folium.Marker(
            location=[hospital["latitude"], hospital["longitude"]],
            tooltip=label,
            popup=folium.Popup(
                f"<b>{hospital['name']}</b><br>"
                f"ETA: {hospital['travel_time_min']} min "
                f"({hospital['distance_km']} km)<br>"
                f"ICU: {hospital['icu_beds_available']} free<br>"
                f"General: {hospital['general_beds_available']} free<br>"
                f"Routing: {hospital.get('routing_method', '?')}<br>"
                f"Tier: {hospital.get('tier', '?')}<br>"
                f"<i>{hospital.get('recommendation', '')}</i>",
                max_width=300),
            icon=folium.Icon(color="red", icon="plus-sign"),
        ).add_to(m)

        if route:
            folium.Marker(location=route[0], tooltip="Ambulance",
                          popup="Ambulance start",
                          icon=folium.Icon(color="orange", icon="star")).add_to(m)

        m.save(MAP_HTML)
        print(f"  map.html → {MAP_HTML}")
    except Exception as e:
        print(f"  [!] map save failed: {e}")


def load_last_output():
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────
# INTERACTIVE CLI
# ─────────────────────────────────────────────

def _get_vitals_from_user():
    print("\n" + "─"*60)
    print("  Patient Vitals")
    print("─"*60)
    try:
        vitals = {
            "age":              float(input("  Age (years)               : ")),
            "sex":                    input("  Sex (male / female)       : ").strip(),
            "heart_rate":       float(input("  Heart Rate (bpm)          : ")),
            "blood_pressure":   float(input("  Blood Pressure (mmHg)     : ")),
            "spo2":             float(input("  SpO2 (%)                  : ")),
            "body_temperature": float(input("  Body Temperature (°C)     : ")),
            "glucose":          float(input("  Glucose (mg/dL)           : ")),
        }
        print("\n" + "─"*60)
        print("  Ambulance GPS")
        print("─"*60)
        amb_lat = float(input("  Latitude   (e.g. 22.6449) : "))
        amb_lon = float(input("  Longitude  (e.g. 88.3747) : "))
        return vitals, amb_lat, amb_lon
    except ValueError:
        print("  [!] Numeric fields must be numbers.")
        return None, None, None


def _collect_extra_jams() -> list:
    """
    Ask for jam points OTHER than the ambulance's current position.
    The current position is always prepended automatically in reroute().
    """
    print("\n  Any other jams further ahead on the route?")
    print("  Enter lat,lon — one per line. Empty line when done.")
    print("  Example: 22.6057,88.3742")
    jams = []
    while True:
        line = input("  Extra jam lat,lon: ").strip()
        if not line:
            break
        try:
            parts = [float(x) for x in line.split(",")]
            if len(parts) == 2:
                jams.append(parts)
                print(f"    Added ({parts[0]:.4f},{parts[1]:.4f})")
            else:
                print("  [!] Need exactly 2 values: lat,lon")
        except ValueError:
            print("  [!] Numbers only, e.g. 22.6057,88.3742")
    return jams


def _get_current_position(orig_lat: float, orig_lon: float) -> tuple[float, float]:
    """
    Ask the driver for their current GPS position.
    If within SAME_POSITION_KM of origin, treat as unchanged.
    """
    print("\n  Enter current ambulance GPS position:")
    print(f"  (press Enter twice to keep original: {orig_lat:.4f}, {orig_lon:.4f})")
    try:
        lat_str = input("  Current Lat : ").strip()
        lon_str = input("  Current Lon : ").strip()

        if not lat_str and not lon_str:
            return orig_lat, orig_lon

        new_lat = float(lat_str) if lat_str else orig_lat
        new_lon = float(lon_str) if lon_str else orig_lon

        from offline_router import _haversine_km
        dist_km = _haversine_km(orig_lat, orig_lon, new_lat, new_lon)

        if dist_km < SAME_POSITION_KM:
            print(f"  Position unchanged (< {SAME_POSITION_KM*1000:.0f} m from origin).")
            return orig_lat, orig_lon

        print(f"  Updated position: ({new_lat:.4f}, {new_lon:.4f}) "
              f"— {dist_km:.2f} km from start")
        return new_lat, new_lon

    except ValueError:
        print("  [!] Invalid input — using original position.")
        return orig_lat, orig_lon


def _ask_reroute(selected_hospital, amb_lat, amb_lon):
    """
    Prompt the driver about congestion ahead.
    Shows ETA improvement after rerouting (Google Maps style).
    """
    conf     = selected_hospital.get("jam_confidence", 0.0)
    prev_eta = selected_hospital.get("travel_time_min")

    if conf >= JAM_POPUP_THRESHOLD:
        print(f"\n  ⚠ JAM AUTO-DETECTED ({conf*100:.0f}% confidence) on route to "
              f"{selected_hospital['name']}")
        if input("  Reroute from current position? (yes/no) : ").strip().lower() \
                in ("yes", "y"):
            new_lat, new_lon = _get_current_position(amb_lat, amb_lon)
            extra_jams       = _collect_extra_jams()
            return reroute_to_same_hospital(
                selected_hospital, new_lat, new_lon,
                extra_jams=extra_jams, prev_eta=prev_eta,
            )

    if input("\n  Is there a jam or congestion ahead? (yes/no) : "
             ).strip().lower() not in ("yes", "y"):
        return None

    new_lat, new_lon = _get_current_position(amb_lat, amb_lon)
    extra_jams       = _collect_extra_jams()

    return reroute_to_same_hospital(
        selected_hospital, new_lat, new_lon,
        extra_jams=extra_jams, prev_eta=prev_eta,
    )


def main():
    running = True
    while running:
        vitals, amb_lat, amb_lon = _get_vitals_from_user()
        if vitals is None:
            continue

        prediction, hospitals = predict_and_shortlist(vitals, amb_lat, amb_lon)
        if not prediction or not hospitals:
            continue

        display_hospital_shortlist(hospitals)
        selected = select_hospital(hospitals)
        print(f"\n  Selected: {selected['name']}")

        selected = compute_route_for_selection(selected, amb_lat, amb_lon)

        while True:
            updated = _ask_reroute(selected, amb_lat, amb_lon)
            if updated is None:
                break
            selected = updated

        again = input("\n  Run again? (yes/no) : ").strip().lower()
        running = again in ("yes", "y")

    print("\n  Goodbye.\n")


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────

def _banner(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def _step(n, total, msg):
    print(f"\n  [{n}/{total}] {msg}")


if __name__ == "__main__":
    main()