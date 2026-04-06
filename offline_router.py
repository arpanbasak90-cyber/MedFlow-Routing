"""
offline_router.py — Smart Offline A* Ambulance Router  v7.1
=============================================================
v7.1 — Two surgical fixes for road-beside-Midland being blocked
-------------------------------------------------------------------------

CHANGES FROM v7.0:
──────────────────────────────────────────────────────────────────────────
FIX 1 — Cone was blocking the direct residential road beside Midland:
  Root cause: The jam cone penalised ALL road types (except SMALL_ROAD_TYPES
  were skipped via `continue`) but the condition was:
      if dist_to_jam < block_radius_m and hw not in SMALL_ROAD_TYPES:
  This meant MEDIUM roads (tertiary, secondary) inside the cone were also
  penalised ×20, even though they are perfectly valid detour roads.
  More critically: the cone check used `continue` which SKIPPED the alley
  cheapening below — so small roads inside the cone got NO discount either.
  FIX: Only penalise MAJOR_ROAD_TYPES inside the cone. Medium and small
  roads are never blocked — they are exactly what the detour should use.

FIX 2 — MODE A alley multiplier 0.15 was not strong enough:
  At 1.25 km distance (rem_km < REROUTE_CLOSE_KM = 2.0), MODE A fires.
  Alleys × 0.15 should prefer them, but tier penalty (SMALL_ROAD_TIER_PENALTY
  "short" = 1.0, "medium" = 2.5) was already applied in _apply_tier_weights.
  The net weight was still comparable to the major road route.
  FIX: Alley multiplier reduced to 0.05 — overwhelmingly prefer alleys/
  residential roads when very close to the hospital.

Everything else unchanged from v7.0.
"""

from __future__ import annotations

import math
import logging
import datetime
from typing import Optional

import networkx as nx

from graph_cache_manager import get_graph, update_graph_for_reroute

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# ROUTING TIERS
# ─────────────────────────────────────────────────────────────────
TIER_SHORT_KM  = 1.5
TIER_LONG_KM   = 5.0

# v7.0: raised to 2.0 km — anything within 2 km uses MODE A (direct alley shortcut)
REROUTE_CLOSE_KM          = 2.0
DIRECT_ALLEY_THRESHOLD_KM = REROUTE_CLOSE_KM

ALLEY_THRESHOLD    = {"short": 5.0, "medium": 60.0, "long": 120.0}
TURN_THRESHOLD_DEG = {"short": 5.0, "medium": 7.0,  "long": 10.0}
MIN_SEGMENT_M      = 10.0
DP_EPSILON         = {"short": 0.00008, "medium": 0.00015, "long": 0.00025}
MAX_GAP_M          = 80.0

# ─────────────────────────────────────────────────────────────────
# MODE B GEOMETRY CONSTANTS  (v7.0)
# ─────────────────────────────────────────────────────────────────
JAM_TAIL_OVERSHOOT  = 1.3    # tail = queue_length × this
MIN_TAIL_DISTANCE_M = 400.0  # floor: never project tail less than 400 m ahead

# v7.0: cone from JAM EPICENTRE, not from ambulance
# Only block edges within ±45° of the jam→hospital bearing
BLOCK_CONE_DEG  = 45.0

# Maximum alley edges in escape leg (Google Maps uses very few)
MAX_ALLEY_EDGES = 6

# ─────────────────────────────────────────────────────────────────
# ROAD TABLES
# ─────────────────────────────────────────────────────────────────
ROAD_SPEEDS_KMPH: dict[str, float] = {
    "motorway": 90,       "motorway_link": 70,
    "trunk": 65,          "trunk_link": 55,
    "primary": 55,        "primary_link": 48,
    "secondary": 45,      "secondary_link": 40,
    "tertiary": 38,       "tertiary_link": 34,
    "unclassified": 30,   "residential": 28,
    "living_street": 20,  "service": 22,
    "road": 28,           "default": 28,
}
CONGESTION_PENALTY_S: dict[str, float] = {
    "motorway": 4,        "motorway_link": 3,   "trunk": 6,       "trunk_link": 4,
    "primary": 6,         "primary_link": 4,    "secondary": 4,   "secondary_link": 3,
    "tertiary": 2,        "tertiary_link": 1,   "unclassified": 0, "residential": 0,
    "living_street": 0,   "service": 0,         "road": 0,        "default": 0,
}
TURN_PENALTY_S    = 3.0
SMALL_ROAD_TYPES  = {"residential", "living_street", "service", "unclassified"}
MAJOR_ROAD_TYPES  = {"motorway", "motorway_link", "trunk", "trunk_link",
                     "primary", "primary_link"}
MEDIUM_ROAD_TYPES = {"secondary", "secondary_link", "tertiary", "tertiary_link"}
DETOUR_FACTOR: dict[str, float] = {
    "motorway": 1.04, "motorway_link": 1.03, "trunk": 1.02,
    "primary": 1.01,  "default": 1.00,
}
ROAD_QUALITY: dict[str, int] = {
    "motorway": 10,      "motorway_link": 9,  "trunk": 9,       "trunk_link": 8,
    "primary": 8,        "primary_link": 7,   "secondary": 6,   "secondary_link": 5,
    "tertiary": 4,       "tertiary_link": 3,  "unclassified": 2, "residential": 2,
    "living_street": 1,  "service": 1,        "road": 2,        "default": 2,
}
SMALL_ROAD_TIER_PENALTY = {"short": 1.0, "medium": 2.5, "long": 4.5}

# ─────────────────────────────────────────────────────────────────
# TRAFFIC MODEL
# ─────────────────────────────────────────────────────────────────
CONGESTION_MULTIPLIERS = {
    "morning_peak": {"primary": 2.2, "secondary": 2.0, "tertiary": 1.8,
                     "residential": 1.3, "default": 1.5},
    "evening_peak": {"primary": 2.5, "secondary": 2.2, "tertiary": 2.0,
                     "residential": 1.4, "default": 1.7},
    "night":        {"primary": 1.1, "secondary": 1.1, "tertiary": 1.0,
                     "residential": 1.0, "default": 1.0},
    "off_peak":     {"primary": 1.4, "secondary": 1.3, "tertiary": 1.2,
                     "residential": 1.1, "default": 1.2},
}
JAM_DETECTION_MULTIPLIER = 1.6

# ─────────────────────────────────────────────────────────────────
# JAM RADIUS TABLE  (metres, by road type and traffic period)
# ─────────────────────────────────────────────────────────────────
JAM_RADIUS_M: dict[str, dict[str, float]] = {
    "morning_peak": {
        "motorway": 1200, "trunk": 1000, "primary": 800,
        "secondary": 500, "tertiary": 300, "default": 200,
    },
    "evening_peak": {
        "motorway": 1500, "trunk": 1200, "primary": 1000,
        "secondary": 600, "tertiary": 400, "default": 250,
    },
    "off_peak": {
        "motorway": 600,  "trunk": 500,  "primary": 400,
        "secondary": 250, "tertiary": 150, "default": 100,
    },
    "night": {
        "motorway": 300,  "trunk": 200,  "primary": 150,
        "secondary": 100, "tertiary": 80, "default": 60,
    },
}
JAM_SLOW_MULTIPLIER = 6.0
JAM_LENGTH_FACTOR   = 3.0   # queue = jam_radius × this


# ═════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═════════════════════════════════════════════════════════════════

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    d = math.radians
    dlat = d(lat2 - lat1)
    dlon = d(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(d(lat1)) * math.cos(d(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _haversine_m(p1, p2) -> float:
    return _haversine_km(p1[0], p1[1], p2[0], p2[1]) * 1000.0


def _bearing(p1, p2) -> float:
    lat1 = math.radians(p1[0]); lon1 = math.radians(p1[1])
    lat2 = math.radians(p2[0]); lon2 = math.radians(p2[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _angle_change(b1: float, b2: float) -> float:
    diff = abs(b1 - b2) % 360
    return diff if diff <= 180 else 360 - diff


def _get_tier(dist_km: float) -> str:
    if dist_km < TIER_SHORT_KM: return "short"
    if dist_km < TIER_LONG_KM:  return "medium"
    return "long"


def _project_point(lat: float, lon: float,
                   bearing_deg: float, dist_m: float) -> tuple[float, float]:
    R = 6_371_000.0
    dlat = (dist_m * math.cos(math.radians(bearing_deg))) / R
    dlon = (dist_m * math.sin(math.radians(bearing_deg))) / (
        R * math.cos(math.radians(lat))
    )
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


# ═════════════════════════════════════════════════════════════════
# JAM ESTIMATION FROM A SINGLE POINT
# ═════════════════════════════════════════════════════════════════

def estimate_jam_radius(jam_lat: float, jam_lon: float, G) -> float:
    """Estimate jam detection radius (m) from road class at epicentre."""
    period       = _get_traffic_period()
    radius_table = JAM_RADIUS_M.get(period, JAM_RADIUS_M["off_peak"])

    best_node = None
    best_dist = float("inf")
    for n, data in G.nodes(data=True):
        if "y" not in data or "x" not in data:
            continue
        d = _haversine_km(jam_lat, jam_lon, data["y"], data["x"]) * 1000
        if d < best_dist:
            best_dist, best_node = d, n

    if best_node is None:
        return radius_table["default"]

    best_class   = "default"
    best_quality = -1
    for _, _, edata in G.edges(best_node, data=True):
        hw = edata.get("road_type", edata.get("highway", "default"))
        if isinstance(hw, list): hw = hw[0]
        hw = str(hw).lower().strip()
        q  = ROAD_QUALITY.get(hw, 0)
        if q > best_quality:
            best_quality, best_class = q, hw

    for key in ("motorway", "trunk", "primary", "secondary", "tertiary"):
        if key in best_class:
            return radius_table.get(key, radius_table["default"])
    return radius_table["default"]


def apply_jam_point(G, jam_lat: float, jam_lon: float) -> tuple:
    """
    Slow all edges whose midpoint is within the estimated jam radius.
    NOTE v7.0: Only call this on G_plan (planning graph), NEVER on G_clean.
    """
    jam_radius_m = estimate_jam_radius(jam_lat, jam_lon, G)
    affected = 0

    for u, v, k, data in G.edges(data=True, keys=True):
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        if "y" not in u_data or "y" not in v_data:
            continue
        mid_lat = (u_data["y"] + v_data["y"]) / 2
        mid_lon = (u_data["x"] + v_data["x"]) / 2
        dist_m  = _haversine_km(jam_lat, jam_lon, mid_lat, mid_lon) * 1000
        if dist_m <= jam_radius_m:
            G[u][v][k]["weight"] = G[u][v][k].get("weight", 1.0) * JAM_SLOW_MULTIPLIER
            affected += 1

    log.info(f"Jam point ({jam_lat:.4f},{jam_lon:.4f}): "
             f"radius={jam_radius_m:.0f} m, {affected} edges slowed")
    return G, jam_radius_m, affected


# ═════════════════════════════════════════════════════════════════
# WAYPOINT REDUCTION — HYBRID DP + TURN + GAP-FILL
# ═════════════════════════════════════════════════════════════════

def reduce_waypoints(
    points:             list,
    dp_epsilon:         float = DP_EPSILON["medium"],
    turn_threshold_deg: float = TURN_THRESHOLD_DEG["medium"],
    min_segment_m:      float = MIN_SEGMENT_M,
    max_gap_m:          float = MAX_GAP_M,
) -> list:
    if len(points) <= 3:
        return list(points)

    kept_set = set(_douglas_peucker_indices(points, dp_epsilon))

    for i in range(1, len(points) - 1):
        if _haversine_m(points[i - 1], points[i]) < min_segment_m:
            continue
        b_in  = _bearing(points[i - 1], points[i])
        b_out = _bearing(points[i],     points[i + 1])
        if _angle_change(b_in, b_out) >= turn_threshold_deg:
            kept_set.add(i)

    kept_indices = sorted(kept_set)

    final_indices = [kept_indices[0]]
    for a, b in zip(kept_indices[:-1], kept_indices[1:]):
        gap = _haversine_m(points[a], points[b])
        if gap > max_gap_m:
            final_indices.extend(range(a + 1, b))
        final_indices.append(b)

    seen = set(); result = []
    for idx in final_indices:
        if idx not in seen:
            seen.add(idx)
            result.append(points[idx])
    return result


def reduce_to_turns(points, turn_threshold_deg=15.0, min_segment_m=MIN_SEGMENT_M):
    return reduce_waypoints(
        points,
        dp_epsilon=DP_EPSILON["medium"],
        turn_threshold_deg=turn_threshold_deg,
        min_segment_m=min_segment_m,
    )


def _douglas_peucker_indices(points: list, epsilon: float) -> list[int]:
    if len(points) <= 2:
        return list(range(len(points)))

    def perp_deg(pt, s, e):
        x0, y0 = pt[1], pt[0]; x1, y1 = s[1], s[0]; x2, y2 = e[1], e[0]
        dx, dy = x2 - x1, y2 - y1
        denom  = math.hypot(dx, dy)
        if denom == 0:
            return math.hypot(x0 - x1, y0 - y1)
        return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / denom

    def _rdp(si, ei):
        if ei <= si + 1:
            return [si, ei]
        md, mi = 0.0, si
        for i in range(si + 1, ei):
            d = perp_deg(points[i], points[si], points[ei])
            if d > md:
                md, mi = d, i
        if md > epsilon:
            return _rdp(si, mi)[:-1] + _rdp(mi, ei)
        return [si, ei]

    return _rdp(0, len(points) - 1)


def _douglas_peucker(points: list, epsilon: float) -> list:
    return [points[i] for i in _douglas_peucker_indices(points, epsilon)]


# ═════════════════════════════════════════════════════════════════
# TRAFFIC ESTIMATION
# ═════════════════════════════════════════════════════════════════

def _get_traffic_period(hour: int = None) -> str:
    h = hour if hour is not None else datetime.datetime.now().hour
    if 7 <= h < 10:      return "morning_peak"
    if 17 <= h < 20:     return "evening_peak"
    if h >= 22 or h < 6: return "night"
    return "off_peak"


def _estimate_congestion(road_type: str, hour: int = None) -> float:
    period = _get_traffic_period(hour)
    mults  = CONGESTION_MULTIPLIERS[period]
    hw     = road_type.lower()
    for key in ("primary", "secondary", "tertiary", "residential"):
        if key in hw:
            return mults.get(key, mults["default"])
    return mults["default"]


def _avg_congestion_on_route(road_types: list) -> float:
    if not road_types:
        return 1.0
    return sum(_estimate_congestion(rt) for rt in road_types) / len(road_types)


def get_current_traffic_summary() -> dict:
    period = _get_traffic_period()
    mults  = CONGESTION_MULTIPLIERS[period]
    avg    = sum(mults.values()) / len(mults)
    return {
        "traffic_period": period,
        "avg_multiplier": round(avg, 2),
        "jam_expected":   avg >= JAM_DETECTION_MULTIPLIER,
        "description": {
            "morning_peak": "Morning rush hour — heavy traffic on main roads",
            "evening_peak": "Evening rush hour — severe congestion expected",
            "night":        "Light traffic — roads mostly clear",
            "off_peak":     "Moderate traffic — some congestion on main roads",
        }[period],
    }


# ═════════════════════════════════════════════════════════════════
# ETA COMPUTATION  (v7.0 — clean, no mysterious discount factors)
# ═════════════════════════════════════════════════════════════════

def _compute_progressive_eta(
    raw_time_s:      float,
    rem_km:          float,
    orig_km:         float,
    road_types_used: list,
    lanes_used:      bool,
    is_rerouted:     bool,
) -> float:
    """
    ETA from clean base weights (congestion already baked in by _add_weights).
    No jam multiplier here — jam ×6 is ONLY applied to G_plan for path
    selection, never to the weights used for time extraction.

    Small alley-escape bonus (0.95×) because alleys skip signalised junctions.
    """
    alley_factor = 0.95 if (is_rerouted and lanes_used) else 1.0
    adjusted_s   = raw_time_s * alley_factor
    eta_min      = round(adjusted_s / 60.0, 2)

    log.info(
        f"ETA: raw={raw_time_s/60:.1f} min "
        f"× alley={alley_factor:.2f} = {eta_min} min "
        f"(rem={rem_km:.2f} km, rerouted={is_rerouted})"
    )
    return eta_min


def _compute_jam_confidence(road_types: list, is_rerouted: bool) -> float:
    """
    v7.0: Confidence based on road type congestion multipliers only.
    If rerouted and path is alley-heavy → cap at 0.25 (path IS clear).
    """
    avg      = _avg_congestion_on_route(road_types)
    raw_conf = min(max((avg - 1.0) / 1.5, 0.0), 1.0)

    if is_rerouted:
        # We successfully rerouted — the new path is clear
        raw_conf = min(raw_conf, 0.25)

    return round(raw_conf, 2)


# ═════════════════════════════════════════════════════════════════
# NODE SNAPPING
# ═════════════════════════════════════════════════════════════════

def _nearest_reachable_node(ox, G, lat, lon, is_destination=False) -> int:
    node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
    if G.has_node(node):
        return node
    radii = [50, 100, 200, 400] if not is_destination else [30, 75, 150, 300, 600]
    nd = [(n, d["y"], d["x"]) for n, d in G.nodes(data=True)
          if "y" in d and "x" in d]
    for r in radii:
        within = [(n, _haversine_km(lat, lon, ny, nx) * 1000) for n, ny, nx in nd]
        within = [(n, dd) for n, dd in within if dd <= r]
        if within:
            return min(within, key=lambda x: x[1])[0]
    return min(nd, key=lambda t: _haversine_km(lat, lon, t[1], t[2]))[0]


def _find_nearest_main_road_node(ox, G, lat, lon,
                                 search_radius_km=1.5) -> Optional[int]:
    """Retained for backward compatibility."""
    nd = [(n, d["y"], d["x"]) for n, d in G.nodes(data=True)
          if "y" in d and "x" in d]
    cands = []
    for n, ny, nx in nd:
        dist = _haversine_km(lat, lon, ny, nx)
        if dist > search_radius_km:
            continue
        for _, _, data in G.edges(n, data=True):
            hw = data.get("highway", "default")
            if isinstance(hw, list): hw = hw[0]
            hw = str(hw).lower()
            if hw in MAJOR_ROAD_TYPES or hw in MEDIUM_ROAD_TYPES:
                cands.append((n, dist))
                break
    if not cands:
        return None
    return min(cands, key=lambda x: x[1])[0]


# ─────────────────────────────────────────────────────────────────
# v7.0 CORE: FIND JAM TAIL REJOIN NODE
# ─────────────────────────────────────────────────────────────────

def _find_jam_tail_rejoin(
    G,
    jam_lat:       float,
    jam_lon:       float,
    dest_lat:      float,
    dest_lon:      float,
    jam_radius_m:  float,
) -> Optional[int]:
    """
    Find a main/medium road node just past the jam tail.

    v7.0: bearing computed from JAM EPICENTRE → hospital (not from ambulance).
    Tail projected from jam epicentre, not from ambulance position.
    Searches 1.0 → 1.5 → 2.5 km from tail point.
    """
    # Bearing: jam epicentre → destination (hospital)
    brg = _bearing((jam_lat, jam_lon), (dest_lat, dest_lon))

    queue_length_m = jam_radius_m * JAM_LENGTH_FACTOR
    tail_dist_m    = max(queue_length_m * JAM_TAIL_OVERSHOOT, MIN_TAIL_DISTANCE_M)
    tail_lat, tail_lon = _project_point(jam_lat, jam_lon, brg, tail_dist_m)

    log.info(
        f"Jam tail: queue={queue_length_m:.0f} m → tail at {tail_dist_m:.0f} m "
        f"along {brg:.1f}° → ({tail_lat:.5f},{tail_lon:.5f})"
    )

    for search_km in (1.0, 1.5, 2.5):
        candidates = []
        for n, data in G.nodes(data=True):
            if "y" not in data or "x" not in data:
                continue
            ny, nx = data["y"], data["x"]
            dist_from_tail = _haversine_km(tail_lat, tail_lon, ny, nx)
            if dist_from_tail > search_km:
                continue
            for _, _, edata in G.edges(n, data=True):
                hw = edata.get("highway", edata.get("road_type", "default"))
                if isinstance(hw, list): hw = hw[0]
                hw = str(hw).lower().strip()
                if hw in MAJOR_ROAD_TYPES or hw in MEDIUM_ROAD_TYPES:
                    candidates.append((n, dist_from_tail))
                    break
        if candidates:
            best = min(candidates, key=lambda x: x[1])[0]
            log.info(f"Rejoin found at search_km={search_km}: node {best}")
            return best

    return None


def _count_small_road_edges(G, path_nodes: list) -> int:
    count = 0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if not G.has_edge(u, v):
            continue
        best = min(G[u][v].values(), key=lambda d: d.get("weight", 9999))
        hw   = best.get("road_type", best.get("highway", "default"))
        if isinstance(hw, list): hw = hw[0]
        if str(hw).lower().strip() in SMALL_ROAD_TYPES:
            count += 1
    return count


# ═════════════════════════════════════════════════════════════════
# HOSPITAL ROUTER  (single destination)
# ═════════════════════════════════════════════════════════════════

class HospitalRouter:

    def __init__(self, hosp_id: str, hosp_lat: float, hosp_lon: float,
                 radius_m: int = 5000):
        self.hosp_id  = hosp_id
        self.hosp_lat = hosp_lat
        self.hosp_lon = hosp_lon
        self.radius_m = radius_m
        self.G: Optional[object] = None
        self._original_dist_km: float = 0.0

    # ── graph ─────────────────────────────────────────────────────

    def load_graph(self, amb_lat: float, amb_lon: float) -> bool:
        G_raw = get_graph(
            self.hosp_id, self.hosp_lat, self.hosp_lon,
            amb_lat, amb_lon, self.radius_m,
        )
        if G_raw is None:
            return False
        self.G = self._add_weights(G_raw)
        return True

    def _add_weights(self, G) -> object:
        """
        Assign clean base weights with congestion baked in.
        NEVER called with a jam-poisoned graph.
        These weights are used for ETA extraction.
        """
        primary_spd = ROAD_SPEEDS_KMPH["primary"]
        for u, v, k, data in G.edges(data=True, keys=True):
            hw = data.get("highway", "default")
            if isinstance(hw, list): hw = hw[0]
            hw = str(hw).lower().strip()

            spd      = ROAD_SPEEDS_KMPH.get(hw, ROAD_SPEEDS_KMPH["default"])
            length_m = data.get("length", 1.0)
            travel_s = (length_m / 1000.0) / spd * 3600.0 * _estimate_congestion(hw)
            penalty  = CONGESTION_PENALTY_S.get(hw, 0)
            factor   = DETOUR_FACTOR.get(hw, DETOUR_FACTOR["default"])

            access = data.get("access", "")
            if isinstance(access, list): access = access[0]
            if str(access).lower() in ("private", "destination"):
                penalty += 30

            if hw in SMALL_ROAD_TYPES:
                travel_s *= 1.6

            alley_saving = 0.0
            if hw in SMALL_ROAD_TYPES:
                ps = (length_m / 1000.0) / primary_spd * 3600.0
                alley_saving = ps - travel_s

            G[u][v][k]["weight"]        = max((travel_s + penalty) * factor, 0.1)
            G[u][v][k]["travel_time_s"] = travel_s
            G[u][v][k]["road_type"]     = hw
            G[u][v][k]["alley_saving"]  = alley_saving
            G[u][v][k]["quality"]       = ROAD_QUALITY.get(hw, 2)
        return G

    def _apply_tier_weights(self, G, tier: str) -> object:
        thr     = ALLEY_THRESHOLD[tier]
        penalty = SMALL_ROAD_TIER_PENALTY[tier]
        for u, v, k, data in G.edges(data=True, keys=True):
            hw     = data.get("road_type", "default")
            saving = data.get("alley_saving", 0.0)
            if hw in SMALL_ROAD_TYPES:
                if tier == "short" and saving > 0 and saving >= thr:
                    G[u][v][k]["weight"] = max(data["weight"] - saving * 0.5, 0.1)
                else:
                    G[u][v][k]["weight"] = data["weight"] * penalty
        return G

    @staticmethod
    def _make_heuristic(G):
        ms = max(ROAD_SPEEDS_KMPH.values()) / 3.6
        def h(u, v):
            return (_haversine_km(
                G.nodes[u].get("y", 0), G.nodes[u].get("x", 0),
                G.nodes[v].get("y", 0), G.nodes[v].get("x", 0),
            ) * 1000 / ms)
        return h

    # ── jam-ahead guard ───────────────────────────────────────────

    def _jam_is_ahead(self, amb_lat, amb_lon, dest_lat, dest_lon,
                      jam_points) -> bool:
        amb_dest_km = _haversine_km(amb_lat, amb_lon, dest_lat, dest_lon)
        for jp in jam_points:
            jlat, jlon  = jp[0], jp[1]
            jam_dest_km = _haversine_km(jlat, jlon, dest_lat, dest_lon)
            jam_amb_km  = _haversine_km(amb_lat, amb_lon, jlat, jlon)
            # Self-jam: ambulance position is the jam epicentre
            if jam_amb_km < 0.05:
                return True
            if jam_dest_km < amb_dest_km and jam_amb_km < amb_dest_km * 1.3:
                return True
        return False

    # ── v7.1 CORE REROUTE ────────────────────────────────────────

    def _build_dynamic_reroute(
        self, G_clean, ox,
        orig:       int,
        dest:       int,
        amb_lat:    float,
        amb_lon:    float,
        dest_lat:   float,
        dest_lon:   float,
        rem_km:     float,
        jam_points: list = None,
    ) -> Optional[list]:
        """
        Google Maps-style reroute around a jam.

        G_clean: pristine weighted graph — used for ETA extraction.
                 DO NOT apply jam weights to this graph.

        All jam-weight manipulation happens on G_plan = G_clean.copy().

        MODE A (rem_km < REROUTE_CLOSE_KM = 2.0 km):
            Very close to hospital — direct alley shortcut.
            v7.1: Alleys ×0.05 (was 0.15) — overwhelmingly prefer small
            roads so the direct road beside Midland always wins.
            Major roads ×5.0 (unchanged).

        MODE B (rem_km >= REROUTE_CLOSE_KM):
            Phase 1: Build G_plan with jam cone blocked.
                     v7.1 FIX: cone ONLY penalises MAJOR_ROAD_TYPES.
                     Medium (tertiary/secondary) and small (residential/
                     service) roads inside the cone are NEVER blocked —
                     they are the valid detour roads.
            Phase 2: Find rejoin node past jam tail (unchanged).
            Phase 3: Fast main-road run on G_clean (unchanged).
        """
        jp_list = jam_points or [[amb_lat, amb_lon]]
        jam_lat, jam_lon = jp_list[0][0], jp_list[0][1]
        jam_radius_m   = estimate_jam_radius(jam_lat, jam_lon, G_clean)
        queue_length_m = jam_radius_m * JAM_LENGTH_FACTOR

        # ── MODE A: very close — direct alley shortcut ─────────────
        if rem_km < REROUTE_CLOSE_KM:
            log.info(
                f"Reroute MODE A ({rem_km:.2f} km < {REROUTE_CLOSE_KM} km) "
                f"— direct alley shortcut"
            )
            G_a = G_clean.copy()
            for u, v, k, data in G_a.edges(data=True, keys=True):
                hw = data.get("road_type", "default")
                if hw in SMALL_ROAD_TYPES:
                    # v7.1 FIX: 0.05 (was 0.15) — overwhelmingly prefer
                    # small roads so road beside Midland always wins
                    G_a[u][v][k]["weight"] *= 0.05
                elif hw in MAJOR_ROAD_TYPES:
                    G_a[u][v][k]["weight"] *= 5.0
            try:
                return nx.astar_path(
                    G_a, orig, dest,
                    heuristic=self._make_heuristic(G_a),
                    weight="weight",
                )
            except nx.NetworkXNoPath:
                return None

        # ── MODE B: three-phase jam-tail escape ───────────────────
        # v7.0: bearing from JAM EPICENTRE → hospital (not from ambulance)
        jam_hosp_brg = _bearing((jam_lat, jam_lon), (dest_lat, dest_lon))

        log.info(
            f"Reroute MODE B ({rem_km:.2f} km) "
            f"| jam=({jam_lat:.4f},{jam_lon:.4f}) "
            f"| jam_radius={jam_radius_m:.0f} m "
            f"| queue={queue_length_m:.0f} m "
            f"| jam→hosp bearing={jam_hosp_brg:.1f}°"
        )

        # ── Phase 2: find rejoin node past jam tail ────────────────
        rejoin = _find_jam_tail_rejoin(
            G_clean, jam_lat, jam_lon, dest_lat, dest_lon, jam_radius_m
        )

        # Guaranteed fallback: nearest any-road node near tail
        if rejoin is None:
            tail_dist_m = max(queue_length_m * JAM_TAIL_OVERSHOOT, MIN_TAIL_DISTANCE_M)
            tail_lat, tail_lon = _project_point(
                jam_lat, jam_lon, jam_hosp_brg, tail_dist_m
            )
            rejoin = _nearest_reachable_node(ox, G_clean, tail_lat, tail_lon)
            log.warning(
                f"MODE B: no major/medium rejoin — fallback nearest node "
                f"at tail ({tail_lat:.5f},{tail_lon:.5f})"
            )

        rejoin_lat = G_clean.nodes[rejoin].get("y", amb_lat)
        rejoin_lon = G_clean.nodes[rejoin].get("x", amb_lon)
        log.info(
            f"MODE B rejoin: node {rejoin} "
            f"({rejoin_lat:.4f},{rejoin_lon:.4f}) "
            f"— {_haversine_km(jam_lat, jam_lon, rejoin_lat, rejoin_lon)*1000:.0f} m "
            f"from jam epicentre"
        )

        # ── Phase 1: build G_plan with directional cone block ─────
        # block_radius_m = full queue length
        block_radius_m = queue_length_m

        # G_plan is used ONLY for A* path selection, NOT for ETA
        G_plan = G_clean.copy()

        for u, v, k, data in G_plan.edges(data=True, keys=True):
            hw     = data.get("road_type", "default")
            u_data = G_clean.nodes[u]
            v_data = G_clean.nodes[v]

            if "y" in u_data and "y" in v_data:
                mid_lat = (u_data["y"] + v_data["y"]) / 2
                mid_lon = (u_data["x"] + v_data["x"]) / 2

                # Distance from JAM EPICENTRE (not ambulance) to edge midpoint
                dist_to_jam = _haversine_km(
                    jam_lat, jam_lon, mid_lat, mid_lon
                ) * 1000

                # v7.1 FIX: ONLY penalise MAJOR roads inside the cone.
                # Medium (tertiary/secondary) and small (residential/service)
                # roads are NEVER blocked — they are the valid detour path.
                # Previously `hw not in SMALL_ROAD_TYPES` also caught medium
                # roads, blocking the road beside Midland Nursing Home.
                if (dist_to_jam < block_radius_m
                        and hw in MAJOR_ROAD_TYPES):       # ← v7.1 change
                    edge_brg   = _bearing((jam_lat, jam_lon), (mid_lat, mid_lon))
                    angle_diff = _angle_change(jam_hosp_brg, edge_brg)
                    if angle_diff < BLOCK_CONE_DEG:
                        # Major road edge inside the jam cone — heavily penalise
                        G_plan[u][v][k]["weight"] *= 20.0
                        continue

            # Alleys: attractive for lateral escape hop
            if hw in SMALL_ROAD_TYPES:
                G_plan[u][v][k]["weight"] *= 0.35
            elif hw in MAJOR_ROAD_TYPES:
                G_plan[u][v][k]["weight"] *= 5.0

        # ── Phase 3: G_main — clean weights, heavily favour main roads
        # Built fresh from self.G (the pristine base), NEVER from G_plan
        G_main = self._add_weights(self.G.copy())
        for u, v, k, data in G_main.edges(data=True, keys=True):
            hw = data.get("road_type", "default")
            if hw in SMALL_ROAD_TYPES:
                G_main[u][v][k]["weight"] *= 5.0
            elif hw in MAJOR_ROAD_TYPES:
                G_main[u][v][k]["weight"] *= 0.7
            elif hw in MEDIUM_ROAD_TYPES:
                G_main[u][v][k]["weight"] *= 0.85

        # ── Run A* for escape and main-road segments ───────────────
        try:
            seg1 = nx.astar_path(
                G_plan, orig, rejoin,
                heuristic=self._make_heuristic(G_plan),
                weight="weight",
            )
        except nx.NetworkXNoPath:
            log.warning("MODE B: escape segment failed — plain A*")
            return None

        try:
            seg2 = nx.astar_path(
                G_main, rejoin, dest,
                heuristic=self._make_heuristic(G_main),
                weight="weight",
            )
        except nx.NetworkXNoPath:
            log.warning("MODE B: main-road segment failed — plain A*")
            return None

        # ── Alley budget check ─────────────────────────────────────
        # Use G_clean (not G_plan) to count road types — G_plan has inflated weights
        alley_count = _count_small_road_edges(G_clean, seg1)
        log.info(
            f"MODE B: escape={len(seg1)} nodes "
            f"({alley_count} alley edges, budget={MAX_ALLEY_EDGES})"
        )

        if alley_count > MAX_ALLEY_EDGES:
            log.info("MODE B: too many alleys — retrying escape with tighter penalty")
            G_plan2 = G_plan.copy()
            for u, v, k, data in G_plan2.edges(data=True, keys=True):
                if data.get("road_type", "default") in SMALL_ROAD_TYPES:
                    G_plan2[u][v][k]["weight"] *= 3.0
            try:
                seg1 = nx.astar_path(
                    G_plan2, orig, rejoin,
                    heuristic=self._make_heuristic(G_plan2),
                    weight="weight",
                )
            except nx.NetworkXNoPath:
                pass  # keep original seg1

        combined = seg1 + seg2[1:]
        log.info(
            f"MODE B done: "
            f"escape={len(seg1)} + main={len(seg2)} = {len(combined)} nodes"
        )
        return combined

    # ── main route entry ──────────────────────────────────────────

    def get_route(
        self,
        origin_lat:  float,
        origin_lon:  float,
        dest_lat:    float,
        dest_lon:    float,
        slow_edges:  list = None,
        _rerouting:  bool = False,
        _orig_dist_km: float = 0.0,
    ) -> dict:
        try:
            import osmnx as ox
        except ImportError:
            return self._fallback(origin_lat, origin_lon, dest_lat, dest_lon)

        if self.G is None:
            if not self.load_graph(origin_lat, origin_lon):
                return self._fallback(origin_lat, origin_lon, dest_lat, dest_lon)

        straight_km = _haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
        tier        = _get_tier(straight_km)

        if not _rerouting and straight_km > 0:
            self._original_dist_km = straight_km
        orig_km = self._original_dist_km if self._original_dist_km > 0 else straight_km

        # ── v7.0: G_clean = pristine weights, used for ETA extraction ──
        # Apply tier weights (alley penalties) but NO jam multipliers
        G_clean = self.G.copy()
        G_clean = self._apply_tier_weights(G_clean, tier)

        # ── G_plan = jam-poisoned planning graph (A* cost only) ────
        # Built separately from G_clean so ETA is never contaminated
        jam_info = []
        G_plan = G_clean.copy()   # start from clean tier-weighted graph
        for jp in (slow_edges or []):
            if len(jp) >= 2:
                G_plan, radius_m, affected = apply_jam_point(G_plan, jp[0], jp[1])
                jam_length_km = round(radius_m * JAM_LENGTH_FACTOR / 1000, 2)
                jam_info.append({
                    "lat":           jp[0],
                    "lon":           jp[1],
                    "radius_m":      round(radius_m),
                    "jam_length_km": jam_length_km,
                    "edges_slowed":  affected,
                })

        orig = _nearest_reachable_node(ox, G_clean, origin_lat, origin_lon)
        dest = _nearest_reachable_node(ox, G_clean, dest_lat, dest_lon,
                                       is_destination=True)

        route_nodes = None
        method      = "astar_offline"

        jam_ahead = (
            _rerouting
            or (
                bool(slow_edges)
                and self._jam_is_ahead(
                    origin_lat, origin_lon, dest_lat, dest_lon, slow_edges
                )
            )
        )

        if jam_ahead:
            method = "astar_jam_rerouted" if slow_edges else "astar_rerouted"
            # Pass G_clean to _build_dynamic_reroute — it builds G_plan internally
            route_nodes = self._build_dynamic_reroute(
                G_clean, ox, orig, dest,
                amb_lat=origin_lat, amb_lon=origin_lon,
                dest_lat=dest_lat, dest_lon=dest_lon,
                rem_km=straight_km,
                jam_points=slow_edges or [],
            )

        if route_nodes is None:
            # Fallback: A* on G_plan (jam-aware path selection, but ETA from G_clean)
            try:
                route_nodes = nx.astar_path(
                    G_plan, orig, dest,
                    heuristic=self._make_heuristic(G_plan),
                    weight="weight",
                )
            except nx.NetworkXNoPath:
                return self._fallback(origin_lat, origin_lon, dest_lat, dest_lon)

        # ── ALWAYS extract ETA from G_clean (no jam poison) ────────
        result = self._extract_route(
            G_clean, route_nodes, tier, method,
            rem_km=straight_km,
            orig_km=orig_km,
            is_rerouted=_rerouting,
        )

        conf = _compute_jam_confidence(
            result.get("road_types_used", []), is_rerouted=_rerouting
        )
        avg  = _avg_congestion_on_route(result.get("road_types_used", []))

        result.update({
            "jam_detected":       conf >= 0.40,
            "jam_confidence":     conf,
            "traffic_multiplier": round(avg, 2),
            "traffic_period":     _get_traffic_period(),
            "snapped_origin":     [G_clean.nodes[orig]["y"], G_clean.nodes[orig]["x"]],
            "snapped_dest":       [G_clean.nodes[dest]["y"], G_clean.nodes[dest]["x"]],
            "tier":               tier,
            "straight_km":        round(straight_km, 3),
            "jam_points":         jam_info,
        })
        return result

    def reroute(
        self,
        new_amb_lat: float,
        new_amb_lon: float,
        dest_lat:    float,
        dest_lon:    float,
        slow_edges:  list = None,
    ) -> dict:
        log.info(f"Rerouting from ({new_amb_lat:.4f},{new_amb_lon:.4f}) "
                 f"to hospital {self.hosp_id}")

        G_raw = update_graph_for_reroute(
            self.hosp_id, self.hosp_lat, self.hosp_lon,
            new_amb_lat, new_amb_lon, self.radius_m,
        )
        if G_raw is not None:
            self.G = self._add_weights(G_raw)

        rem_km = _haversine_km(new_amb_lat, new_amb_lon, dest_lat, dest_lon)
        mode   = "A_close" if rem_km < REROUTE_CLOSE_KM else "B_far"
        log.info(f"Reroute distance remaining: {rem_km:.3f} km → MODE {mode}")

        # Ambulance position is always the primary jam epicentre
        jam_list = list(slow_edges or [])
        if not jam_list:
            jam_list = [[new_amb_lat, new_amb_lon]]
        elif (abs(jam_list[0][0] - new_amb_lat) > 0.0001 or
              abs(jam_list[0][1] - new_amb_lon) > 0.0001):
            jam_list = [[new_amb_lat, new_amb_lon]] + jam_list

        result = self.get_route(
            new_amb_lat, new_amb_lon, dest_lat, dest_lon,
            slow_edges=jam_list,
            _rerouting=True,
            _orig_dist_km=self._original_dist_km,
        )

        result["routing_method"] = (
            "astar_reroute_close" if mode == "A_close" else "astar_reroute_far"
        )
        result["remaining_km"] = round(rem_km, 3)
        result["reroute_mode"] = mode
        return result

    # ── route extraction — ALWAYS from G_clean ────────────────────

    def _extract_route(
        self, G_clean, route_nodes, tier, routing_method,
        rem_km: float = 0.0,
        orig_km: float = 0.0,
        is_rerouted: bool = False,
    ) -> dict:
        """
        Extract route stats from G_clean (no jam multiplier in weights).
        ETA reflects real road speeds + time-of-day congestion only.
        """
        raw         = [[G_clean.nodes[n]["y"], G_clean.nodes[n]["x"]]
                       for n in route_nodes]
        dist_m      = 0.0
        raw_time_s  = 0.0
        quality_sum = 0.0
        road_types: list = []
        lanes_used  = False
        prev        = None

        for u, v in zip(route_nodes[:-1], route_nodes[1:]):
            if not G_clean.has_edge(u, v):
                continue
            best         = min(G_clean[u][v].values(),
                               key=lambda d: d.get("weight", 9999))
            dist_m      += best.get("length", 0)
            # Use travel_time_s (clean), not weight (may have tier penalties)
            raw_time_s  += best.get("travel_time_s", best.get("weight", 0))
            rt           = best.get("road_type", "unknown")
            quality_sum += best.get("quality", 2)
            if rt not in road_types:
                road_types.append(rt)
            if rt in SMALL_ROAD_TYPES:
                lanes_used = True
            # Turn penalty: only at actual road-type transitions
            if prev is not None and prev != rt and "link" not in rt:
                raw_time_s += TURN_PENALTY_S
            prev = rt

        eps  = DP_EPSILON[tier]
        thr  = TURN_THRESHOLD_DEG[tier]
        wpts = reduce_waypoints(
            raw,
            dp_epsilon=eps,
            turn_threshold_deg=thr,
            min_segment_m=MIN_SEGMENT_M,
            max_gap_m=MAX_GAP_M,
        )
        pct = 100 * (1 - len(wpts) / max(len(raw), 1))
        log.info(
            f"Waypoints: {len(raw)} raw → {len(wpts)} kept "
            f"(tier={tier}, -{pct:.0f}%)"
        )

        eta_min = _compute_progressive_eta(
            raw_time_s=raw_time_s,
            rem_km=rem_km if rem_km > 0 else dist_m / 1000.0,
            orig_km=orig_km if orig_km > 0 else dist_m / 1000.0,
            road_types_used=road_types,
            lanes_used=lanes_used,
            is_rerouted=is_rerouted,
        )

        return {
            "success":          True,
            "distance_km":      round(dist_m / 1000, 3),
            "travel_time_min":  eta_min,
            "route_coords":     wpts,
            "raw_waypoints":    len(raw),
            "map_waypoints":    len(wpts),
            "road_types_used":  road_types,
            "lanes_used":       lanes_used,
            "routing_method":   routing_method,
            "road_quality_avg": round(quality_sum / max(len(route_nodes) - 1, 1), 2),
        }

    # ── edge modifiers ────────────────────────────────────────────

    @staticmethod
    def _apply_slow(G, ox, slow_edges, pf=8.0) -> object:
        for item in (slow_edges or []):
            if len(item) == 4:
                lat1, lon1, lat2, lon2 = item
                u = _nearest_reachable_node(ox, G, lat1, lon1)
                v = _nearest_reachable_node(ox, G, lat2, lon2)
            else:
                u, v = item
            if G.has_edge(u, v):
                for k in G[u][v]:
                    G[u][v][k]["weight"] *= pf
        return G

    def _fallback(self, lat1, lon1, lat2, lon2) -> dict:
        dist     = _haversine_km(lat1, lon1, lat2, lon2)
        spd_kmph = ROAD_SPEEDS_KMPH.get("primary", 55)
        eta_min  = round(dist / spd_kmph * 60, 2)
        mult     = _estimate_congestion("primary")
        return {
            "success":            True,
            "distance_km":        round(dist, 3),
            "travel_time_min":    eta_min,
            "route_coords":       [[lat1, lon1], [lat2, lon2]],
            "raw_waypoints":      2,
            "map_waypoints":      2,
            "road_types_used":    [],
            "lanes_used":         False,
            "routing_method":     "straight_line_fallback",
            "tier":               _get_tier(dist),
            "straight_km":        round(dist, 3),
            "road_quality_avg":   0,
            "jam_detected":       mult >= JAM_DETECTION_MULTIPLIER,
            "jam_confidence":     round(min(max((mult - 1.0) / 1.5, 0.0), 1.0), 2),
            "traffic_multiplier": round(mult, 2),
            "traffic_period":     _get_traffic_period(),
            "jam_points":         [],
        }


# ═════════════════════════════════════════════════════════════════
# OFFLINE ROUTER  (public interface for routing_engine.py)
# ═════════════════════════════════════════════════════════════════

class OfflineRouter:

    def __init__(self):
        self._routers: dict[str, HospitalRouter] = {}

    def _router_for(self, hosp_id: str, hosp_lat: float,
                    hosp_lon: float, radius_m: int) -> HospitalRouter:
        if hosp_id not in self._routers:
            self._routers[hosp_id] = HospitalRouter(
                hosp_id, hosp_lat, hosp_lon, radius_m)
        return self._routers[hosp_id]

    def get_route(
        self,
        origin_lat: float, origin_lon: float,
        dest_lat:   float, dest_lon:   float,
        hosp_id:    str   = "unknown",
        hosp_lat:   float = None,
        hosp_lon:   float = None,
        radius_m:   int   = 5000,
        slow_edges: list  = None,
    ) -> dict:
        hl  = hosp_lat if hosp_lat is not None else dest_lat
        hlo = hosp_lon if hosp_lon is not None else dest_lon
        router = self._router_for(hosp_id, hl, hlo, radius_m)
        return router.get_route(
            origin_lat, origin_lon, dest_lat, dest_lon,
            slow_edges=slow_edges or [],
        )

    def reroute(
        self,
        new_amb_lat: float, new_amb_lon: float,
        dest_lat:    float, dest_lon:    float,
        hosp_id:     str   = "unknown",
        hosp_lat:    float = None,
        hosp_lon:    float = None,
        radius_m:    int   = 5000,
        slow_edges:  list  = None,
    ) -> dict:
        hl  = hosp_lat if hosp_lat is not None else dest_lat
        hlo = hosp_lon if hosp_lon is not None else dest_lon
        router = self._router_for(hosp_id, hl, hlo, radius_m)
        return router.reroute(
            new_amb_lat, new_amb_lon, dest_lat, dest_lon,
            slow_edges=slow_edges or [],
        )