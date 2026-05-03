"""
online_router.py — Online A* Router v5.1
=========================================
v5.1 changes
------------
• OSRM public API fallback when osmnx is unavailable.
• Falls back gracefully: OnlineRouter → OSRM → OfflineRouter → straight-line.
"""

from __future__ import annotations

import logging

from offline_router import (
    HospitalRouter,
    OfflineRouter,
    get_current_traffic_summary,
    _haversine_km,
    _estimate_congestion,
    _get_traffic_period,
    JAM_DETECTION_MULTIPLIER,
)
from graph_cache_manager import get_graph

log = logging.getLogger(__name__)


class OnlineRouter:
    """
    Fetches fresh OSM data keyed by hospital ID (via graph_cache_manager).
    Falls back to OSRM then OfflineRouter then straight-line on any error.
    slow_edges = list of [lat, lon] jam epicentre points.
    """

    def __init__(self):
        self._offline = OfflineRouter()

   def get_route(
    self,
    origin_lat:  float,
    origin_lon:  float,
    dest_lat:    float,
    dest_lon:    float,
    hosp_id:     str  = "unknown",
    radius_m:    int  = 5000,
    slow_edges:  list = None,
) -> dict:
    log.info(f"OnlineRouter.get_route: ({origin_lat},{origin_lon}) -> ({dest_lat},{dest_lon})")
    
    # Try OSRM FIRST — fast, no graph download needed
    osrm_result = self._osrm_route(origin_lat, origin_lon, dest_lat, dest_lon)
    if osrm_result.get("routing_method") == "osrm_online":
        log.info("OSRM succeeded — returning real road route")
        return osrm_result
    
    # Only try osmnx if OSRM failed
    try:
        import osmnx as ox  # noqa
    except ImportError:
        log.warning("osmnx not installed — using offline fallback")
        return self._offline_fallback(
            origin_lat, origin_lon, dest_lat, dest_lon,
            hosp_id, dest_lat, dest_lon, radius_m, slow_edges
        )

    dist_km = _haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
    r = max(radius_m, int(dist_km * 500 * 1.30), 5000)

    try:
        G = get_graph(hosp_id, dest_lat, dest_lon, origin_lat, origin_lon, r)
        if G is None:
            raise RuntimeError("graph_cache_manager returned None")

        router = HospitalRouter(hosp_id, dest_lat, dest_lon, r)
        router.G = router._add_weights(G)

        result = router.get_route(
            origin_lat, origin_lon, dest_lat, dest_lon,
            slow_edges=slow_edges or [],
        )
        result["routing_method"] = "astar_online"
        return result

    except Exception as e:
        log.warning(f"osmnx routing failed ({e}) — straight line only")
        return self._straight_line(origin_lat, origin_lon, dest_lat, dest_lon)

    def _offline_fallback(
        self,
        origin_lat, origin_lon, dest_lat, dest_lon,
        hosp_id, hosp_lat, hosp_lon, radius_m, slow_edges,
    ) -> dict:
        try:
            result = self._offline.get_route(
                origin_lat, origin_lon, dest_lat, dest_lon,
                hosp_id=hosp_id, hosp_lat=hosp_lat, hosp_lon=hosp_lon,
                radius_m=radius_m,
                slow_edges=slow_edges or [],
            )
            result["routing_method"] = "astar_offline_fallback"
            return result
        except Exception as e:
            log.warning(f"Offline fallback failed ({e}) — straight line")
            return self._straight_line(origin_lat, origin_lon, dest_lat, dest_lon)

    def _osrm_route(self, lat1, lon1, lat2, lon2) -> dict:
    try:
        import requests
        url = (
            f"https://router.project-osrm.org/route/v1/driving/"
            f"{lon1},{lat1};{lon2},{lat2}"
            f"?overview=full&geometries=geojson&steps=false"
        )
        log.info(f"Calling OSRM: {url}")
        resp = requests.get(url, timeout=8)
        data = resp.json()

        if data.get("code") != "Ok":
            raise ValueError(f"OSRM code: {data.get('code')}")

        coords = [[c[1], c[0]] for c in data["routes"][0]["geometry"]["coordinates"]]
        dist = _haversine_km(lat1, lon1, lat2, lon2)
        mult = _estimate_congestion("primary")
        log.info(f"OSRM success: {len(coords)} waypoints")

        return {
            "success":            True,
            "routing_method":     "osrm_online",
            "route_coords":       coords,
            "distance_km":        round(dist, 3),
            "travel_time_min":    round(dist / 40 * 60 * mult, 2),
            "raw_waypoints":      len(coords),
            "map_waypoints":      len(coords),
            "road_types_used":    ["primary"],
            "lanes_used":         True,
            "jam_detected":       False,
            "jam_confidence":     0.0,
            "traffic_multiplier": round(mult, 2),
            "traffic_period":     _get_traffic_period(),
            "jam_points":         [],
        }
    except Exception as e:
        log.warning(f"OSRM failed ({e}) — straight line")
        return self._straight_line(lat1, lon1, lat2, lon2)

    def _straight_line(self, lat1, lon1, lat2, lon2) -> dict:
        dist = _haversine_km(lat1, lon1, lat2, lon2)
        mult = _estimate_congestion("primary")
        eta  = dist / 40 * 60 * mult
        return {
            "success":            True,
            "routing_method":     "straight_line_fallback",
            "route_coords":       [[lat1, lon1], [lat2, lon2]],
            "distance_km":        round(dist, 3),
            "travel_time_min":    round(eta, 2),
            "raw_waypoints":      2, "map_waypoints": 2,
            "road_types_used":    [],
            "lanes_used":         False,
            "jam_detected":       mult >= JAM_DETECTION_MULTIPLIER,
            "jam_confidence":     round(min((mult - 1.0) / 1.5, 1.0), 2),
            "traffic_multiplier": round(mult, 2),
            "traffic_period":     _get_traffic_period(),
            "jam_points":         [],
        }
