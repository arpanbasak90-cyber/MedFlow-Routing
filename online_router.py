"""
online_router.py — Online A* Router v5.0
=========================================
v5.0 changes
------------
• blocked_edges removed entirely.
• slow_edges now accepts a list of [lat, lon] jam point pairs
  (each pair is expanded to a radius by apply_jam_point in offline_router).
• Falls back gracefully: OnlineRouter → OfflineRouter → straight-line.
• get_current_traffic_summary() re-exported for frontend use.
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
    Falls back to OfflineRouter then straight-line on any error.
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
        slow_edges:  list = None,   # list of [lat, lon] jam points
    ) -> dict:
        try:
            import osmnx as ox  # noqa: F401 — availability check
        except ImportError:
            log.warning("osmnx not installed — straight-line fallback")
            return self._straight_line(origin_lat, origin_lon, dest_lat, dest_lon)

        dist_km = _haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
        r       = max(radius_m, int(dist_km * 500 * 1.30), 5000)

        try:
            G = get_graph(hosp_id, dest_lat, dest_lon,
                          origin_lat, origin_lon, r)
            if G is None:
                raise RuntimeError("graph_cache_manager returned None")

            router   = HospitalRouter(hosp_id, dest_lat, dest_lon, r)
            router.G = router._add_weights(G)

            result = router.get_route(
                origin_lat, origin_lon, dest_lat, dest_lon,
                slow_edges=slow_edges or [],
            )
            result["routing_method"] = "astar_online"
            return result

        except Exception as e:
            log.warning(f"Online routing failed ({e}) — offline fallback")
            return self._offline_fallback(
                origin_lat, origin_lon, dest_lat, dest_lon,
                hosp_id, dest_lat, dest_lon, r, slow_edges,
            )

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

    def _straight_line(self, lat1, lon1, lat2, lon2) -> dict:
        dist = _haversine_km(lat1, lon1, lat2, lon2)
        mult = _estimate_congestion("primary")
        eta  = dist / 40 * 60 * mult
        return {
            "success":            True,
            "routing_method":     "straight_line_fallback",
            "route_coords":       [[lat1,lon1],[lat2,lon2]],
            "distance_km":        round(dist, 3),
            "travel_time_min":    round(eta, 2),
            "raw_waypoints":      2, "map_waypoints": 2,
            "road_types_used":    [],
            "lanes_used":         False,
            "jam_detected":       mult >= JAM_DETECTION_MULTIPLIER,
            "jam_confidence":     round(min((mult-1.0)/1.5, 1.0), 2),
            "traffic_multiplier": round(mult, 2),
            "traffic_period":     _get_traffic_period(),
            "jam_points":         [],
        }