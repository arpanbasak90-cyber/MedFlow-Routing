"""
routing_engine.py — Smart Routing Engine v5.1
===============================================
v5.1 changes
------------
• reroute_to_same_hospital() docstring updated: slow_edges now always
  contains the ambulance position as the first jam point (prepended by
  main.py).  No logic change needed here — the offline router handles
  the _jam_is_ahead guard and jam radius estimation internally.
• All other logic unchanged from v5.0.
"""

from __future__ import annotations

import logging
from utils import detect_connectivity, haversine_km

log = logging.getLogger(__name__)

KOLKATA_CENTER = (22.5726, 88.3639)
DEFAULT_RADIUS = 5000


class RoutingEngine:

    def __init__(
        self,
        city_center_lat: float = KOLKATA_CENTER[0],
        city_center_lon: float = KOLKATA_CENTER[1],
        radius_m:        int   = DEFAULT_RADIUS,
        force_offline:   bool  = False,
    ):
        self.city_center_lat = city_center_lat
        self.city_center_lon = city_center_lon
        self.radius_m        = radius_m
        self.force_offline   = force_offline
        self._online_router  = None
        self._offline_router = None

    # ── lazy loaders ──────────────────────────────────────────────

    def _get_online_router(self):
        if self._online_router is None:
            from online_router import OnlineRouter
            self._online_router = OnlineRouter()
        return self._online_router

    def _get_offline_router(self):
        if self._offline_router is None:
            from offline_router import OfflineRouter
            self._offline_router = OfflineRouter()
        return self._offline_router

    def _is_online(self) -> bool:
        if self.force_offline:
            return False
        online = detect_connectivity()
        log.info(f"Routing mode: {'ONLINE' if online else 'OFFLINE (cached graph)'}")
        return online

    # ── single route to ONE hospital ─────────────────────────────

    def compute_route(
        self,
        origin_lat:  float,
        origin_lon:  float,
        dest_lat:    float,
        dest_lon:    float,
        hosp_id:     str  = "unknown",
        slow_edges:  list = None,
    ) -> dict:
        """
        Compute the fastest road route from origin to a single hospital.
        slow_edges = list of [lat, lon] jam epicentre coordinates.
        """
        dist_km  = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
        radius_m = max(self.radius_m, int(dist_km * 1000 * 0.7), 4000)

        if slow_edges:
            log.info(f"Jam points present ({len(slow_edges)}) — offline router")
            return self._get_offline_router().get_route(
                origin_lat, origin_lon, dest_lat, dest_lon,
                hosp_id=hosp_id, hosp_lat=dest_lat, hosp_lon=dest_lon,
                radius_m=radius_m,
                slow_edges=slow_edges,
            )

        if self._is_online():
            return self._get_online_router().get_route(
                origin_lat, origin_lon, dest_lat, dest_lon,
                hosp_id=hosp_id, radius_m=radius_m,
                slow_edges=[],
            )
        else:
            return self._get_offline_router().get_route(
                origin_lat, origin_lon, dest_lat, dest_lon,
                hosp_id=hosp_id, hosp_lat=dest_lat, hosp_lon=dest_lon,
                radius_m=radius_m,
                slow_edges=[],
            )

    # ── route to the ONE selected hospital ───────────────────────

    def route_to_selected_hospital(
        self,
        ambulance_lat: float,
        ambulance_lon: float,
        hospital:      dict,
        slow_edges:    list = None,
    ) -> dict:
        hosp_id  = hospital["id"]
        dest_lat = hospital["latitude"]
        dest_lon = hospital["longitude"]

        log.info(f"Computing route → {hospital['name']} (id={hosp_id})")

        route = self.compute_route(
            ambulance_lat, ambulance_lon,
            dest_lat, dest_lon,
            hosp_id=hosp_id,
            slow_edges=slow_edges or [],
        )

        return self._merge_route_into_hospital(hospital, route)

    # ── reroute to SAME hospital ──────────────────────────────────

    def reroute_to_same_hospital(
        self,
        hospital:     dict,
        new_amb_lat:  float,
        new_amb_lon:  float,
        slow_edges:   list = None,
    ) -> dict:
        """
        Reroute from a NEW ambulance position to the SAME locked hospital.

        slow_edges = list of [lat, lon] jam epicentre points.

        The first element of slow_edges is ALWAYS the ambulance's current
        position (prepended by main.py before this call).  The offline
        router uses that point as the primary jam epicentre, estimates its
        radius from the road class + traffic period, applies a weight
        penalty to all edges inside that radius, then runs the two-mode
        dynamic reroute (MODE A if < 1.2 km remaining, MODE B otherwise).

        The _jam_is_ahead() guard in HospitalRouter ensures rerouting only
        fires when the jam is actually between the ambulance and the
        hospital — it will not reroute for jams already passed.
        """
        hosp_id  = hospital["id"]
        dest_lat = hospital["latitude"]
        dest_lon = hospital["longitude"]

        log.info(f"Rerouting → same hospital: {hospital['name']} "
                 f"from ({new_amb_lat:.4f},{new_amb_lon:.4f})")

        dist_km  = haversine_km(new_amb_lat, new_amb_lon, dest_lat, dest_lon)
        radius_m = max(self.radius_m, int(dist_km * 1000 * 0.7), 4000)

        route = self._get_offline_router().reroute(
            new_amb_lat, new_amb_lon,
            dest_lat, dest_lon,
            hosp_id=hosp_id,
            hosp_lat=dest_lat,
            hosp_lon=dest_lon,
            radius_m=radius_m,
            slow_edges=slow_edges or [],
        )

        updated = self._merge_route_into_hospital(hospital, route)
        updated["remaining_km"] = route.get("remaining_km", 0.0)
        return updated

    # ── helper: merge route dict into hospital dict ───────────────

    @staticmethod
    def _merge_route_into_hospital(hospital: dict, route: dict) -> dict:
        updated = dict(hospital)
        updated["route"]              = route.get("route_coords",       [])
        updated["distance_km"]        = route.get("distance_km",        hospital["distance_km"])
        updated["travel_time_min"]    = route.get("travel_time_min",    hospital["travel_time_min"])
        updated["routing_method"]     = route.get("routing_method",     "unknown")
        updated["lanes_used"]         = route.get("lanes_used",         False)
        updated["road_types_used"]    = route.get("road_types_used",    [])
        updated["raw_waypoints"]      = route.get("raw_waypoints",      0)
        updated["map_waypoints"]      = route.get("map_waypoints",      0)
        updated["jam_detected"]       = route.get("jam_detected",       False)
        updated["jam_confidence"]     = route.get("jam_confidence",     0.0)
        updated["traffic_multiplier"] = route.get("traffic_multiplier", 1.0)
        updated["traffic_period"]     = route.get("traffic_period",     "unknown")
        updated["tier"]               = route.get("tier",               "medium")
        updated["snapped_origin"]     = route.get("snapped_origin",     [])
        updated["snapped_dest"]       = route.get("snapped_dest",       [])
        updated["jam_points"]         = route.get("jam_points",         [])
        updated["reroute_mode"]       = route.get("reroute_mode",       None)
        updated["remaining_km"]       = route.get("remaining_km",       None)
        return updated

    # ── legacy helper ─────────────────────────────────────────────

    def reroute_if_blocked(
        self,
        origin_lat:  float,
        origin_lon:  float,
        dest_lat:    float,
        dest_lon:    float,
        slow_edges:  list = None,
        hosp_id:     str  = "unknown",
    ) -> dict:
        """Legacy shim — now routes via jam points only."""
        log.info("Legacy reroute_if_blocked → jam-point reroute")
        return self._get_offline_router().reroute(
            origin_lat, origin_lon, dest_lat, dest_lon,
            hosp_id=hosp_id, hosp_lat=dest_lat, hosp_lon=dest_lon,
            slow_edges=slow_edges or [],
        )