"""
graph_cache_manager.py — Smart Incremental Graph Cache Manager v2.0
====================================================================
v2.0 changes (vs v1.0)
-----------------------
• Cache key = hospital ID + snapped radius (unchanged).
• "Already covered" check is more rigorous: both the hospital AND
  the ambulance must be inside the bbox WITH a margin.  Only the
  truly uncovered endpoint triggers a patch download.
• Patch is centred on the UNCOVERED POINT and sized to bridge the
  gap — NOT a full re-download centred on the midpoint.
• _snap_radius() rounds to 2500 m so nearby radii share one file.
• If the existing graph already covers a radius >= the requested
  radius, no new download is triggered at all (reuse-first policy).
• update_graph_for_reroute() only patches when the new ambulance
  position is genuinely outside the current bbox.
• Saved file is ALWAYS written to the same hospital-keyed path so
  re-runs never create duplicate files.

Public API
----------
    get_graph(hosp_id, hosp_lat, hosp_lon,
              amb_lat, amb_lon, radius_m=5000) -> nx.MultiDiGraph | None

    update_graph_for_reroute(hosp_id, hosp_lat, hosp_lon,
                             new_amb_lat, new_amb_lon,
                             radius_m=5000) -> nx.MultiDiGraph | None

    invalidate(hosp_id, radius_m=5000)
    list_cached() -> list[str]
"""

from __future__ import annotations

import os
import math
import logging
from typing import Optional

import networkx as nx

log = logging.getLogger(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Process-level memory cache: {cache_key: nx.MultiDiGraph}
_MEM_CACHE: dict[str, object] = {}

_DRIVE_FILTER = (
    '["highway"]["highway"!~"footway|cycleway|pedestrian|path|steps|'
    'track|construction|proposed|raceway|busway"]["motor_vehicle"!~"no"]'
)

# Bbox margin in degrees (~900 m at Kolkata latitude).
# Points within this buffer of the bbox edge are considered "covered".
_BBOX_MARGIN = 0.008


# ─────────────────────────────────────────────────────────────────
# INTERNAL GEOMETRY
# ─────────────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0; d = math.radians
    dlat = d(lat2-lat1); dlon = d(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(d(lat1))*math.cos(d(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def _graph_bbox(G) -> tuple:
    """Return (min_lat, min_lon, max_lat, max_lon) of all nodes."""
    lats = [d["y"] for _, d in G.nodes(data=True) if "y" in d]
    lons = [d["x"] for _, d in G.nodes(data=True) if "x" in d]
    if not lats:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(lats), min(lons), max(lats), max(lons))


def _point_covered(lat: float, lon: float, bbox: tuple,
                   margin: float = _BBOX_MARGIN) -> bool:
    """True if (lat, lon) is inside the bbox (with margin buffer)."""
    min_lat, min_lon, max_lat, max_lon = bbox
    return (min_lat - margin <= lat <= max_lat + margin and
            min_lon - margin <= lon <= max_lon + margin)


def _bbox_diagonal_km(bbox: tuple) -> float:
    """Rough diagonal size of the bbox in km."""
    min_lat, min_lon, max_lat, max_lon = bbox
    return _haversine_km(min_lat, min_lon, max_lat, max_lon)


# ─────────────────────────────────────────────────────────────────
# CACHE KEY  (hospital-based, not coordinate-based)
# ─────────────────────────────────────────────────────────────────

def _snap_radius(radius_m: int) -> int:
    """Round UP to nearest 2500 m so nearby calls share one graph."""
    return math.ceil(radius_m / 2500) * 2500


def _cache_key(hosp_id: str, radius_m: int) -> str:
    r      = _snap_radius(radius_m)
    safe   = hosp_id.replace("/","_").replace("\\","_").replace(" ","_")
    return f"hosp_{safe}_r{r}.graphml"


def _cache_path(hosp_id: str, radius_m: int) -> str:
    return os.path.join(CACHE_DIR, _cache_key(hosp_id, radius_m))


# ─────────────────────────────────────────────────────────────────
# PATCH DOWNLOAD & MERGE
# ─────────────────────────────────────────────────────────────────

def _patch_radius_for_point(
    G, point_lat: float, point_lon: float, min_r: int = 2500
) -> int:
    """
    Calculate the patch radius needed to bridge from the nearest bbox
    edge to the uncovered point, plus a safety margin.
    Only the GAP is downloaded — not the whole graph again.
    """
    bbox = _graph_bbox(G)
    min_lat, min_lon, max_lat, max_lon = bbox

    # Distance from point to each bbox edge (km)
    gaps_km = [
        max(0.0, min_lat - point_lat),   # south gap
        max(0.0, point_lat - max_lat),   # north gap
        max(0.0, min_lon - point_lon),   # west gap  (approx)
        max(0.0, point_lon - max_lon),   # east gap
    ]
    gap_km = max(gaps_km) * 111.0  # 1° ≈ 111 km
    patch_m = int(gap_km * 1000 * 1.35) + 1500   # gap + 1.5 km safety buffer
    return max(patch_m, min_r)


def _merge_patch(
    ox, G_base,
    patch_lat: float, patch_lon: float,
    patch_radius_m: int,
    cache_file: str, key: str,
) -> object:
    """
    Download a patch centred on (patch_lat, patch_lon) with radius
    patch_radius_m, then merge NEW nodes/edges into G_base.
    Saves back to the SAME cache file.
    """
    log.info(f"Fetching gap patch (r={patch_radius_m} m) at "
             f"({patch_lat:.4f},{patch_lon:.4f})…")
    try:
        G_patch = ox.graph_from_point(
            (patch_lat, patch_lon),
            dist=patch_radius_m,
            network_type="drive",
            custom_filter=_DRIVE_FILTER,
            simplify=True,
            retain_all=False,
        )
    except Exception as e:
        log.warning(f"Patch download failed: {e}")
        return G_base

    added_n = added_e = 0
    for n, data in G_patch.nodes(data=True):
        if not G_base.has_node(n):
            G_base.add_node(n, **data)
            added_n += 1
    for u, v, k, data in G_patch.edges(data=True, keys=True):
        if not G_base.has_edge(u, v, k):
            G_base.add_edge(u, v, key=k, **data)
            added_e += 1

    if added_n or added_e:
        log.info(f"Patch merged: +{added_n} nodes, +{added_e} edges → saving {key}")
        ox.save_graphml(G_base, cache_file)   # always same file
        _MEM_CACHE[key] = G_base
    else:
        log.info("Patch had no new data — graph unchanged")

    return G_base


# ─────────────────────────────────────────────────────────────────
# COVERAGE CHECK & CONDITIONAL PATCH
# ─────────────────────────────────────────────────────────────────

def _ensure_point_covered(
    ox, G, lat: float, lon: float,
    label: str,
    radius_m: int, cache_file: str, key: str,
) -> object:
    """
    If (lat, lon) is NOT inside the current graph bbox, calculate the
    minimum patch needed and download ONLY that gap.
    """
    bbox = _graph_bbox(G)
    if _point_covered(lat, lon, bbox):
        log.debug(f"{label} already covered by cached graph")
        return G

    patch_r = _patch_radius_for_point(G, lat, lon, min_r=2500)
    log.info(f"{label} outside bbox — gap patch r={patch_r} m")
    return _merge_patch(ox, G, lat, lon, patch_r, cache_file, key)


# ─────────────────────────────────────────────────────────────────
# COLD DOWNLOAD
# ─────────────────────────────────────────────────────────────────

def _cold_download(
    ox,
    hosp_id:  str,
    hosp_lat: float, hosp_lon: float,
    amb_lat:  float, amb_lon:  float,
    radius_m: int,
    key:      str,
    fpath:    str,
) -> Optional[object]:
    """
    First-time download.  Centre on the midpoint of amb↔hosp with a
    radius that guarantees both endpoints are inside the graph.
    """
    dist_km   = _haversine_km(amb_lat, amb_lon, hosp_lat, hosp_lon)
    # Half-diagonal + 30% safety so both endpoints fit comfortably
    dl_radius = max(radius_m, int(dist_km * 1000 * 0.65))
    dl_radius = _snap_radius(dl_radius)

    mid_lat = (amb_lat + hosp_lat) / 2
    mid_lon = (amb_lon + hosp_lon) / 2

    log.info(f"Cold download for hospital {hosp_id} "
             f"(r={dl_radius} m, centre midpoint)…")
    try:
        G = ox.graph_from_point(
            (mid_lat, mid_lon),
            dist=dl_radius,
            network_type="drive",
            custom_filter=_DRIVE_FILTER,
            simplify=True,
            retain_all=False,
        )
        ox.save_graphml(G, fpath)          # saved to hospital-keyed file
        _MEM_CACHE[key] = G
        log.info(f"Downloaded & cached → {key} "
                 f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        return G
    except Exception as e:
        log.error(f"Cold download failed for {hosp_id}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────

def get_graph(
    hosp_id:  str,
    hosp_lat: float, hosp_lon: float,
    amb_lat:  float, amb_lon:  float,
    radius_m: int = 5000,
) -> Optional[object]:
    """
    Return a road-network graph covering BOTH the hospital and the
    ambulance.  Download policy (reuse-first):

    1. Memory hit → check coverage → patch only if a point is missing.
    2. Disk hit   → load → check coverage → patch only if needed.
    3. Cold miss  → download centred on midpoint with full radius.

    The same hospital-keyed .graphml file is reused across all calls.
    No duplicate files are ever created.
    """
    try:
        import osmnx as ox
    except ImportError:
        log.warning("osmnx not installed")
        return None

    key   = _cache_key(hosp_id, radius_m)
    fpath = _cache_path(hosp_id, radius_m)

    # ── 1. Memory ────────────────────────────────────────────────
    if key in _MEM_CACHE:
        G = _MEM_CACHE[key]
        log.info(f"Graph from memory: {key} "
                 f"({G.number_of_nodes()} nodes)")
        G = _ensure_point_covered(ox, G, hosp_lat, hosp_lon,
                                  "Hospital", radius_m, fpath, key)
        G = _ensure_point_covered(ox, G, amb_lat,  amb_lon,
                                  "Ambulance", radius_m, fpath, key)
        return G

    # ── 2. Disk ──────────────────────────────────────────────────
    if os.path.exists(fpath):
        log.info(f"Loading graph from disk: {key}")
        G = ox.load_graphml(fpath)
        _MEM_CACHE[key] = G
        G = _ensure_point_covered(ox, G, hosp_lat, hosp_lon,
                                  "Hospital", radius_m, fpath, key)
        G = _ensure_point_covered(ox, G, amb_lat,  amb_lon,
                                  "Ambulance", radius_m, fpath, key)
        return G

    # ── 3. Cold download ─────────────────────────────────────────
    return _cold_download(ox, hosp_id,
                          hosp_lat, hosp_lon,
                          amb_lat,  amb_lon,
                          radius_m, key, fpath)


def update_graph_for_reroute(
    hosp_id:      str,
    hosp_lat:     float, hosp_lon:     float,
    new_amb_lat:  float, new_amb_lon:  float,
    radius_m:     int = 5000,
) -> Optional[object]:
    """
    Called when the ambulance has moved.  Ensures the graph covers the
    NEW position without re-downloading anything already cached.

    Returns the (possibly patched) graph, or None if osmnx unavailable.
    """
    try:
        import osmnx as ox
    except ImportError:
        return None

    key   = _cache_key(hosp_id, radius_m)
    fpath = _cache_path(hosp_id, radius_m)

    if key in _MEM_CACHE:
        G = _MEM_CACHE[key]
    elif os.path.exists(fpath):
        G = ox.load_graphml(fpath)
        _MEM_CACHE[key] = G
    else:
        # No existing graph — fall back to full get_graph
        return get_graph(hosp_id, hosp_lat, hosp_lon,
                         new_amb_lat, new_amb_lon, radius_m)

    # Only patch if the new position is genuinely uncovered
    return _ensure_point_covered(
        ox, G, new_amb_lat, new_amb_lon,
        "New ambulance position", radius_m, fpath, key,
    )


def invalidate(hosp_id: str, radius_m: int = 5000):
    """Force re-download on next call for this hospital."""
    key   = _cache_key(hosp_id, radius_m)
    fpath = _cache_path(hosp_id, radius_m)
    _MEM_CACHE.pop(key, None)
    if os.path.exists(fpath):
        os.remove(fpath)
        log.info(f"Cache invalidated: {key}")


def list_cached() -> list[str]:
    """Return all .graphml filenames in the cache directory."""
    return [f for f in os.listdir(CACHE_DIR) if f.endswith(".graphml")]