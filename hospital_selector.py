"""
hospital_selector.py — Hospital Selector v3.1
===============================================
Changes in v3.1
---------------
• After scoring top N hospitals, always appends the NEAREST GOVT
  hospital even if it has no ICU beds — for high-risk conditions
  the patient may need a large govt facility regardless of bed count.
• Each hospital dict now carries a `recommendation` string explaining
  WHY it was selected — shown to the driver to help them choose.
• Nearest GOVT is marked with `nearest_govt_fallback: True` so the
  frontend can highlight it differently.
"""

import os
import logging
import pandas as pd

from utils import haversine_km, score_hospital, get_required_dept, estimate_travel_time

log = logging.getLogger(__name__)

MAX_SEARCH_RADIUS_KM = 30.0
CANDIDATE_POOL_SIZE  = 8

# Conditions considered HIGH RISK — always show nearest GOVT hospital
HIGH_RISK_CONDITIONS = {
    "heart emergency", "stroke risk", "respiratory emergency",
    "moderate risk - possible heart issue",
    "moderate risk - possible stroke",
    "moderate risk - possible respiratory issue",
}


class HospitalSelector:

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._df: pd.DataFrame | None = None
        self._load()

    def _load(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Hospital CSV not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()
        if "hospital_type" in df.columns:
            df["hospital_type"] = df["hospital_type"].str.upper()
        for col in ["has_cardiology","has_neurology","has_pulmonology","has_icu","emergency_24h"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        for col in ["icu_beds_available","general_beds_available","icu_beds_total","general_beds_total"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        self._df = df
        log.info(f"Loaded {len(df)} hospitals from {self.csv_path}")

    def get_best_hospitals(
        self,
        condition:     str,
        ambulance_lat: float,
        ambulance_lon: float,
        total:         int  = 3,
        min_govt:      int  = 1,
    ) -> list[dict]:
        df = self._df.copy()
        df["distance_km"] = df.apply(
            lambda r: haversine_km(ambulance_lat, ambulance_lon,
                                   r["latitude"], r["longitude"]), axis=1,
        )

        in_radius = df[df["distance_km"] <= MAX_SEARCH_RADIUS_KM]
        df = in_radius if len(in_radius) >= total else df

        emerg = df[df.get("emergency_24h", pd.Series(1, index=df.index)) == 1]
        if len(emerg) >= total:
            df = emerg

        dept_col   = get_required_dept(condition)
        dept_match = pd.Series(False, index=df.index)

        if dept_col and dept_col in df.columns:
            dept_filtered = df[df[dept_col] == 1]
            if len(dept_filtered) >= total:
                df         = dept_filtered
                dept_match = pd.Series(True, index=df.index)
            else:
                log.warning(f"Only {len(dept_filtered)} hospitals with '{dept_col}' — relaxing")
                dept_match = df[dept_col] == 1 if dept_col in df.columns else \
                             pd.Series(False, index=df.index)

        df = df.copy()
        df["_dept_match"] = dept_match if dept_col else False
        df["_score"] = df.apply(
            lambda r: score_hospital(r.to_dict(), r["distance_km"], dept_col), axis=1,
        )
        df = df.sort_values("_score").reset_index(drop=True)
        shortlist = self._enforce_govt_quota(df, total, min_govt)
        result    = self._to_hospital_dicts(shortlist, dept_col, ambulance_lat, ambulance_lon)

        # ── Always append nearest GOVT hospital for high-risk ─────
        is_high_risk = any(k in condition.lower() for k in HIGH_RISK_CONDITIONS)
        if is_high_risk:
            result = self._append_nearest_govt(
                result, df, dept_col, ambulance_lat, ambulance_lon
            )

        return result

    def _append_nearest_govt(
        self, result: list, df: pd.DataFrame,
        dept_col: str | None, amb_lat: float, amb_lon: float,
    ) -> list:
        """
        Find the nearest GOVT hospital regardless of bed availability
        and append it if not already in the list.
        Marked with nearest_govt_fallback=True.
        """
        existing_ids = {h["id"] for h in result}
        govt_df = self._df.copy()
        govt_df["distance_km"] = govt_df.apply(
            lambda r: haversine_km(amb_lat, amb_lon, r["latitude"], r["longitude"]), axis=1,
        )
        govt_df = govt_df[govt_df["hospital_type"] == "GOVT"]
        govt_df = govt_df[~govt_df["hospital_id"].isin(existing_ids)]
        if govt_df.empty:
            return result

        govt_df = govt_df.sort_values("distance_km").reset_index(drop=True)
        nearest = govt_df.iloc[0]
        dist    = float(nearest["distance_km"])

        dept_match = False
        if dept_col and dept_col in nearest.index:
            dept_match = int(nearest[dept_col]) == 1

        icu_free  = int(nearest.get("icu_beds_available", 0))
        gen_free  = int(nearest.get("general_beds_available", 0))
        has_beds  = icu_free > 0 or gen_free > 0

        recommendation = self._build_recommendation(
            nearest, dist, dept_match, icu_free, gen_free,
            is_govt_fallback=True, has_beds=has_beds,
        )

        hosp = {
            "rank":                   len(result) + 1,
            "name":                   nearest["hospital_name"],
            "id":                     nearest["hospital_id"],
            "type":                   nearest["hospital_type"],
            "latitude":               float(nearest["latitude"]),
            "longitude":              float(nearest["longitude"]),
            "city":                   nearest.get("city", ""),
            "distance_km":            round(dist, 3),
            "travel_time_min":        estimate_travel_time(dist),
            "icu_beds_available":     icu_free,
            "general_beds_available": gen_free,
            "icu_beds_total":         int(nearest.get("icu_beds_total", 0)),
            "general_beds_total":     int(nearest.get("general_beds_total", 0)),
            "emergency_24h":          bool(int(nearest.get("emergency_24h", 0))),
            "has_cardiology":         bool(int(nearest.get("has_cardiology", 0))),
            "has_neurology":          bool(int(nearest.get("has_neurology", 0))),
            "has_pulmonology":        bool(int(nearest.get("has_pulmonology", 0))),
            "department_match":       dept_match,
            "route":                  [],
            "routing_method":         "haversine_estimate",
            "lanes_used":             False,
            "road_types_used":        [],
            "nearest_govt_fallback":  True,
            "recommendation":         recommendation,
            "no_beds_warning":        not has_beds,
        }
        result.append(hosp)
        return result

    def _enforce_govt_quota(self, df_sorted, total, min_govt):
        selected = df_sorted.head(total).copy()
        n_govt   = (selected["hospital_type"] == "GOVT").sum()
        if n_govt >= min_govt:
            return selected
        selected_ids = set(selected["hospital_id"])
        govt_pool = df_sorted[
            (df_sorted["hospital_type"] == "GOVT") &
            (~df_sorted["hospital_id"].isin(selected_ids))
        ]
        needed = min_govt - n_govt
        for _, govt_row in govt_pool.head(needed).iterrows():
            pvt_mask = selected["hospital_type"] != "GOVT"
            if pvt_mask.any():
                worst_pvt_idx = selected[pvt_mask]["_score"].idxmax()
                selected.loc[worst_pvt_idx] = govt_row
            else:
                selected = pd.concat(
                    [selected, govt_row.to_frame().T], ignore_index=True
                ).head(total)
        return selected.sort_values("_score").reset_index(drop=True)

    def _build_recommendation(
        self, row, dist_km: float, dept_match: bool,
        icu_free: int, gen_free: int,
        is_govt_fallback: bool = False, has_beds: bool = True,
    ) -> str:
        """
        Build a human-readable recommendation string for each hospital.
        This is shown to the driver to help them choose.
        """
        parts = []

        if dist_km < 1.0:
            parts.append(f"Very close ({dist_km:.1f} km)")
        elif dist_km < 3.0:
            parts.append(f"Nearby ({dist_km:.1f} km)")
        else:
            parts.append(f"{dist_km:.1f} km away")

        if dept_match:
            parts.append("has the required specialty department")

        if icu_free > 0:
            parts.append(f"{icu_free} ICU bed{'s' if icu_free>1 else ''} available")
        elif not has_beds:
            parts.append("⚠ no beds currently available")
        elif gen_free > 0:
            parts.append(f"{gen_free} general bed{'s' if gen_free>1 else ''} available")

        if is_govt_fallback:
            parts.append("nearest govt hospital — large capacity facility")
            if not has_beds:
                parts.append("may still accept emergency patients")

        ht = str(row.get("hospital_type","")).upper()
        if ht == "GOVT":
            parts.append("government facility (lower cost)")
        else:
            parts.append("private facility (faster admission likely)")

        return " · ".join(parts)

    def _to_hospital_dicts(self, df, dept_col, amb_lat, amb_lon) -> list[dict]:
        result = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            dist = float(row["distance_km"])
            eta  = estimate_travel_time(dist)
            dept_match = bool(int(row[dept_col])==1) if dept_col and dept_col in row.index else False
            icu_free   = int(row.get("icu_beds_available",0))
            gen_free   = int(row.get("general_beds_available",0))
            has_beds   = icu_free > 0 or gen_free > 0

            recommendation = self._build_recommendation(
                row, dist, dept_match, icu_free, gen_free,
                is_govt_fallback=False, has_beds=has_beds,
            )

            hosp = {
                "rank":                   rank,
                "name":                   row["hospital_name"],
                "id":                     row["hospital_id"],
                "type":                   row["hospital_type"],
                "latitude":               float(row["latitude"]),
                "longitude":              float(row["longitude"]),
                "city":                   row.get("city",""),
                "distance_km":            round(dist,3),
                "travel_time_min":        eta,
                "icu_beds_available":     icu_free,
                "general_beds_available": gen_free,
                "icu_beds_total":         int(row.get("icu_beds_total",0)),
                "general_beds_total":     int(row.get("general_beds_total",0)),
                "emergency_24h":          bool(int(row.get("emergency_24h",0))),
                "has_cardiology":         bool(int(row.get("has_cardiology",0))),
                "has_neurology":          bool(int(row.get("has_neurology",0))),
                "has_pulmonology":        bool(int(row.get("has_pulmonology",0))),
                "department_match":       dept_match,
                "route":                  [],
                "routing_method":         "haversine_estimate",
                "lanes_used":             False,
                "road_types_used":        [],
                "nearest_govt_fallback":  False,
                "recommendation":         recommendation,
                "no_beds_warning":        not has_beds,
            }
            result.append(hosp)
        return result

    def reload(self):
        self._load()