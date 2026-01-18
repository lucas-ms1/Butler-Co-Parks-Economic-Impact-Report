"""
V2 park-exposure measurement: Euclidean distance to park BOUNDARIES (polygons).

Primary: MetroParks-only boundary distance (headline housing exposure).
Robustness: All-parks boundary distance (appendix).

Inputs:
- data_raw/ButlerParks_NDVI_mean_S2_2025_JunAug.csv (NDVI/OSM polygons)
- data_raw/CURRENTPARCELS/CURRENTPARCELS.shp (parcel polygons)

Outputs:
- data_intermediate/v2/parcel_boundary_distances_v2.csv
  (PIN + boundary distances and distance bands for MetroParks + all parks)
"""

from __future__ import annotations

import json
import os
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely.strtree import STRtree


METROPARK_NAMES = {
    "Rentschler Forest",
    "Voice of America",
    "Gilmore MetroPark",
    "Forest Run",
    "Indian Creek",
    "Governor Bebb",
    "Sebald / Elk Creek",
    "Chrisholm Historic Farmstead",
    "Dudley Woods",
    "Bicentennial Commons",
    "Meadow Ridge",
    "Four Mile Creek",
    "Woodsdale MetroPark",
}


def load_parks_polygons(ndvi_csv_path: str, target_crs: Any) -> gpd.GeoDataFrame:
    df = pd.read_csv(ndvi_csv_path, low_memory=False)
    df = df[df[".geo"].notna()].copy()
    df["geometry"] = df[".geo"].apply(lambda g: shape(json.loads(g)))
    gdf = gpd.GeoDataFrame(df.drop(columns=[".geo"]), geometry="geometry", crs="EPSG:4326").to_crs(target_crs)

    # Keep only polygons tagged as parks when those fields exist
    if "leisure" in gdf.columns or "landuse" in gdf.columns:
        leisure = gdf["leisure"].fillna("") if "leisure" in gdf.columns else ""
        landuse = gdf["landuse"].fillna("") if "landuse" in gdf.columns else ""
        if "leisure" in gdf.columns and "landuse" in gdf.columns:
            gdf = gdf[(leisure == "park") | (landuse == "park")].copy()
        elif "leisure" in gdf.columns:
            gdf = gdf[leisure == "park"].copy()
        else:
            gdf = gdf[landuse == "park"].copy()

    # Drop empty geometries
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def filter_metroparks(parks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if parks_gdf.empty:
        return parks_gdf

    operator = parks_gdf["operator"].fillna("").str.lower() if "operator" in parks_gdf.columns else pd.Series("", index=parks_gdf.index)
    name = parks_gdf["name"].fillna("") if "name" in parks_gdf.columns else pd.Series("", index=parks_gdf.index)

    metro_mask = operator.str.contains("metroparks of butler county", na=False) | name.isin(METROPARK_NAMES)
    return parks_gdf[metro_mask].copy()


def assign_dist_band(dist_miles: float) -> str:
    if dist_miles <= 0.1:
        return "0-0.1"
    if dist_miles <= 0.25:
        return "0.1-0.25"
    if dist_miles <= 0.75:
        return "0.25-0.75"
    if dist_miles <= 1.5:
        return "0.75-1.5"
    if dist_miles <= 3:
        return "1.5-3"
    if dist_miles <= 5:
        return "3-5"
    if dist_miles <= 10:
        return "5-10"
    return ">10"


def compute_boundary_distances(parcels: gpd.GeoDataFrame, parks: gpd.GeoDataFrame) -> list[float]:
    if parks.empty:
        return [np.nan] * len(parcels)

    park_geoms = list(parks.geometry)
    tree = STRtree(park_geoms)
    distances_ft: list[float] = []

    for geom in parcels.geometry:
        if geom is None or geom.is_empty:
            distances_ft.append(np.nan)
            continue
        nearest = tree.nearest(geom)
        nearest_geom = park_geoms[int(nearest)] if isinstance(nearest, (int, np.integer)) else nearest
        distances_ft.append(float(geom.distance(nearest_geom)))

    return distances_ft


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    ndvi_parks_path = os.path.join(project_root, "data_raw", "ButlerParks_NDVI_mean_S2_2025_JunAug.csv")
    parcels_path = os.path.join(project_root, "data_raw", "CURRENTPARCELS", "CURRENTPARCELS.shp")
    output_path = os.path.join(project_root, "data_intermediate", "v2", "parcel_boundary_distances_v2.csv")

    print("=" * 80)
    print("V2 PARK EXPOSURE: DISTANCE TO PARK BOUNDARIES (METROPARKS + ALL PARKS)")
    print("=" * 80)

    print(f"\nLoading parcels: {parcels_path}")
    parcels = gpd.read_file(parcels_path)
    print(f"  Parcels loaded: {len(parcels):,}")
    print(f"  Parcels CRS: {parcels.crs}")

    if "PIN" not in parcels.columns:
        raise ValueError("PIN column not found in parcels layer.")

    # Filter residential
    land_use_col = None
    for col in ["CLASS", "LANDUSE", "LUC"]:
        if col in parcels.columns:
            land_use_col = col
            break
    if land_use_col:
        parcels = parcels[parcels[land_use_col] == "R"].copy()
    print(f"  Residential parcels: {len(parcels):,}")

    target_crs = parcels.crs

    print(f"\nLoading NDVI/OSM park polygons: {ndvi_parks_path}")
    parks_all = load_parks_polygons(ndvi_parks_path, target_crs=target_crs)
    print(f"  Park polygons loaded (all parks): {len(parks_all):,}")

    parks_metro = filter_metroparks(parks_all)
    print(f"  MetroParks polygons: {len(parks_metro):,}")

    print("\nComputing nearest-boundary Euclidean distances...")
    dist_all_ft = compute_boundary_distances(parcels, parks_all)
    dist_metro_ft = compute_boundary_distances(parcels, parks_metro)

    out = pd.DataFrame(
        {
            "PIN": parcels["PIN"].astype(str).values,
            "dist_to_park_boundary_ft_all": dist_all_ft,
            "dist_to_park_boundary_ft_metro": dist_metro_ft,
        }
    )
    out["dist_to_park_boundary_miles_all"] = out["dist_to_park_boundary_ft_all"] / 5280.0
    out["dist_to_park_boundary_miles_metro"] = out["dist_to_park_boundary_ft_metro"] / 5280.0

    out["NearPark_Boundary_All"] = (out["dist_to_park_boundary_miles_all"] <= 1.0).astype(int)
    out["NearPark_Boundary_Metro"] = (out["dist_to_park_boundary_miles_metro"] <= 1.0).astype(int)

    out["DistBand_Boundary_All"] = out["dist_to_park_boundary_miles_all"].apply(assign_dist_band)
    out["DistBand_Boundary_Metro"] = out["dist_to_park_boundary_miles_metro"].apply(assign_dist_band)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"\nWrote: {output_path}")
    print("Distance (miles) summary, all parks:")
    print(out["dist_to_park_boundary_miles_all"].describe())
    print("Distance (miles) summary, MetroParks:")
    print(out["dist_to_park_boundary_miles_metro"].describe())


if __name__ == "__main__":
    main()
