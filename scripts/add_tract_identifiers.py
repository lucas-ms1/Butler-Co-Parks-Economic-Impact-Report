"""
Attach census tract GEOID to each housing sale observation for clustered inference.

Why:
- Hedonic housing residuals are spatially correlated; HC1 SEs understate uncertainty.
- Clustering at a neighborhood geography (census tract) is a minimum defensible fix.

Input:
- data_final/housing_regression_ready.csv  (sales sample; must contain PIN)
- data_raw/CURRENTPARCELS/CURRENTPARCELS.shp (parcel polygons; contains PIN + geometry)
- data_intermediate/butler_health_tracts.gpkg (tract polygons; contains GEOID + geometry)

Output:
- data_final/housing_regression_ready_with_tract.csv (adds GEOID)
- results/sales_pin_to_geoid.csv (audit crosswalk)
"""

from __future__ import annotations

import os

import geopandas as gpd
import pandas as pd


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    sales_path = os.path.join(project_root, "data_final", "housing_regression_ready.csv")
    parcels_path = os.path.join(project_root, "data_raw", "CURRENTPARCELS", "CURRENTPARCELS.shp")
    tracts_path = os.path.join(project_root, "data_intermediate", "butler_health_tracts.gpkg")

    output_path = os.path.join(project_root, "data_final", "housing_regression_ready_with_tract.csv")
    crosswalk_path = os.path.join(project_root, "results", "sales_pin_to_geoid.csv")

    print("=" * 80)
    print("ADD TRACT IDENTIFIERS TO HOUSING SALES (GEOID)")
    print("=" * 80)

    print(f"\nLoading sales data: {sales_path}")
    sales_df = pd.read_csv(sales_path, low_memory=False)
    if "PIN" not in sales_df.columns:
        raise ValueError("Expected 'PIN' column in sales data but it was not found.")
    sales_df["PIN"] = sales_df["PIN"].astype(str)
    sales_pins = set(sales_df["PIN"].unique())
    print(f"  Sales rows: {len(sales_df):,}")
    print(f"  Unique PINs: {len(sales_pins):,}")

    print(f"\nLoading tract polygons: {tracts_path}")
    tracts_gdf = gpd.read_file(tracts_path)
    if "GEOID" not in tracts_gdf.columns:
        raise ValueError("Expected 'GEOID' column in tract polygons but it was not found.")
    tracts_gdf["GEOID"] = tracts_gdf["GEOID"].astype(str)
    target_crs = tracts_gdf.crs
    print(f"  Tracts: {len(tracts_gdf):,}")
    print(f"  Tracts CRS: {target_crs}")

    print(f"\nLoading parcel polygons: {parcels_path}")
    # Try to load only needed columns when supported
    try:
        parcels_gdf = gpd.read_file(parcels_path, columns=["PIN", "geometry"])
    except TypeError:
        parcels_gdf = gpd.read_file(parcels_path)[["PIN", "geometry"]]
    parcels_gdf["PIN"] = parcels_gdf["PIN"].astype(str)
    print(f"  Parcels loaded: {len(parcels_gdf):,}")
    print(f"  Parcels CRS: {parcels_gdf.crs}")

    # Filter parcels to sales PINs to reduce join size
    parcels_sales = parcels_gdf[parcels_gdf["PIN"].isin(sales_pins)].copy()
    print(f"  Parcels matched to sales PINs: {len(parcels_sales):,}")

    # Project to tract CRS for spatial join
    if parcels_sales.crs != target_crs:
        print(f"\nProjecting parcels to tract CRS ({target_crs}) for spatial join...")
        parcels_sales = parcels_sales.to_crs(target_crs)

    # Use parcel centroids for a clean, one-to-one tract assignment
    parcels_sales["geometry"] = parcels_sales.geometry.centroid

    print("\nSpatially joining parcel centroids to census tracts...")
    parcels_with_tract = gpd.sjoin(
        parcels_sales,
        tracts_gdf[["GEOID", "geometry"]],
        how="left",
        predicate="intersects",
    )

    pin_to_geoid = parcels_with_tract[["PIN", "GEOID"]].drop_duplicates()
    matched = pin_to_geoid["GEOID"].notna().sum()
    print(f"  PIN->GEOID matches: {matched:,} / {len(pin_to_geoid):,}")

    # Write audit crosswalk
    os.makedirs(os.path.dirname(crosswalk_path), exist_ok=True)
    pin_to_geoid.to_csv(crosswalk_path, index=False)
    print(f"  Wrote crosswalk: {crosswalk_path}")

    # Merge back to sales
    sales_with_geoid = sales_df.merge(pin_to_geoid, on="PIN", how="left")
    sales_matched = sales_with_geoid["GEOID"].notna().sum()
    print(f"\nSales GEOID coverage: {sales_matched:,} / {len(sales_with_geoid):,}")

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sales_with_geoid.to_csv(output_path, index=False)
    print(f"Wrote updated sales file: {output_path}")


if __name__ == "__main__":
    main()

