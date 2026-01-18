"""
Tract Park Exposure Calculation Script
Economic Impact Report for Butler County Parks

This script calculates park exposure metrics for census tracts by computing
the distance from each tract's centroid to the nearest park and creating
a within_1_mile indicator.
"""

import geopandas as gpd
import pandas as pd
import os

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input paths
health_tracts_path = os.path.join(project_root, 'data_intermediate', 'butler_health_tracts.gpkg')
parks_path = os.path.join(project_root, 'data_intermediate', 'butler_county_parks.shp')

# Output path
output_dir = os.path.join(project_root, 'data_intermediate')
output_path = os.path.join(output_dir, 'butler_health_tracts_park_exposure.gpkg')

print("="*60)
print("TRACT PARK EXPOSURE CALCULATION")
print("="*60)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 1. Load health tracts GeoDataFrame
print(f"\n1. Loading health tracts data...")
print(f"   Path: {health_tracts_path}")
health_tracts_gdf = gpd.read_file(health_tracts_path)
print(f"   Tracts loaded: {len(health_tracts_gdf):,}")
print(f"   CRS: {health_tracts_gdf.crs}")

# 2. Load parks GeoDataFrame
print(f"\n2. Loading parks data...")
print(f"   Path: {parks_path}")
parks_gdf = gpd.read_file(parks_path)
print(f"   Parks loaded: {len(parks_gdf):,}")
print(f"   CRS: {parks_gdf.crs}")

# 3. Ensure CRS consistency
print(f"\n3. Ensuring CRS consistency...")
target_crs = 'EPSG:3402'  # Ohio State Plane South (US Survey Feet)

# Check health tracts CRS
if str(health_tracts_gdf.crs) != target_crs:
    print(f"   Projecting health tracts from {health_tracts_gdf.crs} to {target_crs}...")
    health_tracts_gdf = health_tracts_gdf.to_crs(target_crs)
    print(f"   Health tracts CRS: {health_tracts_gdf.crs}")
else:
    print(f"   Health tracts already in {target_crs}")

# Check parks CRS
if str(parks_gdf.crs) != target_crs:
    print(f"   Projecting parks from {parks_gdf.crs} to {target_crs}...")
    parks_gdf = parks_gdf.to_crs(target_crs)
    print(f"   Parks CRS: {parks_gdf.crs}")
else:
    print(f"   Parks already in {target_crs}")

# 4. Compute tract centroids
print(f"\n4. Computing tract centroids...")
# Create a copy to work with
tracts_with_exposure = health_tracts_gdf.copy()

# Compute centroids
tracts_with_exposure['centroid'] = tracts_with_exposure.geometry.centroid
print(f"   Centroids computed for {len(tracts_with_exposure):,} tracts")

# 5. Calculate distance to nearest park for each tract
print(f"\n5. Calculating distance to nearest park...")
print(f"   Computing distances from tract centroids to parks...")

# For each tract centroid, find the distance to the nearest park
# Using geopandas distance method which is efficient
tracts_with_exposure['dist_to_park_ft'] = tracts_with_exposure['centroid'].apply(
    lambda geom: parks_gdf.geometry.distance(geom).min()
)

print(f"   Distance calculation complete.")
print(f"\n   Distance statistics (feet):")
print(tracts_with_exposure['dist_to_park_ft'].describe())

# 6. Convert distance to miles
print(f"\n6. Converting distance to miles...")
# EPSG:3402 uses US Survey Feet, convert to miles (1 mile = 5280 feet)
tracts_with_exposure['dist_to_park_miles'] = tracts_with_exposure['dist_to_park_ft'] / 5280.0

print(f"   Distance statistics (miles):")
print(tracts_with_exposure['dist_to_park_miles'].describe())

# 7. Create within_1_mile indicator
print(f"\n7. Creating within_1_mile indicator...")
tracts_with_exposure['within_1_mile'] = (tracts_with_exposure['dist_to_park_miles'] <= 1.0).astype(int)

print(f"\n   within_1_mile distribution:")
within_counts = tracts_with_exposure['within_1_mile'].value_counts().sort_index()
print(f"   0 (outside 1 mile): {within_counts.get(0, 0):,} tracts")
print(f"   1 (within 1 mile): {within_counts.get(1, 0):,} tracts")
print(f"   Percentage within 1 mile: {tracts_with_exposure['within_1_mile'].mean() * 100:.1f}%")

# 8. Clean up temporary centroid column
print(f"\n8. Cleaning up temporary columns...")
tracts_with_exposure = tracts_with_exposure.drop(columns=['centroid'])

# 9. Display summary information
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Final GeoDataFrame:")
print(f"  Records: {len(tracts_with_exposure):,}")
print(f"  CRS: {tracts_with_exposure.crs}")
print(f"  Columns: {len(tracts_with_exposure.columns)}")
print(f"\n  New columns added:")
print(f"    - dist_to_park_ft: Distance in feet")
print(f"    - dist_to_park_miles: Distance in miles")
print(f"    - within_1_mile: Binary indicator (1 if <= 1 mile, 0 otherwise)")

# Show sample of results
print(f"\n  Sample results:")
sample_cols = ['GEOID', 'NAME', 'dist_to_park_miles', 'within_1_mile']
if all(col in tracts_with_exposure.columns for col in sample_cols):
    print(tracts_with_exposure[sample_cols].head(10).to_string(index=False))

# 10. Save to GPKG
print(f"\n9. Saving to GeoPackage...")
print(f"   Output path: {output_path}")

tracts_with_exposure.to_file(output_path, driver='GPKG', layer='butler_health_tracts_park_exposure')
print(f"   [OK] Saved successfully!")

print(f"\n{'='*60}")
print("TRACT PARK EXPOSURE CALCULATION COMPLETE")
print(f"{'='*60}")
