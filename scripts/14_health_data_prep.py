"""
Health Data Preparation Script
Economic Impact Report for Butler County Parks

This script loads CDC PLACES Census Tract data and TIGER/Line Census Tract
shapefiles, filters both to Butler County, ensures CRS consistency, and
outputs a combined GeoDataFrame.
"""

import geopandas as gpd
import pandas as pd
import zipfile
import os
import tempfile
import shutil

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input paths
places_csv_path = os.path.join(
    project_root, 
    'data_raw', 
    'Health and Census Tracts',
    'PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260107.csv'
)
tiger_zip_path = os.path.join(
    project_root,
    'data_raw',
    'Health and Census Tracts',
    'tl_2025_39_tract.zip'
)

# Output path
output_dir = os.path.join(project_root, 'data_intermediate')
output_path = os.path.join(output_dir, 'butler_health_tracts.gpkg')

print("="*60)
print("HEALTH DATA PREPARATION")
print("="*60)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 1. Load CDC PLACES CSV data
print(f"\n1. Loading CDC PLACES CSV data...")
print(f"   Path: {places_csv_path}")
places_df = pd.read_csv(places_csv_path, low_memory=False)
print(f"   Total records loaded: {len(places_df):,}")

# Filter to Ohio Butler County
# CountyFIPS == 39017 (Ohio state code 39 + Butler County 017)
# OR CountyName == "Butler" AND StateAbbr == "OH"
print(f"\n   Filtering to Ohio Butler County...")
butler_places = places_df[
    (places_df['CountyFIPS'] == 39017) | 
    ((places_df['CountyName'] == 'Butler') & (places_df['StateAbbr'] == 'OH'))
].copy()
print(f"   Butler County records: {len(butler_places):,}")

if len(butler_places) == 0:
    raise ValueError("No Butler County records found in PLACES data!")

print(f"   Sample TractFIPS: {butler_places['TractFIPS'].head(3).tolist()}")

# 2. Load TIGER/Line shapefile from zip
print(f"\n2. Loading TIGER/Line shapefile...")
print(f"   Path: {tiger_zip_path}")

# Extract zip to temporary directory
temp_dir = tempfile.mkdtemp()
try:
    with zipfile.ZipFile(tiger_zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find the shapefile (should be tl_2025_39_tract.shp)
    shapefile_path = os.path.join(temp_dir, 'tl_2025_39_tract.shp')
    
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found in zip: {shapefile_path}")
    
    # Load the shapefile
    tracts_gdf = gpd.read_file(shapefile_path)
    print(f"   Total tracts loaded: {len(tracts_gdf):,}")
    print(f"   CRS: {tracts_gdf.crs}")
    
    # Filter to Butler County (COUNTYFP == "017")
    print(f"\n   Filtering to Butler County (COUNTYFP == '017')...")
    butler_tracts = tracts_gdf[tracts_gdf['COUNTYFP'] == '017'].copy()
    print(f"   Butler County tracts: {len(butler_tracts):,}")
    
    if len(butler_tracts) == 0:
        raise ValueError("No Butler County tracts found in TIGER/Line data!")
    
    print(f"   Sample GEOID: {butler_tracts['GEOID'].head(3).tolist()}")
    
finally:
    # Clean up temporary directory
    shutil.rmtree(temp_dir)

# 3. Join PLACES data with TIGER/Line geometries
print(f"\n3. Joining PLACES data with TIGER/Line geometries...")
print(f"   Joining on TractFIPS (PLACES) == GEOID (TIGER/Line)...")

# Convert TractFIPS to string for matching (it's numeric in CSV)
butler_places['TractFIPS_str'] = butler_places['TractFIPS'].astype(str)
butler_tracts['GEOID_str'] = butler_tracts['GEOID'].astype(str)

# Perform the join
butler_health_gdf = butler_tracts.merge(
    butler_places,
    left_on='GEOID_str',
    right_on='TractFIPS_str',
    how='inner'
)

print(f"   Joined records: {len(butler_health_gdf):,}")

if len(butler_health_gdf) == 0:
    raise ValueError("No matching records after join! Check TractFIPS/GEOID alignment.")

# 4. Ensure CRS consistency
print(f"\n4. Ensuring CRS consistency...")
print(f"   Current CRS: {butler_health_gdf.crs}")

# Check if CRS is set, if not set to EPSG:4326 (WGS84, typical for TIGER/Line)
if butler_health_gdf.crs is None:
    print("   Warning: No CRS detected, setting to EPSG:4326 (WGS84)")
    butler_health_gdf.set_crs('EPSG:4326', inplace=True)
else:
    print(f"   CRS is set: {butler_health_gdf.crs}")

# Optionally project to a consistent CRS for analysis
# Using EPSG:3402 (Ohio State Plane South) as used in other scripts
target_crs = 'EPSG:3402'
print(f"\n   Projecting to {target_crs} for consistency with other data...")
butler_health_gdf = butler_health_gdf.to_crs(target_crs)
print(f"   Projected CRS: {butler_health_gdf.crs}")

# 5. Clean up temporary columns
print(f"\n5. Cleaning up temporary columns...")
if 'TractFIPS_str' in butler_health_gdf.columns:
    butler_health_gdf.drop(columns=['TractFIPS_str'], inplace=True)
if 'GEOID_str' in butler_health_gdf.columns:
    butler_health_gdf.drop(columns=['GEOID_str'], inplace=True)

# 6. Display summary information
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Final GeoDataFrame:")
print(f"  Records: {len(butler_health_gdf):,}")
print(f"  CRS: {butler_health_gdf.crs}")
print(f"  Columns: {len(butler_health_gdf.columns)}")
print(f"\n  Key columns:")
key_cols = ['GEOID', 'NAME', 'COUNTYFP', 'TractFIPS', 'CountyName', 'TotalPopulation']
for col in key_cols:
    if col in butler_health_gdf.columns:
        print(f"    - {col}")

# 7. Save to GPKG
print(f"\n6. Saving to GeoPackage...")
print(f"   Output path: {output_path}")

butler_health_gdf.to_file(output_path, driver='GPKG', layer='butler_health_tracts')
print(f"   [OK] Saved successfully!")

print(f"\n{'='*60}")
print("HEALTH DATA PREPARATION COMPLETE")
print(f"{'='*60}")
