"""
Health Analysis Dataset Creation Script
Economic Impact Report for Butler County Parks

This script merges health tract data with park exposure metrics, ACS economic
covariates, ACS education variables, and ACS population data to create a final
analysis dataset for health outcomes modeling.
"""

import geopandas as gpd
import pandas as pd
import os

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input paths
health_tracts_path = os.path.join(
    project_root,
    'data_intermediate',
    'butler_health_tracts_park_exposure.gpkg'
)
acs_covariates_path = os.path.join(
    project_root,
    'data_intermediate',
    'butler_acs_dp03_covariates.csv'
)
acs_education_path = os.path.join(
    project_root,
    'data_intermediate',
    'butler_acs_dp02_education.csv'
)
acs_population_path = os.path.join(
    project_root,
    'data_intermediate',
    'butler_acs_dp05_population.csv'
)
acs_demographics_path = os.path.join(
    project_root,
    'data_intermediate',
    'butler_acs_dp05_demographics.csv'
)

# Output paths
output_dir = os.path.join(project_root, 'data_final')
output_gpkg_path = os.path.join(output_dir, 'butler_tract_health_model_data.gpkg')
output_csv_path = os.path.join(output_dir, 'butler_tract_health_model_data.csv')

print("="*60)
print("HEALTH ANALYSIS DATASET CREATION")
print("="*60)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 1. Load health tracts with park exposure
print(f"\n1. Loading health tracts with park exposure...")
print(f"   Path: {health_tracts_path}")
health_gdf = gpd.read_file(health_tracts_path)
print(f"   Records loaded: {len(health_gdf):,}")
print(f"   CRS: {health_gdf.crs}")
print(f"   Columns: {len(health_gdf.columns)}")

# Check GEOID column
print(f"\n   GEOID column type: {health_gdf['GEOID'].dtype}")
print(f"   Sample GEOIDs: {health_gdf['GEOID'].head(3).tolist()}")

# 2. Load ACS economic covariates
print(f"\n2. Loading ACS economic covariates...")
print(f"   Path: {acs_covariates_path}")
acs_econ_df = pd.read_csv(acs_covariates_path)
print(f"   Records loaded: {len(acs_econ_df):,}")
print(f"   Columns: {list(acs_econ_df.columns)}")

# Check GEOID column
print(f"\n   GEOID column type: {acs_econ_df['GEOID'].dtype}")
print(f"   Sample GEOIDs: {acs_econ_df['GEOID'].head(3).tolist()}")

# 2b. Load ACS education variables
print(f"\n2b. Loading ACS education variables...")
print(f"   Path: {acs_education_path}")
acs_educ_df = pd.read_csv(acs_education_path)
print(f"   Records loaded: {len(acs_educ_df):,}")
print(f"   Columns: {list(acs_educ_df.columns)}")

# Check GEOID column
print(f"\n   GEOID column type: {acs_educ_df['GEOID'].dtype}")
print(f"   Sample GEOIDs: {acs_educ_df['GEOID'].head(3).tolist()}")

# 2c. Load ACS population data
print(f"\n2c. Loading ACS population data...")
print(f"   Path: {acs_population_path}")
acs_pop_df = pd.read_csv(acs_population_path)
print(f"   Records loaded: {len(acs_pop_df):,}")
print(f"   Columns: {list(acs_pop_df.columns)}")

# Check GEOID column
print(f"\n   GEOID column type: {acs_pop_df['GEOID'].dtype}")
print(f"   Sample GEOIDs: {acs_pop_df['GEOID'].head(3).tolist()}")

# 2d. Load ACS demographics data
print(f"\n2d. Loading ACS demographics data...")
print(f"   Path: {acs_demographics_path}")
acs_demo_df = pd.read_csv(acs_demographics_path)
print(f"   Records loaded: {len(acs_demo_df):,}")
print(f"   Columns: {list(acs_demo_df.columns)}")

# Check GEOID column
print(f"\n   GEOID column type: {acs_demo_df['GEOID'].dtype}")
print(f"   Sample GEOIDs: {acs_demo_df['GEOID'].head(3).tolist()}")

# 3. Ensure GEOID types match for merging
print(f"\n3. Preparing GEOID for merging...")
# Convert all to string to ensure consistent matching
health_gdf['GEOID'] = health_gdf['GEOID'].astype(str)
acs_econ_df['GEOID'] = acs_econ_df['GEOID'].astype(str)
acs_educ_df['GEOID'] = acs_educ_df['GEOID'].astype(str)
acs_pop_df['GEOID'] = acs_pop_df['GEOID'].astype(str)
acs_demo_df['GEOID'] = acs_demo_df['GEOID'].astype(str)

print(f"   Health tracts GEOID type: {health_gdf['GEOID'].dtype}")
print(f"   ACS economic GEOID type: {acs_econ_df['GEOID'].dtype}")
print(f"   ACS education GEOID type: {acs_educ_df['GEOID'].dtype}")
print(f"   ACS population GEOID type: {acs_pop_df['GEOID'].dtype}")
print(f"   ACS demographics GEOID type: {acs_demo_df['GEOID'].dtype}")

# Check for overlapping GEOIDs
health_geoids = set(health_gdf['GEOID'].unique())
econ_geoids = set(acs_econ_df['GEOID'].unique())
educ_geoids = set(acs_educ_df['GEOID'].unique())
pop_geoids = set(acs_pop_df['GEOID'].unique())
demo_geoids = set(acs_demo_df['GEOID'].unique())

overlap_all = health_geoids.intersection(econ_geoids).intersection(educ_geoids).intersection(pop_geoids).intersection(demo_geoids)
only_health = health_geoids - econ_geoids - educ_geoids - pop_geoids - demo_geoids
only_econ = econ_geoids - health_geoids
only_educ = educ_geoids - health_geoids
only_pop = pop_geoids - health_geoids
only_demo = demo_geoids - health_geoids

print(f"\n   GEOID overlap check:")
print(f"     Overlapping GEOIDs (all datasets): {len(overlap_all):,}")
print(f"     Only in health data: {len(only_health):,}")
print(f"     Only in ACS economic data: {len(only_econ):,}")
print(f"     Only in ACS education data: {len(only_educ):,}")
print(f"     Only in ACS population data: {len(only_pop):,}")
print(f"     Only in ACS demographics data: {len(only_demo):,}")

if len(only_health) > 0:
    print(f"     Warning: Some health tracts not in ACS data: {list(only_health)[:5]}")
if len(only_econ) > 0:
    print(f"     Warning: Some ACS economic tracts not in health data: {list(only_econ)[:5]}")
if len(only_educ) > 0:
    print(f"     Warning: Some ACS education tracts not in health data: {list(only_educ)[:5]}")
if len(only_pop) > 0:
    print(f"     Warning: Some ACS population tracts not in health data: {list(only_pop)[:5]}")
if len(only_demo) > 0:
    print(f"     Warning: Some ACS demographics tracts not in health data: {list(only_demo)[:5]}")

# 4. Merge datasets
print(f"\n4. Merging datasets on GEOID...")
# First merge health tracts with economic covariates
print(f"   4a. Merging health tracts with economic covariates...")
merged_gdf = health_gdf.merge(
    acs_econ_df,
    on='GEOID',
    how='left',
    suffixes=('', '_econ')
)

print(f"      Records after first merge: {len(merged_gdf):,}")
print(f"      Columns after first merge: {len(merged_gdf.columns)}")

# Then merge with education data
print(f"   4b. Merging with education variables...")
merged_gdf = merged_gdf.merge(
    acs_educ_df,
    on='GEOID',
    how='left',
    suffixes=('', '_educ')
)

print(f"      Records after second merge: {len(merged_gdf):,}")
print(f"      Columns after second merge: {len(merged_gdf.columns)}")

# Finally merge with population data
print(f"   4c. Merging with population data...")
merged_gdf = merged_gdf.merge(
    acs_pop_df,
    on='GEOID',
    how='left',
    suffixes=('', '_pop')
)

print(f"      Records after third merge: {len(merged_gdf):,}")
print(f"      Columns after third merge: {len(merged_gdf.columns)}")

# Finally merge with demographics data
print(f"   4d. Merging with demographics data...")
merged_gdf = merged_gdf.merge(
    acs_demo_df,
    on='GEOID',
    how='left',
    suffixes=('', '_demo')
)

print(f"   Final merged records: {len(merged_gdf):,}")
print(f"   Total columns after all merges: {len(merged_gdf.columns)}")

# Check for duplicate NAME columns
name_cols = [col for col in merged_gdf.columns if col.startswith('NAME')]
if len(name_cols) > 1:
    print(f"   Note: Multiple 'NAME' columns found: {name_cols}")
    print(f"         Keeping original 'NAME', others available with suffixes.")

# 5. Check for missing ACS data
print(f"\n5. Checking data completeness...")
acs_econ_cols = ['median_household_income', 'pct_families_below_poverty', 'unemployment_rate']
acs_educ_cols = ['pct_high_school_graduate_or_higher', 'pct_bachelors_degree_or_higher']

print(f"   Economic covariates:")
for col in acs_econ_cols:
    if col in merged_gdf.columns:
        missing = merged_gdf[col].isna().sum()
        print(f"     {col}: {missing} missing values ({missing/len(merged_gdf)*100:.1f}%)")
    else:
        print(f"     Warning: {col} not found in merged data")

print(f"   Education variables:")
for col in acs_educ_cols:
    if col in merged_gdf.columns:
        missing = merged_gdf[col].isna().sum()
        print(f"     {col}: {missing} missing values ({missing/len(merged_gdf)*100:.1f}%)")
    else:
        print(f"     Warning: {col} not found in merged data")

print(f"   Population variable:")
if 'total_population' in merged_gdf.columns:
    missing = merged_gdf['total_population'].isna().sum()
    print(f"     total_population: {missing} missing values ({missing/len(merged_gdf)*100:.1f}%)")
    if missing == 0:
        print(f"     Total population: {merged_gdf['total_population'].sum():,.0f}")
else:
    print(f"     Warning: total_population not found in merged data")

print(f"   Demographics variables:")
acs_demo_cols = ['pct_under_18', 'pct_65_and_over', 'pct_non_hispanic_white', 'pct_black', 'pct_hispanic']
for col in acs_demo_cols:
    if col in merged_gdf.columns:
        missing = merged_gdf[col].isna().sum()
        print(f"     {col}: {missing} missing values ({missing/len(merged_gdf)*100:.1f}%)")
    else:
        print(f"     Warning: {col} not found in merged data")

# 6. Display summary information
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Final dataset:")
print(f"  Records: {len(merged_gdf):,}")
print(f"  CRS: {merged_gdf.crs}")
print(f"  Total columns: {len(merged_gdf.columns)}")

print(f"\n  Key variable categories:")
print(f"    - Health outcomes: {len([c for c in merged_gdf.columns if any(x in c for x in ['_CrudePrev', 'TotalPopulation', 'ACCESS2', 'ARTHRITIS', 'BINGE', 'BPHIGH', 'CANCER', 'CASTHMA', 'CHD', 'CHECKUP', 'CHOLSCREEN', 'COLON_SCREEN', 'COPD', 'CSMOKING', 'DENTAL', 'DEPRESSION', 'DIABETES', 'GHLTH', 'HIGHCHOL', 'LPA', 'MAMMOUSE', 'MHLTH', 'OBESITY', 'PHLTH', 'SLEEP', 'STROKE', 'TEETHLOST', 'HEARING', 'VISION', 'COGNITION', 'MOBILITY', 'SELFCARE', 'INDEPLIVE', 'DISABILITY', 'LONELINESS', 'FOODSTAMP', 'FOODINSECU', 'HOUSINSECU', 'SHUTUTILITY', 'LACKTRPT', 'EMOTIONSPT'])])} variables")
print(f"    - Park exposure: dist_to_park_ft, dist_to_park_miles, within_1_mile")
print(f"    - Economic covariates: median_household_income, pct_families_below_poverty, unemployment_rate")
print(f"    - Education variables: pct_high_school_graduate_or_higher, pct_bachelors_degree_or_higher")
print(f"    - Population: total_population")
print(f"    - Demographics: pct_under_18, pct_65_and_over, pct_non_hispanic_white, pct_black, pct_hispanic")
print(f"    - Geographic identifiers: GEOID, NAME, COUNTYFP, TRACTCE")

# Show sample of key variables
print(f"\n  Sample data (key variables):")
sample_cols = ['GEOID', 'NAME', 'dist_to_park_miles', 'within_1_mile', 
               'median_household_income', 'pct_families_below_poverty', 'unemployment_rate',
               'pct_high_school_graduate_or_higher', 'pct_bachelors_degree_or_higher', 'total_population',
               'pct_under_18', 'pct_65_and_over', 'pct_non_hispanic_white', 'pct_black', 'pct_hispanic']
available_sample_cols = [c for c in sample_cols if c in merged_gdf.columns]
if available_sample_cols:
    print(merged_gdf[available_sample_cols].head(10).to_string(index=False))

# 7. Save to GeoPackage
print(f"\n6. Saving to GeoPackage...")
print(f"   Output path: {output_gpkg_path}")

merged_gdf.to_file(output_gpkg_path, driver='GPKG', layer='butler_tract_health_model_data')
print(f"   [OK] GeoPackage saved successfully!")

# 8. Save CSV copy (without geometry)
print(f"\n7. Saving CSV copy (without geometry)...")
print(f"   Output path: {output_csv_path}")

# Create a copy without geometry for CSV
merged_df = merged_gdf.drop(columns=['geometry']).copy()
merged_df.to_csv(output_csv_path, index=False)
print(f"   [OK] CSV saved successfully!")
print(f"   Note: Geometry column excluded from CSV (spatial data in GeoPackage only)")

print(f"\n{'='*60}")
print("HEALTH ANALYSIS DATASET CREATION COMPLETE")
print(f"{'='*60}")
