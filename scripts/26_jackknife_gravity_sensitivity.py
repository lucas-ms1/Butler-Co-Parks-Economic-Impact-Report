"""
Jackknife Sensitivity Analysis for Frequent Mental Distress Model
Economic Impact Report for Butler County Parks

This script performs a leave-one-out jackknife analysis by re-fitting the
Frequent Mental Distress WLS model (density + reduced demographics, excluding
pct_non_hispanic_white) 86 times, each time dropping one tract. This provides
a robust assessment of the sensitivity of the park_gravity_index_z coefficient
to individual observations.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
from scipy.spatial.distance import cdist
import os

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input paths
input_gpkg_path = os.path.join(project_root, 'data_final', 'butler_tract_health_model_data.gpkg')
parks_path = os.path.join(project_root, 'data_intermediate', 'butler_county_parks.shp')

# Output paths
output_dir = os.path.join(project_root, 'results')
output_path = os.path.join(output_dir, 'mental_distress_jackknife.csv')

print("="*60)
print("JACKKNIFE SENSITIVITY ANALYSIS: FREQUENT MENTAL DISTRESS")
print("="*60)
print("Model: WLS with Density + Reduced Demographics")
print("(Excluding pct_non_hispanic_white)")
print("Method: Leave-One-Out Jackknife")
print("="*60)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print(f"\n1. Loading data...")
print(f"   GeoPackage Path: {input_gpkg_path}")
print(f"   Parks Path: {parks_path}")

# Load from GeoPackage to get geometry
gdf = gpd.read_file(input_gpkg_path)
print(f"   Tracts loaded: {len(gdf):,}")
print(f"   CRS: {gdf.crs}")

# Ensure CRS consistency
target_crs = 'EPSG:3402'  # Ohio State Plane South (US Survey Feet)
if str(gdf.crs) != target_crs:
    print(f"   Projecting tracts to {target_crs}...")
    gdf = gdf.to_crs(target_crs)

# Load parks
gdf_parks = gpd.read_file(parks_path)
if str(gdf_parks.crs) != target_crs:
    print(f"   Projecting parks to {target_crs}...")
    gdf_parks = gdf_parks.to_crs(target_crs)
print(f"   Parks loaded: {len(gdf_parks):,}")

# ============================================================================
# 2. CALCULATE PARK GRAVITY INDEX (if not present)
# ============================================================================
if 'park_gravity_index_z' not in gdf.columns:
    print(f"\n2. Calculating Park Gravity Index...")
    print(f"   Using distance decay coefficient: lambda_d = -1.76 (Macfarlane et al. 2020)")
    
    # Compute tract centroids
    gdf['centroid'] = gdf.geometry.centroid
    centroids = np.array([[geom.x, geom.y] for geom in gdf['centroid']])
    
    # Get park coordinates
    if gdf_parks.geometry.geom_type.iloc[0] == 'Point':
        park_coords = np.array([[geom.x, geom.y] for geom in gdf_parks.geometry])
    else:
        park_coords = np.array([[geom.x, geom.y] for geom in gdf_parks.geometry.centroid])
    
    # Calculate distance matrix
    distance_matrix = cdist(centroids, park_coords, metric='euclidean')
    
    # Convert to meters
    US_SURVEY_FT_TO_M = 0.3048006096
    distance_matrix_m = distance_matrix * US_SURVEY_FT_TO_M
    
    # Add epsilon to avoid log(0)
    epsilon = 1.0  # 1 meter
    distance_matrix_m = np.maximum(distance_matrix_m, epsilon)
    
    # Calculate Park Gravity Index
    lambda_d = -1.76
    gravity_components = np.exp(lambda_d * np.log(distance_matrix_m))
    gravity_sum = np.sum(gravity_components, axis=1)
    park_gravity_index = np.log(gravity_sum)
    
    gdf['park_gravity_index'] = park_gravity_index
    
    # Normalize (Z-score)
    gdf['park_gravity_index_z'] = (
        (gdf['park_gravity_index'] - gdf['park_gravity_index'].mean()) 
        / gdf['park_gravity_index'].std()
    )
    
    # Clean up
    gdf = gdf.drop(columns=['centroid'])
    print(f"   Park Gravity Index calculated and normalized")
else:
    print(f"\n2. Park Gravity Index already present in data")

# ============================================================================
# 3. CALCULATE POPULATION DENSITY (if not present)
# ============================================================================
if 'log_pop_density' not in gdf.columns:
    print(f"\n3. Calculating population density...")
    
    # Calculate tract area in square feet
    gdf['tract_area_sqft'] = gdf.geometry.area
    
    # Convert to square miles
    SQFT_TO_SQMI = 1.0 / 27878400.0
    gdf['tract_area_sqmi'] = gdf['tract_area_sqft'] * SQFT_TO_SQMI
    
    # Calculate population density
    gdf['pop_density'] = gdf['total_population'] / gdf['tract_area_sqmi']
    
    # Calculate log of population density
    epsilon = 0.001
    gdf['log_pop_density'] = np.log(gdf['pop_density'] + epsilon)
    
    print(f"   Population density calculated")
    print(f"     Mean density: {gdf['pop_density'].mean():.2f} people/sq mi")
    print(f"     Mean log density: {gdf['log_pop_density'].mean():.4f}")
else:
    print(f"\n3. Population density already present in data")

# Convert to DataFrame for regression (drop geometry)
df = gdf.drop(columns=['geometry']).copy()
print(f"\n   Final columns: {len(df.columns)}")

# ============================================================================
# 4. PREPARE MODEL SPECIFICATION
# ============================================================================
print(f"\n4. Preparing model specification...")

# Define outcome
outcome_col = None
if 'MHLTH_AgeAdjPrev' in df.columns:
    outcome_col = 'MHLTH_AgeAdjPrev'
    print(f"   Outcome: MHLTH_AgeAdjPrev (age-adjusted)")
elif 'MHLTH_CrudePrev' in df.columns:
    outcome_col = 'MHLTH_CrudePrev'
    print(f"   Outcome: MHLTH_CrudePrev (crude prevalence)")
else:
    raise ValueError("Frequent Mental Distress variable not found!")

# Define control variables
control_vars = {
    'median_household_income': 'median_household_income',
    'pct_families_below_poverty': 'pct_families_below_poverty',
    'unemployment_rate': 'unemployment_rate',
    'pct_bachelors_degree_or_higher': 'pct_bachelors_degree_or_higher'
}

# Check which controls are available
available_controls = {}
for key, col in control_vars.items():
    if col in df.columns:
        available_controls[key] = col
        print(f"   Control available: {col}")
    else:
        print(f"   Warning: Control {col} not found!")

if len(available_controls) == 0:
    raise ValueError("No control variables found!")

# Check for population density
if 'log_pop_density' not in df.columns:
    raise ValueError("log_pop_density not found - required for model!")

# Define reduced demographic variables (excluding pct_non_hispanic_white)
demographic_vars_reduced = {
    'pct_under_18': 'pct_under_18',
    'pct_65_and_over': 'pct_65_and_over',
    'pct_black': 'pct_black',
    'pct_hispanic': 'pct_hispanic'
}

available_demographics_reduced = {}
for key, col in demographic_vars_reduced.items():
    if col in df.columns:
        available_demographics_reduced[key] = col
        print(f"   Demographic available: {col}")
    else:
        print(f"   Warning: Demographic {col} not found!")

if len(available_demographics_reduced) == 0:
    raise ValueError("No reduced demographic variables found!")

# Check for population variable
if 'total_population' not in df.columns:
    raise ValueError("total_population not found - required for WLS weighting!")

# Prepare regression variables
reg_vars = (['park_gravity_index_z'] + list(available_controls.values()) + 
            ['log_pop_density'] + list(available_demographics_reduced.values()) + 
            [outcome_col, 'total_population', 'GEOID', 'NAME'])

# Create subset with non-missing data
df_full = df[reg_vars].copy()
df_full = df_full.dropna()

# Filter to positive population weights
df_full = df_full[df_full['total_population'] > 0].copy()

print(f"\n   Full dataset observations: {len(df_full):,}")

if len(df_full) == 0:
    raise ValueError("No valid observations for model fitting!")

# Prepare independent variables list (for indexing)
X_vars = (['park_gravity_index_z'] + list(available_controls.values()) + 
          ['log_pop_density'] + list(available_demographics_reduced.values()))

# Get index of park_gravity_index_z in X_vars (will be +1 for constant in model)
park_gravity_idx = X_vars.index('park_gravity_index_z') + 1  # +1 for constant

# ============================================================================
# 5. RUN JACKKNIFE ANALYSIS
# ============================================================================
print(f"\n5. Running Jackknife Analysis...")
print(f"   Total iterations: {len(df_full)}")
print(f"   Each iteration drops one tract and re-fits the model")

jackknife_results = []

for i, (idx, row) in enumerate(df_full.iterrows(), 1):
    dropped_geoid = row['GEOID']
    dropped_name = row['NAME']
    
    # Create subset without this tract
    df_jackknife = df_full[df_full.index != idx].copy()
    
    if len(df_jackknife) == 0:
        print(f"   Iteration {i}/{len(df_full)}: Skipping {dropped_geoid} - no observations remaining")
        continue
    
    # Prepare dependent variable
    y = df_jackknife[outcome_col].values
    
    # Prepare independent variables (add constant)
    X = df_jackknife[X_vars].values
    X = sm.add_constant(X)  # Add intercept
    
    # Prepare weights
    weights = df_jackknife['total_population'].values
    
    # Fit WLS model
    try:
        model = sm.WLS(y, X, weights=weights)
        results = model.fit(cov_type='HC1')  # Robust standard errors (HC1)
        
        # Extract park_gravity_index_z coefficient
        park_gravity_coef = results.params[park_gravity_idx]
        park_gravity_se = results.bse[park_gravity_idx]
        park_gravity_pval = results.pvalues[park_gravity_idx]
        
        # Get confidence intervals
        ci_array = results.conf_int()
        if isinstance(ci_array, pd.DataFrame):
            park_gravity_ci_lower = ci_array.iloc[park_gravity_idx, 0]
            park_gravity_ci_upper = ci_array.iloc[park_gravity_idx, 1]
        else:
            park_gravity_ci_lower = ci_array[park_gravity_idx, 0]
            park_gravity_ci_upper = ci_array[park_gravity_idx, 1]
        
        # Store results
        jackknife_results.append({
            'iteration': i,
            'dropped_geoid': dropped_geoid,
            'dropped_name': dropped_name,
            'n_obs': len(df_jackknife),
            'r_squared': results.rsquared,
            'park_gravity_coef': park_gravity_coef,
            'park_gravity_se': park_gravity_se,
            'park_gravity_pvalue': park_gravity_pval,
            'park_gravity_ci_lower': park_gravity_ci_lower,
            'park_gravity_ci_upper': park_gravity_ci_upper,
            'significant_05': 1 if park_gravity_pval < 0.05 else 0
        })
        
        if i % 10 == 0 or i == len(df_full):
            print(f"   Iteration {i}/{len(df_full)}: {dropped_geoid} ({dropped_name}) - "
                  f"Coef: {park_gravity_coef:+.6f}, p={park_gravity_pval:.4f}")
    
    except Exception as e:
        print(f"   Iteration {i}/{len(df_full)}: Error fitting model for {dropped_geoid}: {e}")
        continue

# Convert to DataFrame
jackknife_df = pd.DataFrame(jackknife_results)

if len(jackknife_df) == 0:
    raise ValueError("No successful jackknife iterations!")

# ============================================================================
# 6. SUMMARY STATISTICS
# ============================================================================
print(f"\n6. Summary Statistics...")

coef_min = jackknife_df['park_gravity_coef'].min()
coef_median = jackknife_df['park_gravity_coef'].median()
coef_max = jackknife_df['park_gravity_coef'].max()
coef_mean = jackknife_df['park_gravity_coef'].mean()
coef_std = jackknife_df['park_gravity_coef'].std()

pval_min = jackknife_df['park_gravity_pvalue'].min()
pval_median = jackknife_df['park_gravity_pvalue'].median()
pval_max = jackknife_df['park_gravity_pvalue'].max()

n_significant = jackknife_df['significant_05'].sum()
n_total = len(jackknife_df)
pct_significant = (n_significant / n_total) * 100

print(f"\n   Park Gravity Index Coefficient:")
print(f"     Min:    {coef_min:+.6f}")
print(f"     Median: {coef_median:+.6f}")
print(f"     Mean:   {coef_mean:+.6f}")
print(f"     Max:    {coef_max:+.6f}")
print(f"     Std:    {coef_std:.6f}")

print(f"\n   P-value:")
print(f"     Min:    {pval_min:.6f}")
print(f"     Median: {pval_median:.6f}")
print(f"     Max:    {pval_max:.6f}")

print(f"\n   Significance (p < 0.05):")
print(f"     Significant runs: {n_significant} / {n_total} ({pct_significant:.1f}%)")
print(f"     Non-significant runs: {n_total - n_significant} / {n_total} ({100 - pct_significant:.1f}%)")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print(f"\n7. Saving results...")
print(f"   Output path: {output_path}")

jackknife_df.to_csv(output_path, index=False)
print(f"   [OK] Jackknife results saved successfully!")
print(f"   Total iterations saved: {len(jackknife_df)}")

# ============================================================================
# 8. ONE-PAGE SUMMARY
# ============================================================================
print(f"\n{'='*60}")
print("JACKKNIFE SENSITIVITY ANALYSIS SUMMARY")
print(f"{'='*60}")

print(f"\nModel Specification:")
print(f"  Outcome: {outcome_col}")
print(f"  Predictor: park_gravity_index_z (Z-score normalized)")
print(f"  Controls: {', '.join(available_controls.values())}")
print(f"  Additional Controls: log_pop_density")
print(f"  Demographics: {', '.join(available_demographics_reduced.values())}")
print(f"  Method: Weighted Least Squares (WLS) with population weights")
print(f"  Standard Errors: Robust (HC1)")

print(f"\nJackknife Results:")
print(f"  Total iterations: {n_total}")
print(f"  Observations per iteration: {jackknife_df['n_obs'].iloc[0]} (dropping 1 tract each time)")

print(f"\nPark Gravity Index Coefficient Distribution:")
print(f"  Minimum:   {coef_min:+.6f}")
print(f"  Median:    {coef_median:+.6f}")
print(f"  Mean:      {coef_mean:+.6f}")
print(f"  Maximum:   {coef_max:+.6f}")
print(f"  Std Dev:   {coef_std:.6f}")

print(f"\nP-value Distribution:")
print(f"  Minimum:   {pval_min:.6f}")
print(f"  Median:    {pval_median:.6f}")
print(f"  Maximum:   {pval_max:.6f}")

print(f"\nSignificance Assessment (p < 0.05):")
print(f"  Significant:     {n_significant:3d} / {n_total} ({pct_significant:5.1f}%)")
print(f"  Non-significant: {n_total - n_significant:3d} / {n_total} ({100 - pct_significant:5.1f}%)")

# Identify most and least influential tracts
most_influential = jackknife_df.loc[jackknife_df['park_gravity_coef'].idxmin()]
least_influential = jackknife_df.loc[jackknife_df['park_gravity_coef'].idxmax()]

print(f"\nMost Influential Tract (lowest coefficient when dropped):")
print(f"  GEOID: {most_influential['dropped_geoid']}")
print(f"  Name:  {most_influential['dropped_name']}")
print(f"  Coef:  {most_influential['park_gravity_coef']:+.6f}")
print(f"  P-val: {most_influential['park_gravity_pvalue']:.6f}")

print(f"\nLeast Influential Tract (highest coefficient when dropped):")
print(f"  GEOID: {least_influential['dropped_geoid']}")
print(f"  Name:  {least_influential['dropped_name']}")
print(f"  Coef:  {least_influential['park_gravity_coef']:+.6f}")
print(f"  P-val: {least_influential['park_gravity_pvalue']:.6f}")

print(f"\n{'='*60}")
print("JACKKNIFE ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"\nOutputs created:")
print(f"  1. Jackknife results: {output_path}")
print(f"     (Coefficient and p-value for each leave-one-out iteration)")
