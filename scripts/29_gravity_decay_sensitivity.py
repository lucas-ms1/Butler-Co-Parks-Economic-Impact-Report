"""
Park Gravity Index Decay Parameter Sensitivity Analysis
Economic Impact Report for Butler County Parks

This script tests the sensitivity of the Frequent Mental Distress WLS model
results to different distance decay parameters in the Park Gravity Index calculation.
It re-computes the index using three decay parameters (-1.0, -1.76, -2.5), re-fits
the model for each, and compares the park_gravity_index_z coefficients.
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
output_path = os.path.join(output_dir, 'gravity_decay_sensitivity.csv')

print("="*60)
print("PARK GRAVITY INDEX DECAY PARAMETER SENSITIVITY")
print("="*60)
print("Model: WLS with Density + Reduced Demographics")
print("(Excluding pct_non_hispanic_white)")
print("="*60)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define decay parameters to test
decay_parameters = [-1.0, -1.76, -2.5]
print(f"\nDecay parameters to test: {decay_parameters}")
print(f"  -1.0:  Less distance decay (parks have influence over longer distances)")
print(f"  -1.76: Baseline (Macfarlane et al. 2020)")
print(f"  -2.5:  More distance decay (parks have influence over shorter distances)")

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
# 2. CALCULATE DISTANCE MATRIX (ONCE, REUSED FOR ALL DECAY PARAMETERS)
# ============================================================================
print(f"\n2. Calculating distance matrix...")

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

print(f"   Distance matrix calculated")
print(f"     Shape: {distance_matrix_m.shape}")
print(f"     Min distance: {distance_matrix_m.min():.2f} m")
print(f"     Max distance: {distance_matrix_m.max():.2f} m")
print(f"     Mean distance: {distance_matrix_m.mean():.2f} m")

# Clean up temporary centroid column
gdf = gdf.drop(columns=['centroid'])

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
else:
    print(f"\n3. Population density already present in data")

# ============================================================================
# 4. PREPARE MODEL SPECIFICATION
# ============================================================================
print(f"\n4. Preparing model specification...")

# Define outcome
outcome_col = None
if 'MHLTH_AgeAdjPrev' in gdf.columns:
    outcome_col = 'MHLTH_AgeAdjPrev'
    print(f"   Outcome: MHLTH_AgeAdjPrev (age-adjusted)")
elif 'MHLTH_CrudePrev' in gdf.columns:
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
    if col in gdf.columns:
        available_controls[key] = col
        print(f"   Control available: {col}")
    else:
        print(f"   Warning: Control {col} not found!")

if len(available_controls) == 0:
    raise ValueError("No control variables found!")

# Check for population density
if 'log_pop_density' not in gdf.columns:
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
    if col in gdf.columns:
        available_demographics_reduced[key] = col
        print(f"   Demographic available: {col}")
    else:
        print(f"   Warning: Demographic {col} not found!")

if len(available_demographics_reduced) == 0:
    raise ValueError("No reduced demographic variables found!")

# Check for population variable
if 'total_population' not in gdf.columns:
    raise ValueError("total_population not found - required for WLS weighting!")

# ============================================================================
# 5. RUN SENSITIVITY ANALYSIS FOR EACH DECAY PARAMETER
# ============================================================================
print(f"\n5. Running sensitivity analysis for each decay parameter...")

results_list = []

for lambda_d in decay_parameters:
    print(f"\n   {'='*50}")
    print(f"   Decay Parameter: {lambda_d}")
    print(f"   {'='*50}")
    
    # Calculate Park Gravity Index with this decay parameter
    print(f"   Calculating Park Gravity Index with lambda_d = {lambda_d}...")
    gravity_components = np.exp(lambda_d * np.log(distance_matrix_m))
    gravity_sum = np.sum(gravity_components, axis=1)
    park_gravity_index = np.log(gravity_sum)
    
    # Normalize (Z-score)
    park_gravity_index_z = (
        (park_gravity_index - park_gravity_index.mean()) 
        / park_gravity_index.std()
    )
    
    # Add to GeoDataFrame (temporary column)
    gdf[f'park_gravity_index_z_{lambda_d}'] = park_gravity_index_z
    
    print(f"     Index statistics:")
    print(f"       Min: {park_gravity_index.min():.4f}")
    print(f"       Max: {park_gravity_index.max():.4f}")
    print(f"       Mean: {park_gravity_index.mean():.4f}")
    print(f"       Std: {park_gravity_index.std():.4f}")
    print(f"     Normalized (Z-score) mean: {park_gravity_index_z.mean():.6f}")
    print(f"     Normalized (Z-score) std: {park_gravity_index_z.std():.6f}")
    
    # Prepare regression variables
    reg_vars = ([f'park_gravity_index_z_{lambda_d}'] + list(available_controls.values()) + 
                ['log_pop_density'] + list(available_demographics_reduced.values()) + 
                [outcome_col, 'total_population'])
    
    # Create subset with non-missing data
    df_model = gdf[reg_vars].copy()
    df_model = df_model.dropna()
    
    # Filter to positive population weights
    df_model = df_model[df_model['total_population'] > 0].copy()
    
    print(f"   Observations: {len(df_model):,}")
    
    if len(df_model) == 0:
        print(f"   Warning: No valid observations - skipping")
        continue
    
    # Prepare dependent variable
    y = df_model[outcome_col].values
    
    # Prepare independent variables (add constant)
    X_vars = ([f'park_gravity_index_z_{lambda_d}'] + list(available_controls.values()) + 
              ['log_pop_density'] + list(available_demographics_reduced.values()))
    X = df_model[X_vars].values
    X = sm.add_constant(X)  # Add intercept
    
    # Prepare weights
    weights = df_model['total_population'].values
    
    # Fit WLS model
    print(f"   Fitting WLS model...")
    model = sm.WLS(y, X, weights=weights)
    results = model.fit(cov_type='HC1')  # Robust standard errors (HC1)
    
    # Extract park_gravity_index_z coefficient
    park_gravity_idx = X_vars.index(f'park_gravity_index_z_{lambda_d}') + 1  # +1 for constant
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
    
    # Determine significance
    if park_gravity_pval < 0.01:
        sig = "***"
    elif park_gravity_pval < 0.05:
        sig = "**"
    elif park_gravity_pval < 0.10:
        sig = "*"
    else:
        sig = ""
    
    print(f"\n   Results:")
    print(f"     Coefficient: {park_gravity_coef:+.6f}")
    print(f"     Standard Error: {park_gravity_se:.6f}")
    print(f"     P-value: {park_gravity_pval:.6f} {sig}")
    print(f"     95% CI: [{park_gravity_ci_lower:.6f}, {park_gravity_ci_upper:.6f}]")
    print(f"     R-squared: {results.rsquared:.4f}")
    
    # Store results
    results_list.append({
        'decay_parameter': lambda_d,
        'n_obs': len(df_model),
        'r_squared': results.rsquared,
        'park_gravity_coef': park_gravity_coef,
        'park_gravity_se': park_gravity_se,
        'park_gravity_pvalue': park_gravity_pval,
        'park_gravity_ci_lower': park_gravity_ci_lower,
        'park_gravity_ci_upper': park_gravity_ci_upper,
        'significant_05': 1 if park_gravity_pval < 0.05 else 0,
        'significant_10': 1 if park_gravity_pval < 0.10 else 0,
        'significance': sig
    })

# ============================================================================
# 6. CREATE COMPARISON TABLE
# ============================================================================
print(f"\n6. Creating comparison table...")

results_df = pd.DataFrame(results_list)

# Sort by decay parameter
results_df = results_df.sort_values('decay_parameter')

print(f"\n   Comparison of Results Across Decay Parameters:")
print(f"   {'='*70}")
print(results_df[['decay_parameter', 'n_obs', 'park_gravity_coef', 'park_gravity_se', 
                  'park_gravity_pvalue', 'significant_05', 'r_squared']].to_string(index=False))

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print(f"\n7. Saving results...")
print(f"   Output path: {output_path}")

results_df.to_csv(output_path, index=False)
print(f"   [OK] Results saved successfully!")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print(f"\n{'='*60}")
print("DECAY PARAMETER SENSITIVITY ANALYSIS COMPLETE")
print(f"{'='*60}")

print(f"\nSummary:")
print(f"  Decay parameters tested: {decay_parameters}")
print(f"  Models fitted: {len(results_df)}")

print(f"\nCoefficient Range:")
print(f"  Min: {results_df['park_gravity_coef'].min():+.6f} (lambda_d = {results_df.loc[results_df['park_gravity_coef'].idxmin(), 'decay_parameter']})")
print(f"  Max: {results_df['park_gravity_coef'].max():+.6f} (lambda_d = {results_df.loc[results_df['park_gravity_coef'].idxmax(), 'decay_parameter']})")
print(f"  Range: {results_df['park_gravity_coef'].max() - results_df['park_gravity_coef'].min():.6f}")

print(f"\nSignificance (p < 0.05):")
for _, row in results_df.iterrows():
    sig_status = "Significant" if row['significant_05'] == 1 else "Not significant"
    print(f"  lambda_d = {row['decay_parameter']:+.2f}: {sig_status} (p = {row['park_gravity_pvalue']:.4f})")

print(f"\nOutputs created:")
print(f"  1. Comparison table: {output_path}")
