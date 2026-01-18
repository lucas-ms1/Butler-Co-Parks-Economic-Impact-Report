"""
WLS Influence Diagnostics for Frequent Mental Distress Model
Economic Impact Report for Butler County Parks

This script fits the Frequent Mental Distress WLS model with density + reduced
demographics (excluding pct_non_hispanic_white), computes Cook's distance and
leverage, identifies the top 10 influential tracts, and re-runs the model after
dropping those tracts to assess robustness of the park-gravity coefficient.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy.spatial.distance import cdist
import os

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input paths
input_csv_path = os.path.join(project_root, 'data_final', 'butler_tract_health_model_data.csv')
input_gpkg_path = os.path.join(project_root, 'data_final', 'butler_tract_health_model_data.gpkg')
parks_path = os.path.join(project_root, 'data_intermediate', 'butler_county_parks.shp')

# Output paths
output_dir = os.path.join(project_root, 'results')
output_path = os.path.join(output_dir, 'mental_distress_influence.csv')

print("="*60)
print("WLS INFLUENCE DIAGNOSTICS: FREQUENT MENTAL DISTRESS")
print("="*60)
print("Model: WLS with Density + Reduced Demographics")
print("(Excluding pct_non_hispanic_white)")
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

# Define control variables (same as in script 23)
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

# ============================================================================
# 5. FIT FULL MODEL
# ============================================================================
print(f"\n5. Fitting full WLS model...")

# Prepare data
reg_vars = (['park_gravity_index_z'] + list(available_controls.values()) + 
            ['log_pop_density'] + list(available_demographics_reduced.values()) + 
            [outcome_col, 'total_population', 'GEOID', 'NAME'])

# Create subset with non-missing data
df_model = df[reg_vars].copy()
df_model = df_model.dropna()

# Filter to positive population weights
df_model = df_model[df_model['total_population'] > 0].copy()

print(f"   Observations: {len(df_model):,}")

if len(df_model) == 0:
    raise ValueError("No valid observations for model fitting!")

# Prepare dependent variable
y = df_model[outcome_col].values

# Prepare independent variables (add constant)
X_vars = (['park_gravity_index_z'] + list(available_controls.values()) + 
          ['log_pop_density'] + list(available_demographics_reduced.values()))
X = df_model[X_vars].values
X = sm.add_constant(X)  # Add intercept

# Prepare weights
weights = df_model['total_population'].values

# Fit WLS model
print(f"   Fitting WLS model...")
model_full = sm.WLS(y, X, weights=weights)
results_full = model_full.fit(cov_type='HC1')  # Robust standard errors (HC1)

print(f"\n   Full Model Summary:")
print(f"   R-squared: {results_full.rsquared:.4f}")
print(f"   Observations: {len(df_model):,}")

# Extract park_gravity_index_z coefficient from full model
park_gravity_idx = X_vars.index('park_gravity_index_z') + 1  # +1 for constant
park_gravity_coef_full = results_full.params[park_gravity_idx]
park_gravity_se_full = results_full.bse[park_gravity_idx]
park_gravity_pval_full = results_full.pvalues[park_gravity_idx]

# Get confidence intervals
ci_array_full = results_full.conf_int()
if isinstance(ci_array_full, pd.DataFrame):
    park_gravity_ci_lower_full = ci_array_full.iloc[park_gravity_idx, 0]
    park_gravity_ci_upper_full = ci_array_full.iloc[park_gravity_idx, 1]
else:
    park_gravity_ci_lower_full = ci_array_full[park_gravity_idx, 0]
    park_gravity_ci_upper_full = ci_array_full[park_gravity_idx, 1]

print(f"\n   Park Gravity Index Coefficient (Full Model):")
print(f"     Coefficient: {park_gravity_coef_full:+.6f}")
print(f"     Standard Error: {park_gravity_se_full:.6f}")
print(f"     P-value: {park_gravity_pval_full:.6f}")
print(f"     95% CI: [{park_gravity_ci_lower_full:.6f}, {park_gravity_ci_upper_full:.6f}]")

# ============================================================================
# 6. COMPUTE INFLUENCE MEASURES
# ============================================================================
print(f"\n6. Computing influence measures...")

# Get influence object (works with WLS results)
influence = OLSInfluence(results_full)

# Compute Cook's distance
cooks_d = influence.cooks_distance[0]  # Returns tuple (cooks_d, pvalue)

# Compute leverage (hat matrix diagonal)
leverage = influence.hat_matrix_diag

# Create influence diagnostics dataframe
influence_df = pd.DataFrame({
    'GEOID': df_model['GEOID'].values,
    'NAME': df_model['NAME'].values,
    'cooks_distance': cooks_d,
    'leverage': leverage
})

# Sort by Cook's distance (descending)
influence_df = influence_df.sort_values('cooks_distance', ascending=False)

print(f"\n   Cook's Distance Statistics:")
print(f"     Min: {cooks_d.min():.6f}")
print(f"     Max: {cooks_d.max():.6f}")
print(f"     Mean: {cooks_d.mean():.6f}")
print(f"     Median: {np.median(cooks_d):.6f}")

print(f"\n   Leverage Statistics:")
print(f"     Min: {leverage.min():.6f}")
print(f"     Max: {leverage.max():.6f}")
print(f"     Mean: {leverage.mean():.6f}")
print(f"     Median: {np.median(leverage):.6f}")

# Identify top 10 influential tracts
top_10_influential = influence_df.head(10).copy()
top_10_influential['rank'] = range(1, len(top_10_influential) + 1)

print(f"\n   Top 10 Influential Tracts (by Cook's Distance):")
print(top_10_influential[['rank', 'GEOID', 'NAME', 'cooks_distance', 'leverage']].to_string(index=False))

# Save influence diagnostics
print(f"\n   Saving influence diagnostics to: {output_path}")
top_10_influential[['rank', 'GEOID', 'NAME', 'cooks_distance', 'leverage']].to_csv(
    output_path, index=False
)
print(f"   [OK] Influence diagnostics saved successfully!")

# Get GEOIDs of top 10 influential tracts
top_10_geoids = set(top_10_influential['GEOID'].values)

# ============================================================================
# 7. RE-RUN MODEL WITHOUT TOP 10 INFLUENTIAL TRACTS
# ============================================================================
print(f"\n7. Re-running model without top 10 influential tracts...")

# Create subset excluding top 10 influential tracts
df_model_reduced = df_model[~df_model['GEOID'].isin(top_10_geoids)].copy()

print(f"   Observations after dropping top 10: {len(df_model_reduced):,}")
print(f"   Dropped tracts: {len(df_model) - len(df_model_reduced)}")

if len(df_model_reduced) == 0:
    raise ValueError("No observations remaining after dropping influential tracts!")

# Prepare data for reduced model
y_reduced = df_model_reduced[outcome_col].values
X_reduced = df_model_reduced[X_vars].values
X_reduced = sm.add_constant(X_reduced)
weights_reduced = df_model_reduced['total_population'].values

# Fit reduced WLS model
print(f"   Fitting reduced WLS model...")
model_reduced = sm.WLS(y_reduced, X_reduced, weights=weights_reduced)
results_reduced = model_reduced.fit(cov_type='HC1')

print(f"\n   Reduced Model Summary:")
print(f"   R-squared: {results_reduced.rsquared:.4f}")
print(f"   Observations: {len(df_model_reduced):,}")

# Extract park_gravity_index_z coefficient from reduced model
park_gravity_coef_reduced = results_reduced.params[park_gravity_idx]
park_gravity_se_reduced = results_reduced.bse[park_gravity_idx]
park_gravity_pval_reduced = results_reduced.pvalues[park_gravity_idx]

# Get confidence intervals
ci_array_reduced = results_reduced.conf_int()
if isinstance(ci_array_reduced, pd.DataFrame):
    park_gravity_ci_lower_reduced = ci_array_reduced.iloc[park_gravity_idx, 0]
    park_gravity_ci_upper_reduced = ci_array_reduced.iloc[park_gravity_idx, 1]
else:
    park_gravity_ci_lower_reduced = ci_array_reduced[park_gravity_idx, 0]
    park_gravity_ci_upper_reduced = ci_array_reduced[park_gravity_idx, 1]

print(f"\n   Park Gravity Index Coefficient (Reduced Model):")
print(f"     Coefficient: {park_gravity_coef_reduced:+.6f}")
print(f"     Standard Error: {park_gravity_se_reduced:.6f}")
print(f"     P-value: {park_gravity_pval_reduced:.6f}")
print(f"     95% CI: [{park_gravity_ci_lower_reduced:.6f}, {park_gravity_ci_upper_reduced:.6f}]")

# ============================================================================
# 8. COMPARE RESULTS
# ============================================================================
print(f"\n{'='*60}")
print("COMPARISON: FULL MODEL vs REDUCED MODEL")
print(f"{'='*60}")

comparison_data = {
    'Model': ['Full Model', 'Reduced Model (Top 10 Dropped)'],
    'N': [len(df_model), len(df_model_reduced)],
    'R-squared': [results_full.rsquared, results_reduced.rsquared],
    'Park Gravity Coef': [park_gravity_coef_full, park_gravity_coef_reduced],
    'Park Gravity SE': [park_gravity_se_full, park_gravity_se_reduced],
    'Park Gravity P-value': [park_gravity_pval_full, park_gravity_pval_reduced],
    'Park Gravity CI Lower': [park_gravity_ci_lower_full, park_gravity_ci_lower_reduced],
    'Park Gravity CI Upper': [park_gravity_ci_upper_full, park_gravity_ci_upper_reduced]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Calculate change
coef_change = park_gravity_coef_reduced - park_gravity_coef_full
coef_change_pct = (coef_change / abs(park_gravity_coef_full)) * 100 if park_gravity_coef_full != 0 else np.nan

print(f"\n   Change in Park Gravity Index Coefficient:")
print(f"     Absolute change: {coef_change:+.6f}")
if not np.isnan(coef_change_pct):
    print(f"     Percentage change: {coef_change_pct:+.2f}%")

# Significance interpretation
print(f"\n   Significance Assessment:")
if park_gravity_pval_full < 0.05 and park_gravity_pval_reduced < 0.05:
    print(f"     Both models show significant effect (p < 0.05)")
elif park_gravity_pval_full < 0.05 and park_gravity_pval_reduced >= 0.05:
    print(f"     WARNING: Effect becomes non-significant after dropping influential tracts")
elif park_gravity_pval_full >= 0.05 and park_gravity_pval_reduced < 0.05:
    print(f"     Effect becomes significant after dropping influential tracts")
else:
    print(f"     Both models show non-significant effect")

# Check if coefficient sign changes
if np.sign(park_gravity_coef_full) != np.sign(park_gravity_coef_reduced):
    print(f"     WARNING: Coefficient sign changes after dropping influential tracts!")
else:
    print(f"     Coefficient sign remains the same")

print(f"\n{'='*60}")
print("INFLUENCE DIAGNOSTICS COMPLETE")
print(f"{'='*60}")
print(f"\nOutputs created:")
print(f"  1. Influence diagnostics: {output_path}")
print(f"     (Top 10 influential tracts by Cook's Distance)")
