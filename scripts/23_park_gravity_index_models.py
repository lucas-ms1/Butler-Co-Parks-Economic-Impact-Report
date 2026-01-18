"""
Park Gravity Index Regression Models
Economic Impact Report for Butler County Parks

This script implements the Macfarlane et al. (2020) Park Gravity Index methodology
to calculate utility-based accessibility scores for each census tract, accounting
for proximity to all parks (not just the nearest one).

The script then runs Weighted Least Squares (WLS) regressions with population weights
and performs multicollinearity diagnostics using Variance Inflation Factor (VIF).
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.spatial.distance import cdist
import os
import warnings
warnings.filterwarnings('ignore')

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input paths
tracts_path = os.path.join(project_root, 'data_final', 'butler_tract_health_model_data.gpkg')
parks_path = os.path.join(project_root, 'data_intermediate', 'butler_county_parks.shp')

# Output paths
output_dir = os.path.join(project_root, 'results')
output_path = os.path.join(output_dir, 'park_gravity_index_regressions.csv')
output_path_density = os.path.join(output_dir, 'park_gravity_index_regressions_plus_density.csv')
output_path_density_demo = os.path.join(output_dir, 'park_gravity_index_regressions_plus_density_demographics.csv')
output_path_density_demo_nhwhite_dropped = os.path.join(output_dir, 'park_gravity_index_regressions_plus_density_demographics_nhwhite_dropped.csv')

print("="*60)
print("PARK GRAVITY INDEX REGRESSION MODELS")
print("="*60)
print("Based on Macfarlane et al. (2020) methodology")
print("="*60)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print(f"\n1. Loading data...")

# Load tracts GeoDataFrame
print(f"   Loading tracts from: {tracts_path}")
gdf_tracts = gpd.read_file(tracts_path)
print(f"   Tracts loaded: {len(gdf_tracts):,}")
print(f"   CRS: {gdf_tracts.crs}")

# Load parks GeoDataFrame
print(f"   Loading parks from: {parks_path}")
gdf_parks = gpd.read_file(parks_path)
print(f"   Parks loaded: {len(gdf_parks):,}")
print(f"   Parks CRS: {gdf_parks.crs}")

# Ensure CRS consistency
target_crs = 'EPSG:3402'  # Ohio State Plane South (US Survey Feet)
if str(gdf_tracts.crs) != target_crs:
    print(f"   Projecting tracts to {target_crs}...")
    gdf_tracts = gdf_tracts.to_crs(target_crs)
if str(gdf_parks.crs) != target_crs:
    print(f"   Projecting parks to {target_crs}...")
    gdf_parks = gdf_parks.to_crs(target_crs)

# ============================================================================
# 2. CALCULATE PARK GRAVITY INDEX
# ============================================================================
print(f"\n2. Calculating Park Gravity Index...")
print(f"   Using distance decay coefficient: lambda_d = -1.76 (Macfarlane et al. 2020)")

# Compute tract centroids
print(f"   Computing tract centroids...")
gdf_tracts['centroid'] = gdf_tracts.geometry.centroid
centroids = np.array([[geom.x, geom.y] for geom in gdf_tracts['centroid']])

# Get park coordinates (assuming point geometries)
print(f"   Extracting park coordinates...")
if gdf_parks.geometry.geom_type.iloc[0] == 'Point':
    park_coords = np.array([[geom.x, geom.y] for geom in gdf_parks.geometry])
else:
    # If parks are polygons, use centroids
    park_coords = np.array([[geom.x, geom.y] for geom in gdf_parks.geometry.centroid])

print(f"   Computing distance matrix (tracts × parks)...")
# Calculate distance matrix using scipy (Euclidean distance in projected CRS)
# EPSG:3402 is in US Survey Feet, so distances are in feet
distance_matrix = cdist(centroids, park_coords, metric='euclidean')

# Convert to meters for consistency with typical distance decay applications
# 1 US Survey Foot = 0.3048006096 meters
US_SURVEY_FT_TO_M = 0.3048006096
distance_matrix_m = distance_matrix * US_SURVEY_FT_TO_M

# Add small epsilon to avoid log(0) and division issues
# Use 1 meter as minimum distance
epsilon = 1.0  # 1 meter
distance_matrix_m = np.maximum(distance_matrix_m, epsilon)

print(f"   Distance statistics (meters):")
print(f"     Min: {distance_matrix_m.min():.2f} m")
print(f"     Max: {distance_matrix_m.max():.2f} m")
print(f"     Mean: {distance_matrix_m.mean():.2f} m")
print(f"     Median: {np.median(distance_matrix_m):.2f} m")

# Calculate Park Gravity Index for each tract
# Formula: A_i = ln( sum_j exp(-1.76 * ln(d_ij)) )
# Where d_ij is distance from tract i to park j in meters
print(f"\n   Calculating gravity index for each tract...")
lambda_d = -1.76  # Distance decay coefficient from Macfarlane et al. (2020)

# For each tract, sum over all parks: sum_j exp(-1.76 * ln(d_ij))
# = sum_j exp(-1.76 * ln(d_ij))
# = sum_j d_ij^(-1.76)
gravity_components = np.exp(lambda_d * np.log(distance_matrix_m))
gravity_sum = np.sum(gravity_components, axis=1)  # Sum over parks (axis=1)

# Take natural log
park_gravity_index = np.log(gravity_sum)

# Add to GeoDataFrame
gdf_tracts['park_gravity_index'] = park_gravity_index

print(f"\n   Park Gravity Index statistics:")
print(f"     Min: {park_gravity_index.min():.4f}")
print(f"     Max: {park_gravity_index.max():.4f}")
print(f"     Mean: {park_gravity_index.mean():.4f}")
print(f"     Std: {park_gravity_index.std():.4f}")

# Normalize (Z-score) the Park Gravity Index
print(f"\n   Normalizing Park Gravity Index (Z-score)...")
gdf_tracts['park_gravity_index_z'] = (
    (gdf_tracts['park_gravity_index'] - gdf_tracts['park_gravity_index'].mean()) 
    / gdf_tracts['park_gravity_index'].std()
)
print(f"     Normalized index mean: {gdf_tracts['park_gravity_index_z'].mean():.6f}")
print(f"     Normalized index std: {gdf_tracts['park_gravity_index_z'].std():.6f}")

# ============================================================================
# 2b. CREATE DIAGNOSTICS TABLE
# ============================================================================
print(f"\n2b. Creating Park Gravity Index diagnostics table...")

# Create diagnostics dataframe with key variables
diag_cols = ['GEOID', 'NAME', 'dist_to_park_miles', 'park_gravity_index_z', 'total_population']
available_diag_cols = [col for col in diag_cols if col in gdf_tracts.columns]

if len(available_diag_cols) < len(diag_cols):
    missing = set(diag_cols) - set(available_diag_cols)
    print(f"   Warning: Some columns not found: {missing}")
    print(f"   Available columns: {available_diag_cols}")

# Create diagnostics dataframe - convert GeoDataFrame to DataFrame to drop geometry
diag_df = pd.DataFrame(gdf_tracts[available_diag_cols].copy())

# Sort by park_gravity_index_z
diag_df = diag_df.sort_values('park_gravity_index_z', ascending=False)

# Get top 10 and bottom 10
top_10 = diag_df.head(10).copy()
bottom_10 = diag_df.tail(10).copy()

# Add rank indicator
top_10['rank'] = range(1, len(top_10) + 1)
bottom_10['rank'] = range(len(diag_df) - len(bottom_10) + 1, len(diag_df) + 1)

# Combine and add category
top_10['category'] = 'Top 10'
bottom_10['category'] = 'Bottom 10'
diagnostics_table = pd.concat([top_10, bottom_10], ignore_index=True)

# Reorder columns for display
display_cols = ['rank', 'category'] + [col for col in available_diag_cols if col != 'rank']
diagnostics_table = diagnostics_table[display_cols]

print(f"\n   Top 10 tracts by Park Gravity Index (Z-score):")
print(top_10[available_diag_cols].to_string(index=False))
print(f"\n   Bottom 10 tracts by Park Gravity Index (Z-score):")
print(bottom_10[available_diag_cols].to_string(index=False))

# Save diagnostics table
diag_output_path = os.path.join(output_dir, 'park_gravity_index_diagnostics.csv')
print(f"\n   Saving diagnostics table to: {diag_output_path}")
diagnostics_table.to_csv(diag_output_path, index=False)
print(f"   [OK] Diagnostics table saved successfully!")

# Clean up temporary centroid column
gdf_tracts = gdf_tracts.drop(columns=['centroid'])

# ============================================================================
# 2c. CALCULATE POPULATION DENSITY
# ============================================================================
print(f"\n2c. Calculating tract population density...")

# Calculate tract area in square feet (EPSG:3402 uses US Survey Feet)
gdf_tracts['tract_area_sqft'] = gdf_tracts.geometry.area

# Convert square feet to square miles
# 1 square mile = 27,878,400 square US Survey Feet
SQFT_TO_SQMI = 1.0 / 27878400.0
gdf_tracts['tract_area_sqmi'] = gdf_tracts['tract_area_sqft'] * SQFT_TO_SQMI

# Calculate population density (people per square mile)
gdf_tracts['pop_density'] = gdf_tracts['total_population'] / gdf_tracts['tract_area_sqmi']

# Calculate log of population density (add small epsilon to avoid log(0))
epsilon = 0.001  # Small value to avoid log(0) for zero population
gdf_tracts['log_pop_density'] = np.log(gdf_tracts['pop_density'] + epsilon)

print(f"   Population density statistics:")
print(f"     Min: {gdf_tracts['pop_density'].min():.2f} people/sq mi")
print(f"     Max: {gdf_tracts['pop_density'].max():.2f} people/sq mi")
print(f"     Mean: {gdf_tracts['pop_density'].mean():.2f} people/sq mi")
print(f"     Median: {gdf_tracts['pop_density'].median():.2f} people/sq mi")
print(f"\n   Log population density statistics:")
print(f"     Min: {gdf_tracts['log_pop_density'].min():.4f}")
print(f"     Max: {gdf_tracts['log_pop_density'].max():.4f}")
print(f"     Mean: {gdf_tracts['log_pop_density'].mean():.4f}")
print(f"     Std: {gdf_tracts['log_pop_density'].std():.4f}")

# ============================================================================
# 3. PREPARE REGRESSION DATA
# ============================================================================
print(f"\n3. Preparing regression data...")

# Convert to DataFrame for regression (drop geometry)
df = gdf_tracts.drop(columns=['geometry']).copy()

# Define health outcomes (prefer age-adjusted, fallback to crude)
outcomes_config = {
    'obesity': {
        'preferred': 'OBESITY_AgeAdjPrev',
        'fallback': 'OBESITY_CrudePrev',
        'label': 'Obesity'
    },
    'diabetes': {
        'preferred': 'DIABETES_AgeAdjPrev',
        'fallback': 'DIABETES_CrudePrev',
        'label': 'Diabetes'
    },
    'frequent_mental_distress': {
        'preferred': 'MHLTH_AgeAdjPrev',
        'fallback': 'MHLTH_CrudePrev',
        'label': 'Frequent Mental Distress'
    }
}

# Select actual outcome columns
outcome_vars = {}
for outcome_key, config in outcomes_config.items():
    if config['preferred'] in df.columns:
        outcome_vars[outcome_key] = {
            'column': config['preferred'],
            'label': config['label']
        }
        print(f"   {config['label']}: Using {config['preferred']} (age-adjusted)")
    elif config['fallback'] in df.columns:
        outcome_vars[outcome_key] = {
            'column': config['fallback'],
            'label': config['label']
        }
        print(f"   {config['label']}: Using {config['fallback']} (crude prevalence)")
    else:
        print(f"   Warning: {config['label']} - neither column found!")

if len(outcome_vars) == 0:
    raise ValueError("No health outcome variables found!")

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

# Check for population variable
if 'total_population' not in df.columns:
    raise ValueError("total_population variable not found - required for WLS weighting!")
print(f"   Population variable available: total_population")

# Check for demographic variables
demographic_vars = {
    'pct_under_18': 'pct_under_18',
    'pct_65_and_over': 'pct_65_and_over',
    'pct_non_hispanic_white': 'pct_non_hispanic_white',
    'pct_black': 'pct_black',
    'pct_hispanic': 'pct_hispanic'
}

available_demographics = {}
for key, col in demographic_vars.items():
    if col in df.columns:
        available_demographics[key] = col
        print(f"   Demographic available: {col}")
    else:
        print(f"   Warning: Demographic {col} not found!")

if len(available_demographics) == 0:
    print(f"   Warning: No demographic variables found - demographic models will be skipped")
else:
    print(f"   Demographic variables available: {len(available_demographics)}")

# Create subset of demographics without pct_non_hispanic_white (for reduced models)
available_demographics_reduced = {k: v for k, v in available_demographics.items() 
                                   if k != 'pct_non_hispanic_white'}
if len(available_demographics_reduced) > 0:
    print(f"   Reduced demographic variables (without pct_non_hispanic_white): {len(available_demographics_reduced)}")
    print(f"     Variables: {', '.join(available_demographics_reduced.values())}")

# ============================================================================
# 4. RUN WEIGHTED LEAST SQUARES (WLS) REGRESSIONS
# ============================================================================
print(f"\n4. Running Weighted Least Squares (WLS) regressions...")
print(f"   Weights: total_population")

results_list = []

for outcome_key, outcome_info in outcome_vars.items():
    outcome_col = outcome_info['column']
    outcome_label = outcome_info['label']
    
    print(f"\n   {'='*50}")
    print(f"   Outcome: {outcome_label} ({outcome_col})")
    print(f"   {'='*50}")
    
    # Prepare data for this outcome
    # Select variables: outcome + predictor + controls + population
    reg_vars = ['park_gravity_index_z'] + list(available_controls.values()) + [outcome_col, 'total_population']
    
    # Create subset with non-missing data
    df_outcome = df[reg_vars].copy()
    df_outcome = df_outcome.dropna()
    
    # Filter to positive population weights
    df_outcome = df_outcome[df_outcome['total_population'] > 0].copy()
    
    print(f"   Observations: {len(df_outcome):,}")
    
    if len(df_outcome) == 0:
        print(f"   Skipping - no valid observations")
        continue
    
    # Prepare dependent variable
    y = df_outcome[outcome_col].values
    
    # Prepare independent variables (add constant)
    X_vars = ['park_gravity_index_z'] + list(available_controls.values())
    X = df_outcome[X_vars].values
    X = sm.add_constant(X)  # Add intercept
    
    # Prepare weights
    weights = df_outcome['total_population'].values
    
    # Run WLS regression
    print(f"   Fitting WLS model...")
    model = sm.WLS(y, X, weights=weights)
    results = model.fit(cov_type='HC1')  # Robust standard errors (HC1)
    
    # Print summary
    print(f"\n   Regression Summary:")
    print(results.summary())
    
    # Extract results
    # Get confidence intervals
    ci_array = results.conf_int()
    # Handle both DataFrame and array cases
    if isinstance(ci_array, pd.DataFrame):
        ci_lower_vals = ci_array.iloc[:, 0].values
        ci_upper_vals = ci_array.iloc[:, 1].values
    else:
        # It's a numpy array
        ci_lower_vals = ci_array[:, 0]
        ci_upper_vals = ci_array[:, 1]
    
    coef_names = ['const'] + X_vars
    
    for i, var_name in enumerate(coef_names):
        coef = results.params[i]
        se = results.bse[i]
        pval = results.pvalues[i]
        # Access CI by position
        ci_lower = ci_lower_vals[i]
        ci_upper = ci_upper_vals[i]
        
        results_list.append({
            'outcome': outcome_label,
            'outcome_column': outcome_col,
            'predictor': var_name,
            'coefficient': coef,
            'std_error': se,
            'pvalue': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': len(df_outcome),
            'r_squared': results.rsquared,
            'model_type': 'WLS (Population-Weighted)'
        })
    
    # ============================================================================
    # 5. CALCULATE VARIANCE INFLATION FACTOR (VIF) FOR MULTICOLLINEARITY
    # ============================================================================
    print(f"\n   Variance Inflation Factor (VIF) Diagnostics:")
    print(f"   {'-'*50}")
    
    # Calculate VIF for all predictors (excluding constant)
    # VIF requires the design matrix without constant
    X_no_const = df_outcome[X_vars].values
    
    # Add constant column for VIF calculation (statsmodels expects it)
    X_with_const = sm.add_constant(X_no_const)
    
    # Calculate VIF for each variable
    vif_data = pd.DataFrame()
    vif_data['Variable'] = ['Intercept'] + X_vars
    vif_data['VIF'] = [variance_inflation_factor(X_with_const, i) for i in range(X_with_const.shape[1])]
    
    print(vif_data.to_string(index=False))
    
    # Interpretation
    print(f"\n   VIF Interpretation:")
    print(f"     VIF < 5: Low multicollinearity (acceptable)")
    print(f"     5 <= VIF < 10: Moderate multicollinearity (caution)")
    print(f"     VIF >= 10: High multicollinearity (problematic)")
    
    # Check VIF for predictors only (exclude intercept)
    predictor_vif = vif_data[vif_data['Variable'] != 'Intercept']
    high_vif = predictor_vif[predictor_vif['VIF'] >= 10]
    
    if len(high_vif) > 0:
        print(f"\n   WARNING: High multicollinearity detected in predictors:")
        for _, row in high_vif.iterrows():
            print(f"     {row['Variable']}: VIF = {row['VIF']:.2f}")
    else:
        print(f"   [OK] No high multicollinearity detected in predictors (all VIF < 10)")
        print(f"   Note: Intercept VIF is typically high and not a concern")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print(f"\n{'='*60}")
print("SAVING RESULTS")
print(f"{'='*60}")

if len(results_list) > 0:
    results_df = pd.DataFrame(results_list)
    
    # Add significance stars
    def add_significance(pvalue):
        if pd.isna(pvalue):
            return ''
        elif pvalue < 0.01:
            return '***'
        elif pvalue < 0.05:
            return '**'
        elif pvalue < 0.1:
            return '*'
        else:
            return ''
    
    results_df['significance'] = results_df['pvalue'].apply(add_significance)
    
    # Reorder columns
    column_order = [
        'outcome', 'predictor', 'coefficient', 'std_error', 'pvalue', 
        'significance', 'ci_lower', 'ci_upper', 'n_obs', 'r_squared', 'model_type'
    ]
    results_df = results_df[column_order]
    
    # Sort by outcome, then by predictor
    results_df = results_df.sort_values(['outcome', 'predictor'])
    
    print(f"\n   Results summary:")
    print(f"   Total model results: {len(results_df)}")
    print(f"\n   Results by outcome:")
    for outcome in results_df['outcome'].unique():
        n = len(results_df[results_df['outcome'] == outcome])
        print(f"     {outcome}: {n} coefficients")
    
    print(f"\n   Sample results:")
    print(results_df.head(10).to_string(index=False))
    
    # Save to CSV
    print(f"\n   Saving results to: {output_path}")
    results_df.to_csv(output_path, index=False)
    print(f"   [OK] Results saved successfully!")
    
    # Print key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    
    # Filter to park gravity index coefficients
    park_coefs = results_df[results_df['predictor'] == 'park_gravity_index_z'].copy()
    if len(park_coefs) > 0:
        print(f"\n   Park Gravity Index Effects (Z-score normalized):")
        for _, row in park_coefs.iterrows():
            sig = row['significance']
            coef = row['coefficient']
            pval = row['pvalue']
            print(f"     {row['outcome']}: {coef:+.4f} (p={pval:.4f}) {sig}")
            print(f"       Interpretation: 1 SD increase in Park Gravity Index -> "
                  f"{coef:+.2f} percentage point change in {row['outcome']}")
else:
    print(f"   No results to save!")

# ============================================================================
# 7. RUN WLS REGRESSIONS WITH POPULATION DENSITY CONTROL
# ============================================================================
print(f"\n{'='*60}")
print("RUNNING WLS REGRESSIONS WITH POPULATION DENSITY CONTROL")
print(f"{'='*60}")

# Check for log_pop_density variable
if 'log_pop_density' not in df.columns:
    print(f"   Warning: log_pop_density not found - skipping density models")
else:
    print(f"   Population density control available: log_pop_density")
    
    results_list_density = []
    
    for outcome_key, outcome_info in outcome_vars.items():
        outcome_col = outcome_info['column']
        outcome_label = outcome_info['label']
        
        print(f"\n   {'='*50}")
        print(f"   Outcome: {outcome_label} ({outcome_col}) - WITH DENSITY CONTROL")
        print(f"   {'='*50}")
        
        # Prepare data for this outcome
        # Select variables: outcome + predictor + controls + log_pop_density + population
        reg_vars_density = ['park_gravity_index_z'] + list(available_controls.values()) + ['log_pop_density', outcome_col, 'total_population']
        
        # Create subset with non-missing data
        df_outcome_density = df[reg_vars_density].copy()
        df_outcome_density = df_outcome_density.dropna()
        
        # Filter to positive population weights
        df_outcome_density = df_outcome_density[df_outcome_density['total_population'] > 0].copy()
        
        print(f"   Observations: {len(df_outcome_density):,}")
        
        if len(df_outcome_density) == 0:
            print(f"   Skipping - no valid observations")
            continue
        
        # Prepare dependent variable
        y_density = df_outcome_density[outcome_col].values
        
        # Prepare independent variables (add constant)
        X_vars_density = ['park_gravity_index_z'] + list(available_controls.values()) + ['log_pop_density']
        X_density = df_outcome_density[X_vars_density].values
        X_density = sm.add_constant(X_density)  # Add intercept
        
        # Prepare weights
        weights_density = df_outcome_density['total_population'].values
        
        # Run WLS regression
        print(f"   Fitting WLS model with density control...")
        model_density = sm.WLS(y_density, X_density, weights=weights_density)
        results_density = model_density.fit(cov_type='HC1')  # Robust standard errors (HC1)
        
        # Print summary
        print(f"\n   Regression Summary (with density control):")
        print(results_density.summary())
        
        # Extract results
        # Get confidence intervals
        ci_array_density = results_density.conf_int()
        # Handle both DataFrame and array cases
        if isinstance(ci_array_density, pd.DataFrame):
            ci_lower_vals_density = ci_array_density.iloc[:, 0].values
            ci_upper_vals_density = ci_array_density.iloc[:, 1].values
        else:
            # It's a numpy array
            ci_lower_vals_density = ci_array_density[:, 0]
            ci_upper_vals_density = ci_array_density[:, 1]
        
        coef_names_density = ['const'] + X_vars_density
        
        for i, var_name in enumerate(coef_names_density):
            coef = results_density.params[i]
            se = results_density.bse[i]
            pval = results_density.pvalues[i]
            # Access CI by position
            ci_lower = ci_lower_vals_density[i]
            ci_upper = ci_upper_vals_density[i]
            
            results_list_density.append({
                'outcome': outcome_label,
                'outcome_column': outcome_col,
                'predictor': var_name,
                'coefficient': coef,
                'std_error': se,
                'pvalue': pval,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_obs': len(df_outcome_density),
                'r_squared': results_density.rsquared,
                'model_type': 'WLS (Population-Weighted, with Density Control)'
            })
        
        # Calculate VIF for models with density
        print(f"\n   Variance Inflation Factor (VIF) Diagnostics (with density):")
        print(f"   {'-'*50}")
        
        X_no_const_density = df_outcome_density[X_vars_density].values
        X_with_const_density = sm.add_constant(X_no_const_density)
        
        vif_data_density = pd.DataFrame()
        vif_data_density['Variable'] = ['Intercept'] + X_vars_density
        vif_data_density['VIF'] = [variance_inflation_factor(X_with_const_density, i) for i in range(X_with_const_density.shape[1])]
        
        print(vif_data_density.to_string(index=False))
        
        predictor_vif_density = vif_data_density[vif_data_density['Variable'] != 'Intercept']
        high_vif_density = predictor_vif_density[predictor_vif_density['VIF'] >= 10]
        
        if len(high_vif_density) > 0:
            print(f"\n   WARNING: High multicollinearity detected in predictors:")
            for _, row in high_vif_density.iterrows():
                print(f"     {row['Variable']}: VIF = {row['VIF']:.2f}")
        else:
            print(f"   [OK] No high multicollinearity detected in predictors (all VIF < 10)")
    
    # Save density-controlled results
    if len(results_list_density) > 0:
        print(f"\n{'='*60}")
        print("SAVING RESULTS WITH DENSITY CONTROL")
        print(f"{'='*60}")
        
        results_df_density = pd.DataFrame(results_list_density)
        
        # Add significance stars
        results_df_density['significance'] = results_df_density['pvalue'].apply(add_significance)
        
        # Reorder columns
        results_df_density = results_df_density[column_order]
        
        # Sort by outcome, then by predictor
        results_df_density = results_df_density.sort_values(['outcome', 'predictor'])
        
        print(f"\n   Results summary (with density control):")
        print(f"   Total model results: {len(results_df_density)}")
        print(f"\n   Results by outcome:")
        for outcome in results_df_density['outcome'].unique():
            n = len(results_df_density[results_df_density['outcome'] == outcome])
            print(f"     {outcome}: {n} coefficients")
        
        print(f"\n   Sample results (with density control):")
        print(results_df_density.head(10).to_string(index=False))
        
        # Save to CSV
        print(f"\n   Saving results to: {output_path_density}")
        results_df_density.to_csv(output_path_density, index=False)
        print(f"   [OK] Results with density control saved successfully!")
        
        # Print key findings for density models
        print(f"\n{'='*60}")
        print("KEY FINDINGS (WITH DENSITY CONTROL)")
        print(f"{'='*60}")
        
        park_coefs_density = results_df_density[results_df_density['predictor'] == 'park_gravity_index_z'].copy()
        if len(park_coefs_density) > 0:
            print(f"\n   Park Gravity Index Effects (with density control, Z-score normalized):")
            for _, row in park_coefs_density.iterrows():
                sig = row['significance']
                coef = row['coefficient']
                pval = row['pvalue']
                print(f"     {row['outcome']}: {coef:+.4f} (p={pval:.4f}) {sig}")
                print(f"       Interpretation: 1 SD increase in Park Gravity Index -> "
                      f"{coef:+.2f} percentage point change in {row['outcome']}")

# ============================================================================
# 8. RUN WLS REGRESSIONS WITH POPULATION DENSITY AND DEMOGRAPHIC CONTROLS
# ============================================================================
print(f"\n{'='*60}")
print("RUNNING WLS REGRESSIONS WITH DENSITY AND DEMOGRAPHIC CONTROLS")
print(f"{'='*60}")

# Check for required variables
if 'log_pop_density' not in df.columns:
    print(f"   Warning: log_pop_density not found - skipping density+demographics models")
elif len(available_demographics) == 0:
    print(f"   Warning: No demographic variables found - skipping density+demographics models")
else:
    print(f"   Population density control available: log_pop_density")
    print(f"   Demographic controls available: {', '.join(available_demographics.values())}")
    
    results_list_density_demo = []
    
    for outcome_key, outcome_info in outcome_vars.items():
        outcome_col = outcome_info['column']
        outcome_label = outcome_info['label']
        
        print(f"\n   {'='*50}")
        print(f"   Outcome: {outcome_label} ({outcome_col}) - WITH DENSITY + DEMOGRAPHICS")
        print(f"   {'='*50}")
        
        # Prepare data for this outcome
        # Select variables: outcome + predictor + controls + log_pop_density + demographics + population
        reg_vars_density_demo = (['park_gravity_index_z'] + list(available_controls.values()) + 
                                 ['log_pop_density'] + list(available_demographics.values()) + 
                                 [outcome_col, 'total_population'])
        
        # Create subset with non-missing data
        df_outcome_density_demo = df[reg_vars_density_demo].copy()
        df_outcome_density_demo = df_outcome_density_demo.dropna()
        
        # Filter to positive population weights
        df_outcome_density_demo = df_outcome_density_demo[df_outcome_density_demo['total_population'] > 0].copy()
        
        print(f"   Observations: {len(df_outcome_density_demo):,}")
        
        if len(df_outcome_density_demo) == 0:
            print(f"   Skipping - no valid observations")
            continue
        
        # Prepare dependent variable
        y_density_demo = df_outcome_density_demo[outcome_col].values
        
        # Prepare independent variables (add constant)
        X_vars_density_demo = (['park_gravity_index_z'] + list(available_controls.values()) + 
                               ['log_pop_density'] + list(available_demographics.values()))
        X_density_demo = df_outcome_density_demo[X_vars_density_demo].values
        X_density_demo = sm.add_constant(X_density_demo)  # Add intercept
        
        # Prepare weights
        weights_density_demo = df_outcome_density_demo['total_population'].values
        
        # Run WLS regression
        print(f"   Fitting WLS model with density and demographic controls...")
        model_density_demo = sm.WLS(y_density_demo, X_density_demo, weights=weights_density_demo)
        results_density_demo = model_density_demo.fit(cov_type='HC1')  # Robust standard errors (HC1)
        
        # Print summary
        print(f"\n   Regression Summary (with density + demographics):")
        print(results_density_demo.summary())
        
        # Extract results
        # Get confidence intervals
        ci_array_density_demo = results_density_demo.conf_int()
        # Handle both DataFrame and array cases
        if isinstance(ci_array_density_demo, pd.DataFrame):
            ci_lower_vals_density_demo = ci_array_density_demo.iloc[:, 0].values
            ci_upper_vals_density_demo = ci_array_density_demo.iloc[:, 1].values
        else:
            # It's a numpy array
            ci_lower_vals_density_demo = ci_array_density_demo[:, 0]
            ci_upper_vals_density_demo = ci_array_density_demo[:, 1]
        
        coef_names_density_demo = ['const'] + X_vars_density_demo
        
        for i, var_name in enumerate(coef_names_density_demo):
            coef = results_density_demo.params[i]
            se = results_density_demo.bse[i]
            pval = results_density_demo.pvalues[i]
            # Access CI by position
            ci_lower = ci_lower_vals_density_demo[i]
            ci_upper = ci_upper_vals_density_demo[i]
            
            results_list_density_demo.append({
                'outcome': outcome_label,
                'outcome_column': outcome_col,
                'predictor': var_name,
                'coefficient': coef,
                'std_error': se,
                'pvalue': pval,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_obs': len(df_outcome_density_demo),
                'r_squared': results_density_demo.rsquared,
                'model_type': 'WLS (Population-Weighted, with Density + Demographic Controls)'
            })
        
        # Calculate VIF for models with density and demographics
        print(f"\n   Variance Inflation Factor (VIF) Diagnostics (with density + demographics):")
        print(f"   {'-'*50}")
        
        X_no_const_density_demo = df_outcome_density_demo[X_vars_density_demo].values
        X_with_const_density_demo = sm.add_constant(X_no_const_density_demo)
        
        vif_data_density_demo = pd.DataFrame()
        vif_data_density_demo['Variable'] = ['Intercept'] + X_vars_density_demo
        vif_data_density_demo['VIF'] = [variance_inflation_factor(X_with_const_density_demo, i) 
                                        for i in range(X_with_const_density_demo.shape[1])]
        
        print(vif_data_density_demo.to_string(index=False))
        
        predictor_vif_density_demo = vif_data_density_demo[vif_data_density_demo['Variable'] != 'Intercept']
        high_vif_density_demo = predictor_vif_density_demo[predictor_vif_density_demo['VIF'] >= 10]
        
        if len(high_vif_density_demo) > 0:
            print(f"\n   WARNING: High multicollinearity detected in predictors:")
            for _, row in high_vif_density_demo.iterrows():
                print(f"     {row['Variable']}: VIF = {row['VIF']:.2f}")
        else:
            print(f"   [OK] No high multicollinearity detected in predictors (all VIF < 10)")
    
    # Save density+demographics results
    if len(results_list_density_demo) > 0:
        print(f"\n{'='*60}")
        print("SAVING RESULTS WITH DENSITY + DEMOGRAPHIC CONTROLS")
        print(f"{'='*60}")
        
        results_df_density_demo = pd.DataFrame(results_list_density_demo)
        
        # Add significance stars
        results_df_density_demo['significance'] = results_df_density_demo['pvalue'].apply(add_significance)
        
        # Reorder columns
        results_df_density_demo = results_df_density_demo[column_order]
        
        # Sort by outcome, then by predictor
        results_df_density_demo = results_df_density_demo.sort_values(['outcome', 'predictor'])
        
        print(f"\n   Results summary (with density + demographics):")
        print(f"   Total model results: {len(results_df_density_demo)}")
        print(f"\n   Results by outcome:")
        for outcome in results_df_density_demo['outcome'].unique():
            n = len(results_df_density_demo[results_df_density_demo['outcome'] == outcome])
            print(f"     {outcome}: {n} coefficients")
        
        print(f"\n   Sample results (with density + demographics):")
        print(results_df_density_demo.head(10).to_string(index=False))
        
        # Save to CSV
        print(f"\n   Saving results to: {output_path_density_demo}")
        results_df_density_demo.to_csv(output_path_density_demo, index=False)
        print(f"   [OK] Results with density + demographics saved successfully!")
        
        # Print key findings for density+demographics models
        print(f"\n{'='*60}")
        print("KEY FINDINGS (WITH DENSITY + DEMOGRAPHIC CONTROLS)")
        print(f"{'='*60}")
        
        park_coefs_density_demo = results_df_density_demo[results_df_density_demo['predictor'] == 'park_gravity_index_z'].copy()
        if len(park_coefs_density_demo) > 0:
            print(f"\n   Park Gravity Index Effects (with density + demographics, Z-score normalized):")
            for _, row in park_coefs_density_demo.iterrows():
                sig = row['significance']
                coef = row['coefficient']
                pval = row['pvalue']
                print(f"     {row['outcome']}: {coef:+.4f} (p={pval:.4f}) {sig}")
                print(f"       Interpretation: 1 SD increase in Park Gravity Index -> "
                      f"{coef:+.2f} percentage point change in {row['outcome']}")

# ============================================================================
# 9. RUN WLS REGRESSIONS WITH DENSITY + DEMOGRAPHICS (EXCLUDING pct_non_hispanic_white)
# ============================================================================
print(f"\n{'='*60}")
print("RUNNING WLS REGRESSIONS WITH DENSITY + DEMOGRAPHICS (NH WHITE DROPPED)")
print(f"{'='*60}")

# Check for required variables
if 'log_pop_density' not in df.columns:
    print(f"   Warning: log_pop_density not found - skipping reduced demographics models")
elif len(available_demographics_reduced) == 0:
    print(f"   Warning: No reduced demographic variables found - skipping models")
else:
    print(f"   Population density control available: log_pop_density")
    print(f"   Reduced demographic controls (excluding pct_non_hispanic_white): {', '.join(available_demographics_reduced.values())}")
    
    results_list_density_demo_reduced = []
    vif_tables_list = []  # Store VIF tables for each outcome
    
    for outcome_key, outcome_info in outcome_vars.items():
        outcome_col = outcome_info['column']
        outcome_label = outcome_info['label']
        
        print(f"\n   {'='*50}")
        print(f"   Outcome: {outcome_label} ({outcome_col}) - WITH DENSITY + REDUCED DEMOGRAPHICS")
        print(f"   {'='*50}")
        
        # Prepare data for this outcome
        # Select variables: outcome + predictor + controls + log_pop_density + reduced demographics + population
        reg_vars_density_demo_reduced = (['park_gravity_index_z'] + list(available_controls.values()) + 
                                         ['log_pop_density'] + list(available_demographics_reduced.values()) + 
                                         [outcome_col, 'total_population'])
        
        # Create subset with non-missing data
        df_outcome_density_demo_reduced = df[reg_vars_density_demo_reduced].copy()
        df_outcome_density_demo_reduced = df_outcome_density_demo_reduced.dropna()
        
        # Filter to positive population weights
        df_outcome_density_demo_reduced = df_outcome_density_demo_reduced[df_outcome_density_demo_reduced['total_population'] > 0].copy()
        
        print(f"   Observations: {len(df_outcome_density_demo_reduced):,}")
        
        if len(df_outcome_density_demo_reduced) == 0:
            print(f"   Skipping - no valid observations")
            continue
        
        # Prepare dependent variable
        y_density_demo_reduced = df_outcome_density_demo_reduced[outcome_col].values
        
        # Prepare independent variables (add constant)
        X_vars_density_demo_reduced = (['park_gravity_index_z'] + list(available_controls.values()) + 
                                       ['log_pop_density'] + list(available_demographics_reduced.values()))
        X_density_demo_reduced = df_outcome_density_demo_reduced[X_vars_density_demo_reduced].values
        X_density_demo_reduced = sm.add_constant(X_density_demo_reduced)  # Add intercept
        
        # Prepare weights
        weights_density_demo_reduced = df_outcome_density_demo_reduced['total_population'].values
        
        # Run WLS regression
        print(f"   Fitting WLS model with density and reduced demographic controls...")
        model_density_demo_reduced = sm.WLS(y_density_demo_reduced, X_density_demo_reduced, weights=weights_density_demo_reduced)
        results_density_demo_reduced = model_density_demo_reduced.fit(cov_type='HC1')  # Robust standard errors (HC1)
        
        # Print summary
        print(f"\n   Regression Summary (with density + reduced demographics):")
        print(results_density_demo_reduced.summary())
        
        # Extract results
        # Get confidence intervals
        ci_array_density_demo_reduced = results_density_demo_reduced.conf_int()
        # Handle both DataFrame and array cases
        if isinstance(ci_array_density_demo_reduced, pd.DataFrame):
            ci_lower_vals_density_demo_reduced = ci_array_density_demo_reduced.iloc[:, 0].values
            ci_upper_vals_density_demo_reduced = ci_array_density_demo_reduced.iloc[:, 1].values
        else:
            # It's a numpy array
            ci_lower_vals_density_demo_reduced = ci_array_density_demo_reduced[:, 0]
            ci_upper_vals_density_demo_reduced = ci_array_density_demo_reduced[:, 1]
        
        coef_names_density_demo_reduced = ['const'] + X_vars_density_demo_reduced
        
        for i, var_name in enumerate(coef_names_density_demo_reduced):
            coef = results_density_demo_reduced.params[i]
            se = results_density_demo_reduced.bse[i]
            pval = results_density_demo_reduced.pvalues[i]
            # Access CI by position
            ci_lower = ci_lower_vals_density_demo_reduced[i]
            ci_upper = ci_upper_vals_density_demo_reduced[i]
            
            results_list_density_demo_reduced.append({
                'outcome': outcome_label,
                'outcome_column': outcome_col,
                'predictor': var_name,
                'coefficient': coef,
                'std_error': se,
                'pvalue': pval,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_obs': len(df_outcome_density_demo_reduced),
                'r_squared': results_density_demo_reduced.rsquared,
                'model_type': 'WLS (Population-Weighted, with Density + Reduced Demographic Controls)'
            })
        
        # Calculate VIF for models with density and reduced demographics
        print(f"\n   Variance Inflation Factor (VIF) Diagnostics (with density + reduced demographics):")
        print(f"   {'-'*50}")
        
        X_no_const_density_demo_reduced = df_outcome_density_demo_reduced[X_vars_density_demo_reduced].values
        X_with_const_density_demo_reduced = sm.add_constant(X_no_const_density_demo_reduced)
        
        vif_data_density_demo_reduced = pd.DataFrame()
        vif_data_density_demo_reduced['Variable'] = ['Intercept'] + X_vars_density_demo_reduced
        vif_data_density_demo_reduced['VIF'] = [variance_inflation_factor(X_with_const_density_demo_reduced, i) 
                                                for i in range(X_with_const_density_demo_reduced.shape[1])]
        
        # Add outcome label to VIF table for tracking
        vif_data_density_demo_reduced['outcome'] = outcome_label
        
        print(vif_data_density_demo_reduced.to_string(index=False))
        
        # Store VIF table for saving
        vif_tables_list.append(vif_data_density_demo_reduced)
        
        predictor_vif_density_demo_reduced = vif_data_density_demo_reduced[vif_data_density_demo_reduced['Variable'] != 'Intercept']
        high_vif_density_demo_reduced = predictor_vif_density_demo_reduced[predictor_vif_density_demo_reduced['VIF'] >= 10]
        
        if len(high_vif_density_demo_reduced) > 0:
            print(f"\n   WARNING: High multicollinearity detected in predictors:")
            for _, row in high_vif_density_demo_reduced.iterrows():
                print(f"     {row['Variable']}: VIF = {row['VIF']:.2f}")
        else:
            print(f"   [OK] No high multicollinearity detected in predictors (all VIF < 10)")
    
    # Save density+reduced demographics results
    if len(results_list_density_demo_reduced) > 0:
        print(f"\n{'='*60}")
        print("SAVING RESULTS WITH DENSITY + REDUCED DEMOGRAPHIC CONTROLS")
        print(f"{'='*60}")
        
        results_df_density_demo_reduced = pd.DataFrame(results_list_density_demo_reduced)
        
        # Add significance stars
        results_df_density_demo_reduced['significance'] = results_df_density_demo_reduced['pvalue'].apply(add_significance)
        
        # Reorder columns
        results_df_density_demo_reduced = results_df_density_demo_reduced[column_order]
        
        # Sort by outcome, then by predictor
        results_df_density_demo_reduced = results_df_density_demo_reduced.sort_values(['outcome', 'predictor'])
        
        print(f"\n   Results summary (with density + reduced demographics):")
        print(f"   Total model results: {len(results_df_density_demo_reduced)}")
        print(f"\n   Results by outcome:")
        for outcome in results_df_density_demo_reduced['outcome'].unique():
            n = len(results_df_density_demo_reduced[results_df_density_demo_reduced['outcome'] == outcome])
            print(f"     {outcome}: {n} coefficients")
        
        print(f"\n   Sample results (with density + reduced demographics):")
        print(results_df_density_demo_reduced.head(10).to_string(index=False))
        
        # Save to CSV
        print(f"\n   Saving results to: {output_path_density_demo_nhwhite_dropped}")
        results_df_density_demo_reduced.to_csv(output_path_density_demo_nhwhite_dropped, index=False)
        print(f"   [OK] Results with density + reduced demographics saved successfully!")
        
        # Save VIF tables
        if len(vif_tables_list) > 0:
            vif_tables_df = pd.concat(vif_tables_list, ignore_index=True)
            vif_output_path = os.path.join(output_dir, 'park_gravity_index_vif_density_demographics_nhwhite_dropped.csv')
            print(f"\n   Saving VIF table to: {vif_output_path}")
            vif_tables_df.to_csv(vif_output_path, index=False)
            print(f"   [OK] VIF table saved successfully!")
        
        # Print key findings for density+reduced demographics models
        print(f"\n{'='*60}")
        print("KEY FINDINGS (WITH DENSITY + REDUCED DEMOGRAPHIC CONTROLS)")
        print(f"{'='*60}")
        
        park_coefs_density_demo_reduced = results_df_density_demo_reduced[results_df_density_demo_reduced['predictor'] == 'park_gravity_index_z'].copy()
        if len(park_coefs_density_demo_reduced) > 0:
            print(f"\n   Park Gravity Index Effects (with density + reduced demographics, Z-score normalized):")
            for _, row in park_coefs_density_demo_reduced.iterrows():
                sig = row['significance']
                coef = row['coefficient']
                pval = row['pvalue']
                print(f"     {row['outcome']}: {coef:+.4f} (p={pval:.4f}) {sig}")
                print(f"       Interpretation: 1 SD increase in Park Gravity Index -> "
                      f"{coef:+.2f} percentage point change in {row['outcome']}")

print(f"\n{'='*60}")
print("PARK GRAVITY INDEX REGRESSION ANALYSIS COMPLETE")
print(f"{'='*60}")
