"""
Hierarchical Health Outcomes Regression Models
Economic Impact Report for Butler County Parks

This script runs a hierarchical series of nested regression models for each health outcome:
- Model 0 (Base): Socioeconomic covariates only
- Model 1 (Proximity): Base + dist_to_park + park_acres_10min
- Model 2 (Greenness): Model 1 + ndvi_mean_500m
- Model 3 (Quality): Model 2 + avg_amenity_score_10min + avg_inclusiveness_index_10min
- Model 4 (Spatial): Model 3 with Spatial Error/Lag terms (if available)

Phase 2 Models (with Population Density Control):
- Phase 2 Model 1: Proximity + log_pop_density
- Phase 2 Model 2: Greenness + log_pop_density
- Phase 2 Model 3: Quality + log_pop_density

Interaction Models (Park Access × Poverty):
- Interaction Model 1: Proximity × Poverty
- Interaction Model 2: Greenness × Poverty
- Interaction Model 3: Quality × Poverty

Health Outcomes:
- Obesity
- Mental Health (Frequent Mental Distress)
- Physical Inactivity
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import spatial regression libraries
try:
    from libpysal.weights import Queen
    from spreg import ML_Lag, ML_Error
    HAS_SPATIAL = True
except ImportError:
    try:
        # Try alternative import paths
        import libpysal
        from libpysal.weights import Queen
        from spreg import ML_Lag, ML_Error
        HAS_SPATIAL = True
    except ImportError:
        HAS_SPATIAL = False
        if __name__ == '__main__':
            print("   Note: Spatial regression libraries not available. Model 4 will be skipped.")
            print("   Install with: pip install libpysal spreg")

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input paths
input_path = os.path.join(project_root, 'data_final', 'butler_tract_health_model_data_with_greenness.csv')
input_gpkg_path = os.path.join(project_root, 'data_final', 'butler_tract_health_model_data_with_greenness.gpkg')
quality_metrics_path = os.path.join(project_root, 'data_processed', 'tract_park_quality_metrics.csv')

# Output paths
output_dir = os.path.join(project_root, 'results')
output_path = os.path.join(output_dir, 'hierarchical_health_regressions.csv')

print("="*80)
print("HIERARCHICAL HEALTH OUTCOMES REGRESSION MODELS")
print("="*80)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print(f"\n1. Loading data...")

# Try GPKG first (for spatial models), then CSV
if os.path.exists(input_gpkg_path):
    import geopandas as gpd
    print(f"   Loading from GPKG: {input_gpkg_path}")
    df = gpd.read_file(input_gpkg_path)
    has_geometry = True
    print(f"   Records loaded: {len(df):,}")
    print(f"   CRS: {df.crs}")
elif os.path.exists(input_path):
    print(f"   Loading from CSV: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    has_geometry = False
    print(f"   Records loaded: {len(df):,}")
else:
    raise FileNotFoundError(f"Data file not found: {input_path}")

# Load quality metrics
if os.path.exists(quality_metrics_path):
    print(f"   Loading quality metrics: {quality_metrics_path}")
    quality_df = pd.read_csv(quality_metrics_path)
    quality_df['GEOID'] = quality_df['GEOID'].astype(str)
    df['GEOID'] = df['GEOID'].astype(str)
    df = df.merge(quality_df, on='GEOID', how='left')
    print(f"   Quality metrics merged")
else:
    print(f"   Warning: Quality metrics not found. Model 3 will use alternative measures.")

print(f"   Total columns: {len(df.columns)}")

# ============================================================================
# 1b. CALCULATE POPULATION DENSITY (if not present)
# ============================================================================
print(f"\n1b. Calculating population density...")

if 'log_pop_density' not in df.columns:
    if has_geometry:
        # Calculate from geometry (if GeoDataFrame)
        import geopandas as gpd
        if isinstance(df, gpd.GeoDataFrame):
            # Calculate tract area in square feet
            df['tract_area_sqft'] = df.geometry.area
            # Convert square feet to square miles
            # 1 square mile = 27,878,400 square US Survey Feet
            SQFT_TO_SQMI = 1.0 / 27878400.0
            df['tract_area_sqmi'] = df['tract_area_sqft'] * SQFT_TO_SQMI
        else:
            # If DataFrame but has geometry column, try to use it
            if 'geometry' in df.columns:
                df['tract_area_sqft'] = df.geometry.area
                SQFT_TO_SQMI = 1.0 / 27878400.0
                df['tract_area_sqmi'] = df['tract_area_sqft'] * SQFT_TO_SQMI
            else:
                # Use ALAND (land area in square meters) from CSV
                if 'ALAND' in df.columns:
                    # Convert square meters to square miles
                    # 1 square mile = 2,589,988.11 square meters
                    SQM_TO_SQMI = 1.0 / 2589988.11
                    df['tract_area_sqmi'] = df['ALAND'] * SQM_TO_SQMI
                else:
                    print(f"   Warning: Cannot calculate tract area - no geometry or ALAND column")
                    df['tract_area_sqmi'] = np.nan
    else:
        # Use ALAND from CSV
        if 'ALAND' in df.columns:
            SQM_TO_SQMI = 1.0 / 2589988.11
            df['tract_area_sqmi'] = df['ALAND'] * SQM_TO_SQMI
        else:
            print(f"   Warning: Cannot calculate tract area - no ALAND column")
            df['tract_area_sqmi'] = np.nan
    
    # Calculate population density
    if 'total_population' in df.columns and 'tract_area_sqmi' in df.columns:
        df['pop_density'] = df['total_population'] / df['tract_area_sqmi']
        # Calculate log of population density (add small epsilon to avoid log(0))
        epsilon = 0.001
        df['log_pop_density'] = np.log(df['pop_density'] + epsilon)
        print(f"   Population density calculated")
        print(f"     Mean density: {df['pop_density'].mean():.2f} people/sq mi")
        print(f"     Mean log density: {df['log_pop_density'].mean():.4f}")
    else:
        print(f"   Warning: Cannot calculate population density - missing total_population or tract_area_sqmi")
        df['log_pop_density'] = np.nan
else:
    print(f"   Population density already present in data")

# ============================================================================
# 2. DEFINE HEALTH OUTCOMES
# ============================================================================
print(f"\n2. Defining health outcomes...")

outcomes_config = {
    'obesity': {
        'preferred': 'OBESITY_AgeAdjPrev',
        'fallback': 'OBESITY_CrudePrev',
        'label': 'Obesity'
    },
    'mental_health': {
        'preferred': 'MHLTH_AgeAdjPrev',
        'fallback': 'MHLTH_CrudePrev',
        'label': 'Frequent Mental Distress'
    },
    'physical_inactivity': {
        'preferred': 'LPA_AgeAdjPrev',
        'fallback': 'LPA_CrudePrev',
        'label': 'Physical Inactivity'
    }
}

# Select actual outcome columns
outcome_vars = {}
for outcome_key, config in outcomes_config.items():
    if config['preferred'] in df.columns:
        outcome_vars[outcome_key] = {
            'column': config['preferred'],
            'label': config['label'],
            'type': 'AgeAdjPrev'
        }
        print(f"   {config['label']}: Using {config['preferred']} (age-adjusted)")
    elif config['fallback'] in df.columns:
        outcome_vars[outcome_key] = {
            'column': config['fallback'],
            'label': config['label'],
            'type': 'CrudePrev'
        }
        print(f"   {config['label']}: Using {config['fallback']} (crude prevalence)")
    else:
        print(f"   Warning: {config['label']} - neither column found!")

if len(outcome_vars) == 0:
    raise ValueError("No health outcome variables found!")

# ============================================================================
# 3. DEFINE MODEL VARIABLES
# ============================================================================
print(f"\n3. Defining model variables...")

# Base covariates (Model 0)
base_covariates = {
    'median_household_income': 'median_household_income',
    'pct_families_below_poverty': 'pct_families_below_poverty',
    'unemployment_rate': 'unemployment_rate',
    'pct_bachelors_degree_or_higher': 'pct_bachelors_degree_or_higher',
    'pct_non_hispanic_white': 'pct_non_hispanic_white',
    'pct_black': 'pct_black',
    'pct_hispanic': 'pct_hispanic',
    'pct_under_18': 'pct_under_18',
    'pct_65_and_over': 'pct_65_and_over'
}

# Model 1 variables (Proximity)
model1_vars = {
    'dist_to_park_miles': 'dist_to_park_miles',
    'park_acres_10min': 'park_acres_10min'
}

# Model 2 variables (Greenness)
model2_vars = {
    'ndvi_mean_500m': 'ndvi_mean_500m'
}

# Model 3 variables (Quality)
model3_vars = {
    'avg_amenity_score_10min': 'avg_amenity_score_10min',
    'avg_inclusiveness_index_10min': 'avg_inclusiveness_index_10min'
}

# Check which variables are available
available_base = {}
for key, col in base_covariates.items():
    if col in df.columns:
        available_base[key] = col
    else:
        print(f"   Warning: Base covariate {col} not found!")

available_model1 = {}
for key, col in model1_vars.items():
    if col in df.columns:
        available_model1[key] = col
    else:
        print(f"   Warning: Model 1 variable {col} not found!")

available_model2 = {}
for key, col in model2_vars.items():
    if col in df.columns:
        available_model2[key] = col
    else:
        print(f"   Warning: Model 2 variable {col} not found!")

available_model3 = {}
for key, col in model3_vars.items():
    if col in df.columns:
        available_model3[key] = col
    else:
        print(f"   Warning: Model 3 variable {col} not found!")
        # Fallback: use quality-weighted walkshed if available
        if key == 'avg_amenity_score_10min' and 'quality_weighted_walkshed_10min' in df.columns:
            print(f"     Using quality_weighted_walkshed_10min as proxy")

print(f"\n   Available variables:")
print(f"     Base covariates: {len(available_base)}")
print(f"     Model 1 (Proximity): {len(available_model1)}")
print(f"     Model 2 (Greenness): {len(available_model2)}")
print(f"     Model 3 (Quality): {len(available_model3)}")

# ============================================================================
# 4. HELPER FUNCTION TO FIT MODELS
# ============================================================================
def fit_model(y, X, model_name, outcome_label, model_num):
    """Fit OLS model and extract results"""
    try:
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit(cov_type='HC1')
        
        results = []
        for var in X.columns:
            if var in model.params.index:
                results.append({
                    'outcome': outcome_label,
                    'model': model_name,
                    'model_num': model_num,
                    'variable': var,
                    'coefficient': model.params[var],
                    'std_error': model.bse[var],
                    'pvalue': model.pvalues[var],
                    'ci_lower': model.conf_int().loc[var, 0],
                    'ci_upper': model.conf_int().loc[var, 1],
                    'n_obs': len(y),
                    'r_squared': model.rsquared,
                    'aic': model.aic,
                    'bic': model.bic
                })
        
        # Add model-level statistics
        results.append({
            'outcome': outcome_label,
            'model': model_name,
            'model_num': model_num,
            'variable': '_model_stats',
            'coefficient': np.nan,
            'std_error': np.nan,
            'pvalue': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': len(y),
            'r_squared': model.rsquared,
            'aic': model.aic,
            'bic': model.bic
        })
        
        return results, model
    except Exception as e:
        print(f"      Error fitting {model_name}: {e}")
        return [], None

# ============================================================================
# 5. RUN HIERARCHICAL MODELS FOR EACH OUTCOME
# ============================================================================
print(f"\n5. Running hierarchical regression models...")

all_results = []

for outcome_key, outcome_info in outcome_vars.items():
    outcome_col = outcome_info['column']
    outcome_label = outcome_info['label']
    
    print(f"\n   {'='*70}")
    print(f"   Outcome: {outcome_label} ({outcome_col})")
    print(f"   {'='*70}")
    
    # Create subset with non-missing outcome and all required variables
    required_vars = [outcome_col] + list(available_base.values())
    df_outcome = df[df[outcome_col].notna()].copy()
    
    # Drop rows with missing base covariates
    df_outcome = df_outcome.dropna(subset=required_vars)
    
    if len(df_outcome) == 0:
        print(f"   Skipping - no valid observations")
        continue
    
    print(f"   Observations: {len(df_outcome):,}")
    
    # Prepare dependent variable
    y = df_outcome[outcome_col].values
    
    # ========================================================================
    # MODEL 0: BASE (Socioeconomic covariates only)
    # ========================================================================
    print(f"\n   Model 0 (Base): Socioeconomic covariates only")
    
    X_base = df_outcome[list(available_base.values())].values
    X_base_df = df_outcome[list(available_base.values())].copy()
    
    results_base, model_base = fit_model(
        y, X_base_df, 'Model 0 (Base)', outcome_label, 0
    )
    all_results.extend(results_base)
    
    if model_base is not None:
        print(f"      R-squared: {model_base.rsquared:.4f}")
        print(f"      AIC: {model_base.aic:.2f}")
    
    # ========================================================================
    # MODEL 1: PROXIMITY (Base + dist_to_park + park_acres_10min)
    # ========================================================================
    if len(available_model1) > 0:
        print(f"\n   Model 1 (Proximity): Base + dist_to_park + park_acres_10min")
        
        model1_vars_list = list(available_base.values()) + list(available_model1.values())
        df_model1 = df_outcome.dropna(subset=model1_vars_list)
        
        if len(df_model1) > 0:
            y1 = df_model1[outcome_col].values
            X1_df = df_model1[model1_vars_list].copy()
            
            results_model1, model1 = fit_model(
                y1, X1_df, 'Model 1 (Proximity)', outcome_label, 1
            )
            all_results.extend(results_model1)
            
            if model1 is not None:
                print(f"      R-squared: {model1.rsquared:.4f}")
                print(f"      AIC: {model1.aic:.2f}")
                if 'park_acres_10min' in model1.params.index:
                    coef = model1.params['park_acres_10min']
                    pval = model1.pvalues['park_acres_10min']
                    print(f"      park_acres_10min: {coef:.4f} (p={pval:.4f})")
        else:
            print(f"      Skipping - no valid observations after adding Model 1 variables")
    else:
        print(f"      Skipping - Model 1 variables not available")
    
    # ========================================================================
    # MODEL 2: GREENNESS (Model 1 + ndvi_mean_500m)
    # ========================================================================
    if len(available_model2) > 0 and len(available_model1) > 0:
        print(f"\n   Model 2 (Greenness): Model 1 + ndvi_mean_500m")
        
        model2_vars_list = list(available_base.values()) + list(available_model1.values()) + list(available_model2.values())
        df_model2 = df_outcome.dropna(subset=model2_vars_list)
        
        if len(df_model2) > 0:
            y2 = df_model2[outcome_col].values
            X2_df = df_model2[model2_vars_list].copy()
            
            results_model2, model2 = fit_model(
                y2, X2_df, 'Model 2 (Greenness)', outcome_label, 2
            )
            all_results.extend(results_model2)
            
            if model2 is not None:
                print(f"      R-squared: {model2.rsquared:.4f}")
                print(f"      AIC: {model2.aic:.2f}")
                if 'ndvi_mean_500m' in model2.params.index:
                    coef = model2.params['ndvi_mean_500m']
                    pval = model2.pvalues['ndvi_mean_500m']
                    print(f"      ndvi_mean_500m: {coef:.4f} (p={pval:.4f})")
        else:
            print(f"      Skipping - no valid observations after adding Model 2 variables")
    else:
        print(f"      Skipping - Model 2 variables not available")
    
    # ========================================================================
    # MODEL 3: QUALITY (Model 2 + Quality-Weighted Walkshed)
    # ========================================================================
    if len(available_model2) > 0:
        print(f"\n   Model 3 (Quality): Model 2 + Quality-Weighted Walkshed")
        print(f"      Note: Quality-weighted walkshed incorporates park attractiveness")
        print(f"      (Size × Amenity Score × Inclusiveness Index)")
        
        # Use quality-weighted walkshed as the quality measure
        # (This already incorporates amenity score and inclusiveness via attractiveness)
        model3_vars_list = list(available_base.values()) + list(available_model1.values()) + list(available_model2.values())
        
        # Add quality-weighted walkshed if available
        if 'quality_weighted_walkshed_10min' in df_outcome.columns:
            model3_vars_list.append('quality_weighted_walkshed_10min')
            df_model3 = df_outcome.dropna(subset=model3_vars_list)
            
            if len(df_model3) >= 10:  # Need sufficient observations
                y3 = df_model3[outcome_col].values
                X3_df = df_model3[model3_vars_list].copy()
                
                results_model3, model3 = fit_model(
                    y3, X3_df, 'Model 3 (Quality)', outcome_label, 3
                )
                all_results.extend(results_model3)
                
                if model3 is not None:
                    print(f"      R-squared: {model3.rsquared:.4f}")
                    print(f"      AIC: {model3.aic:.2f}")
                    if 'quality_weighted_walkshed_10min' in model3.params.index:
                        coef = model3.params['quality_weighted_walkshed_10min']
                        pval = model3.pvalues['quality_weighted_walkshed_10min']
                        print(f"      quality_weighted_walkshed_10min: {coef:.4f} (p={pval:.4f})")
            else:
                print(f"      Skipping - insufficient observations ({len(df_model3)}) after adding quality variables")
        else:
            print(f"      Skipping - quality_weighted_walkshed_10min not available")
    else:
        print(f"      Skipping - Model 2 variables not available")
    
    # ========================================================================
    # MODEL 4: SPATIAL (Model 3 with Spatial Error/Lag)
    # ========================================================================
    if HAS_SPATIAL and has_geometry:
        print(f"\n   Model 4 (Spatial): Model 3 with Spatial Error/Lag terms")
        
        # Use Model 3 variables (or Model 2 if Model 3 failed)
        if len(available_model2) > 0 and 'quality_weighted_walkshed_10min' in df_outcome.columns:
            model4_vars_list = (list(available_base.values()) + 
                               list(available_model1.values()) + 
                               list(available_model2.values()) + 
                               ['quality_weighted_walkshed_10min'])
        elif len(available_model2) > 0:
            model4_vars_list = (list(available_base.values()) + 
                               list(available_model1.values()) + 
                               list(available_model2.values()))
        else:
            model4_vars_list = None
        
        if model4_vars_list is not None:
            df_model4 = df_outcome.dropna(subset=model4_vars_list + [outcome_col])
            df_model4 = df_model4[df_model4.geometry.is_valid].copy()
            
            if len(df_model4) >= 10:
                try:
                    # Create spatial weights matrix (Queen contiguity)
                    w = Queen.from_dataframe(df_model4, use_index=False)
                    
                    # Prepare data
                    y4 = df_model4[outcome_col].values
                    X4_vars = model4_vars_list
                    X4 = df_model4[X4_vars].values
                    
                    # Fit Spatial Error Model
                    print(f"      Fitting Spatial Error Model...")
                    try:
                        model_spatial_error = ML_Error(y4, X4, w, name_y=outcome_col, name_x=X4_vars)
                        
                        # Extract p-values from z_stat (list of tuples: (z-stat, p-value))
                        z_stats_error = model_spatial_error.z_stat
                        
                        results_spatial_error = []
                        # betas is an array, extract scalar values
                        betas_array = np.array(model_spatial_error.betas).flatten()
                        std_err_array = np.array(model_spatial_error.std_err).flatten()
                        
                        # betas includes constant at the end, z_stat matches betas order
                        for i, var in enumerate(X4_vars):
                            if i < len(betas_array) - 1:  # Exclude constant (last)
                                z_stat, pval = z_stats_error[i]
                                coef = float(betas_array[i])
                                se = float(std_err_array[i])
                                results_spatial_error.append({
                                    'outcome': outcome_label,
                                    'model': 'Model 4 (Spatial Error)',
                                    'model_num': 4,
                                    'variable': var,
                                    'coefficient': coef,
                                    'std_error': se,
                                    'pvalue': pval,
                                    'ci_lower': coef - 1.96 * se,
                                    'ci_upper': coef + 1.96 * se,
                                    'n_obs': len(y4),
                                    'r_squared': model_spatial_error.pr2,
                                    'aic': model_spatial_error.aic,
                                    'bic': model_spatial_error.schwarz
                                })
                        
                        # Add lambda (spatial error coefficient) - it's the last coefficient
                        lambda_idx = len(betas_array) - 1
                        if lambda_idx < len(z_stats_error):
                            z_stat_lambda, pval_lambda = z_stats_error[lambda_idx]
                            coef_lambda = float(betas_array[lambda_idx])
                            se_lambda = float(std_err_array[lambda_idx])
                            results_spatial_error.append({
                                'outcome': outcome_label,
                                'model': 'Model 4 (Spatial Error)',
                                'model_num': 4,
                                'variable': 'lambda (spatial error)',
                                'coefficient': coef_lambda,
                                'std_error': se_lambda,
                                'pvalue': pval_lambda,
                                'ci_lower': coef_lambda - 1.96 * se_lambda,
                                'ci_upper': coef_lambda + 1.96 * se_lambda,
                                'n_obs': len(y4),
                                'r_squared': model_spatial_error.pr2,
                                'aic': model_spatial_error.aic,
                                'bic': model_spatial_error.schwarz
                            })
                        
                        all_results.extend(results_spatial_error)
                        print(f"      Spatial Error Model R-squared: {model_spatial_error.pr2:.4f}")
                        # Lambda is the last coefficient in betas
                        if len(betas_array) > 0:
                            lambda_val = float(betas_array[-1])
                            print(f"      Lambda (spatial error): {lambda_val:.4f}")
                        
                        # Add model statistics
                        all_results.append({
                            'outcome': outcome_label,
                            'model': 'Model 4 (Spatial Error)',
                            'model_num': 4,
                            'variable': '_model_stats',
                            'coefficient': np.nan,
                            'std_error': np.nan,
                            'pvalue': np.nan,
                            'ci_lower': np.nan,
                            'ci_upper': np.nan,
                            'n_obs': len(y4),
                            'r_squared': model_spatial_error.pr2,
                            'aic': model_spatial_error.aic,
                            'bic': model_spatial_error.schwarz
                        })
                        
                    except Exception as e:
                        print(f"      Error fitting Spatial Error Model: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Fit Spatial Lag Model
                    print(f"      Fitting Spatial Lag Model...")
                    try:
                        model_spatial_lag = ML_Lag(y4, X4, w, name_y=outcome_col, name_x=X4_vars)
                        
                        # Extract p-values from z_stat
                        z_stats_lag = model_spatial_lag.z_stat
                        
                        results_spatial_lag = []
                        # betas is an array, extract scalar values
                        betas_lag_array = np.array(model_spatial_lag.betas).flatten()
                        std_err_lag_array = np.array(model_spatial_lag.std_err).flatten()
                        
                        for i, var in enumerate(X4_vars):
                            if i < len(betas_lag_array) - 1:  # Exclude constant and rho
                                z_stat, pval = z_stats_lag[i]
                                coef = float(betas_lag_array[i])
                                se = float(std_err_lag_array[i])
                                results_spatial_lag.append({
                                    'outcome': outcome_label,
                                    'model': 'Model 4 (Spatial Lag)',
                                    'model_num': 4,
                                    'variable': var,
                                    'coefficient': coef,
                                    'std_error': se,
                                    'pvalue': pval,
                                    'ci_lower': coef - 1.96 * se,
                                    'ci_upper': coef + 1.96 * se,
                                    'n_obs': len(y4),
                                    'r_squared': model_spatial_lag.pr2,
                                    'aic': model_spatial_lag.aic,
                                    'bic': model_spatial_lag.schwarz
                                })
                        
                        # Add rho (spatial lag coefficient) - it's the last coefficient
                        rho_idx = len(betas_lag_array) - 1
                        if rho_idx < len(z_stats_lag):
                            z_stat_rho, pval_rho = z_stats_lag[rho_idx]
                            coef_rho = float(betas_lag_array[rho_idx])
                            se_rho = float(std_err_lag_array[rho_idx])
                            results_spatial_lag.append({
                                'outcome': outcome_label,
                                'model': 'Model 4 (Spatial Lag)',
                                'model_num': 4,
                                'variable': 'rho (spatial lag)',
                                'coefficient': coef_rho,
                                'std_error': se_rho,
                                'pvalue': pval_rho,
                                'ci_lower': coef_rho - 1.96 * se_rho,
                                'ci_upper': coef_rho + 1.96 * se_rho,
                                'n_obs': len(y4),
                                'r_squared': model_spatial_lag.pr2,
                                'aic': model_spatial_lag.aic,
                                'bic': model_spatial_lag.schwarz
                            })
                        
                        all_results.extend(results_spatial_lag)
                        print(f"      Spatial Lag Model R-squared: {model_spatial_lag.pr2:.4f}")
                        if len(betas_lag_array) > 0:
                            rho_val = float(betas_lag_array[-1])
                            print(f"      Rho (spatial lag): {rho_val:.4f}")
                        
                        # Add model statistics
                        all_results.append({
                            'outcome': outcome_label,
                            'model': 'Model 4 (Spatial Lag)',
                            'model_num': 4,
                            'variable': '_model_stats',
                            'coefficient': np.nan,
                            'std_error': np.nan,
                            'pvalue': np.nan,
                            'ci_lower': np.nan,
                            'ci_upper': np.nan,
                            'n_obs': len(y4),
                            'r_squared': model_spatial_lag.pr2,
                            'aic': model_spatial_lag.aic,
                            'bic': model_spatial_lag.schwarz
                        })
                        
                    except Exception as e:
                        print(f"      Error fitting Spatial Lag Model: {e}")
                        import traceback
                        traceback.print_exc()
                
                except Exception as e:
                    print(f"      Error creating spatial weights matrix: {e}")
            else:
                print(f"      Skipping - insufficient observations ({len(df_model4)})")
        else:
            print(f"      Skipping - Model 3/2 variables not available")
    else:
        if not HAS_SPATIAL:
            print(f"\n   Model 4 (Spatial): Skipped - spatial regression libraries not available")
        elif not has_geometry:
            print(f"\n   Model 4 (Spatial): Skipped - no geometry data available")
        else:
            print(f"\n   Model 4 (Spatial): Skipped - insufficient model variables")
    
    # ========================================================================
    # PHASE 2: MODELS WITH POPULATION DENSITY CONTROL
    # ========================================================================
    if 'log_pop_density' in df_outcome.columns and df_outcome['log_pop_density'].notna().sum() > 0:
        print(f"\n   {'='*70}")
        print(f"   PHASE 2: Models with Population Density Control")
        print(f"   {'='*70}")
        
        # Phase 2 Model 1: Proximity + Density
        if len(available_model1) > 0:
            print(f"\n   Phase 2 Model 1: Proximity + Population Density")
            
            phase2_model1_vars = list(available_base.values()) + list(available_model1.values()) + ['log_pop_density']
            df_phase2_1 = df_outcome.dropna(subset=phase2_model1_vars)
            
            if len(df_phase2_1) > 0:
                y_p2_1 = df_phase2_1[outcome_col].values
                X_p2_1_df = df_phase2_1[phase2_model1_vars].copy()
                
                results_p2_1, model_p2_1 = fit_model(
                    y_p2_1, X_p2_1_df, 'Phase 2 Model 1 (Proximity + Density)', outcome_label, 5
                )
                all_results.extend(results_p2_1)
                
                if model_p2_1 is not None:
                    print(f"      R-squared: {model_p2_1.rsquared:.4f}")
                    print(f"      AIC: {model_p2_1.aic:.2f}")
                    if 'park_acres_10min' in model_p2_1.params.index:
                        coef = model_p2_1.params['park_acres_10min']
                        pval = model_p2_1.pvalues['park_acres_10min']
                        print(f"      park_acres_10min: {coef:.4f} (p={pval:.4f})")
                    if 'log_pop_density' in model_p2_1.params.index:
                        coef = model_p2_1.params['log_pop_density']
                        pval = model_p2_1.pvalues['log_pop_density']
                        print(f"      log_pop_density: {coef:.4f} (p={pval:.4f})")
            else:
                print(f"      Skipping - no valid observations")
        
        # Phase 2 Model 2: Greenness + Density
        if len(available_model2) > 0 and len(available_model1) > 0:
            print(f"\n   Phase 2 Model 2: Greenness + Population Density")
            
            phase2_model2_vars = (list(available_base.values()) + 
                                 list(available_model1.values()) + 
                                 list(available_model2.values()) + 
                                 ['log_pop_density'])
            df_phase2_2 = df_outcome.dropna(subset=phase2_model2_vars)
            
            if len(df_phase2_2) > 0:
                y_p2_2 = df_phase2_2[outcome_col].values
                X_p2_2_df = df_phase2_2[phase2_model2_vars].copy()
                
                results_p2_2, model_p2_2 = fit_model(
                    y_p2_2, X_p2_2_df, 'Phase 2 Model 2 (Greenness + Density)', outcome_label, 6
                )
                all_results.extend(results_p2_2)
                
                if model_p2_2 is not None:
                    print(f"      R-squared: {model_p2_2.rsquared:.4f}")
                    print(f"      AIC: {model_p2_2.aic:.2f}")
                    if 'ndvi_mean_500m' in model_p2_2.params.index:
                        coef = model_p2_2.params['ndvi_mean_500m']
                        pval = model_p2_2.pvalues['ndvi_mean_500m']
                        print(f"      ndvi_mean_500m: {coef:.4f} (p={pval:.4f})")
                    if 'log_pop_density' in model_p2_2.params.index:
                        coef = model_p2_2.params['log_pop_density']
                        pval = model_p2_2.pvalues['log_pop_density']
                        print(f"      log_pop_density: {coef:.4f} (p={pval:.4f})")
            else:
                print(f"      Skipping - no valid observations")
        
        # Phase 2 Model 3: Quality + Density
        if len(available_model2) > 0 and 'quality_weighted_walkshed_10min' in df_outcome.columns:
            print(f"\n   Phase 2 Model 3: Quality + Population Density")
            
            phase2_model3_vars = (list(available_base.values()) + 
                                 list(available_model1.values()) + 
                                 list(available_model2.values()) + 
                                 ['quality_weighted_walkshed_10min', 'log_pop_density'])
            df_phase2_3 = df_outcome.dropna(subset=phase2_model3_vars)
            
            if len(df_phase2_3) >= 10:
                y_p2_3 = df_phase2_3[outcome_col].values
                X_p2_3_df = df_phase2_3[phase2_model3_vars].copy()
                
                results_p2_3, model_p2_3 = fit_model(
                    y_p2_3, X_p2_3_df, 'Phase 2 Model 3 (Quality + Density)', outcome_label, 7
                )
                all_results.extend(results_p2_3)
                
                if model_p2_3 is not None:
                    print(f"      R-squared: {model_p2_3.rsquared:.4f}")
                    print(f"      AIC: {model_p2_3.aic:.2f}")
                    if 'quality_weighted_walkshed_10min' in model_p2_3.params.index:
                        coef = model_p2_3.params['quality_weighted_walkshed_10min']
                        pval = model_p2_3.pvalues['quality_weighted_walkshed_10min']
                        print(f"      quality_weighted_walkshed_10min: {coef:.4f} (p={pval:.4f})")
                    if 'log_pop_density' in model_p2_3.params.index:
                        coef = model_p2_3.params['log_pop_density']
                        pval = model_p2_3.pvalues['log_pop_density']
                        print(f"      log_pop_density: {coef:.4f} (p={pval:.4f})")
            else:
                print(f"      Skipping - insufficient observations ({len(df_phase2_3)})")
    else:
        print(f"\n   Phase 2 Models: Skipped - population density not available")
    
    # ========================================================================
    # INTERACTION MODELS: PARK ACCESS × POVERTY
    # ========================================================================
    if 'pct_families_below_poverty' in df_outcome.columns:
        print(f"\n   {'='*70}")
        print(f"   INTERACTION MODELS: Park Access × Poverty")
        print(f"   {'='*70}")
        
        # Create interaction terms
        if 'park_acres_10min' in df_outcome.columns:
            df_outcome['park_acres_10min_x_poverty'] = (
                df_outcome['park_acres_10min'] * df_outcome['pct_families_below_poverty']
            )
        
        if 'dist_to_park_miles' in df_outcome.columns:
            df_outcome['dist_to_park_x_poverty'] = (
                df_outcome['dist_to_park_miles'] * df_outcome['pct_families_below_poverty']
            )
        
        if 'ndvi_mean_500m' in df_outcome.columns:
            df_outcome['ndvi_500m_x_poverty'] = (
                df_outcome['ndvi_mean_500m'] * df_outcome['pct_families_below_poverty']
            )
        
        if 'quality_weighted_walkshed_10min' in df_outcome.columns:
            df_outcome['quality_walkshed_x_poverty'] = (
                df_outcome['quality_weighted_walkshed_10min'] * df_outcome['pct_families_below_poverty']
            )
        
        # Interaction Model 1: Proximity × Poverty
        if 'park_acres_10min' in df_outcome.columns and 'park_acres_10min_x_poverty' in df_outcome.columns:
            print(f"\n   Interaction Model 1: Proximity × Poverty")
            
            # Base + proximity + interaction
            interaction1_vars = (list(available_base.values()) + 
                              list(available_model1.values()) + 
                              ['park_acres_10min_x_poverty'])
            df_int1 = df_outcome.dropna(subset=interaction1_vars)
            
            if len(df_int1) > 0:
                y_int1 = df_int1[outcome_col].values
                X_int1_df = df_int1[interaction1_vars].copy()
                
                results_int1, model_int1 = fit_model(
                    y_int1, X_int1_df, 'Interaction Model 1 (Proximity × Poverty)', outcome_label, 8
                )
                all_results.extend(results_int1)
                
                if model_int1 is not None:
                    print(f"      R-squared: {model_int1.rsquared:.4f}")
                    print(f"      AIC: {model_int1.aic:.2f}")
                    if 'park_acres_10min_x_poverty' in model_int1.params.index:
                        coef = model_int1.params['park_acres_10min_x_poverty']
                        pval = model_int1.pvalues['park_acres_10min_x_poverty']
                        print(f"      park_acres_10min × poverty: {coef:.4f} (p={pval:.4f})")
            else:
                print(f"      Skipping - no valid observations")
        
        # Interaction Model 2: Greenness × Poverty
        if ('ndvi_mean_500m' in df_outcome.columns and 
            'ndvi_500m_x_poverty' in df_outcome.columns and 
            len(available_model1) > 0):
            print(f"\n   Interaction Model 2: Greenness × Poverty")
            
            interaction2_vars = (list(available_base.values()) + 
                              list(available_model1.values()) + 
                              list(available_model2.values()) + 
                              ['ndvi_500m_x_poverty'])
            df_int2 = df_outcome.dropna(subset=interaction2_vars)
            
            if len(df_int2) > 0:
                y_int2 = df_int2[outcome_col].values
                X_int2_df = df_int2[interaction2_vars].copy()
                
                results_int2, model_int2 = fit_model(
                    y_int2, X_int2_df, 'Interaction Model 2 (Greenness × Poverty)', outcome_label, 9
                )
                all_results.extend(results_int2)
                
                if model_int2 is not None:
                    print(f"      R-squared: {model_int2.rsquared:.4f}")
                    print(f"      AIC: {model_int2.aic:.2f}")
                    if 'ndvi_500m_x_poverty' in model_int2.params.index:
                        coef = model_int2.params['ndvi_500m_x_poverty']
                        pval = model_int2.pvalues['ndvi_500m_x_poverty']
                        print(f"      ndvi_500m × poverty: {coef:.4f} (p={pval:.4f})")
            else:
                print(f"      Skipping - no valid observations")
        
        # Interaction Model 3: Quality × Poverty
        if ('quality_weighted_walkshed_10min' in df_outcome.columns and 
            'quality_walkshed_x_poverty' in df_outcome.columns and 
            len(available_model2) > 0):
            print(f"\n   Interaction Model 3: Quality × Poverty")
            
            interaction3_vars = (list(available_base.values()) + 
                              list(available_model1.values()) + 
                              list(available_model2.values()) + 
                              ['quality_weighted_walkshed_10min', 'quality_walkshed_x_poverty'])
            df_int3 = df_outcome.dropna(subset=interaction3_vars)
            
            if len(df_int3) >= 10:
                y_int3 = df_int3[outcome_col].values
                X_int3_df = df_int3[interaction3_vars].copy()
                
                results_int3, model_int3 = fit_model(
                    y_int3, X_int3_df, 'Interaction Model 3 (Quality × Poverty)', outcome_label, 10
                )
                all_results.extend(results_int3)
                
                if model_int3 is not None:
                    print(f"      R-squared: {model_int3.rsquared:.4f}")
                    print(f"      AIC: {model_int3.aic:.2f}")
                    if 'quality_walkshed_x_poverty' in model_int3.params.index:
                        coef = model_int3.params['quality_walkshed_x_poverty']
                        pval = model_int3.pvalues['quality_walkshed_x_poverty']
                        print(f"      quality_walkshed × poverty: {coef:.4f} (p={pval:.4f})")
            else:
                print(f"      Skipping - insufficient observations ({len(df_int3)})")
    else:
        print(f"\n   Interaction Models: Skipped - poverty variable not available")

# ============================================================================
# 6. CREATE RESULTS TABLE
# ============================================================================
print(f"\n{'='*80}")
print("CREATING RESULTS TABLE")
print(f"{'='*80}")

if len(all_results) == 0:
    print("   Warning: No results to save!")
else:
    results_df = pd.DataFrame(all_results)
    
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
    
    # Separate model stats from variable results
    model_stats = results_df[results_df['variable'] == '_model_stats'].copy()
    variable_results = results_df[results_df['variable'] != '_model_stats'].copy()
    
    # Reorder columns for variable results
    column_order = [
        'outcome', 'model', 'model_num', 'variable',
        'coefficient', 'std_error', 'pvalue', 'significance',
        'ci_lower', 'ci_upper', 'n_obs', 'r_squared', 'aic', 'bic'
    ]
    variable_results = variable_results[column_order]
    
    # Sort
    variable_results = variable_results.sort_values(['outcome', 'model_num', 'variable'])
    model_stats = model_stats.sort_values(['outcome', 'model_num'])
    
    print(f"\n   Results summary:")
    print(f"   Total variable results: {len(variable_results)}")
    print(f"   Total model statistics: {len(model_stats)}")
    
    print(f"\n   Results by outcome:")
    for outcome in variable_results['outcome'].unique():
        n = len(variable_results[variable_results['outcome'] == outcome])
        print(f"     {outcome}: {n} variable results")
    
    print(f"\n   Results by model:")
    for model in variable_results['model'].unique():
        n = len(variable_results[variable_results['model'] == model])
        print(f"     {model}: {n} variable results")
    
    # ============================================================================
    # 7. SAVE RESULTS
    # ============================================================================
    print(f"\n7. Saving results...")
    print(f"   Output path: {output_path}")
    
    variable_results.to_csv(output_path, index=False)
    print(f"   [OK] Variable results saved successfully!")
    
    # Also save model statistics separately
    stats_path = output_path.replace('.csv', '_model_stats.csv')
    model_stats.to_csv(stats_path, index=False)
    print(f"   [OK] Model statistics saved to: {stats_path}")
    
    print(f"\n   Sample variable results:")
    print(variable_results.head(15).to_string(index=False))
    
    print(f"\n   Model statistics:")
    print(model_stats[['outcome', 'model', 'n_obs', 'r_squared', 'aic', 'bic']].to_string(index=False))

print(f"\n{'='*80}")
print("HIERARCHICAL REGRESSION ANALYSIS COMPLETE")
print(f"{'='*80}")
