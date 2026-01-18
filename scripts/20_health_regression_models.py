"""
Health Outcomes Regression Models
Economic Impact Report for Butler County Parks

This script estimates the association between park proximity and tract-level
health outcomes (physical inactivity, obesity, diabetes, frequent mental distress)
using OLS regression with robust standard errors.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input path
input_path = os.path.join(project_root, 'data_final', 'butler_tract_health_model_data.csv')

# Output paths
output_dir = os.path.join(project_root, 'results')
output_path = os.path.join(output_dir, 'health_regressions.csv')
output_wls_path = os.path.join(output_dir, 'health_regressions_wls.csv')

print("="*60)
print("HEALTH OUTCOMES REGRESSION MODELS")
print("="*60)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
print(f"\n1. Loading data from: {input_path}")
df = pd.read_csv(input_path, low_memory=False)
print(f"   Tracts loaded: {len(df):,}")
print(f"   Columns: {len(df.columns)}")

# 2. Select health outcome variables
print(f"\n2. Selecting health outcome variables...")

# Define outcomes with their preferred and fallback column names
outcomes_config = {
    'physical_inactivity': {
        'preferred': 'LPA_AgeAdjPrev',
        'fallback': 'LPA_CrudePrev',
        'label': 'Physical Inactivity'
    },
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

# Select the actual outcome columns
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
        print(f"   Warning: {config['label']} - neither {config['preferred']} nor {config['fallback']} found!")

if len(outcome_vars) == 0:
    raise ValueError("No health outcome variables found!")

# 3. Prepare predictor and covariate variables
print(f"\n3. Preparing predictor and covariate variables...")

# Predictors
predictors = {
    'within_1_mile': 'within_1_mile',
    'dist_to_park_miles': 'dist_to_park_miles'
}

# Covariates
covariates = {
    'median_household_income': 'median_household_income',
    'pct_families_below_poverty': 'pct_families_below_poverty',
    'unemployment_rate': 'unemployment_rate',
    'pct_bachelors_degree_or_higher': 'pct_bachelors_degree_or_higher'
}

# Check which variables are available
available_predictors = {}
for key, col in predictors.items():
    if col in df.columns:
        available_predictors[key] = col
        print(f"   Predictor available: {col}")
    else:
        print(f"   Warning: Predictor {col} not found!")

available_covariates = {}
for key, col in covariates.items():
    if col in df.columns:
        available_covariates[key] = col
        print(f"   Covariate available: {col}")
    else:
        print(f"   Warning: Covariate {col} not found!")

if len(available_predictors) == 0:
    raise ValueError("No predictor variables found!")

# Check for population variable for WLS
has_population = 'total_population' in df.columns
if has_population:
    print(f"   Population variable available: total_population (for WLS weighting)")
else:
    print(f"   Warning: total_population not found - WLS models will be skipped")

# 4. Prepare data for regression
print(f"\n4. Preparing data for regression...")

# Create a working copy
df_reg = df.copy()

# Check for missing values in key variables
all_vars = list(available_predictors.values()) + list(available_covariates.values()) + list([v['column'] for v in outcome_vars.values()])
if has_population:
    all_vars.append('total_population')
missing_check = df_reg[all_vars].isna().sum()
if missing_check.sum() > 0:
    print(f"   Missing values:")
    for var, count in missing_check[missing_check > 0].items():
        print(f"     {var}: {count} ({count/len(df_reg)*100:.1f}%)")

# 5. Run regressions
print(f"\n5. Running regressions...")

results_list = []
results_wls_list = []

# Run models for each outcome
for outcome_key, outcome_info in outcome_vars.items():
    outcome_col = outcome_info['column']
    outcome_label = outcome_info['label']
    
    print(f"\n   {'='*50}")
    print(f"   Outcome: {outcome_label} ({outcome_col})")
    print(f"   {'='*50}")
    
    # Create subset with non-missing outcome
    df_outcome = df_reg[df_reg[outcome_col].notna()].copy()
    
    # For WLS, also need non-missing population
    if has_population:
        df_outcome_wls = df_outcome[df_outcome['total_population'].notna() & (df_outcome['total_population'] > 0)].copy()
    else:
        df_outcome_wls = pd.DataFrame()
    
    print(f"   Observations: {len(df_outcome):,}")
    if has_population:
        print(f"   Observations with population weights: {len(df_outcome_wls):,}")
    
    if len(df_outcome) == 0:
        print(f"   Skipping - no valid observations")
        continue
    
    # Model 1: within_1_mile only
    if 'within_1_mile' in available_predictors:
        print(f"\n   Model 1: within_1_mile + covariates")
        
        # Build formula
        formula_vars = ['within_1_mile'] + list(available_covariates.values())
        formula = f"{outcome_col} ~ {' + '.join(formula_vars)}"
        
        try:
            model = sm.OLS.from_formula(formula, data=df_outcome).fit(cov_type='HC1')
            
            # Extract results for within_1_mile
            if 'within_1_mile' in model.params.index:
                coef = model.params['within_1_mile']
                se = model.bse['within_1_mile']
                pvalue = model.pvalues['within_1_mile']
                ci_lower = model.conf_int().loc['within_1_mile', 0]
                ci_upper = model.conf_int().loc['within_1_mile', 1]
                
                results_list.append({
                    'outcome': outcome_label,
                    'outcome_variable': outcome_col,
                    'predictor': 'within_1_mile',
                    'model': 'within_1_mile + covariates',
                    'coefficient': coef,
                    'std_error': se,
                    'pvalue': pvalue,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'n_obs': len(df_outcome),
                    'r_squared': model.rsquared
                })
                
                print(f"      Coefficient: {coef:.4f}")
                print(f"      SE: {se:.4f}")
                print(f"      P-value: {pvalue:.4f}")
                print(f"      R-squared: {model.rsquared:.4f}")
            
            # WLS version with population weights
            if has_population and len(df_outcome_wls) > 0:
                try:
                    weights = df_outcome_wls['total_population']
                    model_wls = sm.WLS.from_formula(formula, data=df_outcome_wls, weights=weights).fit(cov_type='HC1')
                    
                    if 'within_1_mile' in model_wls.params.index:
                        coef_wls = model_wls.params['within_1_mile']
                        se_wls = model_wls.bse['within_1_mile']
                        pvalue_wls = model_wls.pvalues['within_1_mile']
                        ci_lower_wls = model_wls.conf_int().loc['within_1_mile', 0]
                        ci_upper_wls = model_wls.conf_int().loc['within_1_mile', 1]
                        
                        results_wls_list.append({
                            'outcome': outcome_label,
                            'outcome_variable': outcome_col,
                            'predictor': 'within_1_mile',
                            'model': 'within_1_mile + covariates',
                            'coefficient': coef_wls,
                            'std_error': se_wls,
                            'pvalue': pvalue_wls,
                            'ci_lower': ci_lower_wls,
                            'ci_upper': ci_upper_wls,
                            'n_obs': len(df_outcome_wls),
                            'r_squared': model_wls.rsquared
                        })
                        
                        print(f"      WLS Coefficient: {coef_wls:.4f} (SE: {se_wls:.4f}, p: {pvalue_wls:.4f})")
                
                except Exception as e:
                    print(f"      Error fitting WLS model: {e}")
        
        except Exception as e:
            print(f"      Error fitting model: {e}")
    
    # Model 2: dist_to_park_miles only
    if 'dist_to_park_miles' in available_predictors:
        print(f"\n   Model 2: dist_to_park_miles + covariates")
        
        # Build formula
        formula_vars = ['dist_to_park_miles'] + list(available_covariates.values())
        formula = f"{outcome_col} ~ {' + '.join(formula_vars)}"
        
        try:
            model = sm.OLS.from_formula(formula, data=df_outcome).fit(cov_type='HC1')
            
            # Extract results for dist_to_park_miles
            if 'dist_to_park_miles' in model.params.index:
                coef = model.params['dist_to_park_miles']
                se = model.bse['dist_to_park_miles']
                pvalue = model.pvalues['dist_to_park_miles']
                ci_lower = model.conf_int().loc['dist_to_park_miles', 0]
                ci_upper = model.conf_int().loc['dist_to_park_miles', 1]
                
                results_list.append({
                    'outcome': outcome_label,
                    'outcome_variable': outcome_col,
                    'predictor': 'dist_to_park_miles',
                    'model': 'dist_to_park_miles + covariates',
                    'coefficient': coef,
                    'std_error': se,
                    'pvalue': pvalue,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'n_obs': len(df_outcome),
                    'r_squared': model.rsquared
                })
                
                print(f"      Coefficient: {coef:.4f}")
                print(f"      SE: {se:.4f}")
                print(f"      P-value: {pvalue:.4f}")
                print(f"      R-squared: {model.rsquared:.4f}")
            
            # WLS version with population weights
            if has_population and len(df_outcome_wls) > 0:
                try:
                    weights = df_outcome_wls['total_population']
                    model_wls = sm.WLS.from_formula(formula, data=df_outcome_wls, weights=weights).fit(cov_type='HC1')
                    
                    if 'dist_to_park_miles' in model_wls.params.index:
                        coef_wls = model_wls.params['dist_to_park_miles']
                        se_wls = model_wls.bse['dist_to_park_miles']
                        pvalue_wls = model_wls.pvalues['dist_to_park_miles']
                        ci_lower_wls = model_wls.conf_int().loc['dist_to_park_miles', 0]
                        ci_upper_wls = model_wls.conf_int().loc['dist_to_park_miles', 1]
                        
                        results_wls_list.append({
                            'outcome': outcome_label,
                            'outcome_variable': outcome_col,
                            'predictor': 'dist_to_park_miles',
                            'model': 'dist_to_park_miles + covariates',
                            'coefficient': coef_wls,
                            'std_error': se_wls,
                            'pvalue': pvalue_wls,
                            'ci_lower': ci_lower_wls,
                            'ci_upper': ci_upper_wls,
                            'n_obs': len(df_outcome_wls),
                            'r_squared': model_wls.rsquared
                        })
                        
                        print(f"      WLS Coefficient: {coef_wls:.4f} (SE: {se_wls:.4f}, p: {pvalue_wls:.4f})")
                
                except Exception as e:
                    print(f"      Error fitting WLS model: {e}")
        
        except Exception as e:
            print(f"      Error fitting model: {e}")
    
    # Model 3: Both predictors
    if 'within_1_mile' in available_predictors and 'dist_to_park_miles' in available_predictors:
        print(f"\n   Model 3: within_1_mile + dist_to_park_miles + covariates")
        
        # Build formula
        formula_vars = ['within_1_mile', 'dist_to_park_miles'] + list(available_covariates.values())
        formula = f"{outcome_col} ~ {' + '.join(formula_vars)}"
        
        try:
            model = sm.OLS.from_formula(formula, data=df_outcome).fit(cov_type='HC1')
            
            # Extract results for both predictors
            for pred in ['within_1_mile', 'dist_to_park_miles']:
                if pred in model.params.index:
                    coef = model.params[pred]
                    se = model.bse[pred]
                    pvalue = model.pvalues[pred]
                    ci_lower = model.conf_int().loc[pred, 0]
                    ci_upper = model.conf_int().loc[pred, 1]
                    
                    results_list.append({
                        'outcome': outcome_label,
                        'outcome_variable': outcome_col,
                        'predictor': pred,
                        'model': 'both + covariates',
                        'coefficient': coef,
                        'std_error': se,
                        'pvalue': pvalue,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'n_obs': len(df_outcome),
                        'r_squared': model.rsquared
                    })
                    
                    print(f"      {pred} coefficient: {coef:.4f} (SE: {se:.4f}, p: {pvalue:.4f})")
            
            # WLS version with population weights
            if has_population and len(df_outcome_wls) > 0:
                try:
                    weights = df_outcome_wls['total_population']
                    model_wls = sm.WLS.from_formula(formula, data=df_outcome_wls, weights=weights).fit(cov_type='HC1')
                    
                    for pred in ['within_1_mile', 'dist_to_park_miles']:
                        if pred in model_wls.params.index:
                            coef_wls = model_wls.params[pred]
                            se_wls = model_wls.bse[pred]
                            pvalue_wls = model_wls.pvalues[pred]
                            ci_lower_wls = model_wls.conf_int().loc[pred, 0]
                            ci_upper_wls = model_wls.conf_int().loc[pred, 1]
                            
                            results_wls_list.append({
                                'outcome': outcome_label,
                                'outcome_variable': outcome_col,
                                'predictor': pred,
                                'model': 'both + covariates',
                                'coefficient': coef_wls,
                                'std_error': se_wls,
                                'pvalue': pvalue_wls,
                                'ci_lower': ci_lower_wls,
                                'ci_upper': ci_upper_wls,
                                'n_obs': len(df_outcome_wls),
                                'r_squared': model_wls.rsquared
                            })
                            
                            print(f"      WLS {pred} coefficient: {coef_wls:.4f} (SE: {se_wls:.4f}, p: {pvalue_wls:.4f})")
                
                except Exception as e:
                    print(f"      Error fitting WLS model: {e}")
        
        except Exception as e:
            print(f"      Error fitting model: {e}")

# 6. Create tidy results table
print(f"\n{'='*60}")
print("CREATING TIDY RESULTS TABLE")
print(f"{'='*60}")

if len(results_list) == 0:
    print("   Warning: No results to save!")
else:
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
        'outcome', 'outcome_variable', 'predictor', 'model',
        'coefficient', 'std_error', 'pvalue', 'significance',
        'ci_lower', 'ci_upper', 'n_obs', 'r_squared'
    ]
    results_df = results_df[column_order]
    
    # Sort by outcome, then by predictor
    results_df = results_df.sort_values(['outcome', 'predictor', 'model'])
    
    print(f"\n   Results summary:")
    print(f"   Total model results: {len(results_df)}")
    print(f"\n   Results by outcome:")
    for outcome in results_df['outcome'].unique():
        n = len(results_df[results_df['outcome'] == outcome])
        print(f"     {outcome}: {n} results")
    
    print(f"\n   Sample results:")
    print(results_df.head(10).to_string(index=False))
    
    # 7. Save results
    print(f"\n7. Saving results...")
    print(f"   Output path: {output_path}")
    
    results_df.to_csv(output_path, index=False)
    print(f"   [OK] Saved successfully!")
    
    print(f"\n   Results saved with columns:")
    print(f"     {', '.join(results_df.columns)}")

# 8. Create and save WLS results table
if has_population and len(results_wls_list) > 0:
    print(f"\n{'='*60}")
    print("CREATING WLS RESULTS TABLE")
    print(f"{'='*60}")
    
    results_wls_df = pd.DataFrame(results_wls_list)
    
    # Add significance stars
    results_wls_df['significance'] = results_wls_df['pvalue'].apply(add_significance)
    
    # Reorder columns
    results_wls_df = results_wls_df[column_order]
    
    # Sort by outcome, then by predictor
    results_wls_df = results_wls_df.sort_values(['outcome', 'predictor', 'model'])
    
    print(f"\n   WLS Results summary:")
    print(f"   Total WLS model results: {len(results_wls_df)}")
    print(f"\n   WLS Results by outcome:")
    for outcome in results_wls_df['outcome'].unique():
        n = len(results_wls_df[results_wls_df['outcome'] == outcome])
        print(f"     {outcome}: {n} results")
    
    print(f"\n   Sample WLS results:")
    print(results_wls_df.head(10).to_string(index=False))
    
    # Save WLS results
    print(f"\n8. Saving WLS results...")
    print(f"   Output path: {output_wls_path}")
    
    results_wls_df.to_csv(output_wls_path, index=False)
    print(f"   [OK] WLS results saved successfully!")
else:
    print(f"\n8. WLS models:")
    if not has_population:
        print(f"   Skipped - total_population variable not available")
    elif len(results_wls_list) == 0:
        print(f"   Skipped - no WLS results to save")

print(f"\n{'='*60}")
print("HEALTH REGRESSION ANALYSIS COMPLETE")
print(f"{'='*60}")
