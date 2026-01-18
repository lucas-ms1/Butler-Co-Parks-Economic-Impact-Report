"""
Phase 2: Quality-Weighted Park Analysis & Economic Valuation
Economic Impact Report for Butler County Parks

This script:
1. Creates Chen Quality Score from park amenities data
2. Runs density-corrected regressions to isolate protective effects
3. Calculates economic valuation of health benefits

Based on Chen et al. (2023) quality-weighted approach.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input paths
input_path = os.path.join(project_root, 'data_final', 'butler_tract_health_model_data_with_greenness.csv')
input_gpkg_path = os.path.join(project_root, 'data_final', 'butler_tract_health_model_data_with_greenness.gpkg')
amenities_path = os.path.join(project_root, 'data_raw', 'park_amenities.csv')

# Output paths
output_dir = os.path.join(project_root, 'results')
output_path = os.path.join(output_dir, 'phase2_economic_valuation.csv')
summary_path = os.path.join(output_dir, 'phase2_economic_summary.txt')

print("="*80)
print("PHASE 2: QUALITY-WEIGHTED PARK ANALYSIS & ECONOMIC VALUATION")
print("="*80)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# PART 1: CREATE CHEN QUALITY SCORE FROM PARK AMENITIES
# ============================================================================
print(f"\n{'='*80}")
print("PART 1: Creating Chen Quality Score from Park Amenities")
print(f"{'='*80}")

if os.path.exists(amenities_path):
    print(f"\n1. Loading park amenities data...")
    amenities_df = pd.read_csv(amenities_path)
    print(f"   Parks loaded: {len(amenities_df)}")
    
    # Calculate Chen Quality Score
    # Formula: Quality = Amenity_Count * Inclusiveness_Score
    amenities_df['Chen_Quality_Score'] = (
        amenities_df['Amenity Count'] * amenities_df['Inclusiveness Score']
    )
    
    # Create a binary indicator for Active parks (if needed)
    amenities_df['Is_Active'] = (amenities_df['Active vs. Passive'] == 'Active').astype(int)
    amenities_df['Is_Mixed'] = (amenities_df['Active vs. Passive'] == 'Mixed').astype(int)
    
    print(f"\n   Park Quality Scores:")
    quality_summary = amenities_df[['Park Name', 'Amenity Count', 
                                    'Inclusiveness Score', 'Chen_Quality_Score',
                                    'Active vs. Passive']].sort_values(
        by='Chen_Quality_Score', ascending=False)
    print(quality_summary.to_string(index=False))
    
    print(f"\n   Quality Score Statistics:")
    print(f"     Mean: {amenities_df['Chen_Quality_Score'].mean():.2f}")
    print(f"     Min: {amenities_df['Chen_Quality_Score'].min():.2f}")
    print(f"     Max: {amenities_df['Chen_Quality_Score'].max():.2f}")
    print(f"     Std: {amenities_df['Chen_Quality_Score'].std():.2f}")
    
else:
    print(f"\n   Warning: Park amenities file not found at {amenities_path}")
    print(f"   Will use existing quality_weighted_walkshed_10min as proxy")
    amenities_df = None

# ============================================================================
# PART 2: LOAD TRACT DATA AND CALCULATE POPULATION DENSITY
# ============================================================================
print(f"\n{'='*80}")
print("PART 2: Loading Tract Data and Calculating Population Density")
print(f"{'='*80}")

# Try GPKG first, then CSV
if os.path.exists(input_gpkg_path):
    import geopandas as gpd
    print(f"\n2. Loading from GPKG: {input_gpkg_path}")
    df = gpd.read_file(input_gpkg_path)
    has_geometry = True
    print(f"   Records loaded: {len(df):,}")
    
    # Calculate tract area from geometry
    df['tract_area_sqft'] = df.geometry.area
    SQFT_TO_SQMI = 1.0 / 27878400.0
    df['tract_area_sqmi'] = df['tract_area_sqft'] * SQFT_TO_SQMI
    
elif os.path.exists(input_path):
    print(f"\n2. Loading from CSV: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    has_geometry = False
    print(f"   Records loaded: {len(df):,}")
    
    # Calculate tract area from ALAND (square meters)
    if 'ALAND' in df.columns:
        SQM_TO_SQMI = 1.0 / 2589988.11
        df['tract_area_sqmi'] = df['ALAND'] * SQM_TO_SQMI
    else:
        print(f"   Warning: Cannot calculate tract area - no ALAND column")
        df['tract_area_sqmi'] = np.nan
else:
    raise FileNotFoundError(f"Data file not found: {input_path}")

# Calculate population density
if 'total_population' in df.columns and 'tract_area_sqmi' in df.columns:
    df['pop_density_sq_mile'] = df['total_population'] / df['tract_area_sqmi']
    epsilon = 0.001
    df['log_pop_density'] = np.log(df['pop_density_sq_mile'] + epsilon)
    print(f"\n   Population density calculated")
    print(f"     Mean density: {df['pop_density_sq_mile'].mean():.2f} people/sq mi")
    print(f"     Mean log density: {df['log_pop_density'].mean():.4f}")
else:
    print(f"\n   Warning: Cannot calculate population density")
    df['log_pop_density'] = np.nan

# ============================================================================
# PART 3: DENSITY-CORRECTED REGRESSIONS
# ============================================================================
print(f"\n{'='*80}")
print("PART 3: Density-Corrected Regressions (The Statistical Fix)")
print(f"{'='*80}")

# Health Outcomes to Analyze
outcomes_config = {
    'MHLTH_CrudePrev': {
        'label': 'Frequent Mental Distress',
        'preferred': 'MHLTH_AgeAdjPrev'
    },
    'OBESITY_CrudePrev': {
        'label': 'Obesity',
        'preferred': 'OBESITY_AgeAdjPrev'
    },
    'LPA_CrudePrev': {
        'label': 'Physical Inactivity',
        'preferred': 'LPA_AgeAdjPrev'
    }
}

# Select actual outcome columns
outcome_vars = {}
for col, config in outcomes_config.items():
    if config['preferred'] in df.columns:
        outcome_vars[col] = {
            'column': config['preferred'],
            'label': config['label']
        }
    elif col in df.columns:
        outcome_vars[col] = {
            'column': col,
            'label': config['label']
        }

if len(outcome_vars) == 0:
    raise ValueError("No health outcome variables found!")

# Park access variables to test
park_vars = {
    'quality_weighted_walkshed_10min': 'Quality-Weighted Walkshed',
    'park_acres_10min': 'Park Acres (10min)',
    'ndvi_mean_500m': 'NDVI (Greenness)',
    'dist_to_park_miles': 'Distance to Park'
}

# Base controls
base_controls = [
    'median_household_income',
    'pct_families_below_poverty',
    'unemployment_rate',
    'pct_bachelors_degree_or_higher'
]

# Check which variables are available
available_park_vars = {}
for var, label in park_vars.items():
    if var in df.columns:
        available_park_vars[var] = label

available_controls = [c for c in base_controls if c in df.columns]

print(f"\n3. Available variables:")
print(f"   Park access variables: {list(available_park_vars.keys())}")
print(f"   Control variables: {available_controls}")
print(f"   Population density: {'log_pop_density' in df.columns}")

# Store regression results
regression_results = []

print(f"\n{'='*80}")
print("DENSITY-CORRECTED REGRESSION RESULTS")
print(f"{'='*80}")

for outcome_col, outcome_info in outcome_vars.items():
    outcome_label = outcome_info['label']
    print(f"\n{'-'*80}")
    print(f"Outcome: {outcome_label} ({outcome_col})")
    print(f"{'-'*80}")
    
    # Test each park variable
    for park_var, park_label in available_park_vars.items():
        # Prepare model variables
        model_vars = available_controls + [park_var]
        if 'log_pop_density' in df.columns:
            model_vars.append('log_pop_density')
        
        # Create subset with non-missing data
        df_model = df.dropna(subset=[outcome_col] + model_vars)
        
        if len(df_model) < 10:
            print(f"\n   {park_label}: Skipping - insufficient data ({len(df_model)} obs)")
            continue
        
        # Prepare data
        y = df_model[outcome_col].values
        X = df_model[model_vars].copy()
        X = sm.add_constant(X)
        
        # Fit model with robust standard errors
        try:
            model = sm.OLS(y, X, missing='drop').fit(cov_type='HC1')
            
            # Extract coefficient for park variable
            if park_var in model.params.index:
                coeff = model.params[park_var]
                se = model.bse[park_var]
                p_val = model.pvalues[park_var]
                ci_lower = model.conf_int().loc[park_var, 0]
                ci_upper = model.conf_int().loc[park_var, 1]
                
                # Determine if protective
                is_protective = (coeff < 0 and p_val < 0.1)
                
                print(f"\n   {park_label}:")
                print(f"     Coefficient: {coeff:.6f}")
                print(f"     Std Error: {se:.6f}")
                print(f"     P-value: {p_val:.4f}")
                print(f"     95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
                print(f"     R-squared: {model.rsquared:.4f}")
                print(f"     N: {len(df_model)}")
                
                if is_protective:
                    print(f"     -> *** SIGNIFICANT PROTECTIVE EFFECT FOUND ***")
                elif coeff < 0:
                    print(f"     -> Protective but not significant (p={p_val:.4f})")
                else:
                    print(f"     -> Warning: Positive coefficient (not protective)")
                
                # Store results
                regression_results.append({
                    'outcome': outcome_label,
                    'outcome_col': outcome_col,
                    'park_variable': park_label,
                    'park_var_col': park_var,
                    'coefficient': coeff,
                    'std_error': se,
                    'pvalue': p_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'r_squared': model.rsquared,
                    'n_obs': len(df_model),
                    'is_protective': is_protective,
                    'has_density_control': 'log_pop_density' in model_vars
                })
            else:
                print(f"\n   {park_label}: Variable not in model")
                
        except Exception as e:
            print(f"\n   {park_label}: Error fitting model - {e}")

# ============================================================================
# PART 3b: LOAD INTERACTION MODEL RESULTS (Significant Protective Effects)
# ============================================================================
print(f"\n{'='*80}")
print("PART 3b: Loading Interaction Model Results")
print(f"{'='*80}")

# Load hierarchical regression results to get interaction model coefficients
hierarchical_results_path = os.path.join(project_root, 'results', 'hierarchical_health_regressions.csv')

interaction_results = []

if os.path.exists(hierarchical_results_path):
    print(f"\n3b. Loading interaction model results from hierarchical regressions...")
    hier_df = pd.read_csv(hierarchical_results_path)
    
    # Filter to interaction models with significant protective effects
    interaction_vars = [
        'park_acres_10min_x_poverty',
        'quality_walkshed_x_poverty',
        'ndvi_500m_x_poverty'
    ]
    
    for var in interaction_vars:
        var_results = hier_df[
            (hier_df['variable'] == var) & 
            (hier_df['coefficient'] < 0) &  # Protective (negative)
            (hier_df['pvalue'] < 0.1)  # Significant
        ].copy()
        
        if len(var_results) > 0:
            for idx, row in var_results.iterrows():
                interaction_results.append({
                    'outcome': row['outcome'],
                    'outcome_col': None,  # Will map later
                    'park_variable': f"Interaction: {var}",
                    'park_var_col': var,
                    'coefficient': row['coefficient'],
                    'std_error': row['std_error'],
                    'pvalue': row['pvalue'],
                    'ci_lower': row['ci_lower'],
                    'ci_upper': row['ci_upper'],
                    'r_squared': row['r_squared'],
                    'n_obs': row['n_obs'],
                    'is_protective': True,
                    'has_density_control': True,  # Interaction models include base controls
                    'model_type': 'interaction'
                })
    
    print(f"   Found {len(interaction_results)} significant interaction effects")
    
    # Also check for Physical Inactivity quality-weighted walkshed (density-corrected)
    phys_inact_quality = hier_df[
        (hier_df['outcome'] == 'Physical Inactivity') &
        (hier_df['variable'] == 'quality_weighted_walkshed_10min') &
        (hier_df['model'].str.contains('Phase 2', na=False)) &
        (hier_df['coefficient'] < 0) &
        (hier_df['pvalue'] < 0.1)
    ]
    
    if len(phys_inact_quality) > 0:
        for idx, row in phys_inact_quality.iterrows():
            interaction_results.append({
                'outcome': row['outcome'],
                'outcome_col': None,
                'park_variable': 'Quality-Weighted Walkshed (Density-Corrected)',
                'park_var_col': 'quality_weighted_walkshed_10min',
                'coefficient': row['coefficient'],
                'std_error': row['std_error'],
                'pvalue': row['pvalue'],
                'ci_lower': row['ci_lower'],
                'ci_upper': row['ci_upper'],
                'r_squared': row['r_squared'],
                'n_obs': row['n_obs'],
                'is_protective': True,
                'has_density_control': True,
                'model_type': 'density_corrected'
            })
        print(f"   Added Physical Inactivity quality effect (density-corrected)")
    
    # Store interaction results for later merging
    if len(interaction_results) > 0:
        interaction_df = pd.DataFrame(interaction_results)
        print(f"   Interaction effects loaded successfully")
else:
    print(f"   Warning: Hierarchical regression results not found")
    interaction_df = pd.DataFrame()

# ============================================================================
# PART 4: ECONOMIC VALUATION CALCULATOR
# ============================================================================
print(f"\n{'='*80}")
print("PART 4: Economic Valuation Calculator")
print(f"{'='*80}")

# Constants (2024 Estimates based on CDC/ACSM data)
# Sources: CDC, ACSM, Finkelstein et al. (2009), Cawley & Meyerhoefer (2012)
COST_OBESITY = 2500       # Annual excess medical cost per obese person (USD)
COST_INACTIVITY = 1900    # Annual cost per inactive person (USD)
COST_MENTAL_DISTRESS = 1000  # Conservative annual medical cost avoidance (USD)

# Impact scenario: Improvement in park quality/access
# This represents a modest improvement (e.g., +0.1 units in quality-weighted walkshed)
IMPACT_SCENARIOS = {
    'conservative': 0.05,   # 5% improvement
    'moderate': 0.10,        # 10% improvement
    'aggressive': 0.20       # 20% improvement
}

print(f"\n4. Economic Impact Scenarios:")
print(f"   Conservative: {IMPACT_SCENARIOS['conservative']*100:.0f}% improvement in park access")
print(f"   Moderate: {IMPACT_SCENARIOS['moderate']*100:.0f}% improvement in park access")
print(f"   Aggressive: {IMPACT_SCENARIOS['aggressive']*100:.0f}% improvement in park access")

# Calculate total population
total_pop = df['total_population'].sum() if 'total_population' in df.columns else 0
print(f"\n   Total Population: {total_pop:,.0f}")

# Create results DataFrame
results_df = pd.DataFrame(regression_results)

# Merge with interaction results if available
if 'interaction_df' in locals() and len(interaction_df) > 0:
    results_df = pd.concat([results_df, interaction_df], ignore_index=True)
    print(f"\n   Merged interaction model results")
    print(f"   Total protective effects (including interactions): {len(results_df[results_df['is_protective']])}")

# Filter to protective effects only
protective_results = results_df[results_df['is_protective']].copy()

print(f"\n{'='*80}")
print("ECONOMIC VALUATION RESULTS")
print(f"{'='*80}")

valuation_results = []

if len(protective_results) > 0:
    print(f"\n   Found {len(protective_results)} protective effects to value:")
    
    for scenario_name, impact_multiplier in IMPACT_SCENARIOS.items():
        print(f"\n   {'-'*80}")
        print(f"   Scenario: {scenario_name.upper()} ({impact_multiplier*100:.0f}% improvement)")
        print(f"   {'-'*80}")
        
        scenario_total_savings = 0
        
        for idx, row in protective_results.iterrows():
            outcome = row['outcome']
            park_var = row['park_variable']
            beta = row['coefficient']
            
            # Determine cost multiplier
            if 'Obesity' in outcome:
                unit_cost = COST_OBESITY
            elif 'Inactivity' in outcome or 'Physical' in outcome:
                unit_cost = COST_INACTIVITY
            elif 'Mental' in outcome or 'Distress' in outcome:
                unit_cost = COST_MENTAL_DISTRESS
            else:
                unit_cost = 1000  # Default conservative estimate
            
            # Calculate prevalence reduction
            # For interaction terms, we need to interpret differently
            # Interaction coefficient represents effect per unit of park access × poverty
            if 'x_poverty' in row['park_var_col'].lower() or 'interaction' in row['park_variable'].lower():
                # For interaction terms: effect is conditional on poverty level
                # We'll use average poverty rate to estimate average effect
                avg_poverty = df['pct_families_below_poverty'].mean() if 'pct_families_below_poverty' in df.columns else 10.0
                
                if 'quality_walkshed' in row['park_var_col'].lower():
                    # Quality walkshed × poverty: increase quality by 10%, at average poverty
                    base_quality = df['quality_weighted_walkshed_10min'].mean() if 'quality_weighted_walkshed_10min' in df.columns else 1.0
                    var_increase = base_quality * impact_multiplier * (avg_poverty / 100)
                elif 'park_acres' in row['park_var_col'].lower():
                    # Park acres × poverty: increase acres by 10%, at average poverty
                    avg_acres = df['park_acres_10min'].mean() if 'park_acres_10min' in df.columns else 1.0
                    var_increase = avg_acres * impact_multiplier * (avg_poverty / 100)
                else:
                    var_increase = impact_multiplier * (avg_poverty / 100)
            elif 'quality_weighted' in row['park_var_col'].lower():
                # Quality-weighted walkshed: 0.1 unit increase
                var_increase = impact_multiplier
            elif 'park_acres' in row['park_var_col'].lower():
                # Park acres: 10% increase in average acres
                avg_acres = df[row['park_var_col']].mean() if row['park_var_col'] in df.columns else 1.0
                var_increase = avg_acres * impact_multiplier
            elif 'ndvi' in row['park_var_col'].lower():
                # NDVI: 0.1 unit increase (reasonable for greenness improvement)
                var_increase = impact_multiplier
            elif 'dist' in row['park_var_col'].lower():
                # Distance: negative means closer is better, so we use negative of improvement
                var_increase = -impact_multiplier  # Negative because distance decrease = improvement
            else:
                var_increase = impact_multiplier
            
            # Prevalence reduction (percentage points)
            prev_reduction = abs(beta * var_increase)
            
            # Avoided cases = (Reduction % / 100) * Total Population
            avoided_cases = (prev_reduction / 100) * total_pop
            
            # Total savings
            total_savings = avoided_cases * unit_cost
            scenario_total_savings += total_savings
            
            print(f"\n   {outcome} ({park_var}):")
            print(f"     Coefficient: {beta:.6f}")
            print(f"     Variable increase: {var_increase:.4f}")
            print(f"     Prevalence reduction: {prev_reduction:.4f} percentage points")
            print(f"     Projected avoided cases: {int(avoided_cases):,}")
            print(f"     Unit cost: ${unit_cost:,.0f}")
            print(f"     Estimated annual savings: ${total_savings:,.2f}")
            
            valuation_results.append({
                'scenario': scenario_name,
                'outcome': outcome,
                'park_variable': park_var,
                'coefficient': beta,
                'variable_increase': var_increase,
                'prevalence_reduction_pp': prev_reduction,
                'avoided_cases': int(avoided_cases),
                'unit_cost': unit_cost,
                'annual_savings': total_savings
            })
        
        print(f"\n   {'='*60}")
        print(f"   SCENARIO TOTAL ANNUAL SAVINGS: ${scenario_total_savings:,.2f}")
        print(f"   {'='*60}")
        
else:
    print(f"\n   Warning: No protective effects found in density-corrected models.")
    print(f"   Cannot calculate economic valuation.")
    print(f"\n   Recommendation:")
    print(f"   - Check interaction models (Park Access × Poverty)")
    print(f"   - These may reveal benefits hidden in low-income subgroups")

# ============================================================================
# PART 5: SAVE RESULTS
# ============================================================================
print(f"\n{'='*80}")
print("PART 5: Saving Results")
print(f"{'='*80}")

# Save regression results
if len(results_df) > 0:
    results_df.to_csv(output_path, index=False)
    print(f"\n5. Regression results saved to: {output_path}")

# Save valuation results
if len(valuation_results) > 0:
    valuation_df = pd.DataFrame(valuation_results)
    valuation_path = output_path.replace('.csv', '_valuation.csv')
    valuation_df.to_csv(valuation_path, index=False)
    print(f"   Valuation results saved to: {valuation_path}")

# Create summary text file
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("PHASE 2: QUALITY-WEIGHTED PARK ANALYSIS & ECONOMIC VALUATION\n")
    f.write("="*80 + "\n\n")
    
    f.write("PART 1: CHEN QUALITY SCORE\n")
    f.write("-"*80 + "\n")
    if amenities_df is not None:
        f.write(f"Parks analyzed: {len(amenities_df)}\n")
        f.write(f"Mean Quality Score: {amenities_df['Chen_Quality_Score'].mean():.2f}\n")
        f.write(f"Range: {amenities_df['Chen_Quality_Score'].min():.2f} - {amenities_df['Chen_Quality_Score'].max():.2f}\n\n")
    
    f.write("PART 2: DENSITY-CORRECTED REGRESSIONS\n")
    f.write("-"*80 + "\n")
    f.write(f"Total models tested: {len(results_df)}\n")
    f.write(f"Protective effects found: {len(protective_results)}\n\n")
    
    if len(protective_results) > 0:
        f.write("Protective Effects Summary:\n")
        for idx, row in protective_results.iterrows():
            f.write(f"  - {row['outcome']} ({row['park_variable']}): ")
            f.write(f"coeff={row['coefficient']:.6f}, p={row['pvalue']:.4f}\n")
        f.write("\n")
    
    f.write("PART 3: ECONOMIC VALUATION\n")
    f.write("-"*80 + "\n")
    if len(valuation_results) > 0:
        valuation_summary = pd.DataFrame(valuation_results)
        for scenario in IMPACT_SCENARIOS.keys():
            scenario_data = valuation_summary[valuation_summary['scenario'] == scenario]
            if len(scenario_data) > 0:
                total = scenario_data['annual_savings'].sum()
                f.write(f"{scenario.upper()} Scenario: ${total:,.2f} annual savings\n")
    else:
        f.write("No protective effects available for valuation.\n")
        f.write("Recommendation: Use interaction models (Park Access × Poverty)\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print(f"   Summary report saved to: {summary_path}")

print(f"\n{'='*80}")
print("PHASE 2 ANALYSIS COMPLETE")
print(f"{'='*80}")
