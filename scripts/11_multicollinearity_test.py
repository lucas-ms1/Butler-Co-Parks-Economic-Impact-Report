"""
Multicollinearity Test: School Districts vs Park Distance
Economic Impact Report for Butler County Parks

This script tests for multicollinearity between school districts and park distance
variables. The hypothesis is that property taxes fund both parks and schools,
creating a correlation that may cause school district controls to absorb park effects.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import os

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input and output paths
input_path = os.path.join(project_root, 'data_final', 'housing_regression_ready.csv')
output_path = os.path.join(project_root, 'results', 'multicollinearity_test_results.txt')

print("="*80)
print("MULTICOLLINEARITY TEST: SCHOOL DISTRICTS vs PARK DISTANCE")
print("="*80)

# 1. Load Data
print(f"\n1. Loading data from: {input_path}")
df = pd.read_csv(input_path, low_memory=False)
print(f"   Rows loaded: {len(df):,}")

# Create variables
df['ln_price'] = np.log(df['PRICE'])
df['SALEDT'] = pd.to_datetime(df['SALEDT'])
df['SaleYear'] = df['SALEDT'].dt.year

# Check for school district column
school_col = None
for col in ['SCHOOLDIST', 'SCHOOL_DIST', 'SchoolDistrict']:
    if col in df.columns:
        school_col = col
        break

if not school_col:
    print("   ERROR: No school district column found!")
    exit(1)

print(f"   Using school district column: {school_col}")
print(f"   Unique school districts: {df[school_col].nunique()}")

# 2. Descriptive Statistics: Park Distance by School District
print(f"\n{'='*80}")
print("2. DESCRIPTIVE STATISTICS: PARK DISTANCE BY SCHOOL DISTRICT")
print(f"{'='*80}")

# Calculate mean distance by school district
school_dist_stats = df.groupby(school_col).agg({
    'dist_to_park_miles': ['mean', 'std', 'min', 'max', 'count'],
    'NearPark': 'mean'  # Proportion within 1 mile
}).round(4)

school_dist_stats.columns = ['Mean_Distance', 'Std_Distance', 'Min_Distance', 
                             'Max_Distance', 'Count', 'Prop_NearPark']

print(f"\n   Park Distance Statistics by School District:")
print(f"   {'School District':<40s} {'Mean Dist':>12s} {'Std Dev':>12s} {'% Near Park':>12s} {'Count':>8s}")
print(f"   {'-'*100}")
for district, row in school_dist_stats.iterrows():
    print(f"   {str(district):<40s} {row['Mean_Distance']:>12.4f} {row['Std_Distance']:>12.4f} "
          f"{row['Prop_NearPark']*100:>11.2f}% {int(row['Count']):>8d}")

# Test for significant differences in mean distance across districts
print(f"\n   Testing for significant differences in mean park distance across districts...")
# One-way ANOVA
districts = df[school_col].unique()
district_groups = [df[df[school_col] == d]['dist_to_park_miles'].values 
                   for d in districts if len(df[df[school_col] == d]) > 10]
f_stat, p_value = stats.f_oneway(*district_groups)
print(f"   One-way ANOVA: F-statistic = {f_stat:.4f}, p-value = {p_value:.6f}")
if p_value < 0.05:
    print(f"   *** Significant differences in park distance across school districts (p < 0.05)")
else:
    print(f"   No significant differences in park distance across school districts")

# 3. Correlation Analysis
print(f"\n{'='*80}")
print("3. CORRELATION ANALYSIS")
print(f"{'='*80}")

# Create school district dummy variables
school_dummies = pd.get_dummies(df[school_col], prefix='School', drop_first=False)
print(f"\n   Created {len(school_dummies.columns)} school district dummy variables")

# Calculate correlations between park distance and each school district dummy
print(f"\n   Correlation between park distance and school district dummies:")
print(f"   {'School District':<40s} {'Correlation':>12s} {'P-value':>12s}")
print(f"   {'-'*70}")

correlations = []
for col in school_dummies.columns:
    district_name = col.replace('School_', '')
    corr, p_val = stats.pearsonr(df['dist_to_park_miles'], school_dummies[col])
    correlations.append({
        'District': district_name,
        'Correlation': corr,
        'P-value': p_val
    })
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
    print(f"   {district_name:<40s} {corr:>12.4f} {p_val:>12.6f} {sig}")

corr_df = pd.DataFrame(correlations)
max_corr = corr_df['Correlation'].abs().max()
print(f"\n   Maximum absolute correlation: {max_corr:.4f}")

# 4. Variance Inflation Factor (VIF) Analysis
print(f"\n{'='*80}")
print("4. VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
print(f"{'='*80}")

# Prepare data for VIF calculation
# We need to create a design matrix with park distance and school district dummies
print(f"\n   Calculating VIF for models with and without school district controls...")

# Model 1: Without school districts (baseline)
# Use only continuous variables for VIF to avoid issues with dummies
X_no_school = df[['NearPark', 'dist_to_park_miles', 'SQFT', 'YEAR_BUILT', 'ACRES']].copy()
X_no_school = X_no_school.dropna()
X_no_school = X_no_school.astype(float)  # Ensure all numeric

# Calculate VIF for model without school districts
try:
    vif_no_school = pd.DataFrame()
    vif_no_school['Variable'] = X_no_school.columns
    vif_no_school['VIF'] = [variance_inflation_factor(X_no_school.values, i) 
                            for i in range(X_no_school.shape[1])]

    print(f"\n   VIF (Model WITHOUT School District Controls - Continuous Variables Only):")
    print(f"   {'Variable':<30s} {'VIF':>12s} {'Interpretation':>20s}")
    print(f"   {'-'*65}")
    for _, row in vif_no_school.iterrows():
        vif_val = row['VIF']
        if vif_val < 5:
            interp = "Low"
        elif vif_val < 10:
            interp = "Moderate"
        else:
            interp = "High (problematic)"
        print(f"   {row['Variable']:<30s} {vif_val:>12.2f} {interp:>20s}")
except Exception as e:
    print(f"   Warning: Could not calculate VIF for model without school districts: {e}")
    vif_no_school = pd.DataFrame({'Variable': X_no_school.columns, 'VIF': [np.nan] * len(X_no_school.columns)})

# Model 2: With school districts - test key variables
# Get indices where we have complete data
complete_idx = df[['NearPark', 'dist_to_park_miles', 'SQFT', 'YEAR_BUILT', 'ACRES', school_col]].dropna().index
X_with_school = df.loc[complete_idx, ['NearPark', 'dist_to_park_miles', 'SQFT', 'YEAR_BUILT', 'ACRES']].copy()
X_with_school = X_with_school.astype(float)  # Ensure all numeric

# Add school district dummies (drop first to avoid perfect multicollinearity)
school_dummies_subset = pd.get_dummies(df.loc[complete_idx, school_col], prefix='School', drop_first=True)

# Calculate VIF for key variables (park distance and first few school districts)
# Test with a subset to avoid computational issues
print(f"\n   Calculating VIF for model WITH school district controls...")
print(f"   (Testing key variables only)")

# Test VIF for park variables with individual school district dummies
key_vars = ['NearPark', 'dist_to_park_miles']
vif_results = []

# Test each school district dummy separately with park variables
for school_var in school_dummies_subset.columns[:5]:  # Test first 5 districts
    X_test = pd.concat([X_with_school[key_vars], school_dummies_subset[[school_var]]], axis=1)
    X_test = X_test.astype(float)
    try:
        vif_park = variance_inflation_factor(X_test.values, 0)  # VIF for NearPark
        vif_dist = variance_inflation_factor(X_test.values, 1)  # VIF for dist_to_park_miles
        vif_results.append({
            'School_District': school_var.replace('School_', ''),
            'VIF_NearPark': vif_park,
            'VIF_DistToPark': vif_dist
        })
    except Exception as e:
        vif_results.append({
            'School_District': school_var.replace('School_', ''),
            'VIF_NearPark': np.nan,
            'VIF_DistToPark': np.nan
        })

vif_with_school = pd.DataFrame(vif_results)

print(f"\n   VIF for Park Variables (with individual school district controls):")
print(f"   {'School District':<30s} {'VIF NearPark':>15s} {'VIF DistToPark':>15s}")
print(f"   {'-'*65}")
for _, row in vif_with_school.iterrows():
    print(f"   {row['School_District']:<30s} {row['VIF_NearPark']:>15.2f} {row['VIF_DistToPark']:>15.2f}")

# 5. Regression Coefficient Comparison
print(f"\n{'='*80}")
print("5. REGRESSION COEFFICIENT COMPARISON")
print(f"{'='*80}")

# Model without school districts
print(f"\n   Model 1: WITHOUT School District Controls")
formula_no_school = 'ln_price ~ NearPark + SQFT + YEAR_BUILT + ACRES + C(SaleYear)'
model_no_school = sm.OLS.from_formula(formula_no_school, data=df).fit(cov_type='HC1')
coef_no_school = model_no_school.params['NearPark']
se_no_school = model_no_school.bse['NearPark']
pval_no_school = model_no_school.pvalues['NearPark']
pct_no_school = (np.exp(coef_no_school) - 1) * 100

print(f"   NearPark coefficient: {coef_no_school:.6f}")
print(f"   Standard error: {se_no_school:.6f}")
print(f"   P-value: {pval_no_school:.6f}")
print(f"   Percentage premium: {pct_no_school:.2f}%")
print(f"   R-squared: {model_no_school.rsquared:.4f}")

# Model with school districts
print(f"\n   Model 2: WITH School District Controls")
formula_with_school = f'ln_price ~ NearPark + SQFT + YEAR_BUILT + ACRES + C(SaleYear) + C({school_col})'
model_with_school = sm.OLS.from_formula(formula_with_school, data=df).fit(cov_type='HC1')
coef_with_school = model_with_school.params['NearPark']
se_with_school = model_with_school.bse['NearPark']
pval_with_school = model_with_school.pvalues['NearPark']
pct_with_school = (np.exp(coef_with_school) - 1) * 100

print(f"   NearPark coefficient: {coef_with_school:.6f}")
print(f"   Standard error: {se_with_school:.6f}")
print(f"   P-value: {pval_with_school:.6f}")
print(f"   Percentage premium: {pct_with_school:.2f}%")
print(f"   R-squared: {model_with_school.rsquared:.4f}")

# Calculate change
coef_change = coef_with_school - coef_no_school
coef_change_pct = ((coef_with_school / coef_no_school) - 1) * 100 if coef_no_school != 0 else np.nan
se_change = se_with_school - se_no_school

print(f"\n   Change when adding school district controls:")
print(f"   Coefficient change: {coef_change:.6f} ({coef_change_pct:.2f}%)")
print(f"   Standard error change: {se_change:.6f}")
print(f"   Significance change: {'Significant' if pval_no_school < 0.05 else 'Not significant'} -> "
      f"{'Significant' if pval_with_school < 0.05 else 'Not significant'}")

# 6. Conditional Independence Test
print(f"\n{'='*80}")
print("6. CONDITIONAL INDEPENDENCE TEST")
print(f"{'='*80}")

# Test if park distance is independent of school district, conditional on other variables
# We can use a regression of park distance on school districts (controlling for other factors)
print(f"\n   Testing if park distance is independent of school district...")
print(f"   (Regressing park distance on school districts + controls)")

# Create a model where distance is the outcome
formula_dist = f'dist_to_park_miles ~ C({school_col}) + SQFT + YEAR_BUILT + ACRES + C(SaleYear)'
model_dist = sm.OLS.from_formula(formula_dist, data=df).fit(cov_type='HC1')

# F-test for joint significance of school district dummies
# Get all school district parameter names
school_params = [p for p in model_dist.params.index if school_col in p or 'C(' in p and school_col in p]
# Create a constraint matrix - test that all school district coefficients are zero
# We'll use the model's built-in F-test for the model with vs without school districts
# Compare model with school districts vs model without
formula_dist_no_school = 'dist_to_park_miles ~ SQFT + YEAR_BUILT + ACRES + C(SaleYear)'
model_dist_no_school = sm.OLS.from_formula(formula_dist_no_school, data=df).fit(cov_type='HC1')

# Calculate F-statistic manually
n = len(df)
k_full = len(model_dist.params)
k_reduced = len(model_dist_no_school.params)
rss_full = model_dist.ssr
rss_reduced = model_dist_no_school.ssr

f_stat = ((rss_reduced - rss_full) / (k_full - k_reduced)) / (rss_full / (n - k_full))
f_pval = 1 - stats.f.cdf(f_stat, k_full - k_reduced, n - k_full)

print(f"   F-test for joint significance of school districts:")
print(f"   F-statistic: {f_stat:.4f}")
print(f"   P-value: {f_pval:.6f}")
if f_pval < 0.05:
    print(f"   *** School districts are significantly related to park distance (p < 0.05)")
    print(f"   This suggests multicollinearity: school districts predict park distance")
else:
    print(f"   School districts are not significantly related to park distance")

# 7. Summary and Interpretation
print(f"\n{'='*80}")
print("7. SUMMARY AND INTERPRETATION")
print(f"{'='*80}")

print(f"\n   Key Findings:")
print(f"   1. Park distance varies significantly across school districts: "
      f"{'YES' if p_value < 0.05 else 'NO'} (ANOVA p = {p_value:.6f})")
print(f"   2. Maximum correlation between distance and district dummies: {max_corr:.4f}")
if not vif_no_school.empty and 'NearPark' in vif_no_school['Variable'].values:
    nearpark_vif = vif_no_school[vif_no_school['Variable'] == 'NearPark']['VIF'].values[0]
    if not np.isnan(nearpark_vif):
        print(f"   3. VIF for park variables (without school controls): {nearpark_vif:.2f}")
    else:
        print(f"   3. VIF calculation had issues (likely due to data structure)")
else:
    print(f"   3. VIF calculation had issues (likely due to data structure)")
print(f"   4. Park coefficient changes by {abs(coef_change_pct):.2f}% when adding school controls")
print(f"   5. School districts predict park distance: {'YES' if f_pval < 0.05 else 'NO'} "
      f"(F-test p = {f_pval:.6f})")

print(f"\n   Interpretation:")
if p_value < 0.05 and f_pval < 0.05:
    print(f"   *** STRONG EVIDENCE OF MULTICOLLINEARITY ***")
    print(f"   School districts and park distance are significantly correlated.")
    print(f"   This supports the hypothesis that property taxes fund both parks and schools,")
    print(f"   creating a correlation that causes school district controls to absorb park effects.")
    print(f"   Recommendation: The model WITHOUT school district controls captures the")
    print(f"   'bundled community value' of parks, which is appropriate for economic impact analysis.")
elif p_value < 0.05 or f_pval < 0.05:
    print(f"   MODERATE EVIDENCE OF MULTICOLLINEARITY")
    print(f"   Some correlation exists between school districts and park distance.")
    print(f"   This may partially explain why park effects diminish when controlling for districts.")
else:
    print(f"   WEAK EVIDENCE OF MULTICOLLINEARITY")
    print(f"   Limited correlation between school districts and park distance.")
    print(f"   The coefficient change may be due to other factors.")

# Save results to file
print(f"\n{'='*80}")
print("8. SAVING RESULTS")
print(f"{'='*80}")

with open(output_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("MULTICOLLINEARITY TEST: SCHOOL DISTRICTS vs PARK DISTANCE\n")
    f.write("="*80 + "\n\n")
    
    f.write("SUMMARY STATISTICS BY SCHOOL DISTRICT\n")
    f.write("-"*80 + "\n")
    f.write(school_dist_stats.to_string())
    f.write("\n\n")
    
    f.write("CORRELATION ANALYSIS\n")
    f.write("-"*80 + "\n")
    f.write(corr_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("VIF ANALYSIS (Model WITHOUT School Districts)\n")
    f.write("-"*80 + "\n")
    f.write(vif_no_school.to_string(index=False))
    f.write("\n\n")
    
    f.write("REGRESSION COEFFICIENT COMPARISON\n")
    f.write("-"*80 + "\n")
    f.write(f"Model WITHOUT school districts:\n")
    f.write(f"  NearPark coefficient: {coef_no_school:.6f} ({pct_no_school:.2f}%)\n")
    f.write(f"  P-value: {pval_no_school:.6f}\n")
    f.write(f"  R-squared: {model_no_school.rsquared:.4f}\n\n")
    f.write(f"Model WITH school districts:\n")
    f.write(f"  NearPark coefficient: {coef_with_school:.6f} ({pct_with_school:.2f}%)\n")
    f.write(f"  P-value: {pval_with_school:.6f}\n")
    f.write(f"  R-squared: {model_with_school.rsquared:.4f}\n\n")
    f.write(f"Change: {coef_change:.6f} ({coef_change_pct:.2f}%)\n\n")
    
    f.write("CONDITIONAL INDEPENDENCE TEST\n")
    f.write("-"*80 + "\n")
    f.write(f"F-test for school districts predicting park distance:\n")
    f.write(f"  F-statistic: {f_stat:.4f}\n")
    f.write(f"  P-value: {f_pval:.6f}\n")

print(f"\n   Results saved to: {output_path}")

print(f"\n{'='*80}")
print("MULTICOLLINEARITY TEST COMPLETE")
print(f"{'='*80}")

