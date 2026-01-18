"""
Distance Band Benefits Summary Statistics
Economic Impact Report for Butler County Parks

This script calculates comprehensive summary statistics for property value
benefits per distance band, including:
- Property counts and prices by band
- Regression coefficients (premiums) per band
- Total value uplift per band
- Average benefit per property per band
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input path
input_path = os.path.join(project_root, 'data_final', 'housing_regression_ready.csv')

# Output paths
output_dir = os.path.join(project_root, 'results')
output_path = os.path.join(output_dir, 'distance_band_benefits_summary.csv')
summary_path = os.path.join(output_dir, 'distance_band_benefits_summary.txt')

print("="*80)
print("DISTANCE BAND BENEFITS SUMMARY STATISTICS")
print("="*80)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print(f"\n1. Loading data...")
print(f"   Path: {input_path}")

df = pd.read_csv(input_path, low_memory=False)
print(f"   Records loaded: {len(df):,}")

# Create ln_price
df['ln_price'] = np.log(df['PRICE'])

# Create SaleYear
df['SALEDT'] = pd.to_datetime(df['SALEDT'])
df['SaleYear'] = df['SALEDT'].dt.year

# ============================================================================
# 2. DESCRIPTIVE STATISTICS BY DISTANCE BAND
# ============================================================================
print(f"\n2. Calculating descriptive statistics by distance band...")

# Get distance bands
dist_bands = sorted(df['DistBand'].unique())
print(f"   Distance bands found: {dist_bands}")

# Calculate statistics per band
band_stats = []

for band in dist_bands:
    band_df = df[df['DistBand'] == band].copy()
    
    stats = {
        'distance_band': band,
        'property_count': len(band_df),
        'pct_of_total': (len(band_df) / len(df)) * 100,
        'mean_price': band_df['PRICE'].mean(),
        'median_price': band_df['PRICE'].median(),
        'std_price': band_df['PRICE'].std(),
        'min_price': band_df['PRICE'].min(),
        'max_price': band_df['PRICE'].max(),
        'q25_price': band_df['PRICE'].quantile(0.25),
        'q75_price': band_df['PRICE'].quantile(0.75),
        'mean_distance': band_df['dist_to_park_miles'].mean(),
        'max_distance': band_df['dist_to_park_miles'].max(),
        'total_value': band_df['PRICE'].sum()
    }
    
    band_stats.append(stats)

band_stats_df = pd.DataFrame(band_stats)

print(f"\n   Descriptive Statistics by Distance Band:")
print(f"   {'='*80}")
print(f"   {'Band':<15} {'Count':>10} {'% Total':>10} {'Mean Price':>12} {'Median Price':>13} {'Total Value':>15}")
print(f"   {'-'*80}")
for idx, row in band_stats_df.iterrows():
    print(f"   {row['distance_band']:<15} {row['property_count']:>10,} {row['pct_of_total']:>9.2f}% "
          f"${row['mean_price']:>11,.0f} ${row['median_price']:>12,.0f} ${row['total_value']:>14,.0f}")

# ============================================================================
# 3. REGRESSION ANALYSIS: DISTANCE BAND COEFFICIENTS
# ============================================================================
print(f"\n3. Running regression to estimate premiums by distance band...")

# Find reference band (furthest)
furthest_band = dist_bands[-1]
print(f"   Reference category: {furthest_band}")

# Run TWO regressions:
# 1. Without school district (bundled community value - includes park + school effects)
# 2. With school district (isolated park effect)

print(f"\n   Model 1: Bundled Community Value (No School District Controls)")
formula1 = f'ln_price ~ C(DistBand, Treatment(reference="{furthest_band}")) + SQFT + YEAR_BUILT + ACRES + C(SaleYear)'
model1 = sm.OLS.from_formula(formula1, data=df).fit(cov_type='HC1')
print(f"   R-squared: {model1.rsquared:.4f}, N: {len(df):,}")

print(f"\n   Model 2: Isolated Park Effect (With School District Controls)")
formula2 = f'ln_price ~ C(DistBand, Treatment(reference="{furthest_band}")) + SQFT + YEAR_BUILT + ACRES + C(SaleYear) + C(SCHOOLDIST)'
model2 = sm.OLS.from_formula(formula2, data=df).fit(cov_type='HC1')
print(f"   R-squared: {model2.rsquared:.4f}, N: {len(df):,}")

# Use Model 1 (bundled value) for benefits calculation
model = model1
print(f"\n   Using Model 1 (Bundled Community Value) for benefits calculation")

# Extract distance band coefficients
dist_band_params = model.params[model.params.index.str.contains('DistBand')]

# Create results dataframe
regression_results = []

for param_name in dist_band_params.index:
    # Extract band name
    band_name = param_name.replace(f'C(DistBand, Treatment(reference="{furthest_band}"))[T.', '').replace(']', '')
    
    coef = dist_band_params[param_name]
    se = model.bse[param_name]
    pvalue = model.pvalues[param_name]
    ci_lower = model.conf_int().loc[param_name, 0]
    ci_upper = model.conf_int().loc[param_name, 1]
    
    # Calculate percentage premium
    pct_premium = (np.exp(coef) - 1) * 100
    pct_premium_lower = (np.exp(ci_lower) - 1) * 100
    pct_premium_upper = (np.exp(ci_upper) - 1) * 100
    
    regression_results.append({
        'distance_band': band_name,
        'coefficient': coef,
        'std_error': se,
        'pvalue': pvalue,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'pct_premium': pct_premium,
        'pct_premium_lower': pct_premium_lower,
        'pct_premium_upper': pct_premium_upper,
        'significant': pvalue < 0.05
    })

regression_df = pd.DataFrame(regression_results)

print(f"\n   Regression Results (relative to {furthest_band}):")
print(f"   {'='*80}")
print(f"   {'Band':<15} {'Coefficient':>12} {'Std Error':>12} {'% Premium':>12} {'P-value':>10} {'Sig':>5}")
print(f"   {'-'*80}")
for idx, row in regression_df.iterrows():
    sig = "***" if row['pvalue'] < 0.01 else "**" if row['pvalue'] < 0.05 else "*" if row['pvalue'] < 0.1 else ""
    print(f"   {row['distance_band']:<15} {row['coefficient']:>12.6f} {row['std_error']:>12.6f} "
          f"{row['pct_premium']:>11.2f}% {row['pvalue']:>10.6f} {sig:>5}")

# ============================================================================
# 4. CALCULATE ECONOMIC BENEFITS PER BAND
# ============================================================================
print(f"\n4. Calculating economic benefits per distance band...")

# Merge statistics with regression results
benefits_df = band_stats_df.merge(regression_df, on='distance_band', how='left')

# For reference band, set premium to 0
benefits_df.loc[benefits_df['distance_band'] == furthest_band, 'pct_premium'] = 0.0
benefits_df.loc[benefits_df['distance_band'] == furthest_band, 'coefficient'] = 0.0

# Calculate value uplift per property
# Uplift = mean_price * (exp(coef) - 1) = mean_price * pct_premium / 100
benefits_df['avg_benefit_per_property'] = (
    benefits_df['mean_price'] * benefits_df['pct_premium'] / 100
)

# Calculate total value uplift per band
benefits_df['total_value_uplift'] = (
    benefits_df['property_count'] * benefits_df['avg_benefit_per_property']
)

# Calculate cumulative benefits (sum of all bands closer than this one)
benefits_df = benefits_df.sort_values('mean_distance')
benefits_df['cumulative_value_uplift'] = benefits_df['total_value_uplift'].cumsum()

# Calculate benefit per mile (for bands with distance range)
def calculate_benefit_per_mile(row):
    """Calculate benefit per mile for distance band"""
    band = row['distance_band']
    if band == furthest_band:
        return 0.0
    
    # Parse band to get distance range
    if '-' in band:
        parts = band.split('-')
        if parts[0] == '0':
            max_dist = float(parts[1])
            range_miles = max_dist
        else:
            min_dist = float(parts[0])
            max_dist = float(parts[1])
            range_miles = max_dist - min_dist
    elif band.startswith('>'):
        return 0.0  # Reference band
    else:
        return np.nan
    
    if range_miles > 0:
        return row['avg_benefit_per_property'] / range_miles
    else:
        return np.nan

benefits_df['benefit_per_mile'] = benefits_df.apply(calculate_benefit_per_mile, axis=1)

print(f"\n   Economic Benefits by Distance Band:")
print(f"   {'='*80}")
print(f"   {'Band':<15} {'Properties':>12} {'Mean Price':>12} {'Premium %':>12} {'Avg Benefit':>14} {'Total Uplift':>15}")
print(f"   {'-'*80}")
for idx, row in benefits_df.iterrows():
    if pd.notna(row['pct_premium']):
        print(f"   {row['distance_band']:<15} {row['property_count']:>12,} ${row['mean_price']:>11,.0f} "
              f"{row['pct_premium']:>11.2f}% ${row['avg_benefit_per_property']:>13,.0f} ${row['total_value_uplift']:>14,.0f}")
    else:
        print(f"   {row['distance_band']:<15} {row['property_count']:>12,} ${row['mean_price']:>11,.0f} "
              f"{'Reference':>12} ${0:>13,.0f} ${0:>14,.0f}")

# ============================================================================
# 5. SUMMARY STATISTICS
# ============================================================================
print(f"\n5. Summary Statistics:")
print(f"   {'='*80}")

total_properties = benefits_df['property_count'].sum()
total_value = benefits_df['total_value'].sum()
total_uplift = benefits_df['total_value_uplift'].sum()

print(f"\n   Overall Statistics:")
print(f"     Total Properties: {total_properties:,}")
print(f"     Total Property Value: ${total_value:,.0f}")
print(f"     Total Value Uplift (Attributable to Parks): ${total_uplift:,.0f}")
print(f"     Average Benefit per Property: ${total_uplift/total_properties:,.0f}")
print(f"     Uplift as % of Total Value: {(total_uplift/total_value)*100:.2f}%")

# Properties within 1 mile
within_1mile = benefits_df[benefits_df['mean_distance'] <= 1.0]
if len(within_1mile) > 0:
    within_1mile_count = within_1mile['property_count'].sum()
    within_1mile_uplift = within_1mile['total_value_uplift'].sum()
    print(f"\n   Properties Within 1 Mile:")
    print(f"     Count: {within_1mile_count:,} ({(within_1mile_count/total_properties)*100:.2f}%)")
    print(f"     Value Uplift: ${within_1mile_uplift:,.0f}")
    print(f"     Average Benefit per Property: ${within_1mile_uplift/within_1mile_count:,.0f}")

# Closest band (0-0.1 miles)
closest_band = benefits_df[benefits_df['distance_band'] == '0-0.1']
if len(closest_band) > 0:
    row = closest_band.iloc[0]
    print(f"\n   Closest Band (0-0.1 miles):")
    print(f"     Properties: {row['property_count']:,}")
    print(f"     Premium: {row['pct_premium']:.2f}%")
    print(f"     Average Benefit: ${row['avg_benefit_per_property']:,.0f} per property")
    print(f"     Total Uplift: ${row['total_value_uplift']:,.0f}")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print(f"\n6. Saving results...")

# Save detailed results
benefits_df.to_csv(output_path, index=False)
print(f"   Detailed results saved to: {output_path}")

# Save summary text file
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("DISTANCE BAND BENEFITS SUMMARY STATISTICS\n")
    f.write("="*80 + "\n\n")
    
    f.write("DESCRIPTIVE STATISTICS BY DISTANCE BAND\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Band':<15} {'Count':>10} {'% Total':>10} {'Mean Price':>12} {'Median Price':>13} {'Total Value':>15}\n")
    f.write("-"*80 + "\n")
    for idx, row in band_stats_df.iterrows():
        f.write(f"{row['distance_band']:<15} {row['property_count']:>10,} {row['pct_of_total']:>9.2f}% "
                f"${row['mean_price']:>11,.0f} ${row['median_price']:>12,.0f} ${row['total_value']:>14,.0f}\n")
    
    f.write("\n\nREGRESSION RESULTS (Relative to " + furthest_band + ")\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Band':<15} {'Coefficient':>12} {'Std Error':>12} {'% Premium':>12} {'P-value':>10} {'Sig':>5}\n")
    f.write("-"*80 + "\n")
    for idx, row in regression_df.iterrows():
        sig = "***" if row['pvalue'] < 0.01 else "**" if row['pvalue'] < 0.05 else "*" if row['pvalue'] < 0.1 else ""
        f.write(f"{row['distance_band']:<15} {row['coefficient']:>12.6f} {row['std_error']:>12.6f} "
                f"{row['pct_premium']:>11.2f}% {row['pvalue']:>10.6f} {sig:>5}\n")
    
    f.write("\n\nECONOMIC BENEFITS BY DISTANCE BAND\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Band':<15} {'Properties':>12} {'Mean Price':>12} {'Premium %':>12} {'Avg Benefit':>14} {'Total Uplift':>15}\n")
    f.write("-"*80 + "\n")
    for idx, row in benefits_df.iterrows():
        if pd.notna(row['pct_premium']):
            f.write(f"{row['distance_band']:<15} {row['property_count']:>12,} ${row['mean_price']:>11,.0f} "
                   f"{row['pct_premium']:>11.2f}% ${row['avg_benefit_per_property']:>13,.0f} ${row['total_value_uplift']:>14,.0f}\n")
        else:
            f.write(f"{row['distance_band']:<15} {row['property_count']:>12,} ${row['mean_price']:>11,.0f} "
                   f"{'Reference':>12} ${0:>13,.0f} ${0:>14,.0f}\n")
    
    f.write("\n\nSUMMARY STATISTICS\n")
    f.write("-"*80 + "\n")
    f.write(f"Total Properties: {total_properties:,}\n")
    f.write(f"Total Property Value: ${total_value:,.0f}\n")
    f.write(f"Total Value Uplift (Attributable to Parks): ${total_uplift:,.0f}\n")
    f.write(f"Average Benefit per Property: ${total_uplift/total_properties:,.0f}\n")
    f.write(f"Uplift as % of Total Value: {(total_uplift/total_value)*100:.2f}%\n")
    
    if len(within_1mile) > 0:
        f.write(f"\nProperties Within 1 Mile:\n")
        f.write(f"  Count: {within_1mile_count:,} ({(within_1mile_count/total_properties)*100:.2f}%)\n")
        f.write(f"  Value Uplift: ${within_1mile_uplift:,.0f}\n")
        f.write(f"  Average Benefit per Property: ${within_1mile_uplift/within_1mile_count:,.0f}\n")

print(f"   Summary report saved to: {summary_path}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
