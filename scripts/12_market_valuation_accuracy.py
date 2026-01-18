"""
Market Valuation Accuracy Test
Economic Impact Report for Butler County Parks

This script validates the accuracy of assessed market values (MKTVCurYr) 
by comparing them to actual sale prices (PRICE) for properties that sold.
This helps assess the reliability of using market values for full-stock analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input and output paths
input_path = os.path.join(project_root, 'data_final', 'housing_regression_ready.csv')
output_path = os.path.join(project_root, 'results', 'market_valuation_accuracy_test.txt')
output_csv_path = os.path.join(project_root, 'results', 'market_valuation_accuracy.csv')

print("="*80)
print("MARKET VALUATION ACCURACY TEST")
print("="*80)

# 1. Load Data
print(f"\n1. Loading data from: {input_path}")
df = pd.read_csv(input_path, low_memory=False)
print(f"   Rows loaded: {len(df):,}")

# Check for required columns
required_cols = ['PRICE', 'MKTVCurYr']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"   ERROR: Missing required columns: {missing_cols}")
    exit(1)

# Filter to properties with both sale price and market value
df_valid = df.dropna(subset=['PRICE', 'MKTVCurYr']).copy()
print(f"   Properties with both PRICE and MKTVCurYr: {len(df_valid):,}")

# Filter out invalid values
df_valid = df_valid[(df_valid['PRICE'] > 0) & (df_valid['MKTVCurYr'] > 0)].copy()
print(f"   After filtering invalid values: {len(df_valid):,}")

# Create variables
df_valid['SALEDT'] = pd.to_datetime(df_valid['SALEDT'])
df_valid['SaleYear'] = df_valid['SALEDT'].dt.year

# CPI Data for inflation adjustment (Consumer Price Index for All Urban Consumers)
# Source: U.S. Bureau of Labor Statistics, annual averages
# Assume MKTVCurYr is from 2023 (current assessment year)
assessment_year = 2023
cpi_data = {
    2014: 236.736,
    2015: 237.017,
    2016: 240.007,
    2017: 245.120,
    2018: 251.107,
    2019: 255.657,
    2020: 258.811,
    2021: 270.970,
    2022: 292.655,
    2023: 304.700
}

# Calculate inflation adjustment factors (to convert sale prices to 2023 dollars)
cpi_base = cpi_data[assessment_year]
df_valid['CPI_adjustment'] = df_valid['SaleYear'].map(
    lambda y: cpi_base / cpi_data.get(y, cpi_base)
)

# Create inflation-adjusted sale prices (in 2023 dollars)
df_valid['PRICE_adjusted'] = df_valid['PRICE'] * df_valid['CPI_adjustment']

print(f"\n   Inflation Adjustment:")
print(f"     Assessment Year: {assessment_year}")
print(f"     Base CPI ({assessment_year}): {cpi_base:.3f}")
print(f"     Adjusting sale prices from {df_valid['SaleYear'].min()}-{df_valid['SaleYear'].max()} to {assessment_year} dollars")

# Calculate error metrics (NOMINAL - without inflation adjustment)
df_valid['Error'] = df_valid['MKTVCurYr'] - df_valid['PRICE']
df_valid['AbsoluteError'] = np.abs(df_valid['Error'])
df_valid['PercentError'] = (df_valid['Error'] / df_valid['PRICE']) * 100
df_valid['AbsolutePercentError'] = np.abs(df_valid['PercentError'])
df_valid['Ratio'] = df_valid['MKTVCurYr'] / df_valid['PRICE']

# Calculate error metrics (INFLATION-ADJUSTED - with inflation adjustment)
df_valid['Error_adjusted'] = df_valid['MKTVCurYr'] - df_valid['PRICE_adjusted']
df_valid['AbsoluteError_adjusted'] = np.abs(df_valid['Error_adjusted'])
df_valid['PercentError_adjusted'] = (df_valid['Error_adjusted'] / df_valid['PRICE_adjusted']) * 100
df_valid['AbsolutePercentError_adjusted'] = np.abs(df_valid['PercentError_adjusted'])
df_valid['Ratio_adjusted'] = df_valid['MKTVCurYr'] / df_valid['PRICE_adjusted']

# 2. Overall Accuracy Metrics
print(f"\n{'='*80}")
print("2. OVERALL ACCURACY METRICS")
print(f"{'='*80}")

# Calculate summary statistics (NOMINAL)
n = len(df_valid)
mean_price = df_valid['PRICE'].mean()
mean_price_adj = df_valid['PRICE_adjusted'].mean()
mean_mktval = df_valid['MKTVCurYr'].mean()
mean_error = df_valid['Error'].mean()
mean_error_adj = df_valid['Error_adjusted'].mean()
mae = df_valid['AbsoluteError'].mean()
mae_adj = df_valid['AbsoluteError_adjusted'].mean()
rmse = np.sqrt((df_valid['Error']**2).mean())
rmse_adj = np.sqrt((df_valid['Error_adjusted']**2).mean())
mape = df_valid['AbsolutePercentError'].mean()
mape_adj = df_valid['AbsolutePercentError_adjusted'].mean()
median_ape = df_valid['AbsolutePercentError'].median()
median_ape_adj = df_valid['AbsolutePercentError_adjusted'].median()
correlation = df_valid['PRICE'].corr(df_valid['MKTVCurYr'])
correlation_adj = df_valid['PRICE_adjusted'].corr(df_valid['MKTVCurYr'])

# Calculate percentage within thresholds
within_5pct = (df_valid['AbsolutePercentError'] <= 5).sum() / n * 100
within_10pct = (df_valid['AbsolutePercentError'] <= 10).sum() / n * 100
within_20pct = (df_valid['AbsolutePercentError'] <= 20).sum() / n * 100
within_5pct_adj = (df_valid['AbsolutePercentError_adjusted'] <= 5).sum() / n * 100
within_10pct_adj = (df_valid['AbsolutePercentError_adjusted'] <= 10).sum() / n * 100
within_20pct_adj = (df_valid['AbsolutePercentError_adjusted'] <= 20).sum() / n * 100

print(f"\n   Sample Size: {n:,} properties")
print(f"\n   NOTE: Sale prices span {df_valid['SaleYear'].min()}-{df_valid['SaleYear'].max()}, market values are from {assessment_year}")
print(f"   Comparing NOMINAL prices (not adjusted for time) vs INFLATION-ADJUSTED prices (2023 dollars)")

print(f"\n   Mean Values:")
print(f"     Mean Sale Price (Nominal): ${mean_price:,.2f}")
print(f"     Mean Sale Price (2023 $): ${mean_price_adj:,.2f}")
print(f"     Mean Market Value: ${mean_mktval:,.2f}")
print(f"     Mean Difference (Nominal): ${mean_error:,.2f} ({mean_error/mean_price*100:.2f}%)")
print(f"     Mean Difference (2023 $): ${mean_error_adj:,.2f} ({mean_error_adj/mean_price_adj*100:.2f}%)")

print(f"\n   Error Metrics - NOMINAL (Not Adjusted for Time):")
print(f"     Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"     Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"     Mean Absolute Percent Error (MAPE): {mape:.2f}%")
print(f"     Median Absolute Percent Error: {median_ape:.2f}%")
print(f"     Pearson correlation: {correlation:.4f}")

print(f"\n   Error Metrics - INFLATION-ADJUSTED (2023 Dollars):")
print(f"     Mean Absolute Error (MAE): ${mae_adj:,.2f}")
print(f"     Root Mean Squared Error (RMSE): ${rmse_adj:,.2f}")
print(f"     Mean Absolute Percent Error (MAPE): {mape_adj:.2f}%")
print(f"     Median Absolute Percent Error: {median_ape_adj:.2f}%")
print(f"     Pearson correlation: {correlation_adj:.4f}")

print(f"\n   Accuracy Thresholds - NOMINAL:")
print(f"     Within 5%: {within_5pct:.2f}%")
print(f"     Within 10%: {within_10pct:.2f}%")
print(f"     Within 20%: {within_20pct:.2f}%")

print(f"\n   Accuracy Thresholds - INFLATION-ADJUSTED:")
print(f"     Within 5%: {within_5pct_adj:.2f}%")
print(f"     Within 10%: {within_10pct_adj:.2f}%")
print(f"     Within 20%: {within_20pct_adj:.2f}%")

# 3. Accuracy by Distance to Park
print(f"\n{'='*80}")
print("3. ACCURACY BY DISTANCE TO PARK")
print(f"{'='*80}")

if 'dist_to_park_miles' in df_valid.columns:
    # Create distance bands
    df_valid['DistBand'] = pd.cut(
        df_valid['dist_to_park_miles'],
        bins=[0, 0.1, 0.25, 0.75, 1.5, 3, np.inf],
        labels=['0-0.1', '0.1-0.25', '0.25-0.75', '0.75-1.5', '1.5-3', '>3']
    )
    
    distance_stats = df_valid.groupby('DistBand', observed=True).agg({
        'PRICE': 'count',
        'AbsolutePercentError': ['mean', 'median', 'std'],
        'Error': 'mean',
        'Ratio': 'mean'
    }).round(2)
    
    print(f"\n   {'Distance Band':<15} {'Count':>8} {'MAPE':>10} {'Median APE':>12} {'Mean Bias':>12} {'Mean Ratio':>12}")
    print(f"   {'-'*80}")
    for band in distance_stats.index:
        count = int(distance_stats.loc[band, ('PRICE', 'count')])
        mape = distance_stats.loc[band, ('AbsolutePercentError', 'mean')]
        median_ape = distance_stats.loc[band, ('AbsolutePercentError', 'median')]
        bias = distance_stats.loc[band, ('Error', 'mean')]
        ratio = distance_stats.loc[band, ('Ratio', 'mean')]
        print(f"   {str(band):<15} {count:>8,} {mape:>9.2f}% {median_ape:>11.2f}% ${bias:>10,.0f} {ratio:>11.3f}")

# 4. Accuracy by Sale Year
print(f"\n{'='*80}")
print("4. ACCURACY BY SALE YEAR")
print(f"{'='*80}")

year_stats = df_valid.groupby('SaleYear').agg({
    'PRICE': 'count',
    'AbsolutePercentError': ['mean', 'median'],
    'Error': 'mean',
    'Ratio': 'mean',
    'MKTVCurYr': 'mean'
}).round(2)

print(f"\n   {'Year':<6} {'Count':>8} {'MAPE':>10} {'Median APE':>12} {'Mean Bias':>12} {'Mean Ratio':>12}")
print(f"   {'-'*70}")
for year in sorted(df_valid['SaleYear'].unique()):
    if year in year_stats.index:
        count = int(year_stats.loc[year, ('PRICE', 'count')])
        mape = year_stats.loc[year, ('AbsolutePercentError', 'mean')]
        median_ape = year_stats.loc[year, ('AbsolutePercentError', 'median')]
        bias = year_stats.loc[year, ('Error', 'mean')]
        ratio = year_stats.loc[year, ('Ratio', 'mean')]
        print(f"   {year:<6} {count:>8,} {mape:>9.2f}% {median_ape:>11.2f}% ${bias:>10,.0f} {ratio:>11.3f}")

# 5. Accuracy by Price Range
print(f"\n{'='*80}")
print("5. ACCURACY BY PRICE RANGE")
print(f"{'='*80}")

# Create price quartiles
df_valid['PriceQuartile'] = pd.qcut(
    df_valid['PRICE'],
    q=4,
    labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']
)

price_stats = df_valid.groupby('PriceQuartile', observed=True).agg({
    'PRICE': ['count', 'mean'],
    'AbsolutePercentError': ['mean', 'median'],
    'Error': 'mean',
    'Ratio': 'mean'
}).round(2)

print(f"\n   {'Price Quartile':<20} {'Count':>8} {'Mean Price':>12} {'MAPE':>10} {'Median APE':>12} {'Mean Bias':>12} {'Mean Ratio':>12}")
print(f"   {'-'*95}")
for quartile in price_stats.index:
    count = int(price_stats.loc[quartile, ('PRICE', 'count')])
    mean_price = price_stats.loc[quartile, ('PRICE', 'mean')]
    mape = price_stats.loc[quartile, ('AbsolutePercentError', 'mean')]
    median_ape = price_stats.loc[quartile, ('AbsolutePercentError', 'median')]
    bias = price_stats.loc[quartile, ('Error', 'mean')]
    ratio = price_stats.loc[quartile, ('Ratio', 'mean')]
    print(f"   {str(quartile):<20} {count:>8,} ${mean_price:>11,.0f} {mape:>9.2f}% {median_ape:>11.2f}% ${bias:>10,.0f} {ratio:>11.3f}")

# 6. Recent Sales Only (2022-2023) - Minimal Time Gap
print(f"\n{'='*80}")
print("6. RECENT SALES ONLY (2022-2023) - MINIMAL TIME GAP")
print(f"{'='*80}")

df_recent = df_valid[df_valid['SaleYear'].isin([2022, 2023])].copy()
n_recent = len(df_recent)

if n_recent > 0:
    mape_recent = df_recent['AbsolutePercentError'].mean()
    mape_recent_adj = df_recent['AbsolutePercentError_adjusted'].mean()
    median_ape_recent = df_recent['AbsolutePercentError'].median()
    median_ape_recent_adj = df_recent['AbsolutePercentError_adjusted'].median()
    mean_error_recent = df_recent['Error'].mean()
    mean_error_recent_adj = df_recent['Error_adjusted'].mean()
    correlation_recent = df_recent['PRICE'].corr(df_recent['MKTVCurYr'])
    correlation_recent_adj = df_recent['PRICE_adjusted'].corr(df_recent['MKTVCurYr'])
    
    print(f"\n   Sample Size: {n_recent:,} properties (sales from 2022-2023 only)")
    print(f"   These sales have minimal time gap from assessment year ({assessment_year})")
    
    print(f"\n   Error Metrics - NOMINAL:")
    print(f"     Mean Absolute Percent Error (MAPE): {mape_recent:.2f}%")
    print(f"     Median Absolute Percent Error: {median_ape_recent:.2f}%")
    print(f"     Mean Error: ${mean_error_recent:,.2f}")
    print(f"     Correlation: {correlation_recent:.4f}")
    
    print(f"\n   Error Metrics - INFLATION-ADJUSTED:")
    print(f"     Mean Absolute Percent Error (MAPE): {mape_recent_adj:.2f}%")
    print(f"     Median Absolute Percent Error: {median_ape_recent_adj:.2f}%")
    print(f"     Mean Error: ${mean_error_recent_adj:,.2f}")
    print(f"     Correlation: {correlation_recent_adj:.4f}")
else:
    print(f"\n   No recent sales found in 2022-2023")

# 7. Systematic Bias Analysis
print(f"\n{'='*80}")
print("7. SYSTEMATIC BIAS ANALYSIS")
print(f"{'='*80}")

# Test if mean error is significantly different from zero (NOMINAL)
t_stat, p_value = stats.ttest_1samp(df_valid['Error'], 0)
t_stat_adj, p_value_adj = stats.ttest_1samp(df_valid['Error_adjusted'], 0)

print(f"\n   One-sample t-test: Is mean error significantly different from zero?")
print(f"\n   NOMINAL (Not Adjusted for Time):")
print(f"     Mean Error: ${mean_error:,.2f}")
print(f"     t-statistic: {t_stat:.4f}")
print(f"     p-value: {p_value:.6f}")
if p_value < 0.05:
    direction = "overvalued" if mean_error > 0 else "undervalued"
    print(f"     *** Market values are significantly {direction} relative to sale prices")
else:
    print(f"     Market values are not significantly biased on average")

print(f"\n   INFLATION-ADJUSTED (2023 Dollars):")
print(f"     Mean Error: ${mean_error_adj:,.2f}")
print(f"     t-statistic: {t_stat_adj:.4f}")
print(f"     p-value: {p_value_adj:.6f}")
if p_value_adj < 0.05:
    direction_adj = "overvalued" if mean_error_adj > 0 else "undervalued"
    print(f"     *** Market values are significantly {direction_adj} relative to inflation-adjusted sale prices")
else:
    print(f"     Market values are not significantly biased on average")

# Calculate percentage overvalued vs undervalued
overvalued = (df_valid['Error'] > 0).sum() / n * 100
undervalued = (df_valid['Error'] < 0).sum() / n * 100
exact = (df_valid['Error'] == 0).sum() / n * 100
overvalued_adj = (df_valid['Error_adjusted'] > 0).sum() / n * 100
undervalued_adj = (df_valid['Error_adjusted'] < 0).sum() / n * 100
exact_adj = (df_valid['Error_adjusted'] == 0).sum() / n * 100

print(f"\n   Distribution of Errors - NOMINAL:")
print(f"     Overvalued (Market Value > Sale Price): {overvalued:.2f}%")
print(f"     Undervalued (Market Value < Sale Price): {undervalued:.2f}%")
print(f"     Exact Match: {exact:.2f}%")

print(f"\n   Distribution of Errors - INFLATION-ADJUSTED:")
print(f"     Overvalued (Market Value > Adjusted Sale Price): {overvalued_adj:.2f}%")
print(f"     Undervalued (Market Value < Adjusted Sale Price): {undervalued_adj:.2f}%")
print(f"     Exact Match: {exact_adj:.2f}%")

# 8. Outlier Analysis
print(f"\n{'='*80}")
print("8. OUTLIER ANALYSIS")
print(f"{'='*80}")

# Define outliers as properties with >50% absolute percent error
outliers = df_valid[df_valid['AbsolutePercentError'] > 50].copy()
outliers_adj = df_valid[df_valid['AbsolutePercentError_adjusted'] > 50].copy()
outlier_pct = len(outliers) / n * 100
outlier_pct_adj = len(outliers_adj) / n * 100

print(f"\n   Properties with >50% absolute error - NOMINAL: {len(outliers):,} ({outlier_pct:.2f}%)")
if len(outliers) > 0:
    print(f"     Mean sale price of outliers: ${outliers['PRICE'].mean():,.2f}")
    print(f"     Mean market value of outliers: ${outliers['MKTVCurYr'].mean():,.2f}")
    print(f"     Mean absolute percent error: {outliers['AbsolutePercentError'].mean():.2f}%")

print(f"\n   Properties with >50% absolute error - INFLATION-ADJUSTED: {len(outliers_adj):,} ({outlier_pct_adj:.2f}%)")
if len(outliers_adj) > 0:
    print(f"     Mean sale price (adjusted) of outliers: ${outliers_adj['PRICE_adjusted'].mean():,.2f}")
    print(f"     Mean market value of outliers: ${outliers_adj['MKTVCurYr'].mean():,.2f}")
    print(f"     Mean absolute percent error: {outliers_adj['AbsolutePercentError_adjusted'].mean():.2f}%")

# 9. Save Results
print(f"\n{'='*80}")
print("8. SAVING RESULTS")
print(f"{'='*80}")

# Save detailed results to CSV
results_df = df_valid[[
    'PRICE', 'PRICE_adjusted', 'MKTVCurYr', 
    'Error', 'Error_adjusted', 'AbsoluteError', 'AbsoluteError_adjusted',
    'PercentError', 'PercentError_adjusted', 
    'AbsolutePercentError', 'AbsolutePercentError_adjusted',
    'Ratio', 'Ratio_adjusted',
    'SaleYear', 'CPI_adjustment', 'dist_to_park_miles', 'DistBand', 'PriceQuartile'
]].copy()

# Ensure output directory exists
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
results_df.to_csv(output_csv_path, index=False)
print(f"   Detailed results saved to: {output_csv_path}")

# Save summary to text file
with open(output_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("MARKET VALUATION ACCURACY TEST\n")
    f.write("Economic Impact Report for Butler County Parks\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Sample Size: {n:,} properties with both sale price and market value\n\n")
    
    f.write("="*80 + "\n")
    f.write("OVERALL ACCURACY METRICS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"NOTE: Sale prices span {df_valid['SaleYear'].min()}-{df_valid['SaleYear'].max()}, ")
    f.write(f"market values are from {assessment_year}\n")
    f.write(f"Comparing NOMINAL prices (not adjusted for time) vs INFLATION-ADJUSTED prices (2023 dollars)\n\n")
    
    f.write("NOMINAL (Not Adjusted for Time):\n")
    f.write(f"  Mean Sale Price: ${mean_price:,.2f}\n")
    f.write(f"  Mean Market Value: ${mean_mktval:,.2f}\n")
    f.write(f"  Mean Difference: ${mean_error:,.2f} ({mean_error/mean_price*100:.2f}%)\n")
    f.write(f"  Mean Absolute Error (MAE): ${mae:,.2f}\n")
    f.write(f"  Root Mean Squared Error (RMSE): ${rmse:,.2f}\n")
    f.write(f"  Mean Absolute Percent Error (MAPE): {mape:.2f}%\n")
    f.write(f"  Median Absolute Percent Error: {median_ape:.2f}%\n")
    f.write(f"  Pearson Correlation: {correlation:.4f}\n")
    f.write(f"  Accuracy Thresholds:\n")
    f.write(f"    Within 5%: {within_5pct:.2f}%\n")
    f.write(f"    Within 10%: {within_10pct:.2f}%\n")
    f.write(f"    Within 20%: {within_20pct:.2f}%\n\n")
    
    f.write("INFLATION-ADJUSTED (2023 Dollars):\n")
    f.write(f"  Mean Sale Price (2023 $): ${mean_price_adj:,.2f}\n")
    f.write(f"  Mean Market Value: ${mean_mktval:,.2f}\n")
    f.write(f"  Mean Difference: ${mean_error_adj:,.2f} ({mean_error_adj/mean_price_adj*100:.2f}%)\n")
    f.write(f"  Mean Absolute Error (MAE): ${mae_adj:,.2f}\n")
    f.write(f"  Root Mean Squared Error (RMSE): ${rmse_adj:,.2f}\n")
    f.write(f"  Mean Absolute Percent Error (MAPE): {mape_adj:.2f}%\n")
    f.write(f"  Median Absolute Percent Error: {median_ape_adj:.2f}%\n")
    f.write(f"  Pearson Correlation: {correlation_adj:.4f}\n")
    f.write(f"  Accuracy Thresholds:\n")
    f.write(f"    Within 5%: {within_5pct_adj:.2f}%\n")
    f.write(f"    Within 10%: {within_10pct_adj:.2f}%\n")
    f.write(f"    Within 20%: {within_20pct_adj:.2f}%\n\n")
    
    if n_recent > 0:
        f.write("RECENT SALES ONLY (2022-2023):\n")
        f.write(f"  Sample Size: {n_recent:,} properties\n")
        f.write(f"  MAPE (Nominal): {mape_recent:.2f}%\n")
        f.write(f"  MAPE (Adjusted): {mape_recent_adj:.2f}%\n")
        f.write(f"  Correlation (Nominal): {correlation_recent:.4f}\n")
        f.write(f"  Correlation (Adjusted): {correlation_recent_adj:.4f}\n\n")
    
    f.write("="*80 + "\n")
    f.write("SYSTEMATIC BIAS ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write("NOMINAL (Not Adjusted for Time):\n")
    f.write(f"  One-sample t-test (H0: mean error = 0):\n")
    f.write(f"    t-statistic: {t_stat:.4f}\n")
    f.write(f"    p-value: {p_value:.6f}\n")
    if p_value < 0.05:
        direction = "overvalued" if mean_error > 0 else "undervalued"
        f.write(f"    Result: Market values are significantly {direction}\n")
    else:
        f.write(f"    Result: No significant bias detected\n")
    f.write(f"  Distribution:\n")
    f.write(f"    Overvalued: {overvalued:.2f}%\n")
    f.write(f"    Undervalued: {undervalued:.2f}%\n")
    f.write(f"    Exact Match: {exact:.2f}%\n\n")
    
    f.write("INFLATION-ADJUSTED (2023 Dollars):\n")
    f.write(f"  One-sample t-test (H0: mean error = 0):\n")
    f.write(f"    t-statistic: {t_stat_adj:.4f}\n")
    f.write(f"    p-value: {p_value_adj:.6f}\n")
    if p_value_adj < 0.05:
        direction_adj = "overvalued" if mean_error_adj > 0 else "undervalued"
        f.write(f"    Result: Market values are significantly {direction_adj}\n")
    else:
        f.write(f"    Result: No significant bias detected\n")
    f.write(f"  Distribution:\n")
    f.write(f"    Overvalued: {overvalued_adj:.2f}%\n")
    f.write(f"    Undervalued: {undervalued_adj:.2f}%\n")
    f.write(f"    Exact Match: {exact_adj:.2f}%\n\n")
    
    f.write("="*80 + "\n")
    f.write("INTERPRETATION\n")
    f.write("="*80 + "\n\n")
    
    f.write("This analysis compares assessed market values (MKTVCurYr) to actual sale prices (PRICE)\n")
    f.write("for properties that sold. Sale prices span 2014-2023, while market values are from 2023.\n\n")
    
    f.write("IMPORTANT: The nominal comparison mixes different time periods. A 2014 sale price\n")
    f.write("compared to a 2023 market value includes 9 years of inflation and property appreciation.\n")
    f.write("The inflation-adjusted comparison converts all sale prices to 2023 dollars for a fair comparison.\n\n")
    
    f.write("Key Findings:\n\n")
    
    f.write("1. Overall Accuracy:\n")
    f.write(f"   - Nominal MAPE: {mape:.2f}% (includes time effects)\n")
    f.write(f"   - Inflation-Adjusted MAPE: {mape_adj:.2f}% (removes time effects)\n")
    f.write("   - The inflation-adjusted MAPE better reflects true assessment accuracy\n\n")
    
    f.write("2. Systematic Bias:\n")
    f.write(f"   - Nominal: Market values are {('overvalued' if mean_error > 0 else 'undervalued')} by {abs(mean_error/mean_price*100):.2f}%\n")
    f.write(f"   - Inflation-Adjusted: Market values are {('overvalued' if mean_error_adj > 0 else 'undervalued')} by {abs(mean_error_adj/mean_price_adj*100):.2f}%\n")
    f.write("   - The inflation-adjusted bias is more meaningful for current assessments\n\n")
    
    f.write("3. Recent Sales (2022-2023):\n")
    if n_recent > 0:
        f.write(f"   - These sales have minimal time gap from assessment year\n")
        f.write(f"   - Nominal MAPE: {mape_recent:.2f}%\n")
        f.write(f"   - Adjusted MAPE: {mape_recent_adj:.2f}%\n")
        f.write("   - Recent sales provide the most accurate assessment validation\n\n")
    
    f.write("4. Implications for Full-Stock Analysis:\n")
    if abs(mean_error_adj/mean_price_adj) < 0.05:
        f.write("   - Market values appear relatively unbiased after inflation adjustment\n")
        f.write("   - Full-stock analysis using market values should be reasonably reliable\n")
    else:
        f.write("   - Market values show systematic bias even after inflation adjustment\n")
        f.write("   - Full-stock estimates may need adjustment or interpretation with caution\n")
    
    f.write(f"\n5. Correlation: Inflation-adjusted correlation ({correlation_adj:.3f}) is more meaningful\n")
    f.write("   than nominal correlation ({correlation:.3f}) as it removes time effects.\n")

print(f"\n   Summary saved to: {output_path}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")

