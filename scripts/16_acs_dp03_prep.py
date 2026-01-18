"""
ACS DP03 Data Preparation Script
Economic Impact Report for Butler County Parks

This script loads ACS DP03 (Selected Economic Characteristics) data from a ZIP file,
extracts tract-level variables (median household income, poverty, unemployment),
builds a GEOID column, filters to Butler County, and saves the processed data.
"""

import pandas as pd
import zipfile
import os
import tempfile
import shutil

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input path - DP03 ZIP file
dp03_zip_path = os.path.join(
    project_root,
    'data_raw',
    'Health and Census Tracts',
    'ACSDP5Y2023.DP03_2026-01-07T060020.zip'
)

# Output path
output_dir = os.path.join(project_root, 'data_intermediate')
output_path = os.path.join(output_dir, 'butler_acs_dp03_covariates.csv')

print("="*60)
print("ACS DP03 DATA PREPARATION")
print("="*60)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 1. Extract and load DP03 data from ZIP
print(f"\n1. Loading ACS DP03 data from ZIP...")
print(f"   Path: {dp03_zip_path}")

# Extract zip to temporary directory
temp_dir = tempfile.mkdtemp()
try:
    with zipfile.ZipFile(dp03_zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find the data CSV file
    data_file = None
    for file in os.listdir(temp_dir):
        if file.endswith('-Data.csv'):
            data_file = os.path.join(temp_dir, file)
            break
    
    if data_file is None:
        # Check in subdirectories
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('-Data.csv'):
                    data_file = os.path.join(root, file)
                    break
            if data_file:
                break
    
    if data_file is None:
        raise FileNotFoundError(f"Data CSV file not found in ZIP: {dp03_zip_path}")
    
    print(f"   Found data file: {os.path.basename(data_file)}")
    
    # Load the data
    acs_df = pd.read_csv(data_file, low_memory=False)
    print(f"   Total records loaded: {len(acs_df):,}")
    
finally:
    # Clean up temporary directory
    shutil.rmtree(temp_dir)

# 2. Build GEOID column from GEO_ID
print(f"\n2. Building GEOID column from GEO_ID...")
# GEO_ID format: "1400000US39017000100" where last 11 digits are GEOID
# Extract the last 11 digits after "US"
acs_df['GEOID'] = acs_df['GEO_ID'].str.extract(r'US(\d{11})$')
print(f"   GEOID extracted. Sample GEOIDs: {acs_df['GEOID'].head(3).tolist()}")

# Filter to census tracts only (GEO_ID contains "1400000US" for tracts)
print(f"\n   Filtering to census tracts...")
acs_tracts = acs_df[acs_df['GEO_ID'].str.contains('1400000US', na=False)].copy()
print(f"   Census tracts: {len(acs_tracts):,}")

# 3. Filter to Butler County, Ohio
print(f"\n3. Filtering to Butler County, Ohio...")
# Butler County FIPS: 39017 (state 39 + county 017)
# GEOID format: 39017000100 (state 39 + county 017 + tract)
butler_acs = acs_tracts[acs_tracts['GEOID'].str.startswith('39017', na=False)].copy()
print(f"   Butler County tracts: {len(butler_acs):,}")

if len(butler_acs) == 0:
    raise ValueError("No Butler County tracts found in ACS data!")

# 4. Select key variables
print(f"\n4. Selecting key variables...")

# Define variables to extract:
# - DP03_0062E: Median household income (dollars)
# - DP03_0119PE: Percent of families below poverty level
# - DP03_0009PE: Unemployment rate (percent)
# Note: Education variables are typically in DP02, not DP03

variables_to_extract = {
    'GEOID': 'GEOID',
    'NAME': 'NAME',
    'DP03_0062E': 'median_household_income',
    'DP03_0119PE': 'pct_families_below_poverty',
    'DP03_0009PE': 'unemployment_rate'
}

# Check which variables exist
available_vars = {}
missing_vars = []
for acs_var, new_name in variables_to_extract.items():
    if acs_var in butler_acs.columns:
        available_vars[acs_var] = new_name
        print(f"   Found: {acs_var} -> {new_name}")
    else:
        missing_vars.append(acs_var)
        print(f"   Warning: {acs_var} not found in data")

if missing_vars:
    print(f"\n   Missing variables: {missing_vars}")
    print(f"   Proceeding with available variables only.")

# Create output dataframe with selected variables
output_df = butler_acs[list(available_vars.keys())].copy()

# Rename columns
output_df = output_df.rename(columns=available_vars)

# 5. Clean and convert data types
print(f"\n5. Cleaning and converting data types...")

# Convert numeric columns, handling any non-numeric values
numeric_cols = ['median_household_income', 'pct_families_below_poverty', 'unemployment_rate']
for col in numeric_cols:
    if col in output_df.columns:
        # Replace non-numeric values (like '-', 'N', '(X)') with NaN
        output_df[col] = pd.to_numeric(output_df[col], errors='coerce')
        print(f"   Converted {col} to numeric")
        print(f"     Missing values: {output_df[col].isna().sum()}")
        print(f"     Mean: {output_df[col].mean():.2f}")

# 6. Display summary statistics
print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")
print(f"Records: {len(output_df):,}")
print(f"\nVariables extracted:")
for col in output_df.columns:
    if col != 'GEOID' and col != 'NAME':
        print(f"  {col}:")
        print(f"    Mean: {output_df[col].mean():.2f}")
        print(f"    Min: {output_df[col].min():.2f}")
        print(f"    Max: {output_df[col].max():.2f}")
        print(f"    Missing: {output_df[col].isna().sum()}")

print(f"\nSample data:")
print(output_df.head(10).to_string(index=False))

# 7. Save to CSV
print(f"\n6. Saving to CSV...")
print(f"   Output path: {output_path}")

output_df.to_csv(output_path, index=False)
print(f"   [OK] Saved successfully!")

print(f"\n{'='*60}")
print("ACS DP03 DATA PREPARATION COMPLETE")
print(f"{'='*60}")
print(f"\nNote: Education variables (e.g., bachelor's degree, high school completion)")
print(f"      are typically found in ACS DP02 (Social Characteristics), not DP03.")
