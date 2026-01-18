"""
Spatial Autocorrelation Analysis: Moran's I on Model Residuals
Economic Impact Report for Butler County Parks

This script refits the Frequent Mental Distress WLS model (density + reduced
demographics, excluding pct_non_hispanic_white), computes Moran's I on the
model residuals using Queen contiguity weights, and creates a residuals map.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os

# Try to import spatial autocorrelation libraries
try:
    from libpysal.weights import Queen
    from esda.moran import Moran
    SPATIAL_LIBS_AVAILABLE = True
except ImportError:
    try:
        # Alternative: pysal
        import pysal
        from pysal.weights import Queen
        from pysal.esda.moran import Moran
        SPATIAL_LIBS_AVAILABLE = True
    except ImportError:
        SPATIAL_LIBS_AVAILABLE = False
        print("Warning: libpysal/esda not available. Install with: pip install libpysal esda")

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input paths
input_gpkg_path = os.path.join(project_root, 'data_final', 'butler_tract_health_model_data.gpkg')
parks_path = os.path.join(project_root, 'data_intermediate', 'butler_county_parks.shp')

# Output paths
output_dir = os.path.join(project_root, 'results')
figures_dir = os.path.join(project_root, 'figures')
moran_output_path = os.path.join(output_dir, 'moran_residuals_mental_distress.txt')
residuals_map_path = os.path.join(figures_dir, 'mental_distress_residuals_map.png')

print("="*60)
print("SPATIAL AUTOCORRELATION ANALYSIS: MORAN'S I")
print("="*60)
print("Model: WLS with Density + Reduced Demographics")
print("(Excluding pct_non_hispanic_white)")
print("="*60)

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

if not SPATIAL_LIBS_AVAILABLE:
    raise ImportError("Required libraries (libpysal/esda) not available. Please install: pip install libpysal esda")

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
# 5. FIT WLS MODEL
# ============================================================================
print(f"\n5. Fitting WLS model...")

# Prepare regression variables
reg_vars = (['park_gravity_index_z'] + list(available_controls.values()) + 
            ['log_pop_density'] + list(available_demographics_reduced.values()) + 
            [outcome_col, 'total_population', 'GEOID'])

# Create subset with non-missing data
df_model = gdf[reg_vars + ['geometry']].copy()
df_model = df_model.dropna(subset=reg_vars)

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
model = sm.WLS(y, X, weights=weights)
results = model.fit(cov_type='HC1')  # Robust standard errors (HC1)

print(f"\n   Model Summary:")
print(f"   R-squared: {results.rsquared:.4f}")
print(f"   Observations: {len(df_model):,}")

# Extract residuals
residuals = results.resid
df_model['residuals'] = residuals

print(f"\n   Residuals Statistics:")
print(f"     Min:    {residuals.min():.6f}")
print(f"     Mean:   {residuals.mean():.6f}")
print(f"     Max:    {residuals.max():.6f}")
print(f"     Std:    {residuals.std():.6f}")

# ============================================================================
# 6. CREATE SPATIAL WEIGHTS MATRIX (QUEEN CONTIGUITY)
# ============================================================================
print(f"\n6. Creating spatial weights matrix (Queen contiguity)...")

# Ensure geometry is valid
df_model = df_model[df_model.geometry.is_valid].copy()

if len(df_model) == 0:
    raise ValueError("No valid geometries after filtering!")

# Create Queen contiguity weights
try:
    w = Queen.from_dataframe(df_model, use_index=False)
    print(f"   Spatial weights matrix created")
    print(f"     Number of observations: {w.n}")
    print(f"     Number of non-zero weights: {w.nonzero}")
    print(f"     Average number of neighbors: {w.nonzero / w.n:.2f}")
except Exception as e:
    print(f"   Error creating weights matrix: {e}")
    raise

# Check for islands (tracts with no neighbors)
islands = [i for i in range(w.n) if w.cardinalities[i] == 0]
if len(islands) > 0:
    print(f"   Warning: {len(islands)} islands (tracts with no neighbors) detected")
    print(f"   These will be excluded from Moran's I calculation")

# ============================================================================
# 7. COMPUTE MORAN'S I
# ============================================================================
print(f"\n7. Computing Moran's I on residuals...")

# Extract residuals as array
residuals_array = df_model['residuals'].values

# Compute Moran's I
try:
    moran = Moran(residuals_array, w, two_tailed=True)
    
    moran_i = moran.I
    moran_pvalue = moran.p_norm  # Normal approximation p-value
    moran_zscore = moran.z_norm  # Normal approximation z-score
    
    print(f"\n   Moran's I Results:")
    print(f"     Moran's I: {moran_i:.6f}")
    print(f"     Expected I (under null): {moran.EI:.6f}")
    print(f"     Z-score: {moran_zscore:.6f}")
    print(f"     P-value (normal approximation): {moran_pvalue:.6f}")
    
    # Interpretation
    if moran_pvalue < 0.01:
        sig_level = "*** (p < 0.01)"
    elif moran_pvalue < 0.05:
        sig_level = "** (p < 0.05)"
    elif moran_pvalue < 0.10:
        sig_level = "* (p < 0.10)"
    else:
        sig_level = "Not significant"
    
    print(f"     Significance: {sig_level}")
    
    if moran_i > 0:
        print(f"     Interpretation: Positive spatial autocorrelation (clustering of similar residuals)")
    else:
        print(f"     Interpretation: Negative spatial autocorrelation (dispersion of residuals)")
    
except Exception as e:
    print(f"   Error computing Moran's I: {e}")
    raise

# ============================================================================
# 8. SAVE MORAN'S I RESULTS
# ============================================================================
print(f"\n8. Saving Moran's I results...")
print(f"   Output path: {moran_output_path}")

with open(moran_output_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("MORAN'S I TEST FOR SPATIAL AUTOCORRELATION\n")
    f.write("="*60 + "\n\n")
    f.write("Model: Frequent Mental Distress WLS\n")
    f.write("Specification: Density + Reduced Demographics (excluding pct_non_hispanic_white)\n")
    f.write("Test Variable: Model Residuals\n")
    f.write("Spatial Weights: Queen Contiguity\n")
    f.write("="*60 + "\n\n")
    f.write(f"Number of Observations: {len(df_model):,}\n")
    f.write(f"Number of Non-zero Weights: {w.nonzero:,}\n")
    f.write(f"Average Number of Neighbors: {w.nonzero / w.n:.2f}\n")
    if len(islands) > 0:
        f.write(f"Number of Islands (no neighbors): {len(islands)}\n")
    f.write("\n")
    f.write("Moran's I Statistics:\n")
    f.write("-"*60 + "\n")
    f.write(f"Moran's I:              {moran_i:+.6f}\n")
    f.write(f"Expected I (null):      {moran.EI:+.6f}\n")
    f.write(f"Variance:                {moran.VI_norm:.6f}\n")
    f.write(f"Z-score:                 {moran_zscore:+.6f}\n")
    f.write(f"P-value (normal):        {moran_pvalue:.6f}\n")
    f.write(f"Significance:            {sig_level}\n")
    f.write("\n")
    f.write("Interpretation:\n")
    f.write("-"*60 + "\n")
    if moran_i > 0:
        f.write("Positive spatial autocorrelation detected.\n")
        f.write("Similar residual values tend to cluster spatially.\n")
        f.write("This suggests that the model may be missing spatially-structured\n")
        f.write("predictors or that spatial spillovers are present.\n")
    else:
        f.write("Negative spatial autocorrelation detected.\n")
        f.write("Similar residual values tend to be dispersed spatially.\n")
    if moran_pvalue < 0.05:
        f.write("\nThe spatial autocorrelation is statistically significant,\n")
        f.write("suggesting that a spatial econometric model (e.g., spatial lag\n")
        f.write("or spatial error model) may be more appropriate.\n")
    else:
        f.write("\nThe spatial autocorrelation is not statistically significant,\n")
        f.write("suggesting that the standard regression model is adequate.\n")
    f.write("\n")
    f.write("="*60 + "\n")

print(f"   [OK] Moran's I results saved successfully!")

# ============================================================================
# 9. CREATE RESIDUALS MAP
# ============================================================================
print(f"\n9. Creating residuals map...")

fig, ax = plt.subplots(figsize=(12, 10))

# Create color normalization centered at zero
vmin = residuals.min()
vmax = residuals.max()
vcenter = 0
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

# Plot residuals
df_model.plot(
    column='residuals',
    ax=ax,
    cmap='RdBu_r',
    norm=norm,
    edgecolor='black',
    linewidth=0.5,
    legend=True,
    legend_kwds={'label': 'Residuals', 'shrink': 0.8, 'orientation': 'horizontal', 'pad': 0.02}
)

# Customize plot
ax.set_title('Model Residuals: Frequent Mental Distress WLS\n(Spatial Autocorrelation Test)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Easting (US Survey Feet)', fontsize=11)
ax.set_ylabel('Northing (US Survey Feet)', fontsize=11)
ax.axis('off')

# Add text box with Moran's I results
textstr = f"Moran's I: {moran_i:+.4f}\nP-value: {moran_pvalue:.4f}\n{sig_level}"
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig(residuals_map_path, dpi=300, bbox_inches='tight')
print(f"   Saved to: {residuals_map_path}")
plt.close()

# ============================================================================
# 10. COMPLETE
# ============================================================================
print(f"\n{'='*60}")
print("SPATIAL AUTOCORRELATION ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"\nOutputs created:")
print(f"  1. Moran's I results: {moran_output_path}")
print(f"  2. Residuals map: {residuals_map_path}")
print(f"\nSummary:")
print(f"  Moran's I: {moran_i:+.6f}")
print(f"  P-value: {moran_pvalue:.6f}")
print(f"  Significance: {sig_level}")
