"""
Comprehensive Benefits Dashboard
Economic Impact Report for Butler County Parks

Creates a dashboard showing:
- Top: Housing benefits per distance band
- Bottom: Health benefits per distance band  
- Footer: County-wide summary statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input paths
housing_benefits_path = os.path.join(project_root, 'results', 'distance_band_benefits_summary.csv')
health_data_path = os.path.join(project_root, 'data_final', 'butler_tract_health_model_data_with_greenness.csv')
health_data_gpkg_path = os.path.join(project_root, 'data_final', 'butler_tract_health_model_data_with_greenness.gpkg')

# Output path
output_dir = os.path.join(project_root, 'figures')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'comprehensive_benefits_dashboard.png')

print("="*80)
print("COMPREHENSIVE BENEFITS DASHBOARD")
print("="*80)

# ============================================================================
# 1. LOAD HOUSING BENEFITS DATA
# ============================================================================
print(f"\n1. Loading housing benefits data...")
housing_df = pd.read_csv(housing_benefits_path)

# Exclude 0-0.1 mile band
housing_df = housing_df[housing_df['distance_band'] != '0-0.1'].copy()
housing_df = housing_df.sort_values('mean_distance')

print(f"   Housing distance bands: {housing_df['distance_band'].tolist()}")

# ============================================================================
# 2. CALCULATE HEALTH BENEFITS BY DISTANCE BAND
# ============================================================================
print(f"\n2. Calculating health benefits by distance band...")

# Load health data
if os.path.exists(health_data_gpkg_path):
    import geopandas as gpd
    health_df = gpd.read_file(health_data_gpkg_path)
    print(f"   Loaded {len(health_df)} tracts from GPKG")
else:
    health_df = pd.read_csv(health_data_path, low_memory=False)
    print(f"   Loaded {len(health_df)} tracts from CSV")

# Assign distance bands to tracts
def assign_dist_band(dist_miles):
    """Assign distance band category based on distance in miles."""
    if pd.isna(dist_miles):
        return None
    elif dist_miles <= 0.1:
        return '0-0.1'
    elif dist_miles <= 0.25:
        return '0.1-0.25'
    elif dist_miles <= 0.75:
        return '0.25-0.75'
    elif dist_miles <= 1.5:
        return '0.75-1.5'
    elif dist_miles <= 3:
        return '1.5-3'
    else:
        return '>3'

if 'dist_to_park_miles' in health_df.columns:
    health_df['DistBand'] = health_df['dist_to_park_miles'].apply(assign_dist_band)
    
    # Exclude 0-0.1 and >3 bands to match housing analysis
    health_df = health_df[health_df['DistBand'].isin(['0.1-0.25', '0.25-0.75', '0.75-1.5', '1.5-3'])].copy()
    
    # Health outcomes to analyze
    health_outcomes = {
        'MHLTH_CrudePrev': {
            'label': 'Mental Distress',
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
    
    # Calculate health statistics by distance band
    health_band_stats = []
    
    for band in ['0.1-0.25', '0.25-0.75', '0.75-1.5', '1.5-3']:
        band_df = health_df[health_df['DistBand'] == band].copy()
        
        if len(band_df) == 0:
            continue
        
        stats = {'distance_band': band}
        
        # Calculate mean prevalence for each outcome
        for col, info in health_outcomes.items():
            outcome_col = info['preferred'] if info['preferred'] in band_df.columns else col
            if outcome_col in band_df.columns:
                mean_prev = band_df[outcome_col].mean()
                stats[info['label']] = mean_prev
        
        # Calculate population-weighted averages if population data available
        if 'total_population' in band_df.columns:
            stats['total_population'] = band_df['total_population'].sum()
            stats['tract_count'] = len(band_df)
        
        health_band_stats.append(stats)
    
    health_band_df = pd.DataFrame(health_band_stats)
    
    # Calculate differences relative to reference band (1.5-3)
    if '1.5-3' in health_band_df['distance_band'].values:
        ref_row = health_band_df[health_band_df['distance_band'] == '1.5-3'].iloc[0]
        
        for outcome in ['Mental Distress', 'Obesity', 'Physical Inactivity']:
            if outcome in health_band_df.columns:
                ref_value = ref_row[outcome]
                health_band_df[f'{outcome}_diff'] = health_band_df[outcome] - ref_value
                health_band_df[f'{outcome}_pct_diff'] = ((health_band_df[outcome] - ref_value) / ref_value) * 100
        
        print(f"   Health statistics calculated for {len(health_band_df)} distance bands")
    else:
        print(f"   Warning: Reference band (1.5-3) not found in health data")
        health_band_df = pd.DataFrame()
else:
    print(f"   Warning: dist_to_park_miles not found in health data")
    health_band_df = pd.DataFrame()

# ============================================================================
# 3. CALCULATE COUNTY-WIDE SUMMARY STATISTICS
# ============================================================================
print(f"\n3. Calculating county-wide summary statistics...")

# Housing summary
total_properties = housing_df['property_count'].sum()
total_housing_value = housing_df['total_value'].sum()
total_housing_uplift = housing_df['total_value_uplift'].sum()
avg_housing_benefit = total_housing_uplift / total_properties if total_properties > 0 else 0

# Health summary (from Phase 2 economic valuation)
valuation_path = os.path.join(project_root, 'results', 'phase2_economic_valuation_valuation.csv')
if os.path.exists(valuation_path):
    val_df = pd.read_csv(valuation_path)
    # Use moderate scenario for summary
    moderate_val = val_df[val_df['scenario'] == 'moderate']
    total_health_savings = moderate_val['annual_savings'].sum() if len(moderate_val) > 0 else 0
else:
    total_health_savings = 0

# Total population
if 'total_population' in health_df.columns:
    total_pop = health_df['total_population'].sum()
else:
    total_pop = 0

print(f"   County-wide statistics calculated")

# ============================================================================
# 4. LOAD MAP DATA (DISTANCE MAP WITH COLORED PROPERTIES)
# ============================================================================
print(f"\n4. Loading map data for distance visualization...")

# Load parcels and parks for map
parks_path = os.path.join(project_root, 'data_intermediate', 'butler_county_parks.shp')
parcels_path = os.path.join(project_root, 'data_raw', 'CURRENTPARCELS', 'CURRENTPARCELS.shp')

# Distance band color scheme (matching butler_county_distance_map.py)
color_map = {
    '0-0.1': '#c0392b',      # Dark red - very close
    '0.1-0.25': '#e74c3c',   # Red
    '0.25-0.75': '#e67e22',  # Red-orange
    '0.75-1.5': '#f39c12',   # Orange
    '1.5-3': '#f1c40f',      # Yellow-orange
    '3-5': '#f4d03f',        # Light yellow-orange
    '5-10': '#f7dc6f',       # Yellow
    '>10': '#fef9e7'         # Light yellow - very far
}

band_order = ['0-0.1', '0.1-0.25', '0.25-0.75', '0.75-1.5', '1.5-3', '3-5', '5-10', '>10']

def assign_dist_band(dist_miles):
    """Assign distance band category based on distance in miles."""
    if dist_miles <= 0.1:
        return '0-0.1'
    elif dist_miles <= 0.25:
        return '0.1-0.25'
    elif dist_miles <= 0.75:
        return '0.25-0.75'
    elif dist_miles <= 1.5:
        return '0.75-1.5'
    elif dist_miles <= 3:
        return '1.5-3'
    elif dist_miles <= 5:
        return '3-5'
    elif dist_miles <= 10:
        return '5-10'
    else:
        return '>10'

try:
    import geopandas as gpd
    import contextily as ctx
    
    # Load parks
    parks_gdf = gpd.read_file(parks_path)
    target_crs = 'EPSG:3402'
    
    if str(parks_gdf.crs) != target_crs:
        parks_gdf = parks_gdf.to_crs(target_crs)
    
    # Load parcels
    if os.path.exists(parcels_path):
        try:
            parcels_gdf = gpd.read_file(parcels_path)
            if str(parcels_gdf.crs) != target_crs:
                parcels_gdf = parcels_gdf.to_crs(target_crs)
            
            # Filter to residential
            land_use_col = None
            for col in ['CLASS', 'LANDUSE', 'LUC']:
                if col in parcels_gdf.columns:
                    land_use_col = col
                    break
            
            if land_use_col:
                parcels_filtered = parcels_gdf[parcels_gdf[land_use_col] == 'R'].copy()
                print(f"   Filtered to {len(parcels_filtered):,} residential parcels")
            else:
                parcels_filtered = parcels_gdf.copy()
            
            # Calculate distances to nearest park
            print(f"   Calculating distances to nearest park...")
            parcels_filtered['dist_to_park_ft'] = parcels_filtered.geometry.apply(
                lambda geom: parks_gdf.geometry.distance(geom).min()
            )
            parcels_filtered['dist_to_park_miles'] = parcels_filtered['dist_to_park_ft'] / 5280.0
            
            # Assign distance bands
            parcels_filtered['DistBand'] = parcels_filtered['dist_to_park_miles'].apply(assign_dist_band)
            
            print(f"   Distance bands assigned")
            print(f"   Distance range: {parcels_filtered['dist_to_park_miles'].min():.2f} - {parcels_filtered['dist_to_park_miles'].max():.2f} miles")
            
            has_map_data = True
        except Exception as e:
            print(f"   Warning: Could not load parcels: {e}")
            parcels_filtered = None
            has_map_data = False
    else:
        parcels_filtered = None
        has_map_data = False
    
except Exception as e:
    print(f"   Warning: Could not load map data: {e}")
    has_map_data = False
    parks_gdf = None
    parcels_filtered = None

# ============================================================================
# 5. CREATE DASHBOARD WITH MAP
# ============================================================================
print(f"\n5. Creating comprehensive dashboard with map...")

# Create figure with map on left, charts on right
fig = plt.figure(figsize=(22, 14))
# Grid: Map on left, charts and stats on right, gradient circle on far right
# Layout: 
# - Column 0: Map (full height)
# - Column 1: Top row = Housing chart, Bottom row = Health chart
# - Column 2: Statistics (full height) and gradient circle
gs = fig.add_gridspec(2, 3, 
                     width_ratios=[1.4, 1.0, 0.4],  # Map, charts, stats+circle
                     height_ratios=[1.0, 1.0],
                     hspace=0.25, wspace=0.3,
                     left=0.04, right=0.96, top=0.94, bottom=0.06)

# Color scheme
colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']  # Green, Blue, Orange, Red

# ============================================================================
# LEFT SIDE: DISTANCE MAP WITH COLORED PROPERTIES (NO CIRCLE INSIDE)
# ============================================================================
ax_map = fig.add_subplot(gs[0:2, 0])

if has_map_data and parks_gdf is not None and parcels_filtered is not None:
    # Project to Web Mercator for basemap
    display_crs = 'EPSG:3857'
    parcels_plot = parcels_filtered.to_crs(display_crs)
    parks_plot = parks_gdf.to_crs(display_crs)
    
    # Get bounds from parcels
    bounds = parcels_plot.total_bounds
    ax_map.set_xlim(bounds[0], bounds[2])
    ax_map.set_ylim(bounds[1], bounds[3])
    
    # Add basemap
    try:
        ctx.add_basemap(
            ax_map,
            crs=display_crs,
            source=ctx.providers.OpenStreetMap.Mapnik,
            zoom='auto',
            attribution_size=8
        )
        print("   Basemap added successfully")
    except Exception as e:
        print(f"   Warning: Could not add basemap: {e}")
    
    # Plot parcels by distance band (with color coding)
    for band in band_order:
        band_parcels = parcels_plot[parcels_plot['DistBand'] == band]
        if len(band_parcels) > 0:
            band_parcels.plot(
                ax=ax_map,
                color=color_map[band],
                edgecolor='none',
                linewidth=0,
                alpha=0.65,  # Slightly higher alpha to show over basemap
                zorder=2
            )
    
    # Plot parks on top (with higher zorder to ensure visibility)
    parks_plot.plot(
        ax=ax_map,
        color='darkgreen',
        edgecolor='white',
        linewidth=2,
        markersize=250,
        marker='*',
        label='Parks',
        zorder=15  # Higher zorder to ensure parks are visible
    )
    
    # Add park labels
    for idx, row in parks_plot.iterrows():
        if 'park_name' in row:
            ax_map.annotate(
                row['park_name'],
                xy=(row.geometry.x, row.geometry.y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                fontweight='bold',
                color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='darkgreen'),
                zorder=16  # Highest zorder for labels
            )
    
    ax_map.set_axis_off()
    ax_map.set_title('Butler County Properties by Distance to Parks', 
                    fontsize=14, fontweight='bold', pad=10)
    
    # Add summary text on map
    map_text = f'Total Properties: {len(parcels_filtered):,} | Parks: {len(parks_gdf)}'
    ax_map.text(0.5, 0.02, map_text, 
                transform=ax_map.transAxes, 
                fontsize=10, 
                style='italic',
                ha='center',
                color='gray',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                zorder=20)
    
    
else:
    ax_map.text(0.5, 0.5, 'Map data not available',
               transform=ax_map.transAxes, ha='center', va='center', fontsize=12)

# ============================================================================
# RIGHT SIDE TOP: HOUSING BENEFITS BY DISTANCE BAND
# ============================================================================
ax1 = fig.add_subplot(gs[0, 1])

y_pos = np.arange(len(housing_df))
premiums = housing_df['pct_premium'].values
ci_lower = housing_df['pct_premium_lower'].values
ci_upper = housing_df['pct_premium_upper'].values

bar_width = 0.65

# Create bars using FancyBboxPatch
for i, (idx, row) in enumerate(housing_df.iterrows()):
    x = premiums[i]
    y = y_pos[i] - bar_width/2
    width = abs(x) if x != 0 else 0.01
    height = bar_width
    
    # Determine color based on sign
    if x > 0:
        color = colors[i % len(colors)]
        edge_color = '#1a7a1a' if i == 2 else '#0d4d0d'  # Darker green for positive
    else:
        color = '#d62728'
        edge_color = '#8b1a1a'
    
    # Create FancyBboxPatch
    if x >= 0:
        fancy_box = FancyBboxPatch(
            (0, y), width, height,
            boxstyle="round,pad=0.03",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2,
            alpha=0.85,
            zorder=2
        )
    else:
        fancy_box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.03",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2,
            alpha=0.85,
            zorder=2
        )
    
    ax1.add_patch(fancy_box)
    
    # Add value label
    label_x = x + (0.8 if x >= 0 else -0.8)
    ax1.text(label_x, y_pos[i], f'{x:.2f}%',
             ha='center',
             va='center',
             fontweight='bold',
             fontsize=11,
             color='white' if abs(x) > 2 else 'black',
             zorder=3)

# Add error bars
yerr_lower = premiums - ci_lower
yerr_upper = ci_upper - premiums
ax1.errorbar(premiums, y_pos,
            xerr=(yerr_lower, yerr_upper),
            fmt='none',
            ecolor='black',
            elinewidth=2,
            capsize=5,
            capthick=2,
            alpha=0.8,
            zorder=1)

# Add reference line
ax1.axvline(x=0, color='black', linestyle='-', linewidth=2.5, alpha=0.6, zorder=0)

# Customize
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"{band} miles" for band in housing_df['distance_band']], fontsize=12, fontweight='bold')
ax1.set_xlabel('Property Value Premium (%)', fontsize=13, fontweight='bold')
ax1.set_title('Housing Benefits by Distance Band\n(Property Value Premium Relative to 1.5-3 miles)', 
             fontsize=15, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, axis='x', zorder=0, linestyle='--')
ax1.set_xlim(min(premiums) * 1.4, max(premiums) * 1.4)

# Add significance indicators
for i, (idx, row) in enumerate(housing_df.iterrows()):
    pval = row['pvalue']
    if pd.notna(pval):
        if pval < 0.001:
            sig_text = '***'
        elif pval < 0.01:
            sig_text = '**'
        elif pval < 0.05:
            sig_text = '*'
        else:
            sig_text = ''
        
        if sig_text:
            x_pos = ci_upper[i] + (max(ci_upper) - min(ci_lower)) * 0.08
            ax1.text(x_pos, y_pos[i], sig_text,
                    va='center', fontsize=16, fontweight='bold', color='green')

# ============================================================================
# RIGHT SIDE BOTTOM: HEALTH BENEFITS BY DISTANCE BAND (ALL OUTCOMES)
# ============================================================================
ax2 = fig.add_subplot(gs[1, 1])

if len(health_band_df) > 0:
    # Get all available health outcomes
    health_outcomes_list = ['Mental Distress', 'Obesity', 'Physical Inactivity']
    available_outcomes = [outcome for outcome in health_outcomes_list 
                         if f'{outcome}_diff' in health_band_df.columns]
    
    if len(available_outcomes) > 0:
        health_bands = health_band_df['distance_band'].values
        n_bands = len(health_bands)
        n_outcomes = len(available_outcomes)
        
        # Create grouped bar chart
        bar_width_health = 0.25
        x = np.arange(n_bands)
        
        # Color scheme for outcomes
        outcome_colors = {
            'Mental Distress': '#2ca02c',      # Green
            'Obesity': '#1f77b4',              # Blue
            'Physical Inactivity': '#ff7f0e'    # Orange
        }
        
        # Plot bars for each outcome
        for i, outcome in enumerate(available_outcomes):
            outcome_diff = health_band_df[f'{outcome}_diff'].values
            # Flip sign for visualization (negative diff = benefit)
            benefits = -outcome_diff  # Negative prevalence diff = positive benefit
            
            # Calculate x positions for grouped bars
            offset = (i - (n_outcomes - 1) / 2) * bar_width_health
            
            for j, (band, benefit) in enumerate(zip(health_bands, benefits)):
                y = j
                x_pos = x[j] + offset
                width = bar_width_health
                height = abs(benefit) if benefit != 0 else 0.01
                
                # Color based on benefit (positive = good)
                if benefit > 0:
                    color = outcome_colors.get(outcome, '#2ca02c')
                    edge_color = '#1a7a1a' if outcome == 'Mental Distress' else '#0d4d0d'
                else:
                    color = '#d62728'  # Red
                    edge_color = '#8b1a1a'
                
                # Create FancyBboxPatch
                fancy_box = FancyBboxPatch(
                    (x_pos - width/2, y - height/2), width, height,
                    boxstyle="round,pad=0.02",
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=1.5,
                    alpha=0.85,
                    zorder=2
                )
                
                ax2.add_patch(fancy_box)
                
                # Add value label if significant
                if abs(benefit) > 0.3:
                    ax2.text(x_pos, y + height/2 + 0.15, f'{benefit:.2f}',
                            ha='center',
                            va='bottom',
                            fontweight='bold',
                            fontsize=8,
                            color='black',
                            zorder=3)
        
        # Add reference line
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.6, zorder=0)
        
        # Customize
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{band} miles" for band in health_bands], fontsize=11, fontweight='bold')
        ax2.set_ylabel('Prevalence Reduction (percentage points)', fontsize=13, fontweight='bold')
        ax2.set_title('Health Benefits by Distance Band\n(Reduction Relative to 1.5-3 miles)', 
                     fontsize=15, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='y', zorder=0, linestyle='--')
        
        # Set y-axis limits
        all_benefits = []
        for outcome in available_outcomes:
            benefits = -health_band_df[f'{outcome}_diff'].values
            all_benefits.extend(benefits)
        y_max = max(abs(b) for b in all_benefits) if all_benefits else 1
        ax2.set_ylim(-y_max * 0.2, y_max * 1.3)
        
        # Create legend
        legend_elements = []
        for outcome in available_outcomes:
            color = outcome_colors.get(outcome, '#2ca02c')
            legend_elements.append(
                mpatches.Patch(facecolor=color, label=outcome, alpha=0.85, edgecolor='black', linewidth=1.5)
            )
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # Add note
        ax2.text(0.5, -0.12, 'Note: Negative prevalence difference = Protective health benefit',
                transform=ax2.transAxes, ha='center', fontsize=9, style='italic', color='gray')
    else:
        # Fallback: show message if no health data available
        ax2.text(0.5, 0.5, 'Health benefits by distance band\n(Data processing required)',
                transform=ax2.transAxes, ha='center', va='center', fontsize=12, style='italic')
else:
    # Fallback: show message if health data not available
    ax2.text(0.5, 0.5, 'Health benefits by distance band\n(Data processing required)',
            transform=ax2.transAxes, ha='center', va='center', fontsize=12, style='italic')

# ============================================================================
# RIGHT SIDE: COUNTY-WIDE SUMMARY STATISTICS (NEXT TO CHARTS)
# ============================================================================
ax3 = fig.add_subplot(gs[0:2, 2])
ax3.axis('off')

# Remove summary statistics - user will add them separately
# Just show the gradient circle

# ============================================================================
# GRADIENT CIRCLE LEGEND (OUTSIDE MAP, ON THE RIGHT, ABOVE STATS)
# ============================================================================
# Create a separate axes for the legend in the stats area
ax_legend = ax3  # We'll overlay it on the stats axis

# Create gradient circle showing distance bands
# Use equal aspect ratio to make it circular
ax_legend.set_aspect('equal')
legend_size = 0.45  # Make circle bigger
center_x = 0.5
center_y = 0.5  # Center it vertically
num_bands = len(band_order)

# Draw circles from outside in (furthest to closest)
for i, band in enumerate(reversed(band_order)):
    if band in color_map:
        # Calculate radius (larger for closer bands)
        outer_radius = (i + 1) / num_bands * legend_size
        inner_radius = i / num_bands * legend_size if i > 0 else 0
        
        # Create circle/ring patch
        if inner_radius == 0:
            # Innermost circle
            circle = plt.Circle(
                (center_x, center_y),
                outer_radius,
                transform=ax_legend.transAxes,
                facecolor=color_map[band],
                alpha=0.8,
                edgecolor='white',
                linewidth=2,
                zorder=1
            )
            ax_legend.add_patch(circle)
        else:
            # Ring (annulus)
            from matplotlib.patches import Wedge
            wedge = Wedge(
                (center_x, center_y),
                outer_radius,
                0, 360,
                width=outer_radius - inner_radius,
                transform=ax_legend.transAxes,
                facecolor=color_map[band],
                alpha=0.8,
                edgecolor='white',
                linewidth=2,
                zorder=1
            )
            ax_legend.add_patch(wedge)
        
        # Add text label for each band (on the right side)
        if i < num_bands - 1:  # Don't label the outermost
            label_radius = (outer_radius + inner_radius) / 2 if inner_radius > 0 else outer_radius / 2
            label_x = center_x + outer_radius + 0.02
            label_y = center_y + (label_radius - legend_size/2) * 0.5
            
            ax_legend.text(
                label_x, label_y,
                band,
                transform=ax_legend.transAxes,
                fontsize=8,
                fontweight='bold',
                color='black',
                ha='left',
                va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor=color_map[band], linewidth=1.5),
                zorder=2
            )

# Remove title - just show the circle

# Add center label (furthest)
ax_legend.text(
    center_x, center_y,
    '>10 mi',
    transform=ax_legend.transAxes,
    fontsize=8,
    fontweight='bold',
    ha='center',
    va='center',
    color='black',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=1.5),
    zorder=4
)

# Now add the statistics text below the circle
ax3.axis('off')

# Footer statistics
footer_text = []
footer_text.append("DISTANCE BAND DETAILS")
footer_text.append("─" * 80)

for idx, row in housing_df.iterrows():
    band = row['distance_band']
    props = row['property_count']
    premium = row['pct_premium']
    uplift = row['total_value_uplift'] / 1e6
    footer_text.append(f"{band:12} miles: {props:>6,} properties | Premium: {premium:>7.2f}% | Uplift: ${uplift:>8.1f}M")

# Footer removed - user will add statistics separately

# Remove overall title as requested
# fig.suptitle('Butler County Parks: Comprehensive Benefits Dashboard', 
#             fontsize=18, fontweight='bold', y=0.97)

# ============================================================================
# 6. SAVE FIGURE
# ============================================================================
print(f"\n6. Saving dashboard...")
print(f"   Output path: {output_path}")

plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=300)
print(f"   Dashboard saved successfully!")

plt.close()

print(f"\n{'='*80}")
print("DASHBOARD CREATION COMPLETE")
print(f"{'='*80}")
