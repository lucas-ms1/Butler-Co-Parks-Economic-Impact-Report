"""
Create V2 Housing Benefits Bar Chart
Creates a FancyBboxPatch bar chart for housing property value benefits by distance band
using MetroParks boundary exposure with clustered SEs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import os
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Get paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
input_path = os.path.join(project_root, 'results', 'v2', 'housing_regression_clustered_boundary_v2.csv')
output_dir = os.path.join(project_root, 'figures', 'v2')
os.makedirs(output_dir, exist_ok=True)

print("Creating v2 housing benefits bar chart (MetroParks boundary)...")

# Load data
df = pd.read_csv(input_path)

# Filter to Metro distance bands bundled model
mask = (df['model'] == 'Metro_DistanceBands_Bundled') & df['variable'].str.contains('DistBand_Boundary_Metro_0to3')
df_bands = df[mask].copy()

# Extract band names from variable column
def extract_band(v):
    m = re.search(r'\[T\.(.+?)\]$', v)
    if m:
        return m.group(1)
    # Fallback: try splitting by colon
    if ':' in v:
        return v.split(':')[-1]
    return v

df_bands['band'] = df_bands['variable'].apply(extract_band)

# Sort by distance (band order)
band_order = ['0-0.1', '0.1-0.25', '0.25-0.75', '0.75-1.5', '1.5-3']
df_bands['band_order'] = df_bands['band'].apply(lambda x: band_order.index(x) if x in band_order else 999)
df_bands = df_bands.sort_values('band_order').copy()

print(f"   Distance bands: {df_bands['band'].tolist()}")
print(f"   Premiums: {df_bands['pct_premium'].tolist()}")

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create horizontal bar chart
y_pos = np.arange(len(df_bands))
premiums = df_bands['pct_premium'].values
ci_lower = df_bands['pct_premium_lower'].values
ci_upper = df_bands['pct_premium_upper'].values
pvalues = df_bands['pvalue'].values

# Color scheme - green for positive, red for negative
bar_width = 0.6

# Create bars using FancyBboxPatch
for i, (idx, row) in enumerate(df_bands.iterrows()):
    premium = premiums[i]
    y = y_pos[i] - bar_width/2
    width = abs(premium) if premium != 0 else 0.01
    height = bar_width
    
    # Determine color based on sign
    if premium > 0:
        color = '#2ca02c'  # Green for positive
        edge_color = '#1a7a1a'
    else:
        color = '#d62728'  # Red for negative
        edge_color = '#8b1a1a'
    
    # Create FancyBboxPatch
    if premium >= 0:
        # Positive bars extend right from zero
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
        # Negative bars extend left from zero
        fancy_box = FancyBboxPatch(
            (premium, y), width, height,
            boxstyle="round,pad=0.03",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2,
            alpha=0.85,
            zorder=2
        )
    
    ax.add_patch(fancy_box)
    
    # Add value label
    if premium >= 0:
        label_x = premium + 1.0  # Position to the right of positive bars
        ha = 'left'
    else:
        label_x = premium - 1.0  # Position to the left of negative bars
        ha = 'right'
    
    ax.text(label_x, y_pos[i], f'{premium:+.2f}%',
           ha=ha,
           va='center',
           fontweight='bold',
           fontsize=11,
           color='white' if abs(premium) > 15 else 'black',
           zorder=3)
    
    # Add p-value below the bar
    pvalue = pvalues[i]
    if pd.notna(pvalue):
        # Determine significance indicator
        if pvalue < 0.01:
            sig_text = '***'
            sig_color = '#2ca02c'  # Green
        elif pvalue < 0.05:
            sig_text = '**'
            sig_color = '#2ca02c'  # Green
        elif pvalue < 0.10:
            sig_text = '*'
            sig_color = '#ff7f0e'  # Orange
        else:
            sig_text = f'p={pvalue:.3f}'
            sig_color = '#d62728'  # Red for insignificant
        
        # Position p-value below the bar
        ax.text(premium, y_pos[i] - bar_width/2 - 0.2, sig_text,
               ha='center',
               va='top',
               fontsize=9,
               color=sig_color,
               fontweight='bold' if pvalue < 0.10 else 'normal',
               zorder=3)

# Add error bars for confidence intervals
yerr_lower = premiums - ci_lower
yerr_upper = ci_upper - premiums
ax.errorbar(premiums, y_pos,
           xerr=(yerr_lower, yerr_upper),
           fmt='none',
           ecolor='black',
           elinewidth=2,
           capsize=5,
           capthick=2,
           alpha=0.8,
           zorder=1)

# Add reference line at zero
ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.6, zorder=0)

# Customize
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{band} miles" for band in df_bands['band']], fontsize=12, fontweight='bold')
ax.set_xlabel('Property Value Premium (%)', fontsize=13, fontweight='bold')
ax.set_title('Housing Benefits by Distance Band (MetroParks Boundary)\n(Property Value Premium Relative to 1.5-3 miles, Clustered SE @ Tract)', 
            fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x', zorder=0, linestyle='--')

# Set y-axis limits to ensure all bars are visible
y_min = -0.5
y_max = len(df_bands) - 0.5 + bar_width
ax.set_ylim(y_min, y_max)

# Set x-axis limits with proper padding for labels
x_max = max(abs(p) for p in premiums) if len(premiums) > 0 else 1
x_min = min(premiums) if len(premiums) > 0 else -1
x_max_val = max(x_max, abs(x_min))
# Add extra padding for text labels (at least 5 units)
padding = max(5.0, x_max_val * 0.15)
ax.set_xlim(x_min - padding, x_max_val + padding)

# Add note
ax.text(0.5, -0.12, 'Note: Premiums are relative to the 1.5-3 mile reference band. Standard errors clustered at census tract.',
       transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='gray')

# Adjust layout with extra bottom margin for the note
plt.tight_layout(rect=[0, 0.05, 1, 0.98])

# Save figure
output_path = os.path.join(output_dir, 'housing_benefits_v2.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"  Created: housing_benefits_v2.png")
print(f"    Bands: {list(df_bands['band'])}")
print(f"    Premiums: {[f'{p:+.2f}%' for p in premiums]}")
