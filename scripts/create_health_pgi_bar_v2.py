"""
Create V2 Health PGI Coefficients Bar Chart
Creates a FancyBboxPatch bar chart for health PGI regression coefficients
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Get paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
input_path = os.path.join(project_root, 'results', 'v2', 'health_pgi_regression_v2.csv')
output_dir = os.path.join(project_root, 'figures', 'v2')
os.makedirs(output_dir, exist_ok=True)

print("Creating v2 health PGI coefficients bar chart...")

# Load data
df = pd.read_csv(input_path)

# Filter to WLS (population-weighted) as primary
df_wls = df[df['model_type'] == 'WLS_pop_HC1'].copy()

# Sort by coefficient value for better visualization
df_wls = df_wls.sort_values('beta_pp_per_1SD_PGI').copy()

print(f"   Outcomes: {df_wls['outcome'].tolist()}")
print(f"   Coefficients: {df_wls['beta_pp_per_1SD_PGI'].tolist()}")

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create horizontal bar chart
y_pos = np.arange(len(df_wls))
coefs = df_wls['beta_pp_per_1SD_PGI'].values
ci_lower = df_wls['ci_lower'].values
ci_upper = df_wls['ci_upper'].values
pvalues = df_wls['pvalue'].values

# Color scheme - red for positive (worse), green for negative (better)
bar_width = 0.6

# Create bars using FancyBboxPatch
for i, (idx, row) in enumerate(df_wls.iterrows()):
    coef = coefs[i]
    y = y_pos[i] - bar_width/2
    width = abs(coef) if coef != 0 else 0.01
    height = bar_width
    
    # Determine color based on sign (positive = worse = red, negative = better = green)
    if coef > 0:
        color = '#d62728'  # Red for positive (worse outcomes)
        edge_color = '#8b1a1a'
    else:
        color = '#2ca02c'  # Green for negative (better outcomes)
        edge_color = '#1a7a1a'
    
    # Create FancyBboxPatch
    if coef >= 0:
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
            (coef, y), width, height,
            boxstyle="round,pad=0.03",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2,
            alpha=0.85,
            zorder=2
        )
    
    ax.add_patch(fancy_box)
    
    # Add value label
    if coef >= 0:
        label_x = coef + 0.05  # Position to the right of positive bars
        ha = 'left'
    else:
        label_x = coef - 0.05  # Position to the left of negative bars
        ha = 'right'
    
    ax.text(label_x, y_pos[i], f'{coef:+.3f}pp',
           ha=ha,
           va='center',
           fontweight='bold',
           fontsize=11,
           color='white' if abs(coef) > 0.15 else 'black',
           zorder=3)
    
    # Add p-value next to the bar
    pvalue = pvalues[i]
    if pd.notna(pvalue):
        # Determine significance indicator
        if pvalue < 0.01:
            sig_text = '***'
            sig_color = '#2ca02c'  # Green for significant
        elif pvalue < 0.05:
            sig_text = '**'
            sig_color = '#2ca02c'  # Green
        elif pvalue < 0.10:
            sig_text = '*'
            sig_color = '#ff7f0e'  # Orange
        else:
            sig_text = f'ns (p={pvalue:.3f})'
            sig_color = '#666666'  # Gray for insignificant
        
        # Position p-value next to the end of the bar
        if coef >= 0:
            pvalue_x = coef + 0.1  # Position to the right of positive bars
            pvalue_ha = 'left'
        else:
            pvalue_x = coef - 0.1  # Position to the left of negative bars
            pvalue_ha = 'right'
        
        ax.text(pvalue_x, y_pos[i], sig_text,
               ha=pvalue_ha,
               va='center',
               fontsize=9,
               color=sig_color,
               fontweight='bold' if pvalue < 0.10 else 'normal',
               style='italic' if pvalue >= 0.10 else 'normal',
               zorder=3)

# Add error bars for confidence intervals
yerr_lower = coefs - ci_lower
yerr_upper = ci_upper - coefs
ax.errorbar(coefs, y_pos,
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
ax.set_yticklabels(df_wls['outcome'].values, fontsize=12, fontweight='bold')
ax.set_xlabel('β (pp prevalence per +1 SD PGI)', fontsize=13, fontweight='bold')
ax.set_title('Health Associations vs Park Gravity Index (PGI)\nPositive β = higher prevalence (worse). WLS (population-weighted), HC1 SEs.', 
            fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x', zorder=0, linestyle='--')

# Set y-axis limits to ensure all bars are visible
y_min = -0.5
y_max = len(df_wls) - 0.5 + bar_width
ax.set_ylim(y_min, y_max)

# Set x-axis limits with proper padding for labels
x_max = max(abs(c) for c in coefs) if len(coefs) > 0 else 1
x_min = min(coefs) if len(coefs) > 0 else -1
x_max_val = max(x_max, abs(x_min))
# Add extra padding for text labels
padding = max(0.15, x_max_val * 0.2)
ax.set_xlim(x_min - padding, x_max_val + padding)

# Add note
ax.text(0.5, -0.12, 'Note: Coefficients are percentage-point changes in prevalence per +1 SD increase in PGI. Positive values indicate worse outcomes.',
       transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='gray')

# Adjust layout with extra bottom margin for the note
plt.tight_layout(rect=[0, 0.05, 1, 0.98])

# Save figure
output_path = os.path.join(output_dir, 'health_pgi_coefficients_v2.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"  Created: health_pgi_coefficients_v2.png")
print(f"    Outcomes: {list(df_wls['outcome'])}")
print(f"    Coefficients: {[f'{c:+.3f}pp' for c in coefs]}")
