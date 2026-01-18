"""
Distance Band Benefits Visualization
Economic Impact Report for Butler County Parks

Creates a visualization of property value benefits by distance band,
excluding the 0-0.1 mile band and using FancyBboxPatch for styled bars.
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
plt.rcParams['font.size'] = 11

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input path
input_path = os.path.join(project_root, 'results', 'distance_band_benefits_summary.csv')

# Output path
output_dir = os.path.join(project_root, 'figures')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'distance_band_benefits.png')

print("="*80)
print("DISTANCE BAND BENEFITS VISUALIZATION")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print(f"\n1. Loading data...")
df = pd.read_csv(input_path)

# Exclude 0-0.1 mile band
df = df[df['distance_band'] != '0-0.1'].copy()
df = df.sort_values('mean_distance')

print(f"   Distance bands to visualize: {df['distance_band'].tolist()}")

# ============================================================================
# 2. CREATE VISUALIZATION
# ============================================================================
print(f"\n2. Creating visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Color scheme
colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']  # Green, Blue, Orange, Red
if len(df) > len(colors):
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))

# ============================================================================
# SUBPLOT 1: Premium Percentage by Distance Band
# ============================================================================
y_pos = np.arange(len(df))
premiums = df['pct_premium'].values
ci_lower = df['pct_premium_lower'].values
ci_upper = df['pct_premium_upper'].values

# Create bars using FancyBboxPatch
bar_width = 0.6
for i, (idx, row) in enumerate(df.iterrows()):
    x = premiums[i]
    y = y_pos[i] - bar_width/2
    width = abs(x) if x != 0 else 0.01
    height = bar_width
    
    # Determine color based on sign
    if x > 0:
        color = colors[i] if i < len(colors) else '#2ca02c'
        edge_color = '#1a7a1a'
    else:
        color = '#d62728'
        edge_color = '#8b1a1a'
    
    # Create FancyBboxPatch
    if x >= 0:
        # Positive bars extend right from zero
        fancy_box = FancyBboxPatch(
            (0, y), width, height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=1.5,
            alpha=0.8,
            zorder=2
        )
    else:
        # Negative bars extend left from zero
        fancy_box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=1.5,
            alpha=0.8,
            zorder=2
        )
    
    ax1.add_patch(fancy_box)
    
    # Add value label
    label_x = x + (0.5 if x >= 0 else -0.5)
    ax1.text(label_x, y_pos[i], f'{x:.2f}%',
             ha='center' if x >= 0 else 'center',
             va='center',
             fontweight='bold',
             fontsize=10,
             color='white' if abs(x) > 2 else 'black',
             zorder=3)

# Add error bars
yerr_lower = premiums - ci_lower
yerr_upper = ci_upper - premiums
ax1.errorbar(premiums, y_pos,
            xerr=(yerr_lower, yerr_upper),
            fmt='none',
            ecolor='black',
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
            alpha=0.7,
            zorder=1)

# Add reference line at zero
ax1.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.5, zorder=0)

# Customize
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"{band} miles" for band in df['distance_band']], fontsize=11)
ax1.set_xlabel('Property Value Premium (%)', fontsize=12, fontweight='bold')
ax1.set_title('Property Value Premium by Distance Band\n(Relative to 1.5-3 miles)', 
             fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, axis='x', zorder=0)
ax1.set_xlim(min(premiums) * 1.3, max(premiums) * 1.3)

# Add significance indicators
for i, (idx, row) in enumerate(df.iterrows()):
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
            x_pos = ci_upper[i] + (max(ci_upper) - min(ci_lower)) * 0.05
            ax1.text(x_pos, y_pos[i], sig_text,
                    va='center', fontsize=14, fontweight='bold', color='green')

# ============================================================================
# SUBPLOT 2: Total Value Uplift by Distance Band
# ============================================================================
uplift = df['total_value_uplift'].values / 1e6  # Convert to millions

# Create bars using FancyBboxPatch
for i, (idx, row) in enumerate(df.iterrows()):
    x = 0
    y = y_pos[i] - bar_width/2
    width = uplift[i]
    height = bar_width
    
    # Determine color based on sign
    if uplift[i] > 0:
        color = colors[i] if i < len(colors) else '#2ca02c'
        edge_color = '#1a7a1a'
    else:
        color = '#d62728'
        edge_color = '#8b1a1a'
    
    # Create FancyBboxPatch
    fancy_box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor=edge_color,
        linewidth=1.5,
        alpha=0.8,
        zorder=2
    )
    
    ax2.add_patch(fancy_box)
    
    # Add value label
    label_x = width / 2 if width > 0 else width / 2
    ax2.text(label_x, y_pos[i], f'${width:.1f}M',
             ha='center',
             va='center',
             fontweight='bold',
             fontsize=10,
             color='white' if abs(width) > 10 else 'black',
             zorder=3)

# Customize
ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"{band} miles" for band in df['distance_band']], fontsize=11)
ax2.set_xlabel('Total Value Uplift (Millions USD)', fontsize=12, fontweight='bold')
ax2.set_title('Total Economic Benefit by Distance Band', 
             fontsize=14, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, axis='x', zorder=0)
ax2.set_xlim(0, max(uplift) * 1.15)

# Add reference line at zero
ax2.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.5, zorder=0)

# ============================================================================
# 3. ADD SUMMARY TEXT
# ============================================================================
# Calculate summary statistics
total_uplift = df['total_value_uplift'].sum() / 1e6
total_properties = df['property_count'].sum()
avg_benefit = df['avg_benefit_per_property'].mean()

summary_text = (
    f"Total Value Uplift: ${total_uplift:.1f}M | "
    f"Properties: {total_properties:,} | "
    f"Avg Benefit: ${avg_benefit:,.0f}/property"
)

fig.text(0.5, 0.02, summary_text, 
        ha='center', fontsize=10, style='italic', color='gray',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

# Overall title
fig.suptitle('Property Value Benefits by Distance to Parks\n(Bundled Community Value Model)', 
            fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08)

# ============================================================================
# 4. SAVE FIGURE
# ============================================================================
print(f"\n3. Saving figure...")
print(f"   Output path: {output_path}")

plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=300)
print(f"   Figure saved successfully!")

plt.close()

print(f"\n{'='*80}")
print("VISUALIZATION COMPLETE")
print(f"{'='*80}")
