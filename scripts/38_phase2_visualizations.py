"""
Phase 2 Visualizations: Quality-Weighted Park Analysis & Economic Valuation
Economic Impact Report for Butler County Parks

This script creates visualizations for:
1. Park Quality Scores (Chen et al. 2023 method)
2. Regression coefficients with confidence intervals
3. Economic valuation by scenario
4. Interaction effects (Park Access × Poverty)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input paths
amenities_path = os.path.join(project_root, 'data_raw', 'park_amenities.csv')
regression_path = os.path.join(project_root, 'results', 'phase2_economic_valuation.csv')
valuation_path = os.path.join(project_root, 'results', 'phase2_economic_valuation_valuation.csv')

# Output paths
output_dir = os.path.join(project_root, 'figures')
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("PHASE 2 VISUALIZATIONS")
print("="*80)

# ============================================================================
# FIGURE 1: PARK QUALITY SCORES
# ============================================================================
print(f"\n1. Creating Park Quality Scores visualization...")

if os.path.exists(amenities_path):
    amenities_df = pd.read_csv(amenities_path)
    amenities_df['Chen_Quality_Score'] = (
        amenities_df['Amenity Count'] * amenities_df['Inclusiveness Score']
    )
    amenities_df = amenities_df.sort_values('Chen_Quality_Score', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create color gradient based on quality score
    colors = plt.cm.viridis((amenities_df['Chen_Quality_Score'] - amenities_df['Chen_Quality_Score'].min()) / 
                            (amenities_df['Chen_Quality_Score'].max() - amenities_df['Chen_Quality_Score'].min()))
    
    bars = ax.barh(amenities_df['Park Name'], amenities_df['Chen_Quality_Score'], color=colors)
    
    # Add value labels
    for i, (idx, row) in enumerate(amenities_df.iterrows()):
        ax.text(row['Chen_Quality_Score'] + 0.5, i, 
               f"{int(row['Chen_Quality_Score'])}", 
               va='center', fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Chen Quality Score\n(Amenity Count × Inclusiveness Score)', 
                 fontsize=12, fontweight='bold')
    ax.set_title('Park Quality Scores\n(Chen et al. 2023 Method)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, amenities_df['Chen_Quality_Score'].max() * 1.15)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add note
    note_text = f"Mean Score: {amenities_df['Chen_Quality_Score'].mean():.1f} | Range: {amenities_df['Chen_Quality_Score'].min():.0f}-{amenities_df['Chen_Quality_Score'].max():.0f}"
    ax.text(0.5, -0.08, note_text, transform=ax.transAxes, 
           fontsize=9, style='italic', ha='center', color='gray')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'phase2_park_quality_scores.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {output_path}")
    plt.close()

# ============================================================================
# FIGURE 2: REGRESSION COEFFICIENTS WITH CONFIDENCE INTERVALS
# ============================================================================
print(f"\n2. Creating regression coefficients visualization...")

if os.path.exists(regression_path):
    reg_df = pd.read_csv(regression_path)
    
    # Filter to protective effects and significant results
    protective_df = reg_df[reg_df['is_protective'] == True].copy()
    
    if len(protective_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        y_pos = np.arange(len(protective_df))
        coefficients = protective_df['coefficient'].values
        ci_lower = protective_df['ci_lower'].values
        ci_upper = protective_df['ci_upper'].values
        
        # Create labels
        labels = []
        for idx, row in protective_df.iterrows():
            outcome_short = row['outcome'].replace('Frequent Mental Distress', 'Mental Distress')
            var_short = row['park_variable'].replace('Quality-Weighted Walkshed', 'Quality Walkshed')
            var_short = var_short.replace('Interaction: ', '')
            labels.append(f"{outcome_short}\n{var_short}")
        
        # Color by significance
        colors = ['#2ca02c' if p < 0.01 else '#ff7f0e' if p < 0.05 else '#1f77b4' 
                 for p in protective_df['pvalue']]
        
        # Plot error bars
        yerr_lower = coefficients - ci_lower
        yerr_upper = ci_upper - coefficients
        
        ax.errorbar(coefficients, y_pos, 
                   xerr=(yerr_lower, yerr_upper),
                   fmt='o', capsize=5, capthick=2, markersize=10,
                   color='steelblue', ecolor='steelblue', elinewidth=2)
        
        # Add reference line at zero
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='No Effect')
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Coefficient (Protective Effect)', fontsize=12, fontweight='bold')
        ax.set_title('Park Access Health Effects\n(Density-Corrected & Interaction Models)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add significance indicators
        for i, (idx, row) in enumerate(protective_df.iterrows()):
            pval = row['pvalue']
            if pval < 0.001:
                sig_text = '***'
            elif pval < 0.01:
                sig_text = '**'
            elif pval < 0.05:
                sig_text = '*'
            else:
                sig_text = ''
            
            if sig_text:
                ax.text(row['coefficient'] + (ci_upper[i] - coefficients[i]) * 1.1, i,
                       sig_text, va='center', fontsize=12, fontweight='bold', color='green')
        
        # Add legend for significance
        legend_elements = [
            mpatches.Patch(facecolor='green', label='p < 0.001 (***)'),
            mpatches.Patch(facecolor='orange', label='p < 0.01 (**)'),
            mpatches.Patch(facecolor='blue', label='p < 0.05 (*)'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='No Effect')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'phase2_regression_coefficients.png')
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        print(f"   Saved: {output_path}")
        plt.close()

# ============================================================================
# FIGURE 3: ECONOMIC VALUATION BY SCENARIO
# ============================================================================
print(f"\n3. Creating economic valuation visualization...")

if os.path.exists(valuation_path):
    val_df = pd.read_csv(valuation_path)
    
    # Aggregate by scenario
    scenario_totals = val_df.groupby('scenario')['annual_savings'].sum().sort_values()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Total savings by scenario
    scenarios = scenario_totals.index.tolist()
    scenario_labels = [s.capitalize() for s in scenarios]
    colors_scenario = ['#2ca02c', '#ff7f0e', '#1f77b4']
    
    bars1 = ax1.bar(scenario_labels, scenario_totals.values, color=colors_scenario, alpha=0.8)
    
    # Add value labels
    for i, (scenario, total) in enumerate(scenario_totals.items()):
        ax1.text(i, total + max(scenario_totals) * 0.02, 
                f'${total:,.0f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.set_ylabel('Annual Health Cost Savings (USD)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Economic Impact by Scenario', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(scenario_totals) * 1.15)
    
    # Subplot 2: Breakdown by outcome and scenario
    pivot_df = val_df.pivot_table(
        values='annual_savings', 
        index='outcome', 
        columns='scenario', 
        aggfunc='sum'
    )
    
    x = np.arange(len(pivot_df.index))
    width = 0.25
    
    for i, scenario in enumerate(['conservative', 'moderate', 'aggressive']):
        if scenario in pivot_df.columns:
            offset = (i - 1) * width
            ax2.bar(x + offset, pivot_df[scenario].values, width, 
                   label=scenario.capitalize(), color=colors_scenario[i], alpha=0.8)
    
    ax2.set_xlabel('Health Outcome', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Annual Savings (USD)', fontsize=12, fontweight='bold')
    ax2.set_title('Savings by Outcome and Scenario', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([o.replace('Frequent Mental Distress', 'Mental\nDistress') 
                         for o in pivot_df.index], fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'phase2_economic_valuation.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {output_path}")
    plt.close()

# ============================================================================
# FIGURE 4: INTERACTION EFFECTS DETAILED VIEW
# ============================================================================
print(f"\n4. Creating interaction effects visualization...")

if os.path.exists(regression_path):
    reg_df = pd.read_csv(regression_path)
    interaction_df = reg_df[reg_df['model_type'] == 'interaction'].copy()
    
    if len(interaction_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(interaction_df))
        coefficients = interaction_df['coefficient'].values
        ci_lower = interaction_df['ci_lower'].values
        ci_upper = interaction_df['ci_upper'].values
        
        # Create labels
        labels = []
        for idx, row in interaction_df.iterrows():
            outcome_short = row['outcome'].replace('Frequent Mental Distress', 'Mental Distress')
            var_short = row['park_variable'].replace('Interaction: ', '').replace('_x_poverty', ' × Poverty')
            var_short = var_short.replace('park_acres_10min', 'Park Acres')
            var_short = var_short.replace('quality_walkshed', 'Quality Walkshed')
            labels.append(f"{outcome_short}\n{var_short}")
        
        # Plot
        yerr_lower = coefficients - ci_lower
        yerr_upper = ci_upper - coefficients
        
        ax.errorbar(coefficients, y_pos, 
                   xerr=(yerr_lower, yerr_upper),
                   fmt='o', capsize=6, capthick=2.5, markersize=12,
                   color='#2ca02c', ecolor='#2ca02c', elinewidth=2.5)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Interaction Coefficient\n(Negative = Protective Effect in High-Poverty Areas)', 
                     fontsize=12, fontweight='bold')
        ax.set_title('Park Access × Poverty Interaction Effects\n(Benefits Strongest in Low-Income Neighborhoods)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add significance stars
        for i, (idx, row) in enumerate(interaction_df.iterrows()):
            pval = row['pvalue']
            if pval < 0.001:
                sig_text = '***'
            elif pval < 0.01:
                sig_text = '**'
            elif pval < 0.05:
                sig_text = '*'
            else:
                sig_text = ''
            
            if sig_text:
                ax.text(row['coefficient'] + (ci_upper[i] - coefficients[i]) * 1.15, i,
                       sig_text, va='center', fontsize=14, fontweight='bold', color='green')
        
        # Add note
        note_text = "Interaction models reveal protective effects that are hidden in global models"
        ax.text(0.5, -0.12, note_text, transform=ax.transAxes, 
               fontsize=9, style='italic', ha='center', color='gray')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'phase2_interaction_effects.png')
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        print(f"   Saved: {output_path}")
        plt.close()

# ============================================================================
# FIGURE 5: COMPREHENSIVE DASHBOARD
# ============================================================================
print(f"\n5. Creating comprehensive dashboard...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Load data
if os.path.exists(amenities_path):
    amenities_df = pd.read_csv(amenities_path)
    amenities_df['Chen_Quality_Score'] = (
        amenities_df['Amenity Count'] * amenities_df['Inclusiveness Score']
    )
    amenities_df = amenities_df.sort_values('Chen_Quality_Score', ascending=False).head(6)

if os.path.exists(valuation_path):
    val_df = pd.read_csv(valuation_path)
    scenario_totals = val_df.groupby('scenario')['annual_savings'].sum()

if os.path.exists(regression_path):
    reg_df = pd.read_csv(regression_path)
    protective_df = reg_df[reg_df['is_protective'] == True].copy()
    interaction_df = reg_df[reg_df['model_type'] == 'interaction'].copy()

# Panel 1: Top Parks by Quality (top-left)
ax1 = fig.add_subplot(gs[0, 0])
if 'amenities_df' in locals() and len(amenities_df) > 0:
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(amenities_df)))
    ax1.barh(range(len(amenities_df)), amenities_df['Chen_Quality_Score'].values, color=colors)
    ax1.set_yticks(range(len(amenities_df)))
    ax1.set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                         for name in amenities_df['Park Name']], fontsize=8)
    ax1.set_xlabel('Quality Score', fontsize=9, fontweight='bold')
    ax1.set_title('Top Parks by Quality', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

# Panel 2: Economic Impact Summary (top-center)
ax2 = fig.add_subplot(gs[0, 1:])
if 'scenario_totals' in locals() and len(scenario_totals) > 0:
    scenarios = scenario_totals.index.tolist()
    scenario_labels = [s.capitalize() + f'\n({s.capitalize()} Scenario)' for s in scenarios]
    colors_scenario = ['#2ca02c', '#ff7f0e', '#1f77b4']
    
    bars = ax2.bar(scenario_labels, scenario_totals.values, color=colors_scenario, alpha=0.8)
    for i, (scenario, total) in enumerate(scenario_totals.items()):
        ax2.text(i, total + max(scenario_totals) * 0.03, 
                f'${total:,.0f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Annual Savings (USD)', fontsize=10, fontweight='bold')
    ax2.set_title('Economic Impact: Annual Health Cost Savings', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(scenario_totals) * 1.2)

# Panel 3: Protective Effects (middle-left, spans 2 rows)
ax3 = fig.add_subplot(gs[1:, 0])
if 'protective_df' in locals() and len(protective_df) > 0:
    y_pos = np.arange(len(protective_df))
    coefficients = protective_df['coefficient'].values
    ci_lower = protective_df['ci_lower'].values
    ci_upper = protective_df['ci_upper'].values
    
    labels = []
    for idx, row in protective_df.iterrows():
        outcome_short = row['outcome'].replace('Frequent Mental Distress', 'Mental Distress')
        var_short = row['park_variable'].replace('Quality-Weighted Walkshed', 'Quality')
        var_short = var_short.replace('Interaction: ', '')
        labels.append(f"{outcome_short[:12]}\n{var_short[:15]}")
    
    yerr_lower = coefficients - ci_lower
    yerr_upper = ci_upper - coefficients
    
    ax3.errorbar(coefficients, y_pos, 
                xerr=(yerr_lower, yerr_upper),
                fmt='o', capsize=4, capthick=2, markersize=8,
                color='steelblue', ecolor='steelblue', elinewidth=2)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.set_xlabel('Coefficient', fontsize=9, fontweight='bold')
    ax3.set_title('Protective Effects\n(Density-Corrected)', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

# Panel 4: Interaction Effects (middle-right)
ax4 = fig.add_subplot(gs[1, 1:])
if 'interaction_df' in locals() and len(interaction_df) > 0:
    y_pos = np.arange(len(interaction_df))
    coefficients = interaction_df['coefficient'].values
    ci_lower = interaction_df['ci_lower'].values
    ci_upper = interaction_df['ci_upper'].values
    
    labels = []
    for idx, row in interaction_df.iterrows():
        outcome_short = row['outcome'].replace('Frequent Mental Distress', 'Mental Distress')
        labels.append(outcome_short)
    
    yerr_lower = coefficients - ci_lower
    yerr_upper = ci_upper - coefficients
    
    ax4.errorbar(coefficients, y_pos, 
                xerr=(yerr_lower, yerr_upper),
                fmt='s', capsize=5, capthick=2.5, markersize=10,
                color='#2ca02c', ecolor='#2ca02c', elinewidth=2.5)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels, fontsize=9)
    ax4.set_xlabel('Interaction Coefficient', fontsize=9, fontweight='bold')
    ax4.set_title('Park × Poverty Interactions\n(Stronger Benefits in Low-Income Areas)', 
                 fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

# Panel 5: Savings Breakdown (bottom-right)
ax5 = fig.add_subplot(gs[2, 1:])
if 'val_df' in locals() and len(val_df) > 0:
    # Aggregate by outcome for moderate scenario
    moderate_df = val_df[val_df['scenario'] == 'moderate'].copy()
    if len(moderate_df) > 0:
        outcome_savings = moderate_df.groupby('outcome')['annual_savings'].sum()
        
        outcome_labels = [o.replace('Frequent Mental Distress', 'Mental\nDistress') 
                         for o in outcome_savings.index]
        colors_outcome = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        wedges, texts, autotexts = ax5.pie(outcome_savings.values, 
                                           labels=outcome_labels,
                                           autopct='%1.1f%%',
                                           colors=colors_outcome,
                                           startangle=90,
                                           textprops={'fontsize': 9, 'fontweight': 'bold'})
        
        ax5.set_title('Savings Breakdown\n(Moderate Scenario)', fontsize=10, fontweight='bold')

# Overall title
fig.suptitle('Phase 2: Quality-Weighted Park Analysis & Economic Valuation', 
            fontsize=16, fontweight='bold', y=0.98)

# Add summary text
summary_text = (
    "Key Finding: Park benefits are strongest in low-income neighborhoods.\n"
    "Interaction models reveal protective effects hidden in global models."
)
fig.text(0.5, 0.02, summary_text, ha='center', fontsize=9, style='italic', color='gray')

output_path = os.path.join(output_dir, 'phase2_comprehensive_dashboard.png')
plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=300)
print(f"   Saved: {output_path}")
plt.close()

print(f"\n{'='*80}")
print("ALL VISUALIZATIONS COMPLETE")
print(f"{'='*80}")
print(f"\nFigures saved to: {output_dir}")
print(f"  - phase2_park_quality_scores.png")
print(f"  - phase2_regression_coefficients.png")
print(f"  - phase2_economic_valuation.png")
print(f"  - phase2_interaction_effects.png")
print(f"  - phase2_comprehensive_dashboard.png")
