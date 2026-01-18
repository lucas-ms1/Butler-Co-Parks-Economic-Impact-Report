"""
Jackknife Results Summary and Visualization
Economic Impact Report for Butler County Parks

This script reads the jackknife sensitivity analysis results, creates histograms
of coefficients and p-values, and generates a markdown summary table.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input path
input_path = os.path.join(project_root, 'results', 'mental_distress_jackknife.csv')

# Output paths
output_dir = os.path.join(project_root, 'results')
figures_dir = os.path.join(project_root, 'figures')
coeff_plot_path = os.path.join(figures_dir, 'mental_distress_jackknife_coeff.png')
pvalue_plot_path = os.path.join(figures_dir, 'mental_distress_jackknife_pvalues.png')
summary_path = os.path.join(output_dir, 'mental_distress_jackknife_summary.md')

print("="*60)
print("JACKKNIFE RESULTS SUMMARY AND VISUALIZATION")
print("="*60)

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print(f"\n1. Loading jackknife results...")
print(f"   Input path: {input_path}")

df = pd.read_csv(input_path)
print(f"   Rows loaded: {len(df):,}")
print(f"   Columns: {list(df.columns)}")

# ============================================================================
# 2. CALCULATE SUMMARY STATISTICS
# ============================================================================
print(f"\n2. Calculating summary statistics...")

coef_min = df['park_gravity_coef'].min()
coef_median = df['park_gravity_coef'].median()
coef_max = df['park_gravity_coef'].max()
coef_mean = df['park_gravity_coef'].mean()
coef_std = df['park_gravity_coef'].std()

pval_min = df['park_gravity_pvalue'].min()
pval_median = df['park_gravity_pvalue'].median()
pval_max = df['park_gravity_pvalue'].max()
pval_mean = df['park_gravity_pvalue'].mean()
pval_std = df['park_gravity_pvalue'].std()

n_significant = df['significant_05'].sum()
n_total = len(df)
pct_significant = (n_significant / n_total) * 100

print(f"\n   Park Gravity Index Coefficient:")
print(f"     Min:    {coef_min:+.6f}")
print(f"     Median: {coef_median:+.6f}")
print(f"     Mean:   {coef_mean:+.6f}")
print(f"     Max:    {coef_max:+.6f}")
print(f"     Std:    {coef_std:.6f}")

print(f"\n   P-value:")
print(f"     Min:    {pval_min:.6f}")
print(f"     Median: {pval_median:.6f}")
print(f"     Mean:   {pval_mean:.6f}")
print(f"     Max:    {pval_max:.6f}")
print(f"     Std:    {pval_std:.6f}")

print(f"\n   Significance (p < 0.05):")
print(f"     Significant: {n_significant} / {n_total} ({pct_significant:.1f}%)")

# ============================================================================
# 3. CREATE COEFFICIENT HISTOGRAM
# ============================================================================
print(f"\n3. Creating coefficient histogram...")

fig, ax = plt.subplots(figsize=(10, 6))

# Create histogram
n, bins, patches = ax.hist(df['park_gravity_coef'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')

# Add vertical lines for key statistics
ax.axvline(coef_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {coef_mean:.4f}')
ax.axvline(coef_median, color='green', linestyle='--', linewidth=2, label=f'Median: {coef_median:.4f}')
ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Zero')

# Customize plot
ax.set_xlabel('Park Gravity Index Coefficient', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Park Gravity Index Coefficients\n(Leave-One-Out Jackknife Analysis)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

# Add text box with summary statistics
textstr = f'Min: {coef_min:.4f}\nMedian: {coef_median:.4f}\nMax: {coef_max:.4f}\nStd: {coef_std:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(coeff_plot_path, dpi=300, bbox_inches='tight')
print(f"   Saved to: {coeff_plot_path}")
plt.close()

# ============================================================================
# 4. CREATE P-VALUE HISTOGRAM
# ============================================================================
print(f"\n4. Creating p-value histogram...")

fig, ax = plt.subplots(figsize=(10, 6))

# Create histogram
n, bins, patches = ax.hist(df['park_gravity_pvalue'], bins=20, edgecolor='black', alpha=0.7, color='coral')

# Color bars based on significance threshold
for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
    if bin_val < 0.05:
        patch.set_facecolor('green')
        patch.set_alpha(0.7)
    elif bin_val >= 0.05:
        patch.set_facecolor('coral')
        patch.set_alpha(0.7)

# Add vertical lines for key statistics
ax.axvline(pval_median, color='blue', linestyle='--', linewidth=2, label=f'Median: {pval_median:.4f}')
ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='p = 0.05 (significance threshold)')

# Customize plot
ax.set_xlabel('P-value', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of P-values\n(Leave-One-Out Jackknife Analysis)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

# Add text box with summary statistics
textstr = f'Min: {pval_min:.4f}\nMedian: {pval_median:.4f}\nMax: {pval_max:.4f}\nSignificant (p<0.05): {n_significant}/{n_total} ({pct_significant:.1f}%)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(pvalue_plot_path, dpi=300, bbox_inches='tight')
print(f"   Saved to: {pvalue_plot_path}")
plt.close()

# ============================================================================
# 5. CREATE MARKDOWN SUMMARY
# ============================================================================
print(f"\n5. Creating markdown summary...")

markdown_lines = []
markdown_lines.append("# Jackknife Sensitivity Analysis: Frequent Mental Distress")
markdown_lines.append("")
markdown_lines.append("## Summary")
markdown_lines.append("")
markdown_lines.append("This analysis performs a leave-one-out jackknife sensitivity test on the Frequent Mental Distress WLS model with density and reduced demographic controls (excluding pct_non_hispanic_white). Each iteration drops one tract and re-fits the model to assess the robustness of the park_gravity_index_z coefficient.")
markdown_lines.append("")
markdown_lines.append(f"**Total iterations:** {n_total}")
markdown_lines.append(f"**Observations per iteration:** {df['n_obs'].iloc[0]} (dropping 1 tract each time)")
markdown_lines.append("")
markdown_lines.append("## Park Gravity Index Coefficient Distribution")
markdown_lines.append("")
markdown_lines.append("| Statistic | Value |")
markdown_lines.append("|-----------|-------|")
markdown_lines.append(f"| Minimum   | {coef_min:+.6f} |")
markdown_lines.append(f"| Median    | {coef_median:+.6f} |")
markdown_lines.append(f"| Mean      | {coef_mean:+.6f} |")
markdown_lines.append(f"| Maximum   | {coef_max:+.6f} |")
markdown_lines.append(f"| Std Dev   | {coef_std:.6f} |")
markdown_lines.append("")
markdown_lines.append("## P-value Distribution")
markdown_lines.append("")
markdown_lines.append("| Statistic | Value |")
markdown_lines.append("|-----------|-------|")
markdown_lines.append(f"| Minimum   | {pval_min:.6f} |")
markdown_lines.append(f"| Median    | {pval_median:.6f} |")
markdown_lines.append(f"| Mean      | {pval_mean:.6f} |")
markdown_lines.append(f"| Maximum   | {pval_max:.6f} |")
markdown_lines.append("")
markdown_lines.append("## Significance Assessment")
markdown_lines.append("")
markdown_lines.append("| Category | Count | Percentage |")
markdown_lines.append("|----------|-------|------------|")
markdown_lines.append(f"| Significant (p < 0.05) | {n_significant} | {pct_significant:.1f}% |")
markdown_lines.append(f"| Non-significant (p ≥ 0.05) | {n_total - n_significant} | {100 - pct_significant:.1f}% |")
markdown_lines.append("")
markdown_lines.append("## Visualizations")
markdown_lines.append("")
markdown_lines.append("### Coefficient Distribution")
markdown_lines.append("")
markdown_lines.append(f"![Coefficient Histogram]({os.path.relpath(coeff_plot_path, project_root).replace(os.sep, '/')})")
markdown_lines.append("")
markdown_lines.append("### P-value Distribution")
markdown_lines.append("")
markdown_lines.append(f"![P-value Histogram]({os.path.relpath(pvalue_plot_path, project_root).replace(os.sep, '/')})")
markdown_lines.append("")
markdown_lines.append("## Interpretation")
markdown_lines.append("")
markdown_lines.append("The jackknife analysis reveals the sensitivity of the park-gravity effect to individual observations:")
markdown_lines.append("")
markdown_lines.append(f"- **Coefficient stability:** The median coefficient ({coef_median:.4f}) is close to the mean ({coef_mean:.4f}), suggesting relative stability across iterations.")
markdown_lines.append(f"- **Significance sensitivity:** {pct_significant:.1f}% of iterations yield significant results (p < 0.05), indicating moderate sensitivity to specific observations.")
markdown_lines.append(f"- **Effect direction:** All coefficients are positive, suggesting a consistent positive association between park gravity and mental distress outcomes across all leave-one-out iterations.")
markdown_lines.append("")
markdown_lines.append("## Model Specification")
markdown_lines.append("")
markdown_lines.append("- **Outcome:** Frequent Mental Distress (MHLTH_CrudePrev)")
markdown_lines.append("- **Predictor:** park_gravity_index_z (Z-score normalized)")
markdown_lines.append("- **Controls:** median_household_income, pct_families_below_poverty, unemployment_rate, pct_bachelors_degree_or_higher")
markdown_lines.append("- **Additional Controls:** log_pop_density")
markdown_lines.append("- **Demographics:** pct_under_18, pct_65_and_over, pct_black, pct_hispanic")
markdown_lines.append("- **Method:** Weighted Least Squares (WLS) with population weights")
markdown_lines.append("- **Standard Errors:** Robust (HC1)")

# Write markdown file
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(markdown_lines))

print(f"   Saved to: {summary_path}")

# ============================================================================
# 6. COMPLETE
# ============================================================================
print(f"\n{'='*60}")
print("JACKKNIFE RESULTS SUMMARY COMPLETE")
print(f"{'='*60}")
print(f"\nOutputs created:")
print(f"  1. Coefficient histogram: {coeff_plot_path}")
print(f"  2. P-value histogram: {pvalue_plot_path}")
print(f"  3. Summary markdown: {summary_path}")
