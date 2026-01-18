"""
Health Regression Results Summary
Economic Impact Report for Butler County Parks

This script reads health regression results (OLS and WLS), filters to park exposure
coefficients, creates clean markdown tables ranked by p-value, and generates
coefficient plots for both estimation methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Get the script directory and construct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Input paths
ols_input_path = os.path.join(project_root, 'results', 'health_regressions.csv')
wls_input_path = os.path.join(project_root, 'results', 'health_regressions_wls.csv')

# Output paths
output_dir = os.path.join(project_root, 'results')
figures_dir = os.path.join(project_root, 'figures')
ols_markdown_path = os.path.join(output_dir, 'health_results_summary.md')
wls_markdown_path = os.path.join(output_dir, 'health_results_summary_wls.md')
ols_plot_path = os.path.join(figures_dir, 'health_coefficients.png')
wls_plot_path = os.path.join(figures_dir, 'health_coefficients_wls.png')

print("="*60)
print("HEALTH REGRESSION RESULTS SUMMARY")
print("="*60)

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Helper function to process results and create outputs
def process_results(input_path, markdown_path, plot_path, method_name, method_description):
    """Process regression results and create markdown table and plot."""
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {method_name} RESULTS")
    print(f"{'='*60}")
    
    # Load regression results
    print(f"\n1. Loading {method_name} regression results...")
    print(f"   Path: {input_path}")
    if not os.path.exists(input_path):
        print(f"   Warning: {input_path} not found. Skipping {method_name} results.")
        return None
    
    results_df = pd.read_csv(input_path)
    print(f"   Total results loaded: {len(results_df):,}")
    
    # Filter to park exposure coefficients
    print(f"\n2. Filtering to park exposure coefficients...")
    park_predictors = ['within_1_mile', 'dist_to_park_miles']
    park_results = results_df[results_df['predictor'].isin(park_predictors)].copy()
    print(f"   Park exposure results: {len(park_results):,}")
    
    if len(park_results) == 0:
        print(f"   Warning: No park exposure coefficients found in {method_name} results!")
        return None
    
    # Prepare data for summary
    print(f"\n3. Preparing summary data...")
    
    # Create a clean summary table
    summary_df = park_results.copy()
    
    # Recalculate significance from p-values (more robust than reading from CSV)
    def add_significance(pvalue):
        if pd.isna(pvalue):
            return ''
        elif pvalue < 0.01:
            return '***'
        elif pvalue < 0.05:
            return '**'
        elif pvalue < 0.1:
            return '*'
        else:
            return ''
    
    summary_df['significance'] = summary_df['pvalue'].apply(add_significance)
    
    # Format values for display
    summary_df['coef_formatted'] = summary_df['coefficient'].apply(lambda x: f"{x:+.4f}")
    summary_df['se_formatted'] = summary_df['std_error'].apply(lambda x: f"({x:.4f})")
    summary_df['pvalue_formatted'] = summary_df['pvalue'].apply(lambda x: f"{x:.4f}")
    summary_df['ci_formatted'] = summary_df.apply(
        lambda row: f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]", axis=1
    )
    
    # Create a combined coefficient + SE + significance string
    summary_df['coef_display'] = summary_df.apply(
        lambda row: f"{row['coef_formatted']} {row['se_formatted']}{row['significance']}", axis=1
    )
    
    # Sort by p-value
    summary_df = summary_df.sort_values('pvalue')
    
    # Create markdown table
    print(f"\n4. Creating {method_name} markdown table...")
    
    markdown_lines = []
    markdown_lines.append(f"# Health Outcomes: Park Exposure Effects ({method_name})")
    markdown_lines.append("")
    markdown_lines.append(f"Results from {method_description}.")
    markdown_lines.append("")
    markdown_lines.append("## Summary Table (Ranked by P-value)")
    markdown_lines.append("")
    markdown_lines.append("| Outcome | Predictor | Model | Coefficient (SE) | P-value | 95% CI | N | R² |")
    markdown_lines.append("|---------|-----------|-------|------------------|---------|--------|---|----|")
    
    for _, row in summary_df.iterrows():
        outcome = row['outcome']
        predictor = row['predictor']
        model = row['model']
        coef_display = row['coef_display']
        pvalue = row['pvalue_formatted']
        ci = row['ci_formatted']
        n_obs = int(row['n_obs'])
        r_sq = f"{row['r_squared']:.4f}"
        
        markdown_lines.append(
            f"| {outcome} | {predictor} | {model} | {coef_display} | {pvalue} | {ci} | {n_obs} | {r_sq} |"
        )
    
    markdown_lines.append("")
    markdown_lines.append("**Significance levels:** *** p<0.01, ** p<0.05, * p<0.1")
    markdown_lines.append("")
    
    # Write markdown file
    with open(markdown_path, 'w') as f:
        f.write('\n'.join(markdown_lines))
    
    print(f"   Markdown table saved to: {markdown_path}")
    
    # Display summary
    print(f"\n   Results summary:")
    print(f"     Total coefficients: {len(summary_df)}")
    print(f"     Significant at p<0.05: {len(summary_df[summary_df['pvalue'] < 0.05])}")
    print(f"     Significant at p<0.10: {len(summary_df[summary_df['pvalue'] < 0.10])}")
    
    # Create coefficient plot
    print(f"\n5. Creating {method_name} coefficient plot...")
    
    # Prepare data for plotting
    plot_df = summary_df.copy()
    
    # Create a label for each result
    plot_df['label'] = plot_df.apply(
        lambda row: f"{row['outcome']}\n({row['predictor']})", axis=1
    )
    
    # Sort by coefficient value for better visualization
    plot_df = plot_df.sort_values('coefficient')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get colors based on significance
    colors = []
    for pval in plot_df['pvalue']:
        if pval < 0.01:
            colors.append('#2ca02c')  # Green for highly significant
        elif pval < 0.05:
            colors.append('#1f77b4')  # Blue for significant
        elif pval < 0.10:
            colors.append('#ff7f0e')  # Orange for marginally significant
        else:
            colors.append('#d62728')  # Red for not significant
    
    # Plot coefficients with error bars
    y_pos = np.arange(len(plot_df))
    x_coef = plot_df['coefficient'].values
    x_ci_lower = plot_df['ci_lower'].values
    x_ci_upper = plot_df['ci_upper'].values
    
    # Create horizontal error bars
    ax.errorbar(x_coef, y_pos, 
                xerr=[x_coef - x_ci_lower, x_ci_upper - x_coef],
                fmt='o', capsize=5, capthick=2, elinewidth=2,
                color='gray', alpha=0.6, label='95% CI')
    
    # Plot points with colors
    scatter = ax.scatter(x_coef, y_pos, c=colors, s=150, 
                         edgecolors='black', linewidths=1.5, zorder=5)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Set labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['label'].values, fontsize=10)
    ax.set_xlabel('Coefficient (Percentage Points)', fontsize=12, fontweight='bold')
    ax.set_title(f'Park Exposure Effects on Health Outcomes\n({method_description})', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Add legend for significance
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='p < 0.01'),
        Patch(facecolor='#1f77b4', label='p < 0.05'),
        Patch(facecolor='#ff7f0e', label='p < 0.10'),
        Patch(facecolor='#d62728', label='p ≥ 0.10')
    ]
    ax.legend(handles=legend_elements, loc='upper right', title='Significance Level', 
              fontsize=10, title_fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print(f"   Coefficient plot saved to: {plot_path}")
    
    # Display plot summary
    print(f"\n   Plot summary:")
    print(f"     Outcomes shown: {plot_df['outcome'].nunique()}")
    print(f"     Predictors shown: {plot_df['predictor'].nunique()}")
    print(f"     Coefficient range: [{plot_df['coefficient'].min():.4f}, {plot_df['coefficient'].max():.4f}]")
    
    return summary_df

# Process OLS results
ols_summary = process_results(
    ols_input_path,
    ols_markdown_path,
    ols_plot_path,
    'OLS',
    'OLS with Robust Standard Errors (HC1)'
)

# Process WLS results
wls_summary = process_results(
    wls_input_path,
    wls_markdown_path,
    wls_plot_path,
    'WLS',
    'WLS (Population-Weighted) with Robust Standard Errors (HC1)'
)

# Print sample of results
print(f"\n{'='*60}")
print("SAMPLE RESULTS")
print(f"{'='*60}")

if ols_summary is not None:
    print(f"\nOLS Results (Top 5 by P-value):")
    print(ols_summary[['outcome', 'predictor', 'coefficient', 'std_error', 'pvalue', 'significance']].head().to_string(index=False))

if wls_summary is not None:
    print(f"\nWLS Results (Top 5 by P-value):")
    print(wls_summary[['outcome', 'predictor', 'coefficient', 'std_error', 'pvalue', 'significance']].head().to_string(index=False))

print(f"\n{'='*60}")
print("HEALTH RESULTS SUMMARY COMPLETE")
print(f"{'='*60}")
print(f"\nOutputs created:")
if ols_summary is not None:
    print(f"  OLS:")
    print(f"    1. Markdown table: {ols_markdown_path}")
    print(f"    2. Coefficient plot: {ols_plot_path}")
if wls_summary is not None:
    print(f"  WLS:")
    print(f"    1. Markdown table: {wls_markdown_path}")
    print(f"    2. Coefficient plot: {wls_plot_path}")
