#!/usr/bin/env python3
"""
Plot Budget Sweep Results
=========================

Creates publication-quality figures from budget sweep data.

Usage:
------
python plot_budget_sweep.py --indir budget_sweep_results

Outputs:
--------
- budget_sweep_phase_diagram.png/pdf
- budget_sweep_scatter.png/pdf
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="budget_sweep_results")
    ap.add_argument("--outdir", type=str, default="budget_sweep_results")
    args = ap.parse_args()

    # Load data
    with open(f"{args.indir}/sweep_SUMMARY.json") as f:
        summary = json.load(f)
    
    with open(f"{args.indir}/sweep_plot_data.json") as f:
        plot_data = json.load(f)

    budgets = np.array(plot_data["budgets"])
    edges_mean = np.array(plot_data["edges_mean"])
    edges_std = np.array(plot_data["edges_std"])
    gini_mean = np.array(plot_data["gini_mean"])
    gini_std = np.array(plot_data["gini_std"])
    corr_mean = np.array([x if x is not None else np.nan for x in plot_data["corr_wd_mean"]])
    corr_std = np.array([x if x is not None else np.nan for x in plot_data["corr_wd_std"]])

    meta = summary["meta"]
    all_results = summary["all_results"]

    # =========================================================================
    # Figure 1: Phase Diagram
    # =========================================================================
    fig1, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Edges vs Budget
    ax1 = axes[0]
    ax1.errorbar(budgets, edges_mean, yerr=edges_std, fmt='o-', 
                 color='#2ecc71', capsize=3, capthick=1.5, markersize=8, linewidth=2)
    ax1.set_xlabel('Per-Node Budget (Λ)', fontsize=12)
    ax1.set_ylabel('Number of Edges', fontsize=12)
    ax1.set_title('A. Sparsity vs Capacity', fontsize=12, fontweight='bold')
    ax1.axhline(y=45, color='gray', linestyle='--', alpha=0.5, label='Max (N=10)')
    ax1.set_ylim(0, max(edges_mean) * 1.3)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Panel B: Gini vs Budget
    ax2 = axes[1]
    ax2.errorbar(budgets, gini_mean, yerr=gini_std, fmt='s-',
                 color='#3498db', capsize=3, capthick=1.5, markersize=8, linewidth=2)
    ax2.set_xlabel('Per-Node Budget (Λ)', fontsize=12)
    ax2.set_ylabel('Gini Coefficient', fontsize=12)
    ax2.set_title('B. Inequality vs Capacity', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 0.6)
    ax2.grid(True, alpha=0.3)

    # Panel C: Locality Correlation vs Budget
    ax3 = axes[2]
    mask = ~np.isnan(corr_mean)
    ax3.errorbar(budgets[mask], corr_mean[mask], yerr=corr_std[mask], fmt='^-',
                 color='#e74c3c', capsize=3, capthick=1.5, markersize=8, linewidth=2)
    ax3.set_xlabel('Per-Node Budget (Λ)', fontsize=12)
    ax3.set_ylabel('Weight-Distance Correlation', fontsize=12)
    ax3.set_title('C. Locality vs Capacity', fontsize=12, fontweight='bold')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylim(-0.6, 0.2)
    ax3.grid(True, alpha=0.3)

    # Shade the "strong locality" region
    ax3.axhspan(-0.6, -0.25, alpha=0.1, color='green', label='Strong locality')
    ax3.legend(loc='upper left')

    plt.suptitle(f'Budget Sweep: Locality Phase Transition\n'
                 f'N={meta["N"]}, {meta["seeds_per_budget"]} seeds/budget, '
                 f'{meta["total_runs"]} total runs',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    fig1.savefig(f'{args.outdir}/budget_sweep_phase_diagram.png', dpi=150, 
                 bbox_inches='tight', facecolor='white')
    fig1.savefig(f'{args.outdir}/budget_sweep_phase_diagram.pdf', dpi=300,
                 bbox_inches='tight', facecolor='white')
    print(f"Saved: {args.outdir}/budget_sweep_phase_diagram.png/pdf")

    # =========================================================================
    # Figure 2: Scatter Plot (all individual runs)
    # =========================================================================
    fig2, ax = plt.subplots(figsize=(10, 7))

    # Color by budget
    budget_vals = np.array([r["budget"] for r in all_results])
    edges_vals = np.array([r["edges"] for r in all_results])
    corr_vals = np.array([r["corr_wd"] if r["corr_wd"] is not None else np.nan 
                          for r in all_results])

    # Filter out NaN
    valid = ~np.isnan(corr_vals)
    
    sc = ax.scatter(edges_vals[valid], corr_vals[valid], c=budget_vals[valid],
                    cmap='viridis', s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Budget (Λ)', fontsize=11)

    ax.set_xlabel('Number of Edges', fontsize=12)
    ax.set_ylabel('Weight-Distance Correlation (Locality)', fontsize=12)
    ax.set_title('Budget Sweep: Sparsity-Locality Phase Space\n'
                 '(each point = one run, color = budget)',
                 fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Add region annotations
    ax.add_patch(Rectangle((2, -0.6), 8, 0.35, fill=True, 
                            facecolor='green', alpha=0.1, zorder=0))
    ax.text(6, -0.55, 'SPARSE + LOCAL\n(Low Budget)', ha='center', 
            fontsize=10, color='darkgreen', fontweight='bold')

    ax.add_patch(Rectangle((15, -0.15), 15, 0.35, fill=True,
                            facecolor='red', alpha=0.1, zorder=0))
    ax.text(22, 0.05, 'DENSE + NON-LOCAL\n(High Budget)', ha='center',
            fontsize=10, color='darkred')

    fig2.savefig(f'{args.outdir}/budget_sweep_scatter.png', dpi=150,
                 bbox_inches='tight', facecolor='white')
    fig2.savefig(f'{args.outdir}/budget_sweep_scatter.pdf', dpi=300,
                 bbox_inches='tight', facecolor='white')
    print(f"Saved: {args.outdir}/budget_sweep_scatter.png/pdf")

    # =========================================================================
    # Figure 3: Correlation Analysis
    # =========================================================================
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Budget vs |corr_wd| (locality strength)
    ax1 = axes[0]
    abs_corr_mean = np.abs(corr_mean)
    ax1.errorbar(budgets[mask], abs_corr_mean[mask], yerr=corr_std[mask], fmt='o-',
                 color='#9b59b6', capsize=3, markersize=8, linewidth=2)
    ax1.set_xlabel('Per-Node Budget (Λ)', fontsize=12)
    ax1.set_ylabel('|Weight-Distance Correlation|', fontsize=12)
    ax1.set_title('A. Locality Strength vs Capacity', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Fit exponential decay
    try:
        from scipy.optimize import curve_fit
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        valid_budgets = budgets[mask]
        valid_abs_corr = abs_corr_mean[mask]
        popt, _ = curve_fit(exp_decay, valid_budgets, valid_abs_corr, 
                           p0=[0.3, 0.2, 0.1], maxfev=5000)
        x_fit = np.linspace(budgets.min(), budgets.max(), 100)
        y_fit = exp_decay(x_fit, *popt)
        ax1.plot(x_fit, y_fit, 'k--', alpha=0.5, label=f'Exp fit: τ={1/popt[1]:.1f}')
        ax1.legend()
    except:
        pass

    # Panel B: Edges vs Locality Correlation
    ax2 = axes[1]
    ax2.scatter(edges_vals[valid], corr_vals[valid], c=budget_vals[valid],
                cmap='viridis', s=60, alpha=0.6, edgecolors='black', linewidths=0.3)
    ax2.set_xlabel('Number of Edges', fontsize=12)
    ax2.set_ylabel('Weight-Distance Correlation', fontsize=12)
    ax2.set_title('B. Locality vs Sparsity', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # Fit linear trend
    valid_edges = edges_vals[valid]
    valid_corr = corr_vals[valid]
    if len(valid_edges) > 2:
        z = np.polyfit(valid_edges, valid_corr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_edges.min(), valid_edges.max(), 100)
        ax2.plot(x_line, p(x_line), 'r--', alpha=0.7, 
                 label=f'Linear fit: slope={z[0]:.4f}')
        ax2.legend()

    plt.tight_layout()
    fig3.savefig(f'{args.outdir}/budget_sweep_correlation.png', dpi=150,
                 bbox_inches='tight', facecolor='white')
    print(f"Saved: {args.outdir}/budget_sweep_correlation.png")

    # =========================================================================
    # Print Summary Statistics
    # =========================================================================
    print("\n" + "=" * 60)
    print("BUDGET SWEEP SUMMARY")
    print("=" * 60)
    
    # Find critical budget (where locality drops below threshold)
    locality_threshold = -0.25
    critical_idx = np.where(corr_mean > locality_threshold)[0]
    if len(critical_idx) > 0:
        critical_budget = budgets[critical_idx[0]]
        print(f"Critical budget (locality > {locality_threshold}): Λ ≈ {critical_budget:.1f}")
    
    # Correlation between budget and locality
    valid_mask = ~np.isnan(corr_mean)
    if np.sum(valid_mask) > 2:
        r = np.corrcoef(budgets[valid_mask], corr_mean[valid_mask])[0, 1]
        print(f"Budget-Locality correlation: r = {r:.3f}")
    
    # Min/max locality
    print(f"Strongest locality: corr_wd = {np.nanmin(corr_mean):.3f} at Λ = {budgets[np.nanargmin(corr_mean)]:.1f}")
    print(f"Weakest locality: corr_wd = {np.nanmax(corr_mean):.3f} at Λ = {budgets[np.nanargmax(corr_mean)]:.1f}")

    print("\nPlotting complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())